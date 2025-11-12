"""
Common utility functions for weather data collection scripts.
Extracted to reduce code duplication across weather processors.
"""
import math
from typing import Dict
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger
import argparse
import subprocess
import sys

from liga_predictor.database import get_db


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in km between two lat/lon points using Haversine formula"""
    R = 6371.0  # Earth radius in km
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def load_matches_needing_weather(db=None) -> pd.DataFrame:
    """
    Load all matches that need weather data from the database.
    Returns DataFrame with match_id, match_datetime, home_team_id, lat, lon.
    """
    if db is None:
        db = get_db()

    df = db.query_to_dataframe("""
        SELECT m.match_id, m.match_datetime, m.home_team_id, t.lat, t.lon
        FROM matches m
        JOIN team_locations t ON t.team_id = m.home_team_id
        WHERE m.temperature_celsius IS NULL
          AND m.match_datetime IS NOT NULL
          AND t.lat IS NOT NULL AND t.lon IS NOT NULL
        ORDER BY m.match_datetime
    """)
    return df


def calculate_confidence_meteostat(distance_km: float, exact_hour: bool = True) -> float:
    """
    Calculate confidence score for Meteostat data.
    Base: 0.95, distance penalty: -0.01 per 5km beyond 5km (cap -0.15), hour rounding: -0.02
    """
    base = 0.95
    distance_penalty = 0.0
    if distance_km > 5.0:
        penalty_km = (distance_km - 5.0) / 5.0
        distance_penalty = min(0.01 * penalty_km, 0.15)
    hour_penalty = 0.0 if exact_hour else 0.02
    confidence = max(base - distance_penalty - hour_penalty, 0.7)
    return confidence


def calculate_confidence_open_meteo(exact_hour: bool = True) -> float:
    """
    Calculate confidence score for Open-Meteo data.
    Base: 0.85, hour rounding penalty: -0.02
    """
    base = 0.85
    hour_penalty = 0.0 if exact_hour else 0.02
    confidence = max(base - hour_penalty, 0.7)
    return confidence


def calculate_confidence_dwd(distance_km: float, exact_hour: bool = True) -> float:
    """
    Calculate confidence score for DWD data.
    Base: 0.90, distance penalty: -0.01 per 5km beyond 5km (cap -0.1), hour rounding: -0.02
    """
    base = 0.90
    distance_penalty = 0.0
    if distance_km > 5.0:
        penalty_km = (distance_km - 5.0) / 5.0
        distance_penalty = min(0.01 * penalty_km, 0.1)
    hour_penalty = 0.0 if exact_hour else 0.02
    confidence = max(base - distance_penalty - hour_penalty, 0.7)
    return confidence


# Multi-source weather fetching orchestrator
# Runs staged pipeline: Meteostat → Open-Meteo → DWD to achieve 95%+ coverage


def get_weather_coverage(db) -> float:
    """Calculate current weather coverage percentage"""
    result = db.execute_query("""
        SELECT
            COUNT(*) as total_matches,
            COUNT(temperature_celsius) as with_weather
        FROM matches
        WHERE is_finished = 1
    """)[0]

    if result['total_matches'] == 0:
        return 0.0

    coverage = (result['with_weather'] / result['total_matches']) * 100
    return coverage


def get_matches_needing_weather(db) -> int:
    """Get count of matches still needing weather"""
    result = db.execute_query("""
        SELECT COUNT(*) as count
        FROM matches m
        JOIN team_locations t ON t.team_id = m.home_team_id
        WHERE m.temperature_celsius IS NULL
          AND m.match_datetime IS NOT NULL
          AND t.lat IS NOT NULL AND t.lon IS NOT NULL
          AND m.is_finished = 1
    """)[0]
    return result['count']


def run_stage(stage_name: str, script_path: str, limit: int = None, sleep: float = None, **kwargs) -> bool:
    """Run a weather fetching stage"""
    logger.info(f"=== Stage: {stage_name} ===")

    cmd = [sys.executable, str(script_path)]
    if limit:
        cmd.extend(["--limit", str(limit)])
    if sleep is not None:
        cmd.extend(["--sleep", str(sleep)])
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Stage {stage_name} completed successfully")
        if result.stdout:
            logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Stage {stage_name} failed: {e}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return False


if __name__ == "__main__":
    print("Use CLI instead: liga-predictor fetch-weather")
    import sys
    sys.exit(1)
