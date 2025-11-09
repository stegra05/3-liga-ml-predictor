"""
Populate matches weather fields using Meteostat (hourly historical).
Strategy:
 1) Map each home team to nearest station with hourly coverage across match dates
 2) For each station, fetch hourly time series for full date range
 3) Join hourly weather to matches by station & hour and update matches
"""
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple
import pandas as pd
from loguru import logger
import argparse

sys.path.append(str(Path(__file__).parent.parent.parent))
from database.db_manager import get_db


def load_matches(db) -> pd.DataFrame:
    df = db.query_to_dataframe("""
        SELECT m.match_id, m.match_datetime, m.home_team_id, t.lat, t.lon
        FROM matches m
        JOIN team_locations t ON t.team_id = m.home_team_id
        WHERE m.temperature_celsius IS NULL
          AND m.match_datetime IS NOT NULL
          AND t.lat IS NOT NULL AND t.lon IS NOT NULL
        ORDER BY m.match_datetime
    """)
    if df.empty:
        return df
    # Ensure timezone aware Europe/Berlin
    df["match_datetime"] = pd.to_datetime(df["match_datetime"]).dt.tz_localize(None)
    df["match_datetime"] = df["match_datetime"].dt.tz_localize(ZoneInfo("Europe/Berlin"))
    # Round to top of hour for join
    df["match_hour"] = df["match_datetime"].dt.floor("h")
    return df


def map_team_to_stations(df_matches: pd.DataFrame, k: int = 3) -> Tuple[Dict[int, List[str]], Dict[str, Tuple[float, float]]]:
    """
    Map teams to nearest stations and return station coordinates for distance calculation.
    Returns: (team_to_stations dict, station_coords dict with lat/lon)
    """
    from meteostat import Stations
    import math
    team_to_stations: Dict[int, List[str]] = {}
    station_coords: Dict[str, Tuple[float, float]] = {}
    logger.info(f"Selecting up to {k} nearest stations per team")
    # Unique teams and their coords
    coords = (
        df_matches[["home_team_id", "lat", "lon"]]
        .drop_duplicates("home_team_id")
        .itertuples(index=False, name=None)
    )
    for home_team_id, lat, lon in coords:
        try:
            st = Stations()
            st = st.nearby(lat, lon)
            res = st.fetch(k)
            if not res.empty:
                team_to_stations[home_team_id] = list(res.index)
                # Store station coordinates for distance calculation
                for station_id in res.index:
                    if station_id not in station_coords:
                        try:
                            station_coords[station_id] = (float(res.loc[station_id, 'latitude']), float(res.loc[station_id, 'longitude']))
                        except (KeyError, ValueError):
                            # Fallback: try alternative column names
                            try:
                                station_coords[station_id] = (float(res.loc[station_id, 'lat']), float(res.loc[station_id, 'lon']))
                            except (KeyError, ValueError):
                                logger.debug(f"Could not extract coordinates for station {station_id}")
                                continue
        except Exception:
            continue
    logger.info(f"Mapped {len(team_to_stations)} teams to station lists")
    return team_to_stations, station_coords


def calculate_confidence_meteostat(distance_km: float, exact_hour: bool = True) -> float:
    """
    Calculate confidence score for Meteostat data.
    Base: 0.95, distance penalty: -0.01 per 5km beyond 5km (cap -0.1), hour rounding: -0.02
    """
    base = 0.95
    distance_penalty = 0.0
    if distance_km > 5.0:
        penalty_km = (distance_km - 5.0) / 5.0
        distance_penalty = min(0.01 * penalty_km, 0.1)
    hour_penalty = 0.0 if exact_hour else 0.02
    confidence = max(base - distance_penalty - hour_penalty, 0.7)
    return confidence


def fetch_hourly_for_stations(team_to_stations: Dict[int, List[str]], df_matches: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from meteostat import Hourly
    # Group matches by station
    dfm = df_matches.copy()
    dfm["stations"] = dfm["home_team_id"].map(team_to_stations)
    dfm = dfm.dropna(subset=["stations"])
    if dfm.empty:
        return dfm, pd.DataFrame()
    # Explode to one row per station
    dfm = dfm.explode("stations").rename(columns={"stations": "station"})
    start = dfm["match_hour"].min()
    end = dfm["match_hour"].max()
    # Make naive local datetimes for Meteostat
    start_naive = pd.Timestamp(start).tz_localize(None) if pd.Timestamp(start).tzinfo else pd.Timestamp(start)
    end_naive = pd.Timestamp(end).tz_localize(None) if pd.Timestamp(end).tzinfo else pd.Timestamp(end)
    # Cap end to now to avoid future year warnings
    now_naive = pd.Timestamp.now(ZoneInfo("Europe/Berlin")).replace(tzinfo=None)
    if end_naive > now_naive:
        end_naive = now_naive
    logger.info(f"Fetching hourly weather for stations: {dfm['station'].nunique()} stations, {start}..{end}")
    frames: List[pd.DataFrame] = []
    for station in sorted(dfm["station"].unique()):
        try:
            h = Hourly(station, start_naive, end_naive, timezone="Europe/Berlin").fetch()
            if h.empty:
                continue
            h = h.reset_index().rename(columns={
                "time": "match_hour",
                "temp": "temperature_celsius",
                "rhum": "humidity_percent",
                "prcp": "precipitation_mm",
                "wspd": "wind_speed_kmh",
            })
            h["station"] = station
            # Some datasets use different column names; normalize if present
            for alt, std in [
                ("temp", "temperature_celsius"),
                ("rh", "humidity_percent"),
                ("precipitation", "precipitation_mm"),
                ("wspd", "wind_speed_kmh"),
                ("ws", "wind_speed_kmh"),
            ]:
                if alt in h.columns and std not in h.columns:
                    h[std] = h[alt]
            keep = ["station", "match_hour", "temperature_celsius", "humidity_percent", "precipitation_mm", "wind_speed_kmh"]
            frames.append(h[keep])
        except Exception as e:
            logger.debug(f"Station {station} fetch failed: {e}")
            continue
    if not frames:
        return dfm, pd.DataFrame()
    return dfm, pd.concat(frames, ignore_index=True)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in km between two lat/lon points"""
    import math
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


def update_matches_with_weather(db, df_exploded: pd.DataFrame, df_hourly: pd.DataFrame, 
                                station_coords: Dict[str, Tuple[float, float]]) -> Tuple[int, int]:
    if df_hourly.empty:
        return 0, len(df_exploded)
    # Join hourly to exploded matches, then aggregate median across stations per match_id
    joined = df_exploded.merge(
        df_hourly,
        on=["station", "match_hour"],
        how="left",
    )
    # Calculate distance for each match-station pair
    joined["distance_km"] = joined.apply(
        lambda row: haversine_distance(
            row["lat"], row["lon"],
            station_coords.get(row["station"], (row["lat"], row["lon"]))[0],
            station_coords.get(row["station"], (row["lat"], row["lon"]))[1]
        ) if row["station"] in station_coords else 0.0,
        axis=1
    )
    # Check if exact hour match (match_datetime == match_hour)
    joined["exact_hour"] = (joined["match_datetime"] == joined["match_hour"])
    
    # Aggregate: median for weather values, min distance for confidence calculation
    agg = joined.groupby("match_id", as_index=False).agg({
        "temperature_celsius": "median",
        "humidity_percent": "median",
        "wind_speed_kmh": "median",
        "precipitation_mm": "median",
        "distance_km": "min",  # Use closest station distance
        "exact_hour": "any",  # True if any station had exact hour
        "lat": "first",
        "lon": "first",
    })
    
    updated = 0
    skipped = 0
    for row in agg.itertuples(index=False):
        match_id = int(row.match_id)
        temp = getattr(row, "temperature_celsius")
        hum = getattr(row, "humidity_percent")
        wind = getattr(row, "wind_speed_kmh")
        prcp = getattr(row, "precipitation_mm")
        if pd.isna(temp) and pd.isna(hum) and pd.isna(wind) and pd.isna(prcp):
            skipped += 1
            continue
        
        distance_km = float(getattr(row, "distance_km"))
        exact_hour = bool(getattr(row, "exact_hour"))
        confidence = calculate_confidence_meteostat(distance_km, exact_hour)
        
        db.execute_insert("""
            UPDATE matches
               SET temperature_celsius = ?,
                   humidity_percent = ?,
                   wind_speed_kmh = ?,
                   precipitation_mm = ?,
                   weather_source = 'meteostat',
                   weather_confidence = ?
             WHERE match_id = ?
        """, (
            None if pd.isna(temp) else float(temp),
            None if pd.isna(hum) else float(hum),
            None if pd.isna(wind) else float(wind),
            None if pd.isna(prcp) else float(prcp),
            confidence,
            match_id,
        ))
        updated += 1
    return updated, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    db = get_db()
    logger.info("=== Meteostat weather population start ===")
    df_matches = load_matches(db)
    if df_matches.empty:
        logger.info("No matches needing weather")
        return
    if args.limit:
        df_matches = df_matches.head(args.limit)
    team_to_stations, station_coords = map_team_to_stations(df_matches, k=3)
    if not team_to_stations:
        logger.warning("No station mappings found")
        return
    df_exp, df_hourly = fetch_hourly_for_stations(team_to_stations, df_matches)
    updated, skipped = update_matches_with_weather(db, df_exp, df_hourly, station_coords)
    logger.success(f"Weather update done: updated={updated}, skipped={skipped}")


if __name__ == "__main__":
    # Optional limit via env var or manual edit
    main()


