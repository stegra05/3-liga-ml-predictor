"""
Fetch historical weather for matches using Open-Meteo archive API.
Populates matches.temperature_celsius, humidity_percent, wind_speed_kmh, precipitation_mm, weather_condition (left blank).
"""
from pathlib import Path
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from loguru import logger
import requests
import argparse
import time

sys.path.append(str(Path(__file__).parent.parent.parent))
from database.db_manager import get_db

API = "https://archive-api.open-meteo.com/v1/archive"


def calculate_confidence_open_meteo(exact_hour: bool = True) -> float:
    """
    Calculate confidence score for Open-Meteo data.
    Base: 0.85, hour rounding penalty: -0.02
    """
    base = 0.85
    hour_penalty = 0.0 if exact_hour else 0.02
    confidence = max(base - hour_penalty, 0.7)
    return confidence


def fetch_hour(session: requests.Session, lat: float, lon: float, dt: datetime):
    date_str = dt.strftime("%Y-%m-%d")
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": "Europe/Berlin",
        "windspeed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    # Basic retry with backoff for 429/5xx
    for attempt in range(5):
        r = session.get(API, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(1.5 + attempt)  # backoff
            continue
        r.raise_for_status()
        break
    data = r.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time") or []
    exact_hour = False
    try:
        idx = times.index(dt.strftime("%Y-%m-%dT%H:00"))
        exact_hour = True
    except ValueError:
        # fallback: closest hour
        idx = None
        for i, t in enumerate(times):
            if t.startswith(date_str):
                idx = i
                break
    if idx is None:
        return None
    return {
        "temperature_celsius": (hourly.get("temperature_2m") or [None])[idx],
        "humidity_percent": (hourly.get("relative_humidity_2m") or [None])[idx],
        "precipitation_mm": (hourly.get("precipitation") or [None])[idx],
        "wind_speed_kmh": (hourly.get("wind_speed_10m") or [None])[idx],
        "exact_hour": exact_hour,
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch historical weather for matches")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of matches to update")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep between API calls (seconds)")
    args = parser.parse_args()

    db = get_db()
    logger.info("=== Fetching weather for matches with missing weather ===")
    rows = db.execute_query("""
        SELECT m.match_id, m.match_datetime, m.home_team_id, t.lat, t.lon
        FROM matches m
        JOIN team_locations t ON t.team_id = m.home_team_id
        WHERE m.temperature_celsius IS NULL
          AND m.match_datetime IS NOT NULL
          AND t.lat IS NOT NULL AND t.lon IS NOT NULL
        ORDER BY m.match_datetime
    """)
    total = len(rows)
    if args.limit:
        rows = rows[:args.limit]
    logger.info(f"Processing {len(rows)} matches (of {total} total needing weather)")
    updated = 0
    skipped = 0
    cache = {}
    session = requests.Session()
    session.headers.update({"User-Agent": "3Liga-WeatherFetcher/1.0"})
    for r in rows:
        mid = r["match_id"]
        dt = r["match_datetime"]
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt)
        # Treat naive as Europe/Berlin local time
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("Europe/Berlin"))
        else:
            dt = dt.astimezone(ZoneInfo("Europe/Berlin"))
        lat, lon = r["lat"], r["lon"]
        key = (round(lat, 4), round(lon, 4), dt.strftime("%Y-%m-%dT%H"))
        w = cache.get(key)
        if w is None:
            try:
                w = fetch_hour(session, lat, lon, dt)
                cache[key] = w
            except Exception as e:
                logger.debug(f"API error for match {mid}: {e}")
                # cool down after rate-limit or transient errors
                time.sleep(2.0)
                w = None
        if not w:
            skipped += 1
            continue
        
        exact_hour = w.get("exact_hour", True)
        confidence = calculate_confidence_open_meteo(exact_hour)
        
        db.execute_insert("""
            UPDATE matches
            SET temperature_celsius = ?, humidity_percent = ?, wind_speed_kmh = ?, precipitation_mm = ?,
                weather_source = 'open_meteo', weather_confidence = ?
            WHERE match_id = ?
        """, (w["temperature_celsius"], w["humidity_percent"], w["wind_speed_kmh"], w["precipitation_mm"], confidence, mid))
        updated += 1
        if (updated + skipped) % 50 == 0:
            logger.info(f"Progress: updated={updated}, skipped={skipped}")
        if args.sleep:
            time.sleep(args.sleep)
    logger.success(f"Weather fetch complete: updated={updated}, skipped={skipped}")


if __name__ == "__main__":
    main()


