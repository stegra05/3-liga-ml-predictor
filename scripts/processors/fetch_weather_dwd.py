"""
Fetch historical weather for matches using DWD (Deutscher Wetterdienst) via wetterdienst.
This is a fallback for matches still missing weather data after Meteostat and Open-Meteo.
"""
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple, Optional
import pandas as pd
from loguru import logger
import argparse

sys.path.append(str(Path(__file__).parent.parent.parent))
from database.db_manager import get_db


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


def find_nearest_dwd_station(lat: float, lon: float, max_distance_km: float = 50.0) -> Optional[Dict]:
    """
    Find nearest DWD station to given coordinates.
    Returns station info dict with station_id, lat, lon, distance_km or None if none found.
    """
    try:
        from wetterdienst import Wetterdienst
        from wetterdienst.provider.dwd.observation import DwdObservationRequest
        from wetterdienst.provider.dwd.observation import DwdObservationDataset
        
        # Get available stations
        request = DwdObservationRequest(
            parameter=DwdObservationDataset.HOURLY.TEMPERATURE_AIR_200,
            resolution=DwdObservationDataset.HOURLY.Resolution.HOURLY,
            start_date=datetime(2000, 1, 1),  # Wide range
            end_date=datetime.now(),
        )
        
        # Get station list
        stations_df = request.all().df
        
        if stations_df.empty:
            return None
        
        # Calculate distances
        stations_df['distance_km'] = stations_df.apply(
            lambda row: haversine_distance(lat, lon, row['latitude'], row['longitude']),
            axis=1
        )
        
        # Filter by max distance and sort
        nearby = stations_df[stations_df['distance_km'] <= max_distance_km].sort_values('distance_km')
        
        if nearby.empty:
            return None
        
        nearest = nearby.iloc[0]
        return {
            'station_id': nearest['station_id'],
            'lat': float(nearest['latitude']),
            'lon': float(nearest['longitude']),
            'distance_km': float(nearest['distance_km']),
        }
    except Exception as e:
        logger.debug(f"Error finding DWD station: {e}")
        return None


def fetch_dwd_weather(station_id: str, dt: datetime) -> Optional[Dict]:
    """
    Fetch hourly weather from DWD station for specific datetime.
    Returns dict with weather data or None.
    """
    try:
        from wetterdienst import Wetterdienst
        from wetterdienst.provider.dwd.observation import DwdObservationRequest
        from wetterdienst.provider.dwd.observation import DwdObservationDataset
        
        # Request hourly data for the date
        date_str = dt.strftime("%Y-%m-%d")
        request = DwdObservationRequest(
            parameter=[
                DwdObservationDataset.HOURLY.TEMPERATURE_AIR_200,
                DwdObservationDataset.HOURLY.HUMIDITY,
                DwdObservationDataset.HOURLY.PRECIPITATION_HEIGHT,
                DwdObservationDataset.HOURLY.WIND_SPEED,
            ],
            resolution=DwdObservationDataset.HOURLY.Resolution.HOURLY,
            start_date=datetime.strptime(date_str, "%Y-%m-%d"),
            end_date=datetime.strptime(date_str, "%Y-%m-%d"),
        ).filter_by_station_id(station_id)
        
        df = request.values.all().df
        
        if df.empty:
            return None
        
        # Find matching hour
        target_hour = dt.hour
        exact_hour = False
        
        # Try to find exact hour
        df['hour'] = pd.to_datetime(df['date']).dt.hour
        matching = df[df['hour'] == target_hour]
        
        if not matching.empty:
            row = matching.iloc[0]
            exact_hour = True
        else:
            # Use closest hour
            df['hour_diff'] = abs(df['hour'] - target_hour)
            row = df.loc[df['hour_diff'].idxmin()]
        
        # Extract values (column names may vary, try common ones)
        temp = None
        humidity = None
        precipitation = None
        wind_speed = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'temperature' in col_lower or 'temp' in col_lower:
                temp = row.get(col)
            elif 'humidity' in col_lower or 'feuchte' in col_lower:
                humidity = row.get(col)
            elif 'precipitation' in col_lower or 'niederschlag' in col_lower:
                precipitation = row.get(col)
            elif 'wind' in col_lower and 'speed' in col_lower:
                wind_speed = row.get(col)
        
        # Try direct column access
        if temp is None:
            temp = row.get('value') if 'value' in row.index else None
        
        if temp is None and humidity is None and precipitation is None and wind_speed is None:
            return None
        
        return {
            "temperature_celsius": float(temp) if temp is not None and pd.notna(temp) else None,
            "humidity_percent": float(humidity) if humidity is not None and pd.notna(humidity) else None,
            "precipitation_mm": float(precipitation) if precipitation is not None and pd.notna(precipitation) else None,
            "wind_speed_kmh": float(wind_speed) if wind_speed is not None and pd.notna(wind_speed) else None,
            "exact_hour": exact_hour,
        }
    except Exception as e:
        logger.debug(f"Error fetching DWD weather for station {station_id} at {dt}: {e}")
        return None


def load_matches(db) -> pd.DataFrame:
    """Load matches still missing weather data"""
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
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch weather from DWD for remaining gaps")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of matches to update")
    parser.add_argument("--max-distance", type=float, default=50.0, help="Max distance to station in km")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep between API calls (seconds)")
    args = parser.parse_args()
    
    db = get_db()
    logger.info("=== DWD weather population start ===")
    df_matches = load_matches(db)
    
    if df_matches.empty:
        logger.info("No matches needing weather")
        return
    
    if args.limit:
        df_matches = df_matches.head(args.limit)
    
    logger.info(f"Processing {len(df_matches)} matches")
    
    updated = 0
    skipped = 0
    failed = 0
    
    # Cache station lookups per team
    team_stations: Dict[int, Dict] = {}
    
    for idx, row in df_matches.iterrows():
        match_id = int(row['match_id'])
        home_team_id = int(row['home_team_id'])
        lat = float(row['lat'])
        lon = float(row['lon'])
        dt = row['match_datetime']
        
        try:
            # Find or reuse station for this team
            if home_team_id not in team_stations:
                station_info = find_nearest_dwd_station(lat, lon, args.max_distance)
                if not station_info:
                    skipped += 1
                    logger.debug(f"Match {match_id}: No DWD station within {args.max_distance}km")
                    continue
                team_stations[home_team_id] = station_info
            
            station_info = team_stations[home_team_id]
            station_id = station_info['station_id']
            distance_km = station_info['distance_km']
            
            # Fetch weather
            weather = fetch_dwd_weather(station_id, dt)
            
            if not weather:
                skipped += 1
                logger.debug(f"Match {match_id}: No weather data from station {station_id}")
                continue
            
            # Calculate confidence
            exact_hour = weather.get("exact_hour", True)
            confidence = calculate_confidence_dwd(distance_km, exact_hour)
            
            # Update database
            db.execute_insert("""
                UPDATE matches
                   SET temperature_celsius = ?,
                       humidity_percent = ?,
                       wind_speed_kmh = ?,
                       precipitation_mm = ?,
                       weather_source = 'dwd',
                       weather_confidence = ?
                 WHERE match_id = ?
            """, (
                weather.get("temperature_celsius"),
                weather.get("humidity_percent"),
                weather.get("wind_speed_kmh"),
                weather.get("precipitation_mm"),
                confidence,
                match_id,
            ))
            
            updated += 1
            
            if (updated + skipped + failed) % 50 == 0:
                logger.info(f"Progress: updated={updated}, skipped={skipped}, failed={failed}")
            
            if args.sleep:
                import time
                time.sleep(args.sleep)
                
        except Exception as e:
            failed += 1
            logger.error(f"Error processing match {match_id}: {e}")
            continue
    
    logger.success(f"DWD weather update done: updated={updated}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()

