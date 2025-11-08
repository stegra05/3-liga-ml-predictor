"""
Weather Data Collector for 3. Liga Matches
Fetches historical weather data from Open-Meteo API (FREE)
"""

import requests
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from database.db_manager import get_db


class WeatherCollector:
    """Collects historical weather data for football matches"""

    OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    RATE_LIMIT_DELAY = 0.5  # seconds between requests (10k/day limit)

    def __init__(self, locations_file: str = "config/stadium_locations.json"):
        """
        Initialize weather collector

        Args:
            locations_file: Path to stadium locations JSON
        """
        self.db = get_db()
        self.locations = self._load_locations(locations_file)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': '3Liga-Dataset-Weather-Collector/1.0'
        })
        logger.info(f"Weather collector initialized with {len(self.locations)} stadium locations")

    def _load_locations(self, locations_file: str) -> Dict:
        """Load stadium locations from JSON file"""
        path = Path(locations_file)
        if not path.exists():
            raise FileNotFoundError(f"Stadium locations file not found: {locations_file}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get('stadiums', {})

    def get_matches_without_weather(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get matches that don't have weather data yet

        Args:
            limit: Maximum number of matches to fetch (None = all)

        Returns:
            List of match dictionaries
        """
        query = """
            SELECT m.match_id, m.match_datetime, m.season, m.matchday,
                   t.team_name as home_team
            FROM matches m
            JOIN teams t ON m.home_team_id = t.team_id
            WHERE m.temperature_celsius IS NULL
            AND m.is_finished = 1
            AND m.match_datetime IS NOT NULL
            AND DATE(m.match_datetime) != '1970-01-01'
            ORDER BY m.match_datetime ASC
        """

        if limit:
            query += f" LIMIT {limit}"

        matches = self.db.execute_query(query)

        # Convert to list of dicts
        result = []
        for match in matches:
            result.append({
                'match_id': match['match_id'],
                'match_datetime': match['match_datetime'],
                'season': match['season'],
                'matchday': match['matchday'],
                'home_team': match['home_team']
            })

        logger.info(f"Found {len(result)} matches without weather data")
        return result

    def fetch_weather(self, team_name: str, match_datetime: str) -> Optional[Dict]:
        """
        Fetch weather data from Open-Meteo API

        Args:
            team_name: Home team name (to look up stadium location)
            match_datetime: Match datetime string

        Returns:
            Dictionary with weather data or None on error
        """
        # Get stadium location
        location = self.locations.get(team_name)
        if not location:
            logger.warning(f"No location data for team: {team_name}")
            return None

        # Parse match datetime
        try:
            dt = datetime.fromisoformat(match_datetime)
        except (ValueError, TypeError) as e:
            logger.error(f"Could not parse datetime {match_datetime}: {e}")
            return None

        # Prepare API parameters
        date_str = dt.strftime('%Y-%m-%d')
        params = {
            'latitude': location['latitude'],
            'longitude': location['longitude'],
            'start_date': date_str,
            'end_date': date_str,
            'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation',
            'timezone': 'Europe/Berlin'
        }

        try:
            # Make API request
            response = self.session.get(
                self.OPEN_METEO_ARCHIVE_URL,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            # Extract hourly data
            hourly = data.get('hourly', {})
            if not hourly:
                logger.warning(f"No hourly data returned for {team_name} on {date_str}")
                return None

            # Get weather at match time (hour)
            hour = dt.hour
            if hour >= len(hourly.get('temperature_2m', [])):
                logger.warning(f"Hour {hour} not available in data (length: {len(hourly.get('temperature_2m', []))})")
                # Use closest hour available
                hour = min(hour, len(hourly.get('temperature_2m', [])) - 1)

            weather = {
                'temperature_celsius': hourly['temperature_2m'][hour],
                'humidity_percent': hourly['relative_humidity_2m'][hour],
                'wind_speed_kmh': hourly['wind_speed_10m'][hour],
                'precipitation_mm': hourly['precipitation'][hour],
                'weather_condition': self._determine_weather_condition(
                    hourly['temperature_2m'][hour],
                    hourly['precipitation'][hour]
                )
            }

            # Validate data
            if weather['temperature_celsius'] is None:
                logger.warning(f"Temperature is None for {team_name} on {date_str}")
                return None

            return weather

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {team_name} on {date_str}: {e}")
            return None
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing API response for {team_name} on {date_str}: {e}")
            return None

    def _determine_weather_condition(self, temperature: float, precipitation: float) -> str:
        """
        Determine categorical weather condition from temperature and precipitation

        Args:
            temperature: Temperature in Celsius
            precipitation: Precipitation in mm

        Returns:
            Weather condition string
        """
        if precipitation > 5.0:
            if temperature < 0:
                return 'snow'
            else:
                return 'heavy_rain'
        elif precipitation > 0.5:
            if temperature < 0:
                return 'light_snow'
            else:
                return 'rain'
        elif temperature < 0:
            return 'freezing'
        elif temperature > 30:
            return 'hot'
        else:
            return 'clear'

    def update_match_weather(self, match_id: int, weather: Dict) -> None:
        """
        Update match with weather data

        Args:
            match_id: Match ID
            weather: Weather dictionary
        """
        query = """
            UPDATE matches SET
                temperature_celsius = ?,
                humidity_percent = ?,
                wind_speed_kmh = ?,
                precipitation_mm = ?,
                weather_condition = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE match_id = ?
        """

        self.db.execute_insert(query, (
            weather['temperature_celsius'],
            weather['humidity_percent'],
            weather['wind_speed_kmh'],
            weather['precipitation_mm'],
            weather['weather_condition'],
            match_id
        ))

    def collect_weather_for_all_matches(self, batch_size: int = 100):
        """
        Collect weather data for all matches without weather

        Args:
            batch_size: Number of matches to process before logging progress
        """
        logger.info("=== Starting weather data collection ===")
        start_time = datetime.now()

        matches = self.get_matches_without_weather()
        total_matches = len(matches)

        if total_matches == 0:
            logger.info("No matches need weather data collection")
            return

        collected = 0
        failed = 0
        skipped = 0

        for idx, match in enumerate(matches):
            try:
                # Fetch weather
                weather = self.fetch_weather(
                    match['home_team'],
                    match['match_datetime']
                )

                if weather:
                    # Update database
                    self.update_match_weather(match['match_id'], weather)
                    collected += 1

                    logger.debug(f"Match {match['match_id']}: {weather['temperature_celsius']:.1f}°C, "
                               f"{weather['precipitation_mm']}mm rain")
                else:
                    skipped += 1
                    logger.warning(f"Skipped match {match['match_id']} (no weather data available)")

                # Rate limiting
                time.sleep(self.RATE_LIMIT_DELAY)

                # Progress logging
                if (idx + 1) % batch_size == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = (idx + 1) / elapsed if elapsed > 0 else 0
                    remaining = (total_matches - idx - 1) / rate if rate > 0 else 0

                    logger.info(f"Progress: {idx + 1}/{total_matches} matches "
                              f"({collected} collected, {failed} failed, {skipped} skipped) "
                              f"- {remaining/60:.1f} min remaining")

            except Exception as e:
                logger.error(f"Error processing match {match['match_id']}: {e}")
                failed += 1
                continue

        # Final summary
        duration = (datetime.now() - start_time).total_seconds()
        logger.success(f"""
=== Weather data collection complete ===
Total matches: {total_matches}
Successfully collected: {collected}
Failed: {failed}
Skipped (no data): {skipped}
Duration: {duration/60:.1f} minutes
Rate: {total_matches/duration*60:.1f} matches/minute
        """)

        # Log to database
        self.db.log_collection(
            source='open_meteo',
            collection_type='weather_data',
            status='success' if failed == 0 else 'partial',
            records_collected=collected,
            error_message=f"{failed} failed, {skipped} skipped" if (failed > 0 or skipped > 0) else None,
            started_at=start_time
        )

    def validate_weather_data(self) -> Dict:
        """
        Validate collected weather data

        Returns:
            Dictionary with validation statistics
        """
        logger.info("Validating weather data...")

        query = """
            SELECT
                COUNT(*) as total_matches,
                COUNT(temperature_celsius) as with_temp,
                MIN(temperature_celsius) as min_temp,
                MAX(temperature_celsius) as max_temp,
                AVG(temperature_celsius) as avg_temp,
                MIN(humidity_percent) as min_humidity,
                MAX(humidity_percent) as max_humidity,
                AVG(precipitation_mm) as avg_precipitation,
                MAX(precipitation_mm) as max_precipitation
            FROM matches
            WHERE is_finished = 1
        """

        result = self.db.execute_query(query)[0]

        stats = {
            'total_matches': result['total_matches'],
            'with_weather': result['with_temp'],
            'coverage': result['with_temp'] / result['total_matches'] * 100 if result['total_matches'] > 0 else 0,
            'temp_range': (result['min_temp'], result['max_temp']),
            'avg_temp': result['avg_temp'],
            'humidity_range': (result['min_humidity'], result['max_humidity']),
            'avg_precipitation': result['avg_precipitation'],
            'max_precipitation': result['max_precipitation']
        }

        logger.info(f"""
=== Weather Data Validation ===
Total finished matches: {stats['total_matches']}
Matches with weather: {stats['with_weather']} ({stats['coverage']:.1f}%)
Temperature range: {stats['temp_range'][0]:.1f}°C to {stats['temp_range'][1]:.1f}°C
Average temperature: {stats['avg_temp']:.1f}°C
Humidity range: {stats['humidity_range'][0]:.1f}% to {stats['humidity_range'][1]:.1f}%
Average precipitation: {stats['avg_precipitation']:.2f}mm
Max precipitation: {stats['max_precipitation']:.1f}mm
        """)

        # Check for suspicious values
        warnings = []
        if stats['temp_range'][0] < -20:
            warnings.append(f"Suspiciously low temperature: {stats['temp_range'][0]:.1f}°C")
        if stats['temp_range'][1] > 45:
            warnings.append(f"Suspiciously high temperature: {stats['temp_range'][1]:.1f}°C")

        if warnings:
            logger.warning("Data quality warnings:")
            for warning in warnings:
                logger.warning(f"  - {warning}")

        return stats


def main():
    """Main execution"""
    logger.info("=== 3. Liga Weather Data Collector ===")

    collector = WeatherCollector()

    # Collect weather for all matches
    collector.collect_weather_for_all_matches(batch_size=50)

    # Validate results
    stats = collector.validate_weather_data()

    if stats['coverage'] >= 95:
        logger.success(f"✓ Weather data collection successful: {stats['coverage']:.1f}% coverage")
    elif stats['coverage'] >= 80:
        logger.warning(f"⚠ Weather data collection partial: {stats['coverage']:.1f}% coverage")
    else:
        logger.error(f"✗ Weather data collection incomplete: {stats['coverage']:.1f}% coverage")

    print(f"\n✓ Weather data collection complete!")
    print(f"  Coverage: {stats['with_weather']}/{stats['total_matches']} matches ({stats['coverage']:.1f}%)")
    print(f"  Temperature: {stats['temp_range'][0]:.1f}°C to {stats['temp_range'][1]:.1f}°C")
    print(f"  Average precipitation: {stats['avg_precipitation']:.2f}mm")


if __name__ == "__main__":
    main()
