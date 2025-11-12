"""
Build/expand team_locations by inferring each team's primary home venue and geocoding it.
Uses Open-Meteo Geocoding API (no API key).
"""
from pathlib import Path
from loguru import logger
import requests

from liga_predictor.database import get_db

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"


class TeamLocationBuilder:
    """Builds team location data from home venues"""

    def __init__(self):
        self.db = get_db()

    def geocode(self, name: str):
        """Geocode a location name using Open-Meteo API"""
        try:
            resp = requests.get(GEOCODE_URL, params={"name": name, "count": 1, "language": "de", "format": "json"}, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            if data and data.get("results"):
                r = data["results"][0]
                return float(r["latitude"]), float(r["longitude"]), r.get("name")
        except Exception as e:
            logger.debug(f"Geocode failed for '{name}': {e}")
        return None, None, None

    def build_locations(self):
        """Build team_locations from home venues"""
        logger.info("=== Building team_locations from home venues ===")
        # Ensure table exists
        self.db.execute_insert("""
            CREATE TABLE IF NOT EXISTS team_locations (
                team_id INTEGER PRIMARY KEY,
                team_name TEXT,
                lat REAL,
                lon REAL,
                source TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """, ())

        # Teams
        teams = self.db.execute_query("SELECT team_id, team_name FROM teams ORDER BY team_name")
        # Existing locations
        existing = {r["team_id"] for r in self.db.execute_query("SELECT team_id FROM team_locations WHERE lat IS NOT NULL AND lon IS NOT NULL")}

        inserted = 0
        skipped = 0
        for t in teams:
            tid = t["team_id"]
            if tid in existing:
                continue
            # Find most frequent home venue
            row = self.db.execute_query("""
                SELECT venue, COUNT(*) c
                FROM matches
                WHERE home_team_id = ? AND venue IS NOT NULL
                GROUP BY venue
                ORDER BY c DESC
                LIMIT 1
            """, (tid,))
            if not row or not row[0]["venue"]:
                skipped += 1
                continue
            venue = row[0]["venue"].strip()
            lat, lon, resolved = self.geocode(venue + ", Deutschland")
            if lat is None:
                # Try without country
                lat, lon, resolved = self.geocode(venue)
            if lat is None:
                skipped += 1
                continue
            self.db.execute_insert("""
                INSERT OR REPLACE INTO team_locations (team_id, team_name, lat, lon, source, updated_at)
                VALUES (?, ?, ?, ?, 'open-meteo-geocoding', CURRENT_TIMESTAMP)
            """, (tid, t["team_name"], lat, lon))
            inserted += 1
            if (inserted + skipped) % 10 == 0:
                logger.info(f"Progress: {inserted} inserted, {skipped} skipped")

        logger.success(f"team_locations update complete: {inserted} inserted/updated, {skipped} skipped")


def main():
    builder = TeamLocationBuilder()
    builder.build_locations()


if __name__ == "__main__":
    main()
