#!/usr/bin/env python3
"""
3. Liga Match Predictor - Production Script
Uses the winning Random Forest Classifier model to predict upcoming match results.

Usage:
    python main.py                    # Predict next matchday (default)
    python main.py predict            # Explicit prediction
    python main.py predict --update-data      # Update data before predicting
    python main.py predict --matchday 15      # Predict specific matchday
    python main.py predict --season 2025      # Predict for specific season
"""

import argparse
import sys
import pickle
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, List, Dict
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
from loguru import logger
import requests

# Add project to path
sys.path.append(str(Path(__file__).parent))

from database.db_manager import get_db
from kicktipp_predictor import config
from kicktipp_predictor.data_loader import prepare_features
from kicktipp_predictor.models.classifiers import ClassifierExperiment
from scripts.collectors.openligadb_collector import OpenLigaDBCollector
from scripts.processors.ml_data_exporter import MLDataExporter


class MatchPredictor:
    """Production match prediction system"""

    def __init__(self, model_path: str = "models/rf_classifier.pkl", weather_mode: str = "live", ext_data: bool = False):
        """
        Initialize predictor

        Args:
            model_path: Path to save/load trained model
            weather_mode: Weather fetching mode - 'live', 'estimate', or 'off'
            ext_data: If True, include heavy external data collection (FBref, matchday standings)
        """
        self.db = get_db()
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.features = None
        self.feature_medians = {}

        # Winner model configuration
        self.default_scores = config.DEFAULT_SCORES

        # Weather configuration
        self.weather_mode = weather_mode
        self._weather_cache = {}
        
        # External data collection flag
        self.ext_data = ext_data
        
        # Load stadium locations from JSON
        stadium_json_path = Path(__file__).parent / "config" / "stadium_locations.json"
        try:
            with open(stadium_json_path, 'r', encoding='utf-8') as f:
                self._stadium_locations = json.load(f)
            logger.debug(f"Loaded {len(self._stadium_locations.get('stadiums', {}))} stadium locations from JSON")
        except Exception as e:
            logger.warning(f"Could not load stadium_locations.json: {e}")
            self._stadium_locations = {'stadiums': {}}

        logger.info("Match Predictor initialized")

    def _get_finished_match_count(self) -> int:
        """
        Return the number of finished matches in the database.
        Used to detect if the DB has grown since last persisted model.
        """
        try:
            conn = self.db.get_connection()
            result = pd.read_sql_query(
                "SELECT COUNT(*) as cnt FROM matches WHERE is_finished = 1",
                conn
            )
            conn.close()
            if not result.empty:
                return int(result.iloc[0]['cnt'])
        except Exception:
            pass
        return 0

    def get_current_season(self) -> str:
        """Get the current season based on today's date"""
        today = datetime.now()
        year = today.year
        month = today.month

        # Season runs from July to June
        if month >= 7:
            season = f"{year}-{year+1}"
        else:
            season = f"{year-1}-{year}"

        return season

    def find_next_matchday(self, season: Optional[str] = None) -> Tuple[str, int, bool]:
        """
        Find the next matchday with unfinished matches

        Args:
            season: Specific season to check (default: current season)

        Returns:
            (season, matchday, has_data) tuple
        """
        if season is None:
            season = self.get_current_season()
        else:
            # Allow passing just the start year (e.g., "2025")
            try:
                if season.isdigit() and len(season) == 4:
                    start_year = int(season)
                    season = f"{start_year}-{start_year+1}"
            except Exception:
                pass

        conn = self.db.get_connection()

        # Find next matchday with unfinished matches
        query = """
        SELECT season, matchday,
               COUNT(*) as total_matches,
               SUM(CASE WHEN is_finished = 1 THEN 1 ELSE 0 END) as finished_matches,
               SUM(CASE WHEN is_finished = 0 THEN 1 ELSE 0 END) as upcoming_matches,
               MIN(match_datetime) as first_match_date
        FROM matches
        WHERE season = ?
        GROUP BY season, matchday
        ORDER BY matchday ASC
        """

        df = pd.read_sql_query(query, conn, params=(season,))
        conn.close()

        if df.empty:
            logger.warning(f"No data found for season {season}")
            return season, 1, False

        # Find first matchday with unfinished matches
        upcoming = df[df['upcoming_matches'] > 0]

        if not upcoming.empty:
            next_md = upcoming.iloc[0]
            logger.info(f"Next matchday: {season} MD {next_md['matchday']} "
                       f"({int(next_md['upcoming_matches'])} matches)")
            return season, int(next_md['matchday']), True

        # All matchdays finished, predict next one
        last_matchday = int(df['matchday'].max())
        next_matchday = last_matchday + 1
        logger.info(f"All matchdays finished. Next would be: {season} MD {next_matchday}")
        return season, next_matchday, False

    def check_data_availability(self, season: str, matchday: int) -> Dict:
        """
        Check if data is available for prediction

        Args:
            season: Season to check
            matchday: Matchday to check

        Returns:
            Dictionary with availability status
        """
        conn = self.db.get_connection()

        query = """
        SELECT
            COUNT(*) as match_count,
            COALESCE(SUM(CASE WHEN is_finished = 0 THEN 1 ELSE 0 END), 0) as upcoming,
            COALESCE(SUM(CASE WHEN is_finished = 1 THEN 1 ELSE 0 END), 0) as finished
        FROM matches
        WHERE season = ? AND matchday = ?
        """

        result = pd.read_sql_query(query, conn, params=(season, matchday))
        # Guard against NULLs/NANs
        result = result.fillna(0)
        conn.close()

        match_count = int(result.iloc[0]['match_count'])
        upcoming = int(result.iloc[0]['upcoming'])
        finished = int(result.iloc[0]['finished'])

        return {
            'exists': match_count > 0,
            'match_count': match_count,
            'upcoming': upcoming,
            'finished': finished,
            'can_predict': upcoming > 0
        }

    def load_upcoming_from_db(self, season: str, matchday: int) -> pd.DataFrame:
        """
        Load upcoming matches from database for a specific season and matchday

        Args:
            season: Season string (e.g., "2025-2026")
            matchday: Matchday number

        Returns:
            DataFrame with match info (empty if no matches found)
        """
        conn = self.db.get_connection()

        query = """
        SELECT m.match_id, m.season, m.matchday, m.match_datetime,
               ht.team_name AS home_team, at.team_name AS away_team,
               m.home_team_id, m.away_team_id
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.team_id
        JOIN teams at ON m.away_team_id = at.team_id
        WHERE m.season = ? AND m.matchday = ? AND m.is_finished = 0
        ORDER BY m.match_datetime
        """

        df = pd.read_sql_query(query, conn, params=(season, matchday))
        conn.close()

        if not df.empty:
            logger.info(f"Loaded {len(df)} upcoming matches from database for {season} MD {matchday}")

        return df

    def update_matchday_data(self, season: str, matchday: int) -> bool:
        """
        Orchestrate complete data acquisition for a matchday:
        1. Build team locations if needed (for travel distance)
        2. Backfill prior matchdays if needed for ratings
        3. Fetch and insert match schedule
        4. Collect league standings (OpenLigaDB)
        5. Collect referees (Transfermarkt)
        6. Collect betting odds via OddsPortal scraper
        7. Persist weather data (forecast <7d, estimate otherwise)
        8. Calculate ratings and head-to-head statistics
        9. Import detailed match statistics (CSV, if available)

        Args:
            season: Season (e.g., "2025-2026")
            matchday: Matchday number

        Returns:
            True if successful
        """
        logger.info(f"=== Updating data for {season} matchday {matchday} ===")

        try:
            # Extract year from season (e.g., "2025-2026" -> "2025")
            season_year = season.split('-')[0]

            collector = OpenLigaDBCollector()

            # Step 1: Build team locations (needed early for travel distance calculations)
            logger.info("Step 1: Checking team locations...")
            try:
                conn = self.db.get_connection()
                # Check if table exists first
                table_check = pd.read_sql_query("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='team_locations'
                """, conn)
                
                if not table_check.empty:
                    locs_check = pd.read_sql_query("""
                        SELECT COUNT(*) as count
                        FROM team_locations
                        WHERE lat IS NOT NULL AND lon IS NOT NULL
                    """, conn)
                    
                    if locs_check.iloc[0]['count'] == 0:
                        logger.info("Building team locations...")
                        from scripts.processors.build_team_locations import main as build_team_locs
                        build_team_locs()
                        logger.success("Team locations built")
                    else:
                        logger.debug("Team locations already populated, skipping")
                else:
                    # Table doesn't exist, create it and build locations
                    logger.info("Creating team_locations table and building locations...")
                    from scripts.processors.build_team_locations import main as build_team_locs
                    build_team_locs()
                    logger.success("Team locations built")
                conn.close()
            except Exception as e:
                logger.warning(f"Team location check/build failed: {e}")

            # Step 2: Check and backfill ALL missing matches across all seasons
            logger.info("Step 2: Checking for missing matches across all seasons...")
            conn = self.db.get_connection()
            
            try:
                # Check what seasons we have
                existing_seasons = pd.read_sql_query("""
                    SELECT DISTINCT season FROM matches ORDER BY season
                """, conn)
                
                # Check if we need to backfill historical data
                if existing_seasons.empty or len(existing_seasons) < 10:  # Should have ~15+ seasons
                    conn.close()
                    logger.info("Backfilling all historical data from 2009 to present...")
                    try:
                        from datetime import datetime
                        current_year = datetime.now().year
                        collector.collect_all_historical_data(start_year=2009, end_year=current_year)
                        logger.success("Historical data backfill completed")
                    except Exception as e:
                        logger.warning(f"Historical backfill failed: {e}")
                else:
                    # Check if current season's prior matchdays are missing
                    prior_check = pd.read_sql_query("""
                        SELECT DISTINCT matchday
                        FROM matches
                        WHERE season = ? AND matchday < ? AND is_finished = 1
                        ORDER BY matchday
                    """, conn, params=(season, matchday))

                    if prior_check.empty or len(prior_check) < (matchday - 1):
                        conn.close()
                        logger.info(f"Backfilling prior matchdays 1-{matchday-1} for {season}...")
                        try:
                            # Use full season fetch method for reliability
                            target_matchdays = list(range(1, matchday))
                            collector.collect_matchday_range_from_full_season(
                                season_year=season_year,
                                target_matchdays=target_matchdays
                            )
                        except Exception as e:
                            logger.warning(f"Backfill of prior matchdays failed: {e}")
                    else:
                        conn.close()
                    
                    # Also refresh any recently finished matches that might have updated results
                    logger.info("Step 2b: Refreshing recently finished matches...")
                    try:
                        from datetime import timedelta
                        cutoff_date = (datetime.now() - timedelta(days=7)).isoformat()
                        # Get matches that finished in last 7 days and refresh them
                        refresh_conn = self.db.get_connection()
                        recent_matches = pd.read_sql_query("""
                            SELECT DISTINCT season, matchday
                            FROM matches
                            WHERE is_finished = 1
                              AND match_datetime >= ?
                            ORDER BY season, matchday
                        """, refresh_conn, params=(cutoff_date,))
                        refresh_conn.close()
                        
                        if not recent_matches.empty:
                            for _, row in recent_matches.iterrows():
                                s = row['season']
                                md = row['matchday']
                                sy = s.split('-')[0]
                                try:
                                    raw = collector.get_matchdata_for_matchday(sy, md)
                                    if raw:
                                        parsed = [collector.parse_match_data(m, s) for m in raw if collector.parse_match_data(m, s)]
                                        if parsed:
                                            collector.insert_matches_to_db(parsed, raw_matches=raw)
                                except Exception as e:
                                    logger.debug(f"Failed to refresh {s} MD {md}: {e}")
                    except Exception as e:
                        logger.debug(f"Recent matches refresh failed: {e}")
            except Exception as e:
                logger.warning(f"Error in Step 2: {e}")
                try:
                    conn.close()
                except:
                    pass

            # Step 3: Fetch and insert target matchday fixtures
            logger.info(f"Step 3: Checking and fetching fixtures for {season} MD {matchday}...")
            
            # Check if matches already exist for this matchday
            conn = self.db.get_connection()
            try:
                existing_check = pd.read_sql_query("""
                    SELECT COUNT(*) as count
                    FROM matches
                    WHERE season = ? AND matchday = ?
                """, conn, params=(season, matchday))
                
                existing_count = existing_check.iloc[0]['count'] if not existing_check.empty else 0
                
                # 3. Liga has 20 teams = 10 matches per matchday
                expected_matches = 10
                
                if existing_count >= expected_matches:
                    logger.info(f"All {expected_matches} matches for {season} MD {matchday} already exist in database, skipping API call")
                    conn.close()
                else:
                    conn.close()
                    logger.info(f"Found {existing_count}/{expected_matches} matches, fetching remaining from API...")
                    raw_matches = collector.get_matchdata_for_matchday(season_year, matchday)

                    if not raw_matches:
                        logger.error(f"No matches found for {season} MD {matchday}")
                        return False

                    # Parse and insert matches
                    parsed_matches = []
                    for match in raw_matches:
                        parsed = collector.parse_match_data(match, season)
                        if parsed:
                            parsed_matches.append(parsed)

                    if parsed_matches:
                        # Pass raw_matches to enable event insertion
                        inserted = collector.insert_matches_to_db(parsed_matches, raw_matches=raw_matches)
                        logger.success(f"Inserted/updated {inserted} matches")
                    else:
                        logger.error("No valid matches could be parsed")
                        return False
            except Exception as e:
                logger.warning(f"Error checking existing matches: {e}")
                conn.close()
                # Fallback to fetching if check fails
                raw_matches = collector.get_matchdata_for_matchday(season_year, matchday)
                if raw_matches:
                    parsed_matches = []
                    for match in raw_matches:
                        parsed = collector.parse_match_data(match, season)
                        if parsed:
                            parsed_matches.append(parsed)
                    if parsed_matches:
                        inserted = collector.insert_matches_to_db(parsed_matches, raw_matches=raw_matches)
                        logger.success(f"Inserted/updated {inserted} matches")

            # Step 4: Collect league standings (OpenLigaDB)
            logger.info(f"Step 4: Checking and collecting league standings for {season}...")
            try:
                # Check if standings already exist for current/latest matchday
                conn = self.db.get_connection()
                try:
                    # First check if we have standings for the current matchday
                    current_matchday_check = pd.read_sql_query("""
                        SELECT COUNT(DISTINCT team_id) as team_count
                        FROM league_standings
                        WHERE season = ? AND matchday = ?
                    """, conn, params=(season, matchday))
                    
                    if not current_matchday_check.empty:
                        team_count = current_matchday_check.iloc[0]['team_count']
                        expected_teams = 20  # 3. Liga has 20 teams
                        
                        if team_count >= expected_teams:
                            logger.info(f"Standings for {season} MD {matchday} already exist ({team_count} teams), skipping API call")
                            conn.close()
                        else:
                            # Check latest matchday with standings
                            latest_standings_check = pd.read_sql_query("""
                                SELECT MAX(matchday) as max_matchday, COUNT(DISTINCT team_id) as team_count
                                FROM league_standings
                                WHERE season = ?
                            """, conn, params=(season,))
                            
                            if not latest_standings_check.empty:
                                max_matchday = latest_standings_check.iloc[0]['max_matchday']
                                latest_team_count = latest_standings_check.iloc[0]['team_count']
                                
                                # If we have standings for a later matchday with all teams, skip
                                if max_matchday is not None and max_matchday >= matchday and latest_team_count >= expected_teams:
                                    logger.info(f"Standings for {season} MD {max_matchday} already exist ({latest_team_count} teams), skipping API call")
                                    conn.close()
                                else:
                                    conn.close()
                                    logger.info(f"Standings incomplete or outdated (current MD {matchday}: {team_count} teams, latest MD {max_matchday}: {latest_team_count} teams), fetching from API...")
                                    standings_raw = collector.get_league_table(season_year)
                                    if standings_raw:
                                        standings_parsed = collector.parse_league_table(standings_raw, season)
                                        standings_inserted = collector.insert_standings_to_db(standings_parsed)
                                        logger.success(f"Inserted/updated {standings_inserted} league standings")
                            else:
                                conn.close()
                                logger.info("No standings found in database, fetching from API...")
                                standings_raw = collector.get_league_table(season_year)
                                if standings_raw:
                                    standings_parsed = collector.parse_league_table(standings_raw, season)
                                    standings_inserted = collector.insert_standings_to_db(standings_parsed)
                                    logger.success(f"Inserted/updated {standings_inserted} league standings")
                    else:
                        # No standings exist, fetch them
                        conn.close()
                        logger.info("No standings found in database, fetching from API...")
                        standings_raw = collector.get_league_table(season_year)
                        if standings_raw:
                            standings_parsed = collector.parse_league_table(standings_raw, season)
                            standings_inserted = collector.insert_standings_to_db(standings_parsed)
                            logger.success(f"Inserted/updated {standings_inserted} league standings")
                except Exception as check_error:
                    logger.warning(f"Error checking existing standings: {check_error}")
                    conn.close()
                    # Fallback to fetching if check fails
                    standings_raw = collector.get_league_table(season_year)
                    if standings_raw:
                        standings_parsed = collector.parse_league_table(standings_raw, season)
                        standings_inserted = collector.insert_standings_to_db(standings_parsed)
                        logger.success(f"Inserted/updated {standings_inserted} league standings")
            except Exception as e:
                logger.warning(f"League standings collection failed: {e}")

            # Step 5: Collect referees (FotMob - preferred over Transfermarkt)
            logger.info(f"Step 5: Collecting referees for {season} MD {matchday}...")
            try:
                from scripts.collectors.fotmob_collector import FotMobCollector
                fotmob_collector = FotMobCollector(use_selenium=True)
                # Collect referees from FotMob (more reliable than Transfermarkt scraping)
                referee_stats = fotmob_collector.collect_referees_for_matchday(
                    season=season,
                    matchday=matchday,
                    skip_existing=True
                )
                logger.success(f"Referee collection: {referee_stats.get('referees_updated', 0)} referees updated, {referee_stats.get('matches_skipped', 0)} skipped")
            except ImportError:
                logger.warning("FotMob collector not available, trying Transfermarkt fallback...")
                try:
                    from scripts.collectors.transfermarkt_referee_collector import TransfermarktRefereeCollector
                    referee_collector = TransfermarktRefereeCollector()
                    referee_stats = referee_collector.collect_matchday_range(
                        season=season,
                        start_matchday=matchday,
                        end_matchday=matchday,
                        use_match_reports=False,
                        skip_complete_matchdays=True,
                        skip_existing_referees=False
                    )
                    logger.success(f"Referee collection (Transfermarkt): {referee_stats.get('matches_updated', 0)} matches updated")
                except ImportError:
                    logger.warning("Transfermarkt referee collector also not available, skipping referee collection")
                except Exception as e:
                    logger.warning(f"Transfermarkt referee collection failed: {e}")
            except Exception as e:
                logger.warning(f"FotMob referee collection failed: {e}")

            # Step 6: Collect betting odds via OddsPortal scraper
            logger.info(f"Step 6: Collecting betting odds for {season} MD {matchday}...")
            try:
                from scripts.collectors.oddsportal_collector import OddsPortalCollector
                odds_collector = OddsPortalCollector(use_selenium=True)
                # Collect odds specifically for this matchday (more reliable than season-wide scraping)
                odds_stats = odds_collector.collect_matchday_odds(season, matchday, skip_existing=True)
                logger.success(f"Odds collection: {odds_stats.get('matches_inserted', 0)} matches inserted, {odds_stats.get('matches_skipped', 0)} skipped, {odds_stats.get('matches_not_found', 0)} not found")
            except ImportError:
                logger.warning("OddsPortal collector not available, skipping odds collection")
            except Exception as e:
                logger.warning(f"Odds collection failed: {e}")

            # Step 7: Persist weather data
            logger.info(f"Step 7: Persisting weather data for {season} MD {matchday}...")
            weather_updated = self.persist_weather_for_matchday(season, matchday)
            logger.success(f"Weather updated for {weather_updated} matches")
            
            # Also run weather processors for any missing historical weather
            logger.info("Step 7b: Running weather processors for missing historical weather...")
            try:
                # Call fetch_weather_multi programmatically via subprocess
                import subprocess
                weather_script = Path(__file__).parent / "scripts" / "processors" / "fetch_weather_multi.py"
                if weather_script.exists():
                    result = subprocess.run(
                        [sys.executable, str(weather_script), "--limit", "100"],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
                    if result.returncode == 0:
                        logger.success("Weather processors completed")
                    else:
                        logger.debug(f"Weather processors returned code {result.returncode}")
            except Exception as e:
                logger.debug(f"Weather processors failed: {e}")

            # Step 7c: Backfill weather_condition for processor-updated rows
            logger.info("Step 7c: Backfilling weather_condition for processor-updated rows...")
            try:
                conn = self.db.get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE matches
                    SET weather_condition = CASE
                        WHEN precipitation_mm > 5.0 THEN 'heavy_rain'
                        WHEN precipitation_mm > 0.5 THEN 'rain'
                        WHEN temperature_celsius <= 0 AND precipitation_mm > 0 THEN 'snow'
                        WHEN wind_speed_kmh > 40 THEN 'windy'
                        WHEN temperature_celsius > 25 THEN 'hot'
                        WHEN temperature_celsius < 5 THEN 'cold'
                        ELSE 'clear'
                    END,
                    updated_at = CURRENT_TIMESTAMP
                    WHERE weather_condition IS NULL
                      AND temperature_celsius IS NOT NULL
                """)
                rows_updated = cursor.rowcount
                conn.commit()
                conn.close()
                if rows_updated > 0:
                    logger.success(f"Filled weather_condition for {rows_updated} processor-updated rows")
                else:
                    logger.debug("No rows needed weather_condition backfill")
            except Exception as e:
                logger.debug(f"Weather condition backfill failed: {e}")

            # Step 8: Calculate ratings and head-to-head
            logger.info("Step 8: Calculating team ratings...")
            try:
                from scripts.processors.rating_calculator import RatingCalculator
                rating_calc = RatingCalculator()
                rating_calc.calculate_all_ratings(season=season)
                logger.success("Team ratings calculated")
                
                # Update latest ratings with corrected form metrics (including most recent matches)
                logger.info("Step 8b: Updating latest ratings with corrected form metrics...")
                try:
                    from scripts.processors.update_latest_ratings import update_latest_ratings
                    update_latest_ratings(season=season)
                    logger.success("Latest ratings updated")
                except Exception as e:
                    logger.warning(f"Latest ratings update failed: {e}")
            except Exception as e:
                logger.warning(f"Rating calculation failed: {e}")

            logger.info("Step 9: Building head-to-head statistics...")
            try:
                from scripts.processors.build_head_to_head import compute_h2h
                compute_h2h()
                logger.success("Head-to-head statistics updated")
            except Exception as e:
                logger.warning(f"H2H calculation failed: {e}")

            # Step 9b: Build matchday-level standings (only if --ext-data flag is set)
            if self.ext_data:
                logger.info(f"Step 9b: Building matchday-level standings for {season}...")
                try:
                    self._build_matchday_standings(season)
                    logger.success("Matchday-level standings built")
                except Exception as e:
                    logger.warning(f"Matchday standings build failed: {e}")
            else:
                logger.info("Step 9b: Skipping matchday-level standings (--ext-data not set)")

            # Step 10: Collect FBref data (standings, team stats, player stats) (only if --ext-data flag is set)
            if self.ext_data:
                logger.info(f"Step 10: Collecting FBref data for {season}...")
                try:
                    from scripts.collectors.fbref_collector import FBrefCollector
                    fbref_collector = FBrefCollector(use_selenium=False)  # Try without Selenium first
                    fbref_stats = fbref_collector.collect_season_data(season)
                    logger.success(f"FBref collection: {fbref_stats.get('standings', {}).get('teams_collected', 0)} teams, "
                                 f"{fbref_stats.get('player_stats', {}).get('players_collected', 0)} players")
                except ImportError:
                    logger.debug("FBref collector not available, skipping FBref collection")
                except Exception as e:
                    logger.warning(f"FBref collection failed (may need Selenium): {e}")
                    # Retry with Selenium if non-Selenium attempt failed
                    try:
                        logger.info("Retrying FBref collection with Selenium...")
                        fbref_collector = FBrefCollector(use_selenium=True)
                        fbref_stats = fbref_collector.collect_season_data(season)
                        logger.success(f"FBref collection (Selenium): {fbref_stats.get('standings', {}).get('teams_collected', 0)} teams, "
                                     f"{fbref_stats.get('player_stats', {}).get('players_collected', 0)} players")
                    except Exception as e2:
                        logger.warning(f"FBref collection failed with Selenium as well: {e2}")
            else:
                logger.info("Step 10: Skipping FBref collection (--ext-data not set)")

            # Step 11: Import detailed match statistics (CSV, if available)
            logger.info("Step 11: Checking for detailed match statistics (CSV import)...")
            try:
                from scripts.processors.import_existing_data import DataImporter
                importer = DataImporter()
                # Check if CSV file exists
                csv_path = Path("data/raw/fotmob_stats_all.csv")
                if csv_path.exists():
                    logger.info("Found FotMob stats CSV, importing...")
                    stats_count = importer.import_fotmob_statistics()
                    logger.success(f"Imported {stats_count} detailed statistics records")
                else:
                    logger.debug("No FotMob stats CSV found, skipping detailed stats import")
            except ImportError:
                logger.debug("Data importer not available, skipping detailed stats import")
            except Exception as e:
                logger.debug(f"Detailed stats import failed: {e}")

            logger.success(f"=== Data update complete for {season} MD {matchday} ===")
            return True

        except Exception as e:
            logger.error(f"Error updating data: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _build_matchday_standings(self, season: str) -> int:
        """
        Build league standings for each matchday by computing from match results
        
        Args:
            season: Season string (e.g., "2025-2026")
            
        Returns:
            Number of standings records inserted/updated
        """
        conn = self.db.get_connection()
        
        # Get all finished matches for this season, ordered by matchday
        matches_df = pd.read_sql_query("""
            SELECT matchday, home_team_id, away_team_id, home_goals, away_goals, result
            FROM matches
            WHERE season = ? AND is_finished = 1 AND home_goals IS NOT NULL AND away_goals IS NOT NULL
            ORDER BY matchday ASC, match_datetime ASC
        """, conn, params=(season,))
        
        if matches_df.empty:
            conn.close()
            return 0
        
        # Get all unique matchdays
        matchdays = sorted(matches_df['matchday'].unique())
        
        # Track cumulative standings per team per matchday
        # We'll build cumulatively: stats for matchday N include all matches up to N
        team_stats = {}  # team_id -> current cumulative stats
        
        inserted = 0
        
        for matchday in matchdays:
            # Process all matches up to and including this matchday
            matches_up_to = matches_df[matches_df['matchday'] <= matchday]
            
            # Reset stats for this matchday (we'll rebuild from scratch up to this point)
            # Actually, we need to process incrementally - only process new matches since last matchday
            # For simplicity, rebuild from scratch each time (inefficient but correct)
            team_stats = {}
            
            for _, match in matches_up_to.iterrows():
                home_id = match['home_team_id']
                away_id = match['away_team_id']
                home_goals = int(match['home_goals'])
                away_goals = int(match['away_goals'])
                result = match['result']
                
                # Initialize if needed
                if home_id not in team_stats:
                    team_stats[home_id] = {
                        'matches_played': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                        'goals_for': 0, 'goals_against': 0, 'points': 0
                    }
                if away_id not in team_stats:
                    team_stats[away_id] = {
                        'matches_played': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                        'goals_for': 0, 'goals_against': 0, 'points': 0
                    }
                
                # Update home team stats
                home_stats = team_stats[home_id]
                home_stats['matches_played'] += 1
                home_stats['goals_for'] += home_goals
                home_stats['goals_against'] += away_goals
                
                if result == 'H':
                    home_stats['wins'] += 1
                    home_stats['points'] += 3
                elif result == 'D':
                    home_stats['draws'] += 1
                    home_stats['points'] += 1
                else:  # 'A'
                    home_stats['losses'] += 1
                
                # Update away team stats
                away_stats = team_stats[away_id]
                away_stats['matches_played'] += 1
                away_stats['goals_for'] += away_goals
                away_stats['goals_against'] += home_goals
                
                if result == 'A':
                    away_stats['wins'] += 1
                    away_stats['points'] += 3
                elif result == 'D':
                    away_stats['draws'] += 1
                    away_stats['points'] += 1
                else:  # 'H'
                    away_stats['losses'] += 1
            
            # Calculate goal difference and sort by points, goal difference, goals for
            standings_list = []
            for team_id, stats in team_stats.items():
                stats_copy = stats.copy()
                stats_copy['goal_difference'] = stats_copy['goals_for'] - stats_copy['goals_against']
                standings_list.append((team_id, stats_copy))
            
            # Sort: points DESC, goal_diff DESC, goals_for DESC
            standings_list.sort(key=lambda x: (x[1]['points'], x[1]['goal_difference'], x[1]['goals_for']), reverse=True)
            
            # Insert standings with positions for this matchday
            for position, (team_id, stats) in enumerate(standings_list, start=1):
                query = """
                    INSERT OR REPLACE INTO league_standings
                    (season, matchday, team_id, position, matches_played, wins, draws, losses,
                     goals_for, goals_against, goal_difference, points)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                try:
                    conn.execute(query, (
                        season, matchday, team_id, position,
                        stats['matches_played'], stats['wins'], stats['draws'], stats['losses'],
                        stats['goals_for'], stats['goals_against'], stats['goal_difference'], stats['points']
                    ))
                    inserted += 1
                except Exception as e:
                    logger.debug(f"Error inserting standing for team {team_id} matchday {matchday}: {e}")
        
        conn.commit()
        conn.close()
        
        return inserted

    def export_training_data(self) -> pd.DataFrame:
        """
        Export all completed matches for training

        Returns:
            DataFrame with all training data
        """
        logger.info("Exporting training data from database...")

        exporter = MLDataExporter()
        df = exporter.export_comprehensive_dataset(min_season="2009-2010")

        # Only use finished matches for training (no data leakage)
        finished_matches = df[df['result'].notna()].copy()

        logger.info(f"Exported {len(finished_matches)} completed matches for training")

        return finished_matches

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple:
        """
        Prepare features and targets for training

        Args:
            df: Raw dataset

        Returns:
            X, y_class, y_home, y_away, features
        """
        # Choose features (numerical only for Random Forest)
        features = [f for f in config.NUMERICAL_FEATURES if f in df.columns]

        # Handle missing values
        for feat in features:
            if feat in config.CATEGORICAL_FEATURES:
                df[feat] = df[feat].fillna('MISSING')
            else:
                median_val = df[feat].median()
                self.feature_medians[feat] = median_val
                df[feat] = df[feat].fillna(median_val)

        # Prepare features and targets
        X = df[features].copy()

        # Create multiclass target from result
        result_map = {'A': 0, 'D': 1, 'H': 2}
        y_class = df['result'].map(result_map)

        y_home = df['home_goals']
        y_away = df['away_goals']

        self.features = features

        logger.info(f"Prepared {len(features)} features for training")

        return X, y_class, y_home, y_away, features

    def train_model(self, force_retrain: bool = False, cutoff_datetime: Optional[pd.Timestamp] = None, persist: bool = True) -> ClassifierExperiment:
        """
        Train or load the Random Forest Classifier

        Args:
            force_retrain: Force retraining even if model exists
            cutoff_datetime: If provided, only train on matches strictly before this datetime
            persist: When True, save the trained model to disk

        Returns:
            Trained model
        """
        # If no cutoff and a saved model exists (and not forcing retrain), load it,
        # then auto-retrain if DB has grown since last persistence
        if cutoff_datetime is None and self.model_path.exists() and not force_retrain:
            logger.info(f"Loading existing model from {self.model_path}")
            with open(self.model_path, 'rb') as f:
                saved = pickle.load(f)
                self.model = saved['model']
                self.features = saved['features']
                self.feature_medians = saved['feature_medians']
                self.default_scores = saved['default_scores']
                saved_n = int(saved.get('n_samples', 0))
            # Prefer comparing against current training dataset size (same logic as training)
            try:
                current_n = len(self.export_training_data())
            except Exception:
                # Fallback to DB finished count if exporter fails
                current_n = self._get_finished_match_count()
            if current_n <= saved_n:
                logger.info(f"Model up-to-date (n_samples={saved_n}, current_trainable={current_n}). Using saved model.")
                return self.model
            else:
                logger.info(
                    f"Auto-retrain triggered: trainable samples grew {saved_n} -> {current_n}. Retraining and persisting..."
                )

        # Train new model
        if cutoff_datetime is not None:
            logger.info(f"Training model up to cutoff {pd.to_datetime(cutoff_datetime)} (no leakage)")
        else:
            logger.info("Training new Random Forest Classifier...")

        # Get all training data (only completed matches)
        df = self.export_training_data()
        # Apply cutoff if requested (strictly before)
        if cutoff_datetime is not None:
            df['match_datetime'] = pd.to_datetime(df['match_datetime'])
            before_count = len(df)
            cdt = pd.to_datetime(cutoff_datetime)
            # Drop timezone to compare against naive datetimes in df
            try:
                cdt = cdt.tz_localize(None)
            except TypeError:
                pass  # already naive
            df = df[df['match_datetime'] < cdt]
            logger.info(f"Filtered training data by cutoff: {before_count} -> {len(df)} rows")

        # Prepare features
        X, y_class, y_home, y_away, features = self.prepare_training_data(df)

        # Train Random Forest
        experiment = ClassifierExperiment(default_scores=self.default_scores)
        model = experiment.train_random_forest(X, y_class)

        self.model = experiment

        # Save model if requested
        if persist:
            logger.info(f"Saving model to {self.model_path}")
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': experiment,
                    'features': features,
                    'feature_medians': self.feature_medians,
                    'default_scores': self.default_scores,
                    'trained_on': datetime.now().isoformat(),
                    'n_samples': len(X)
                }, f)
            logger.success(f"Model trained on {len(X)} matches and saved")
        else:
            logger.success(f"Model trained on {len(X)} matches (not persisted)")

        return experiment

    def fetch_future_matches(self, season: str, matchday: int) -> pd.DataFrame:
        """
        Fetch future matches from OpenLigaDB API

        Args:
            season: Season (e.g., "2025-2026")
            matchday: Matchday number

        Returns:
            DataFrame with basic match info
        """
        logger.info(f"Fetching matches from API for {season} MD {matchday}...")

        # Extract year from season
        season_year = season.split('-')[0]

        collector = OpenLigaDBCollector()
        api_matches = collector.get_matchdata_for_matchday(season_year, matchday)

        if not api_matches:
            logger.error(f"No matches found in API for {season} MD {matchday}")
            return pd.DataFrame()

        # Parse matches and get team names from database
        conn = self.db.get_connection()
        matches = []

        for match in api_matches:
            parsed = collector.parse_match_data(match, season)

            # Skip matches that couldn't be parsed
            if parsed is None:
                continue

            # Extract basic match info
            try:
                home_team_id = parsed.get('home_team_id')
                away_team_id = parsed.get('away_team_id')

                # Get team names from database
                home_team_name = None
                away_team_name = None

                if home_team_id:
                    result = pd.read_sql_query(
                        "SELECT team_name FROM teams WHERE team_id = ?",
                        conn,
                        params=(home_team_id,)
                    )
                    if not result.empty:
                        home_team_name = result.iloc[0]['team_name']

                if away_team_id:
                    result = pd.read_sql_query(
                        "SELECT team_name FROM teams WHERE team_id = ?",
                        conn,
                        params=(away_team_id,)
                    )
                    if not result.empty:
                        away_team_name = result.iloc[0]['team_name']

                # Try to find match_id in database if match already exists
                match_id = None
                if home_team_id and away_team_id and parsed.get('match_datetime'):
                    match_id_result = pd.read_sql_query(
                        """
                        SELECT match_id FROM matches
                        WHERE season = ? AND matchday = ?
                          AND home_team_id = ? AND away_team_id = ?
                          AND match_datetime = ?
                        LIMIT 1
                        """,
                        conn,
                        params=(season, matchday, home_team_id, away_team_id, parsed.get('match_datetime'))
                    )
                    if not match_id_result.empty:
                        match_id = int(match_id_result.iloc[0]['match_id'])

                match_info = {
                    'season': season,
                    'matchday': matchday,
                    'match_datetime': parsed.get('match_datetime'),
                    'home_team': home_team_name,
                    'away_team': away_team_name,
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                }
                if match_id is not None:
                    match_info['match_id'] = match_id

                # Only add if we have essential info
                if match_info['home_team'] and match_info['away_team'] and match_info['match_datetime']:
                    matches.append(match_info)
                else:
                    logger.warning(f"Skipping match with incomplete data: home={home_team_name}, away={away_team_name}")

            except Exception as e:
                logger.warning(f"Error parsing match: {e}")
                continue

        conn.close()

        if not matches:
            logger.error(f"No valid matches could be parsed from API")
            return pd.DataFrame()

        df = pd.DataFrame(matches)
        logger.success(f"Fetched {len(df)} future matches from API")

        return df

    def _calculate_rest_days(self, team_name: str, match_date: pd.Timestamp, conn) -> int:
        """
        Calculate rest days for a team (days since last match)

        Args:
            team_name: Team name
            match_date: Upcoming match date
            conn: Database connection

        Returns:
            Number of rest days (no cap - can be >7 during breaks)
        """
        try:
            # Get team_id for querying
            team_id_result = pd.read_sql_query(
                "SELECT team_id FROM teams WHERE team_name = ? LIMIT 1",
                conn,
                params=(team_name,)
            )

            if team_id_result.empty:
                logger.warning(f"Team not found: {team_name}, using default rest days")
                return 7  # Default only if team not found

            team_id = team_id_result.iloc[0]['team_id']

            # Query for last match
            result = pd.read_sql_query(
                """
                SELECT MAX(match_datetime) as last_match
                FROM matches
                WHERE (home_team_id = ? OR away_team_id = ?)
                  AND match_datetime < ?
                  AND is_finished = 1
                """,
                conn,
                params=(team_id, team_id, match_date.isoformat())
            )

            if not result.empty and result.iloc[0]['last_match'] is not None:
                last_match = pd.to_datetime(result.iloc[0]['last_match'])
                rest_days = (match_date - last_match).days
                # Return actual rest days without any cap (can be >7 during breaks)
                return max(0, rest_days)  # Ensure non-negative

        except Exception as e:
            logger.warning(f"Could not calculate rest days for {team_name}: {e}")

        # Only default to 7 if truly no previous match found (should be rare)
        # During breaks, previous matches should exist, so this shouldn't be hit
        return 7

    def _calculate_travel_distance(self, home_team: str, away_team: str, conn) -> float:
        """
        Calculate travel distance between team locations using haversine formula.
        Uses team_locations table for consistent distance calculation.

        Args:
            home_team: Home team name
            away_team: Away team name
            conn: Database connection

        Returns:
            Distance in kilometers (default to 300 if not available)
        """
        try:
            import numpy as np
            
            # Get team coordinates from team_locations table
            home_coords = pd.read_sql_query(
                """
                SELECT lat, lon
                FROM team_locations tl
                JOIN teams t ON tl.team_id = t.team_id
                WHERE t.team_name = ? AND tl.lat IS NOT NULL AND tl.lon IS NOT NULL
                LIMIT 1
                """,
                conn,
                params=(home_team,)
            )

            away_coords = pd.read_sql_query(
                """
                SELECT lat, lon
                FROM team_locations tl
                JOIN teams t ON tl.team_id = t.team_id
                WHERE t.team_name = ? AND tl.lat IS NOT NULL AND tl.lon IS NOT NULL
                LIMIT 1
                """,
                conn,
                params=(away_team,)
            )

            if not home_coords.empty and not away_coords.empty:
                lat1 = home_coords.iloc[0]['lat']
                lon1 = home_coords.iloc[0]['lon']
                lat2 = away_coords.iloc[0]['lat']
                lon2 = away_coords.iloc[0]['lon']

                if pd.notna(lat1) and pd.notna(lon1) and pd.notna(lat2) and pd.notna(lon2):
                    # Haversine formula for great-circle distance
                    R = 6371.0  # Earth radius in km
                    lat1_rad = np.radians(lat1)
                    lon1_rad = np.radians(lon1)
                    lat2_rad = np.radians(lat2)
                    lon2_rad = np.radians(lon2)
                    
                    dlat = lat2_rad - lat1_rad
                    dlon = lon2_rad - lon1_rad
                    
                    a = np.sin(dlat/2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2.0)**2
                    c = 2 * np.arcsin(np.sqrt(a))
                    distance = R * c
                    
                    return round(float(distance), 2)

        except Exception as e:
            logger.debug(f"Could not calculate travel distance for {home_team} vs {away_team}: {e}")

        return 300.0  # Default to 300km

    def _get_head_to_head_stats(self, home_team: str, away_team: str, conn) -> dict:
        """
        Get head-to-head statistics between two teams

        Args:
            home_team: Home team name
            away_team: Away team name
            conn: Database connection

        Returns:
            Dictionary with h2h statistics
        """
        try:
            # Query head_to_head table
            result = pd.read_sql_query(
                """
                SELECT team_a_wins, draws, team_b_wins, total_matches
                FROM head_to_head h
                WHERE (
                    (h.team_a_id = (SELECT team_id FROM teams WHERE team_name = ? LIMIT 1)
                     AND h.team_b_id = (SELECT team_id FROM teams WHERE team_name = ? LIMIT 1))
                    OR
                    (h.team_a_id = (SELECT team_id FROM teams WHERE team_name = ? LIMIT 1)
                     AND h.team_b_id = (SELECT team_id FROM teams WHERE team_name = ? LIMIT 1))
                )
                LIMIT 1
                """,
                conn,
                params=(home_team, away_team, away_team, home_team)
            )

            if not result.empty:
                row = result.iloc[0]
                total = row['total_matches']

                if total > 0:
                    # Determine which team is which in the database
                    home_is_team_a = True  # We'll figure this out

                    # Try to determine orientation
                    result2 = pd.read_sql_query(
                        """
                        SELECT team_a_id, team_b_id
                        FROM head_to_head
                        WHERE team_a_id = (SELECT team_id FROM teams WHERE team_name = ? LIMIT 1)
                          AND team_b_id = (SELECT team_id FROM teams WHERE team_name = ? LIMIT 1)
                        LIMIT 1
                        """,
                        conn,
                        params=(home_team, away_team)
                    )

                    if result2.empty:
                        # Teams are swapped in database
                        home_is_team_a = False

                    if home_is_team_a:
                        home_wins = row['team_a_wins']
                        away_wins = row['team_b_wins']
                    else:
                        home_wins = row['team_b_wins']
                        away_wins = row['team_a_wins']

                    draws = row['draws']

                    return {
                        'h2h_home_wins': home_wins,
                        'h2h_away_wins': away_wins,
                        'h2h_home_win_rate': home_wins / total if total > 0 else 0,
                        'h2h_draw_rate': draws / total if total > 0 else 0,
                        'h2h_match_count': total
                    }

        except Exception as e:
            logger.warning(f"Could not get h2h stats for {home_team} vs {away_team}: {e}")

        # Default values
        return {
            'h2h_home_wins': 0,
            'h2h_away_wins': 0,
            'h2h_home_win_rate': 0.0,
            'h2h_draw_rate': 0.0,
            'h2h_match_count': 0
        }

    def _resolve_stadium_coords(self, home_team: str, conn) -> Tuple[Optional[float], Optional[float]]:
        """
        Resolve stadium coordinates for a team.
        
        Tries JSON config first, then database team_locations table.
        
        Args:
            home_team: Home team name
            conn: Database connection
            
        Returns:
            Tuple of (latitude, longitude) or (None, None) if not found
        """
        # Try JSON config first (case-insensitive matching)
        stadiums = self._stadium_locations.get('stadiums', {})
        for team_name, team_data in stadiums.items():
            if team_name.lower() == home_team.lower():
                lat = team_data.get('latitude')
                lon = team_data.get('longitude')
                if lat is not None and lon is not None:
                    logger.debug(f"Found coordinates for {home_team} in JSON config: ({lat}, {lon})")
                    return float(lat), float(lon)
        
        # Fallback to database
        try:
            result = pd.read_sql_query(
                """
                SELECT lat, lon
                FROM team_locations
                WHERE team_id = (SELECT team_id FROM teams WHERE team_name = ? LIMIT 1)
                  AND lat IS NOT NULL AND lon IS NOT NULL
                LIMIT 1
                """,
                conn,
                params=(home_team,)
            )
            
            if not result.empty:
                lat = result.iloc[0]['lat']
                lon = result.iloc[0]['lon']
                if pd.notna(lat) and pd.notna(lon):
                    logger.debug(f"Found coordinates for {home_team} in database: ({lat}, {lon})")
                    return float(lat), float(lon)
        except Exception as e:
            logger.debug(f"Could not query team_locations for {home_team}: {e}")
        
        logger.warning(f"Could not resolve coordinates for {home_team}")
        return None, None

    def _estimate_weather(self, match_date: pd.Timestamp, conn) -> dict:
        """
        Estimate weather for future match using historical seasonal patterns

        Uses regression on historical weather data for the same month/time period
        to predict weather conditions.

        Args:
            match_date: Match datetime
            conn: Database connection

        Returns:
            Dictionary with weather features (estimated from historical patterns)
        """
        try:
            # Extract month and approximate day range (±14 days window)
            target_month = match_date.month
            target_day = match_date.day

            # Query historical weather for same month/day range across all years
            result = pd.read_sql_query(
                """
                SELECT
                    AVG(temperature_celsius) as avg_temp,
                    AVG(humidity_percent) as avg_humidity,
                    AVG(wind_speed_kmh) as avg_wind,
                    AVG(precipitation_mm) as avg_precip,
                    COUNT(*) as sample_count
                FROM matches
                WHERE strftime('%m', match_datetime) = ?
                  AND ABS(CAST(strftime('%d', match_datetime) AS INTEGER) - ?) <= 14
                  AND temperature_celsius IS NOT NULL
                  AND is_finished = 1
                """,
                conn,
                params=(f"{target_month:02d}", target_day)
            )

            if not result.empty and result.iloc[0]['sample_count'] > 5:  # Need at least 5 samples
                row = result.iloc[0]
                logger.debug(
                    f"Using weather estimate from {int(row['sample_count'])} historical matches "
                    f"in similar time period (month {target_month}, day ~{target_day})"
                )

                return {
                    'temperature_celsius': round(row['avg_temp'], 1) if pd.notna(row['avg_temp']) else 15.0,
                    'humidity_percent': round(row['avg_humidity'], 1) if pd.notna(row['avg_humidity']) else 70.0,
                    'wind_speed_kmh': round(row['avg_wind'], 1) if pd.notna(row['avg_wind']) else 10.0,
                    'precipitation_mm': round(row['avg_precip'], 2) if pd.notna(row['avg_precip']) else 0.0,
                }

        except Exception as e:
            logger.debug(f"Could not estimate weather from historical patterns: {e}")

        # Fallback: Use seasonal defaults based on month
        seasonal_defaults = {
            12: {'temp': 3.0, 'humidity': 85, 'wind': 15, 'precip': 1.5},   # Winter
            1:  {'temp': 2.0, 'humidity': 85, 'wind': 15, 'precip': 1.5},
            2:  {'temp': 4.0, 'humidity': 80, 'wind': 14, 'precip': 1.2},
            3:  {'temp': 8.0, 'humidity': 75, 'wind': 13, 'precip': 1.0},   # Spring
            4:  {'temp': 12.0, 'humidity': 70, 'wind': 12, 'precip': 0.8},
            5:  {'temp': 16.0, 'humidity': 68, 'wind': 11, 'precip': 0.9},
            6:  {'temp': 20.0, 'humidity': 65, 'wind': 10, 'precip': 0.7},  # Summer
            7:  {'temp': 22.0, 'humidity': 65, 'wind': 10, 'precip': 0.8},
            8:  {'temp': 21.0, 'humidity': 68, 'wind': 10, 'precip': 0.9},
            9:  {'temp': 17.0, 'humidity': 72, 'wind': 11, 'precip': 0.8},  # Autumn
            10: {'temp': 12.0, 'humidity': 78, 'wind': 12, 'precip': 1.0},
            11: {'temp': 7.0, 'humidity': 82, 'wind': 14, 'precip': 1.3},
        }

        month = match_date.month
        defaults = seasonal_defaults.get(month, seasonal_defaults[5])  # Default to May

        logger.debug(f"Using seasonal default weather for month {month}")

        return {
            'temperature_celsius': defaults['temp'],
            'humidity_percent': defaults['humidity'],
            'wind_speed_kmh': defaults['wind'],
            'precipitation_mm': defaults['precip'],
        }

    def _get_live_weather(self, match_dt: pd.Timestamp, home_team: str, conn) -> Optional[dict]:
        """
        Fetch live weather forecast from Open-Meteo API for a future match.
        
        Uses forecast API for future dates, with caching to avoid repeated API calls.
        Falls back to None if coordinates unavailable or API fails.
        
        Args:
            match_dt: Match datetime (should be timezone-aware)
            home_team: Home team name
            conn: Database connection
            
        Returns:
            Dictionary with weather features or None if unavailable
        """
        # Resolve coordinates
        lat, lon = self._resolve_stadium_coords(home_team, conn)
        if lat is None or lon is None:
            logger.debug(f"Cannot fetch live weather: no coordinates for {home_team}")
            return None
        
        # Ensure timezone-aware datetime (Europe/Berlin)
        if match_dt.tzinfo is None:
            match_dt = match_dt.tz_localize(ZoneInfo("Europe/Berlin"))
        else:
            match_dt = match_dt.astimezone(ZoneInfo("Europe/Berlin"))
        
        # Check cache
        cache_key = (round(lat, 4), round(lon, 4), match_dt.strftime('%Y-%m-%dT%H'))
        if cache_key in self._weather_cache:
            logger.debug(f"Using cached weather for {home_team} at {match_dt}")
            return self._weather_cache[cache_key]
        
        # Calculate forecast_days (max 16 days for free API)
        # Use date-based difference to avoid floor from Timedelta.days
        now = pd.Timestamp.now(ZoneInfo("Europe/Berlin"))
        days_ahead = (match_dt.date() - now.date()).days
        if days_ahead < 0:
            logger.debug(f"Match date {match_dt} is in the past, cannot use forecast API")
            return None
        if days_ahead > 16:
            logger.debug(f"Match date {match_dt} is more than 16 days ahead, forecast unavailable")
            return None
        
        # Build API request
        api_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "timezone": "Europe/Berlin",
            "windspeed_unit": "kmh",
            "precipitation_unit": "mm",
            # Include all days up to and including the match date
            "forecast_days": min(max(days_ahead, 0) + 1, 16)
        }
        
        # Make API call with retry
        session = requests.Session()
        session.headers.update({"User-Agent": "3Liga-Predictor/1.0"})
        
        for attempt in range(2):  # Max 2 attempts
            try:
                response = session.get(api_url, params=params, timeout=10)
                if response.status_code == 429:
                    logger.debug("Rate limited, waiting before retry...")
                    time.sleep(1.5 + attempt)
                    continue
                response.raise_for_status()
                data = response.json()
                break
            except requests.exceptions.RequestException as e:
                if attempt == 1:
                    logger.debug(f"Open-Meteo API error for {home_team}: {e}")
                    return None
                time.sleep(0.5)
        else:
            return None
        
        # Extract hourly data
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            logger.debug(f"No hourly data in API response for {home_team}")
            return None
        
        # Find exact hour or nearest same-day hour
        target_hour_str = match_dt.strftime("%Y-%m-%dT%H:00")
        exact_hour = False
        idx = None
        
        try:
            idx = times.index(target_hour_str)
            exact_hour = True
        except ValueError:
            # Find nearest hour on same day
            target_date = match_dt.strftime("%Y-%m-%d")
            for i, t in enumerate(times):
                if t.startswith(target_date):
                    idx = i
                    break
        
        if idx is None:
            logger.debug(f"Could not find matching hour for {match_dt} in API response")
            return None
        
        # Extract weather values
        temps = hourly.get("temperature_2m", [])
        humidities = hourly.get("relative_humidity_2m", [])
        precipitations = hourly.get("precipitation", [])
        wind_speeds = hourly.get("wind_speed_10m", [])
        
        weather_data = {
            'temperature_celsius': temps[idx] if idx < len(temps) and temps[idx] is not None else None,
            'humidity_percent': humidities[idx] if idx < len(humidities) and humidities[idx] is not None else None,
            'precipitation_mm': precipitations[idx] if idx < len(precipitations) and precipitations[idx] is not None else None,
            'wind_speed_kmh': wind_speeds[idx] if idx < len(wind_speeds) and wind_speeds[idx] is not None else None,
        }
        
        # Check if we got valid data
        if all(v is None for v in weather_data.values()):
            logger.debug(f"No valid weather data in API response for {home_team}")
            return None
        
        # Fill None values with defaults if needed
        if weather_data['temperature_celsius'] is None:
            weather_data['temperature_celsius'] = 15.0
        if weather_data['humidity_percent'] is None:
            weather_data['humidity_percent'] = 70.0
        if weather_data['wind_speed_kmh'] is None:
            weather_data['wind_speed_kmh'] = 10.0
        if weather_data['precipitation_mm'] is None:
            weather_data['precipitation_mm'] = 0.0
        
        # Round values
        weather_data['temperature_celsius'] = round(weather_data['temperature_celsius'], 1)
        weather_data['humidity_percent'] = round(weather_data['humidity_percent'], 1)
        weather_data['wind_speed_kmh'] = round(weather_data['wind_speed_kmh'], 1)
        weather_data['precipitation_mm'] = round(weather_data['precipitation_mm'], 2)
        
        # Cache result
        self._weather_cache[cache_key] = weather_data
        
        source_note = "exact hour" if exact_hour else "nearest hour"
        logger.debug(f"Fetched live weather for {home_team} ({source_note}): "
                    f"temp={weather_data['temperature_celsius']}°C, "
                    f"humidity={weather_data['humidity_percent']}%, "
                    f"wind={weather_data['wind_speed_kmh']}km/h, "
                    f"precip={weather_data['precipitation_mm']}mm")
        
        return weather_data

    def persist_weather_for_matchday(self, season: str, matchday: int) -> int:
        """
        Persist weather data for all matches in a matchday to the database
        
        For matches < 7 days away: fetch forecast from Open-Meteo
        For matches >= 7 days away: use historical estimate
        
        Args:
            season: Season string
            matchday: Matchday number
            
        Returns:
            Number of matches updated
        """
        logger.info(f"Persisting weather data for {season} MD {matchday}...")
        
        conn = self.db.get_connection()
        
        # Get all matches for this matchday
        query = """
        SELECT m.match_id, m.match_datetime, ht.team_name as home_team
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.team_id
        WHERE m.season = ? AND m.matchday = ?
        ORDER BY m.match_datetime
        """
        
        matches = pd.read_sql_query(query, conn, params=(season, matchday))
        
        if matches.empty:
            logger.warning(f"No matches found for {season} MD {matchday}")
            conn.close()
            return 0
        
        updated_count = 0
        now = pd.Timestamp.now(ZoneInfo("Europe/Berlin"))
        
        for _, match_row in matches.iterrows():
            match_id = match_row['match_id']
            match_dt = pd.to_datetime(match_row['match_datetime'])
            home_team = match_row['home_team']
            
            # Ensure timezone-aware
            if match_dt.tzinfo is None:
                match_dt = match_dt.tz_localize(ZoneInfo("Europe/Berlin"))
            else:
                match_dt = match_dt.astimezone(ZoneInfo("Europe/Berlin"))
            
            # Calculate days ahead
            days_ahead = (match_dt.date() - now.date()).days
            
            # Fetch weather based on timeframe
            if days_ahead >= 0 and days_ahead < 7:
                # Use forecast API
                weather_data = self._get_live_weather(match_dt, home_team, conn)
                if weather_data:
                    weather_source = 'open_meteo_forecast'
                    weather_confidence = 0.8
                else:
                    # Fallback to estimate
                    weather_data = self._estimate_weather(match_dt, conn)
                    weather_source = 'historical_estimate'
                    weather_confidence = 0.5
            else:
                # Use historical estimate
                weather_data = self._estimate_weather(match_dt, conn)
                weather_source = 'historical_estimate'
                weather_confidence = 0.5
            
            # Map numeric weather to condition text
            temp = weather_data.get('temperature_celsius', 15.0)
            precip = weather_data.get('precipitation_mm', 0.0)
            wind = weather_data.get('wind_speed_kmh', 10.0)
            
            weather_condition = None
            if precip > 5.0:
                weather_condition = 'heavy_rain'
            elif precip > 0.5:
                weather_condition = 'rain'
            elif temp <= 0 and precip > 0:
                weather_condition = 'snow'
            elif wind > 40:
                weather_condition = 'windy'
            elif temp > 25:
                weather_condition = 'hot'
            elif temp < 5:
                weather_condition = 'cold'
            else:
                weather_condition = 'clear'
            
            # Update database
            update_query = """
            UPDATE matches
            SET temperature_celsius = ?,
                humidity_percent = ?,
                wind_speed_kmh = ?,
                precipitation_mm = ?,
                weather_source = ?,
                weather_confidence = ?,
                weather_condition = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE match_id = ?
            """
            
            try:
                conn.execute(update_query, (
                    weather_data.get('temperature_celsius'),
                    weather_data.get('humidity_percent'),
                    weather_data.get('wind_speed_kmh'),
                    weather_data.get('precipitation_mm'),
                    weather_source,
                    weather_confidence,
                    weather_condition,
                    match_id
                ))
                updated_count += 1
            except Exception as e:
                logger.warning(f"Failed to update weather for match {match_id}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.success(f"Updated weather for {updated_count}/{len(matches)} matches")
        return updated_count

    def _get_betting_odds(self, match_id: int, conn) -> Optional[dict]:
        """
        Query betting odds from database for a match.
        Uses aggregated oddsportal_avg bookmaker data.

        Args:
            match_id: Match ID
            conn: Database connection

        Returns:
            Dictionary with odds features or None if not found
        """
        try:
            # Query aggregated odds (same as MLDataExporter)
            odds_query = """
                SELECT
                    AVG(odds_home) AS odds_home,
                    AVG(odds_draw) AS odds_draw,
                    AVG(odds_away) AS odds_away,
                    AVG(implied_prob_home) AS implied_prob_home,
                    AVG(implied_prob_draw) AS implied_prob_draw,
                    AVG(implied_prob_away) AS implied_prob_away
                FROM betting_odds
                WHERE match_id = ? AND bookmaker = 'oddsportal_avg'
            """
            
            result = pd.read_sql_query(odds_query, conn, params=(match_id,))
            
            if not result.empty:
                row = result.iloc[0]
                # Check if we have valid odds
                if pd.notna(row['odds_home']) and pd.notna(row['odds_draw']) and pd.notna(row['odds_away']):
                    return {
                        'odds_home': float(row['odds_home']),
                        'odds_draw': float(row['odds_draw']),
                        'odds_away': float(row['odds_away']),
                        'implied_prob_home': float(row['implied_prob_home']) if pd.notna(row['implied_prob_home']) else None,
                        'implied_prob_draw': float(row['implied_prob_draw']) if pd.notna(row['implied_prob_draw']) else None,
                        'implied_prob_away': float(row['implied_prob_away']) if pd.notna(row['implied_prob_away']) else None,
                    }
        except Exception as e:
            logger.debug(f"Could not query betting odds for match_id {match_id}: {e}")
        
        return None

    def _calculate_odds_heuristic(self, home_elo: float, away_elo: float,
                                   home_pi: float, away_pi: float,
                                   home_form: int, away_form: int) -> dict:
        """
        Calculate betting odds heuristic from team ratings and form

        Uses Elo ratings and form to estimate fair odds.

        Args:
            home_elo: Home team Elo rating
            away_elo: Away team Elo rating
            home_pi: Home team PI rating
            away_pi: Away team PI rating
            home_form: Home team points in last 5
            away_form: Away team points in last 5

        Returns:
            Dictionary with odds features
        """
        try:
            # Calculate Elo win probability with home advantage
            HOME_ADVANTAGE = 100  # Elo points for home advantage
            elo_diff = (home_elo + HOME_ADVANTAGE) - away_elo

            # Elo formula for win probability
            prob_home_elo = 1 / (1 + 10 ** (-elo_diff / 400))
            prob_away_elo = 1 - prob_home_elo

            # PI probabilities (already 0-1)
            prob_home_pi = max(0.1, min(0.9, home_pi + 0.1))  # Add home advantage
            prob_away_pi = max(0.1, min(0.9, away_pi))

            # Form factor (normalize to 0-1, max 15 points in last 5)
            form_factor = (home_form - away_form) / 30.0  # -1 to 1 range
            form_factor = max(-0.3, min(0.3, form_factor))  # Cap at ±0.3

            # Combine probabilities (weighted average)
            prob_home = 0.5 * prob_home_elo + 0.3 * prob_home_pi + form_factor
            prob_away = 0.5 * prob_away_elo + 0.3 * prob_away_pi - form_factor

            # Ensure probabilities are positive
            prob_home = max(0.1, prob_home)
            prob_away = max(0.1, prob_away)

            # Ensure probabilities sum to less than 1 (leave room for draw)
            total = prob_home + prob_away
            if total >= 0.95:
                prob_home = prob_home / total * 0.95
                prob_away = prob_away / total * 0.95

            # Draw probability (remaining probability)
            prob_draw = max(0.15, 1.0 - prob_home - prob_away)

            # Normalize to sum to 1.0
            total = prob_home + prob_draw + prob_away
            if total > 0:
                prob_home = prob_home / total
                prob_draw = prob_draw / total
                prob_away = prob_away / total
            else:
                # Fallback to equal probabilities if something went wrong
                prob_home = 0.38
                prob_draw = 0.28
                prob_away = 0.34

            # Convert probabilities to odds (with bookmaker margin)
            MARGIN = 1.05  # 5% bookmaker margin
            odds_home = round(MARGIN / prob_home, 2)
            odds_draw = round(MARGIN / prob_draw, 2)
            odds_away = round(MARGIN / prob_away, 2)

            return {
                'odds_home': odds_home,
                'odds_draw': odds_draw,
                'odds_away': odds_away,
                'implied_prob_home': round(prob_home, 3),
                'implied_prob_draw': round(prob_draw, 3),
                'implied_prob_away': round(prob_away, 3),
            }

        except Exception as e:
            logger.debug(f"Could not calculate odds heuristic: {e}")

            # Default odds (balanced match)
            return {
                'odds_home': 2.50,
                'odds_draw': 3.20,
                'odds_away': 2.80,
                'implied_prob_home': 0.38,
                'implied_prob_draw': 0.28,
                'implied_prob_away': 0.34,
            }

    def engineer_features_for_matches(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for future matches from database

        Args:
            matches: DataFrame with basic match info (teams, datetime)

        Returns:
            DataFrame with all features engineered
        """
        logger.info(f"Engineering features for {len(matches)} matches...")

        conn = self.db.get_connection()
        enriched = []

        for idx, match in matches.iterrows():
            features = {
                'season': match['season'],
                'matchday': match['matchday'],
                'match_datetime': match['match_datetime'],
                'home_team': match['home_team'],
                'away_team': match['away_team'],
            }

            # Get latest ratings - prefer "current" ratings (match_id IS NULL) which include latest match results
            # Fallback to latest match-specific rating if current rating doesn't exist
            home_ratings = pd.read_sql_query("""
                SELECT elo_rating, pi_rating, points_last_5, points_last_10,
                       goals_scored_last_5, goals_conceded_last_5
                FROM team_ratings
                WHERE team_id = (SELECT team_id FROM teams WHERE team_name = ? LIMIT 1)
                  AND (match_id IS NULL OR match_id = (
                      SELECT match_id FROM team_ratings tr2
                      WHERE tr2.team_id = team_ratings.team_id
                      ORDER BY tr2.matchday DESC, tr2.created_at DESC
                      LIMIT 1
                  ))
                ORDER BY CASE WHEN match_id IS NULL THEN 0 ELSE 1 END,
                         matchday DESC, created_at DESC
                LIMIT 1
            """, conn, params=(match['home_team'],))

            if not home_ratings.empty:
                features['home_elo'] = home_ratings.iloc[0]['elo_rating']
                features['home_pi'] = home_ratings.iloc[0]['pi_rating']
                features['home_points_l5'] = int(home_ratings.iloc[0]['points_last_5']) if pd.notna(home_ratings.iloc[0]['points_last_5']) else 0
                features['home_points_l10'] = int(home_ratings.iloc[0]['points_last_10']) if pd.notna(home_ratings.iloc[0]['points_last_10']) else 0
                features['home_goals_scored_l5'] = float(home_ratings.iloc[0]['goals_scored_last_5']) if pd.notna(home_ratings.iloc[0]['goals_scored_last_5']) else 0.0
                features['home_goals_conceded_l5'] = float(home_ratings.iloc[0]['goals_conceded_last_5']) if pd.notna(home_ratings.iloc[0]['goals_conceded_last_5']) else 0.0
            else:
                # Use defaults
                features['home_elo'] = 1500
                features['home_pi'] = 0.5
                features['home_points_l5'] = 0
                features['home_points_l10'] = 0
                features['home_goals_scored_l5'] = 0.0
                features['home_goals_conceded_l5'] = 0.0

            # Get latest ratings for away team
            away_ratings = pd.read_sql_query("""
                SELECT elo_rating, pi_rating, points_last_5, points_last_10,
                       goals_scored_last_5, goals_conceded_last_5
                FROM team_ratings
                WHERE team_id = (SELECT team_id FROM teams WHERE team_name = ? LIMIT 1)
                  AND (match_id IS NULL OR match_id = (
                      SELECT match_id FROM team_ratings tr2
                      WHERE tr2.team_id = team_ratings.team_id
                      ORDER BY tr2.matchday DESC, tr2.created_at DESC
                      LIMIT 1
                  ))
                ORDER BY CASE WHEN match_id IS NULL THEN 0 ELSE 1 END,
                         matchday DESC, created_at DESC
                LIMIT 1
            """, conn, params=(match['away_team'],))

            if not away_ratings.empty:
                features['away_elo'] = away_ratings.iloc[0]['elo_rating']
                features['away_pi'] = away_ratings.iloc[0]['pi_rating']
                features['away_points_l5'] = int(away_ratings.iloc[0]['points_last_5']) if pd.notna(away_ratings.iloc[0]['points_last_5']) else 0
                features['away_points_l10'] = int(away_ratings.iloc[0]['points_last_10']) if pd.notna(away_ratings.iloc[0]['points_last_10']) else 0
                features['away_goals_scored_l5'] = float(away_ratings.iloc[0]['goals_scored_last_5']) if pd.notna(away_ratings.iloc[0]['goals_scored_last_5']) else 0.0
                features['away_goals_conceded_l5'] = float(away_ratings.iloc[0]['goals_conceded_last_5']) if pd.notna(away_ratings.iloc[0]['goals_conceded_last_5']) else 0.0
            else:
                # Use defaults
                features['away_elo'] = 1500
                features['away_pi'] = 0.5
                features['away_points_l5'] = 0
                features['away_points_l10'] = 0
                features['away_goals_scored_l5'] = 0.0
                features['away_goals_conceded_l5'] = 0.0

            # Calculate derived features
            features['elo_diff'] = features['home_elo'] - features['away_elo']
            features['pi_diff'] = features['home_pi'] - features['away_pi']
            features['form_diff_l5'] = features['home_points_l5'] - features['away_points_l5']
            features['goal_diff_l5'] = (features['home_goals_scored_l5'] - features['home_goals_conceded_l5']) - \
                                       (features['away_goals_scored_l5'] - features['away_goals_conceded_l5'])

            # Parse datetime for temporal features
            match_dt = pd.to_datetime(match['match_datetime'])
            features['day_of_week'] = match_dt.dayofweek
            features['month'] = match_dt.month
            features['year'] = match_dt.year
            features['is_midweek'] = 1 if match_dt.dayofweek in [1, 2, 3] else 0

            # Calculate rest days (days since last match)
            home_rest_days = self._calculate_rest_days(
                match['home_team'], match_dt, conn
            )
            away_rest_days = self._calculate_rest_days(
                match['away_team'], match_dt, conn
            )

            features['rest_days_home'] = home_rest_days
            features['rest_days_away'] = away_rest_days
            features['rest_days_diff'] = home_rest_days - away_rest_days

            # Calculate travel distance
            travel_distance = self._calculate_travel_distance(
                match['home_team'], match['away_team'], conn
            )
            features['travel_distance_km'] = travel_distance

            # Get head-to-head statistics
            h2h_stats = self._get_head_to_head_stats(
                match['home_team'], match['away_team'], conn
            )
            features.update(h2h_stats)

            # Get weather data based on mode
            if self.weather_mode == 'off':
                # Skip weather, use conservative defaults
                weather_estimate = {
                    'temperature_celsius': 15.0,
                    'humidity_percent': 70.0,
                    'wind_speed_kmh': 10.0,
                    'precipitation_mm': 0.0,
                }
            elif self.weather_mode == 'live':
                # Try live forecast first, fallback to estimate
                weather_estimate = self._get_live_weather(match_dt, match['home_team'], conn)
                if weather_estimate is None:
                    logger.debug(f"Live weather unavailable for {match['home_team']}, using estimate")
                    weather_estimate = self._estimate_weather(match_dt, conn)
            else:  # 'estimate' mode
                weather_estimate = self._estimate_weather(match_dt, conn)
            
            features.update(weather_estimate)

            # Get betting odds: try database first, fallback to heuristic
            match_id = None
            if 'match_id' in match:
                match_id_val = match['match_id']
                if pd.notna(match_id_val):
                    try:
                        match_id = int(match_id_val)
                    except (ValueError, TypeError):
                        match_id = None
            
            odds_data = None
            if match_id is not None:
                odds_data = self._get_betting_odds(match_id, conn)
            
            if odds_data is None:
                # Fallback to heuristic calculation
                # Note: Using heuristic odds (not from OddsPortal)
                odds_data = self._calculate_odds_heuristic(
                    features['home_elo'], features['away_elo'],
                    features['home_pi'], features['away_pi'],
                    features['home_points_l5'], features['away_points_l5']
                )
            else:
                # Calculate implied probabilities if not present
                if odds_data.get('implied_prob_home') is None and odds_data.get('odds_home'):
                    odds_data['implied_prob_home'] = 1.0 / odds_data['odds_home']
                if odds_data.get('implied_prob_draw') is None and odds_data.get('odds_draw'):
                    odds_data['implied_prob_draw'] = 1.0 / odds_data['odds_draw']
                if odds_data.get('implied_prob_away') is None and odds_data.get('odds_away'):
                    odds_data['implied_prob_away'] = 1.0 / odds_data['odds_away']
            
            features.update(odds_data)

            enriched.append(features)

        conn.close()

        df_enriched = pd.DataFrame(enriched)
        logger.success(f"Engineered features for {len(df_enriched)} matches")

        return df_enriched

    def get_upcoming_matches(self, season: str, matchday: int, require_db: bool = False) -> pd.DataFrame:
        """
        Get upcoming matches for prediction
        Priority: DB → CSV → API (unless require_db=True, then DB-only)

        Args:
            season: Season
            matchday: Matchday number
            require_db: If True, only return DB matches (no CSV/API fallback)

        Returns:
            DataFrame with match features
        """
        logger.info(f"Loading matches for {season} MD {matchday}...")

        # Priority 1: Try to load from database
        db_matches = self.load_upcoming_from_db(season, matchday)
        if not db_matches.empty:
            # Engineer features for DB matches
            matches_with_features = self.engineer_features_for_matches(db_matches)
            logger.info(f"Found {len(matches_with_features)} matches in database")
            return matches_with_features

        # If require_db is True, only use DB matches - return empty if none found
        if require_db:
            logger.warning(f"No matches found in database for {season} MD {matchday} (require_db=True)")
            return pd.DataFrame()

        # Priority 2: Try to load from existing CSV dataset
        try:
            df = pd.read_csv('data/processed/3liga_ml_dataset_full.csv')

            # Filter for the specific matchday
            matches = df[(df['season'] == season) & (df['matchday'] == matchday)].copy()

            # Filter for unfinished matches only
            unfinished = matches[matches['result'].isna() | (matches['home_goals'].isna())]

            if len(unfinished) == 0:
                # If all finished, take all matches from this matchday for demonstration
                matches_to_predict = matches.copy()
            else:
                matches_to_predict = unfinished.copy()

            if len(matches_to_predict) > 0:
                logger.info(f"Found {len(matches_to_predict)} matches in existing CSV dataset")
                return matches_to_predict

        except Exception as e:
            logger.warning(f"Could not load from existing CSV dataset: {e}")

        # Priority 3: Fetch from API and engineer features
        logger.info("No matches in DB or CSV, fetching from API...")

        # Fetch basic match info from API
        future_matches = self.fetch_future_matches(season, matchday)

        if future_matches.empty:
            logger.error(f"Could not fetch matches from API")
            return pd.DataFrame()

        # Engineer features for these matches
        matches_with_features = self.engineer_features_for_matches(future_matches)

        logger.info(f"Found {len(matches_with_features)} matches to predict")

        return matches_with_features

    def prepare_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for prediction (handle missing values)

        Args:
            df: Raw match data

        Returns:
            Prepared features
        """
        # Use same features as training
        if self.features is None:
            raise ValueError("Model must be trained first")

        # Handle missing values using training medians
        for feat in self.features:
            if feat not in df.columns:
                # h2h_total_matches was removed, use h2h_match_count instead
                if feat == 'h2h_total_matches' and 'h2h_match_count' in df.columns:
                    df[feat] = df['h2h_match_count']
                else:
                    logger.debug(f"Feature {feat} not in prediction data, using 0")
                    df[feat] = 0
            elif feat in config.CATEGORICAL_FEATURES:
                df[feat] = df[feat].fillna('MISSING')
            else:
                median_val = self.feature_medians.get(feat, 0)
                df[feat] = df[feat].fillna(median_val)

        X = df[self.features].copy()

        return X

    def predict_matches(self, season: str, matchday: int, require_db: bool = False) -> pd.DataFrame:
        """
        Predict results for all matches in a matchday

        Args:
            season: Season
            matchday: Matchday number
            require_db: If True, only use DB matches (no CSV/API fallback)

        Returns:
            DataFrame with predictions
        """
        # Get matches to predict (first, so we can derive cutoff to avoid leakage)
        matches = self.get_upcoming_matches(season, matchday, require_db=require_db)

        if matches.empty:
            logger.error(f"No matches found for {season} MD {matchday}")
            return pd.DataFrame()

        # Determine cutoff: start of the earliest kickoff in this matchday
        # Always use cutoff to ensure zero data leakage (train only on data before matchday)
        match_times = pd.to_datetime(matches['match_datetime'])
        cutoff = match_times.min()
        # Ensure cutoff is timezone-aware for comparison
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(ZoneInfo("Europe/Berlin"))

        # Train model respecting leakage constraints (always use cutoff)
        # This ensures we only train on matches that finished before the earliest kickoff
        logger.info(f"Training model with cutoff {cutoff} to prevent data leakage")
        self.train_model(force_retrain=True, cutoff_datetime=cutoff, persist=False)

        # Prepare features
        X = self.prepare_prediction_features(matches)

        # Make predictions
        home_goals, away_goals = self.model.predict_random_forest(X)

        # Add predictions to results along with key features
        results = pd.DataFrame({
            'match_datetime': matches['match_datetime'].values,
            'home_team': matches['home_team'].values,
            'away_team': matches['away_team'].values,
            'predicted_home_goals': home_goals.astype(int),
            'predicted_away_goals': away_goals.astype(int),
        })
        
        # Add key features for explanation
        feature_cols = ['home_elo', 'away_elo', 'elo_diff', 'home_pi', 'away_pi', 'pi_diff',
                       'home_points_l5', 'away_points_l5', 'form_diff_l5',
                       'home_goals_scored_l5', 'home_goals_conceded_l5',
                       'away_goals_scored_l5', 'away_goals_conceded_l5',
                       'odds_home', 'odds_draw', 'odds_away', 'implied_prob_home',
                       'rest_days_home', 'rest_days_away', 'rest_days_diff',
                       'travel_distance_km', 'h2h_match_count', 'h2h_home_win_rate',
                       'temperature_celsius', 'precipitation_mm', 'wind_speed_kmh']
        
        for col in feature_cols:
            if col in matches.columns:
                results[col] = matches[col].values

        # Add actual results if available
        if 'home_goals' in matches.columns and matches['home_goals'].notna().any():
            results['actual_home_goals'] = matches['home_goals'].values
            results['actual_away_goals'] = matches['away_goals'].values

        return results

    def print_predictions(self, predictions: pd.DataFrame):
        """
        Print predictions in a nice format with feature explanations

        Args:
            predictions: Prediction results DataFrame
        """
        print("\n" + "=" * 80)
        print(f"{'3. LIGA MATCH PREDICTIONS':^80}")
        print("=" * 80)
        print(f"\nModel: Random Forest Classifier (Winner Model)")
        print(f"Predictions generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nTotal matches: {len(predictions)}")
        print("\n" + "-" * 80)

        for idx, row in predictions.iterrows():
            # Parse datetime
            match_date = pd.to_datetime(row['match_datetime'])
            date_str = match_date.strftime('%a, %d %b %Y %H:%M')

            # Format prediction
            pred_score = f"{int(row['predicted_home_goals'])}-{int(row['predicted_away_goals'])}"
            
            # Determine predicted result
            if row['predicted_home_goals'] > row['predicted_away_goals']:
                pred_result = "Home Win"
            elif row['predicted_home_goals'] < row['predicted_away_goals']:
                pred_result = "Away Win"
            else:
                pred_result = "Draw"

            print(f"\n{date_str}")
            print(f"  {row['home_team']:30s} vs {row['away_team']:30s}")
            print(f"  Prediction: {pred_score} ({pred_result})")

            # Show key features
            print(f"\n  Key Features:")
            
            # Elo ratings
            if 'home_elo' in row and pd.notna(row['home_elo']):
                home_elo = int(row['home_elo'])
                away_elo = int(row['away_elo'])
                elo_diff = int(row.get('elo_diff', home_elo - away_elo))
                print(f"    Elo Ratings:     Home {home_elo} vs Away {away_elo} (diff: {elo_diff:+d})")
            
            # PI ratings
            if 'home_pi' in row and pd.notna(row['home_pi']):
                home_pi = row['home_pi']
                away_pi = row['away_pi']
                pi_diff = row.get('pi_diff', home_pi - away_pi)
                print(f"    PI Ratings:      Home {home_pi:.3f} vs Away {away_pi:.3f} (diff: {pi_diff:+.3f})")
            
            # Form (last 5 matches)
            if 'home_points_l5' in row and pd.notna(row['home_points_l5']):
                home_form = int(row['home_points_l5'])
                away_form = int(row['away_points_l5'])
                form_diff = int(row.get('form_diff_l5', home_form - away_form))
                print(f"    Form (last 5):    Home {home_form} pts vs Away {away_form} pts (diff: {form_diff:+d})")
            
            # Betting odds
            if 'odds_home' in row and pd.notna(row['odds_home']):
                odds_h = row['odds_home']
                odds_d = row.get('odds_draw', 0)
                odds_a = row.get('odds_away', 0)
                # Check if odds are valid (positive and reasonable)
                if odds_h > 0 and odds_d > 0 and odds_a > 0 and odds_h < 100 and odds_d < 100 and odds_a < 100:
                    odds_source = "database" if 'implied_prob_home' in row and pd.notna(row.get('implied_prob_home')) else "heuristic"
                    print(f"    Betting Odds:    Home {odds_h:.2f} | Draw {odds_d:.2f} | Away {odds_a:.2f} ({odds_source})")
                else:
                    print(f"    Betting Odds:    Invalid (using defaults)")
            
            # Goal difference (last 5)
            if 'home_goals_scored_l5' in row and pd.notna(row['home_goals_scored_l5']):
                home_gf = int(row['home_goals_scored_l5'])
                home_ga = int(row.get('home_goals_conceded_l5', 0))
                away_gf = int(row.get('away_goals_scored_l5', 0))
                away_ga = int(row.get('away_goals_conceded_l5', 0))
                home_gd = home_gf - home_ga
                away_gd = away_gf - away_ga
                print(f"    Goal Diff (L5):   Home {home_gd:+d} ({home_gf}-{home_ga}) vs Away {away_gd:+d} ({away_gf}-{away_ga})")
            
            # Weather conditions
            if 'temperature_celsius' in row and pd.notna(row['temperature_celsius']):
                temp = row['temperature_celsius']
                precip = row.get('precipitation_mm', 0)
                wind = row.get('wind_speed_kmh', 0)
                weather_desc = []
                if temp < 5:
                    weather_desc.append("cold")
                elif temp > 25:
                    weather_desc.append("hot")
                if precip > 0.5:
                    weather_desc.append("rainy")
                if wind > 25:
                    weather_desc.append("windy")
                if not weather_desc:
                    weather_desc.append("clear")
                print(f"    Weather:         {temp:.1f}°C, {precip:.1f}mm rain, {wind:.1f} km/h wind ({', '.join(weather_desc)})")
            
            # Rest days
            if 'rest_days_home' in row and pd.notna(row['rest_days_home']):
                rest_h = int(row['rest_days_home'])
                rest_a = int(row.get('rest_days_away', 0))
                rest_diff = int(row.get('rest_days_diff', rest_h - rest_a))
                print(f"    Rest Days:       Home {rest_h}d vs Away {rest_a}d (diff: {rest_diff:+d})")
            
            # Travel distance
            if 'travel_distance_km' in row and pd.notna(row['travel_distance_km']):
                travel = row['travel_distance_km']
                print(f"    Travel Distance: {travel:.0f} km")
            
            # Head-to-head
            if 'h2h_match_count' in row and pd.notna(row['h2h_match_count']) and row['h2h_match_count'] > 0:
                h2h_count = int(row['h2h_match_count'])
                h2h_rate = row.get('h2h_home_win_rate', 0)
                print(f"    H2H History:     {h2h_count} matches, Home win rate: {h2h_rate:.1%}")

            # Show actual if available
            if 'actual_home_goals' in row and pd.notna(row['actual_home_goals']):
                actual_score = f"{int(row['actual_home_goals'])}-{int(row['actual_away_goals'])}"
                print(f"\n  Actual Result:     {actual_score}")

        print("\n" + "=" * 80)
        print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Predict 3. Liga match results using Random Forest Classifier'
    )
    parser.add_argument(
        '--season',
        type=str,
        help='Season to predict (e.g., 2025-2026). Default: current season'
    )
    parser.add_argument(
        '--matchday',
        type=int,
        help='Specific matchday to predict. Default: next upcoming matchday'
    )
    parser.add_argument(
        '--update-data',
        action='store_true',
        help='Update/fetch data for the matchday before predicting'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save predictions to CSV file'
    )
    parser.add_argument(
        '--weather-mode',
        type=str,
        choices=['live', 'estimate', 'off'],
        default='live',
        help='Weather fetching mode: live (forecast API), estimate (historical), or off (defaults). Default: live'
    )
    parser.add_argument(
        '--ext-data',
        action='store_true',
        help='Include heavy external data collection (FBref, matchday-level standings). Default: False'
    )

    args = parser.parse_args()

    # Initialize predictor
    predictor = MatchPredictor(weather_mode=args.weather_mode, ext_data=args.ext_data)

    # Determine which matchday to predict
    # Normalize season format if provided as start year (e.g., "2025" -> "2025-2026")
    season_arg = args.season
    if season_arg and season_arg.isdigit() and len(season_arg) == 4:
        start_year = int(season_arg)
        season_arg = f"{start_year}-{start_year+1}"

    if args.matchday and season_arg:
        season = season_arg
        matchday = args.matchday
    elif args.matchday:
        season = predictor.get_current_season()
        matchday = args.matchday
    else:
        season, matchday, has_data = predictor.find_next_matchday(season_arg)

    logger.info(f"Target: {season} Matchday {matchday}")

    # Step 1: Check data availability
    data_status = predictor.check_data_availability(season, matchday)
    logger.info(f"Data availability: exists={data_status['exists']}, can_predict={data_status['can_predict']}")

    # Step 2 or 3: If --update-data is set, always run update (refresh odds/weather/ratings/H2H)
    if args.update_data:
        logger.info("--update-data flag set, running data acquisition pipeline...")
        success = predictor.update_matchday_data(season, matchday)
        if not success:
            logger.error("Data acquisition failed, but continuing with prediction attempt...")
        # Re-check availability after update
        data_status = predictor.check_data_availability(season, matchday)

    # Step 3: If no data available, acquire it
    if not data_status['exists'] or not data_status['can_predict']:
        logger.warning(f"No data available for {season} MD {matchday}")

        if args.update_data:
            # Already tried above, but try again if it failed
            logger.info("Attempting data acquisition again...")
            success = predictor.update_matchday_data(season, matchday)
            if not success:
                print(f"\n⚠️  Failed to acquire data for {season} Matchday {matchday}")
                print(f"\nTo retry, run:")
                print(f"  python main.py predict --season {season} --matchday {matchday} --update-data")
                return
            # Re-check after acquisition
            data_status = predictor.check_data_availability(season, matchday)
            if not data_status['can_predict']:
                logger.error("Data acquisition completed but no upcoming matches found")
                return
        else:
            print(f"\n⚠️  No data available for {season} Matchday {matchday}")
            print(f"\nTo fetch data, run:")
            print(f"  python main.py predict --season {season} --matchday {matchday} --update-data")
            return

    # Step 2: Data is available - retrain (leakage-safe) and predict
    logger.info("Making predictions...")
    # Use DB-only when --update-data was set (ensures we use freshly updated data)
    predictions = predictor.predict_matches(season, matchday, require_db=args.update_data)

    if predictions.empty:
        logger.error("No predictions generated")
        return

    # Display predictions
    predictor.print_predictions(predictions)

    # Save to file if requested
    if args.output:
        predictions.to_csv(args.output, index=False)
        logger.success(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    print("This script is deprecated. Use: python main.py predict [args]", file=sys.stderr)
    print("Example: python main.py predict --season 2025 --matchday 15", file=sys.stderr)
    sys.exit(2)
