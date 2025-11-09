#!/usr/bin/env python3
"""
3. Liga Match Predictor - Production Script
Uses the winning Random Forest Classifier model to predict upcoming match results.

Usage:
    python predict.py                    # Predict next matchday
    python predict.py --update-data      # Update data before predicting
    python predict.py --matchday 15      # Predict specific matchday
    python predict.py --season 2025      # Predict for specific season
"""

import argparse
import sys
import pickle
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, List, Dict

import pandas as pd
import numpy as np
from loguru import logger

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

    def __init__(self, model_path: str = "models/rf_classifier.pkl"):
        """
        Initialize predictor

        Args:
            model_path: Path to save/load trained model
        """
        self.db = get_db()
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.features = None
        self.feature_medians = {}

        # Winner model configuration
        self.default_scores = config.DEFAULT_SCORES

        logger.info("Match Predictor initialized")

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
            SUM(CASE WHEN is_finished = 0 THEN 1 ELSE 0 END) as upcoming,
            SUM(CASE WHEN is_finished = 1 THEN 1 ELSE 0 END) as finished
        FROM matches
        WHERE season = ? AND matchday = ?
        """

        result = pd.read_sql_query(query, conn, params=(season, matchday))
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

    def update_matchday_data(self, season: str, matchday: int) -> bool:
        """
        Fetch/update data for a specific matchday

        Args:
            season: Season (e.g., "2025-2026")
            matchday: Matchday number

        Returns:
            True if successful
        """
        logger.info(f"Updating data for {season} matchday {matchday}...")

        try:
            # Extract year from season (e.g., "2025-2026" -> "2025")
            season_year = season.split('-')[0]

            collector = OpenLigaDBCollector()
            matches = collector.get_matchdata_for_matchday(season_year, matchday)

            if not matches:
                logger.error(f"No matches found for {season} MD {matchday}")
                return False

            # TODO: Process and insert matches into database
            # For now, this would require implementing the full data pipeline
            logger.warning("Data fetching implemented but database insertion requires full pipeline")
            logger.info(f"Found {len(matches)} matches for {season} MD {matchday}")

            return True

        except Exception as e:
            logger.error(f"Error updating data: {e}")
            return False

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

    def train_model(self, force_retrain: bool = False) -> ClassifierExperiment:
        """
        Train or load the Random Forest Classifier

        Args:
            force_retrain: Force retraining even if model exists

        Returns:
            Trained model
        """
        # Load existing model if available
        if self.model_path.exists() and not force_retrain:
            logger.info(f"Loading existing model from {self.model_path}")
            with open(self.model_path, 'rb') as f:
                saved = pickle.load(f)
                self.model = saved['model']
                self.features = saved['features']
                self.feature_medians = saved['feature_medians']
                self.default_scores = saved['default_scores']
            return self.model

        # Train new model
        logger.info("Training new Random Forest Classifier...")

        # Get all training data (only completed matches)
        df = self.export_training_data()

        # Prepare features
        X, y_class, y_home, y_away, features = self.prepare_training_data(df)

        # Train Random Forest
        experiment = ClassifierExperiment(default_scores=self.default_scores)
        model = experiment.train_random_forest(X, y_class)

        self.model = experiment

        # Save model
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

                match_info = {
                    'season': season,
                    'matchday': matchday,
                    'match_datetime': parsed.get('match_datetime'),
                    'home_team': home_team_name,
                    'away_team': away_team_name,
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                }

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
            Number of rest days (default to 7 if no previous match found)
        """
        try:
            # Find most recent match before this date
            query = """
            SELECT MAX(match_datetime) as last_match
            FROM matches
            WHERE (home_team = ? OR away_team = ?)
              AND match_datetime < ?
              AND is_finished = 1
            """

            # Get team_id for querying
            team_id_result = pd.read_sql_query(
                "SELECT team_id FROM teams WHERE team_name = ? LIMIT 1",
                conn,
                params=(team_name,)
            )

            if team_id_result.empty:
                return 7  # Default

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

            if not result.empty and result.iloc[0]['last_match']:
                last_match = pd.to_datetime(result.iloc[0]['last_match'])
                rest_days = (match_date - last_match).days
                return max(0, rest_days)  # Ensure non-negative

        except Exception as e:
            logger.warning(f"Could not calculate rest days for {team_name}: {e}")

        return 7  # Default to 1 week

    def _calculate_travel_distance(self, home_team: str, away_team: str, conn) -> float:
        """
        Calculate travel distance between team cities using geopy

        Args:
            home_team: Home team name
            away_team: Away team name
            conn: Database connection

        Returns:
            Distance in kilometers (default to 300 if not available)
        """
        try:
            from geopy.geocoders import Nominatim
            from geopy.distance import geodesic
            from geopy.exc import GeocoderTimedOut, GeocoderServiceError

            # Get team cities from database
            home_city = pd.read_sql_query(
                """
                SELECT city
                FROM teams
                WHERE team_name = ?
                LIMIT 1
                """,
                conn,
                params=(home_team,)
            )

            away_city = pd.read_sql_query(
                """
                SELECT city
                FROM teams
                WHERE team_name = ?
                LIMIT 1
                """,
                conn,
                params=(away_team,)
            )

            if not home_city.empty and not away_city.empty:
                home_city_name = home_city.iloc[0]['city']
                away_city_name = away_city.iloc[0]['city']

                if home_city_name and away_city_name:
                    # Same city = 0km
                    if home_city_name.lower() == away_city_name.lower():
                        return 0.0

                    # Use geopy to geocode cities and calculate distance
                    geolocator = Nominatim(user_agent="3liga-predictor")

                    # Geocode with Germany context
                    home_location = geolocator.geocode(f"{home_city_name}, Germany", timeout=5)
                    away_location = geolocator.geocode(f"{away_city_name}, Germany", timeout=5)

                    if home_location and away_location:
                        home_coords = (home_location.latitude, home_location.longitude)
                        away_coords = (away_location.latitude, away_location.longitude)

                        distance = geodesic(home_coords, away_coords).kilometers
                        return round(distance, 2)

        except ImportError:
            logger.debug(f"geopy not installed, using default distance")
        except Exception as e:
            # Catch GeocoderTimedOut, GeocoderServiceError, and any other exceptions
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
                        'h2h_total_matches': total,
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
            'h2h_total_matches': 0,
            'h2h_home_wins': 0,
            'h2h_away_wins': 0,
            'h2h_home_win_rate': 0.0,
            'h2h_draw_rate': 0.0,
            'h2h_match_count': 0
        }

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
                    COUNT(*) as sample_count,
                    -- Also get standard deviations for uncertainty
                    STDEV(temperature_celsius) as std_temp
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

            # Ensure probabilities sum to less than 1 (leave room for draw)
            total = prob_home + prob_away
            if total >= 0.95:
                prob_home = prob_home / total * 0.95
                prob_away = prob_away / total * 0.95

            # Draw probability (remaining probability)
            prob_draw = max(0.15, 1.0 - prob_home - prob_away)

            # Normalize to sum to 1.0
            total = prob_home + prob_draw + prob_away
            prob_home = prob_home / total
            prob_draw = prob_draw / total
            prob_away = prob_away / total

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

            # Get latest ratings for home team
            home_ratings = pd.read_sql_query("""
                SELECT elo_rating, pi_rating, points_last_5, points_last_10,
                       goals_scored_last_5, goals_conceded_last_5
                FROM team_ratings
                WHERE team_id = (SELECT team_id FROM teams WHERE team_name = ? LIMIT 1)
                ORDER BY matchday DESC, created_at DESC
                LIMIT 1
            """, conn, params=(match['home_team'],))

            if not home_ratings.empty:
                features['home_elo'] = home_ratings.iloc[0]['elo_rating']
                features['home_pi'] = home_ratings.iloc[0]['pi_rating']
                features['home_points_l5'] = home_ratings.iloc[0]['points_last_5']
                features['home_points_l10'] = home_ratings.iloc[0]['points_last_10']
                features['home_goals_scored_l5'] = home_ratings.iloc[0]['goals_scored_last_5']
                features['home_goals_conceded_l5'] = home_ratings.iloc[0]['goals_conceded_last_5']
            else:
                # Use defaults
                features['home_elo'] = 1500
                features['home_pi'] = 0.5
                features['home_points_l5'] = 0
                features['home_points_l10'] = 0
                features['home_goals_scored_l5'] = 0
                features['home_goals_conceded_l5'] = 0

            # Get latest ratings for away team
            away_ratings = pd.read_sql_query("""
                SELECT elo_rating, pi_rating, points_last_5, points_last_10,
                       goals_scored_last_5, goals_conceded_last_5
                FROM team_ratings
                WHERE team_id = (SELECT team_id FROM teams WHERE team_name = ? LIMIT 1)
                ORDER BY matchday DESC, created_at DESC
                LIMIT 1
            """, conn, params=(match['away_team'],))

            if not away_ratings.empty:
                features['away_elo'] = away_ratings.iloc[0]['elo_rating']
                features['away_pi'] = away_ratings.iloc[0]['pi_rating']
                features['away_points_l5'] = away_ratings.iloc[0]['points_last_5']
                features['away_points_l10'] = away_ratings.iloc[0]['points_last_10']
                features['away_goals_scored_l5'] = away_ratings.iloc[0]['goals_scored_last_5']
                features['away_goals_conceded_l5'] = away_ratings.iloc[0]['goals_conceded_last_5']
            else:
                # Use defaults
                features['away_elo'] = 1500
                features['away_pi'] = 0.5
                features['away_points_l5'] = 0
                features['away_points_l10'] = 0
                features['away_goals_scored_l5'] = 0
                features['away_goals_conceded_l5'] = 0

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

            # Estimate weather from previous week
            weather_estimate = self._estimate_weather(match_dt, conn)
            features.update(weather_estimate)

            # Calculate betting odds heuristic from team ratings
            odds_heuristic = self._calculate_odds_heuristic(
                features['home_elo'], features['away_elo'],
                features['home_pi'], features['away_pi'],
                features['home_points_l5'], features['away_points_l5']
            )
            features.update(odds_heuristic)

            enriched.append(features)

        conn.close()

        df_enriched = pd.DataFrame(enriched)
        logger.success(f"Engineered features for {len(df_enriched)} matches")

        return df_enriched

    def get_upcoming_matches(self, season: str, matchday: int) -> pd.DataFrame:
        """
        Get upcoming matches for prediction

        Args:
            season: Season
            matchday: Matchday number

        Returns:
            DataFrame with match features
        """
        logger.info(f"Loading matches for {season} MD {matchday}...")

        # First, try to load from existing dataset (database/CSV)
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
                logger.info(f"Found {len(matches_to_predict)} matches in existing dataset")
                return matches_to_predict

        except Exception as e:
            logger.warning(f"Could not load from existing dataset: {e}")

        # If no matches found in dataset, fetch from API and engineer features
        logger.info("No matches in dataset, fetching from API...")

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
                logger.warning(f"Feature {feat} not in prediction data, using 0")
                df[feat] = 0
            elif feat in config.CATEGORICAL_FEATURES:
                df[feat] = df[feat].fillna('MISSING')
            else:
                median_val = self.feature_medians.get(feat, 0)
                df[feat] = df[feat].fillna(median_val)

        X = df[self.features].copy()

        return X

    def predict_matches(self, season: str, matchday: int) -> pd.DataFrame:
        """
        Predict results for all matches in a matchday

        Args:
            season: Season
            matchday: Matchday number

        Returns:
            DataFrame with predictions
        """
        # Ensure model is trained
        if self.model is None:
            self.train_model()

        # Get matches to predict
        matches = self.get_upcoming_matches(season, matchday)

        if matches.empty:
            logger.error(f"No matches found for {season} MD {matchday}")
            return pd.DataFrame()

        # Prepare features
        X = self.prepare_prediction_features(matches)

        # Make predictions
        home_goals, away_goals = self.model.predict_random_forest(X)

        # Add predictions to results
        results = pd.DataFrame({
            'match_datetime': matches['match_datetime'].values,
            'home_team': matches['home_team'].values,
            'away_team': matches['away_team'].values,
            'predicted_home_goals': home_goals.astype(int),
            'predicted_away_goals': away_goals.astype(int),
        })

        # Add actual results if available
        if 'home_goals' in matches.columns and matches['home_goals'].notna().any():
            results['actual_home_goals'] = matches['home_goals'].values
            results['actual_away_goals'] = matches['away_goals'].values

        return results

    def print_predictions(self, predictions: pd.DataFrame):
        """
        Print predictions in a nice format

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

            print(f"\n{date_str}")
            print(f"  {row['home_team']:30s} vs {row['away_team']:30s}")
            print(f"  Prediction: {pred_score}")

            # Show actual if available
            if 'actual_home_goals' in row and pd.notna(row['actual_home_goals']):
                actual_score = f"{int(row['actual_home_goals'])}-{int(row['actual_away_goals'])}"
                print(f"  Actual:     {actual_score}")

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
        '--retrain',
        action='store_true',
        help='Force model retraining'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save predictions to CSV file'
    )

    args = parser.parse_args()

    # Initialize predictor
    predictor = MatchPredictor()

    # Determine which matchday to predict
    if args.matchday and args.season:
        season = args.season
        matchday = args.matchday
    elif args.matchday:
        season = predictor.get_current_season()
        matchday = args.matchday
    else:
        season, matchday, has_data = predictor.find_next_matchday(args.season)

    logger.info(f"Target: {season} Matchday {matchday}")

    # Check data availability
    data_status = predictor.check_data_availability(season, matchday)

    if not data_status['exists']:
        logger.warning(f"No data found for {season} MD {matchday}")

        if args.update_data:
            logger.info("Attempting to fetch data...")
            predictor.update_matchday_data(season, matchday)
        else:
            print(f"\n⚠️  No data available for {season} Matchday {matchday}")
            print(f"\nTo fetch data, run:")
            print(f"  python predict.py --season {season} --matchday {matchday} --update-data")
            return

    # Train or load model
    logger.info("Preparing model...")
    predictor.train_model(force_retrain=args.retrain)

    # Make predictions
    logger.info("Making predictions...")
    predictions = predictor.predict_matches(season, matchday)

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
    main()
