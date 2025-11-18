"""
ML Data Exporter for 3. Liga Dataset
Exports comprehensive feature-engineered data for machine learning models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, Tuple, Optional
from datetime import datetime

from liga_predictor.database import get_db


class MLDataExporter:
    """Exports ML-ready datasets from the database"""

    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize ML data exporter

        Args:
            output_dir: Directory to save exported data
        """
        self.db = get_db()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ML Data Exporter initialized (output: {output_dir})")

    def export_comprehensive_dataset(self,
                                     min_season: str = "2009-2010",
                                     include_current: bool = False) -> pd.DataFrame:
        """
        Export comprehensive dataset with all features

        Args:
            min_season: Minimum season to include (default: 2009-2010 for all available seasons)
            include_current: Include current incomplete season

        Returns:
            DataFrame with comprehensive features
        """
        logger.info(f"Exporting comprehensive dataset from {min_season}")

        # Build comprehensive query joining all data sources
        query = """
        WITH season_mapping AS (
            -- Map each season to its previous season for lookback features
            -- First get distinct seasons, then apply LAG window function
            SELECT season,
                   LAG(season) OVER (ORDER BY season) as prev_season
            FROM (
                SELECT DISTINCT season
                FROM matches
            )
        ),
        final_standings AS (
            -- Get only the final standings per team per season (max matchday)
            SELECT
                ls.*,
                ROW_NUMBER() OVER (PARTITION BY ls.team_id, ls.season ORDER BY ls.matchday DESC) as rn
            FROM league_standings ls
        ),
        -- POINT-IN-TIME H2H: Calculate head-to-head stats using ONLY matches before current match
        -- This prevents data leakage by ensuring we don't use future match results
        h2h_dynamic AS (
            SELECT
                m1.match_id as current_match_id,
                m1.home_team_id,
                m1.away_team_id,
                -- Total historical matches between these two teams (before current match)
                COUNT(m2.match_id) as h2h_total_matches,
                -- Wins from home team perspective
                SUM(CASE
                    WHEN (m2.home_team_id = m1.home_team_id AND m2.result = 'H')
                      OR (m2.away_team_id = m1.home_team_id AND m2.result = 'A')
                    THEN 1 ELSE 0
                END) as h2h_home_wins,
                -- Draws
                SUM(CASE WHEN m2.result = 'D' THEN 1 ELSE 0 END) as h2h_draws,
                -- Wins from away team perspective
                SUM(CASE
                    WHEN (m2.home_team_id = m1.away_team_id AND m2.result = 'H')
                      OR (m2.away_team_id = m1.away_team_id AND m2.result = 'A')
                    THEN 1 ELSE 0
                END) as h2h_away_wins
            FROM matches m1
            LEFT JOIN matches m2 ON (
                -- Match involves the same two teams
                ((m2.home_team_id = m1.home_team_id AND m2.away_team_id = m1.away_team_id) OR
                 (m2.home_team_id = m1.away_team_id AND m2.away_team_id = m1.home_team_id))
                -- CRITICAL: Only use matches that occurred BEFORE the current match
                AND m2.match_datetime < m1.match_datetime
                AND m2.is_finished = 1
                AND m2.home_goals IS NOT NULL
            )
            WHERE m1.is_finished = 1
            GROUP BY m1.match_id, m1.home_team_id, m1.away_team_id
        ),
        odds_agg AS (
            SELECT
                match_id,
                AVG(odds_home) AS odds_home,
                AVG(odds_draw) AS odds_draw,
                AVG(odds_away) AS odds_away,
                AVG(implied_prob_home) AS implied_prob_home,
                AVG(implied_prob_draw) AS implied_prob_draw,
                AVG(implied_prob_away) AS implied_prob_away
            FROM betting_odds
            WHERE bookmaker = 'oddsportal_avg'
            GROUP BY match_id
        ),
        -- Calculate rest days: find previous match for each team
        home_team_matches AS (
            SELECT
                m1.match_id as current_match_id,
                m1.home_team_id as team_id,
                m1.match_datetime as current_match_datetime,
                (SELECT MAX(m2.match_datetime)
                 FROM matches m2
                 WHERE m2.is_finished = 1
                   AND m2.match_datetime < m1.match_datetime
                   AND (m2.home_team_id = m1.home_team_id OR m2.away_team_id = m1.home_team_id)
                ) as previous_match_datetime
            FROM matches m1
            WHERE m1.is_finished = 1
        ),
        away_team_matches AS (
            SELECT
                m1.match_id as current_match_id,
                m1.away_team_id as team_id,
                m1.match_datetime as current_match_datetime,
                (SELECT MAX(m2.match_datetime)
                 FROM matches m2
                 WHERE m2.is_finished = 1
                   AND m2.match_datetime < m1.match_datetime
                   AND (m2.home_team_id = m1.away_team_id OR m2.away_team_id = m1.away_team_id)
                ) as previous_match_datetime
            FROM matches m1
            WHERE m1.is_finished = 1
        ),
        rest_days_home AS (
            SELECT
                current_match_id as match_id,
                CAST(julianday(current_match_datetime) - julianday(previous_match_datetime) AS INTEGER) as rest_days
            FROM home_team_matches
            WHERE previous_match_datetime IS NOT NULL
        ),
        rest_days_away AS (
            SELECT
                current_match_id as match_id,
                CAST(julianday(current_match_datetime) - julianday(previous_match_datetime) AS INTEGER) as rest_days
            FROM away_team_matches
            WHERE previous_match_datetime IS NOT NULL
        )
        SELECT
            -- Match identifiers
            m.match_id,
            m.season,
            m.matchday,
            m.match_datetime,

            -- Teams
            ht.team_name as home_team,
            at.team_name as away_team,
            m.home_team_id,
            m.away_team_id,

            -- Match result (target variables)
            m.home_goals,
            m.away_goals,
            m.result,

            -- Match context
            m.venue,
            m.attendance,

            -- Home team ratings (BEFORE match)
            htr.elo_rating as home_elo,
            htr.pi_rating as home_pi,
            htr.points_last_5 as home_points_l5,
            htr.points_last_10 as home_points_l10,
            htr.goals_scored_last_5 as home_goals_scored_l5,
            htr.goals_conceded_last_5 as home_goals_conceded_l5,

            -- Away team ratings (BEFORE match)
            atr.elo_rating as away_elo,
            atr.pi_rating as away_pi,
            atr.points_last_5 as away_points_l5,
            atr.points_last_10 as away_points_l10,
            atr.goals_scored_last_5 as away_goals_scored_l5,
            atr.goals_conceded_last_5 as away_goals_conceded_l5,

            -- Betting odds
            bo.odds_home,
            bo.odds_draw,
            bo.odds_away,
            bo.implied_prob_home,
            bo.implied_prob_draw,
            bo.implied_prob_away,

            -- Home team match statistics
            hms.possession_percent as home_possession,
            hms.shots_total as home_shots,
            hms.shots_on_target as home_shots_on_target,
            hms.big_chances as home_big_chances,
            hms.passes_total as home_passes,
            hms.pass_accuracy_percent as home_pass_accuracy,
            hms.tackles_total as home_tackles,
            hms.interceptions as home_interceptions,
            hms.fouls_committed as home_fouls,
            hms.corners as home_corners,
            hms.yellow_cards as home_yellow_cards,
            hms.red_cards as home_red_cards,

            -- Away team match statistics
            ams.possession_percent as away_possession,
            ams.shots_total as away_shots,
            ams.shots_on_target as away_shots_on_target,
            ams.big_chances as away_big_chances,
            ams.passes_total as away_passes,
            ams.pass_accuracy_percent as away_pass_accuracy,
            ams.tackles_total as away_tackles,
            ams.interceptions as away_interceptions,
            ams.fouls_committed as away_fouls,
            ams.corners as away_corners,
            ams.yellow_cards as away_yellow_cards,
            ams.red_cards as away_red_cards,

            -- Weather conditions at kickoff
            -- WARNING: These are OBSERVED weather values (post-match).
            -- For real prediction, you need FORECAST data (2-3 days before match).
            -- This creates minor leakage in backtesting but impact is minimal.
            -- TODO: Integrate weather forecast API for production predictions.
            m.temperature_celsius,
            m.humidity_percent,
            m.wind_speed_kmh,
            m.precipitation_mm,

            -- Derived features (rest/travel)
            rdh.rest_days as rest_days_home,
            rda.rest_days as rest_days_away,
            NULL as travel_distance_km,

            -- FBref features: Previous season final standings (team quality indicators)
            hls.position as home_prev_season_position,
            hls.points as home_prev_season_points,
            hls.wins as home_prev_season_wins,
            hls.goal_difference as home_prev_season_goal_diff,
            als.position as away_prev_season_position,
            als.points as away_prev_season_points,
            als.wins as away_prev_season_wins,
            als.goal_difference as away_prev_season_goal_diff,

            -- Head-to-head statistics (POINT-IN-TIME: only includes matches before current match)
            h2h.h2h_total_matches,
            h2h.h2h_home_wins,
            h2h.h2h_draws,
            h2h.h2h_away_wins

        FROM matches m

        -- Join teams
        JOIN teams ht ON m.home_team_id = ht.team_id
        JOIN teams at ON m.away_team_id = at.team_id

        -- Join ratings (CRITICAL: ratings are calculated BEFORE each match for prediction)
        LEFT JOIN team_ratings htr ON m.match_id = htr.match_id AND m.home_team_id = htr.team_id
        LEFT JOIN team_ratings atr ON m.match_id = atr.match_id AND m.away_team_id = atr.team_id

        -- Join betting odds (aggregated to one row per match)
        LEFT JOIN odds_agg bo ON m.match_id = bo.match_id

        -- Join match statistics (these are POST-match, for analysis only)
        LEFT JOIN match_statistics hms ON m.match_id = hms.match_id AND m.home_team_id = hms.team_id
        LEFT JOIN match_statistics ams ON m.match_id = ams.match_id AND m.away_team_id = ams.team_id

        -- Join point-in-time head-to-head statistics
        LEFT JOIN h2h_dynamic h2h ON m.match_id = h2h.current_match_id

        -- Join rest days calculations
        LEFT JOIN rest_days_home rdh ON m.match_id = rdh.match_id
        LEFT JOIN rest_days_away rda ON m.match_id = rda.match_id

        -- Join FBref league standings from PREVIOUS season (predictive features)
        -- Use final_standings CTE with rn=1 to avoid duplicates from multiple matchday records
        LEFT JOIN season_mapping sm ON m.season = sm.season
        LEFT JOIN final_standings hls ON m.home_team_id = hls.team_id AND sm.prev_season = hls.season AND hls.rn = 1
        LEFT JOIN final_standings als ON m.away_team_id = als.team_id AND sm.prev_season = als.season AND als.rn = 1

        WHERE m.is_finished = 1
            AND m.home_goals IS NOT NULL
            AND m.season >= ?

        ORDER BY m.match_datetime ASC
        """

        df = self.db.query_to_dataframe(query, params=(min_season,))
        logger.info(f"Loaded {len(df)} matches from database")

        # Engineer additional features
        df = self._engineer_features(df)
        # Enrich with travel distance if coordinates available
        df = self._add_travel_distance(df)

        # Add data quality flags
        df = self._add_quality_flags(df)

        logger.success(f"Exported dataset with {len(df)} matches and {len(df.columns)} features")
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features from raw data

        Args:
            df: Raw dataframe

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering additional features...")

        # Elo difference (key feature)
        df['elo_diff'] = df['home_elo'] - df['away_elo']

        # Pi-rating difference
        df['pi_diff'] = df['home_pi'] - df['away_pi']

        # Form difference (points in last 5)
        df['form_diff_l5'] = df['home_points_l5'] - df['away_points_l5']

        # Goal difference indicators
        df['goal_diff_l5'] = (df['home_goals_scored_l5'] - df['home_goals_conceded_l5']) - \
                             (df['away_goals_scored_l5'] - df['away_goals_conceded_l5'])

        # FBref derived features (previous season quality indicators)
        if 'home_prev_season_position' in df.columns:
            # Position difference (lower is better, so home - away; negative = home was better)
            df['prev_season_position_diff'] = df['home_prev_season_position'] - df['away_prev_season_position']

            # Points difference (higher is better)
            df['prev_season_points_diff'] = df['home_prev_season_points'] - df['away_prev_season_points']

            # Quality tier indicators (position-based)
            # Top 3 = promotion candidates, Bottom 3 = relegation candidates
            df['home_was_top_tier'] = (df['home_prev_season_position'] <= 3).astype(int)
            df['home_was_bottom_tier'] = (df['home_prev_season_position'] >= 18).astype(int)
            df['away_was_top_tier'] = (df['away_prev_season_position'] <= 3).astype(int)
            df['away_was_bottom_tier'] = (df['away_prev_season_position'] >= 18).astype(int)

            # New teams (no previous season data) - useful indicator
            df['home_is_new_team'] = df['home_prev_season_position'].isna().astype(int)
            df['away_is_new_team'] = df['away_prev_season_position'].isna().astype(int)

        # Head-to-head features (already in home/away perspective from point-in-time calculation)
        if 'h2h_total_matches' in df.columns:
            # Fill NaN values with 0 (no previous matchups)
            df['h2h_total_matches'] = df['h2h_total_matches'].fillna(0)
            df['h2h_home_wins'] = df['h2h_home_wins'].fillna(0)
            df['h2h_draws'] = df['h2h_draws'].fillna(0)
            df['h2h_away_wins'] = df['h2h_away_wins'].fillna(0)

            # Calculate H2H win rates (avoid division by zero)
            df['h2h_home_win_rate'] = np.where(
                df['h2h_total_matches'] > 0,
                df['h2h_home_wins'] / df['h2h_total_matches'],
                np.nan
            )
            df['h2h_draw_rate'] = np.where(
                df['h2h_total_matches'] > 0,
                df['h2h_draws'] / df['h2h_total_matches'],
                np.nan
            )
            df['h2h_away_win_rate'] = np.where(
                df['h2h_total_matches'] > 0,
                df['h2h_away_wins'] / df['h2h_total_matches'],
                np.nan
            )

            # H2H match count (useful indicator for new matchups)
            df['h2h_match_count'] = df['h2h_total_matches']

        # Rest days features (only if present and non-null)
        if 'rest_days_home' in df.columns and df['rest_days_home'].notna().any() and 'rest_days_away' in df.columns:
            df['rest_days_diff'] = df['rest_days_home'] - df['rest_days_away']
            df['home_rest_advantage'] = (df['rest_days_home'] > df['rest_days_away']).astype(int)
            df['away_rest_advantage'] = (df['rest_days_away'] > df['rest_days_home']).astype(int)
            # Categorize rest days
            df['home_rest_category'] = pd.cut(
                df['rest_days_home'],
                bins=[-1, 2, 4, 7, float('inf')],
                labels=['short', 'medium', 'normal', 'long']
            )
            df['away_rest_category'] = pd.cut(
                df['rest_days_away'],
                bins=[-1, 2, 4, 7, float('inf')],
                labels=['short', 'medium', 'normal', 'long']
            )

        # Travel distance features (only if present and non-null)
        if 'travel_distance_km' in df.columns and df['travel_distance_km'].notna().any():
            # Categorize travel distance
            df['travel_category'] = pd.cut(
                df['travel_distance_km'],
                bins=[0, 50, 200, float('inf')],
                labels=['local', 'regional', 'long']
            )
            df['is_long_travel'] = (df['travel_distance_km'] > 200).astype(int)
            df['is_local_match'] = (df['travel_distance_km'] < 50).astype(int)

        # Date features
        df['match_datetime'] = pd.to_datetime(df['match_datetime'])
        df['day_of_week'] = df['match_datetime'].dt.dayofweek
        df['month'] = df['match_datetime'].dt.month
        df['year'] = df['match_datetime'].dt.year
        # Derived context features
        df['is_midweek'] = df['day_of_week'].isin([1, 2, 3]).astype(int)

        # Weather features
        if 'temperature_celsius' in df.columns:
            # Temperature categories
            df['is_hot'] = (df['temperature_celsius'] > 25).astype(int)
            df['is_cold'] = (df['temperature_celsius'] < 10).astype(int)

            # Precipitation categories
            df['is_rainy'] = (df['precipitation_mm'] > 0.5).astype(int)
            df['is_heavy_rain'] = (df['precipitation_mm'] > 5.0).astype(int)

            # Wind categories
            df['is_windy'] = (df['wind_speed_kmh'] > 30).astype(int)

            # Extreme weather flag (any adverse condition)
            df['is_extreme_weather'] = (
                (df['temperature_celsius'] < 5) |
                (df['temperature_celsius'] > 30) |
                (df['precipitation_mm'] > 5.0) |
                (df['wind_speed_kmh'] > 40)
            ).astype(int)

        # Categorical encoding prep
        df['is_home'] = 1  # Always 1 for this format (home team perspective)

        # Target encoding (for classification)
        df['target_home_win'] = (df['result'] == 'H').astype(int)
        df['target_draw'] = (df['result'] == 'D').astype(int)
        df['target_away_win'] = (df['result'] == 'A').astype(int)

        # Multi-class target (0=Away win, 1=Draw, 2=Home win)
        df['target_multiclass'] = df['result'].map({'A': 0, 'D': 1, 'H': 2})

        # Regression targets
        df['target_home_goals'] = df['home_goals']
        df['target_away_goals'] = df['away_goals']
        df['target_total_goals'] = df['home_goals'] + df['away_goals']

        logger.info(f"Engineered {len(df.columns)} total features")
        return df

    def _add_travel_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute great-circle (haversine) distance between home and away team locations.
        Requires team_locations table with lat/lon.
        """
        try:
            locs = self.db.query_to_dataframe("SELECT team_id, lat, lon FROM team_locations WHERE lat IS NOT NULL AND lon IS NOT NULL")
            if locs.empty:
                return df
            locs = locs.set_index('team_id')
            # Map coordinates
            df['home_lat'] = df['home_team_id'].map(locs['lat'])
            df['home_lon'] = df['home_team_id'].map(locs['lon'])
            df['away_lat'] = df['away_team_id'].map(locs['lat'])
            df['away_lon'] = df['away_team_id'].map(locs['lon'])

            import numpy as np
            def haversine_np(lat1, lon1, lat2, lon2):
                R = 6371.0  # km
                lat1 = np.radians(lat1)
                lon1 = np.radians(lon1)
                lat2 = np.radians(lat2)
                lon2 = np.radians(lon2)
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
                c = 2 * np.arcsin(np.sqrt(a))
                return R * c

            mask = df[['home_lat','home_lon','away_lat','away_lon']].notna().all(axis=1)
            if mask.any():
                df.loc[mask, 'travel_distance_km'] = haversine_np(
                    df.loc[mask, 'home_lat'].values,
                    df.loc[mask, 'home_lon'].values,
                    df.loc[mask, 'away_lat'].values,
                    df.loc[mask, 'away_lon'].values
                )
            # Cleanup helper columns
            df = df.drop(columns=['home_lat','home_lon','away_lat','away_lon'])
            return df
        except Exception as e:
            logger.warning(f"Travel distance computation failed: {e}")
            return df

    def _add_quality_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add data quality indicators"""

        # Flag indicating if match has detailed statistics
        df['has_detailed_stats'] = df['home_possession'].notna().astype(int)

        # Flag indicating if match has betting odds
        df['has_odds'] = df['odds_home'].notna().astype(int)

        # Flag indicating if match has ratings
        df['has_ratings'] = df['home_elo'].notna().astype(int)

        # Flag indicating if match has weather data
        if 'temperature_celsius' in df.columns:
            df['has_weather'] = df['temperature_celsius'].notna().astype(int)

        return df

    def create_train_test_split(self,
                                df: pd.DataFrame,
                                test_size: float = 0.2,
                                val_size: float = 0.1,
                                temporal: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits

        Args:
            df: Full dataset
            test_size: Proportion for test set
            val_size: Proportion for validation set
            temporal: If True, use temporal split (recommended for time series)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Creating train/val/test split (test={test_size}, val={val_size}, temporal={temporal})")

        if temporal:
            # Temporal split: train on older data, test on recent
            df = df.sort_values('match_datetime')

            n = len(df)
            test_idx = int(n * (1 - test_size))
            val_idx = int(test_idx * (1 - val_size))

            train_df = df.iloc[:val_idx].copy()
            val_df = df.iloc[val_idx:test_idx].copy()
            test_df = df.iloc[test_idx:].copy()

            logger.info(f"Temporal split: train up to {train_df['match_datetime'].max()}")
            logger.info(f"                val: {val_df['match_datetime'].min()} to {val_df['match_datetime'].max()}")
            logger.info(f"                test from {test_df['match_datetime'].min()}")

        else:
            # Random split (not recommended for time series, but included for completeness)
            from sklearn.model_selection import train_test_split

            train_val, test_df = train_test_split(df, test_size=test_size, random_state=42)
            train_df, val_df = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)

        logger.success(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df

    def get_feature_columns(self, include_post_match: bool = False) -> Dict[str, list]:
        """
        Get categorized feature columns for ML

        Args:
            include_post_match: Include post-match statistics (for analysis, not prediction)

        Returns:
            Dictionary with categorized feature lists
        """
        features = {
            'categorical': [
                'home_team', 'away_team', 'venue',
                'day_of_week', 'month', 'is_midweek'
            ],
            'rating_features': [
                'home_elo', 'away_elo', 'elo_diff',
                'home_pi', 'away_pi', 'pi_diff'
            ],
            'form_features': [
                'home_points_l5', 'away_points_l5', 'form_diff_l5',
                'home_points_l10', 'away_points_l10',
                'home_goals_scored_l5', 'home_goals_conceded_l5',
                'away_goals_scored_l5', 'away_goals_conceded_l5',
                'goal_diff_l5'
            ],
            'odds_features': [
                'odds_home', 'odds_draw', 'odds_away',
                'implied_prob_home', 'implied_prob_draw', 'implied_prob_away'
            ],
            'fbref_features': [
                'home_prev_season_position', 'away_prev_season_position',
                'home_prev_season_points', 'away_prev_season_points',
                'home_prev_season_wins', 'away_prev_season_wins',
                'home_prev_season_goal_diff', 'away_prev_season_goal_diff',
                'prev_season_position_diff', 'prev_season_points_diff',
                'home_was_top_tier', 'home_was_bottom_tier',
                'away_was_top_tier', 'away_was_bottom_tier',
                'home_is_new_team', 'away_is_new_team'
            ],
            'targets': [
                'target_home_win', 'target_draw', 'target_away_win',
                'target_multiclass',
                'target_home_goals', 'target_away_goals', 'target_total_goals'
            ],
            'identifiers': [
                'match_id', 'season', 'matchday', 'match_datetime',
                'home_team', 'away_team'
            ]
        }

        if include_post_match:
            features['post_match_stats'] = [
                'home_possession', 'away_possession',
                'home_shots', 'away_shots',
                'home_shots_on_target', 'away_shots_on_target',
                'home_pass_accuracy', 'away_pass_accuracy',
                'home_corners', 'away_corners',
                'home_fouls', 'away_fouls'
            ]

        # All predictive features (for training)
        features['all_predictive'] = (
            features['categorical'] +
            features['rating_features'] +
            features['form_features'] +
            features['odds_features'] +
            features['fbref_features']
        )

        return features

    def export_to_csv(self, save_splits: bool = True):
        """
        Export full dataset and splits to CSV files

        Args:
            save_splits: If True, also save train/val/test splits
        """
        logger.info("=== Exporting ML datasets to CSV ===")

        # Export full dataset
        df_full = self.export_comprehensive_dataset()
        output_file = self.output_dir / "3liga_ml_dataset_full.csv"
        df_full.to_csv(output_file, index=False)
        logger.success(f"Saved full dataset: {output_file} ({len(df_full)} rows)")

        if save_splits:
            # Create and save splits
            train_df, val_df, test_df = self.create_train_test_split(df_full)

            train_file = self.output_dir / "3liga_ml_dataset_train.csv"
            val_file = self.output_dir / "3liga_ml_dataset_val.csv"
            test_file = self.output_dir / "3liga_ml_dataset_test.csv"

            train_df.to_csv(train_file, index=False)
            val_df.to_csv(val_file, index=False)
            test_df.to_csv(test_file, index=False)

            logger.success(f"Saved train set: {train_file} ({len(train_df)} rows)")
            logger.success(f"Saved val set: {val_file} ({len(val_df)} rows)")
            logger.success(f"Saved test set: {test_file} ({len(test_df)} rows)")

        # Save feature documentation
        features = self.get_feature_columns(include_post_match=True)
        feature_doc = self.output_dir / "feature_documentation.txt"
        with open(feature_doc, 'w') as f:
            f.write("=== 3. Liga ML Dataset Feature Documentation ===\n\n")
            for category, feature_list in features.items():
                f.write(f"\n{category.upper()}:\n")
                for feat in feature_list:
                    f.write(f"  - {feat}\n")

        logger.success(f"Saved feature documentation: {feature_doc}")

        # Generate dataset summary
        self._generate_summary(df_full, train_df, val_df, test_df)

    def _generate_summary(self, df_full, train_df, val_df, test_df):
        """Generate and save dataset summary statistics"""
        summary_file = self.output_dir / "dataset_summary.txt"

        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("3. LIGA ML DATASET SUMMARY\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("DATASET SIZES:\n")
            f.write(f"  Full dataset: {len(df_full):,} matches\n")
            f.write(f"  Train set: {len(train_df):,} matches\n")
            f.write(f"  Validation set: {len(val_df):,} matches\n")
            f.write(f"  Test set: {len(test_df):,} matches\n\n")

            f.write("TEMPORAL COVERAGE:\n")
            f.write(f"  Date range: {df_full['match_datetime'].min()} to {df_full['match_datetime'].max()}\n")
            f.write(f"  Seasons: {df_full['season'].nunique()}\n")
            f.write(f"  Season list: {', '.join(sorted(df_full['season'].unique()))}\n\n")

            f.write("DATA QUALITY:\n")
            f.write(f"  Matches with ratings: {df_full['has_ratings'].sum():,} ({df_full['has_ratings'].mean()*100:.1f}%)\n")
            f.write(f"  Matches with detailed stats: {df_full['has_detailed_stats'].sum():,} ({df_full['has_detailed_stats'].mean()*100:.1f}%)\n")
            f.write(f"  Matches with betting odds: {df_full['has_odds'].sum():,} ({df_full['has_odds'].mean()*100:.1f}%)\n\n")

            f.write("TARGET DISTRIBUTION (Full Dataset):\n")
            f.write(f"  Home wins: {df_full['target_home_win'].sum():,} ({df_full['target_home_win'].mean()*100:.1f}%)\n")
            f.write(f"  Draws: {df_full['target_draw'].sum():,} ({df_full['target_draw'].mean()*100:.1f}%)\n")
            f.write(f"  Away wins: {df_full['target_away_win'].sum():,} ({df_full['target_away_win'].mean()*100:.1f}%)\n\n")

            f.write("FEATURE STATISTICS:\n")
            f.write(f"  Total features: {len(df_full.columns)}\n")
            f.write(f"  Missing data:\n")
            missing = df_full.isnull().sum()
            for col in missing[missing > 0].sort_values(ascending=False).head(10).index:
                pct = (missing[col] / len(df_full)) * 100
                f.write(f"    {col}: {missing[col]:,} ({pct:.1f}%)\n")

        logger.success(f"Saved dataset summary: {summary_file}")


if __name__ == "__main__":
    print("Use CLI instead: liga-predictor export-ml-data")
    import sys
    sys.exit(1)
