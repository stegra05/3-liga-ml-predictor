"""
Rating Systems Calculator for 3. Liga
Calculates Elo ratings, Pi-ratings, and team form metrics
Research shows these are the most important features for gradient boosting models
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from loguru import logger
from pathlib import Path
from datetime import datetime
import argparse

from liga_predictor.database import get_db
from liga_predictor.models import TeamRating


class RatingCalculator:
    """Calculates team rating systems"""

    def __init__(self, initial_elo: float = 1500.0, k_factor: float = 32.0):
        """
        Initialize rating calculator

        Args:
            initial_elo: Initial Elo rating for new teams
            k_factor: K-factor for Elo calculations (higher = more volatile)
        """
        self.db = get_db()
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        logger.info(f"Rating calculator initialized (initial_elo={initial_elo}, k={k_factor})")

    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for team A using Elo formula

        Args:
            rating_a: Team A's rating
            rating_b: Team B's rating

        Returns:
            Expected score (0-1)
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_elo(self, rating: float, expected: float, actual: float, k: float = None) -> float:
        """
        Update Elo rating after a match

        Args:
            rating: Current rating
            expected: Expected score (0-1)
            actual: Actual score (1=win, 0.5=draw, 0=loss)
            k: K-factor (uses self.k_factor if None)

        Returns:
            New rating
        """
        if k is None:
            k = self.k_factor
        return rating + k * (actual - expected)

    def calculate_pi_rating(self, team_data: pd.DataFrame) -> float:
        """
        Calculate Pi-rating for a team based on recent performance
        Pi-ratings are identified in research as the most suitable features for gradient boosting

        Formula based on: points earned / max possible points, with recency weighting

        Args:
            team_data: DataFrame with recent matches (should be sorted by date)

        Returns:
            Pi-rating value
        """
        if len(team_data) == 0:
            return 0.5  # Neutral rating

        # Get points from results
        points_map = {'H': 3, 'D': 1, 'A': 0}  # From home team perspective

        # Calculate weighted points (more recent = higher weight)
        weights = np.exp(np.linspace(-1, 0, len(team_data)))  # Exponential decay
        weights = weights / weights.sum()

        points = []
        for _, match in team_data.iterrows():
            if match['is_home']:
                result = match['result']
            else:
                # Reverse result for away team
                result = 'H' if match['result'] == 'A' else ('A' if match['result'] == 'H' else 'D')

            points.append(points_map.get(result, 0))

        weighted_points = np.average(points, weights=weights)
        pi_rating = weighted_points / 3.0  # Normalize to 0-1

        return float(pi_rating)

    def calculate_all_ratings(self, season: str = None, force_recalculate: bool = False):
        """
        Calculate all rating systems for all matches

        Args:
            season: Specific season to calculate (None = all seasons)
            force_recalculate: If True, recalculate even if ratings exist
        """
        logger.info(f"=== Calculating all rating systems {f'for season {season}' if season else 'for all seasons'} ===")

        # Get all matches sorted by date
        query = """
            SELECT m.*, ht.team_name as home_team, at.team_name as away_team
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.team_id
            JOIN teams at ON m.away_team_id = at.team_id
            WHERE m.is_finished = 1 AND m.home_goals IS NOT NULL
        """

        if season:
            query += f" AND m.season = '{season}'"

        query += " ORDER BY m.match_datetime ASC"

        matches_df = self.db.query_to_dataframe(query)
        logger.info(f"Processing {len(matches_df)} finished matches")

        # Initialize Elo ratings dict
        elo_ratings = {}  # team_id -> current_elo

        # Initialize match history for Pi-ratings (last 10 matches per team)
        team_history = {}  # team_id -> list of recent matches

        ratings_to_insert = []

        for idx, match in matches_df.iterrows():
            match_id = match['match_id']
            home_team_id = match['home_team_id']
            away_team_id = match['away_team_id']

            # Initialize Elo if not exists
            if home_team_id not in elo_ratings:
                elo_ratings[home_team_id] = self.initial_elo
            if away_team_id not in elo_ratings:
                elo_ratings[away_team_id] = self.initial_elo

            # Get current Elo ratings
            home_elo = elo_ratings[home_team_id]
            away_elo = elo_ratings[away_team_id]

            # Calculate expected scores
            home_expected = self.calculate_expected_score(home_elo, away_elo)
            away_expected = 1 - home_expected

            # Determine actual scores
            result = match['result']
            if result == 'H':
                home_actual, away_actual = 1.0, 0.0
            elif result == 'A':
                home_actual, away_actual = 0.0, 1.0
            else:  # Draw
                home_actual, away_actual = 0.5, 0.5

            # Update Elo ratings
            new_home_elo = self.update_elo(home_elo, home_expected, home_actual)
            new_away_elo = self.update_elo(away_elo, away_expected, away_actual)

            # Calculate Pi-ratings based on recent history
            if home_team_id not in team_history:
                team_history[home_team_id] = []
            if away_team_id not in team_history:
                team_history[away_team_id] = []

            home_matches_df = pd.DataFrame(team_history[home_team_id][-10:]) if team_history[home_team_id] else pd.DataFrame()
            away_matches_df = pd.DataFrame(team_history[away_team_id][-10:]) if team_history[away_team_id] else pd.DataFrame()

            home_pi = self.calculate_pi_rating(home_matches_df)
            away_pi = self.calculate_pi_rating(away_matches_df)

            # Calculate form metrics (points in last 5 and 10 matches)
            home_points_5 = self._calculate_recent_points(team_history[home_team_id], 5)
            away_points_5 = self._calculate_recent_points(team_history[away_team_id], 5)
            home_points_10 = self._calculate_recent_points(team_history[home_team_id], 10)
            away_points_10 = self._calculate_recent_points(team_history[away_team_id], 10)

            # Calculate goals metrics
            home_goals_5 = self._calculate_recent_goals(team_history[home_team_id], 5, scored=True)
            home_conceded_5 = self._calculate_recent_goals(team_history[home_team_id], 5, scored=False)
            away_goals_5 = self._calculate_recent_goals(team_history[away_team_id], 5, scored=True)
            away_conceded_5 = self._calculate_recent_goals(team_history[away_team_id], 5, scored=False)

            # Store ratings for this match (BEFORE the match - for prediction)
            ratings_to_insert.append({
                'team_id': home_team_id,
                'match_id': match_id,
                'season': match['season'],
                'matchday': match['matchday'],
                'elo_rating': home_elo,
                'pi_rating': home_pi,
                'points_last_5': home_points_5,
                'points_last_10': home_points_10,
                'goals_scored_last_5': home_goals_5,
                'goals_conceded_last_5': home_conceded_5
            })

            ratings_to_insert.append({
                'team_id': away_team_id,
                'match_id': match_id,
                'season': match['season'],
                'matchday': match['matchday'],
                'elo_rating': away_elo,
                'pi_rating': away_pi,
                'points_last_5': away_points_5,
                'points_last_10': away_points_10,
                'goals_scored_last_5': away_goals_5,
                'goals_conceded_last_5': away_conceded_5
            })

            # Update Elo ratings for next iteration
            elo_ratings[home_team_id] = new_home_elo
            elo_ratings[away_team_id] = new_away_elo

            # Update match history
            team_history[home_team_id].append({
                'match_id': match_id,
                'is_home': True,
                'result': result,
                'goals_scored': match['home_goals'],
                'goals_conceded': match['away_goals']
            })
            team_history[away_team_id].append({
                'match_id': match_id,
                'is_home': False,
                'result': result,
                'goals_scored': match['away_goals'],
                'goals_conceded': match['home_goals']
            })

            if (idx + 1) % 500 == 0:
                logger.info(f"Processed {idx + 1}/{len(matches_df)} matches")

        # Insert all ratings into database
        logger.info(f"Inserting {len(ratings_to_insert)} rating records into database")
        self._bulk_insert_ratings(ratings_to_insert)

        logger.success(f"Rating calculation complete: {len(ratings_to_insert)} records")

    def _calculate_recent_points(self, match_history: List[Dict], n_matches: int) -> int:
        """Calculate points earned in last n matches"""
        if not match_history:
            return 0

        recent = match_history[-n_matches:]
        points = 0
        for match in recent:
            if match['is_home']:
                result = match['result']
            else:
                # Reverse for away team
                result = 'H' if match['result'] == 'A' else ('A' if match['result'] == 'H' else 'D')

            if result == 'H':
                points += 3
            elif result == 'D':
                points += 1

        return points

    def _calculate_recent_goals(self, match_history: List[Dict], n_matches: int, scored: bool = True) -> float:
        """Calculate average goals scored or conceded in last n matches"""
        if not match_history:
            return 0.0

        recent = match_history[-n_matches:]
        goals = [m['goals_scored'] if scored else m['goals_conceded'] for m in recent]
        return float(np.mean(goals)) if goals else 0.0

    def _bulk_insert_ratings(self, ratings: List[Dict]):
        """Bulk insert ratings into database using ORM"""
        if not ratings:
            return

        for r in ratings:
            self.db.merge_or_create(
                TeamRating,
                filter_dict={
                    'team_id': r['team_id'],
                    'match_id': r.get('match_id'),
                    'season': r['season'],
                    'matchday': r['matchday']
                },
                defaults={
                    'elo_rating': r['elo_rating'],
                    'pi_rating': r['pi_rating'],
                    'points_last_5': r['points_last_5'],
                    'points_last_10': r['points_last_10'],
                    'goals_scored_last_5': r['goals_scored_last_5'],
                    'goals_conceded_last_5': r['goals_conceded_last_5']
                }
            )


def update_latest_ratings(season: str = None):
    """
    Update the latest ratings for all teams by calculating post-match ratings
    for their most recent finished matches.

    This creates new rating records that include the most recent match results,
    which are needed for accurate predictions.

    Args:
        season: Optional season to update (default: all seasons)
    """
    logger.info("=== Updating Latest Team Ratings ===")

    db = get_db()
    conn = db.get_connection()

    # Get all teams
    teams_df = pd.read_sql_query("SELECT team_id, team_name FROM teams", conn)
    logger.info(f"Found {len(teams_df)} teams")

    updated_count = 0

    for _, team_row in teams_df.iterrows():
        team_id = team_row['team_id']
        team_name = team_row['team_name']

        # Get the most recent finished match for this team
        query = """
            SELECT m.match_id, m.season, m.matchday, m.match_datetime,
                   m.home_team_id, m.away_team_id, m.home_goals, m.away_goals, m.result
            FROM matches m
            WHERE (m.home_team_id = ? OR m.away_team_id = ?)
              AND m.is_finished = 1
              AND m.home_goals IS NOT NULL
        """
        params = [team_id, team_id]

        if season:
            query += " AND m.season = ?"
            params.append(season)

        query += " ORDER BY m.match_datetime DESC LIMIT 1"

        latest_match = pd.read_sql_query(query, conn, params=params)

        if latest_match.empty:
            logger.debug(f"No finished matches found for {team_name}")
            continue

        match = latest_match.iloc[0]
        match_id = match['match_id']

        # Get the last 5 finished matches for this team (including the latest)
        form_query = """
            SELECT m.match_id, m.match_datetime, m.home_team_id, m.away_team_id,
                   m.home_goals, m.away_goals, m.result, m.season, m.matchday
            FROM matches m
            WHERE (m.home_team_id = ? OR m.away_team_id = ?)
              AND m.is_finished = 1
              AND m.home_goals IS NOT NULL
              AND m.match_datetime <= ?
        """
        if season:
            form_query += " AND m.season = ?"
            form_params = [team_id, team_id, match['match_datetime'], season]
        else:
            form_params = [team_id, team_id, match['match_datetime']]

        form_query += " ORDER BY m.match_datetime DESC LIMIT 5"

        recent_matches = pd.read_sql_query(form_query, conn, params=form_params)

        # Calculate points from last 5 matches
        points_last_5 = 0
        goals_scored = []
        goals_conceded = []

        for _, m in recent_matches.iterrows():
            is_home = m['home_team_id'] == team_id
            if is_home:
                result = m['result']
                goals_scored.append(m['home_goals'])
                goals_conceded.append(m['away_goals'])
            else:
                # Reverse result for away team
                if m['result'] == 'H':
                    result = 'A'
                elif m['result'] == 'A':
                    result = 'H'
                else:
                    result = 'D'
                goals_scored.append(m['away_goals'])
                goals_conceded.append(m['home_goals'])

            if result == 'H':
                points_last_5 += 3
            elif result == 'D':
                points_last_5 += 1

        goals_scored_avg = sum(goals_scored) / len(goals_scored) if goals_scored else 0.0
        goals_conceded_avg = sum(goals_conceded) / len(goals_conceded) if goals_conceded else 0.0

        # Get the latest Elo and PI ratings (these are correct as they're updated after each match)
        rating_query = """
            SELECT elo_rating, pi_rating, points_last_10
            FROM team_ratings
            WHERE team_id = ? AND match_id = ?
            LIMIT 1
        """
        current_rating = pd.read_sql_query(rating_query, conn, params=(team_id, match_id))

        if not current_rating.empty:
            elo_rating = current_rating.iloc[0]['elo_rating']
            pi_rating = current_rating.iloc[0]['pi_rating']
            points_last_10 = current_rating.iloc[0]['points_last_10']
        else:
            # Fallback: get latest rating
            fallback_query = """
                SELECT elo_rating, pi_rating, points_last_10
                FROM team_ratings
                WHERE team_id = ?
                ORDER BY matchday DESC, created_at DESC
                LIMIT 1
            """
            fallback_rating = pd.read_sql_query(fallback_query, conn, params=(team_id,))
            if not fallback_rating.empty:
                elo_rating = fallback_rating.iloc[0]['elo_rating']
                pi_rating = fallback_rating.iloc[0]['pi_rating']
                points_last_10 = fallback_rating.iloc[0]['points_last_10']
            else:
                logger.warning(f"No existing ratings found for {team_name}, skipping")
                continue

        # CRITICAL FIX: Do NOT update the existing rating record for this match.
        # The existing record (where match_id is set) represents the state BEFORE the match,
        # which is what we need for training. Updating it with post-match form would cause leakage.
        
        # We ONLY update the "current" rating (match_id = NULL) which represents the state
        # AFTER the match, ready for the NEXT prediction.

        # Also create a "current" rating record (match_id = NULL) for easy querying
        # This represents the team's state after their most recent match
        insert_current_query = """
            INSERT OR REPLACE INTO team_ratings
            (team_id, match_id, season, matchday, elo_rating, pi_rating,
             points_last_5, points_last_10, goals_scored_last_5, goals_conceded_last_5,
             created_at)
            VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """

        cursor.execute(insert_current_query, (
            team_id, match['season'], match['matchday'],
            elo_rating, pi_rating,
            points_last_5, points_last_10,
            goals_scored_avg, goals_conceded_avg
        ))

        updated_count += 1

    conn.commit()
    conn.close()

    logger.success(f"Updated latest ratings for {updated_count} teams")
    return updated_count


if __name__ == "__main__":
    print("Use CLI instead: liga-predictor calculate-ratings")
    import sys
    sys.exit(1)
