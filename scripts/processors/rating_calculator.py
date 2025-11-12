"""
Rating Systems Calculator for 3. Liga
Calculates Elo ratings, Pi-ratings, and team form metrics
Research shows these are the most important features for gradient boosting models
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from loguru import logger
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from database.db_manager import get_db


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
        """Bulk insert ratings into database"""
        if not ratings:
            return

        query = """
            INSERT OR REPLACE INTO team_ratings
            (team_id, match_id, season, matchday, elo_rating, pi_rating,
             points_last_5, points_last_10, goals_scored_last_5, goals_conceded_last_5)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params_list = [
            (r['team_id'], r['match_id'], r['season'], r['matchday'],
             r['elo_rating'], r['pi_rating'], r['points_last_5'], r['points_last_10'],
             r['goals_scored_last_5'], r['goals_conceded_last_5'])
            for r in ratings
        ]

        self.db.execute_many(query, params_list)


def main():
    """Main execution"""
    calculator = RatingCalculator(initial_elo=1500.0, k_factor=32.0)

    # Calculate ratings for all matches
    calculator.calculate_all_ratings()

    # Print statistics
    stats = calculator.db.get_database_stats()
    print("\n=== Database Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Show sample ratings
    query = """
        SELECT t.team_name, tr.season, tr.matchday,
               tr.elo_rating, tr.pi_rating, tr.points_last_5
        FROM team_ratings tr
        JOIN teams t ON tr.team_id = t.team_id
        ORDER BY tr.season DESC, tr.matchday DESC
        LIMIT 20
    """
    ratings_sample = calculator.db.query_to_dataframe(query)
    print("\n=== Sample Ratings (Recent) ===")
    print(ratings_sample.to_string(index=False))


if __name__ == "__main__":
    import sys
    print("This script is deprecated. Use: python main.py rating-calculator [args]", file=sys.stderr)
    sys.exit(2)
