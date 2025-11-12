#!/usr/bin/env python3
"""
Update Latest Team Ratings
Recalculates and stores the most current ratings for all teams based on their latest finished matches.
This ensures predictions use ratings that include the most recent match results.

Usage:
    python scripts/processors/update_latest_ratings.py [--season SEASON]
"""

import sys
from pathlib import Path
import argparse
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent.parent))
from database.db_manager import get_db
from scripts.processors.rating_calculator import RatingCalculator
import pandas as pd


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
        
        # Update the existing rating record for this match with corrected form metrics
        # The form metrics should reflect the state AFTER this match
        update_query = """
            UPDATE team_ratings
            SET points_last_5 = ?,
                goals_scored_last_5 = ?,
                goals_conceded_last_5 = ?,
                created_at = CURRENT_TIMESTAMP
            WHERE team_id = ? AND match_id = ?
        """
        
        cursor = conn.cursor()
        cursor.execute(update_query, (
            points_last_5, goals_scored_avg, goals_conceded_avg,
            team_id, match_id
        ))
        
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


def main():
    parser = argparse.ArgumentParser(
        description='Update latest team ratings with corrected form metrics'
    )
    parser.add_argument(
        '--season',
        type=str,
        help='Specific season to update (e.g., 2025-2026). Default: all seasons'
    )
    
    args = parser.parse_args()
    
    update_latest_ratings(season=args.season)
    
    logger.info("Rating update complete!")


if __name__ == "__main__":
    main()

