"""
OpenLigaDB API Collector for 3. Liga
Fetches comprehensive match data, league tables, and events
"""

import requests
from typing import List, Dict, Optional, Any
from datetime import datetime
import pandas as pd
from loguru import logger
import time
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from database.db_manager import get_db
from scripts.utils.team_mapper import TeamMapper


class OpenLigaDBCollector:
    """Collects data from OpenLigaDB API"""

    BASE_URL = "https://api.openligadb.de"
    LEAGUE_CODE = "bl3"  # 3. Liga

    def __init__(self):
        """Initialize OpenLigaDB collector"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (3Liga-Dataset-Collector/1.0)'
        })
        self.db = get_db()
        self.team_mapper = TeamMapper()
        logger.info("OpenLigaDB collector initialized")

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Any]:
        """
        Make API request with error handling

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response or None on error
        """
        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {url} - {e}")
            return None

    def get_available_seasons(self) -> List[str]:
        """
        Get list of available seasons for 3. Liga

        Returns:
            List of season strings (e.g., ['2009', '2010', ...])
        """
        endpoint = f"availableLeagues/{self.LEAGUE_CODE}"
        data = self._make_request(endpoint)

        if data:
            # OpenLigaDB returns list of available years
            seasons = [item.get('league', {}).get('leagueShortcut', '') for item in data]
            # Filter for actual season years
            seasons = [s.replace(self.LEAGUE_CODE, '').strip() for s in seasons if self.LEAGUE_CODE in s.lower()]
            logger.info(f"Found {len(seasons)} available seasons")
            return seasons

        return []

    def get_matchdata_for_season(self, season: str) -> List[Dict]:
        """
        Get all match data for a specific season

        Args:
            season: Season year (e.g., '2024' for 2024/2025 season)

        Returns:
            List of match dictionaries
        """
        endpoint = f"getmatchdata/{self.LEAGUE_CODE}/{season}"
        logger.info(f"Fetching matches for season {season}")

        data = self._make_request(endpoint)

        if data:
            logger.success(f"Retrieved {len(data)} matches for season {season}")
            return data

        logger.warning(f"No data retrieved for season {season}")
        return []

    def get_matchdata_for_matchday(self, season: str, matchday: int) -> List[Dict]:
        """
        Get match data for specific matchday

        Args:
            season: Season year
            matchday: Matchday number

        Returns:
            List of match dictionaries
        """
        endpoint = f"getmatchdata/{self.LEAGUE_CODE}/{season}/{matchday}"
        logger.info(f"Fetching matches for season {season}, matchday {matchday}")

        data = self._make_request(endpoint)

        if data:
            logger.success(f"Retrieved {len(data)} matches")
            return data

        return []

    def get_league_table(self, season: str) -> List[Dict]:
        """
        Get league table/standings for a season

        Args:
            season: Season year

        Returns:
            List of standings dictionaries
        """
        endpoint = f"getbltable/{self.LEAGUE_CODE}/{season}"
        logger.info(f"Fetching league table for season {season}")

        data = self._make_request(endpoint)

        if data:
            logger.success(f"Retrieved standings for {len(data)} teams")
            return data

        return []

    def parse_match_data(self, match: Dict, season_str: str) -> Dict:
        """
        Parse match data from OpenLigaDB format to our format

        Args:
            match: Raw match data from API
            season_str: Season string (e.g., "2024-2025")

        Returns:
            Parsed match dictionary
        """
        match_id = match.get('matchID')
        group = match.get('group')
        matchday = group.get('groupOrderID', 0) if group else 0

        # Parse datetime
        match_datetime_str = match.get('matchDateTime') or match.get('matchDateTimeUTC')
        if not match_datetime_str:
            logger.warning(f"Match {match_id} has no datetime, skipping")
            return None

        try:
            match_datetime = datetime.fromisoformat(match_datetime_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError) as e:
            logger.warning(f"Could not parse datetime for match {match_id}: {e}")
            return None

        # Teams
        team1 = match.get('team1') or {}
        team2 = match.get('team2') or {}

        home_team_name = team1.get('teamName') or team1.get('shortName') or ''
        away_team_name = team2.get('teamName') or team2.get('shortName') or ''

        if not home_team_name or not away_team_name:
            logger.warning(f"Match {match_id} missing team names, skipping")
            return None

        home_team_openligadb_id = team1.get('teamId')
        away_team_openligadb_id = team2.get('teamId')

        # Get or create team IDs
        home_team_id = self.db.get_or_create_team(
            team_name=home_team_name,
            openligadb_id=home_team_openligadb_id
        )
        away_team_id = self.db.get_or_create_team(
            team_name=away_team_name,
            openligadb_id=away_team_openligadb_id
        )

        # Match results
        results = match.get('matchResults', [])
        final_result = None
        home_goals = None
        away_goals = None

        # Find final result
        for result in results:
            if result.get('resultTypeID') == 2:  # Final result
                home_goals = result.get('pointsTeam1')
                away_goals = result.get('pointsTeam2')
                break

        # Determine result
        result_code = None
        if home_goals is not None and away_goals is not None:
            if home_goals > away_goals:
                result_code = 'H'
            elif away_goals > home_goals:
                result_code = 'A'
            else:
                result_code = 'D'

        # Match status
        is_finished = match.get('matchIsFinished', False)

        # Venue
        location = match.get('location') or {}
        venue = location.get('locationStadium') or location.get('locationCity') or None

        parsed = {
            'openligadb_match_id': match_id,
            'season': season_str,
            'matchday': matchday,
            'match_datetime': match_datetime,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'result': result_code,
            'is_finished': is_finished,
            'venue': venue if venue else None,
            'raw_data': match  # Store raw data for goals parsing
        }

        return parsed

    def parse_league_table(self, standings_data: List[Dict], season_str: str) -> List[Dict]:
        """
        Parse league table data

        Args:
            standings_data: Raw standings data from API
            season_str: Season string

        Returns:
            List of parsed standings dictionaries
        """
        parsed_standings = []

        for idx, entry in enumerate(standings_data, start=1):
            team_name = entry.get('teamName', entry.get('shortName', ''))
            openligadb_id = entry.get('teamInfoId')

            team_id = self.db.get_or_create_team(
                team_name=team_name,
                openligadb_id=openligadb_id
            )

            # Get position - use index if rank is not available
            position = entry.get('rank')
            if position is None:
                position = idx

            standing = {
                'season': season_str,
                'matchday': entry.get('matchday', 34),  # Final matchday usually
                'team_id': team_id,
                'position': position,
                'matches_played': entry.get('matches', 0),
                'wins': entry.get('won', 0),
                'draws': entry.get('draw', 0),
                'losses': entry.get('lost', 0),
                'goals_for': entry.get('goals', 0),
                'goals_against': entry.get('opponentGoals', 0),
                'goal_difference': entry.get('goalDiff', 0),
                'points': entry.get('points', 0)
            }

            parsed_standings.append(standing)

        return parsed_standings

    def parse_match_events(self, match: Dict, match_db_id: int) -> List[Dict]:
        """
        Parse match events (goals, cards) from match data

        Args:
            match: Raw match data
            match_db_id: Database match ID

        Returns:
            List of event dictionaries
        """
        events = []
        goals = match.get('goals', [])

        for goal in goals:
            # Determine team
            team_name = goal.get('scoreTeam1') > goal.get('scoreTeam2', 0)  # Simplified

            event = {
                'match_id': match_db_id,
                'team_id': None,  # Would need to resolve from match teams
                'event_type': 'goal',
                'minute': goal.get('matchMinute'),
                'player_name': goal.get('goalGetterName'),
                'is_penalty': goal.get('isPenalty', False),
                'is_own_goal': goal.get('isOwnGoal', False)
            }

            if goal.get('goalAssistName'):
                event['assist_player_name'] = goal.get('goalAssistName')

            events.append(event)

        return events

    def insert_matches_to_db(self, matches: List[Dict]) -> int:
        """
        Insert matches into database

        Args:
            matches: List of parsed match dictionaries

        Returns:
            Number of matches inserted/updated
        """
        inserted = 0

        for match in matches:
            try:
                # Check if match exists
                existing_id = self.db.get_match_id(
                    season=match['season'],
                    home_team_id=match['home_team_id'],
                    away_team_id=match['away_team_id'],
                    match_datetime=match['match_datetime']
                )

                if existing_id:
                    # Update existing match
                    query = """
                        UPDATE matches
                        SET home_goals = ?, away_goals = ?, result = ?,
                            is_finished = ?, venue = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE match_id = ?
                    """
                    conn = self.db.get_connection()
                    conn.execute(query, (
                        match['home_goals'], match['away_goals'], match['result'],
                        match['is_finished'], match['venue'], existing_id
                    ))
                    conn.commit()
                    conn.close()
                else:
                    # Insert new match
                    query = """
                        INSERT INTO matches
                        (openligadb_match_id, season, matchday, match_datetime,
                         home_team_id, away_team_id, home_goals, away_goals,
                         result, is_finished, venue)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    self.db.execute_insert(query, (
                        match['openligadb_match_id'], match['season'],
                        match['matchday'], match['match_datetime'],
                        match['home_team_id'], match['away_team_id'],
                        match['home_goals'], match['away_goals'],
                        match['result'], match['is_finished'], match['venue']
                    ))

                inserted += 1

            except Exception as e:
                logger.error(f"Error inserting match {match.get('openligadb_match_id')}: {e}")

        return inserted

    def insert_standings_to_db(self, standings: List[Dict]) -> int:
        """
        Insert league standings into database

        Args:
            standings: List of parsed standings dictionaries

        Returns:
            Number of standings inserted
        """
        inserted = 0

        for standing in standings:
            try:
                query = """
                    INSERT OR REPLACE INTO league_standings
                    (season, matchday, team_id, position, matches_played,
                     wins, draws, losses, goals_for, goals_against,
                     goal_difference, points)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                self.db.execute_insert(query, (
                    standing['season'], standing['matchday'], standing['team_id'],
                    standing['position'], standing['matches_played'],
                    standing['wins'], standing['draws'], standing['losses'],
                    standing['goals_for'], standing['goals_against'],
                    standing['goal_difference'], standing['points']
                ))
                inserted += 1

            except Exception as e:
                logger.error(f"Error inserting standing: {e}")

        return inserted

    def collect_season(self, season_year: str) -> Dict[str, int]:
        """
        Collect all data for a specific season

        Args:
            season_year: Season year (e.g., '2024')

        Returns:
            Dictionary with collection statistics
        """
        logger.info(f"=== Collecting season {season_year} ===")
        start_time = datetime.now()

        # Create season string (e.g., "2024-2025")
        season_str = f"{season_year}-{int(season_year) + 1}"

        stats = {
            'matches_collected': 0,
            'standings_collected': 0,
            'errors': 0
        }

        # Collect matches
        try:
            matches_raw = self.get_matchdata_for_season(season_year)
            matches_parsed = []
            for idx, m in enumerate(matches_raw):
                try:
                    parsed = self.parse_match_data(m, season_str)
                    if parsed:
                        matches_parsed.append(parsed)
                except Exception as parse_error:
                    logger.debug(f"Failed to parse match {idx}: {parse_error}")
                    continue

            if matches_parsed:
                stats['matches_collected'] = self.insert_matches_to_db(matches_parsed)
                logger.success(f"Inserted/updated {stats['matches_collected']} matches")
            else:
                logger.warning(f"No valid matches parsed for season {season_year}")
        except Exception as e:
            logger.error(f"Error collecting matches for season {season_year}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            stats['errors'] += 1

        # Collect league table
        try:
            standings_raw = self.get_league_table(season_year)
            if standings_raw:
                standings_parsed = self.parse_league_table(standings_raw, season_str)
                stats['standings_collected'] = self.insert_standings_to_db(standings_parsed)
                logger.success(f"Inserted {stats['standings_collected']} standings")
        except Exception as e:
            logger.error(f"Error collecting standings for season {season_year}: {e}")
            stats['errors'] += 1

        # Log collection
        duration = (datetime.now() - start_time).total_seconds()
        self.db.log_collection(
            source='openligadb',
            collection_type='full_season',
            season=season_str,
            status='success' if stats['errors'] == 0 else 'partial',
            records_collected=stats['matches_collected'] + stats['standings_collected'],
            started_at=start_time
        )

        logger.info(f"Season {season_year} collection completed in {duration:.1f}s")
        return stats

    def collect_all_historical_data(self, start_year: int = 2009, end_year: Optional[int] = None) -> None:
        """
        Collect all historical data from start_year to end_year

        Args:
            start_year: Start season year (default: 2009)
            end_year: End season year (default: current year)
        """
        if end_year is None:
            end_year = datetime.now().year

        logger.info(f"=== Starting historical data collection: {start_year}-{end_year} ===")

        total_stats = {
            'matches_collected': 0,
            'standings_collected': 0,
            'seasons_processed': 0,
            'errors': 0
        }

        for year in range(start_year, end_year + 1):
            try:
                season_stats = self.collect_season(str(year))
                total_stats['matches_collected'] += season_stats['matches_collected']
                total_stats['standings_collected'] += season_stats['standings_collected']
                total_stats['errors'] += season_stats['errors']
                total_stats['seasons_processed'] += 1

                # Rate limiting - be nice to the API
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error collecting season {year}: {e}")
                total_stats['errors'] += 1

        logger.success(f"""
=== Historical data collection complete ===
Seasons processed: {total_stats['seasons_processed']}
Matches collected: {total_stats['matches_collected']}
Standings collected: {total_stats['standings_collected']}
Errors: {total_stats['errors']}
        """)


def main():
    """Main execution function"""
    collector = OpenLigaDBCollector()

    # Initialize database first
    logger.info("Initializing database...")
    collector.db.initialize_schema()

    # Collect all historical data
    collector.collect_all_historical_data(start_year=2009, end_year=2024)

    # Get current season too
    current_year = datetime.now().year
    if datetime.now().month >= 7:  # Season starts around July
        collector.collect_season(str(current_year))

    # Print database stats
    stats = collector.db.get_database_stats()
    print("\n=== Database Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
