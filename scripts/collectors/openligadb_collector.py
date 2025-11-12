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

        # Find final result (different seasons use different resultTypeIDs/order)
        final = None
        # 1) Prefer explicit 'Endergebnis'
        for r in results or []:
            if str(r.get('resultName', '')).lower().startswith('endergebnis'):
                final = r
                break
        # 2) Prefer typeID==2 (common in newer seasons)
        if final is None:
            for r in results or []:
                if r.get('resultTypeID') == 2:
                    final = r
                    break
        # 3) Fallback: smallest resultOrderID (often full-time)
        if final is None and results:
            final = sorted(results, key=lambda x: (x.get('resultOrderID', 99), x.get('resultID', 1e9)))[0]

        if final:
            home_goals = final.get('pointsTeam1')
            away_goals = final.get('pointsTeam2')

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

        # Venue, Attendance, Referee (best-effort extraction)
        location = match.get('location') or {}
        venue = location.get('locationStadium') or location.get('locationCity') or None
        # Attendance candidates observed in some feeds/APIs; OpenLigaDB often doesn't provide it
        attendance = None
        for att_key in ('numberOfViewers', 'attendance', 'spectators'):
            val = match.get(att_key)
            if val is not None:
                try:
                    attendance = int(val)
                    break
                except Exception:
                    try:
                        # Sometimes nested or string with separators
                        attendance = int(str(val).replace('.', '').replace(',', ''))
                        break
                    except Exception:
                        pass

        # Referee (if present, structure varies; try common shapes)
        referee = None
        # 1) Flat key
        if match.get('referee'):
            referee = match.get('referee')
        # 2) Officials list with names
        if referee is None:
            for officials_key in ('matchOfficials', 'officials', 'referees'):
                officials = match.get(officials_key)
                if isinstance(officials, list) and officials:
                    # Prefer a head referee if labeled, else first with a name
                    head = None
                    for o in officials:
                        name = o.get('officialName') or o.get('name') or o.get('officialShortName')
                        role = (o.get('officialTypeName') or o.get('role') or '').lower()
                        if name and ('schiedsrichter' in role or 'referee' in role or 'head' in role):
                            head = name
                            break
                    if not head:
                        # Fallback to first available name
                        for o in officials:
                            name = o.get('officialName') or o.get('name') or o.get('officialShortName')
                            if name:
                                head = name
                                break
                    referee = head
                    if referee:
                        break

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
            'attendance': attendance,
            'referee': referee,
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

    def parse_match_events(self, match: Dict, match_db_id: int, home_team_id: int = None, away_team_id: int = None) -> List[Dict]:
        """
        Parse match events (goals, cards, substitutions) from match data

        Args:
            match: Raw match data
            match_db_id: Database match ID
            home_team_id: Home team ID (for resolving which team scored)
            away_team_id: Away team ID (for resolving which team scored)

        Returns:
            List of event dictionaries
        """
        events = []
        goals = match.get('goals', [])

        # Get team IDs from match if not provided
        if home_team_id is None or away_team_id is None:
            team1 = match.get('team1') or {}
            team2 = match.get('team2') or {}
            team1_id = team1.get('teamId')
            team2_id = team2.get('teamId')
            
            if team1_id and team2_id:
                # Get team IDs from database
                team1_name = team1.get('teamName') or team1.get('shortName', '')
                team2_name = team2.get('teamName') or team2.get('shortName', '')
                
                if team1_name:
                    t1_db_id = self.db.get_or_create_team(team_name=team1_name, openligadb_id=team1_id)
                    if home_team_id is None:
                        home_team_id = t1_db_id
                if team2_name:
                    t2_db_id = self.db.get_or_create_team(team_name=team2_name, openligadb_id=team2_id)
                    if away_team_id is None:
                        away_team_id = t2_db_id

        for goal in goals:
            # Determine which team scored based on score progression
            score_team1 = goal.get('scoreTeam1', 0)
            score_team2 = goal.get('scoreTeam2', 0)
            
            # Determine team_id: if scoreTeam1 increased, it's home team; if scoreTeam2 increased, it's away team
            # We compare with previous goal or initial score
            team_id = None
            if score_team1 > score_team2:
                team_id = home_team_id
            elif score_team2 > score_team1:
                team_id = away_team_id
            else:
                # Equal scores - try to infer from goal getter (would need team lookup)
                # For now, use home team as fallback
                team_id = home_team_id

            event = {
                'match_id': match_db_id,
                'team_id': team_id,
                'event_type': 'goal',
                'minute': goal.get('matchMinute'),
                'minute_extra': goal.get('matchMinuteExtra'),
                'player_id': None,  # Would need player lookup
                'player_name': goal.get('goalGetterName'),
                'is_penalty': goal.get('isPenalty', False),
                'is_own_goal': goal.get('isOwnGoal', False),
                'assist_player_id': None,
                'assist_player_name': goal.get('goalAssistName'),
                'player_out_id': None,
                'player_out_name': None
            }

            events.append(event)

        # Parse cards and substitutions if available in match data
        # OpenLigaDB may have these in different structures
        # For now, we focus on goals which are most reliably available

        return events

    def insert_match_events(self, events: List[Dict]) -> int:
        """
        Insert match events into database with idempotency

        Args:
            events: List of event dictionaries from parse_match_events

        Returns:
            Number of events inserted
        """
        if not events:
            return 0

        inserted = 0
        conn = self.db.get_connection()

        try:
            for event in events:
                # Use INSERT OR IGNORE for idempotency (based on match_id + minute + player_name)
                query = """
                    INSERT OR IGNORE INTO match_events
                    (match_id, team_id, event_type, minute, minute_extra, player_id, player_name,
                     is_penalty, is_own_goal, assist_player_id, assist_player_name, 
                     player_out_id, player_out_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                try:
                    conn.execute(query, (
                        event['match_id'],
                        event.get('team_id'),
                        event['event_type'],
                        event.get('minute'),
                        event.get('minute_extra'),
                        event.get('player_id'),
                        event.get('player_name'),
                        event.get('is_penalty', False),
                        event.get('is_own_goal', False),
                        event.get('assist_player_id'),
                        event.get('assist_player_name'),
                        event.get('player_out_id'),
                        event.get('player_out_name')
                    ))
                    inserted += 1
                except Exception as e:
                    logger.debug(f"Error inserting event: {e}")
                    continue

            conn.commit()
        finally:
            conn.close()

        return inserted

    def insert_matches_to_db(self, matches: List[Dict], raw_matches: List[Dict] = None) -> int:
        """
        Insert matches into database using openligadb_match_id as primary key
        Optionally also insert match events if raw_matches provided

        Args:
            matches: List of parsed match dictionaries
            raw_matches: Optional list of raw match data from API (for event parsing)

        Returns:
            Number of matches inserted/updated
        """
        inserted = 0
        match_id_map = {}  # Map openligadb_match_id -> db match_id

        for match in matches:
            try:
                openligadb_match_id = match.get('openligadb_match_id')
                if not openligadb_match_id:
                    logger.warning(f"Match missing openligadb_match_id, skipping")
                    continue

                # Check if match exists by openligadb_match_id (most reliable)
                query_check = "SELECT match_id FROM matches WHERE openligadb_match_id = ? LIMIT 1"
                existing = self.db.execute_query(query_check, (openligadb_match_id,))

                if existing:
                    # Update existing match (can update all fields including matchday/season if corrected)
                    existing_id = existing[0]['match_id']
                    match_id_map[openligadb_match_id] = existing_id
                    query = """
                        UPDATE matches
                        SET season = ?, matchday = ?, match_datetime = ?,
                            home_team_id = ?, away_team_id = ?,
                            home_goals = ?, away_goals = ?, result = ?,
                            is_finished = ?, venue = ?, attendance = ?, referee = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE match_id = ?
                    """
                    conn = self.db.get_connection()
                    conn.execute(query, (
                        match['season'], match['matchday'], match['match_datetime'],
                        match['home_team_id'], match['away_team_id'],
                        match['home_goals'], match['away_goals'], match['result'],
                        match['is_finished'], match['venue'], match.get('attendance'), match.get('referee'), existing_id
                    ))
                    conn.commit()
                    conn.close()
                else:
                    # Insert new match
                    query = """
                        INSERT INTO matches
                        (openligadb_match_id, season, matchday, match_datetime,
                         home_team_id, away_team_id, home_goals, away_goals,
                         result, is_finished, venue, attendance, referee)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    conn = self.db.get_connection()
                    cursor = conn.cursor()
                    cursor.execute(query, (
                        match['openligadb_match_id'], match['season'],
                        match['matchday'], match['match_datetime'],
                        match['home_team_id'], match['away_team_id'],
                        match['home_goals'], match['away_goals'],
                        match['result'], match['is_finished'], match['venue'],
                        match.get('attendance'), match.get('referee')
                    ))
                    match_id_map[openligadb_match_id] = cursor.lastrowid
                    conn.commit()
                    conn.close()

                inserted += 1

            except Exception as e:
                logger.error(f"Error inserting match {match.get('openligadb_match_id')}: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        # Insert match events if raw_matches provided
        if raw_matches and match_id_map:
            all_events = []
            for raw_match, parsed_match in zip(raw_matches, matches):
                openligadb_id = parsed_match.get('openligadb_match_id')
                db_match_id = match_id_map.get(openligadb_id)
                if db_match_id:
                    events = self.parse_match_events(
                        raw_match, 
                        db_match_id,
                        home_team_id=parsed_match.get('home_team_id'),
                        away_team_id=parsed_match.get('away_team_id')
                    )
                    all_events.extend(events)
            
            if all_events:
                events_inserted = self.insert_match_events(all_events)
                logger.debug(f"Inserted {events_inserted} match events")

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

    def collect_matchday_range_from_full_season(self, season_year: str, target_matchdays: List[int]) -> Dict[str, int]:
        """
        Collect specific matchdays by fetching full season and filtering locally
        Useful when individual matchday endpoints don't return data

        Args:
            season_year: Season year (e.g., '2022')
            target_matchdays: List of matchday numbers to collect

        Returns:
            Dictionary with collection statistics
        """
        logger.info(f"=== Collecting matchdays {target_matchdays} for season {season_year} (via full season fetch) ===")
        start_time = datetime.now()

        season_str = f"{season_year}-{int(season_year) + 1}"

        stats = {
            'matches_collected': 0,
            'matchdays_found': 0,
            'errors': 0
        }

        # Fetch full season
        logger.info(f"Fetching full season data for {season_year}...")
        matches_raw = self.get_matchdata_for_season(season_year)

        if not matches_raw:
            logger.warning(f"No matches found for season {season_year}")
            return stats

        # Filter by target matchdays
        target_set = set(target_matchdays)
        filtered_matches = []
        found_matchdays = set()

        for m in matches_raw:
            group = m.get('group')
            matchday = group.get('groupOrderID', 0) if group else 0
            if matchday in target_set:
                filtered_matches.append(m)
                found_matchdays.add(matchday)

        logger.info(f"Found {len(filtered_matches)} matches across matchdays {sorted(found_matchdays)}")

        if not filtered_matches:
            logger.warning(f"No matches found for target matchdays {target_matchdays}")
            return stats

        # Parse and insert filtered matches
        matches_parsed = []
        for idx, m in enumerate(filtered_matches):
            try:
                parsed = self.parse_match_data(m, season_str)
                if parsed:
                    matches_parsed.append(parsed)
            except Exception as parse_error:
                logger.debug(f"Failed to parse match {idx}: {parse_error}")
                continue

        if matches_parsed:
            stats['matches_collected'] = self.insert_matches_to_db(matches_parsed)
            stats['matchdays_found'] = len(found_matchdays)
            logger.success(f"Inserted/updated {stats['matches_collected']} matches from {len(found_matchdays)} matchdays")
        else:
            logger.warning(f"No valid matches parsed")

        duration = (datetime.now() - start_time).total_seconds()
        logger.success(f"""
=== Full season fetch collection complete ===
Season: {season_str}
Target matchdays: {target_matchdays}
Matchdays found: {sorted(found_matchdays)}
Matches collected: {stats['matches_collected']}
Duration: {duration:.1f}s
        """)
        return stats

    def _matchday_has_all_matches(self, season: str, matchday: int, expected_matches: int = 10) -> bool:
        """
        Check if a matchday already has all expected matches in the database
        
        Args:
            season: Season string (e.g., "2022-2023")
            matchday: Matchday number
            expected_matches: Expected number of matches per matchday (default: 10 for 3. Liga)
            
        Returns:
            True if matchday has all expected matches
        """
        query = """
            SELECT COUNT(*) as count
            FROM matches
            WHERE season = ? AND matchday = ?
        """
        result = self.db.execute_query(query, (season, matchday))
        if result and result[0]['count'] >= expected_matches:
            return True
        return False

    def collect_matchday_range(self, season_year: str, start_matchday: int, end_matchday: int) -> Dict[str, int]:
        """
        Collect matches for a specific range of matchdays within a season

        Args:
            season_year: Season year (e.g., '2022')
            start_matchday: First matchday to collect (inclusive)
            end_matchday: Last matchday to collect (inclusive)

        Returns:
            Dictionary with collection statistics
        """
        logger.info(f"=== Collecting matchdays {start_matchday}-{end_matchday} for season {season_year} ===")
        start_time = datetime.now()

        # Create season string (e.g., "2022-2023")
        season_str = f"{season_year}-{int(season_year) + 1}"

        stats = {
            'matches_collected': 0,
            'matchdays_processed': 0,
            'matchdays_skipped': 0,
            'errors': 0
        }

        # Collect matches for each matchday in range
        for matchday in range(start_matchday, end_matchday + 1):
            try:
                # Check if matchday already has all matches
                if self._matchday_has_all_matches(season_str, matchday):
                    logger.info(f"Matchday {matchday} already has all matches, skipping API call")
                    stats['matchdays_skipped'] += 1
                    continue
                
                logger.info(f"Collecting matchday {matchday}...")
                matches_raw = self.get_matchdata_for_matchday(season_year, matchday)
                
                if not matches_raw:
                    logger.warning(f"No matches found for matchday {matchday}")
                    continue

                matches_parsed = []
                for idx, m in enumerate(matches_raw):
                    try:
                        parsed = self.parse_match_data(m, season_str)
                        if parsed:
                            matches_parsed.append(parsed)
                    except Exception as parse_error:
                        logger.debug(f"Failed to parse match {idx} on matchday {matchday}: {parse_error}")
                        continue

                if matches_parsed:
                    inserted = self.insert_matches_to_db(matches_parsed)
                    stats['matches_collected'] += inserted
                    stats['matchdays_processed'] += 1
                    logger.success(f"Matchday {matchday}: Inserted/updated {inserted} matches")
                else:
                    logger.warning(f"Matchday {matchday}: No valid matches parsed")

                # Rate limiting - be nice to the API
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error collecting matchday {matchday} for season {season_year}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                stats['errors'] += 1

        # Log collection
        duration = (datetime.now() - start_time).total_seconds()
        self.db.log_collection(
            source='openligadb',
            collection_type='matchday_range',
            season=season_str,
            matchday=start_matchday,
            status='success' if stats['errors'] == 0 else 'partial',
            records_collected=stats['matches_collected'],
            started_at=start_time
        )

        logger.success(f"""
=== Matchday range collection complete ===
Season: {season_str}
Matchdays: {start_matchday}-{end_matchday}
Matchdays processed: {stats['matchdays_processed']}
Matchdays skipped: {stats['matchdays_skipped']}
Matches collected: {stats['matches_collected']}
Errors: {stats['errors']}
Duration: {duration:.1f}s
        """)
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
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenLigaDB Collector for 3. Liga",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect full season
  python openligadb_collector.py --season 2022

  # Collect specific matchday range
  python openligadb_collector.py --season 2022 --start-matchday 22 --end-matchday 38

  # Collect all historical data
  python openligadb_collector.py --all-historical

  # Use season string format
  python openligadb_collector.py --season-str 2022-2023 --start-matchday 22 --end-matchday 38
        """
    )
    parser.add_argument(
        '--season',
        type=str,
        help='Season year (e.g., "2022" for 2022-2023 season)'
    )
    parser.add_argument(
        '--season-str',
        type=str,
        help='Season string (e.g., "2022-2023"). If provided, extracts year automatically.'
    )
    parser.add_argument(
        '--start-matchday',
        type=int,
        help='First matchday to collect (for range collection)'
    )
    parser.add_argument(
        '--end-matchday',
        type=int,
        help='Last matchday to collect (for range collection)'
    )
    parser.add_argument(
        '--full-season',
        action='store_true',
        help='Collect full season (all matchdays)'
    )
    parser.add_argument(
        '--all-historical',
        action='store_true',
        help='Collect all historical data from 2009 to current year'
    )
    parser.add_argument(
        '--init-db',
        action='store_true',
        default=True,
        help='Initialize database schema (default: True)'
    )
    parser.add_argument(
        '--use-full-season-fetch',
        action='store_true',
        help='Fetch full season and filter locally (useful when matchday endpoints fail)'
    )

    args = parser.parse_args()

    collector = OpenLigaDBCollector()

    # Initialize database if requested
    if args.init_db:
        logger.info("Initializing database...")
        collector.db.initialize_schema()

    # Determine season year from arguments
    season_year = None
    if args.season:
        season_year = args.season
    elif args.season_str:
        # Extract year from season string (e.g., "2022-2023" -> "2022")
        season_year = args.season_str.split('-')[0]

    # Execute based on arguments
    if args.all_historical:
        # Collect all historical data
        collector.collect_all_historical_data(start_year=2009, end_year=2024)
        
        # Get current season too
        current_year = datetime.now().year
        if datetime.now().month >= 7:  # Season starts around July
            collector.collect_season(str(current_year))
    
    elif season_year:
        if args.start_matchday is not None and args.end_matchday is not None:
            # Collect matchday range
            if args.use_full_season_fetch:
                # Use full season fetch method
                target_matchdays = list(range(args.start_matchday, args.end_matchday + 1))
                collector.collect_matchday_range_from_full_season(
                    season_year=season_year,
                    target_matchdays=target_matchdays
                )
            else:
                # Use individual matchday endpoints
                collector.collect_matchday_range(
                    season_year=season_year,
                    start_matchday=args.start_matchday,
                    end_matchday=args.end_matchday
                )
        elif args.full_season:
            # Collect full season
            collector.collect_season(season_year)
        else:
            # Default: collect full season if no range specified
            logger.info("No matchday range specified, collecting full season...")
            collector.collect_season(season_year)
    else:
        # Default behavior: collect all historical data
        logger.info("No arguments provided, collecting all historical data...")
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
    import sys
    print("This script is deprecated. Use: python main.py collect-openligadb [args]", file=sys.stderr)
    sys.exit(2)
