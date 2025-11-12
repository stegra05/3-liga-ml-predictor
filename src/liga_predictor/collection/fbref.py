"""
FBref Web Scraper for 3. Liga
Collects team statistics, player statistics, and match data
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime
from loguru import logger
import time
import re

from liga_predictor.database import get_db
from liga_predictor.utils.team_mapper import TeamMapper
from liga_predictor.utils.scraper import BaseScraper
from liga_predictor.models import Player, PlayerSeasonStats, FBrefCollectionLog


class FBrefCollector(BaseScraper):
    """Collects 3. Liga data from FBref"""

    BASE_URL = "https://fbref.com"
    COMPETITION_ID = "59"  # 3. Liga competition code

    # Available seasons on FBref (2018-2019 onwards)
    AVAILABLE_SEASONS = [
        "2018-2019", "2019-2020", "2020-2021", "2021-2022",
        "2022-2023", "2023-2024", "2024-2025", "2025-2026"
    ]

    def __init__(self, use_selenium: bool = False):
        """
        Initialize FBref collector

        Args:
            use_selenium: If True, use Selenium for browser automation (slower but bypasses bot detection)
        """
        # Initialize base scraper with per-request driver pattern (not persistent)
        super().__init__(use_selenium=use_selenium, persistent_driver=False)

        self.db = get_db()
        self.team_mapper = TeamMapper()

        if use_selenium:
            logger.info("FBref collector initialized with Selenium (browser automation)")
        else:
            logger.info("FBref collector initialized with requests (may be blocked)")
            logger.warning("If you encounter 403 errors, run with use_selenium=True")

    def _make_request(self, url: str, delay: float = 3.0) -> Optional[BeautifulSoup]:
        """Make HTTP request with error handling and rate limiting"""
        return super()._make_request(url, delay=delay, wait_for="table" if self.use_selenium else None)

    def _extract_tables_from_html(self, html_content: str) -> List[pd.DataFrame]:
        """
        Extract tables from HTML, handling FBref's commented tables

        Args:
            html_content: Raw HTML content as string

        Returns:
            List of pandas DataFrames
        """
        # FBref wraps many tables in HTML comments, need to uncomment them
        html_cleaned = html_content.replace('<!--', '').replace('-->', '')

        try:
            tables = pd.read_html(html_cleaned)
            return tables
        except ValueError:
            logger.warning("No tables found in HTML content")
            return []

    def get_season_url(self, season: str) -> str:
        """
        Get FBref URL for a specific season

        Args:
            season: Season string (e.g., "2023-2024")

        Returns:
            Full URL to season page
        """
        return f"{self.BASE_URL}/en/comps/{self.COMPETITION_ID}/{season}/{season}-3-Liga-Stats"

    def collect_season_standings(self, season: str) -> Dict[str, int]:
        """
        Collect league standings/table for a season

        Args:
            season: Season string (e.g., "2023-2024")

        Returns:
            Statistics dictionary
        """
        logger.info(f"=== Collecting standings for {season} ===")
        url = self.get_season_url(season)

        soup = self._make_request(url)
        if not soup:
            return {"teams_collected": 0, "error": "Failed to fetch page"}

        stats = {"teams_collected": 0, "errors": 0}

        try:
            # Extract tables from page
            tables = self._extract_tables_from_html(str(soup))

            if not tables:
                logger.warning(f"No tables found for {season}")
                return stats

            # First table is typically the league standings
            standings_df = tables[0]

            # FBref columns: Rk, Squad, MP, W, D, L, GF, GA, GD, Pts, Pts/MP
            logger.info(f"Found standings table with {len(standings_df)} rows")

            for _, row in standings_df.iterrows():
                try:
                    fbref_team_name = str(row.get('Squad', ''))
                    if not fbref_team_name or fbref_team_name == 'Squad':
                        continue

                    # Map FBref team name to database standard name
                    standard_team_name = self.team_mapper.get_standard_name_from_fbref(fbref_team_name)
                    team_id = self.team_mapper.get_team_id(standard_team_name)

                    if not team_id:
                        logger.warning(f"Could not find team ID for: {fbref_team_name} -> {standard_team_name}")
                        stats["errors"] += 1
                        continue

                    # Extract standing data
                    # Note: This is season-end data, not matchday-specific
                    # Store as final matchday (38) for now
                    standing_data = {
                        'position': int(row.get('Rk', 0)),
                        'matches_played': int(row.get('MP', 0)),
                        'wins': int(row.get('W', 0)),
                        'draws': int(row.get('D', 0)),
                        'losses': int(row.get('L', 0)),
                        'goals_for': int(row.get('GF', 0)),
                        'goals_against': int(row.get('GA', 0)),
                        'goal_difference': int(row.get('GD', 0)),
                        'points': int(row.get('Pts', 0))
                    }

                    # Insert to database
                    self.db.insert_league_standing(season, 38, team_id, standing_data)
                    logger.debug(f"Team {standard_team_name}: {standing_data['points']} pts, Pos {standing_data['position']}")
                    stats["teams_collected"] += 1

                except Exception as e:
                    logger.error(f"Error processing team row: {e}")
                    stats["errors"] += 1

            logger.success(f"Collected standings for {stats['teams_collected']} teams")

        except Exception as e:
            logger.error(f"Error collecting standings for {season}: {e}")
            stats["errors"] += 1

        return stats

    def collect_team_season_stats(self, season: str) -> Dict[str, int]:
        """
        Collect team-level season statistics

        Args:
            season: Season string (e.g., "2023-2024")

        Returns:
            Statistics dictionary
        """
        logger.info(f"=== Collecting team stats for {season} ===")
        url = self.get_season_url(season)

        soup = self._make_request(url)
        if not soup:
            return {"teams_collected": 0, "error": "Failed to fetch page"}

        stats = {"teams_collected": 0, "errors": 0}

        try:
            tables = self._extract_tables_from_html(str(soup))

            # Look for "Squad Standard Stats" table
            # FBref often has multi-level column headers
            for table in tables[1:6]:  # Check first few tables after standings
                # Check if this looks like a stats table
                if len(table.columns) > 10:  # Stats tables have many columns
                    # Handle multi-level columns if present
                    if isinstance(table.columns, pd.MultiIndex):
                        # Flatten multi-level columns
                        table.columns = ['_'.join(col).strip('_') for col in table.columns.values]

                    # Find the column that contains team names
                    team_col = None
                    for col in table.columns:
                        if 'Squad' in str(col) or 'Team' in str(col):
                            team_col = col
                            break

                    if team_col is None and len(table.columns) > 0:
                        # First column is usually team name
                        team_col = table.columns[0]

                    if team_col is None:
                        continue

                    logger.info(f"Found stats table with {len(table.columns)} columns, team column: {team_col}")

                    for idx, row in table.iterrows():
                        try:
                            fbref_team_name = str(row[team_col]).strip()

                            # Skip header rows and empty values
                            if not fbref_team_name or fbref_team_name in ['Squad', 'Team', 'nan']:
                                continue

                            # Skip if it's just a number (ranking)
                            if fbref_team_name.isdigit():
                                continue

                            standard_team_name = self.team_mapper.get_standard_name_from_fbref(fbref_team_name)
                            team_id = self.team_mapper.get_team_id(standard_team_name)

                            if not team_id:
                                logger.warning(f"Could not find team ID for: {fbref_team_name}")
                                stats["errors"] += 1
                                continue

                            # Extract available statistics
                            # Note: We're just counting for now, will add DB insertion next
                            logger.debug(f"Collected stats for {standard_team_name}")
                            stats["teams_collected"] += 1

                        except Exception as e:
                            logger.error(f"Error processing team stats row: {e}")
                            stats["errors"] += 1

                    break  # Found the right table, stop looking

            logger.success(f"Collected stats for {stats['teams_collected']} teams")

        except Exception as e:
            logger.error(f"Error collecting team stats for {season}: {e}")
            stats["errors"] += 1

        return stats

    def collect_player_season_stats(self, season: str) -> Dict[str, int]:
        """
        Collect player season statistics

        Args:
            season: Season string (e.g., "2023-2024")

        Returns:
            Statistics dictionary
        """
        logger.info(f"=== Collecting player stats for {season} ===")

        # Try the main stats page first (most likely to have player data)
        player_stats_url = f"{self.BASE_URL}/en/comps/{self.COMPETITION_ID}/{season}/stats/{season}-3-Liga-Stats"

        logger.info("Note: Player-level statistics may not be available for 3. Liga on FBref")
        logger.info("FBref typically only provides detailed player stats for top-tier leagues")

        soup = self._make_request(player_stats_url)
        if not soup:
            return {"players_collected": 0, "error": "Failed to fetch page"}

        stats = {"players_collected": 0, "errors": 0}

        try:
            tables = self._extract_tables_from_html(str(soup))

            if not tables:
                logger.warning(f"No player stats tables found for {season}")
                return stats

            # Look for player stats table (might not be first table)
            player_df = None
            for idx, table in enumerate(tables):
                if isinstance(table.columns, pd.MultiIndex):
                    # Flatten to check columns
                    flat_cols = ['_'.join(str(col).strip() for col in c if str(col) != 'nan').strip('_') for c in table.columns.values]
                else:
                    flat_cols = list(table.columns)

                # Check if this table has player names (look for Player column)
                has_player_col = any('Player' in str(col) and 'Nation' not in str(col) for col in flat_cols)
                has_squad_col = any('Squad' in str(col) for col in flat_cols)

                # Player stats tables have both Player and Squad columns
                if has_player_col and has_squad_col:
                    player_df = table
                    logger.info(f"Found player stats in table {idx}")
                    break

            if player_df is None:
                logger.warning(f"Could not find individual player statistics table for {season}")
                logger.info("This is expected for 3. Liga - FBref may not provide player-level data for this league")
                return stats

            # Handle multi-level columns if present
            if isinstance(player_df.columns, pd.MultiIndex):
                # Flatten multi-level columns
                player_df.columns = ['_'.join(str(col).strip() for col in c if str(col) != 'nan').strip('_') for c in player_df.columns.values]

            logger.info(f"Found player stats table with {len(player_df)} rows, {len(player_df.columns)} columns")
            logger.debug(f"Columns: {list(player_df.columns)[:10]}...")  # Show first 10 columns

            # Find the player name column
            player_col = None
            for col in player_df.columns:
                if 'Player' in str(col) and 'Nation' not in str(col):
                    player_col = col
                    break

            if player_col is None:
                logger.warning("Could not find Player column in table")
                return stats

            for _, row in player_df.iterrows():
                try:
                    player_name = str(row.get(player_col, '')).strip()

                    # Skip header rows and empty values
                    if not player_name or player_name in ['Player', 'nan'] or player_name.isdigit():
                        continue

                    # Extract team info - look for Squad column
                    squad_col = None
                    for col in player_df.columns:
                        if 'Squad' in str(col):
                            squad_col = col
                            break

                    if squad_col is None:
                        logger.debug(f"No Squad column found, skipping player {player_name}")
                        continue

                    fbref_team_name = str(row.get(squad_col, '')).strip()
                    if not fbref_team_name or fbref_team_name == 'nan':
                        continue

                    standard_team_name = self.team_mapper.get_standard_name_from_fbref(fbref_team_name)
                    team_id = self.team_mapper.get_team_id(standard_team_name)

                    if not team_id:
                        logger.warning(f"Could not find team for player {player_name}: {fbref_team_name}")
                        stats["errors"] += 1
                        continue

                    # Extract player stats - safely get columns
                    def safe_get(col_name, default=0):
                        """Safely get column value"""
                        for col in player_df.columns:
                            if col_name in str(col):
                                val = row.get(col, default)
                                try:
                                    return int(float(val)) if val not in [None, '', 'nan'] else default
                                except:
                                    return default
                        return default

                    player_stats_data = {
                        'full_name': player_name,
                        'team_id': team_id,
                        'season': season,
                        'position': str(row.get('Pos', '')).strip() if 'Pos' in player_df.columns else '',
                        'matches_played': safe_get('MP', 0),
                        'starts': safe_get('Starts', 0),
                        'minutes_played': safe_get('Min', 0),
                        'goals': safe_get('Gls', 0),
                        'assists': safe_get('Ast', 0),
                        'yellow_cards': safe_get('CrdY', 0),
                        'red_cards': safe_get('CrdR', 0)
                    }

                    # Only save if they have some playing time
                    if player_stats_data['matches_played'] > 0 or player_stats_data['minutes_played'] > 0:
                        # Get or create player, then upsert season stats
                        nationality = str(row.get('Nation', '')).strip().replace('flag', '').strip() if 'Nation' in player_df.columns else None
                        player = self.db.merge_or_create(
                            Player,
                            filter_dict={'full_name': player_name},
                            defaults={'nationality': nationality, 'position': player_stats_data['position']}
                        )
                        player_id = player.player_id

                        # Upsert season stats
                        self.db.merge_or_create(
                            PlayerSeasonStats,
                            filter_dict={'player_id': player_id, 'team_id': team_id, 'season': season},
                            defaults={**player_stats_data, 'source': 'fbref'}
                        )

                        logger.debug(f"Player {player_name} ({fbref_team_name}): {player_stats_data['goals']}G, {player_stats_data['assists']}A in {player_stats_data['matches_played']} matches")
                        stats["players_collected"] += 1

                except Exception as e:
                    logger.error(f"Error processing player row: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    stats["errors"] += 1

            logger.success(f"Collected stats for {stats['players_collected']} players")

        except Exception as e:
            logger.error(f"Error collecting player stats for {season}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            stats["errors"] += 1

        return stats

    def collect_season_data(self, season: str) -> Dict[str, Any]:
        """
        Collect all available data for a season

        Args:
            season: Season string (e.g., "2023-2024")

        Returns:
            Combined statistics dictionary
        """
        if season not in self.AVAILABLE_SEASONS:
            logger.warning(f"Season {season} not in available seasons list")
            logger.info(f"Available: {self.AVAILABLE_SEASONS}")

        logger.info(f"\n{'='*60}")
        logger.info(f"COLLECTING DATA FOR SEASON: {season}")
        logger.info(f"{'='*60}\n")

        start_time = datetime.now()
        combined_stats = {
            'season': season,
            'started_at': start_time,
            'standings': {},
            'team_stats': {},
            'player_stats': {}
        }

        # 1. Collect standings
        logger.info("Step 1/3: Collecting league standings...")
        combined_stats['standings'] = self.collect_season_standings(season)

        # 2. Collect team stats
        logger.info("\nStep 2/3: Collecting team statistics...")
        combined_stats['team_stats'] = self.collect_team_season_stats(season)

        # 3. Collect player stats
        logger.info("\nStep 3/3: Collecting player statistics...")
        combined_stats['player_stats'] = self.collect_player_season_stats(season)

        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        combined_stats['completed_at'] = end_time
        combined_stats['duration_seconds'] = duration

        logger.info(f"\n{'='*60}")
        logger.success(f"SEASON {season} COLLECTION COMPLETE")
        logger.info(f"Duration: {duration:.1f} seconds")
        logger.info(f"Teams: {combined_stats['standings'].get('teams_collected', 0)}")
        logger.info(f"Players: {combined_stats['player_stats'].get('players_collected', 0)}")
        logger.info(f"{'='*60}\n")

        # Log to database
        self._log_collection(season, combined_stats)

        return combined_stats

    def _log_collection(self, season: str, stats: Dict) -> None:
        """
        Log collection run to database

        Args:
            season: Season collected
            stats: Statistics dictionary
        """
        try:
            log_data = {
                'season': season,
                'collection_type': 'full_season',
                'items_attempted': (
                    stats.get('standings', {}).get('teams_collected', 0) +
                    stats.get('player_stats', {}).get('players_collected', 0)
                ),
                'items_collected': (
                    stats.get('standings', {}).get('teams_collected', 0) +
                    stats.get('player_stats', {}).get('players_collected', 0)
                ),
                'items_failed': (
                    stats.get('standings', {}).get('errors', 0) +
                    stats.get('player_stats', {}).get('errors', 0)
                ),
                'started_at': stats.get('started_at'),
                'completed_at': stats.get('completed_at'),
                'duration_seconds': int(stats.get('duration_seconds', 0)),
                'status': 'success' if stats.get('standings', {}).get('teams_collected', 0) > 0 else 'failed',
                'teams_processed': stats.get('standings', {}).get('teams_collected', 0),
                'players_processed': stats.get('player_stats', {}).get('players_collected', 0)
            }

            session = self.db.get_session()
            try:
                log_entry = FBrefCollectionLog(
                    season=log_data['season'],
                    collection_type=log_data['collection_type'],
                    items_attempted=log_data['items_attempted'],
                    items_collected=log_data['items_collected'],
                    items_failed=log_data['items_failed'],
                    started_at=log_data['started_at'],
                    completed_at=log_data['completed_at'],
                    duration_seconds=log_data['duration_seconds'],
                    status=log_data['status'],
                    teams_processed=log_data['teams_processed'],
                    players_processed=log_data['players_processed']
                )
                session.add(log_entry)
                session.commit()
            finally:
                session.close()
            logger.debug("Collection logged to database")

        except Exception as e:
            logger.error(f"Failed to log collection: {e}")

    def collect_all_seasons(self, seasons: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Collect data for multiple seasons

        Args:
            seasons: List of seasons to collect, or None for all available

        Returns:
            Summary statistics
        """
        if seasons is None:
            seasons = self.AVAILABLE_SEASONS

        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING MULTI-SEASON COLLECTION")
        logger.info(f"Seasons to collect: {len(seasons)}")
        logger.info(f"{'='*60}\n")

        overall_start = datetime.now()
        results = {}

        for i, season in enumerate(seasons, 1):
            logger.info(f"\n[{i}/{len(seasons)}] Processing season: {season}")
            try:
                results[season] = self.collect_season_data(season)
            except Exception as e:
                logger.error(f"Failed to collect season {season}: {e}")
                results[season] = {'error': str(e)}

        overall_duration = (datetime.now() - overall_start).total_seconds()

        # Final summary
        total_teams = sum(r.get('standings', {}).get('teams_collected', 0) for r in results.values() if 'error' not in r)
        total_players = sum(r.get('player_stats', {}).get('players_collected', 0) for r in results.values() if 'error' not in r)
        successful_seasons = sum(1 for r in results.values() if 'error' not in r)

        logger.info(f"\n{'='*60}")
        logger.success(f"MULTI-SEASON COLLECTION COMPLETE")
        logger.info(f"Total duration: {overall_duration/60:.1f} minutes")
        logger.info(f"Successful seasons: {successful_seasons}/{len(seasons)}")
        logger.info(f"Total teams processed: {total_teams}")
        logger.info(f"Total players processed: {total_players}")
        logger.info(f"{'='*60}\n")

        return {
            'results': results,
            'summary': {
                'total_seasons': len(seasons),
                'successful_seasons': successful_seasons,
                'total_teams': total_teams,
                'total_players': total_players,
                'total_duration_seconds': overall_duration
            }
        }


if __name__ == "__main__":
    print("Use CLI instead: liga-predictor collect-fbref")
    import sys
    sys.exit(1)
