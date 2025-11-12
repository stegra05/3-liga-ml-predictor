"""
FotMob Web Scraper for 3. Liga
Collects match statistics from FotMob
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from loguru import logger
import time
import sys
from pathlib import Path
import re
import json

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from database.db_manager import get_db
from scripts.utils.team_mapper import TeamMapper


class FotMobCollector:
    """Collects 3. Liga match statistics from FotMob"""

    BASE_URL = "https://www.fotmob.com"
    LEAGUE_ID = "208"  # 3. Liga league code on FotMob

    def __init__(self, use_selenium: bool = True):
        """
        Initialize FotMob collector

        Args:
            use_selenium: If True, use Selenium for browser automation (recommended for FotMob)
        """
        self.use_selenium = use_selenium
        self.session = requests.Session()

        # Headers to mimic real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

        self.db = get_db()
        self.team_mapper = TeamMapper()
        self.driver = None  # Will be initialized if using Selenium

        logger.info("FotMob collector initialized")
        if use_selenium:
            logger.info("Using Selenium for browser automation")

    def __del__(self):
        """Clean up Selenium driver on deletion"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass

    def _init_selenium_driver(self):
        """Initialize Selenium driver with stealth options"""
        if self.driver is not None:
            return

        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager

            logger.debug("Initializing Selenium WebDriver")

            chrome_options = Options()
            chrome_options.add_argument('--headless=new')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument(f'user-agent={self.session.headers["User-Agent"]}')

            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.success("Selenium WebDriver initialized")

        except ImportError:
            logger.error("Selenium not installed. Install with: pip install selenium webdriver-manager")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Selenium: {e}")
            raise

    def _make_request_selenium(self, url: str, wait_for_selector: str = None) -> Optional[BeautifulSoup]:
        """
        Make request using Selenium

        Args:
            url: URL to fetch
            wait_for_selector: Optional CSS selector to wait for

        Returns:
            BeautifulSoup object or None on error
        """
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            if self.driver is None:
                self._init_selenium_driver()

            logger.debug(f"Fetching: {url}")
            self.driver.get(url)

            # Wait for content to load
            if wait_for_selector:
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_selector))
                )
            else:
                time.sleep(3)  # Default wait for page load

            page_source = self.driver.page_source
            return BeautifulSoup(page_source, 'html.parser')

        except Exception as e:
            logger.error(f"Selenium request failed for {url}: {e}")
            return None

    def _extract_json_from_script(self, soup: BeautifulSoup, script_id: str = "__NEXT_DATA__") -> Optional[Dict]:
        """
        Extract JSON data from Next.js script tag

        Args:
            soup: BeautifulSoup object
            script_id: ID of script tag containing JSON

        Returns:
            Parsed JSON data or None
        """
        try:
            script_tag = soup.find('script', {'id': script_id, 'type': 'application/json'})
            if script_tag and script_tag.string:
                return json.loads(script_tag.string)
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from script tag: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
            return None

    def get_match_statistics(self, fotmob_match_id: int, match_slug: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get match statistics from FotMob for a specific match

        Args:
            fotmob_match_id: FotMob match ID
            match_slug: Optional match slug (e.g., "energie-cottbus-vs-vfl-osnabruck/2qbovm")

        Returns:
            Dictionary with match statistics or None
        """
        # FotMob URLs can be either:
        # 1. /matches/{slug} - requires slug
        # 2. We'll construct a generic one if no slug provided
        if match_slug:
            url = f"{self.BASE_URL}/matches/{match_slug}"
        else:
            # Try to fetch via the match ID in a different format
            # This might not work, but let's try
            url = f"{self.BASE_URL}/match/{fotmob_match_id}"

        if not self.use_selenium:
            logger.warning("FotMob requires Selenium for reliable scraping")
            return None

        soup = self._make_request_selenium(url)
        if not soup:
            return None

        # Extract JSON data embedded in Next.js page
        data = self._extract_json_from_script(soup)
        if not data:
            logger.warning(f"No JSON data found for match {fotmob_match_id}")
            return None

        try:
            # Navigate the Next.js data structure
            page_props = data.get('props', {}).get('pageProps', {})

            if not page_props:
                logger.warning(f"No page props found for match {fotmob_match_id}")
                return None

            # Extract basic match info from header
            header = page_props.get('header', {})
            teams = header.get('teams', [])

            if len(teams) < 2:
                logger.warning(f"Invalid team data for match {fotmob_match_id}")
                return None

            home_team = teams[0]
            away_team = teams[1]

            status = header.get('status', {})

            # Extract statistics from content
            content = page_props.get('content', {}) or {}
            stats_data = content.get('stats', {}).get('Periods', {}).get('All', {}).get('stats', []) if content else []

            # Extract referee information
            referee = None
            # Check header for referee/officials
            if header:
                referee = header.get('referee') or header.get('refereeName')
                if not referee:
                    officials = header.get('officials') or header.get('matchOfficials')
                    if isinstance(officials, list) and officials:
                        for official in officials:
                            role = (official.get('role') or official.get('type') or '').lower()
                            if 'referee' in role or 'schiedsrichter' in role:
                                referee = official.get('name') or official.get('officialName')
                                break
            
            # Check content/matchDetails for referee
            if not referee and content:
                match_details = content.get('matchDetails') or content.get('details') or {}
                referee = match_details.get('referee') or match_details.get('refereeName')
                if not referee:
                    officials = match_details.get('officials') or match_details.get('matchOfficials')
                    if isinstance(officials, list) and officials:
                        for official in officials:
                            role = (official.get('role') or official.get('type') or '').lower()
                            if 'referee' in role or 'schiedsrichter' in role:
                                referee = official.get('name') or official.get('officialName')
                                break

            result = {
                'fotmob_match_id': fotmob_match_id,
                'home_team': home_team.get('name', ''),
                'home_team_id': home_team.get('id'),
                'away_team': away_team.get('name', ''),
                'away_team_id': away_team.get('id'),
                'home_score': home_team.get('score'),
                'away_score': away_team.get('score'),
                'finished': status.get('finished', False),
                'match_date': status.get('utcTime'),
                'referee': referee,
                'has_stats': False,
                'statistics': {}
            }

            # Parse statistics
            if stats_data:
                result['has_stats'] = True
                result['statistics'] = self._parse_statistics(stats_data)

            return result

        except Exception as e:
            logger.error(f"Error parsing match data for {fotmob_match_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _parse_statistics(self, stats_data: List[Dict]) -> Dict[str, Dict]:
        """
        Parse FotMob statistics into structured format

        Args:
            stats_data: List of stat category dictionaries from FotMob

        Returns:
            Dictionary with home and away statistics
        """
        home_stats = {}
        away_stats = {}

        # Mapping of FotMob stat titles to database columns
        stat_mapping = {
            'Ball possession': ('possession', float),
            'Total shots': ('shots_total', int),
            'Shots on target': ('shots_on_target', int),
            'Shots off target': ('shots_off_target', int),
            'Blocked shots': ('shots_blocked', int),
            'Big chances': ('big_chances', int),
            'Big chances missed': ('big_chances_missed', int),
            'Passes': ('passes_total', int),
            'Accurate passes': ('passes_accurate', int, 'with_percent'),
            'Key passes': ('key_passes', int),
            'Crosses': ('crosses_total', int),
            'Accurate crosses': ('crosses_accurate', int, 'with_percent_in_parens'),
            'Long balls': ('long_balls_total', int),
            'Accurate long balls': ('long_balls_accurate', int, 'with_percent_in_parens'),
            'Touches in opposition box': ('touches_opp_box', int),
            'Corners': ('corners', int),
            'Tackles': ('tackles_total', int),
            'Tackles won': ('tackles_won', int),
            'Interceptions': ('interceptions', int),
            'Clearances': ('clearances', int),
            'Goalkeeper saves': ('keeper_saves', int),
            'Duels won': ('duels_won', int),
            'Aerial duels won': ('aerials_won', int, 'with_percent_in_parens'),
            'Fouls committed': ('fouls_committed', int),
            'Fouls won': ('fouls_won', int),
            'Yellow cards': ('yellow_cards', int),
            'Red cards': ('red_cards', int),
            'Offsides': ('offsides', int),
            'Touches': ('touches', int),
            'Dribbles attempted': ('dribbles_attempted', int),
            'Successful dribbles': ('dribbles_successful', int, 'with_percent_in_parens'),
        }

        # FotMob stats are nested in categories like "Top stats", "Attack", "Defence", etc.
        for category in stats_data:
            category_stats = category.get('stats', [])

            for stat in category_stats:
                stat_title = stat.get('title', '')
                stat_values = stat.get('stats', [])

                # FotMob provides stats as [home_value, away_value]
                if len(stat_values) >= 2:
                    home_value = stat_values[0]
                    away_value = stat_values[1]

                    if stat_title in stat_mapping:
                        mapping_info = stat_mapping[stat_title]
                        db_key = mapping_info[0]
                        value_type = mapping_info[1]
                        has_special_format = len(mapping_info) > 2

                        try:
                            # Handle special formats
                            if has_special_format:
                                if mapping_info[2] == 'with_percent':
                                    # Format like "437 (83%)"
                                    if isinstance(home_value, str) and '(' in home_value:
                                        home_value = home_value.split('(')[0].strip()
                                        # Extract percentage for pass_accuracy_percent
                                        home_percent = stat_values[0].split('(')[1].replace(')', '').replace('%', '').strip()
                                        home_stats['pass_accuracy_percent'] = float(home_percent)

                                    if isinstance(away_value, str) and '(' in away_value:
                                        away_value = away_value.split('(')[0].strip()
                                        away_percent = stat_values[1].split('(')[1].replace(')', '').replace('%', '').strip()
                                        away_stats['pass_accuracy_percent'] = float(away_percent)

                                elif mapping_info[2] == 'with_percent_in_parens':
                                    # Format like "23 (49)" or "23 (49%)" - just extract the number
                                    if isinstance(home_value, str) and '(' in home_value:
                                        home_value = home_value.split('(')[0].strip()

                                    if isinstance(away_value, str) and '(' in away_value:
                                        away_value = away_value.split('(')[0].strip()

                            # Convert values
                            if home_value is not None and home_value != '':
                                if isinstance(home_value, str):
                                    home_value = home_value.replace('%', '').strip()
                                home_stats[db_key] = value_type(home_value)

                            if away_value is not None and away_value != '':
                                if isinstance(away_value, str):
                                    away_value = away_value.replace('%', '').strip()
                                away_stats[db_key] = value_type(away_value)

                        except (ValueError, TypeError) as e:
                            logger.debug(f"Failed to convert {stat_title} (home={home_value}, away={away_value}): {e}")
                            continue

        return {'home': home_stats, 'away': away_stats}

    def _matchday_has_all_stats(self, season: str, matchday: int) -> bool:
        """
        Check if a matchday already has statistics for all matches
        
        Args:
            season: Season string (e.g., "2024-2025")
            matchday: Matchday number
            
        Returns:
            True if all matches in matchday have statistics
        """
        # Count total matches for this matchday
        total_query = """
            SELECT COUNT(*) as total
            FROM matches
            WHERE season = ? AND matchday = ?
        """
        total_result = self.db.execute_query(total_query, (season, matchday))
        if not total_result or total_result[0]['total'] == 0:
            return False
        
        total_matches = total_result[0]['total']
        
        # Count matches with statistics
        stats_query = """
            SELECT COUNT(DISTINCT ms.match_id) as with_stats
            FROM match_statistics ms
            JOIN matches m ON ms.match_id = m.match_id
            WHERE m.season = ? AND m.matchday = ? AND ms.source = 'fotmob'
        """
        stats_result = self.db.execute_query(stats_query, (season, matchday))
        if not stats_result:
            return False
        
        matches_with_stats = stats_result[0]['with_stats']
        return matches_with_stats >= total_matches

    def collect_matchday(self, season: str, matchday: int, update_existing: bool = False) -> Dict[str, int]:
        """
        Collect statistics for a specific matchday

        Args:
            season: Season in format "2024-2025"
            matchday: Matchday number
            update_existing: If True, update matches that already have statistics

        Returns:
            Dictionary with collection statistics
        """
        start_time = datetime.now()
        stats = {'collected': 0, 'skipped': 0, 'errors': 0, 'not_found': 0}

        logger.info(f"Collecting FotMob statistics for {season} matchday {matchday}")

        # Check if matchday already has all statistics (unless updating)
        if not update_existing and self._matchday_has_all_stats(season, matchday):
            logger.info(f"Matchday {matchday} already has statistics for all matches, skipping")
            return stats

        # Get matches from database for this season/matchday
        query = """
            SELECT m.match_id, m.season, m.matchday, m.match_datetime,
                   m.home_goals, m.away_goals, m.fotmob_match_id,
                   ht.team_name as home_team, ht.team_id as home_team_id,
                   at.team_name as away_team, at.team_id as away_team_id
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.team_id
            JOIN teams at ON m.away_team_id = at.team_id
            WHERE m.season = ? AND m.matchday = ?
            ORDER BY m.match_datetime
        """

        matches = self.db.execute_query(query, (season, matchday))

        if not matches:
            logger.warning(f"No matches found for {season} matchday {matchday}")
            return stats

        logger.info(f"Found {len(matches)} matches to process")

        for match in matches:
            match_id = match['match_id']
            home_team = match['home_team']
            away_team = match['away_team']

            logger.info(f"Processing: {home_team} vs {away_team}")

            # Check if statistics already exist
            existing_stats = self.db.execute_query(
                "SELECT COUNT(*) as count FROM match_statistics WHERE match_id = ? AND source = 'fotmob'",
                (match_id,)
            )

            if existing_stats and existing_stats[0]['count'] > 0 and not update_existing:
                logger.debug(f"Statistics already exist for match {match_id}, skipping")
                stats['skipped'] += 1
                continue

            # Try to find FotMob match ID
            fotmob_match_id = match['fotmob_match_id']

            if not fotmob_match_id:
                # Try to find match on FotMob by team names and date
                fotmob_match_id = self._find_fotmob_match_id(
                    home_team, away_team, match['match_datetime'], season, matchday
                )

                if fotmob_match_id:
                    # Store FotMob match ID for future use
                    self.db.execute_insert(
                        "UPDATE matches SET fotmob_match_id = ? WHERE match_id = ?",
                        (fotmob_match_id, match_id)
                    )
                    logger.info(f"Found FotMob match ID: {fotmob_match_id}")
                else:
                    logger.warning(f"Could not find FotMob match ID for {home_team} vs {away_team}")
                    stats['not_found'] += 1
                    continue

            # Get match statistics from FotMob
            match_data = self.get_match_statistics(fotmob_match_id)

            if not match_data or not match_data.get('has_stats'):
                logger.warning(f"No statistics available for match {fotmob_match_id}")
                stats['errors'] += 1
                continue

            # Insert statistics into database
            try:
                self._insert_match_statistics(match_id, match, match_data['statistics'])
                stats['collected'] += 1
                logger.success(f"Collected statistics for {home_team} vs {away_team}")
            except Exception as e:
                logger.error(f"Failed to insert statistics: {e}")
                stats['errors'] += 1

            # Rate limiting
            time.sleep(2)

        # Log collection
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = None if stats['collected'] > 0 else f"MD {matchday}: {stats['collected']} collected, {stats['skipped']} skipped, {stats['errors']} errors, {stats['not_found']} not found"
        self.db.log_collection(
            source='fotmob',
            collection_type='match_statistics',
            season=season,
            matchday=matchday,
            status='success' if stats['collected'] > 0 else 'partial',
            records_collected=stats['collected'],
            started_at=start_time,
            error_message=error_msg
        )

        logger.info(f"Collection complete: {stats}")
        return stats

    def _find_fotmob_match_id(self, home_team: str, away_team: str, match_date: str,
                              season: str, matchday: int) -> Optional[int]:
        """
        Find FotMob match ID by searching the league matches page

        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Match datetime (ISO format)
            season: Season
            matchday: Matchday

        Returns:
            FotMob match ID or None
        """
        try:
            from datetime import datetime as dt
            from difflib import SequenceMatcher

            # Parse match date to get the date range for the round
            match_dt = dt.fromisoformat(match_date.replace('Z', '+00:00')) if match_date else None

            # Fetch league matches page
            # FotMob groups matches by round, so we need to parse the matches page
            url = f"{self.BASE_URL}/leagues/{self.LEAGUE_ID}/matches/3-liga?group=by-round"

            logger.debug(f"Fetching league matches page to find match ID for {home_team} vs {away_team}")
            soup = self._make_request_selenium(url)

            if not soup:
                logger.warning("Could not fetch league matches page")
                return None

            # Extract JSON data
            data = self._extract_json_from_script(soup)
            if not data:
                logger.warning("No JSON data found on league matches page")
                return None

            # Navigate to matches data
            page_props = data.get('props', {}).get('pageProps', {})
            matches_data = page_props.get('matches', {})

            # Get all matches
            all_matches = matches_data.get('allMatches', [])

            if not all_matches:
                logger.warning("No matches data found in league matches page")
                return None

            # Filter matches by round/matchday
            round_matches = [m for m in all_matches if str(m.get('round')) == str(matchday)]

            if not round_matches:
                logger.debug(f"Could not find any matches for round {matchday}, trying all matches")
                # Fall back to searching all matches
                round_matches = all_matches

            # Search for our match in this round
            return self._match_teams_in_list(round_matches, home_team, away_team, match_dt)

        except Exception as e:
            logger.error(f"Error finding FotMob match ID: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _match_teams_in_list(self, matches: List[Dict], home_team: str, away_team: str,
                             match_dt: Optional[Any]) -> Optional[int]:
        """
        Find a match in a list by matching team names

        Args:
            matches: List of match dictionaries from FotMob
            home_team: Home team name from database
            away_team: Away team name from database
            match_dt: Optional datetime object for date verification

        Returns:
            FotMob match ID or None
        """
        from difflib import SequenceMatcher
        from datetime import datetime as dt, timedelta

        best_match = None
        best_score = 0.0

        for match in matches:
            # Extract team names from match
            fotmob_home = match.get('home', {}).get('name', '')
            fotmob_away = match.get('away', {}).get('name', '')
            match_id = match.get('id')

            if not fotmob_home or not fotmob_away or not match_id:
                continue

            # Calculate similarity scores
            home_similarity = SequenceMatcher(None, home_team.lower(), fotmob_home.lower()).ratio()
            away_similarity = SequenceMatcher(None, away_team.lower(), fotmob_away.lower()).ratio()

            # Also try with normalized names (remove common prefixes)
            home_normalized = self.team_mapper._normalize(home_team)
            away_normalized = self.team_mapper._normalize(away_team)
            fotmob_home_normalized = self.team_mapper._normalize(fotmob_home)
            fotmob_away_normalized = self.team_mapper._normalize(fotmob_away)

            home_similarity_norm = SequenceMatcher(None, home_normalized, fotmob_home_normalized).ratio()
            away_similarity_norm = SequenceMatcher(None, away_normalized, fotmob_away_normalized).ratio()

            # Use the better score
            home_score = max(home_similarity, home_similarity_norm)
            away_score = max(away_similarity, away_similarity_norm)

            # Average similarity
            avg_score = (home_score + away_score) / 2

            # Verify date if provided (should be within 24 hours)
            date_match = True
            if match_dt:
                match_time_str = match.get('status', {}).get('utcTime')
                if match_time_str:
                    try:
                        fotmob_dt = dt.fromisoformat(match_time_str.replace('Z', '+00:00'))
                        time_diff = abs((fotmob_dt - match_dt).total_seconds())
                        # Allow up to 24 hours difference (for timezone issues)
                        if time_diff > 86400:
                            date_match = False
                    except:
                        pass

            # Require high similarity (>0.7) for both teams
            if avg_score > best_score and home_score > 0.7 and away_score > 0.7 and date_match:
                best_score = avg_score
                best_match = match_id
                logger.debug(f"Match candidate: {fotmob_home} vs {fotmob_away} (ID: {match_id}, similarity: {avg_score:.2f})")

        if best_match and best_score > 0.75:
            logger.info(f"Found match ID {best_match} with {best_score:.2%} similarity")
            return best_match

        return None

    def _insert_match_statistics(self, match_id: int, match: Dict, statistics: Dict[str, Dict]) -> None:
        """
        Insert match statistics into database

        Args:
            match_id: Database match ID
            match: Match data from database
            statistics: Statistics dictionary with 'home' and 'away' keys
        """
        home_stats = statistics.get('home', {})
        away_stats = statistics.get('away', {})

        # Insert home team statistics
        self._insert_team_statistics(
            match_id=match_id,
            team_id=match['home_team_id'],
            is_home=True,
            stats=home_stats
        )

        # Insert away team statistics
        self._insert_team_statistics(
            match_id=match_id,
            team_id=match['away_team_id'],
            is_home=False,
            stats=away_stats
        )

    def _insert_team_statistics(self, match_id: int, team_id: int, is_home: bool, stats: Dict) -> None:
        """
        Insert statistics for one team in a match

        Args:
            match_id: Match ID
            team_id: Team ID
            is_home: True if home team
            stats: Statistics dictionary
        """
        # Build INSERT query with all available fields
        query = """
            INSERT OR REPLACE INTO match_statistics (
                match_id, team_id, is_home,
                possession_percent,
                shots_total, shots_on_target, shots_off_target, shots_blocked,
                big_chances, big_chances_missed,
                passes_total, passes_accurate, pass_accuracy_percent,
                key_passes, crosses_total, crosses_accurate,
                long_balls_total, long_balls_accurate,
                corners, offsides,
                tackles_total, tackles_won, interceptions, clearances,
                duels_total, duels_won, aerials_total, aerials_won,
                fouls_committed, fouls_won,
                yellow_cards, red_cards,
                touches, dribbles_attempted, dribbles_successful,
                source, has_complete_stats, updated_at
            ) VALUES (
                ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?,
                'fotmob', ?, CURRENT_TIMESTAMP
            )
        """

        # Calculate total duels and aerials if we have won counts
        duels_total = stats.get('duels_won')  # FotMob only provides won duels
        aerials_total = stats.get('aerials_won')  # FotMob only provides won aerials

        params = (
            match_id, team_id, is_home,
            stats.get('possession'),
            stats.get('shots_total'), stats.get('shots_on_target'),
            stats.get('shots_off_target'), stats.get('shots_blocked'),
            stats.get('big_chances'), stats.get('big_chances_missed'),
            stats.get('passes_total'), stats.get('passes_accurate'),
            stats.get('pass_accuracy_percent'),
            stats.get('key_passes'), stats.get('crosses_total'),
            stats.get('crosses_accurate'),
            stats.get('long_balls_total'), stats.get('long_balls_accurate'),
            stats.get('corners'), stats.get('offsides'),
            stats.get('tackles_total'), stats.get('tackles_won'),
            stats.get('interceptions'), stats.get('clearances'),
            duels_total, stats.get('duels_won'),
            aerials_total, stats.get('aerials_won'),
            stats.get('fouls_committed'), stats.get('fouls_won'),
            stats.get('yellow_cards'), stats.get('red_cards'),
            stats.get('touches'), stats.get('dribbles_attempted'),
            stats.get('dribbles_successful'),
            1 if len(stats) >= 10 else 0  # has_complete_stats
        )

        self.db.execute_insert(query, params)

    def collect_season(self, season: str, start_matchday: int = 1,
                       end_matchday: int = 38, update_existing: bool = False) -> Dict[str, int]:
        """
        Collect statistics for an entire season

        Args:
            season: Season in format "2024-2025"
            start_matchday: First matchday to collect
            end_matchday: Last matchday to collect
            update_existing: If True, update matches that already have statistics

        Returns:
            Dictionary with total collection statistics
        """
        total_stats = {'collected': 0, 'skipped': 0, 'errors': 0, 'not_found': 0, 'matchdays_skipped': 0}

        logger.info(f"Collecting FotMob statistics for season {season}, matchdays {start_matchday}-{end_matchday}")

        for matchday in range(start_matchday, end_matchday + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Matchday {matchday}/{end_matchday}")
            logger.info(f"{'='*60}")

            # Check if matchday already has all stats (collect_matchday will also check, but we can log it here)
            if not update_existing and self._matchday_has_all_stats(season, matchday):
                logger.info(f"Matchday {matchday} already has all statistics, skipping")
                total_stats['matchdays_skipped'] += 1
                continue

            stats = self.collect_matchday(season, matchday, update_existing)

            total_stats['collected'] += stats['collected']
            total_stats['skipped'] += stats['skipped']
            total_stats['errors'] += stats['errors']
            total_stats['not_found'] += stats['not_found']

        logger.success(f"\nSeason collection complete!")
        logger.info(f"Total stats: {total_stats}")
        if total_stats.get('matchdays_skipped', 0) > 0:
            logger.info(f"Skipped {total_stats['matchdays_skipped']} matchdays that already had complete statistics")

        return total_stats

    def collect_referees_for_matchday(self, season: str, matchday: int, skip_existing: bool = True) -> Dict[str, int]:
        """
        Collect referee data for a matchday using FotMob
        
        Args:
            season: Season in format "2024-2025"
            matchday: Matchday number
            skip_existing: If True, skip matches that already have referee data
            
        Returns:
            Dictionary with collection statistics
        """
        stats = {
            'matches_processed': 0,
            'referees_found': 0,
            'referees_updated': 0,
            'matches_skipped': 0,
            'errors': 0
        }
        
        logger.info(f"Collecting referees from FotMob for {season} MD {matchday}")
        
        # Get matches for this matchday that have fotmob_match_id
        query = """
            SELECT m.match_id, m.fotmob_match_id, m.referee,
                   ht.team_name as home_team, at.team_name as away_team
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.team_id
            JOIN teams at ON m.away_team_id = at.team_id
            WHERE m.season = ? AND m.matchday = ? AND m.fotmob_match_id IS NOT NULL
            ORDER BY m.match_datetime
        """
        
        matches = self.db.execute_query(query, (season, matchday))
        
        if not matches:
            logger.warning(f"No matches with FotMob IDs found for {season} MD {matchday}")
            return stats
        
        logger.info(f"Found {len(matches)} matches with FotMob IDs")
        
        for match in matches:
            match_id = match['match_id']
            fotmob_match_id = match['fotmob_match_id']
            existing_referee = match['referee']
            
            # Skip if already has referee and skip_existing is True
            if skip_existing and existing_referee:
                stats['matches_skipped'] += 1
                continue
            
            stats['matches_processed'] += 1
            
            try:
                # Get match data from FotMob
                match_data = self.get_match_statistics(fotmob_match_id)
                
                if match_data and match_data.get('referee'):
                    referee = match_data['referee']
                    # Update match with referee
                    update_query = """
                        UPDATE matches
                        SET referee = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE match_id = ?
                    """
                    self.db.execute_insert(update_query, (referee, match_id))
                    stats['referees_found'] += 1
                    stats['referees_updated'] += 1
                    logger.debug(f"Updated referee for {match['home_team']} vs {match['away_team']}: {referee}")
                else:
                    logger.debug(f"No referee data found for {match['home_team']} vs {match['away_team']}")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.debug(f"Error collecting referee for match {match_id}: {e}")
                stats['errors'] += 1
        
        logger.info(f"Referee collection complete: {stats}")
        return stats

    def identify_gaps(self, season: Optional[str] = None, min_season: str = "2014-2015") -> List[Tuple[str, int]]:
        """
        Identify matches missing FotMob statistics

        Args:
            season: Optional season to check (if None, checks all seasons)
            min_season: Minimum season to check (FotMob data availability starts around 2014-2015)

        Returns:
            List of (season, matchday) tuples with missing statistics
        """
        query = """
            SELECT DISTINCT m.season, m.matchday
            FROM matches m
            LEFT JOIN match_statistics ms ON m.match_id = ms.match_id AND ms.source = 'fotmob'
            WHERE m.is_finished = 1
            AND ms.stat_id IS NULL
            AND m.season >= ?
        """

        params = [min_season]
        if season:
            query += " AND m.season = ?"
            params.append(season)

        query += " ORDER BY m.season DESC, m.matchday DESC"  # Start with most recent

        results = self.db.execute_query(query, tuple(params))
        gaps = [(r['season'], r['matchday']) for r in results]

        logger.info(f"Found {len(gaps)} matchdays with missing statistics (from {min_season} onwards)")
        return gaps

    def fill_gaps(self, limit: Optional[int] = None, season: Optional[str] = None) -> Dict[str, int]:
        """
        Automatically fill missing statistics

        Args:
            limit: Maximum number of matchdays to fill (None = all)
            season: Optional season filter

        Returns:
            Dictionary with collection statistics
        """
        gaps = self.identify_gaps(season)

        if not gaps:
            logger.info("No gaps found!")
            return {'collected': 0, 'skipped': 0, 'errors': 0, 'not_found': 0}

        if limit:
            gaps = gaps[:limit]
            logger.info(f"Filling first {limit} gaps")

        total_stats = {'collected': 0, 'skipped': 0, 'errors': 0, 'not_found': 0}

        for season_str, matchday in gaps:
            logger.info(f"\nFilling gap: {season_str} matchday {matchday}")
            stats = self.collect_matchday(season_str, matchday, update_existing=False)

            total_stats['collected'] += stats['collected']
            total_stats['skipped'] += stats['skipped']
            total_stats['errors'] += stats['errors']
            total_stats['not_found'] += stats['not_found']

        logger.success(f"\nGap filling complete: {total_stats}")
        return total_stats


def main():
    """Main CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='FotMob Statistics Collector for 3. Liga')
    parser.add_argument('--season', type=str, help='Season (e.g., 2024-2025)')
    parser.add_argument('--matchday', type=int, help='Specific matchday to collect')
    parser.add_argument('--start-matchday', type=int, default=1, help='Start matchday for season collection')
    parser.add_argument('--end-matchday', type=int, default=38, help='End matchday for season collection')
    parser.add_argument('--fill-gaps', action='store_true', help='Fill missing statistics')
    parser.add_argument('--identify-gaps', action='store_true', help='Identify missing statistics')
    parser.add_argument('--limit', type=int, help='Limit number of matchdays to process')
    parser.add_argument('--update', action='store_true', help='Update existing statistics')
    parser.add_argument('--no-selenium', action='store_true', help='Disable Selenium (not recommended)')

    args = parser.parse_args()

    # Initialize collector
    collector = FotMobCollector(use_selenium=not args.no_selenium)

    try:
        if args.identify_gaps:
            # Identify gaps
            gaps = collector.identify_gaps(season=args.season)
            print(f"\nFound {len(gaps)} matchdays with missing statistics:")
            for season, matchday in gaps:
                print(f"  {season} matchday {matchday}")

        elif args.fill_gaps:
            # Fill gaps
            collector.fill_gaps(limit=args.limit, season=args.season)

        elif args.matchday and args.season:
            # Collect specific matchday
            collector.collect_matchday(args.season, args.matchday, update_existing=args.update)

        elif args.season:
            # Collect entire season
            collector.collect_season(
                args.season,
                start_matchday=args.start_matchday,
                end_matchday=args.end_matchday,
                update_existing=args.update
            )

        else:
            parser.print_help()
            print("\nExamples:")
            print("  python scripts/collectors/fotmob_collector.py --season 2024-2025 --matchday 15")
            print("  python scripts/collectors/fotmob_collector.py --season 2024-2025")
            print("  python scripts/collectors/fotmob_collector.py --fill-gaps --limit 10")
            print("  python scripts/collectors/fotmob_collector.py --identify-gaps")

    finally:
        # Clean up
        if collector.driver:
            collector.driver.quit()


if __name__ == "__main__":
    main()
