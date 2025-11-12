"""
OddsPortal Collector for 3. Liga Betting Odds
Scrapes historical and current betting odds from OddsPortal.com
"""

import sys
from pathlib import Path
import time
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import random
from loguru import logger

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from database.db_manager import get_db
from scripts.utils.team_mapper import TeamMapper


class OddsPortalCollector:
    """Collects betting odds from OddsPortal.com"""

    BASE_URL = "https://www.oddsportal.com"
    LEAGUE_PATH = "/football/germany/3-liga"

    # Season mapping for URL construction
    SEASONS = {
        "2009-2010": "2009-2010",
        "2010-2011": "2010-2011",
        "2011-2012": "2011-2012",
        "2012-2013": "2012-2013",
        "2013-2014": "2013-2014",
        "2014-2015": "2014-2015",
        "2015-2016": "2015-2016",
        "2016-2017": "2016-2017",
        "2017-2018": "2017-2018",
        "2018-2019": "2018-2019",
        "2019-2020": "2019-2020",
        "2020-2021": "2020-2021",
        "2021-2022": "2021-2022",
        "2022-2023": "2022-2023",
        "2023-2024": "2023-2024",
        "2024-2025": "2024-2025",
        "2025-2026": "2025-2026",
    }

    def __init__(self, use_selenium: bool = True):
        """Initialize collector with Selenium (recommended for JavaScript sites)"""
        self.use_selenium = use_selenium
        self.db = get_db()
        self.team_mapper = TeamMapper()
        self.driver = None
        logger.info("OddsPortal collector initialized")

    def _init_driver(self):
        """Initialize Selenium WebDriver with stealth settings"""
        if self.driver is not None:
            return self.driver

        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager

            chrome_options = Options()
            chrome_options.add_argument('--headless=new')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument(
                'user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )

            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("Selenium WebDriver initialized")
            return self.driver

        except Exception as e:
            logger.error(f"Failed to initialize Selenium WebDriver: {e}")
            raise

    def _close_driver(self):
        """Close Selenium WebDriver"""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                logger.debug("Selenium WebDriver closed")
            except Exception as e:
                logger.debug(f"Error closing driver: {e}")

    def _random_delay(self, min_seconds: float = 2.0, max_seconds: float = 5.0):
        """Random delay to avoid detection"""
        delay = random.uniform(min_seconds, max_seconds)
        time.sleep(delay)

    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team names from OddsPortal format"""
        team_name = team_name.strip()

        # Common OddsPortal variations
        replacements = {
            'VfL Osnabruck': 'VfL Osnabrueck',
            'Preussen Munster': 'SC Preussen Muenster',
            'Stutt. Kickers': 'Stuttgarter Kickers',
            'Stuttgarter K.': 'Stuttgarter Kickers',
            '1860 Munchen': 'TSV 1860 München',
            'Munchen 1860': 'TSV 1860 München',
            'RW Essen': 'Rot-Weiss Essen',
            'C. Zeiss Jena': 'FC Carl Zeiss Jena',
            'Waldhof Mannheim': 'SV Waldhof Mannheim',
            'Wehen': 'SV Wehen Wiesbaden',
            'Hansa': 'FC Hansa Rostock',
            'Viktoria Köln': 'Viktoria Köln',  # Keep as-is, will be normalized by TeamMapper
            'FC Viktoria Köln': 'Viktoria Köln',  # Remove FC prefix
            'Viktoria Koln': 'Viktoria Köln',  # Fix umlaut
        }

        for old, new in replacements.items():
            if old.lower() in team_name.lower():
                return new

        return team_name

    def _get_existing_odds_match_ids(self, season: Optional[str] = None) -> set:
        """
        Get match_ids that already have odds data

        Args:
            season: Optional season filter

        Returns:
            Set of match_ids with existing odds
        """
        if season:
            query = """
                SELECT DISTINCT bo.match_id
                FROM betting_odds bo
                JOIN matches m ON bo.match_id = m.match_id
                WHERE m.season = ?
                AND bo.bookmaker = 'oddsportal_avg'
            """
            result = self.db.execute_query(query, (season,))
        else:
            query = """
                SELECT DISTINCT match_id
                FROM betting_odds
                WHERE bookmaker = 'oddsportal_avg'
            """
            result = self.db.execute_query(query)
        
        if result:
            return {row['match_id'] for row in result}
        return set()
    
    def _season_has_complete_odds(self, season: str, min_coverage: float = 0.95) -> bool:
        """
        Check if a season already has complete odds coverage
        
        Args:
            season: Season string (e.g., "2023-2024")
            min_coverage: Minimum coverage ratio (0.0-1.0) to consider complete (default: 95%)
            
        Returns:
            True if season has sufficient odds coverage
        """
        # Count total matches in season
        total_query = """
            SELECT COUNT(*) as total
            FROM matches
            WHERE season = ?
        """
        total_result = self.db.execute_query(total_query, (season,))
        if not total_result or total_result[0]['total'] == 0:
            return False
        
        total_matches = total_result[0]['total']
        
        # Count matches with odds
        odds_query = """
            SELECT COUNT(DISTINCT bo.match_id) as with_odds
            FROM betting_odds bo
            JOIN matches m ON bo.match_id = m.match_id
            WHERE m.season = ?
            AND bo.bookmaker = 'oddsportal_avg'
        """
        odds_result = self.db.execute_query(odds_query, (season,))
        if not odds_result:
            return False
        
        matches_with_odds = odds_result[0]['with_odds']
        coverage = matches_with_odds / total_matches if total_matches > 0 else 0
        
        return coverage >= min_coverage

    def _find_match_id(self, home_team: str, away_team: str,
                       match_date: datetime, season: str) -> Optional[int]:
        """
        Find match_id in database by teams, date, and season

        Args:
            home_team: Normalized home team name
            away_team: Normalized away team name
            match_date: Match datetime
            season: Season string

        Returns:
            match_id or None
        """
        # Try to get team IDs
        home_team_id = self.team_mapper.get_team_id(home_team)
        away_team_id = self.team_mapper.get_team_id(away_team)

        if not home_team_id or not away_team_id:
            # Teams not found - skip silently
            return None

        # Search with date range (±2 days to handle date discrepancies)
        query = """
            SELECT match_id, match_datetime, home_goals, away_goals
            FROM matches
            WHERE home_team_id = ?
            AND away_team_id = ?
            AND season = ?
            AND date(match_datetime) BETWEEN date(?, '-2 days') AND date(?, '+2 days')
            ORDER BY ABS(julianday(match_datetime) - julianday(?))
            LIMIT 1
        """

        match_date_str = match_date.strftime('%Y-%m-%d %H:%M:%S')
        result = self.db.execute_query(
            query,
            (home_team_id, away_team_id, season,
             match_date_str, match_date_str, match_date_str)
        )

        if result:
            return result[0]['match_id']

        # Match not found - skip silently
        return None

    def _calculate_implied_probability(self, odds: float) -> Optional[float]:
        """Calculate implied probability from decimal odds"""
        if odds and odds > 0:
            return 1.0 / odds
        return None

    def _insert_or_update_odds(self, match_data: Dict) -> bool:
        """
        Insert or update betting odds for a match
        
        Args:
            match_data: Dictionary with match and odds data
                - If 'match_id' is present, use it directly
                - Otherwise, use 'home_team', 'away_team', 'match_datetime', 'season' to find match
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # If match_id is provided directly, use it (from collect_matchday_odds)
            if 'match_id' in match_data and match_data['match_id']:
                match_id = match_data['match_id']
            else:
                # Otherwise, find match by teams and date
                # Normalize team names
                home_team = self._normalize_team_name(match_data['home_team'])
                away_team = self._normalize_team_name(match_data['away_team'])

                # Use TeamMapper for further normalization
                # Try multiple variations to improve matching
                home_team_standard = self.team_mapper.get_standard_name(home_team)
                if not self.team_mapper.get_team_id(home_team_standard):
                    # Try without common prefixes
                    for prefix in ['FC ', 'SV ', 'TSV ', 'SC ', '1. FC ', '1. ']:
                        if home_team.startswith(prefix):
                            alt_name = home_team[len(prefix):].strip()
                            alt_standard = self.team_mapper.get_standard_name(alt_name)
                            if self.team_mapper.get_team_id(alt_standard):
                                home_team_standard = alt_standard
                                break
                
                away_team_standard = self.team_mapper.get_standard_name(away_team)
                if not self.team_mapper.get_team_id(away_team_standard):
                    # Try without common prefixes
                    for prefix in ['FC ', 'SV ', 'TSV ', 'SC ', '1. FC ', '1. ']:
                        if away_team.startswith(prefix):
                            alt_name = away_team[len(prefix):].strip()
                            alt_standard = self.team_mapper.get_standard_name(alt_name)
                            if self.team_mapper.get_team_id(alt_standard):
                                away_team_standard = alt_standard
                                break
                
                home_team = home_team_standard
                away_team = away_team_standard

                # Ensure match_datetime is a datetime object
                match_datetime = match_data.get('match_datetime')
                if isinstance(match_datetime, str):
                    import pandas as pd
                    match_datetime = pd.to_datetime(match_datetime).to_pydatetime()
                elif not isinstance(match_datetime, datetime):
                    import pandas as pd
                    match_datetime = pd.to_datetime(match_datetime).to_pydatetime()

                # Find match in database
                match_id = self._find_match_id(
                    home_team, away_team,
                    match_datetime,
                    match_data.get('season', '')
                )

                if not match_id:
                    logger.warning(
                        f"Match not found in database: {home_team} vs {away_team}, "
                        f"{match_data.get('match_datetime')}, {match_data.get('season')}"
                    )
                    return False

            # Calculate implied probabilities
            implied_home = self._calculate_implied_probability(match_data.get('odds_home'))
            implied_draw = self._calculate_implied_probability(match_data.get('odds_draw'))
            implied_away = self._calculate_implied_probability(match_data.get('odds_away'))

            # Check if odds already exist
            check_query = """
                SELECT odds_id FROM betting_odds
                WHERE match_id = ? AND bookmaker = 'oddsportal_avg'
            """
            existing = self.db.execute_query(check_query, (match_id,))

            if existing:
                # Update existing odds
                query = """
                    UPDATE betting_odds
                    SET odds_home = ?,
                        odds_draw = ?,
                        odds_away = ?,
                        implied_prob_home = ?,
                        implied_prob_draw = ?,
                        implied_prob_away = ?,
                        collected_at = CURRENT_TIMESTAMP
                    WHERE match_id = ? AND bookmaker = 'oddsportal_avg'
                """
                params = (
                    match_data.get('odds_home'),
                    match_data.get('odds_draw'),
                    match_data.get('odds_away'),
                    implied_home, implied_draw, implied_away,
                    match_id
                )
                # Updating existing odds
            else:
                # Insert new odds
                query = """
                    INSERT INTO betting_odds (
                        match_id, bookmaker, odds_home, odds_draw, odds_away,
                        implied_prob_home, implied_prob_draw, implied_prob_away,
                        odds_type, collected_at
                    ) VALUES (?, 'oddsportal_avg', ?, ?, ?, ?, ?, ?, 'closing', CURRENT_TIMESTAMP)
                """
                params = (
                    match_id,
                    match_data.get('odds_home'),
                    match_data.get('odds_draw'),
                    match_data.get('odds_away'),
                    implied_home, implied_draw, implied_away
                )
                # Inserting new odds

            conn = self.db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
                return True
            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Error inserting/updating odds: {e}")
            return False

    def _parse_match_row_selenium(self, row) -> Optional[Dict]:
        """
        Parse match data from Selenium WebElement row

        Args:
            row: Selenium WebElement representing a match row

        Returns:
            Dictionary with match data or None
        """
        from datetime import datetime  # Import at function level to avoid scoping issues
        try:
            from selenium.webdriver.common.by import By

            # Extract date from date header (some rows contain the header, others don't)
            date_header = row.find_elements(By.CSS_SELECTOR, '[data-testid="date-header"]')
            match_date = None
            has_date = False
            if date_header:
                date_text = date_header[0].text.strip()
                try:
                    # Format: "09 Nov 2025"
                    match_date = datetime.strptime(date_text, '%d %b %Y')
                    has_date = True
                except:
                    match_date = None
                    has_date = False

            # Extract time (rows usually have the time)
            time_element = row.find_elements(By.CSS_SELECTOR, '[data-testid="time-item"]')
            time_str = None
            has_time = False
            if time_element:
                time_str = time_element[0].text.strip()
                if time_str:
                    has_time = True

            # Combine date and time conservatively:
            # - If both present: combine
            # - If only date: return date at 00:00 (has_date=True, has_time=False)
            # - If only time: return today's date with that time (has_date=False, has_time=True) so caller can replace the date
            match_datetime = None
            if has_date and has_time:
                try:
                    hour, minute = map(int, time_str.split(':'))
                    match_datetime = match_date.replace(hour=hour, minute=minute)
                except:
                    match_datetime = match_date
                    has_time = False
            elif has_date:
                match_datetime = match_date.replace(hour=0, minute=0)
            elif has_time:
                try:
                    hour, minute = map(int, time_str.split(':'))
                    today = datetime.now().date()
                    match_datetime = datetime(today.year, today.month, today.day, hour, minute)
                except:
                    match_datetime = None

            # Try multiple methods to extract team names
            # Method 1: Look for data-testid="event-participants" or similar
            participants_elem = row.find_elements(By.CSS_SELECTOR, '[data-testid="event-participants"]')
            if participants_elem:
                participants_text = participants_elem[0].text.strip()
                # Format is usually "Team1 - Team2" or "Team1 vs Team2"
                import re
                # Try to split by common separators
                if ' - ' in participants_text:
                    teams = participants_text.split(' - ', 1)
                elif ' vs ' in participants_text:
                    teams = participants_text.split(' vs ', 1)
                elif ' v ' in participants_text:
                    teams = participants_text.split(' v ', 1)
                else:
                    # Try regex pattern
                    match = re.match(r'(.+?)\s*[-–]\s*(.+)', participants_text)
                    if match:
                        teams = [match.group(1).strip(), match.group(2).strip()]
                    else:
                        teams = None
                
                if teams and len(teams) == 2:
                    home_team = teams[0].strip()
                    away_team = teams[1].strip()
                else:
                    # Fallback to p tags method
                    all_p_tags = row.find_elements(By.TAG_NAME, 'p')
                    potential_teams = []
                    for p in all_p_tags:
                        text = p.text.strip()
                        if not text or text == '/' or text in ['Germany', '1', 'X', '2', "B's"]:
                            continue
                        if ':' in text and len(text) <= 5:
                            continue
                        try:
                            float(text)  # Skip odds values
                            continue
                        except ValueError:
                            pass
                        if len(text) > 2:
                            potential_teams.append(text)
                    
                    if len(potential_teams) < 2:
                        return None
                    home_team = potential_teams[0]
                    away_team = potential_teams[1]
            else:
                # Fallback: Extract team names from p tags
                all_p_tags = row.find_elements(By.TAG_NAME, 'p')
                potential_teams = []
                for p in all_p_tags:
                    text = p.text.strip()
                    if not text or text == '/' or text in ['Germany', '1', 'X', '2', "B's"]:
                        continue
                    if ':' in text and len(text) <= 5:
                        continue
                    try:
                        float(text)  # Skip odds values
                        continue
                    except ValueError:
                        pass
                    if len(text) > 2:
                        potential_teams.append(text)
                
                if len(potential_teams) < 2:
                    return None
                home_team = potential_teams[0]
                away_team = potential_teams[1]

            # Extract odds - look for odds values in the row
            odds_values = []
            # Try to find odds in various ways
            # Method 1: Look for data-testid="odds" or similar
            odds_elements = row.find_elements(By.CSS_SELECTOR, '[data-testid*="odds"], .odds, [class*="odds"]')
            for elem in odds_elements:
                try:
                    text = elem.text.strip()
                    odds_val = float(text)
                    if 1.01 <= odds_val <= 100:
                        odds_values.append(odds_val)
                except (ValueError, AttributeError):
                    pass
            
            # Method 2: Look for all text elements that are decimal numbers
            if len(odds_values) < 3:
                all_text_elements = row.find_elements(By.XPATH, './/*[text()]')
                for elem in all_text_elements:
                    try:
                        text = elem.text.strip()
                        if not text:
                            continue
                        odds_val = float(text)
                        if 1.01 <= odds_val <= 100 and odds_val not in odds_values:
                            odds_values.append(odds_val)
                            if len(odds_values) >= 3:
                                break
                    except (ValueError, AttributeError):
                        pass
            # Pick the most plausible 1X2 triple from all decimals in the row.
            # Heuristic: choose consecutive triple with realistic overround (≈1.02–1.20).
            odds_home, odds_draw, odds_away = None, None, None
            def evaluate_triplet(a: float, b: float, c: float) -> float:
                """
                Return a score for (home, draw, away) where lower is better.
                Factors:
                  - Overround closeness to 1.06 (typical ~4–8% margin)
                  - Plausible ranges, especially draw odds (rarely <2.0 or >6.0)
                """
                # Reject obvious garbage
                if any(x < 1.05 or x > 20.0 for x in (a, b, c)):
                    return float('inf')
                overround = (1.0 / a) + (1.0 / b) + (1.0 / c)
                # Penalize underround (<1.0) and extreme overround (>1.25)
                if overround < 0.98 or overround > 1.25:
                    return float('inf')
                score = abs(overround - 1.06) * 100.0  # weight overround
                # Strong preference for realistic draw range
                if not (2.0 <= b <= 6.0):
                    score += 50.0
                # Mild preference against extremely short or long home/away
                if a < 1.15 or a > 8.0:
                    score += 5.0
                if c < 1.15 or c > 8.0:
                    score += 5.0
                return score

            best_score = None
            best_triplet = None
            for i in range(max(0, len(odds_values) - 2)):
                a, b, c = odds_values[i], odds_values[i + 1], odds_values[i + 2]
                score = evaluate_triplet(a, b, c)
                if score == float('inf'):
                    continue
                if best_score is None or score < best_score:
                    best_score = score
                    best_triplet = (a, b, c)
            if best_triplet:
                odds_home, odds_draw, odds_away = best_triplet
            else:
                # Fallback: use the first three numbers if nothing passed the heuristic
                odds_home = odds_values[0] if len(odds_values) > 0 else None
                odds_draw = odds_values[1] if len(odds_values) > 1 else None
                odds_away = odds_values[2] if len(odds_values) > 2 else None

            # Extract score if available (look for pattern like "0-1" or "4-0")
            score = None
            participants_elem = row.find_elements(By.CSS_SELECTOR, '[data-testid="event-participants"]')
            if participants_elem:
                full_text = participants_elem[0].text
                # Look for score pattern: digits-digits
                import re
                score_match = re.search(r'(\d+)\s*[–-]\s*(\d+)', full_text)
                if score_match:
                    score = f"{score_match.group(1)}-{score_match.group(2)}"

            return {
                'home_team': home_team,
                'away_team': away_team,
                'match_datetime': match_datetime,
                'has_date': has_date,
                'has_time': has_time,
                'odds_home': odds_home,
                'odds_draw': odds_draw,
                'odds_away': odds_away,
                'score': score
            }

        except Exception as e:
            # Error parsing match row - skip silently
            return None

    def collect_upcoming_matches(self) -> Dict[str, int]:
        """
        Collect odds for upcoming matches

        Returns:
            Statistics dictionary
        """
        logger.info("=== Collecting upcoming matches ===")

        stats = {
            'matches_found': 0,
            'matches_inserted': 0,
            'matches_updated': 0,
            'matches_skipped': 0,
            'errors': 0
        }

        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            driver = self._init_driver()

            # Navigate to upcoming matches page
            url = f"{self.BASE_URL}{self.LEAGUE_PATH}/"
            logger.info(f"Navigating to: {url}")
            driver.get(url)

            # Wait for page to load
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            self._random_delay()

            # Attempt to load all dynamically rendered rows (infinite-scroll style)
            # Repeatedly scroll to bottom until the number of rows stops increasing
            last_count = -1
            stable_iters = 0
            for _ in range(12):  # up to ~12 scroll cycles
                match_rows = driver.find_elements(By.CSS_SELECTOR, '.eventRow, [data-testid="event-row"], div[class*="eventRow"]')
                count = len(match_rows)
                if count == last_count:
                    stable_iters += 1
                else:
                    stable_iters = 0
                if stable_iters >= 2:
                    break
                last_count = count
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.0)
            # Small extra delay to allow odds text to populate in rows
            time.sleep(1.0)

            # Final fetch of rows
            match_rows = driver.find_elements(By.CSS_SELECTOR, '.eventRow, [data-testid="event-row"], div[class*="eventRow"]')
            stats['matches_found'] = len(match_rows)
            logger.info(f"Found {len(match_rows)} upcoming matches")

            # Determine current season
            current_month = datetime.now().month
            current_year = datetime.now().year
            if current_month >= 7:  # Season starts in July/August
                season = f"{current_year}-{current_year + 1}"
            else:
                season = f"{current_year - 1}-{current_year}"

            # Track current date as we iterate (OddsPortal groups by date)
            current_date = None

            for row in match_rows:
                try:
                    match_data = self._parse_match_row_selenium(row)
                    if not match_data:
                        continue

                    # If this row has a date header, update current_date to that date
                    if match_data.get('has_date') and match_data.get('match_datetime'):
                        current_date = match_data['match_datetime'].date()
                    # If we have only time, attach it to the most recent current_date
                    if match_data.get('has_time') and not match_data.get('has_date') and current_date and match_data.get('match_datetime'):
                        mt = match_data['match_datetime']
                        match_data['match_datetime'] = datetime(
                            current_date.year, current_date.month, current_date.day, mt.hour, mt.minute
                        )

                    match_data['season'] = season

                    # Insert/update odds
                    if self._insert_or_update_odds(match_data):
                        stats['matches_inserted'] += 1

                except Exception as e:
                    # Error processing match - skip
                    stats['errors'] += 1

        except Exception as e:
            logger.error(f"Error collecting upcoming matches: {e}")
            stats['errors'] += 1
        finally:
            self._close_driver()

        logger.info(f"Upcoming matches collection complete: {stats}")
        return stats

    def collect_recent_matches(self, days: int = 7) -> Dict[str, int]:
        """
        Collect odds for recent matches (last N days)

        Args:
            days: Number of days to look back

        Returns:
            Statistics dictionary
        """
        logger.info(f"=== Collecting recent matches (last {days} days) ===")

        stats = {
            'matches_found': 0,
            'matches_inserted': 0,
            'matches_skipped': 0,
            'errors': 0
        }

        # Determine season(s) to check
        cutoff_date = datetime.now() - timedelta(days=days)

        # For simplicity, check current season
        current_month = datetime.now().month
        current_year = datetime.now().year
        if current_month >= 7:
            season = f"{current_year}-{current_year + 1}"
        else:
            season = f"{current_year - 1}-{current_year}"

        # Use season collection with existing odds filter
        stats = self.collect_season_odds(season, skip_existing=True)

        logger.info(f"Recent matches collection complete: {stats}")
        return stats

    def collect_season_odds(self, season: str, skip_existing: bool = True) -> Dict[str, int]:
        """
        Collect all odds for a season

        Args:
            season: Season string (e.g., "2023-2024")
            skip_existing: If True, skip matches that already have odds

        Returns:
            Statistics dictionary
        """
        logger.info(f"=== Collecting odds for season {season} ===")

        stats = {
            'matches_found': 0,
            'matches_inserted': 0,
            'matches_skipped': 0,
            'errors': 0
        }

        if season not in self.SEASONS:
            logger.warning(f"Season {season} not in mapping")
            return stats

        # Check if season already has complete odds coverage
        if skip_existing and self._season_has_complete_odds(season):
            logger.info(f"Season {season} already has complete odds coverage (≥95%), skipping scraping")
            return stats

        # Get existing odds if we're skipping
        existing_match_ids = set()
        if skip_existing:
            existing_match_ids = self._get_existing_odds_match_ids(season)
            logger.info(f"Found {len(existing_match_ids)} matches with existing odds")

        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            driver = self._init_driver()

            # Build URL for season results
            if season == "2024-2025" or season == "2025-2026":
                # Current season uses base results URL
                url = f"{self.BASE_URL}{self.LEAGUE_PATH}/results/"
            else:
                # Historical seasons have season-specific URLs
                url = f"{self.BASE_URL}{self.LEAGUE_PATH}-{season}/results/"

            logger.info(f"Navigating to: {url}")
            driver.get(url)

            # Wait for page to load
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            self._random_delay()

            # Note: OddsPortal may paginate results
            # This is a simplified version - full implementation would handle pagination
            logger.warning("Note: Pagination handling not yet implemented - may miss matches")

            # Find match rows
            match_rows = driver.find_elements(By.CSS_SELECTOR, '.eventRow')
            stats['matches_found'] = len(match_rows)
            logger.info(f"Found {len(match_rows)} matches")

            # Track current date as we iterate (OddsPortal groups by date)
            current_date = None

            for row in match_rows:
                try:
                    match_data = self._parse_match_row_selenium(row)
                    if not match_data:
                        continue

                    # If this row has a date, update current_date
                    if match_data['match_datetime'] and match_data['match_datetime'].hour != datetime.now().hour:
                        current_date = match_data['match_datetime']
                    # If this row doesn't have a proper date but we have a current_date, use it
                    elif current_date and match_data['match_datetime'].date() == datetime.now().date():
                        # This match has time but not date, combine with current_date
                        match_time = match_data['match_datetime']
                        match_data['match_datetime'] = current_date.replace(
                            hour=match_time.hour,
                            minute=match_time.minute
                        )

                    match_data['season'] = season

                    # Check if we should skip
                    if skip_existing:
                        # Quick check by team names (not perfect but faster)
                        home_team = self.team_mapper.get_standard_name(
                            self._normalize_team_name(match_data['home_team'])
                        )
                        away_team = self.team_mapper.get_standard_name(
                            self._normalize_team_name(match_data['away_team'])
                        )

                        # Try to find match_id
                        temp_match_id = self._find_match_id(
                            home_team, away_team,
                            match_data['match_datetime'],
                            season
                        )

                        if temp_match_id and temp_match_id in existing_match_ids:
                            stats['matches_skipped'] += 1
                            continue

                    # Insert/update odds
                    if self._insert_or_update_odds(match_data):
                        stats['matches_inserted'] += 1

                except Exception as e:
                    # Error processing match - skip
                    stats['errors'] += 1

        except Exception as e:
            logger.error(f"Error collecting season {season}: {e}")
            stats['errors'] += 1
        finally:
            self._close_driver()

        logger.info(f"Season {season} complete: {stats}")
        return stats

    def collect_matchday_odds(self, season: str, matchday: int, skip_existing: bool = True) -> Dict[str, int]:
        """
        Collect odds for a specific matchday by matching against database matches
        
        Args:
            season: Season string (e.g., "2025-2026")
            matchday: Matchday number
            skip_existing: If True, skip matches that already have odds
            
        Returns:
            Statistics dictionary
        """
        logger.info(f"=== Collecting odds for {season} MD {matchday} ===")
        
        stats = {
            'matches_found': 0,
            'matches_inserted': 0,
            'matches_skipped': 0,
            'matches_not_found': 0,
            'errors': 0
        }
        
        # Get matches for this matchday from database
        query = """
            SELECT m.match_id, m.match_datetime,
                   ht.team_name as home_team, at.team_name as away_team
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.team_id
            JOIN teams at ON m.away_team_id = at.team_id
            WHERE m.season = ? AND m.matchday = ?
            ORDER BY m.match_datetime
        """
        
        db_matches = self.db.execute_query(query, (season, matchday))
        
        if not db_matches:
            logger.warning(f"No matches found in database for {season} MD {matchday}")
            return stats
        
        logger.info(f"Found {len(db_matches)} matches in database for MD {matchday}")
        
        # Get existing match IDs with odds (for skip logic)
        existing_match_ids = set()
        if skip_existing:
            existing_query = """
                SELECT DISTINCT match_id
                FROM betting_odds
                WHERE bookmaker = 'oddsportal_avg'
            """
            existing_results = self.db.execute_query(existing_query)
            existing_match_ids = {r['match_id'] for r in existing_results} if existing_results else set()
        
        # Check if matches are upcoming (not finished)
        # Get first match to check if it's finished
        first_match = db_matches[0] if db_matches else None
        is_upcoming = False
        if first_match:
            # Check if any matches are finished
            finished_query = """
                SELECT COUNT(*) as finished_count
                FROM matches
                WHERE season = ? AND matchday = ? AND is_finished = 1
            """
            finished_result = self.db.execute_query(finished_query, (season, matchday))
            if finished_result and finished_result[0]['finished_count'] == 0:
                is_upcoming = True
        
        # Collect odds for the entire season (to get all matches from OddsPortal)
        # Then match them to our database matches
        logger.info(f"Scraping OddsPortal for {'upcoming' if is_upcoming else 'finished'} matches...")
        season_matches = []
        
        try:
            if not self.use_selenium:
                logger.error("Selenium required for OddsPortal collection")
                return stats
            
            self._init_driver()
            driver = self.driver
            
            # Import Selenium components
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            # Navigate to appropriate page: main page for upcoming, results page for finished
            if is_upcoming:
                url = f"{self.BASE_URL}{self.LEAGUE_PATH}/"
                logger.info(f"Navigating to upcoming matches page: {url}")
            else:
                url = f"{self.BASE_URL}{self.LEAGUE_PATH}/results"
                logger.info(f"Navigating to results page: {url}")
            driver.get(url)
            
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            self._random_delay()
            
            # Wait for match rows to load (OddsPortal loads dynamically)
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '.eventRow, [data-testid="event-row"]'))
                )
            except:
                pass  # Match rows may not have loaded yet, continuing anyway
            
            # Attempt to load all dynamically rendered rows by scrolling
            last_count = -1
            stable_iters = 0
            for _ in range(14):
                rows_now = driver.find_elements(By.CSS_SELECTOR, '.eventRow, [data-testid="event-row"], div[class*="eventRow"]')
                count = len(rows_now)
                if count == last_count:
                    stable_iters += 1
                else:
                    stable_iters = 0
                if stable_iters >= 2:
                    break
                last_count = count
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.0)
            time.sleep(1.0)
            
            # Find match rows - try multiple selectors (final)
            match_rows = driver.find_elements(By.CSS_SELECTOR, '.eventRow, [data-testid="event-row"], div[class*="eventRow"]')
            logger.info(f"Found {len(match_rows)} matches on OddsPortal")
            
            current_date = None
            
            for i, row in enumerate(match_rows):
                try:
                    match_data = self._parse_match_row_selenium(row)
                    if not match_data:
                        # Log first failure to understand why parsing fails
                        if i == 0:
                            try:
                                from selenium.webdriver.common.by import By
                                row_text = row.text[:300]  # First 300 chars
                                logger.warning(f"Failed to parse first row. Row text: {row_text}...")
                                # Also check what elements are found
                                participants = row.find_elements(By.CSS_SELECTOR, '[data-testid="event-participants"]')
                                p_tags = row.find_elements(By.TAG_NAME, 'p')
                                logger.debug(f"  Found {len(participants)} participants elements, {len(p_tags)} p tags")
                            except Exception as e:
                                logger.debug(f"  Error inspecting row: {e}")
                        continue
                    
                    # Handle date parsing for matchday scraping (same logic as upcoming)
                    if match_data.get('has_date') and match_data.get('match_datetime'):
                        current_date = match_data['match_datetime'].date()
                    if match_data.get('has_time') and not match_data.get('has_date') and current_date and match_data.get('match_datetime'):
                        mt = match_data['match_datetime']
                        match_data['match_datetime'] = datetime(
                            current_date.year, current_date.month, current_date.day, mt.hour, mt.minute
                        )
                    
                    match_data['season'] = season
                    season_matches.append(match_data)
                    
                except Exception as e:
                    stats['errors'] += 1
                    if i < 3:  # Log first few errors
                        logger.debug(f"Error parsing row {i+1}: {e}")
            
            # Now match OddsPortal matches to database matches
            logger.info(f"Matching {len(season_matches)} OddsPortal matches to {len(db_matches)} database matches...")
            
            # Log sample OddsPortal matches for debugging (only if no matches found)
            if not season_matches:
                logger.warning("No matches found on OddsPortal page - check if page loaded correctly")
            
            for db_match in db_matches:
                db_match_id = db_match['match_id']
                db_home = db_match['home_team']
                db_away = db_match['away_team']
                db_datetime = db_match['match_datetime']
                
                # Skip if already has odds
                if skip_existing and db_match_id in existing_match_ids:
                    stats['matches_skipped'] += 1
                    continue
                
                # Find matching OddsPortal match
                matched_odds = None
                best_match_score = 0
                
                for odds_match in season_matches:
                    # Normalize team names for comparison
                    odds_home_raw = odds_match.get('home_team', '')
                    odds_away_raw = odds_match.get('away_team', '')
                    
                    odds_home = self._normalize_team_name(odds_home_raw)
                    odds_away = self._normalize_team_name(odds_away_raw)
                    
                    odds_home_std = self.team_mapper.get_standard_name(odds_home)
                    odds_away_std = self.team_mapper.get_standard_name(odds_away)
                    
                    # Try without prefixes for both
                    def remove_prefixes(name):
                        """Remove common team prefixes"""
                        prefixes = ['FC ', 'SV ', 'TSV ', 'SC ', '1. FC ', '1. ', 'TSG ', 'SSV ', 'MSV ', 'VfL ', 'VfB ', 'Rot-Weiss ', 'Rot-Weiß ']
                        for prefix in prefixes:
                            if name.startswith(prefix):
                                return name[len(prefix):].strip()
                        return name
                    
                    def normalize_for_comparison(name):
                        """Normalize team name for comparison"""
                        # Remove prefixes
                        name = remove_prefixes(name)
                        # Remove common suffixes
                        name = name.replace(' 1899', '').replace(' 1904', '').replace(' 1846', '').replace(' 05', '').replace(' II', '').replace(' 2', '')
                        # Normalize umlauts and special chars (convert to ASCII-friendly)
                        name = name.replace('ä', 'a').replace('ö', 'o').replace('ü', 'u').replace('ß', 'ss')
                        return name.strip()
                    
                    odds_home_normalized = normalize_for_comparison(odds_home_std)
                    odds_away_normalized = normalize_for_comparison(odds_away_std)
                    db_home_normalized = normalize_for_comparison(db_home)
                    db_away_normalized = normalize_for_comparison(db_away)
                    
                    # Check if teams match (multiple strategies)
                    home_match = (odds_home_std == db_home or 
                                 odds_home_normalized == db_home_normalized or
                                 odds_home_std.lower() == db_home.lower() or
                                 odds_home_normalized.lower() == db_home_normalized.lower() or
                                 odds_home_raw == db_home or  # Direct match with raw name
                                 odds_home_raw.lower() == db_home.lower())
                    away_match = (odds_away_std == db_away or 
                                 odds_away_normalized == db_away_normalized or
                                 odds_away_std.lower() == db_away.lower() or
                                 odds_away_normalized.lower() == db_away_normalized.lower() or
                                 odds_away_raw == db_away or  # Direct match with raw name
                                 odds_away_raw.lower() == db_away.lower())
                    
                    # Also check date (within 1 day)
                    date_match = False
                    if odds_match.get('match_datetime') and db_datetime:
                        try:
                            import pandas as pd
                            odds_date = pd.to_datetime(odds_match['match_datetime']).date()
                            db_date = pd.to_datetime(db_datetime).date()
                            date_match = abs((odds_date - db_date).days) <= 1
                        except:
                            pass
                    
                    # Calculate match score (higher is better)
                    match_score = 0
                    if home_match:
                        match_score += 2
                    if away_match:
                        match_score += 2
                    if date_match:
                        match_score += 1
                    
                    # Require at least team match, prefer with date match
                    if (home_match and away_match) and (date_match or not db_datetime or match_score > best_match_score):
                        if match_score > best_match_score:
                            matched_odds = odds_match
                            best_match_score = match_score
                
                if matched_odds:
                    # Insert/update odds
                    # Convert db_datetime to datetime object if it's a string
                    import pandas as pd
                    if isinstance(db_datetime, str):
                        db_datetime_obj = pd.to_datetime(db_datetime).to_pydatetime()
                    elif isinstance(db_datetime, datetime):
                        db_datetime_obj = db_datetime
                    else:
                        db_datetime_obj = pd.to_datetime(db_datetime).to_pydatetime()
                    
                    matched_odds['match_datetime'] = db_datetime_obj
                    matched_odds['match_id'] = db_match_id  # Pass match_id directly to avoid lookup issues
                    logger.info(f"✓ Matched: {db_home} vs {db_away} -> Odds: {matched_odds.get('odds_home')}/{matched_odds.get('odds_draw')}/{matched_odds.get('odds_away')}")
                    if self._insert_or_update_odds(matched_odds):
                        stats['matches_inserted'] += 1
                        stats['matches_found'] += 1
                    else:
                        stats['errors'] += 1
                else:
                    logger.warning(f"✗ Could not find OddsPortal match for: {db_home} vs {db_away} (MD {matchday})")
                    # Log available OddsPortal teams for debugging (only first 3 failures)
                    if season_matches and stats['matches_not_found'] < 3:
                        logger.debug(f"  Available OddsPortal teams (sample): {[(m.get('home_team'), m.get('away_team')) for m in season_matches[:5]]}")
                    stats['matches_not_found'] += 1
        
        except Exception as e:
            logger.error(f"Error collecting matchday odds: {e}")
            import traceback
            logger.error(traceback.format_exc())
            stats['errors'] += 1
        finally:
            self._close_driver()
        
        # Summary
        total = stats['matches_inserted'] + stats['matches_skipped'] + stats['matches_not_found']
        if total > 0:
            logger.info(f"MD {matchday} odds: {stats['matches_inserted']} inserted, {stats['matches_skipped']} skipped, {stats['matches_not_found']} not found")
        else:
            logger.warning(f"MD {matchday} odds: No matches processed")
        return stats

    def collect_all_seasons(self, start_season: str = "2009-2010",
                           end_season: str = "2025-2026",
                           skip_existing: bool = True) -> Dict[str, int]:
        """
        Collect odds for all seasons

        Args:
            start_season: Starting season
            end_season: Ending season
            skip_existing: If True, skip matches that already have odds

        Returns:
            Combined statistics
        """
        logger.info(f"=== Collecting all seasons ({start_season} to {end_season}) ===")

        total_stats = {
            'seasons_processed': 0,
            'matches_found': 0,
            'matches_inserted': 0,
            'matches_skipped': 0,
            'errors': 0
        }

        seasons = list(self.SEASONS.keys())
        start_idx = seasons.index(start_season) if start_season in seasons else 0
        end_idx = seasons.index(end_season) + 1 if end_season in seasons else len(seasons)

        for season in seasons[start_idx:end_idx]:
            logger.info(f"Processing season: {season}")

            stats = self.collect_season_odds(season, skip_existing=skip_existing)

            total_stats['seasons_processed'] += 1
            total_stats['matches_found'] += stats['matches_found']
            total_stats['matches_inserted'] += stats['matches_inserted']
            total_stats['matches_skipped'] += stats.get('matches_skipped', 0)
            total_stats['errors'] += stats['errors']

            # Delay between seasons
            self._random_delay(3, 6)

        logger.info(f"All seasons complete: {total_stats}")
        return total_stats


def main():
    """Main execution with CLI argument parsing"""
    import argparse

    parser = argparse.ArgumentParser(
        description='OddsPortal Collector for 3. Liga Betting Odds'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['upcoming', 'recent', 'season', 'all'],
        default='recent',
        help='Collection mode (default: recent)'
    )
    parser.add_argument(
        '--season',
        type=str,
        help='Specific season to collect (e.g., 2023-2024)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days for recent mode (default: 7)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip matches that already have odds (default: True)'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_false',
        dest='skip_existing',
        help='Re-scrape all matches, even if they have odds'
    )

    args = parser.parse_args()

    collector = OddsPortalCollector(use_selenium=True)

    try:
        if args.mode == 'upcoming':
            stats = collector.collect_upcoming_matches()
        elif args.mode == 'recent':
            stats = collector.collect_recent_matches(days=args.days)
        elif args.mode == 'season':
            if not args.season:
                logger.error("--season required for 'season' mode")
                sys.exit(1)
            stats = collector.collect_season_odds(args.season, skip_existing=args.skip_existing)
        elif args.mode == 'all':
            stats = collector.collect_all_seasons(skip_existing=args.skip_existing)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)

        # Print summary
        print("\n" + "=" * 60)
        print("ODDSPORTAL COLLECTION SUMMARY")
        print("=" * 60)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("=" * 60)

    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import sys
    print("This script is deprecated. Use: python main.py collect-oddsportal [args]", file=sys.stderr)
    sys.exit(2)
