"""
Transfermarkt Referee Data Collector for 3. Liga
Scrapes referee names from Transfermarkt.com
"""

import sys
from pathlib import Path
import time
import re
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import requests
from bs4 import BeautifulSoup
from loguru import logger

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from database.db_manager import get_db
from scripts.utils.team_mapper import TeamMapper


class TransfermarktRefereeCollector:
    """Collects referee names from Transfermarkt.com"""
    
    BASE_URL = "https://www.transfermarkt.com"
    COMPETITION_ID = "L3"  # 3. Liga
    
    # Season mapping (Transfermarkt uses year of season start)
    SEASONS = {
        "2009-2010": 2009,
        "2010-2011": 2010,
        "2011-2012": 2011,
        "2012-2013": 2012,
        "2013-2014": 2013,
        "2014-2015": 2014,
        "2015-2016": 2015,
        "2016-2017": 2016,
        "2017-2018": 2017,
        "2018-2019": 2018,
        "2019-2020": 2019,
        "2020-2021": 2020,
        "2021-2022": 2021,
        "2022-2023": 2022,
        "2023-2024": 2023,
        "2024-2025": 2024,
        "2025-2026": 2025,
    }
    
    def __init__(self):
        """Initialize collector"""
        self.db = get_db()
        self.team_mapper = TeamMapper()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        logger.info("Transfermarkt referee collector initialized")
    
    def _normalize_team_name(self, team_name: str) -> str:
        """Normalize team names from Transfermarkt format"""
        team_name = team_name.strip()
        
        # Common Transfermarkt variations
        replacements = {
            '1860 München': 'TSV 1860 München',
            '1860 Munich': 'TSV 1860 München',
            'Aachen': 'Alemannia Aachen',
            'RW Essen': 'Rot-Weiss Essen',
            'Rot-Weiß Essen': 'Rot-Weiss Essen',
            'Carl Zeiss Jena': 'FC Carl Zeiss Jena',
            'Dynamo Dresden': 'SG Dynamo Dresden',
            'Hansa Rostock': 'FC Hansa Rostock',
            'VfL Osnabrück': 'VfL Osnabrueck',
            'Osnabrück': 'VfL Osnabrueck',
            'Preußen Münster': 'SC Preussen Muenster',
            'Preussen Münster': 'SC Preussen Muenster',
        }
        
        for old, new in replacements.items():
            if old.lower() in team_name.lower():
                return new
        
        return team_name
    
    def _normalize_referee_name(self, referee_str: str) -> Optional[str]:
        """Normalize referee name"""
        if not referee_str:
            return None
        
        # Clean up referee name
        referee = referee_str.strip()
        
        # Remove common prefixes/suffixes
        referee = re.sub(r'^Schiedsrichter:\s*', '', referee, flags=re.IGNORECASE)
        referee = re.sub(r'\s+', ' ', referee)  # Normalize whitespace
        
        if not referee or referee == '-' or referee.lower() == 'unknown':
            return None
        
        return referee
    
    def _scrape_matchday(self, season_year: int, matchday: int) -> List[Dict]:
        """
        Scrape referee for a single matchday
        
        Args:
            season_year: Year season starts (e.g., 2024 for 2024-2025)
            matchday: Matchday number
            
        Returns:
            List of match data dictionaries with referee names
        """
        url = f"{self.BASE_URL}/3-liga/spieltag/wettbewerb/{self.COMPETITION_ID}/saison_id/{season_year}/spieltag/{matchday}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            matches = []
            
            # Find all match tables
            all_tables = soup.find_all('table')
            
            for table in all_tables:
                try:
                    # Get first row for team names
                    first_row = table.find('tr')
                    if not first_row:
                        continue
                    
                    cells = first_row.find_all('td')
                    if len(cells) < 8:
                        continue
                    
                    # Extract team names (cell 0 = home, cell 7 = away)
                    home_text = cells[0].get_text(strip=True)
                    away_text = cells[7].get_text(strip=True)
                    
                    # Clean team names (remove rankings like "(11.)")
                    home_team = re.sub(r'^\(\d+\.\)', '', home_text).strip()
                    away_team = re.sub(r'\(\d+\.\)$', '', away_text).strip()
                    
                    if not home_team or not away_team:
                        continue
                    
                    # Referee information is NOT on matchday pages - must visit match report page
                    # Find match report link
                    referee = None
                    report_link = table.find('a', href=re.compile(r'/spielbericht/'))
                    
                    if report_link:
                        report_href = report_link.get('href')
                        if report_href:
                            # Build full URL
                            if not report_href.startswith('http'):
                                report_url = f"{self.BASE_URL}{report_href}"
                            else:
                                report_url = report_href
                            
                            # Scrape individual match report page for referee
                            referee = self._scrape_match_report_referee(report_url)
                            time.sleep(0.3)  # Be polite between requests
                    
                    if referee and home_team and away_team:
                        matches.append({
                            'home_team': self._normalize_team_name(home_team),
                            'away_team': self._normalize_team_name(away_team),
                            'referee': referee,
                            'matchday': matchday
                        })
                
                except Exception as e:
                    logger.debug(f"Error parsing match table: {e}")
                    continue
            
            return matches
        
        except Exception as e:
            logger.error(f"Error scraping matchday {matchday} for season {season_year}: {e}")
            return []
    
    def _scrape_match_report_referee(self, match_url: str) -> Optional[str]:
        """
        Scrape referee from individual match report page
        
        Args:
            match_url: Full URL to match report page
            
        Returns:
            Referee name or None
        """
        try:
            response = self.session.get(match_url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Method 1: Look for referee profile link (most reliable)
            # Transfermarkt uses links with /profil/schiedsrichter/ in href
            referee_link = soup.find('a', href=re.compile(r'/profil/schiedsrichter/'))
            if referee_link:
                # Get referee name from title attribute or link text
                referee_name = referee_link.get('title') or referee_link.get_text(strip=True)
                if referee_name:
                    return self._normalize_referee_name(referee_name)
            
            # Method 2: Look for "Referee:" text (English) - Transfermarkt uses English
            referee_labels = soup.find_all(string=re.compile(r'Referee:', re.IGNORECASE))
            for label in referee_labels:
                parent = label.find_parent()
                if parent:
                    # Look for link next to "Referee:" label
                    link = parent.find('a', href=re.compile(r'/profil/schiedsrichter/'))
                    if link:
                        referee_name = link.get('title') or link.get_text(strip=True)
                        if referee_name:
                            return self._normalize_referee_name(referee_name)
                    
                    # Fallback: try to extract from parent text
                    parent_text = parent.get_text(strip=True)
                    match = re.search(r'Referee[:\s]+([^\n\r]+)', parent_text, re.IGNORECASE)
                    if match:
                        referee_name = match.group(1).strip()
                        # Clean up - remove any extra text after name
                        referee_name = re.sub(r'\s*\|.*$', '', referee_name)  # Remove pipe and after
                        return self._normalize_referee_name(referee_name)
            
            # Method 3: Look in spieldaten section
            spieldaten_sections = soup.find_all(class_=re.compile(r'spieldaten', re.IGNORECASE))
            for section in spieldaten_sections:
                referee_link = section.find('a', href=re.compile(r'/profil/schiedsrichter/'))
                if referee_link:
                    referee_name = referee_link.get('title') or referee_link.get_text(strip=True)
                    if referee_name:
                        return self._normalize_referee_name(referee_name)
            
            return None
        
        except Exception as e:
            logger.debug(f"Error scraping match report {match_url}: {e}")
            return None
    
    def _scrape_season(self, season: str, use_match_reports: bool = False, 
                      skip_complete_matchdays: bool = True) -> List[Dict]:
        """
        Scrape referees for entire season
        
        Args:
            season: Season in format "2024-2025"
            use_match_reports: If True, also scrape individual match reports (slower but more complete)
            skip_complete_matchdays: If True, skip matchdays that already have referee data
            
        Returns:
            List of match data dictionaries
        """
        if season not in self.SEASONS:
            logger.warning(f"Season {season} not in mapping")
            return []
        
        season_year = self.SEASONS[season]
        logger.info(f"Scraping referees for season {season} (year {season_year})")
        
        all_matches = []
        skipped_matchdays = 0
        
        # 3. Liga has 38 matchdays
        for matchday in range(1, 39):
            # Check if matchday already has referee data
            if skip_complete_matchdays:
                has_referees = self._matchday_has_referees(season, matchday)
                if has_referees:
                    logger.debug(f"Skipping matchday {matchday} (already has referee data)")
                    skipped_matchdays += 1
                    continue
                else:
                    # Log why we're scraping (for debugging)
                    coverage_info = self._get_matchday_coverage(season, matchday)
                    if coverage_info:
                        logger.debug(f"Matchday {matchday}: {coverage_info['with_referee']}/{coverage_info['total']} matches have referees ({coverage_info['coverage']:.0%}), scraping...")
            
            logger.debug(f"Scraping matchday {matchday}...")
            matches = self._scrape_matchday(season_year, matchday)
            
            # If match reports are enabled and referee not found, try individual pages
            if use_match_reports:
                for match in matches:
                    if not match.get('referee'):
                        # Would need match URL here - skip for now
                        pass
            
            all_matches.extend(matches)
            
            # Be polite - small delay between requests
            time.sleep(0.2)
        
        if skipped_matchdays > 0:
            logger.info(f"Skipped {skipped_matchdays} matchdays that already have referee data")
        logger.info(f"Scraped {len([m for m in all_matches if m.get('referee')])} matches with referees for {season}")
        return all_matches
    
    def _match_to_database(self, scraped_match: Dict, season: str) -> Optional[int]:
        """
        Match scraped data to database match_id
        
        Args:
            scraped_match: Dictionary with home_team, away_team, matchday
            season: Season string
            
        Returns:
            match_id from database or None
        """
        query = """
        SELECT m.match_id
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.team_id
        JOIN teams at ON m.away_team_id = at.team_id
        WHERE m.season = ?
        AND m.matchday = ?
        AND ht.team_name = ?
        AND at.team_name = ?
        """
        
        home_team = scraped_match['home_team']
        away_team = scraped_match['away_team']
        matchday = scraped_match['matchday']
        
        # Try with normalized names
        normalized_home = self.team_mapper.get_standard_name(home_team)
        normalized_away = self.team_mapper.get_standard_name(away_team)
        
        result = self.db.execute_query(query, (season, matchday, normalized_home, normalized_away))
        
        if result:
            return result[0]['match_id']
        
        # Try without normalization
        result = self.db.execute_query(query, (season, matchday, home_team, away_team))
        
        if result:
            return result[0]['match_id']
        
        logger.debug(f"No match found: {home_team} vs {away_team}, MD{matchday}, {season}")
        return None
    
    def _match_has_referee(self, match_id: int) -> bool:
        """Check if match already has referee data"""
        query = "SELECT referee FROM matches WHERE match_id = ?"
        result = self.db.execute_query(query, (match_id,))
        if result and result[0]['referee']:
            return True
        return False
    
    def _update_referee(self, match_id: int, referee: str, skip_if_exists: bool = True) -> bool:
        """
        Update referee in database
        
        Args:
            match_id: Match ID
            referee: Referee name
            skip_if_exists: If True, skip update if referee already exists
            
        Returns:
            True if updated, False if skipped or error
        """
        if skip_if_exists and self._match_has_referee(match_id):
            return False
        
        try:
            query = """
            UPDATE matches 
            SET referee = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE match_id = ?
            """
            conn = self.db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(query, (referee, match_id))
                conn.commit()
                return True
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Error updating referee for match {match_id}: {e}")
            return False
    
    def _get_matchday_coverage(self, season: str, matchday: int) -> Optional[Dict]:
        """
        Get referee coverage statistics for a matchday
        
        Args:
            season: Season string
            matchday: Matchday number
            
        Returns:
            Dictionary with coverage info or None
        """
        query = """
            SELECT 
                COUNT(*) as total_matches,
                COUNT(referee) as matches_with_referee
            FROM matches
            WHERE season = ? AND matchday = ?
        """
        result = self.db.execute_query(query, (season, matchday))
        
        if result:
            row = result[0]
            total = row['total_matches']
            with_referee = row['matches_with_referee']
            
            if total == 0:
                return None
            
            coverage = with_referee / total if total > 0 else 0
            return {
                'total': total,
                'with_referee': with_referee,
                'coverage': coverage
            }
        
        return None
    
    def _matchday_matches_are_upcoming(self, season: str, matchday: int) -> bool:
        """
        Check if all matches for a matchday are upcoming (not finished)
        
        Args:
            season: Season string
            matchday: Matchday number
            
        Returns:
            True if all matches are upcoming
        """
        query = """
            SELECT 
                COUNT(*) as total_matches,
                COUNT(CASE WHEN is_finished = 1 THEN 1 END) as finished_matches
            FROM matches
            WHERE season = ? AND matchday = ?
        """
        result = self.db.execute_query(query, (season, matchday))
        
        if result:
            row = result[0]
            total = row['total_matches']
            finished = row['finished_matches']
            
            # If no matches exist, consider it upcoming
            if total == 0:
                return True
            
            # If all matches are upcoming (none finished), return True
            return finished == 0
        
        return True

    def _matchday_has_referees(self, season: str, matchday: int, min_coverage: float = 0.9) -> bool:
        """
        Check if matchday already has sufficient referee coverage
        
        Args:
            season: Season string
            matchday: Matchday number
            min_coverage: Minimum coverage ratio (0.0-1.0) to consider complete
            
        Returns:
            True if matchday has sufficient coverage
        """
        coverage_info = self._get_matchday_coverage(season, matchday)
        if not coverage_info:
            return False
        
        return coverage_info['coverage'] >= min_coverage
    
    def _update_referees_batch(self, updates: List[tuple]) -> int:
        """
        Update multiple referees in database in batch
        
        Args:
            updates: List of (referee, match_id) tuples
            
        Returns:
            Number of successful updates
        """
        if not updates:
            return 0
        
        try:
            query = """
            UPDATE matches 
            SET referee = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE match_id = ?
            """
            conn = self.db.get_connection()
            try:
                cursor = conn.cursor()
                cursor.executemany(query, updates)
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Error batch updating referees: {e}")
            return 0
    
    def collect_matchday_range(self, season: str, start_matchday: int, end_matchday: int,
                               use_match_reports: bool = False,
                               skip_complete_matchdays: bool = True,
                               skip_existing_referees: bool = True,
                               batch_size: int = 50) -> Dict[str, int]:
        """
        Collect referees for a specific range of matchdays
        
        Args:
            season: Season in format "2024-2025"
            start_matchday: First matchday to collect (inclusive)
            end_matchday: Last matchday to collect (inclusive)
            use_match_reports: If True, scrape individual match reports (slower)
            skip_complete_matchdays: If True, skip matchdays that already have referee data
            skip_existing_referees: If True, skip matches that already have referee data
            batch_size: Number of updates to batch before saving to database
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'matches_scraped': 0,
            'matches_matched': 0,
            'matches_updated': 0,
            'matches_skipped': 0,
            'errors': 0
        }
        
        if season not in self.SEASONS:
            logger.warning(f"Season {season} not in mapping")
            return stats
        
        season_year = self.SEASONS[season]
        all_matches = []
        
        # Scrape only the specified matchday range
        for matchday in range(start_matchday, end_matchday + 1):
            # Check if matchday already has referee data
            if skip_complete_matchdays:
                has_referees = self._matchday_has_referees(season, matchday)
                if has_referees:
                    logger.debug(f"Skipping matchday {matchday} (already has referee data)")
                    continue
                else:
                    # Log why we're scraping (for debugging)
                    coverage_info = self._get_matchday_coverage(season, matchday)
                    if coverage_info:
                        logger.debug(f"Matchday {matchday}: {coverage_info['with_referee']}/{coverage_info['total']} matches have referees ({coverage_info['coverage']:.0%}), scraping...")
            
            # Check if matches are upcoming (referees may not be published yet)
            matches_are_upcoming = self._matchday_matches_are_upcoming(season, matchday)
            if matches_are_upcoming:
                logger.debug(f"Matchday {matchday} matches are upcoming - referees may not be published yet on Transfermarkt, skipping scraping")
                continue
            
            logger.debug(f"Scraping matchday {matchday}...")
            matches = self._scrape_matchday(season_year, matchday)
            if len(matches) == 0:
                logger.debug(f"No referee data found for matchday {matchday} (matches may be upcoming or data not available)")
            all_matches.extend(matches)
            time.sleep(0.2)  # Be polite between requests
        
        stats['matches_scraped'] = len(all_matches)
        
        # Batch process matches
        batch_updates = []
        
        for match in all_matches:
            if not match.get('referee'):
                continue
            
            match_id = self._match_to_database(match, season)
            
            if match_id:
                stats['matches_matched'] += 1
                
                # Check if match already has referee
                if skip_existing_referees and self._match_has_referee(match_id):
                    stats['matches_skipped'] += 1
                    continue
                
                batch_updates.append((match['referee'], match_id))
                
                # Save batch when it reaches batch_size
                if len(batch_updates) >= batch_size:
                    updated = self._update_referees_batch(batch_updates)
                    stats['matches_updated'] += updated
                    batch_updates = []
                    logger.debug(f"Saved batch of {updated} referee updates")
        
        # Save remaining updates
        if batch_updates:
            updated = self._update_referees_batch(batch_updates)
            stats['matches_updated'] += updated
            logger.debug(f"Saved final batch of {updated} referee updates")
        
        logger.info(f"Matchday range {start_matchday}-{end_matchday} complete: {stats}")
        return stats

    def collect_season(self, season: str, use_match_reports: bool = False,
                      skip_complete_matchdays: bool = True,
                      skip_existing_referees: bool = True,
                      batch_size: int = 50) -> Dict[str, int]:
        """
        Collect referees for a single season (all 38 matchdays)
        
        Args:
            season: Season in format "2024-2025"
            use_match_reports: If True, scrape individual match reports (slower)
            skip_complete_matchdays: If True, skip matchdays that already have referee data
            skip_existing_referees: If True, skip matches that already have referee data
            batch_size: Number of updates to batch before saving to database
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'matches_scraped': 0,
            'matches_matched': 0,
            'matches_updated': 0,
            'matches_skipped': 0,
            'errors': 0
        }
        
        # Scrape referee data
        matches = self._scrape_season(season, use_match_reports=use_match_reports,
                                     skip_complete_matchdays=skip_complete_matchdays)
        stats['matches_scraped'] = len(matches)
        
        # Batch process matches
        batch_updates = []
        
        for match in matches:
            if not match.get('referee'):
                continue
            
            match_id = self._match_to_database(match, season)
            
            if match_id:
                stats['matches_matched'] += 1
                
                # Check if match already has referee
                if skip_existing_referees and self._match_has_referee(match_id):
                    stats['matches_skipped'] += 1
                    continue
                
                batch_updates.append((match['referee'], match_id))
                
                # Save batch when it reaches batch_size
                if len(batch_updates) >= batch_size:
                    updated = self._update_referees_batch(batch_updates)
                    stats['matches_updated'] += updated
                    batch_updates = []
                    logger.debug(f"Saved batch of {updated} referee updates")
        
        # Save remaining updates
        if batch_updates:
            updated = self._update_referees_batch(batch_updates)
            stats['matches_updated'] += updated
            logger.debug(f"Saved final batch of {updated} referee updates")
        
        logger.info(f"Season {season} complete: {stats}")
        return stats
    
    def collect_all_seasons(self, start_season: str = "2009-2010", 
                           end_season: str = "2025-2026",
                           use_match_reports: bool = False,
                           skip_complete_matchdays: bool = True,
                           skip_existing_referees: bool = True,
                           batch_size: int = 50) -> Dict[str, int]:
        """
        Collect referees for all seasons
        
        Args:
            start_season: Starting season
            end_season: Ending season
            use_match_reports: If True, scrape individual match reports (slower but more complete)
            skip_complete_matchdays: If True, skip matchdays that already have referee data
            skip_existing_referees: If True, skip matches that already have referee data
            batch_size: Number of updates to batch before saving to database
            
        Returns:
            Combined statistics
        """
        total_stats = {
            'seasons_processed': 0,
            'matches_scraped': 0,
            'matches_matched': 0,
            'matches_updated': 0,
            'matches_skipped': 0,
            'errors': 0
        }
        
        seasons = list(self.SEASONS.keys())
        start_idx = seasons.index(start_season) if start_season in seasons else 0
        end_idx = seasons.index(end_season) + 1 if end_season in seasons else len(seasons)
        
        for season in seasons[start_idx:end_idx]:
            logger.info(f"Processing season: {season}")
            
            stats = self.collect_season(season, use_match_reports=use_match_reports,
                                       skip_complete_matchdays=skip_complete_matchdays,
                                       skip_existing_referees=skip_existing_referees,
                                       batch_size=batch_size)
            
            total_stats['seasons_processed'] += 1
            total_stats['matches_scraped'] += stats['matches_scraped']
            total_stats['matches_matched'] += stats['matches_matched']
            total_stats['matches_updated'] += stats['matches_updated']
            total_stats['matches_skipped'] += stats.get('matches_skipped', 0)
            total_stats['errors'] += stats['errors']
            
            # Delay between seasons
            time.sleep(1)
        
        logger.info(f"All seasons complete: {total_stats}")
        return total_stats


def main():
    """Main execution"""
    logger.info("Starting Transfermarkt referee collection")
    
    collector = TransfermarktRefereeCollector()
    
    # Collect all historical data
    stats = collector.collect_all_seasons()
    
    logger.info("Collection complete!")
    
    # Print summary
    print("\n" + "="*60)
    print("REFEREE DATA COLLECTION SUMMARY")
    print("="*60)
    print(f"Seasons processed: {stats['seasons_processed']}")
    print(f"Matches scraped: {stats['matches_scraped']}")
    print(f"Matches matched to database: {stats['matches_matched']}")
    print(f"Matches updated: {stats['matches_updated']}")
    if 'matches_skipped' in stats:
        print(f"Matches skipped (already had data): {stats['matches_skipped']}")
    print(f"Errors: {stats['errors']}")
    print("="*60)
    
    # Check coverage
    db = get_db()
    result = db.execute_query("""
        SELECT 
            COUNT(*) as total_matches,
            COUNT(referee) as with_referee,
            ROUND(COUNT(referee) * 100.0 / COUNT(*), 2) as percentage
        FROM matches 
        WHERE is_finished = 1
    """)
    
    if result:
        row = result[0]
        print("\nDATABASE REFEREE COVERAGE:")
        print(f"Total finished matches: {row['total_matches']}")
        print(f"Matches with referee: {row['with_referee']}")
        print(f"Coverage: {row['percentage']}%")
        print("="*60)


if __name__ == "__main__":
    import sys
    print("This script is deprecated. Use: python main.py collect-transfermarkt-referees [args]", file=sys.stderr)
    sys.exit(2)

