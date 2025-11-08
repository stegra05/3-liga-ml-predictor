"""
Transfermarkt Data Collector for 3. Liga
Scrapes player data, squad values, and transfer information
"""

import requests
from bs4 import BeautifulSoup
import time
import re
import json
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from database.db_manager import get_db


class TransfermarktCollector:
    """Collects player and squad data from Transfermarkt"""

    BASE_URL = "https://www.transfermarkt.de"
    RATE_LIMIT_DELAY = 3.0  # Respectful delay between requests

    # User agent to identify ourselves
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (compatible; 3Liga-Dataset-Collector/1.0; Educational Research)',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }

    def __init__(self, urls_file: str = "config/transfermarkt_urls.json"):
        """Initialize Transfermarkt collector"""
        self.db = get_db()
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.url_mappings = self._load_url_mappings(urls_file)
        logger.info(f"Transfermarkt collector initialized with {len(self.url_mappings)} team URLs")

    def _load_url_mappings(self, urls_file: str) -> Dict:
        """Load Transfermarkt URL mappings from JSON file"""
        path = Path(urls_file)
        if not path.exists():
            raise FileNotFoundError(f"URL mappings file not found: {urls_file}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get('teams', {})

    def _make_request(self, url: str, retries: int = 3) -> Optional[BeautifulSoup]:
        """
        Make HTTP request with retries and rate limiting

        Args:
            url: URL to fetch
            retries: Number of retry attempts

        Returns:
            BeautifulSoup object or None on failure
        """
        for attempt in range(retries):
            try:
                logger.debug(f"Fetching: {url}")
                response = self.session.get(url, timeout=15)
                response.raise_for_status()

                # Rate limiting
                time.sleep(self.RATE_LIMIT_DELAY)

                return BeautifulSoup(response.content, 'html.parser')

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Failed to fetch {url} after {retries} attempts")
                    return None

    def get_team_transfermarkt_url(self, team_name: str, season: str) -> Optional[str]:
        """
        Get Transfermarkt URL for a team using direct URL mappings

        Args:
            team_name: Team name
            season: Season string (e.g., "2023-2024")

        Returns:
            URL string or None
        """
        # Get team data from mappings
        team_data = self.url_mappings.get(team_name)

        if not team_data:
            logger.warning(f"No URL mapping found for team: {team_name}")
            return None

        # Extract season year (first year of the season)
        season_year = season.split('-')[0]

        # Construct URL - use kader (squad) page with query parameter for historical data
        # Format: /kader/verein/{id}/plus/0/galerie/0?saison_id={year}
        url = f"{self.BASE_URL}/{team_data['url_slug']}/kader/verein/{team_data['team_id']}/plus/0/galerie/0?saison_id={season_year}"

        logger.debug(f"Generated URL for {team_name} ({season}): {url}")
        return url

    def parse_market_value(self, value_str: str) -> Optional[int]:
        """
        Parse market value string to integer (in euros)

        Args:
            value_str: String like "€1.50m", "€250k", "€10.00m", "56.80 Mio.", "1.11 Mio."

        Returns:
            Value in euros or None
        """
        if not value_str or value_str == '-':
            return None

        try:
            # Remove currency symbol, whitespace, and newlines
            value_str = value_str.replace('€', '').replace('\n', ' ').strip()

            # Extract just the number part (before any text like "Gesamtmarktwert")
            # Match pattern like "56.80 Mio." or "1.11 Mio." or "250 Tsd."
            import re
            match = re.search(r'([\d.,]+)\s*(Mio\.|Tsd\.|m|k)', value_str, re.IGNORECASE)

            if match:
                number_str = match.group(1).replace(',', '.')  # Convert German comma to decimal point
                unit = match.group(2).lower()
                value = float(number_str)

                # Handle millions
                if unit in ['mio.', 'm']:
                    return int(value * 1_000_000)
                # Handle thousands
                elif unit in ['tsd.', 'k']:
                    return int(value * 1_000)

            # Fallback: try direct conversion
            value_str = value_str.replace(',', '.')
            return int(float(value_str))

        except (ValueError, AttributeError) as e:
            logger.debug(f"Could not parse market value '{value_str}': {e}")
            return None

    def scrape_squad_value(self, team_name: str, season: str) -> Optional[Dict]:
        """
        Scrape squad value for a team in a season from the squad details table

        Args:
            team_name: Team name
            season: Season string (e.g., "2023-2024")

        Returns:
            Dictionary with squad value data
        """
        # Get team URL
        team_url = self.get_team_transfermarkt_url(team_name, season)
        if not team_url:
            return None

        soup = self._make_request(team_url)
        if not soup:
            return None

        try:
            # Find squad value in the "KADERDETAILS NACH POSITIONEN" table
            # This table shows squad details by position with total value
            total_value = None

            # Look for the "Gesamt:" row in the squad details table
            # The table typically has rows for each position (Torwart, Abwehr, Mittelfeld, Sturm)
            # and a final row with "Gesamt:" (Total)

            # Find all table rows
            for row in soup.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                if not cells:
                    continue

                # Look for row containing "Gesamt:" (Total)
                for cell in cells:
                    cell_text = cell.get_text(strip=True)
                    if 'Gesamt:' in cell_text or 'Gesamtmarktwert' in cell_text:
                        # The market value is typically in the next cell or one of the following cells
                        # Try to find the value in this row
                        for value_cell in cells:
                            value_text = value_cell.get_text(strip=True)
                            # Skip cells that just have "Gesamt:" or age values
                            if 'Gesamt' in value_text or value_text.replace(',', '').replace('.', '').isdigit():
                                continue

                            parsed = self.parse_market_value(value_text)
                            if parsed and parsed > 100000:  # Reasonable squad value threshold
                                total_value = parsed
                                logger.debug(f"Found squad value in Gesamt row: {value_text} -> {total_value}")
                                break

                        if total_value:
                            break

                if total_value:
                    break

            # Fallback: try to find in the data-header (current/recent seasons)
            if not total_value:
                market_value_box = soup.find('a', class_='data-header__market-value-wrapper')
                if market_value_box:
                    value_text = market_value_box.text.strip()
                    total_value = self.parse_market_value(value_text)
                    logger.debug(f"Found squad value in header: {value_text} -> {total_value}")

            # Count players from the detailed squad table
            player_rows = soup.find_all('tr', class_=['odd', 'even'])
            num_players = len(player_rows)

            # Calculate average value
            avg_value = int(total_value / num_players) if total_value and num_players > 0 else None

            return {
                'team_name': team_name,
                'season': season,
                'total_squad_value': total_value,
                'num_players': num_players,
                'avg_player_value': avg_value,
                'scraped_at': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error scraping squad value for {team_name}: {e}")
            return None

    def scrape_player_data(self, team_name: str, season: str) -> List[Dict]:
        """
        Scrape player data for a team - squad page shows historical values for the given season

        Args:
            team_name: Team name
            season: Season string (e.g., "2023-2024")

        Returns:
            List of player dictionaries
        """
        team_url = self.get_team_transfermarkt_url(team_name, season)
        if not team_url:
            return []

        soup = self._make_request(team_url)
        if not soup:
            return []

        players = []

        try:
            # Find all player rows in the detailed squad table
            player_rows = soup.find_all('tr', class_=['odd', 'even'])

            for row in player_rows:
                try:
                    # Get all cells in the row
                    cells = row.find_all('td')
                    if len(cells) < 5:  # Minimum cells needed for basic player data
                        continue

                    # Player name (cell 3 in simplified format, cell 3 in detailed format)
                    player_name = None
                    position = None
                    age = None
                    nationality = None
                    height_cm = None
                    preferred_foot = None
                    market_value = None

                    # Check table format based on number of cells
                    if len(cells) == 9:
                        # Simplified format (galerie/0 view)
                        # Cell 3: Player name
                        # Cell 4: Position
                        # Cell 5: Age
                        # Cell 8: Market value

                        name_elem = cells[3].find('a')
                        if not name_elem:
                            continue
                        player_name = name_elem.text.strip()

                        position = cells[4].text.strip() if cells[4] else None

                        age_text = cells[5].text.strip()
                        if age_text and age_text.isdigit():
                            age = int(age_text)

                        # Nationality from cell 6 if available
                        if cells[6]:
                            nat_elem = cells[6].find_all('img', class_='flaggenrahmen')
                            nationality = nat_elem[0].get('alt') if nat_elem else None

                        # Market value in cell 8
                        market_value = self.parse_market_value(cells[8].text.strip()) if cells[8] else None

                    elif len(cells) >= 13:
                        # Detailed format (plus/1 view)
                        # Cell 3: Player name
                        # Cell 4: Position
                        # Cell 5: Birth date with age
                        # Cell 6: Nationality
                        # Cell 8: Height
                        # Cell 9: Foot
                        # Cell 12: Market value

                        name_elem = cells[3].find('a')
                        if not name_elem:
                            continue
                        player_name = name_elem.text.strip()

                        position = cells[4].text.strip() if cells[4] else None

                        age_cell = cells[5].text.strip()
                        age_match = re.search(r'\((\d+)\)', age_cell)
                        age = int(age_match.group(1)) if age_match else None

                        nat_elem = cells[6].find_all('img', class_='flaggenrahmen')
                        nationality = nat_elem[0].get('alt') if nat_elem else None

                        height_text = cells[8].text.strip()
                        height_match = re.search(r'(\d+)', height_text.replace(',', ''))
                        height_cm = int(height_match.group(1)) if height_match else None

                        foot_text = cells[9].text.strip()
                        preferred_foot = foot_text if foot_text and foot_text not in ['-', ''] else None

                        market_value = self.parse_market_value(cells[12].text.strip()) if cells[12] else None
                    else:
                        # Unknown format, skip
                        continue

                    if not player_name:
                        continue

                    player = {
                        'player_name': player_name,
                        'team_name': team_name,
                        'season': season,
                        'position': position,
                        'age': age,
                        'market_value': market_value,
                        'nationality': nationality,
                        'height_cm': height_cm,
                        'preferred_foot': preferred_foot,
                        'scraped_at': datetime.now()
                    }

                    players.append(player)

                except Exception as e:
                    logger.debug(f"Error parsing player row: {e}")
                    continue

            logger.info(f"Scraped {len(players)} players for {team_name} ({season})")
            return players

        except Exception as e:
            logger.error(f"Error scraping players for {team_name}: {e}")
            return []

    def store_squad_value(self, squad_data: Dict) -> None:
        """Store squad value data in database"""

        # Get team_id
        query = "SELECT team_id FROM teams WHERE team_name = ? LIMIT 1"
        result = self.db.execute_query(query, (squad_data['team_name'],))

        if not result:
            logger.warning(f"Team not found in database: {squad_data['team_name']}")
            return

        team_id = result[0]['team_id']

        # Insert squad value
        insert_query = """
            INSERT OR REPLACE INTO squad_values
            (team_id, season, total_squad_value, num_players, avg_player_value, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """

        self.db.execute_insert(insert_query, (
            team_id,
            squad_data['season'],
            squad_data['total_squad_value'],
            squad_data['num_players'],
            squad_data['avg_player_value'],
            squad_data['scraped_at']
        ))

    def store_player_data(self, players: List[Dict]) -> int:
        """
        Store player data in database

        Args:
            players: List of player dictionaries

        Returns:
            Number of players stored
        """
        stored = 0

        for player in players:
            try:
                # Get team_id
                query = "SELECT team_id FROM teams WHERE team_name = ? LIMIT 1"
                result = self.db.execute_query(query, (player['team_name'],))

                if not result:
                    continue

                team_id = result[0]['team_id']

                # Insert player
                insert_query = """
                    INSERT OR REPLACE INTO players
                    (player_name, team_id, season, position, age, market_value, nationality, height_cm, preferred_foot, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                self.db.execute_insert(insert_query, (
                    player['player_name'],
                    team_id,
                    player['season'],
                    player['position'],
                    player['age'],
                    player['market_value'],
                    player['nationality'],
                    player.get('height_cm'),
                    player.get('preferred_foot'),
                    player['scraped_at']
                ))

                stored += 1

            except Exception as e:
                logger.error(f"Error storing player {player.get('player_name', 'unknown')}: {e}")
                continue

        return stored

    def collect_all_teams_data(self, season: str = "2024-2025", limit: Optional[int] = None):
        """
        Collect data for teams that actually played in the 3. Liga in a specific season

        Args:
            season: Season string
            limit: Limit number of teams (for testing)
        """
        logger.info(f"=== Starting Transfermarkt data collection for {season} ===")
        start_time = datetime.now()

        # Get only teams that actually played in this season (from matches)
        query = """
            SELECT DISTINCT t.team_name
            FROM teams t
            WHERE t.team_id IN (
                SELECT DISTINCT home_team_id FROM matches WHERE season = ?
                UNION
                SELECT DISTINCT away_team_id FROM matches WHERE season = ?
            )
            ORDER BY t.team_name
        """
        teams = self.db.execute_query(query, (season, season))

        if limit:
            teams = teams[:limit]

        total_teams = len(teams)
        logger.info(f"Collecting data for {total_teams} teams that played in {season}")

        collected_squads = 0
        collected_players = 0
        failed = 0

        for idx, team in enumerate(teams):
            team_name = team['team_name']

            try:
                logger.info(f"[{idx + 1}/{total_teams}] Processing {team_name}...")

                # Scrape squad value
                squad_data = self.scrape_squad_value(team_name, season)
                if squad_data:
                    self.store_squad_value(squad_data)
                    collected_squads += 1
                    value_str = f"€{squad_data['total_squad_value']:,}" if squad_data['total_squad_value'] else "N/A"
                    logger.success(f"  Squad value: {value_str} ({squad_data['num_players']} players)")

                # Scrape player data
                players = self.scrape_player_data(team_name, season)
                if players:
                    stored = self.store_player_data(players)
                    collected_players += stored
                    logger.success(f"  Players: {stored} stored")

            except Exception as e:
                logger.error(f"Error processing {team_name}: {e}")
                failed += 1
                continue

        duration = (datetime.now() - start_time).total_seconds()

        logger.success(f"""
=== Transfermarkt collection complete ===
Teams processed: {total_teams}
Squad values collected: {collected_squads}
Players collected: {collected_players}
Failed: {failed}
Duration: {duration/60:.1f} minutes
        """)


def main():
    """Main execution"""
    logger.info("=== Transfermarkt Data Collector ===")

    collector = TransfermarktCollector()

    # Get all available seasons from database
    db = get_db()
    seasons_query = "SELECT DISTINCT season FROM matches ORDER BY season DESC"
    seasons_result = db.execute_query(seasons_query)
    seasons = [row['season'] for row in seasons_result]

    logger.info(f"Found {len(seasons)} seasons to collect: {seasons[0]} to {seasons[-1]}")

    # Collect for all seasons (most recent first)
    for season in seasons:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting collection for season {season}")
        logger.info(f"{'='*60}\n")

        collector.collect_all_teams_data(season=season, limit=None)

    logger.success(f"\n{'='*60}")
    logger.success("ALL SEASONS COLLECTION COMPLETE!")
    logger.success(f"Collected data for {len(seasons)} seasons")
    logger.success(f"{'='*60}")


if __name__ == "__main__":
    main()
