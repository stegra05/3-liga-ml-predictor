"""
Debug why player scraping is returning 0 players
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from scripts.collectors.transfermarkt_collector import TransfermarktCollector
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, level="DEBUG")

def debug_player_scraping():
    """Debug player scraping"""
    collector = TransfermarktCollector()

    team_name = "1. FC Kaiserslautern"
    season = "2019-2020"

    team_url = collector.get_team_transfermarkt_url(team_name, season)
    logger.info(f"Fetching: {team_url}")

    soup = collector._make_request(team_url)
    if not soup:
        logger.error("Failed to fetch page")
        return

    logger.info("\n" + "="*80)
    logger.info("SEARCHING FOR PLAYER ROWS")
    logger.info("="*80)

    # Check for rows with class odd/even
    player_rows = soup.find_all('tr', class_=['odd', 'even'])
    logger.info(f"Found {len(player_rows)} rows with class 'odd' or 'even'")

    if player_rows:
        logger.info("\nFirst row structure:")
        first_row = player_rows[0]
        cells = first_row.find_all('td')
        logger.info(f"Number of cells: {len(cells)}")

        for i, cell in enumerate(cells[:15]):  # Show first 15 cells
            logger.info(f"  Cell {i}: {cell.get_text(strip=True)[:50]}")

    logger.info("\n" + "="*80)

if __name__ == "__main__":
    debug_player_scraping()
