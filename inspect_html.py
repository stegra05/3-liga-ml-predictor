"""
Inspect the actual HTML to see what data is available
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from scripts.collectors.transfermarkt_collector import TransfermarktCollector
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, level="DEBUG")

def inspect_squad_page():
    """Inspect the HTML structure of a squad page"""
    collector = TransfermarktCollector()

    team_name = "SSV Ulm 1846"  # Use the team from the user's example
    season = "2021-2022"  # Old season

    team_url = collector.get_team_transfermarkt_url(team_name, season)
    logger.info(f"Fetching: {team_url}")

    soup = collector._make_request(team_url)
    if not soup:
        logger.error("Failed to fetch page")
        return

    logger.info("\n" + "="*80)
    logger.info("SEARCHING FOR 'GESAMT' ROWS")
    logger.info("="*80)

    # Find all rows containing "Gesamt"
    found_any = False
    for row in soup.find_all('tr'):
        row_text = row.get_text()
        if 'Gesamt' in row_text or 'gesamt' in row_text.lower():
            found_any = True
            cells = row.find_all(['td', 'th'])
            logger.info(f"\nFound row with 'Gesamt':")
            logger.info(f"Number of cells: {len(cells)}")
            for i, cell in enumerate(cells):
                cell_text = cell.get_text(strip=True)
                logger.info(f"  Cell {i}: '{cell_text}'")

    if not found_any:
        logger.warning("No rows containing 'Gesamt' found!")

    logger.info("\n" + "="*80)
    logger.info("SEARCHING FOR MARKET VALUE IN HEADER")
    logger.info("="*80)

    mv_box = soup.find('a', class_='data-header__market-value-wrapper')
    if mv_box:
        logger.info(f"Found header market value: {mv_box.text.strip()}")
    else:
        logger.warning("No header market value found")

    logger.info("\n" + "="*80)

if __name__ == "__main__":
    inspect_squad_page()
