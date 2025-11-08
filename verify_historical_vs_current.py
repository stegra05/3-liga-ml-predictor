"""
Verify that we're getting historical values, not current ones
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from scripts.collectors.transfermarkt_collector import TransfermarktCollector
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")

def compare_historical_vs_current():
    """Compare historical vs current market values"""
    collector = TransfermarktCollector()

    team_name = "1. FC Kaiserslautern"

    # Historical season (2019-2020)
    historical_season = "2019-2020"
    logger.info(f"Fetching HISTORICAL data for {team_name} in {historical_season}...")
    historical_data = collector.scrape_squad_value(team_name, historical_season)

    # Current/recent season (2024-2025)
    current_season = "2024-2025"
    logger.info(f"\nFetching CURRENT data for {team_name} in {current_season}...")
    current_data = collector.scrape_squad_value(team_name, current_season)

    logger.info("\n" + "="*60)
    logger.info("COMPARISON: Historical vs Current Squad Values")
    logger.info("="*60)

    if historical_data:
        hist_val = f"€{historical_data['total_squad_value']:,}" if historical_data['total_squad_value'] else "N/A"
        logger.info(f"Historical ({historical_season}): {hist_val}")
        logger.info(f"  Players: {historical_data['num_players']}")

    if current_data:
        curr_val = f"€{current_data['total_squad_value']:,}" if current_data['total_squad_value'] else "N/A"
        logger.info(f"\nCurrent ({current_season}):    {curr_val}")
        logger.info(f"  Players: {current_data['num_players']}")

    if historical_data and current_data and historical_data['total_squad_value'] and current_data['total_squad_value']:
        diff = current_data['total_squad_value'] - historical_data['total_squad_value']
        pct_change = (diff / historical_data['total_squad_value']) * 100

        logger.info(f"\nDifference: €{diff:,} ({pct_change:+.1f}%)")

        if abs(diff) > 1_000_000:  # More than 1M difference
            logger.success("✓ Values are DIFFERENT - Historical scraping is working correctly!")
        else:
            logger.warning("⚠ Values are very similar - May be getting current values for both")

    logger.info("="*60)

if __name__ == "__main__":
    compare_historical_vs_current()
