"""
Verify historical values across multiple teams and seasons
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from scripts.collectors.transfermarkt_collector import TransfermarktCollector
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")

def verify_multiple_teams():
    """Verify historical scraping with multiple teams and seasons"""
    collector = TransfermarktCollector()

    test_cases = [
        ("Eintracht Braunschweig", "2019-2020", "2023-2024"),
        ("1. FC Magdeburg", "2018-2019", "2022-2023"),
        ("Dynamo Dresden", "2017-2018", "2021-2022"),
    ]

    logger.info("="*80)
    logger.info("MULTI-TEAM HISTORICAL VALUE VERIFICATION")
    logger.info("="*80)

    for team_name, old_season, recent_season in test_cases:
        logger.info(f"\n{team_name}:")
        logger.info("-" * 80)

        # Old season
        old_data = collector.scrape_squad_value(team_name, old_season)
        if old_data:
            old_val = f"€{old_data['total_squad_value']:,}" if old_data['total_squad_value'] else "N/A"
            logger.info(f"  {old_season}: {old_val:>15s} ({old_data['num_players']} players)")

        # Recent season
        recent_data = collector.scrape_squad_value(team_name, recent_season)
        if recent_data:
            recent_val = f"€{recent_data['total_squad_value']:,}" if recent_data['total_squad_value'] else "N/A"
            logger.info(f"  {recent_season}: {recent_val:>15s} ({recent_data['num_players']} players)")

        # Calculate difference
        if old_data and recent_data and old_data['total_squad_value'] and recent_data['total_squad_value']:
            diff = recent_data['total_squad_value'] - old_data['total_squad_value']
            pct_change = (diff / old_data['total_squad_value']) * 100

            if abs(pct_change) > 5:
                logger.success(f"  ✓ Difference: €{diff:,} ({pct_change:+.1f}%) - Historical data working!")
            else:
                logger.warning(f"  ⚠ Difference: €{diff:,} ({pct_change:+.1f}%) - Values very similar")

    logger.info("\n" + "="*80)

if __name__ == "__main__":
    verify_multiple_teams()
