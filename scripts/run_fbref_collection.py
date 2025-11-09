"""
Run FBref Full Collection
Collects data for all available seasons (2018-2019 through 2025-2026)
"""

from collectors.fbref_collector import FBrefCollector
from loguru import logger
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def main():
    """Run full FBref data collection"""

    logger.info("="*70)
    logger.info("FBREF FULL COLLECTION - ALL SEASONS")
    logger.info("="*70)
    logger.info("")
    logger.info("This will collect data for the following seasons:")
    logger.info("  • 2018-2019")
    logger.info("  • 2019-2020")
    logger.info("  • 2020-2021")
    logger.info("  • 2021-2022")
    logger.info("  • 2022-2023")
    logger.info("  • 2023-2024")
    logger.info("  • 2024-2025")
    logger.info("  • 2025-2026")
    logger.info("")
    logger.info("Data to be collected:")
    logger.info("  ✓ League standings (final tables)")
    logger.info("  ✓ Team season statistics")
    logger.info("  ✗ Player statistics (not available for 3. Liga)")
    logger.info("")
    logger.info("Estimated time: 20-30 minutes")
    logger.info("="*70)
    logger.info("")

    # Initialize collector with Selenium
    collector = FBrefCollector(use_selenium=True)

    # Run collection for all seasons
    results = collector.collect_all_seasons()

    # Print final summary
    summary = results.get('summary', {})

    logger.info("")
    logger.info("="*70)
    logger.info("COLLECTION COMPLETE!")
    logger.info("="*70)
    logger.info(f"Total seasons processed: {summary.get('successful_seasons', 0)}/{summary.get('total_seasons', 0)}")
    logger.info(f"Total teams: {summary.get('total_teams', 0)}")
    logger.info(f"Total duration: {summary.get('total_duration_seconds', 0)/60:.1f} minutes")
    logger.info("")
    logger.info("Data saved to database: database/3liga.db")
    logger.info("Tables updated:")
    logger.info("  • league_standings")
    logger.info("  • fbref_collection_log")
    logger.info("="*70)


if __name__ == "__main__":
    main()
