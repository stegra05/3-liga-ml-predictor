"""
Rerun FBref Collection for 2018-2019 Season
Quick script to recollect data after fixing team mappings
"""

from collectors.fbref_collector import FBrefCollector
from loguru import logger
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def main():
    """Rerun collection for 2018-2019 season"""

    logger.info("="*70)
    logger.info("RERUNNING FBREF COLLECTION FOR 2018-2019")
    logger.info("="*70)
    logger.info("")
    logger.info("Reason: Added missing team mappings:")
    logger.info("  • BTSV → Eintracht Braunschweig")
    logger.info("  • Fortuna Köln → Fortuna Köln")
    logger.info("  • W'burg Kickers → Würzburger Kickers")
    logger.info("")
    logger.info("This will capture player data for these teams.")
    logger.info("="*70)
    logger.info("")

    # Initialize collector with Selenium
    collector = FBrefCollector(use_selenium=True)

    # Rerun 2018-2019 season
    season = "2018-2019"
    logger.info(f"Collecting data for season: {season}")
    results = collector.collect_season_data(season)

    # Print summary
    logger.info("")
    logger.info("="*70)
    logger.info("COLLECTION COMPLETE!")
    logger.info("="*70)
    logger.info(f"Standings: {results.get('standings', {}).get('teams_collected', 0)} teams")
    logger.info(f"Team stats: {results.get('team_stats', {}).get('teams_collected', 0)} teams")
    logger.info(f"Player stats: {results.get('player_stats', {}).get('players_collected', 0)} players")
    logger.info("")
    logger.info("Note: Data will be merged with existing records (INSERT OR REPLACE)")
    logger.info("="*70)


if __name__ == "__main__":
    main()
