"""
Test script to verify historical market value scraping
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from scripts.collectors.transfermarkt_collector import TransfermarktCollector
from loguru import logger

# Configure logger for testing
logger.remove()
logger.add(sys.stdout, level="DEBUG")

def test_historical_scraping():
    """Test scraping with an older season"""
    collector = TransfermarktCollector()

    # Test with an older season
    test_team = "1. FC Kaiserslautern"
    test_season = "2021-2022"  # Old season to verify historical values

    logger.info(f"Testing historical scraping for {test_team} in {test_season}")
    logger.info("="*60)

    # Test squad value scraping
    logger.info("\n1. Testing squad value scraping...")
    squad_data = collector.scrape_squad_value(test_team, test_season)

    if squad_data:
        logger.success(f"Squad value scraped successfully:")
        logger.info(f"  Team: {squad_data['team_name']}")
        logger.info(f"  Season: {squad_data['season']}")
        logger.info(f"  Total Squad Value: €{squad_data['total_squad_value']:,}" if squad_data['total_squad_value'] else "  Total Squad Value: N/A")
        logger.info(f"  Number of Players: {squad_data['num_players']}")
        logger.info(f"  Average Player Value: €{squad_data['avg_player_value']:,}" if squad_data['avg_player_value'] else "  Average Player Value: N/A")
    else:
        logger.error("Failed to scrape squad value")

    # Test player data scraping (limit to first few for quick test)
    logger.info("\n2. Testing player data scraping...")
    players = collector.scrape_player_data(test_team, test_season)

    if players:
        logger.success(f"Scraped {len(players)} players")
        logger.info("\nFirst 5 players:")
        for i, player in enumerate(players[:5], 1):
            logger.info(f"  {i}. {player['player_name']}")
            logger.info(f"     Position: {player['position']}, Age: {player['age']}")
            logger.info(f"     Market Value: €{player['market_value']:,}" if player['market_value'] else "     Market Value: N/A")
            logger.info(f"     Nationality: {player['nationality']}")
    else:
        logger.error("Failed to scrape player data")

    logger.info("\n" + "="*60)
    logger.success("Test complete!")

if __name__ == "__main__":
    test_historical_scraping()
