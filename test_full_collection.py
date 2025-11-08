"""
Test script to run full collection for a historical season
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from scripts.collectors.transfermarkt_collector import TransfermarktCollector
from database.db_manager import get_db
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")

def test_full_season_collection():
    """Test collecting data for a full historical season"""
    collector = TransfermarktCollector()
    db = get_db()

    # Test with an older season
    test_season = "2019-2020"

    logger.info(f"Testing full collection for season {test_season}")
    logger.info("="*60)

    # Get teams that played in this season
    query = """
        SELECT DISTINCT t.team_name
        FROM teams t
        WHERE t.team_id IN (
            SELECT DISTINCT home_team_id FROM matches WHERE season = ?
            UNION
            SELECT DISTINCT away_team_id FROM matches WHERE season = ?
        )
        ORDER BY t.team_name
        LIMIT 5
    """
    teams = db.execute_query(query, (test_season, test_season))

    logger.info(f"Found {len(teams)} teams to collect (limiting to first 5 for testing)")
    logger.info(f"Teams: {', '.join([t['team_name'] for t in teams])}")
    logger.info("="*60)

    results = []

    for idx, team in enumerate(teams):
        team_name = team['team_name']
        logger.info(f"\n[{idx + 1}/{len(teams)}] Processing {team_name}...")

        try:
            # Scrape squad value
            squad_data = collector.scrape_squad_value(team_name, test_season)
            if squad_data:
                value_str = f"€{squad_data['total_squad_value']:,}" if squad_data['total_squad_value'] else "N/A"
                logger.success(f"  Squad value: {value_str} ({squad_data['num_players']} players)")

                results.append({
                    'team': team_name,
                    'squad_value': squad_data['total_squad_value'],
                    'num_players': squad_data['num_players'],
                    'avg_value': squad_data['avg_player_value']
                })

            # Scrape player data
            players = collector.scrape_player_data(team_name, test_season)
            if players:
                logger.success(f"  Players: {len(players)} scraped")

                # Show a few sample players with their values
                logger.info(f"  Sample players:")
                for player in players[:3]:
                    val_str = f"€{player['market_value']:,}" if player['market_value'] else "N/A"
                    logger.info(f"    - {player['player_name']}: {val_str}")

        except Exception as e:
            logger.error(f"Error processing {team_name}: {e}")

    logger.info("\n" + "="*60)
    logger.success("Collection test complete!")
    logger.info("\nSummary of collected squad values:")
    logger.info("-" * 60)
    for result in results:
        val_str = f"€{result['squad_value']:,}" if result['squad_value'] else "N/A"
        avg_str = f"€{result['avg_value']:,}" if result['avg_value'] else "N/A"
        logger.info(f"{result['team']:30s} | Total: {val_str:>15s} | Avg: {avg_str:>12s}")
    logger.info("="*60)

if __name__ == "__main__":
    test_full_season_collection()
