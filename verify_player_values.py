"""
Check individual player values to verify historical vs current data
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from scripts.collectors.transfermarkt_collector import TransfermarktCollector
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")

def check_player_values():
    """Check individual player values for historical accuracy"""
    collector = TransfermarktCollector()

    team_name = "Eintracht Braunschweig"

    # Historical season
    historical_season = "2019-2020"
    logger.info(f"Fetching players for {team_name} in {historical_season}...")
    historical_players = collector.scrape_player_data(team_name, historical_season)

    # Current season
    current_season = "2024-2025"
    logger.info(f"Fetching players for {team_name} in {current_season}...")
    current_players = collector.scrape_player_data(team_name, current_season)

    logger.info("\n" + "="*80)
    logger.info(f"PLAYER VALUE COMPARISON: {team_name}")
    logger.info("="*80)

    # Show some players from historical season
    logger.info(f"\nHistorical ({historical_season}) - First 10 players:")
    logger.info("-" * 80)
    for i, player in enumerate(historical_players[:10], 1):
        val_str = f"€{player['market_value']:,}" if player['market_value'] else "N/A"
        logger.info(f"{i:2d}. {player['player_name']:30s} | {player['position']:20s} | {val_str:>12s}")

    logger.info(f"\nCurrent ({current_season}) - First 10 players:")
    logger.info("-" * 80)
    for i, player in enumerate(current_players[:10], 1):
        val_str = f"€{player['market_value']:,}" if player['market_value'] else "N/A"
        logger.info(f"{i:2d}. {player['player_name']:30s} | {player['position']:20s} | {val_str:>12s}")

    logger.info("\n" + "="*80)

    # Check if rosters are different (which they should be)
    historical_names = {p['player_name'] for p in historical_players}
    current_names = {p['player_name'] for p in current_players}

    logger.info(f"\nHistorical roster size: {len(historical_names)} players")
    logger.info(f"Current roster size: {len(current_names)} players")

    common_players = historical_names & current_names
    if common_players:
        logger.info(f"\nPlayers appearing in both rosters ({len(common_players)}):")
        for name in sorted(list(common_players)[:5]):
            hist_val = next((p['market_value'] for p in historical_players if p['player_name'] == name), None)
            curr_val = next((p['market_value'] for p in current_players if p['player_name'] == name), None)

            if hist_val and curr_val:
                hist_str = f"€{hist_val:,}"
                curr_str = f"€{curr_val:,}"
                diff = curr_val - hist_val
                logger.info(f"  {name:30s}: {hist_str:>12s} -> {curr_str:>12s} (diff: €{diff:,})")
    else:
        logger.info("\nNo common players between rosters (expected for teams with high turnover)")

    logger.info("="*80)

if __name__ == "__main__":
    check_player_values()
