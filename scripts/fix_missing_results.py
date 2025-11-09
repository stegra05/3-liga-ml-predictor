"""
Fix matches with missing results by re-fetching from OpenLigaDB
"""

import sys
from pathlib import Path
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from database.db_manager import get_db
from scripts.collectors.openligadb_collector import OpenLigaDBCollector


def fix_missing_results():
    """Fix matches that are marked finished but missing results/goals"""
    db = get_db()
    collector = OpenLigaDBCollector()

    logger.info("=== Finding matches with missing results ===")

    # Find matches that are finished but missing results
    query = """
        SELECT match_id, openligadb_match_id, season, matchday,
               home_team_id, away_team_id, home_goals, away_goals, result
        FROM matches
        WHERE is_finished = 1 
          AND (home_goals IS NULL OR away_goals IS NULL OR result IS NULL)
        ORDER BY season, matchday
    """
    matches = db.execute_query(query)

    if not matches:
        logger.success("No matches with missing results found!")
        return

    logger.info(f"Found {len(matches)} matches with missing results")

    fixed = 0
    failed = 0

    for match in matches:
        match_id = match['match_id']
        openligadb_id = match['openligadb_match_id']
        season = match['season']

        if not openligadb_id:
            logger.warning(f"Match {match_id} has no openligadb_match_id, skipping")
            failed += 1
            continue

        # Extract season year from season string (e.g., "2022-2023" -> "2022")
        season_year = season.split('-')[0]

        logger.info(f"Fixing match {match_id} (openligadb_id={openligadb_id}, season={season})")

        # Try to fetch match data from OpenLigaDB
        # We'll fetch the full season and find our match
        matches_raw = collector.get_matchdata_for_season(season_year)

        if not matches_raw:
            logger.warning(f"Could not fetch season data for {season_year}")
            failed += 1
            continue

        # Find the specific match
        target_match = None
        for m in matches_raw:
            if m.get('matchID') == openligadb_id:
                target_match = m
                break

        if not target_match:
            logger.warning(f"Match {openligadb_id} not found in OpenLigaDB response")
            failed += 1
            continue

        # Parse the match data
        try:
            parsed = collector.parse_match_data(target_match, season)
            if not parsed:
                logger.warning(f"Could not parse match {openligadb_id}")
                failed += 1
                continue

            # Update the match in database
            if parsed.get('home_goals') is not None and parsed.get('away_goals') is not None:
                query_update = """
                    UPDATE matches
                    SET home_goals = ?, away_goals = ?, result = ?,
                        is_finished = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE match_id = ?
                """
                db.execute_insert(query_update, (
                    parsed['home_goals'],
                    parsed['away_goals'],
                    parsed['result'],
                    parsed['is_finished'],
                    match_id
                ))
                logger.success(f"Fixed match {match_id}: {parsed['home_goals']}-{parsed['away_goals']} ({parsed['result']})")
                fixed += 1
            else:
                # Match has no results - check if it's actually finished
                # If OpenLigaDB says it's not finished, update our database
                if not parsed.get('is_finished', False):
                    query_update = """
                        UPDATE matches
                        SET is_finished = 0, updated_at = CURRENT_TIMESTAMP
                        WHERE match_id = ?
                    """
                    db.execute_insert(query_update, (match_id,))
                    logger.info(f"Updated match {match_id}: marked as not finished (no results available)")
                    fixed += 1
                else:
                    logger.warning(f"Match {openligadb_id} marked finished but has no results in OpenLigaDB (likely cancelled/postponed)")
                    failed += 1

        except Exception as e:
            logger.error(f"Error processing match {openligadb_id}: {e}")
            failed += 1

    logger.success(f"""
=== Fix complete ===
Fixed: {fixed}
Failed: {failed}
Total: {len(matches)}
    """)


def main():
    fix_missing_results()


if __name__ == "__main__":
    main()

