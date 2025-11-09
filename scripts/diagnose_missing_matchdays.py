"""
Diagnostic script to identify missing matchdays in the database vs OpenLigaDB
"""

import sys
from pathlib import Path
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from database.db_manager import get_db
from scripts.collectors.openligadb_collector import OpenLigaDBCollector


def diagnose_season(season_year: str):
    """Diagnose missing matchdays for a specific season"""
    db = get_db()
    collector = OpenLigaDBCollector()

    season_str = f"{season_year}-{int(season_year) + 1}"

    logger.info(f"=== Diagnosing season {season_str} ===")

    # Get matchdays in database
    query = """
        SELECT matchday, COUNT(*) as matches, 
               SUM(CASE WHEN is_finished = 1 THEN 1 ELSE 0 END) as finished
        FROM matches
        WHERE season = ?
        GROUP BY matchday
        ORDER BY matchday
    """
    db_matchdays = db.execute_query(query, (season_str,))
    db_matchday_set = {r['matchday'] for r in db_matchdays}

    logger.info(f"\nDatabase matchdays ({len(db_matchday_set)}):")
    for r in db_matchdays:
        logger.info(f"  Matchday {r['matchday']:2d}: {r['matches']:2d} matches ({r['finished']:2d} finished)")

    # Get matchdays from OpenLigaDB
    logger.info(f"\nFetching matchdays from OpenLigaDB...")
    matches_raw = collector.get_matchdata_for_season(season_year)

    if not matches_raw:
        logger.warning(f"No matches found in OpenLigaDB for season {season_year}")
        return

    # Extract matchdays from API response
    api_matchday_set = set()
    api_matchday_counts = {}

    for m in matches_raw:
        group = m.get('group')
        matchday = group.get('groupOrderID', 0) if group else 0
        if matchday > 0:
            api_matchday_set.add(matchday)
            api_matchday_counts[matchday] = api_matchday_counts.get(matchday, 0) + 1

    logger.info(f"\nOpenLigaDB matchdays ({len(api_matchday_set)}):")
    for matchday in sorted(api_matchday_set):
        count = api_matchday_counts.get(matchday, 0)
        logger.info(f"  Matchday {matchday:2d}: {count:2d} matches")

    # Find missing matchdays
    missing_in_db = api_matchday_set - db_matchday_set
    missing_in_api = db_matchday_set - api_matchday_set
    expected_matchdays = set(range(1, 39))  # 38 matchdays expected

    logger.info(f"\n=== Analysis ===")
    logger.info(f"Expected matchdays: {len(expected_matchdays)} (1-38)")
    logger.info(f"Database has: {len(db_matchday_set)} matchdays")
    logger.info(f"OpenLigaDB has: {len(api_matchday_set)} matchdays")

    if missing_in_db:
        logger.warning(f"\n⚠️  Matchdays in OpenLigaDB but NOT in database: {sorted(missing_in_db)}")
        logger.info(f"   These can be backfilled using:")
        logger.info(f"   python scripts/collectors/openligadb_collector.py --season {season_year} --start-matchday {min(missing_in_db)} --end-matchday {max(missing_in_db)} --use-full-season-fetch")

    if missing_in_api:
        logger.warning(f"\n⚠️  Matchdays in database but NOT in OpenLigaDB: {sorted(missing_in_api)}")
        logger.info(f"   These may be data quality issues or API gaps")

    missing_expected = expected_matchdays - api_matchday_set
    if missing_expected:
        logger.info(f"\n📊 Matchdays not available in OpenLigaDB: {sorted(missing_expected)}")
        logger.info(f"   This is normal if the season structure differs (e.g., fewer matchdays)")

    # Summary
    logger.info(f"\n=== Summary ===")
    total_db = sum(r['matches'] for r in db_matchdays)
    total_api = sum(api_matchday_counts.values())
    logger.info(f"Total matches in database: {total_db}")
    logger.info(f"Total matches in OpenLigaDB: {total_api}")
    logger.info(f"Missing matches: {total_api - total_db}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose missing matchdays")
    parser.add_argument('--season', type=str, required=True, help='Season year (e.g., "2022")')
    args = parser.parse_args()

    diagnose_season(args.season)


if __name__ == "__main__":
    main()

