"""
Import Existing CSV Data into Database
Imports FotMob statistics and betting odds from existing CSV files
"""

import pandas as pd
from pathlib import Path
from loguru import logger
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from database.db_manager import get_db
from scripts.utils.team_mapper import TeamMapper


class DataImporter:
    """Imports existing CSV data into database"""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.db = get_db()
        self.team_mapper = TeamMapper()
        logger.info(f"Data importer initialized with directory: {data_dir}")

    def import_fotmob_statistics(self, csv_file: str = "fotmob_stats_all.csv") -> int:
        """
        Import FotMob match statistics

        Args:
            csv_file: CSV filename in data_dir

        Returns:
            Number of records imported
        """
        file_path = self.data_dir / csv_file
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return 0

        logger.info(f"Importing FotMob statistics from {csv_file}")
        df = pd.read_csv(file_path)

        logger.info(f"Loaded {len(df)} records from {csv_file}")

        imported = 0
        skipped = 0

        for idx, row in df.iterrows():
            # Skip matches without stats
            if not row.get('has_stats', False):
                skipped += 1
                continue

            try:
                # Get team IDs
                home_team_id = self.team_mapper.get_team_id(row['home_team'])
                away_team_id = self.team_mapper.get_team_id(row['away_team'])

                if not home_team_id or not away_team_id:
                    logger.warning(f"Could not find teams: {row['home_team']} vs {row['away_team']}")
                    skipped += 1
                    continue

                # Parse match date - convert to string for SQLite
                match_date = pd.to_datetime(row['match_date'])
                match_date_str = match_date.strftime('%Y-%m-%d %H:%M:%S')

                # Find match in database
                match_id = self.db.get_match_id(
                    season=row['season'],
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    match_datetime=match_date_str
                )

                if not match_id:
                    # Try to find by date only (more lenient)
                    query = """
                        SELECT match_id FROM matches
                        WHERE season = ? AND home_team_id = ? AND away_team_id = ?
                        AND DATE(match_datetime) = DATE(?)
                        LIMIT 1
                    """
                    result = self.db.execute_query(query, (row['season'], home_team_id, away_team_id, match_date_str))
                    if result:
                        match_id = result[0]['match_id']
                    else:
                        logger.debug(f"Match not found in DB: {row['home_team']} vs {row['away_team']} on {match_date}")
                        skipped += 1
                        continue

                # Insert home team stats
                self._insert_match_stats(match_id, home_team_id, True, row, 'home')

                # Insert away team stats
                self._insert_match_stats(match_id, away_team_id, False, row, 'away')

                imported += 2  # Count both teams

                if (imported + skipped) % 100 == 0:
                    logger.info(f"Progress: {imported} imported, {skipped} skipped")

            except Exception as e:
                logger.error(f"Error importing row {idx}: {e}")
                skipped += 1

        logger.success(f"FotMob import complete: {imported} team-match stats imported, {skipped} skipped")
        return imported

    def _insert_match_stats(self, match_id: int, team_id: int, is_home: bool, row: pd.Series, prefix: str):
        """Insert match statistics for one team"""

        query = """
            INSERT OR REPLACE INTO match_statistics
            (match_id, team_id, is_home, possession_percent,
             shots_total, shots_on_target, shots_off_target, shots_blocked,
             big_chances, big_chances_missed,
             passes_total, passes_accurate, pass_accuracy_percent,
             crosses_total, crosses_accurate,
             tackles_total, interceptions, clearances,
             duels_total, duels_won, aerials_total, aerials_won,
             fouls_committed, fouls_won, yellow_cards, red_cards,
             corners, offsides, source, has_complete_stats)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        # Get column values with prefix
        def get_val(col):
            val = row.get(f'{prefix}_{col}')
            return None if pd.isna(val) else val

        params = (
            match_id, team_id, is_home,
            get_val('possession'),
            get_val('total_shots'), get_val('shots_on_target'), get_val('shots_off_target'), get_val('blocked_shots'),
            get_val('big_chances'), get_val('big_chances_missed'),
            get_val('total_passes'), get_val('accurate_passes'), get_val('pass_accuracy'),
            get_val('total_crosses'), get_val('accurate_crosses'),
            get_val('total_tackles'), get_val('interceptions'), get_val('clearances'),
            get_val('total_duels'), get_val('duels_won'), get_val('total_aerials'), get_val('aerials_won'),
            get_val('fouls_committed'), get_val('fouls_won'), get_val('yellow_cards'), get_val('red_cards'),
            get_val('corners'), get_val('offsides'),
            'fotmob', True
        )

        self.db.execute_insert(query, params)

    def import_betting_odds(self, csv_file: str = None) -> int:
        """
        Import betting odds data from all season files

        Args:
            csv_file: Single CSV filename (if None, imports all season files)

        Returns:
            Number of records imported
        """
        # If specific file provided, use it
        if csv_file:
            file_path = self.data_dir / csv_file
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return 0
            logger.info(f"Importing betting odds from {csv_file}")
            df = pd.read_csv(file_path)
        else:
            # Import from all individual season files
            import glob
            season_files = sorted(glob.glob(str(self.data_dir / "oddsportal_3liga_????-????.csv")))
            if not season_files:
                logger.error(f"No odds season files found in {self.data_dir}")
                return 0

            logger.info(f"Found {len(season_files)} odds season files")
            dfs = []
            for file in season_files:
                dfs.append(pd.read_csv(file))
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined {len(df)} total odds records from {len(season_files)} files")

        logger.info(f"Loaded {len(df)} records from {csv_file}")

        imported = 0
        skipped = 0

        for idx, row in df.iterrows():
            try:
                # Get team IDs
                home_team_id = self.team_mapper.get_team_id(row['homeTeamName'])
                away_team_id = self.team_mapper.get_team_id(row['awayTeamName'])

                if not home_team_id or not away_team_id:
                    logger.debug(f"Could not find teams: {row['homeTeamName']} vs {row['awayTeamName']}")
                    skipped += 1
                    continue

                # Find match in database by season + teams (not date)
                # This eliminates issues with "Unknown" dates (68.7% of records)
                query = """
                    SELECT match_id, home_goals, away_goals FROM matches
                    WHERE season = ? AND home_team_id = ? AND away_team_id = ?
                """
                result = self.db.execute_query(query, (row['season'], home_team_id, away_team_id))

                if not result:
                    logger.debug(f"Match not found: {row['homeTeamName']} vs {row['awayTeamName']} in {row['season']}")
                    skipped += 1
                    continue

                # If multiple matches found (unlikely in league play), use score to disambiguate
                match_id = None
                if len(result) > 1:
                    score_str = row.get('score')
                    if score_str and pd.notna(score_str) and '-' in score_str:
                        try:
                            score_parts = score_str.split('-')
                            expected_home = int(score_parts[0].strip())
                            expected_away = int(score_parts[1].strip())

                            # Find match with matching score
                            for match in result:
                                if match['home_goals'] == expected_home and match['away_goals'] == expected_away:
                                    match_id = match['match_id']
                                    break
                        except (ValueError, IndexError):
                            pass

                    if not match_id:
                        # Use first match if score doesn't help
                        match_id = result[0]['match_id']
                        logger.debug(f"Multiple matches found for {row['homeTeamName']} vs {row['awayTeamName']}, using first")
                else:
                    match_id = result[0]['match_id']

                # Calculate implied probabilities
                odds_home = float(row['odds_home']) if pd.notna(row['odds_home']) else None
                odds_draw = float(row['odds_draw']) if pd.notna(row['odds_draw']) else None
                odds_away = float(row['odds_away']) if pd.notna(row['odds_away']) else None

                prob_home = 1 / odds_home if odds_home and odds_home > 0 else None
                prob_draw = 1 / odds_draw if odds_draw and odds_draw > 0 else None
                prob_away = 1 / odds_away if odds_away and odds_away > 0 else None

                # Insert odds
                insert_query = """
                    INSERT OR REPLACE INTO betting_odds
                    (match_id, bookmaker, odds_home, odds_draw, odds_away,
                     implied_prob_home, implied_prob_draw, implied_prob_away, odds_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                self.db.execute_insert(insert_query, (
                    match_id, 'oddsportal_avg',
                    odds_home, odds_draw, odds_away,
                    prob_home, prob_draw, prob_away,
                    'closing'
                ))

                imported += 1

                if (imported + skipped) % 100 == 0:
                    logger.info(f"Progress: {imported} imported, {skipped} skipped")

            except Exception as e:
                logger.error(f"Error importing odds row {idx}: {e}")
                skipped += 1

        logger.success(f"Odds import complete: {imported} imported, {skipped} skipped")
        return imported

    def import_all(self):
        """Import all existing CSV data"""
        logger.info("=== Starting import of all existing data ===")

        start_time = datetime.now()

        # Import FotMob stats
        fotmob_count = self.import_fotmob_statistics()

        # Import betting odds
        odds_count = self.import_betting_odds()

        duration = (datetime.now() - start_time).total_seconds()

        logger.success(f"""
=== Data import complete ===
Duration: {duration:.1f}s
FotMob statistics: {fotmob_count} records
Betting odds: {odds_count} records
        """)

        # Log to database
        self.db.log_collection(
            source='csv_import',
            collection_type='existing_data',
            status='success',
            records_collected=fotmob_count + odds_count,
            started_at=start_time
        )


def main():
    """Main execution"""
    importer = DataImporter()
    importer.import_all()

    # Print database stats
    stats = importer.db.get_database_stats()
    print("\n=== Database Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
