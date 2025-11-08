"""
Import Existing CSV Data into Database
Imports FotMob statistics and betting odds from existing CSV files
"""

import pandas as pd
from pathlib import Path
from loguru import logger
import sys
from datetime import datetime, timedelta

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

    def _find_team_id_alternative(self, team_name: str):
        """
        Try alternative methods to find team ID when standard lookup fails
        
        Args:
            team_name: Team name from CSV
            
        Returns:
            Team ID or None
        """
        # Known mappings for common CSV name variations
        known_mappings = {
            "Aue": "Erzgebirge Aue",
            "Bayern II": "FC Bayern München II",
            "Braunschweig": "Eintracht Braunschweig",
            "Burghausen": "Wacker Burghausen",
            "Dortmund II": "Borussia Dortmund II",
            "Erfurt": "FC Rot-Weiß Erfurt",
            "Heidenheim": "1. FC Heidenheim 1846",
            "Ingolstadt": "FC Ingolstadt 04",
            "Jena": "FC Carl Zeiss Jena",
            "Offenbach": "Kickers Offenbach",
            "Regensburg": "Jahn Regensburg",
            "SG Dynamo Dresden": "Dynamo Dresden",
            "Sandhausen": "SV Sandhausen",
            "Stuttgart II": "VfB Stuttgart II",
            "Unterhaching": "SpVgg Unterhaching",
            "VfL Osnabruck": "VfL Osnabrück",  # Handle umlaut
            "VfL Osnabrück": "VfL Osnabrück",
            "Wehen": "SV Wehen Wiesbaden",
            "Wuppertal": "Wuppertaler SV",
            # Additional mappings for newer seasons
            "Viktoria Koln": "Viktoria Köln",
            "Viktoria Köln": "Viktoria Köln",
            "RW Essen": "Rot-Weiss Essen",
            "Rot-Weiss Essen": "Rot-Weiss Essen",
            "Preussen Munster": "Preußen Münster",
            "Preussen Münster": "Preußen Münster",
            "Preußen Munster": "Preußen Münster",
            "Wurzburger Kickers": "FC Kickers Würzburg",
            "Würzburger Kickers": "FC Kickers Würzburg",
            "Viktoria Berlin": "FC Viktoria 1889 Berlin",
            "Grossaspach": "SG Sonnenhof Großaspach",
            "Großaspach": "SG Sonnenhof Großaspach",
            "Saarbrucken": "1. FC Saarbrücken",
            "Saarbrücken": "1. FC Saarbrücken",
            "Kaiserslautern": "1. FC Kaiserslautern",
        }
        
        # Check known mappings first
        if team_name in known_mappings:
            mapped_name = known_mappings[team_name]
            query = "SELECT team_id FROM teams WHERE team_name = ? LIMIT 1"
            result = self.db.execute_query(query, (mapped_name,))
            if result:
                logger.debug(f"Mapped '{team_name}' -> '{mapped_name}'")
                return result[0]['team_id']
        
        # Try direct database lookup (case-insensitive, partial match)
        query = """
            SELECT team_id FROM teams 
            WHERE LOWER(team_name) = LOWER(?)
               OR LOWER(team_name) LIKE LOWER(?)
               OR LOWER(?) LIKE LOWER('%' || team_name || '%')
            LIMIT 1
        """
        result = self.db.execute_query(query, (team_name, f"%{team_name}%", team_name))
        if result:
            return result[0]['team_id']
        
        # Try with common name variations
        variations = [
            team_name.replace(" II", "").replace(" 2", "").strip(),
            team_name.replace("SG ", "").replace("FC ", "").replace("TSV ", "").replace("SV ", "").strip(),
            team_name.replace(" 1860", "").strip(),
            team_name.replace("Osnabruck", "Osnabrück"),  # Handle umlaut
        ]
        
        for variant in variations:
            if variant and variant != team_name:
                result = self.db.execute_query(query, (variant, f"%{variant}%", variant))
                if result:
                    logger.debug(f"Found team '{team_name}' via variant '{variant}'")
                    return result[0]['team_id']
        
        return None

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

                # If team not found, try alternative lookup methods
                if not home_team_id:
                    # Try direct database lookup with fuzzy matching
                    home_team_id = self._find_team_id_alternative(row['homeTeamName'])
                    if not home_team_id:
                        logger.warning(f"Could not find home team: {row['homeTeamName']} (season: {row.get('season', 'unknown')})")
                        skipped += 1
                        continue

                if not away_team_id:
                    # Try direct database lookup with fuzzy matching
                    away_team_id = self._find_team_id_alternative(row['awayTeamName'])
                    if not away_team_id:
                        logger.warning(f"Could not find away team: {row['awayTeamName']} (season: {row.get('season', 'unknown')})")
                        skipped += 1
                        continue

                # Find match in database by season + teams
                # Try to use date if available, otherwise match by season + teams + score
                match_datetime = row.get('matchDateTime', '')
                use_date = match_datetime and pd.notna(match_datetime) and 'Unknown' not in str(match_datetime)
                
                if use_date:
                    # Try to parse date and match by date range (±1 day tolerance)
                    try:
                        match_date = pd.to_datetime(match_datetime)
                        date_start = (match_date - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
                        date_end = (match_date + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
                        
                        query = """
                            SELECT match_id, home_goals, away_goals, matchday, match_datetime 
                            FROM matches
                            WHERE season = ? AND home_team_id = ? AND away_team_id = ?
                              AND match_datetime BETWEEN ? AND ?
                        """
                        result = self.db.execute_query(query, (
                            row['season'], home_team_id, away_team_id, date_start, date_end
                        ))
                    except Exception as e:
                        logger.debug(f"Could not parse date '{match_datetime}': {e}")
                        use_date = False
                
                if not use_date or not result:
                    # Fallback: match by season + teams only
                    query = """
                        SELECT match_id, home_goals, away_goals, matchday, match_datetime 
                        FROM matches
                        WHERE season = ? AND home_team_id = ? AND away_team_id = ?
                    """
                    result = self.db.execute_query(query, (row['season'], home_team_id, away_team_id))

                if not result:
                    logger.warning(f"Match not found: {row['homeTeamName']} vs {row['awayTeamName']} in {row['season']}")
                    skipped += 1
                    continue

                # If multiple matches found, use score to disambiguate
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
                                    logger.debug(f"Matched by score: {row['homeTeamName']} vs {row['awayTeamName']} ({expected_home}-{expected_away})")
                                    break
                        except (ValueError, IndexError):
                            pass

                    if not match_id:
                        # Use first match if score doesn't help
                        match_id = result[0]['match_id']
                        logger.warning(f"Multiple matches found for {row['homeTeamName']} vs {row['awayTeamName']} in {row['season']}, using first (matchday {result[0].get('matchday', 'unknown')})")
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
