"""
Database Manager for 3. Liga Dataset
Handles database initialization, connections, and basic operations
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
from loguru import logger


class DatabaseManager:
    """Manages SQLite database connections and operations"""

    def __init__(self, db_path: str = "database/3liga.db"):
        """
        Initialize database manager

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Database manager initialized: {self.db_path}")

    def get_connection(self) -> sqlite3.Connection:
        """
        Get database connection with optimized settings

        Returns:
            SQLite connection object
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable column access by name
        # Optimize SQLite for better performance
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("PRAGMA foreign_keys=ON")  # Enable foreign key constraints
        return conn

    def initialize_schema(self, schema_file: str = "database/schema.sql") -> None:
        """
        Initialize database with schema from SQL file

        Args:
            schema_file: Path to schema SQL file
        """
        schema_path = Path(schema_file)
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")

        logger.info(f"Initializing database schema from {schema_file}")

        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()

        conn = self.get_connection()
        try:
            # Execute schema (handle multiple statements)
            conn.executescript(schema_sql)
            conn.commit()
            logger.success("Database schema initialized successfully")

            # --- Schema Migrations ---
            # Add 'updated_at' column to match_statistics if it doesn't exist
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(match_statistics)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'updated_at' not in columns:
                logger.info("Adding 'updated_at' column to 'match_statistics' table...")
                cursor.execute("ALTER TABLE match_statistics ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
                conn.commit()
                logger.success("'updated_at' column added to 'match_statistics' table.")

            # Add weather_source and weather_confidence columns to matches if they don't exist
            cursor.execute("PRAGMA table_info(matches)")
            match_columns = [col[1] for col in cursor.fetchall()]
            if 'weather_source' not in match_columns:
                logger.info("Adding 'weather_source' column to 'matches' table...")
                cursor.execute("ALTER TABLE matches ADD COLUMN weather_source TEXT")
                conn.commit()
                logger.success("'weather_source' column added to 'matches' table.")
            if 'weather_confidence' not in match_columns:
                logger.info("Adding 'weather_confidence' column to 'matches' table...")
                cursor.execute("ALTER TABLE matches ADD COLUMN weather_confidence REAL")
                conn.commit()
                logger.success("'weather_confidence' column added to 'matches' table.")

            # Add fotmob_match_id column to matches if it doesn't exist
            if 'fotmob_match_id' not in match_columns:
                logger.info("Adding 'fotmob_match_id' column to 'matches' table...")
                cursor.execute("ALTER TABLE matches ADD COLUMN fotmob_match_id INTEGER")
                conn.commit()
                logger.success("'fotmob_match_id' column added to 'matches' table.")
            # --- End Schema Migrations ---

        except sqlite3.Error as e:
            logger.error(f"Error initializing schema: {e}")
            raise
        finally:
            conn.close()

    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[sqlite3.Row]:
        """
        Execute SELECT query and return results

        Args:
            query: SQL query string
            params: Query parameters (for parameterized queries)

        Returns:
            List of result rows
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            results = cursor.fetchall()
            return results
        finally:
            conn.close()

    def execute_insert(self, query: str, params: tuple) -> int:
        """
        Execute INSERT query and return last row ID

        Args:
            query: SQL INSERT query
            params: Query parameters

        Returns:
            ID of inserted row
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def execute_many(self, query: str, params_list: List[tuple]) -> None:
        """
        Execute query with multiple parameter sets (bulk insert)

        Args:
            query: SQL query
            params_list: List of parameter tuples
        """
        if not params_list:
            return

        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            logger.info(f"Bulk insert: {len(params_list)} rows affected")
        finally:
            conn.close()

    def query_to_dataframe(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """
        Execute query and return results as pandas DataFrame

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            DataFrame with query results
        """
        conn = self.get_connection()
        try:
            df = pd.read_sql_query(query, conn, params=params)
            return df
        finally:
            conn.close()

    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> int:
        """
        Insert DataFrame into database table

        Args:
            df: pandas DataFrame
            table_name: Target table name
            if_exists: How to behave if table exists ('fail', 'replace', 'append')

        Returns:
            Number of rows inserted
        """
        conn = self.get_connection()
        try:
            rows = df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            logger.info(f"Inserted {rows} rows into {table_name}")
            return rows if rows else 0
        finally:
            conn.close()

    def log_collection(self, source: str, collection_type: str, season: Optional[str] = None,
                      matchday: Optional[int] = None, status: str = 'started',
                      records_collected: int = 0, error_message: Optional[str] = None,
                      started_at: Optional[datetime] = None) -> int:
        """
        Log data collection activity

        Args:
            source: Data source name (e.g., 'openligadb')
            collection_type: Type of collection (e.g., 'match_data')
            season: Season being collected
            matchday: Matchday being collected
            status: Collection status ('started', 'success', 'partial', 'failed')
            records_collected: Number of records collected
            error_message: Error message if failed
            started_at: Start timestamp

        Returns:
            Log entry ID
        """
        query = """
            INSERT INTO collection_logs
            (source, collection_type, season, matchday, status, records_collected,
             error_message, started_at, completed_at, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        started = started_at or datetime.now()
        completed = datetime.now() if status != 'started' else None
        duration = (completed - started).total_seconds() if completed else None

        params = (
            source, collection_type, season, matchday, status,
            records_collected, error_message, started,
            completed, duration
        )

        return self.execute_insert(query, params)

    def get_or_create_team(self, team_name: str, openligadb_id: Optional[int] = None) -> int:
        """
        Get team ID or create if doesn't exist

        Args:
            team_name: Team name
            openligadb_id: OpenLigaDB team ID

        Returns:
            Team ID
        """
        # Try to find existing team
        query = """
            SELECT team_id FROM teams
            WHERE team_name = ? OR openligadb_id = ?
            LIMIT 1
        """
        result = self.execute_query(query, (team_name, openligadb_id))

        if result:
            return result[0]['team_id']

        # Create new team
        insert_query = """
            INSERT INTO teams (team_name, openligadb_id)
            VALUES (?, ?)
        """
        team_id = self.execute_insert(insert_query, (team_name, openligadb_id))
        logger.info(f"Created new team: {team_name} (ID: {team_id})")
        return team_id

    def get_team_id_by_name(self, team_name: str) -> Optional[int]:
        """
        Get team ID by name (fuzzy matching supported)

        Args:
            team_name: Team name to search for

        Returns:
            Team ID or None if not found
        """
        # Exact match
        query = "SELECT team_id FROM teams WHERE team_name = ? OR team_name_alt = ? LIMIT 1"
        result = self.execute_query(query, (team_name, team_name))

        if result:
            return result[0]['team_id']

        # Fuzzy match (contains)
        query = "SELECT team_id FROM teams WHERE team_name LIKE ? LIMIT 1"
        result = self.execute_query(query, (f"%{team_name}%",))

        return result[0]['team_id'] if result else None

    def get_match_id(self, season: str, home_team_id: int, away_team_id: int,
                    match_datetime: datetime) -> Optional[int]:
        """
        Get match ID for existing match

        Args:
            season: Season string (e.g., "2024-2025")
            home_team_id: Home team ID
            away_team_id: Away team ID
            match_datetime: Match date and time

        Returns:
            Match ID or None if not found
        """
        query = """
            SELECT match_id FROM matches
            WHERE season = ? AND home_team_id = ? AND away_team_id = ?
            AND date(match_datetime) = date(?)
            LIMIT 1
        """
        result = self.execute_query(query, (season, home_team_id, away_team_id, match_datetime))
        return result[0]['match_id'] if result else None

    def get_or_create_player(self, full_name: str, date_of_birth: Optional[str] = None,
                             nationality: Optional[str] = None, position: Optional[str] = None) -> int:
        """
        Get existing player ID or create new player

        Args:
            full_name: Player's full name
            date_of_birth: Date of birth (optional)
            nationality: Nationality (optional)
            position: Position (optional)

        Returns:
            Player ID
        """
        # Check if player exists
        query = "SELECT player_id FROM players WHERE full_name = ? LIMIT 1"
        result = self.execute_query(query, (full_name,))

        if result:
            return result[0]['player_id']

        # Create new player
        insert_query = """
            INSERT INTO players (full_name, date_of_birth, nationality, position)
            VALUES (?, ?, ?, ?)
        """
        player_id = self.execute_insert(insert_query, (full_name, date_of_birth, nationality, position))
        logger.debug(f"Created new player: {full_name} (ID: {player_id})")
        return player_id

    def insert_player_season_stats(self, player_id: int, team_id: int, season: str,
                                   stats: Dict[str, Any], source: str = 'fbref') -> None:
        """
        Insert or update player season statistics

        Args:
            player_id: Player ID
            team_id: Team ID
            season: Season string
            stats: Dictionary of statistics
            source: Data source
        """
        query = """
            INSERT OR REPLACE INTO player_season_stats
            (player_id, team_id, season, matches_played, minutes_played, starts,
             goals, assists, penalties_scored, yellow_cards, red_cards,
             shots_total, shots_on_target, pass_accuracy_percent, tackles_won, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            player_id,
            team_id,
            season,
            stats.get('matches_played', 0),
            stats.get('minutes_played', 0),
            stats.get('starts', 0),
            stats.get('goals', 0),
            stats.get('assists', 0),
            stats.get('penalties_scored', 0),
            stats.get('yellow_cards', 0),
            stats.get('red_cards', 0),
            stats.get('shots_total'),
            stats.get('shots_on_target'),
            stats.get('pass_accuracy_percent'),
            stats.get('tackles_won'),
            source
        )

        self.execute_insert(query, params)

    def insert_league_standing(self, season: str, matchday: int, team_id: int,
                               standing_data: Dict[str, Any]) -> None:
        """
        Insert league standing for a team

        Args:
            season: Season string
            matchday: Matchday number
            team_id: Team ID
            standing_data: Dictionary with standing information
        """
        query = """
            INSERT OR REPLACE INTO league_standings
            (season, matchday, team_id, position, matches_played, wins, draws, losses,
             goals_for, goals_against, goal_difference, points)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            season,
            matchday,
            team_id,
            standing_data.get('position', 0),
            standing_data.get('matches_played', 0),
            standing_data.get('wins', 0),
            standing_data.get('draws', 0),
            standing_data.get('losses', 0),
            standing_data.get('goals_for', 0),
            standing_data.get('goals_against', 0),
            standing_data.get('goal_difference', 0),
            standing_data.get('points', 0)
        )

        self.execute_insert(query, params)

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dictionary with table row counts and other stats
        """
        tables = [
            'teams', 'matches', 'match_statistics', 'match_events',
            'league_standings', 'betting_odds', 'players',
            'player_season_stats', 'team_ratings', 'collection_logs'
        ]

        stats = {}
        conn = self.get_connection()

        try:
            for table in tables:
                cursor = conn.cursor()
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                result = cursor.fetchone()
                stats[table] = result['count']

            # Additional useful stats
            cursor.execute("SELECT COUNT(DISTINCT season) as seasons FROM matches")
            stats['unique_seasons'] = cursor.fetchone()['seasons']

            cursor.execute("SELECT MIN(match_datetime) as first, MAX(match_datetime) as last FROM matches")
            dates = cursor.fetchone()
            stats['date_range'] = {
                'first': dates['first'],
                'last': dates['last']
            }

            return stats

        finally:
            conn.close()

    def vacuum(self) -> None:
        """Optimize database (VACUUM command)"""
        conn = self.get_connection()
        try:
            logger.info("Running VACUUM to optimize database...")
            conn.execute("VACUUM")
            logger.success("Database optimized")
        finally:
            conn.close()


# Convenience function to get database manager instance
_db_instance: Optional[DatabaseManager] = None


def get_db(db_path: str = "database/3liga.db") -> DatabaseManager:
    """
    Get singleton database manager instance

    Args:
        db_path: Path to database file

    Returns:
        DatabaseManager instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager(db_path)
    return _db_instance


def main():
    """Main function for database initialization"""
    # Initialize database when run as script
    db = DatabaseManager()
    db.initialize_schema()
    print("Database initialized successfully!")

    # Print stats
    stats = db.get_database_stats()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
