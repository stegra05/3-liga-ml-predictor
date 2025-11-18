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
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from liga_predictor.models import Base


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
        
        # Create SQLAlchemy engine with optimized settings
        db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(
            db_url,
            poolclass=StaticPool,
            connect_args={
                "check_same_thread": False,
                "timeout": 20
            },
            echo=False
        )
        
        # Configure SQLite pragmas via event listeners
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        
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

    def get_session(self) -> Session:
        """
        Get SQLAlchemy session for ORM operations

        Returns:
            SQLAlchemy session object
        """
        return self.SessionLocal()

    def initialize_schema_with_alembic(self) -> None:
        """
        Initialize database schema using Alembic migrations (RECOMMENDED)

        This is the preferred method for database initialization as it:
        - Tracks schema versions
        - Allows incremental migrations
        - Maintains consistency with ORM models
        """
        try:
            from alembic.config import Config
            from alembic import command

            logger.info("Initializing database schema using Alembic migrations...")

            # Load Alembic config
            alembic_cfg = Config("alembic.ini")

            # Run migrations to head
            command.upgrade(alembic_cfg, "head")

            logger.success("✓ Database schema initialized successfully via Alembic")

        except ImportError:
            logger.error("Alembic not installed. Install with: pip install alembic")
            raise
        except Exception as e:
            logger.error(f"Error running Alembic migrations: {e}")
            raise

    def initialize_schema_legacy(self, schema_file: str = "database/schema.sql") -> None:
        """
        Initialize database with schema from SQL file (LEGACY METHOD)

        DEPRECATED: Use initialize_schema_with_alembic() instead.
        This method is kept for backwards compatibility but bypasses
        Alembic migrations, which can lead to schema drift.

        Args:
            schema_file: Path to schema SQL file
        """
        logger.warning("Using legacy schema initialization. Consider using Alembic migrations instead.")

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
        except sqlite3.Error as e:
            logger.error(f"Error initializing schema: {e}")
            raise
        finally:
            conn.close()

    # Alias for backwards compatibility
    initialize_schema = initialize_schema_with_alembic

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

    def get_or_create_team(self, team_name: str, openligadb_id: Optional[int] = None) -> int:
        """
        Get team ID or create if doesn't exist (using ORM)

        Args:
            team_name: Team name
            openligadb_id: OpenLigaDB team ID

        Returns:
            Team ID
        """
        from liga_predictor.models import Team

        session = self.get_session()
        try:
            team = session.query(Team).filter(
                (Team.team_name == team_name) | (Team.openligadb_id == openligadb_id)
            ).first()

            if not team:
                team = Team(team_name=team_name, openligadb_id=openligadb_id)
                session.add(team)
                session.flush()
                logger.info(f"Created new team: {team_name} (ID: {team.team_id})")

            return team.team_id
        finally:
            session.commit()
            session.close()

    def merge_or_create(self, model_class, filter_dict: Dict[str, Any], defaults: Dict[str, Any]):
        """
        Helper to merge (update if exists, create if not) using SQLAlchemy merge()
        This reduces verbose if/else patterns in collection scripts
        
        Args:
            model_class: SQLAlchemy model class
            filter_dict: Dictionary of filters to find existing record (empty = insert only)
            defaults: Dictionary of values to set (for both create and update)
            
        Returns:
            The model instance (existing or newly created)
        """
        session = self.get_session()
        try:
            existing = None
            if filter_dict:
                # Build filter query
                query = session.query(model_class)
                for key, value in filter_dict.items():
                    query = query.filter(getattr(model_class, key) == value)
                existing = query.first()
            
            if existing:
                # Update existing
                for key, value in defaults.items():
                    setattr(existing, key, value)
                session.commit()
                return existing
            else:
                # Create new
                new_obj = model_class(**{**filter_dict, **defaults})
                session.add(new_obj)
                session.flush()
                session.commit()
                return new_obj
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def get_team_id_by_name(self, team_name: str) -> Optional[int]:
        """
        Get team ID by name (fuzzy matching supported)

        Args:
            team_name: Team name to search for

        Returns:
            Team ID or None if not found
        """
        from liga_predictor.models import Team

        session = self.get_session()
        try:
            # Exact match
            team = session.query(Team).filter(
                (Team.team_name == team_name) | (Team.team_name_alt == team_name)
            ).first()

            if team:
                return team.team_id

            # Fuzzy match (contains) - use raw SQL for LIKE
            result = self.execute_query(
                "SELECT team_id FROM teams WHERE team_name LIKE ? LIMIT 1",
                (f"%{team_name}%",)
            )

            return result[0]['team_id'] if result else None
        finally:
            session.close()

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
        from liga_predictor.models import CollectionLog

        started = started_at or datetime.now()
        completed = datetime.now() if status != 'started' else None
        duration = (completed - started).total_seconds() if completed else None

        session = self.get_session()
        try:
            log_entry = CollectionLog(
                source=source,
                collection_type=collection_type,
                season=season,
                matchday=matchday,
                status=status,
                records_collected=records_collected,
                error_message=error_message,
                started_at=started,
                completed_at=completed,
                duration_seconds=duration
            )
            session.add(log_entry)
            session.commit()
            return log_entry.log_id
        finally:
            session.close()

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
