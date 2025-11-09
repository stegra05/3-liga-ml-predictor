"""
Cleanup Script for Duplicate Betting Odds
Removes duplicate betting odds records, keeping only the most recent record
for each (match_id, bookmaker) combination.
"""

import sys
from pathlib import Path
from typing import Dict, Any
from loguru import logger

# Add parent directory to path to import database module
sys.path.insert(0, str(Path(__file__).parent.parent))
from database.db_manager import DatabaseManager


class DuplicateOddsCleaner:
    """Handles cleanup of duplicate betting odds records"""

    def __init__(self, db_path: str = "database/3liga.db"):
        """
        Initialize the cleaner

        Args:
            db_path: Path to SQLite database file
        """
        self.db = DatabaseManager(db_path)
        self.stats_before: Dict[str, Any] = {}
        self.stats_after: Dict[str, Any] = {}

    def get_current_stats(self) -> Dict[str, Any]:
        """
        Get current statistics about betting odds

        Returns:
            Dictionary with statistics
        """
        conn = self.db.get_connection()
        try:
            stats = {}

            # Total records
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM betting_odds")
            stats['total_records'] = cursor.fetchone()['count']

            # Count duplicate groups (match_id, bookmaker combinations with >1 record)
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM (
                    SELECT match_id, bookmaker
                    FROM betting_odds
                    GROUP BY match_id, bookmaker
                    HAVING COUNT(*) > 1
                )
            """)
            stats['duplicate_groups'] = cursor.fetchone()['count']

            # Count total duplicate records (all records in duplicate groups minus one per group)
            cursor.execute("""
                SELECT 
                    COUNT(*) - COUNT(DISTINCT match_id || '|' || bookmaker) as duplicate_count
                FROM betting_odds
            """)
            stats['duplicate_records'] = cursor.fetchone()['duplicate_count']

            # Count unique (match_id, bookmaker) combinations
            cursor.execute("""
                SELECT COUNT(DISTINCT match_id || '|' || bookmaker) as count
                FROM betting_odds
            """)
            stats['unique_combinations'] = cursor.fetchone()['count']

            # Count matches with odds
            cursor.execute("""
                SELECT COUNT(DISTINCT match_id) as count
                FROM betting_odds
            """)
            stats['matches_with_odds'] = cursor.fetchone()['count']

            return stats

        finally:
            conn.close()

    def validate_before_cleanup(self) -> bool:
        """
        Validate current state before cleanup

        Returns:
            True if validation passes, False otherwise
        """
        logger.info("Validating current database state...")
        self.stats_before = self.get_current_stats()

        logger.info(f"Current betting_odds statistics:")
        logger.info(f"  Total records: {self.stats_before['total_records']:,}")
        logger.info(f"  Unique (match_id, bookmaker) combinations: {self.stats_before['unique_combinations']:,}")
        logger.info(f"  Duplicate groups: {self.stats_before['duplicate_groups']:,}")
        logger.info(f"  Estimated duplicate records: {self.stats_before['duplicate_records']:,}")
        logger.info(f"  Matches with odds: {self.stats_before['matches_with_odds']:,}")

        if self.stats_before['duplicate_groups'] == 0:
            logger.success("No duplicates found! Database is already clean.")
            return False

        if self.stats_before['duplicate_records'] <= 0:
            logger.warning("Duplicate count is zero or negative. Something may be wrong.")
            return False

        return True

    def cleanup_duplicates(self, dry_run: bool = False) -> int:
        """
        Remove duplicate betting odds, keeping most recent record per (match_id, bookmaker)

        Args:
            dry_run: If True, only show what would be deleted without actually deleting

        Returns:
            Number of records that would be/were deleted
        """
        conn = self.db.get_connection()
        try:
            # Use ROW_NUMBER() window function to identify records to keep
            # Keep the most recent record (highest created_at, or highest odds_id as fallback)
            # SQLite: NULL values sort last in DESC order, so we use COALESCE for explicit handling
            delete_query = """
                DELETE FROM betting_odds
                WHERE odds_id NOT IN (
                    SELECT odds_id FROM (
                        SELECT odds_id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY match_id, bookmaker
                                   ORDER BY 
                                       COALESCE(created_at, '1970-01-01') DESC,
                                       odds_id DESC
                               ) as rn
                        FROM betting_odds
                    ) WHERE rn = 1
                )
            """

            if dry_run:
                # In dry-run mode, count what would be deleted
                count_query = """
                    SELECT COUNT(*) as count
                    FROM betting_odds
                    WHERE odds_id NOT IN (
                        SELECT odds_id FROM (
                            SELECT odds_id,
                                   ROW_NUMBER() OVER (
                                       PARTITION BY match_id, bookmaker
                                       ORDER BY 
                                           COALESCE(created_at, '1970-01-01') DESC,
                                           odds_id DESC
                                   ) as rn
                            FROM betting_odds
                        ) WHERE rn = 1
                    )
                """
                cursor = conn.cursor()
                cursor.execute(count_query)
                count = cursor.fetchone()['count']
                logger.info(f"[DRY RUN] Would delete {count:,} duplicate records")
                return count
            else:
                # Actually delete duplicates
                cursor = conn.cursor()
                cursor.execute(delete_query)
                deleted_count = cursor.rowcount
                conn.commit()
                logger.success(f"Deleted {deleted_count:,} duplicate records")
                return deleted_count

        except Exception as e:
            if not dry_run:
                conn.rollback()
                logger.error(f"Error during cleanup: {e}")
                raise
            else:
                raise
        finally:
            conn.close()

    def validate_after_cleanup(self) -> bool:
        """
        Validate cleanup results

        Returns:
            True if validation passes, False otherwise
        """
        logger.info("Validating cleanup results...")
        self.stats_after = self.get_current_stats()

        logger.info(f"After cleanup betting_odds statistics:")
        logger.info(f"  Total records: {self.stats_after['total_records']:,}")
        logger.info(f"  Unique (match_id, bookmaker) combinations: {self.stats_after['unique_combinations']:,}")
        logger.info(f"  Duplicate groups: {self.stats_after['duplicate_groups']:,}")
        logger.info(f"  Matches with odds: {self.stats_after['matches_with_odds']:,}")

        # Validation checks
        issues = []

        # Check: No duplicates should remain
        if self.stats_after['duplicate_groups'] > 0:
            issues.append(f"Still have {self.stats_after['duplicate_groups']} duplicate groups")

        # Check: Should have same number of unique combinations
        if self.stats_after['unique_combinations'] != self.stats_before['unique_combinations']:
            issues.append(
                f"Unique combinations changed: {self.stats_before['unique_combinations']} -> "
                f"{self.stats_after['unique_combinations']}"
            )

        # Check: Should have same number of matches with odds
        if self.stats_after['matches_with_odds'] != self.stats_before['matches_with_odds']:
            issues.append(
                f"Matches with odds changed: {self.stats_before['matches_with_odds']} -> "
                f"{self.stats_after['matches_with_odds']}"
            )

        # Check: Records deleted should match expected
        expected_deleted = self.stats_before['duplicate_records']
        actual_deleted = self.stats_before['total_records'] - self.stats_after['total_records']
        if abs(actual_deleted - expected_deleted) > 1:  # Allow 1 record difference for rounding
            issues.append(
                f"Deletion count mismatch: expected ~{expected_deleted}, actual {actual_deleted}"
            )

        if issues:
            logger.error("Validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        else:
            logger.success("Validation passed! Cleanup successful.")
            logger.info(f"Removed {actual_deleted:,} duplicate records")
            logger.info(f"Database size reduced from {self.stats_before['total_records']:,} to "
                       f"{self.stats_after['total_records']:,} records")
            return True

    def add_unique_constraint(self, dry_run: bool = False) -> bool:
        """
        Add UNIQUE constraint to prevent future duplicates

        Args:
            dry_run: If True, only show what would be created

        Returns:
            True if constraint was added successfully
        """
        conn = self.db.get_connection()
        try:
            # Check if constraint already exists
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND name='idx_betting_odds_unique'
            """)
            exists = cursor.fetchone() is not None

            if exists:
                logger.info("UNIQUE constraint already exists")
                return True

            if dry_run:
                logger.info("[DRY RUN] Would create UNIQUE index: idx_betting_odds_unique")
                return True

            # Create unique index
            create_index_query = """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_betting_odds_unique
                ON betting_odds(match_id, bookmaker)
            """
            cursor.execute(create_index_query)
            conn.commit()
            logger.success("Created UNIQUE constraint on (match_id, bookmaker)")
            return True

        except Exception as e:
            if not dry_run:
                conn.rollback()
                logger.error(f"Error creating constraint: {e}")
                raise
            else:
                raise
        finally:
            conn.close()

    def run_cleanup(self, dry_run: bool = False, add_constraint: bool = True) -> bool:
        """
        Run the complete cleanup process

        Args:
            dry_run: If True, only show what would be done without making changes
            add_constraint: If True, add UNIQUE constraint after cleanup

        Returns:
            True if cleanup was successful
        """
        logger.info("=" * 80)
        logger.info("DUPLICATE ODDS CLEANUP")
        logger.info("=" * 80)

        if dry_run:
            logger.warning("DRY RUN MODE - No changes will be made to the database")
        else:
            logger.warning("⚠️  WARNING: This will permanently delete duplicate records!")
            logger.warning("⚠️  Make sure you have backed up the database before proceeding!")
            logger.info("")

        # Step 1: Validate before cleanup
        if not self.validate_before_cleanup():
            return False

        # Step 2: Cleanup duplicates
        logger.info("")
        logger.info("Step 1: Removing duplicate records...")
        deleted_count = self.cleanup_duplicates(dry_run=dry_run)

        if dry_run:
            logger.info(f"[DRY RUN] Would delete {deleted_count:,} records")
            logger.info("")
            logger.info("To actually perform the cleanup, run without --dry-run flag")
            return True

        # Step 3: Validate after cleanup
        logger.info("")
        logger.info("Step 2: Validating cleanup results...")
        if not self.validate_after_cleanup():
            return False

        # Step 4: Add unique constraint (optional)
        if add_constraint:
            logger.info("")
            logger.info("Step 3: Adding UNIQUE constraint to prevent future duplicates...")
            self.add_unique_constraint(dry_run=False)

        logger.info("")
        logger.success("=" * 80)
        logger.success("CLEANUP COMPLETE")
        logger.success("=" * 80)

        return True


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean up duplicate betting odds records",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview changes without making them)
  python scripts/cleanup_duplicate_odds.py --dry-run

  # Actually perform cleanup
  python scripts/cleanup_duplicate_odds.py

  # Cleanup without adding unique constraint
  python scripts/cleanup_duplicate_odds.py --no-constraint
        """
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default='database/3liga.db',
        help='Path to database file (default: database/3liga.db)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without making them'
    )
    parser.add_argument(
        '--no-constraint',
        action='store_true',
        help='Skip adding UNIQUE constraint after cleanup'
    )

    args = parser.parse_args()

    # Check if database exists
    db_path = Path(args.db_path)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.error("Please provide a valid database path")
        sys.exit(1)

    # Run cleanup
    cleaner = DuplicateOddsCleaner(db_path=str(db_path))
    success = cleaner.run_cleanup(
        dry_run=args.dry_run,
        add_constraint=not args.no_constraint
    )

    if not success:
        logger.error("Cleanup failed or was skipped")
        sys.exit(1)


if __name__ == "__main__":
    main()

