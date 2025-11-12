"""
Team Unification Script
Merges duplicate team records to a single canonical team and updates references.
"""

from loguru import logger
from pathlib import Path
from typing import Dict, List, Tuple

from liga_predictor.database import get_db


# Define canonicalization map: old_name -> canonical_name
CANONICAL_MAP: Dict[str, str] = {
    # Fortuna Köln variants
    "SC Fortuna Köln": "Fortuna Köln",
    # Würzburg variants
    "FC Kickers Würzburg": "Würzburger Kickers",
}


class TeamUnifier:
    """Unifies duplicate team records"""

    def __init__(self):
        self.db = get_db()

    def get_team_id_by_name(self, name: str) -> int:
        """Get team ID by team name"""
        res = self.db.execute_query("SELECT team_id FROM teams WHERE team_name = ? LIMIT 1", (name,))
        return res[0]["team_id"] if res else None

    def unify_team(self, old_name: str, canonical_name: str) -> Tuple[int, int]:
        """
        Update references from old team to canonical team.
        Returns (old_id, canonical_id)
        """
        old_id = self.get_team_id_by_name(old_name)
        if old_id is None:
            logger.info(f"Old team not found, skipping: {old_name}")
            return None, None

        canonical_id = self.get_team_id_by_name(canonical_name)
        if canonical_id is None:
            # Rename old team to canonical if canonical doesn't exist
            logger.info(f"Canonical team not found; renaming '{old_name}' to '{canonical_name}'")
            self.db.execute_insert("UPDATE teams SET team_name = ? WHERE team_id = ?", (canonical_name, old_id))
            return old_id, old_id

        if old_id == canonical_id:
            logger.info(f"Team already canonical: {canonical_name}")
            return old_id, canonical_id

        logger.info(f"Merging '{old_name}' (id={old_id}) -> '{canonical_name}' (id={canonical_id})")

        # Update references across tables
        updates: List[Tuple[str, str]] = [
            ("matches", "home_team_id"),
            ("matches", "away_team_id"),
            ("match_statistics", "team_id"),
            ("league_standings", "team_id"),
            ("team_ratings", "team_id"),
            ("head_to_head", "team_a_id"),
            ("head_to_head", "team_b_id"),
        ]
        for table, col in updates:
            self.db.execute_insert(f"UPDATE {table} SET {col} = ? WHERE {col} = ?", (canonical_id, old_id))

        # Deduplicate head_to_head rows (keep one per pair)
        self.db.execute_insert("""
            DELETE FROM head_to_head
            WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM head_to_head GROUP BY team_a_id, team_b_id
            )
        """, ())

        # Remove old team row
        self.db.execute_insert("DELETE FROM teams WHERE team_id = ?", (old_id,))

        return old_id, canonical_id

    def add_unique_index(self) -> None:
        """Add a unique index on teams(team_name) to prevent future duplicates."""
        self.db.execute_insert("CREATE UNIQUE INDEX IF NOT EXISTS ux_teams_name ON teams(team_name)", ())

    def unify_all(self):
        """Unify all teams in the canonical map"""
        logger.info("=== Unifying duplicate teams ===")
        for old_name, canonical_name in CANONICAL_MAP.items():
            self.unify_team(old_name, canonical_name)
        self.add_unique_index()
        logger.success("Team unification complete")


def main():
    """Main execution"""
    unifier = TeamUnifier()
    unifier.unify_all()


if __name__ == "__main__":
    main()
