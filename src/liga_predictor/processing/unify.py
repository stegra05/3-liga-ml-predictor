"""
Team Unification Script
Merges duplicate team records to a single canonical team and updates references.
"""

from loguru import logger
from pathlib import Path
from typing import Dict, List, Tuple

from liga_predictor.database import get_db
from liga_predictor.models import Team, Match, MatchStatistic, LeagueStanding, TeamRating, HeadToHead


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
        """Get team ID by team name using ORM"""
        session = self.db.get_session()
        try:
            team = session.query(Team).filter(Team.team_name == name).first()
            return team.team_id if team else None
        finally:
            session.close()

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
            session = self.db.get_session()
            try:
                team = session.query(Team).filter(Team.team_id == old_id).first()
                if team:
                    team.team_name = canonical_name
                    session.commit()
            finally:
                session.close()
            return old_id, old_id

        if old_id == canonical_id:
            logger.info(f"Team already canonical: {canonical_name}")
            return old_id, canonical_id

        logger.info(f"Merging '{old_name}' (id={old_id}) -> '{canonical_name}' (id={canonical_id})")

        # Update references across tables using ORM
        session = self.db.get_session()
        try:
            # Update matches
            session.query(Match).filter(Match.home_team_id == old_id).update({Match.home_team_id: canonical_id})
            session.query(Match).filter(Match.away_team_id == old_id).update({Match.away_team_id: canonical_id})
            
            # Update match_statistics
            session.query(MatchStatistic).filter(MatchStatistic.team_id == old_id).update({MatchStatistic.team_id: canonical_id})
            
            # Update league_standings
            session.query(LeagueStanding).filter(LeagueStanding.team_id == old_id).update({LeagueStanding.team_id: canonical_id})
            
            # Update team_ratings
            session.query(TeamRating).filter(TeamRating.team_id == old_id).update({TeamRating.team_id: canonical_id})
            
            # Update head_to_head
            session.query(HeadToHead).filter(HeadToHead.team_a_id == old_id).update({HeadToHead.team_a_id: canonical_id})
            session.query(HeadToHead).filter(HeadToHead.team_b_id == old_id).update({HeadToHead.team_b_id: canonical_id})

            # Deduplicate head_to_head rows (keep one per pair) - use raw SQL for this complex operation
            from sqlalchemy import text
            session.execute(text("""
                DELETE FROM head_to_head
                WHERE rowid NOT IN (
                    SELECT MIN(rowid) FROM head_to_head GROUP BY team_a_id, team_b_id
                )
            """))

            # Remove old team row
            session.query(Team).filter(Team.team_id == old_id).delete()

            session.commit()
        finally:
            session.close()

        return old_id, canonical_id

    def add_unique_index(self) -> None:
        """Add a unique index on teams(team_name) to prevent future duplicates."""
        from sqlalchemy import text
        session = self.db.get_session()
        try:
            session.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ux_teams_name ON teams(team_name)"))
            session.commit()
        finally:
            session.close()

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
