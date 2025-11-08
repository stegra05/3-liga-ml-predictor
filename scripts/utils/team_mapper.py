"""
Team Mapping System for 3. Liga Dataset
Standardizes team names across different data sources
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from loguru import logger
import sys
import unicodedata
import re

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from database.db_manager import get_db


class TeamMapper:
    """Handles team name standardization and mapping"""

    def __init__(self, config_path: str = "config/team_mappings.json"):
        """
        Initialize team mapper

        Args:
            config_path: Path to team mappings configuration file
        """
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.mappings = self._load_mappings()
        # Build normalized lookup tables
        self._build_normalized_maps()

    @staticmethod
    def _normalize(name: str) -> str:
        """
        Normalize team names for robust matching:
        - lowercase
        - remove diacritics
        - strip common prefixes (fc, sc, sg, tsv, sv, 1., 1)
        - remove punctuation/dots/extra spaces
        - unify common abbreviations
        """
        if not name:
            return ""
        s = name.strip()
        # Replace abbreviations that appear in OddsPortal
        s = s.replace("Stutt. ", "Stuttgarter ")
        # Lowercase
        s_lower = s.lower()
        # Remove diacritics
        s_norm = unicodedata.normalize("NFKD", s_lower)
        s_ascii = "".join([c for c in s_norm if not unicodedata.combining(c)])
        # Remove punctuation and dots
        s_ascii = re.sub(r"[^\w\s]", " ", s_ascii)
        # Strip common leading tokens
        tokens = s_ascii.split()
        drop = {"fc", "sc", "sg", "tsv", "sv", "1", "1."}
        tokens = [t for t in tokens if t not in drop]
        # Join and collapse spaces
        s_final = re.sub(r"\s+", " ", " ".join(tokens)).strip()
        return s_final

    def _build_normalized_maps(self) -> None:
        """Precompute normalized name -> canonical mappings from config and DB."""
        self._norm_to_canonical: Dict[str, str] = {}
        # From config teams
        for original, data in self.mappings.get("teams", {}).items():
            standard = data.get("standard_name", original)
            self._norm_to_canonical[self._normalize(original)] = standard
            self._norm_to_canonical[self._normalize(standard)] = standard
        # From config aliases
        for alias, standard in self.mappings.get("aliases", {}).items():
            self._norm_to_canonical[self._normalize(alias)] = standard
            self._norm_to_canonical[self._normalize(standard)] = standard
        # From DB existing teams
        try:
            db = get_db()
            rows = db.execute_query("SELECT team_name FROM teams")
            for r in rows:
                t = r["team_name"]
                self._norm_to_canonical.setdefault(self._normalize(t), t)
        except Exception as e:
            logger.debug(f"Could not pre-load DB team names for normalization: {e}")

    def _load_mappings(self) -> Dict:
        """Load team mappings from config file"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"teams": {}, "aliases": {}}

    def save_mappings(self) -> None:
        """Save team mappings to config file"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.mappings, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved team mappings to {self.config_path}")

    def extract_teams_from_csv(self, csv_path: str,
                               home_col: str = 'homeTeamName',
                               away_col: str = 'awayTeamName') -> set:
        """
        Extract unique team names from CSV file

        Args:
            csv_path: Path to CSV file
            home_col: Column name for home team
            away_col: Column name for away team

        Returns:
            Set of unique team names
        """
        try:
            df = pd.read_csv(csv_path)
            teams = set()

            if home_col in df.columns:
                teams.update(df[home_col].dropna().unique())
            if away_col in df.columns:
                teams.update(df[away_col].dropna().unique())

            logger.info(f"Extracted {len(teams)} teams from {Path(csv_path).name}")
            return teams

        except Exception as e:
            logger.error(f"Error reading {csv_path}: {e}")
            return set()

    def collect_all_team_names(self, data_dir: str = "data/raw") -> Dict[str, set]:
        """
        Collect all team names from all data sources

        Args:
            data_dir: Directory containing raw data files

        Returns:
            Dictionary with source as key and set of team names as value
        """
        data_path = Path(data_dir)
        all_teams = {}

        # Matches file
        matches_file = data_path / "matches_3liga_2009-2025.csv"
        if matches_file.exists():
            all_teams['matches'] = self.extract_teams_from_csv(
                str(matches_file),
                'homeTeamName',
                'awayTeamName'
            )

        # FotMob stats
        fotmob_file = data_path / "fotmob_stats_all.csv"
        if fotmob_file.exists():
            all_teams['fotmob'] = self.extract_teams_from_csv(
                str(fotmob_file),
                'home_team',
                'away_team'
            )

        # OddsPortal
        odds_file = data_path / "oddsportal_3liga_full.csv"
        if odds_file.exists():
            all_teams['oddsportal'] = self.extract_teams_from_csv(
                str(odds_file),
                'homeTeamName',
                'awayTeamName'
            )

        # Get unique teams across all sources
        all_unique = set()
        for teams in all_teams.values():
            all_unique.update(teams)

        all_teams['all_unique'] = all_unique
        logger.info(f"Total unique team names across all sources: {len(all_unique)}")

        return all_teams

    def create_standardized_names(self, team_names: set) -> Dict[str, str]:
        """
        Create standardized team names from various naming conventions

        Args:
            team_names: Set of team names from various sources

        Returns:
            Dictionary mapping variants to standardized names
        """
        standardized = {}

        # Common patterns for standardization
        for team in sorted(team_names):
            # Use the name as-is initially
            standard = team

            # Store all variations
            standardized[team] = standard

        return standardized

    def find_similar_names(self, team_names: List[str], threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """
        Find similar team names that might be the same team

        Args:
            team_names: List of team names
            threshold: Similarity threshold (0-1)

        Returns:
            List of (name1, name2, similarity_score) tuples
        """
        from difflib import SequenceMatcher

        similar_pairs = []

        for i, name1 in enumerate(team_names):
            for name2 in team_names[i + 1:]:
                similarity = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
                if similarity >= threshold and similarity < 1.0:
                    similar_pairs.append((name1, name2, similarity))

        return sorted(similar_pairs, key=lambda x: x[2], reverse=True)

    def generate_mapping_template(self) -> Dict:
        """
        Generate a template for manual team mapping

        Returns:
            Dictionary template for team mappings
        """
        all_teams = self.collect_all_team_names()
        team_list = sorted(all_teams.get('all_unique', []))

        mapping_template = {
            "metadata": {
                "total_teams": len(team_list),
                "generated_at": pd.Timestamp.now().isoformat(),
                "sources": list(all_teams.keys())
            },
            "teams": {},
            "aliases": {}
        }

        # Find similar names for manual review
        similar = self.find_similar_names(team_list, threshold=0.75)

        if similar:
            mapping_template["metadata"]["similar_names_to_review"] = [
                {"name1": n1, "name2": n2, "similarity": round(score, 3)}
                for n1, n2, score in similar[:20]  # Top 20 most similar
            ]

        # Create initial mapping (all teams map to themselves)
        for team in team_list:
            mapping_template["teams"][team] = {
                "standard_name": team,
                "short_name": team[:20],  # Shortened version
                "openligadb_id": None,
                "founded": None,
                "stadium": None
            }

        return mapping_template

    def populate_database_with_teams(self) -> None:
        """Populate database teams table from mappings"""
        db = get_db()

        if not self.mappings.get("teams"):
            logger.warning("No team mappings found. Run generate_mapping_template() first.")
            return

        teams_added = 0
        for team_name, team_data in self.mappings["teams"].items():
            try:
                team_id = db.get_or_create_team(
                    team_name=team_data.get("standard_name", team_name),
                    openligadb_id=team_data.get("openligadb_id")
                )
                teams_added += 1
            except Exception as e:
                logger.error(f"Error adding team {team_name}: {e}")

        logger.success(f"Added {teams_added} teams to database")

    def get_standard_name(self, team_name: str) -> str:
        """
        Get standardized team name

        Args:
            team_name: Original team name from any source

        Returns:
            Standardized team name
        """
        # Fast path: normalized canonical map
        norm = self._normalize(team_name)
        if norm in self._norm_to_canonical:
            return self._norm_to_canonical[norm]

        # Check direct mapping
        if team_name in self.mappings.get("teams", {}):
            return self.mappings["teams"][team_name].get("standard_name", team_name)

        # Check aliases
        if team_name in self.mappings.get("aliases", {}):
            standard = self.mappings["aliases"][team_name]
            if standard in self.mappings.get("teams", {}):
                return self.mappings["teams"][standard].get("standard_name", team_name)

        # Return as-is if no mapping found
        return team_name

    def get_team_id(self, team_name: str) -> Optional[int]:
        """
        Get database team ID for a team name

        Args:
            team_name: Team name (any variant)

        Returns:
            Team ID or None
        """
        standard_name = self.get_standard_name(team_name)
        db = get_db()
        team_id = db.get_team_id_by_name(standard_name)
        if team_id:
            return team_id
        # Attempt fallback lookups with normalized variants
        # Try adding/removing common prefixes
        candidates = [
            standard_name,
            standard_name.replace("Koln", "Köln"),
            standard_name.replace("Munster", "Münster"),
            standard_name.replace("Wurzburger", "Würzburger"),
        ]
        for c in candidates:
            team_id = db.get_team_id_by_name(c)
            if team_id:
                return team_id
        # Last resort: case-insensitive LIKE
        try:
            res = db.execute_query(
                "SELECT team_id FROM teams WHERE LOWER(team_name)=LOWER(?) OR LOWER(team_name) LIKE LOWER(?) LIMIT 1",
                (standard_name, f"%{standard_name}%"),
            )
            if res:
                return res[0]["team_id"]
        except Exception:
            pass
        return None

    def export_team_list(self, output_file: str = "data/processed/team_list.csv") -> None:
        """
        Export standardized team list to CSV

        Args:
            output_file: Output CSV file path
        """
        if not self.mappings.get("teams"):
            logger.warning("No team mappings to export")
            return

        teams_data = []
        for team_name, team_data in self.mappings["teams"].items():
            teams_data.append({
                "original_name": team_name,
                "standard_name": team_data.get("standard_name", team_name),
                "short_name": team_data.get("short_name", ""),
                "openligadb_id": team_data.get("openligadb_id", ""),
                "stadium": team_data.get("stadium", ""),
                "founded": team_data.get("founded", "")
            })

        df = pd.DataFrame(teams_data)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.success(f"Exported team list to {output_file}")


def main():
    """Main function for team mapping initialization"""
    logger.info("Starting team mapping system initialization")

    mapper = TeamMapper()

    # Collect all team names
    logger.info("Collecting team names from all data sources...")
    all_teams = mapper.collect_all_team_names()

    print("\n=== Team Names by Source ===")
    for source, teams in all_teams.items():
        if source != 'all_unique':
            print(f"{source}: {len(teams)} teams")

    print(f"\nTotal unique teams: {len(all_teams['all_unique'])}")

    # Generate mapping template
    logger.info("Generating team mapping template...")
    template = mapper.generate_mapping_template()

    # Print similar names for review
    if "similar_names_to_review" in template["metadata"]:
        print("\n=== Similar Team Names (Manual Review Needed) ===")
        for pair in template["metadata"]["similar_names_to_review"][:10]:
            print(f"  {pair['name1']} <-> {pair['name2']} (similarity: {pair['similarity']})")

    # Save template
    mapper.mappings = template
    mapper.save_mappings()

    print(f"\n✓ Team mapping template created: {mapper.config_path}")
    print(f"✓ Found {template['metadata']['total_teams']} unique teams")
    print("\nNext steps:")
    print("1. Review and edit config/team_mappings.json to standardize team names")
    print("2. Add OpenLigaDB IDs where available")
    print("3. Run mapper.populate_database_with_teams() to add teams to database")


if __name__ == "__main__":
    main()
