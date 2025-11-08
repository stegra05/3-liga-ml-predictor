"""
Clean and standardize team mappings
Automatically consolidates obvious duplicates and standardizes naming
"""

import json
from pathlib import Path
from loguru import logger


def clean_team_mappings(input_file: str = "config/team_mappings.json",
                        output_file: str = "config/team_mappings.json") -> None:
    """
    Clean team mappings by standardizing obvious duplicates

    Args:
        input_file: Input mappings file
        output_file: Output mappings file (can be same as input)
    """

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Define manual standardization rules
    # Format: variant_name -> standard_name
    standardization_rules = {
        # Bayern München variations
        "Bayern München II": "FC Bayern München II",
        "Bayern Munich II": "FC Bayern München II",

        # Carl Zeiss Jena
        "Carl Zeiss Jena": "FC Carl Zeiss Jena",

        # Dynamo Dresden
        "SG Dynamo Dresden": "Dynamo Dresden",

        # Großaspach
        "Sonnenhof Großaspach": "SG Sonnenhof Großaspach",

        # Rot-Weiß Erfurt
        "Rot-Weiß Erfurt": "FC Rot-Weiß Erfurt",
        "RW Erfurt": "FC Rot-Weiß Erfurt",

        # Waldhof Mannheim
        "Waldhof Mannheim": "SV Waldhof Mannheim",

        # Wehen Wiesbaden
        "Wehen Wiesbaden": "SV Wehen Wiesbaden",
        "SV Wehen": "SV Wehen Wiesbaden",

        # Osnabrück (fix umlaut)
        "VfL Osnabruck": "VfL Osnabrück",

        # Preußen Münster (fix umlaut)
        "Preussen Münster": "Preußen Münster",
        "Preussen Munster": "Preußen Münster",

        # KFC Uerdingen
        "KFC Uerdingen": "KFC Uerdingen 05",

        # 1860 München variations
        "1860 München": "TSV 1860 München",
        "1860 Munich": "TSV 1860 München",
        "TSV 1860 Munchen": "TSV 1860 München",

        # Heidenheim
        "1. FC Heidenheim": "1. FC Heidenheim 1846",

        # Viktoria Köln
        "Viktoria Koln": "FC Viktoria Köln",
        "Viktoria Köln": "FC Viktoria Köln",

        # SC Verl
        "Sportclub Verl": "SC Verl",

        # FSV Zwickau
        "Zwickau": "FSV Zwickau",

        # Türkgücü München
        "Turkgucu Munchen": "Türkgücü München",
        "Turkgucu München": "Türkgücü München",

        # Ingolstadt
        "Ingolstadt": "FC Ingolstadt 04",

        # Unterhaching
        "Unterhaching": "SpVgg Unterhaching",

        # Saarbrücken
        "Saarbrucken": "1. FC Saarbrücken",
        "1. FC Saarbrucken": "1. FC Saarbrücken",

        # Magdeburg
        "Magdeburg": "1. FC Magdeburg",

        # Duisburg
        "Duisburg": "MSV Duisburg",

        # Hallescher FC
        "Halle": "Hallescher FC",

        # Energie Cottbus
        "Cottbus": "Energie Cottbus",
        "FC Energie Cottbus": "Energie Cottbus",

        # Chemnitzer FC
        "Chemnitz": "Chemnitzer FC",

        # Würzburger Kickers
        "Wurzburger Kickers": "Würzburger Kickers",
        "Wurzburg": "Würzburger Kickers",

        # Arminia Bielefeld II
        "Arminia Bielefeld II": "DSC Arminia Bielefeld II",

        # Viktoria Berlin
        "Viktoria Berlin": "FC Viktoria 1889 Berlin",

        # Erzgebirge Aue
        "Aue": "Erzgebirge Aue",
        "FC Erzgebirge Aue": "Erzgebirge Aue",
    }

    # Apply standardization rules
    teams = data.get("teams", {})
    aliases = data.get("aliases", {})

    standardized_teams = {}
    updated_aliases = {}

    for team_name, team_info in teams.items():
        if team_name in standardization_rules:
            # This is a variant that should be aliased
            standard_name = standardization_rules[team_name]
            updated_aliases[team_name] = standard_name

            # Check if standard already exists
            if standard_name not in standardized_teams:
                standardized_teams[standard_name] = {
                    "standard_name": standard_name,
                    "short_name": team_info.get("short_name", standard_name[:20]),
                    "openligadb_id": team_info.get("openligadb_id"),
                    "transfermarkt_id": team_info.get("transfermarkt_id"),
                    "founded": team_info.get("founded"),
                    "stadium": team_info.get("stadium")
                }
        else:
            # Keep as-is
            standardized_teams[team_name] = team_info
            if "standard_name" not in team_info:
                team_info["standard_name"] = team_name

    # Update data
    data["teams"] = standardized_teams
    data["aliases"] = updated_aliases

    # Update metadata
    if "metadata" not in data:
        data["metadata"] = {}

    data["metadata"]["total_teams"] = len(standardized_teams)
    data["metadata"]["total_aliases"] = len(updated_aliases)
    data["metadata"]["cleaned"] = True

    # Save cleaned mappings
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.success(f"Cleaned team mappings saved to {output_file}")
    logger.info(f"Standardized teams: {len(standardized_teams)}")
    logger.info(f"Aliases created: {len(updated_aliases)}")

    print(f"\n✓ Team mappings cleaned")
    print(f"  - {len(standardized_teams)} standardized teams")
    print(f"  - {len(updated_aliases)} aliases created")


if __name__ == "__main__":
    clean_team_mappings()
