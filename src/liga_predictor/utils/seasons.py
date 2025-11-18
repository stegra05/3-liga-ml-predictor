"""
Season utility functions for dynamic season handling.
Eliminates hardcoded season lists throughout the codebase.
"""

from datetime import datetime
from typing import List, Dict


def get_current_season() -> str:
    """
    Get the current season based on today's date.

    Football seasons run from August to May.
    - August-December: Current year is the start year
    - January-July: Current year is the end year

    Returns:
        Season string in format "YYYY-YYYY" (e.g., "2024-2025")
    """
    today = datetime.now()
    year = today.year
    month = today.month

    # If we're in Jan-July, the season started last year
    if month < 8:
        start_year = year - 1
    else:
        start_year = year

    return f"{start_year}-{start_year + 1}"


def generate_seasons(start_year: int = 2009, end_year: int = None) -> List[str]:
    """
    Generate list of season strings from start_year to end_year.

    Args:
        start_year: First season start year (default: 2009 for 3. Liga data availability)
        end_year: Last season start year (default: current year)

    Returns:
        List of season strings ["2009-2010", "2010-2011", ...]

    Examples:
        >>> generate_seasons(2020, 2022)
        ['2020-2021', '2021-2022', '2022-2023']
    """
    if end_year is None:
        # Default to current year
        today = datetime.now()
        end_year = today.year if today.month >= 8 else today.year - 1

    return [f"{year}-{year + 1}" for year in range(start_year, end_year + 1)]


def generate_season_mapping(start_year: int = 2009, end_year: int = None) -> Dict[str, int]:
    """
    Generate season mapping dict for APIs that use start year.

    Args:
        start_year: First season start year (default: 2009)
        end_year: Last season start year (default: current year)

    Returns:
        Dict mapping season string to start year
        {"2009-2010": 2009, "2010-2011": 2010, ...}

    Examples:
        >>> generate_season_mapping(2020, 2022)
        {'2020-2021': 2020, '2021-2022': 2021, '2022-2023': 2022}
    """
    if end_year is None:
        today = datetime.now()
        end_year = today.year if today.month >= 8 else today.year - 1

    return {f"{year}-{year + 1}": year for year in range(start_year, end_year + 1)}


def parse_season(season: str) -> tuple[int, int]:
    """
    Parse season string into start and end years.

    Args:
        season: Season string like "2024-2025"

    Returns:
        Tuple of (start_year, end_year)

    Examples:
        >>> parse_season("2024-2025")
        (2024, 2025)

    Raises:
        ValueError: If season format is invalid
    """
    if "-" not in season:
        raise ValueError(f"Invalid season format: {season}. Expected 'YYYY-YYYY'")

    parts = season.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid season format: {season}. Expected 'YYYY-YYYY'")

    try:
        start_year = int(parts[0])
        end_year = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid season format: {season}. Years must be integers")

    if end_year != start_year + 1:
        raise ValueError(f"Invalid season: {season}. End year must be start year + 1")

    return start_year, end_year


def is_valid_season(season: str) -> bool:
    """
    Check if a season string is valid.

    Args:
        season: Season string to validate

    Returns:
        True if valid, False otherwise

    Examples:
        >>> is_valid_season("2024-2025")
        True
        >>> is_valid_season("2024-2026")
        False
    """
    try:
        parse_season(season)
        return True
    except ValueError:
        return False


def get_previous_season(season: str) -> str:
    """
    Get the previous season.

    Args:
        season: Current season string

    Returns:
        Previous season string

    Examples:
        >>> get_previous_season("2024-2025")
        '2023-2024'
    """
    start_year, _ = parse_season(season)
    prev_start = start_year - 1
    return f"{prev_start}-{prev_start + 1}"


def get_next_season(season: str) -> str:
    """
    Get the next season.

    Args:
        season: Current season string

    Returns:
        Next season string

    Examples:
        >>> get_next_season("2024-2025")
        '2025-2026'
    """
    start_year, _ = parse_season(season)
    next_start = start_year + 1
    return f"{next_start}-{next_start + 1}"


# Constants for specific data sources
TRANSFERMARKT_START_YEAR = 2009  # Transfermarkt has data from 2009-2010 onwards
FBREF_START_YEAR = 2018  # FBref has 3. Liga data from 2018-2019 onwards
OPENLIGADB_START_YEAR = 2009  # OpenLigaDB has data from 2009-2010 onwards
ODDSPORTAL_START_YEAR = 2009  # OddsPortal has data from 2009-2010 onwards


if __name__ == "__main__":
    # Test the functions
    print(f"Current season: {get_current_season()}")
    print(f"\nAll seasons (2009-present): {generate_seasons()}")
    print(f"\nFBref seasons (2018-present): {generate_seasons(FBREF_START_YEAR)}")
    print(f"\nSeason mapping: {generate_season_mapping(2020, 2022)}")
