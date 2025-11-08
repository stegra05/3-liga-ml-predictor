"""
Validate and fix Transfermarkt team IDs
"""

import json
import requests
import time
from bs4 import BeautifulSoup
from pathlib import Path
import re

CONFIG_FILE = "config/transfermarkt_urls.json"
BASE_URL = "https://www.transfermarkt.de"

# Use same headers as the actual scraper
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; 3Liga-Dataset-Collector/1.0; Educational Research)',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

def verify_team_id(url_slug: str, team_id: str, team_name: str) -> tuple:
    """Verify if the team ID is correct by checking the squad page"""
    # Use the squad page URL
    squad_url = f"{BASE_URL}/{url_slug}/kader/verein/{team_id}/saison_id/2024/plus/1"

    try:
        response = requests.get(squad_url, headers=HEADERS, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the team name on the page (in the header)
        # Look for the h1 header which contains the team name
        header = soup.find('h1', class_='data-header__headline-wrapper')
        if header:
            page_team_name = header.text.strip()

            # Normalize both names for comparison
            normalized_expected = team_name.lower().replace('.', '').replace(' ', '')
            normalized_found = page_team_name.lower().replace('.', '').replace(' ', '')

            if normalized_expected in normalized_found or normalized_found in normalized_expected:
                print(f"✓ {team_name:40} ID: {team_id:6} -> OK")
                return (True, team_id)
            else:
                print(f"✗ {team_name:40} ID: {team_id:6} -> MISMATCH (found: {page_team_name})")
                return (False, None)
        else:
            print(f"? {team_name:40} ID: {team_id:6} -> Could not verify (no header found)")
            return (None, None)

    except Exception as e:
        print(f"✗ {team_name:40} ID: {team_id:6} -> Error: {str(e)[:50]}")
        return (False, None)
    finally:
        time.sleep(1.5)  # Rate limiting


def main():
    print("=== Transfermarkt Team ID Validator ===\n")

    # Load config
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)

    teams = config['teams']
    print(f"Validating {len(teams)} teams...\n")

    mismatches = []

    for team_name, team_data in sorted(teams.items()):
        url_slug = team_data['url_slug']
        current_id = team_data['team_id']

        # Verify team ID
        is_correct, verified_id = verify_team_id(url_slug, current_id, team_name)

        if is_correct == False:  # Explicit False check (not None)
            mismatches.append({
                'team': team_name,
                'url_slug': url_slug,
                'bad_id': current_id
            })

    # Summary
    print(f"\n{'='*60}")
    print(f"Validation complete!")
    print(f"Teams checked: {len(teams)}")
    print(f"Mismatches found: {len(mismatches)}")

    if mismatches:
        print(f"\n{'='*60}")
        print("Teams with INCORRECT IDs:")
        print(f"{'='*60}")
        for m in mismatches:
            print(f"\nTeam: {m['team']}")
            print(f"  URL slug: {m['url_slug']}")
            print(f"  Bad ID: {m['bad_id']}")
            print(f"  Check manually: https://www.transfermarkt.de/{m['url_slug']}/startseite/verein/")

        print(f"\n{'='*60}")
        print(f"Please manually correct these {len(mismatches)} team IDs in {CONFIG_FILE}")
        print("Search for the correct team on Transfermarkt and update the team_id field.")
    else:
        print("\n✓ All team IDs are correct!")


if __name__ == "__main__":
    main()
