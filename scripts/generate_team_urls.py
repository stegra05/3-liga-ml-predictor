"""
Generate correct Transfermarkt URLs by searching for each team
"""

import json
import requests
from bs4 import BeautifulSoup
import time
import re
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from database.db_manager import get_db

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; 3Liga-Dataset-Collector/1.0; Educational Research)',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

BASE_URL = "https://www.transfermarkt.de"

def search_team(team_name: str):
    """Search for a team and return the first 3. Liga result"""
    search_url = f"{BASE_URL}/schnellsuche/ergebnis/schnellsuche"

    # Try multiple search variants
    # Extract the main city/team name
    # Remove common prefixes and get the main name
    simplified = re.sub(r'^(1\.\s*)?((FC|SV|TSV|SpVgg|VfL|VfB|VfR|SG|FSV|MSV|KFC)\s+)', '', team_name)
    main_name = simplified.split()[0]  # Get first word after prefixes

    search_variants = [
        team_name,  # Full name first
        main_name   # Simplified name (e.g., "Heidenheim", "Saarbrücken")
    ]

    results = []
    for search_term in search_variants:
        if not search_term:
            continue

        params = {'query': search_term}

        try:
            response = requests.get(search_url, params=params, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all club results
            results = soup.find_all('td', class_='hauptlink')

            if len(results) > 0:
                break  # Found results, stop trying variants

            time.sleep(0.5)  # Small delay between variants

        except Exception as e:
            print(f"✗ Error searching with '{search_term}': {e}")
            continue

    try:

        for result in results:
            link = result.find('a')
            if not link:
                continue

            href = link.get('href', '')
            result_name = link.text.strip()

            # Extract team ID and URL slug from href
            # Format: /vereinsname/startseite/verein/TEAM_ID
            match = re.search(r'/([^/]+)/startseite/verein/(\d+)', href)
            if match:
                url_slug = match.group(1)
                team_id = match.group(2)

                # Normalize names for comparison
                norm_search = team_name.lower().replace('.', '').replace(' ', '').replace('ü', 'u').replace('ö', 'o').replace('ä', 'a')
                norm_result = result_name.lower().replace('.', '').replace(' ', '').replace('ü', 'u').replace('ö', 'o').replace('ä', 'a')

                # Check if it's a good match (not youth/reserve teams)
                if 'u19' not in norm_result and 'u17' not in norm_result and 'u23' not in norm_result:
                    # If the search term is in the result or vice versa, it's a match
                    if norm_search in norm_result or norm_result in norm_search:
                        return {
                            'url_slug': url_slug,
                            'team_id': team_id,
                            'found_name': result_name
                        }

        print(f"✗ No result found for: {team_name}")
        return None

    except Exception as e:
        print(f"✗ Error searching for {team_name}: {e}")
        return None

def main():
    print("=== Transfermarkt URL Generator ===\n", flush=True)

    # Get all unique teams from database
    db = get_db()
    query = "SELECT DISTINCT team_name FROM teams ORDER BY team_name"
    teams_result = db.execute_query(query)
    team_names = [row['team_name'] for row in teams_result]

    print(f"Found {len(team_names)} teams in database\n", flush=True)

    results = {}

    for idx, team_name in enumerate(team_names, 1):
        print(f"[{idx}/{len(team_names)}] Searching for: {team_name:40}", end=' ', flush=True)

        result = search_team(team_name)

        if result:
            results[team_name] = {
                'url_slug': result['url_slug'],
                'team_id': result['team_id']
            }
            print(f"✓ ID: {result['team_id']:6} ({result['found_name']})", flush=True)
        else:
            print(f"✗ NOT FOUND", flush=True)

        time.sleep(2)  # Rate limiting

    # Save to file
    output = {
        'teams': results,
        'generated_at': str(time.time()),
        'total_teams': len(results)
    }

    output_file = 'config/transfermarkt_urls_new.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Generated {len(results)}/{len(team_names)} team URLs")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
