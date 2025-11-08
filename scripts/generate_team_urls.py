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
    params = {'query': team_name}

    try:
        response = requests.get(search_url, params=params, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all club results
        results = soup.find_all('td', class_='hauptlink')

        for result in results:
            link = result.find('a', class_='vereinprofil_tooltip')
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

                # Check if it's in the right league (look for info in the same row)
                parent_row = result.find_parent('tr')
                if parent_row:
                    league_info = parent_row.find('td', class_='zentriert')
                    if league_info:
                        league_text = league_info.text.strip()
                        # Accept 3. Liga, 2. Bundesliga, or Bundesliga (teams may have moved)
                        if '3. Liga' in league_text or '2. Bundesliga' in league_text or 'Bundesliga' in league_text:
                            return {
                                'url_slug': url_slug,
                                'team_id': team_id,
                                'found_name': result_name,
                                'league': league_text
                            }

        print(f"✗ No result found for: {team_name}")
        return None

    except Exception as e:
        print(f"✗ Error searching for {team_name}: {e}")
        return None

def main():
    print("=== Transfermarkt URL Generator ===\n")

    # Get all unique teams from database
    db = get_db()
    query = "SELECT DISTINCT team_name FROM teams ORDER BY team_name"
    teams_result = db.execute_query(query)
    team_names = [row['team_name'] for row in teams_result]

    print(f"Found {len(team_names)} teams in database\n")

    results = {}

    for idx, team_name in enumerate(team_names, 1):
        print(f"[{idx}/{len(team_names)}] Searching for: {team_name:40}", end=' ')

        result = search_team(team_name)

        if result:
            results[team_name] = {
                'url_slug': result['url_slug'],
                'team_id': result['team_id']
            }
            print(f"✓ ID: {result['team_id']:6} ({result['found_name']})")
        else:
            print(f"✗ NOT FOUND")

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
