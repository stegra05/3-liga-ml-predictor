"""
Build Head-to-Head (H2H) table from historical match results.
"""
from pathlib import Path
from loguru import logger
from collections import defaultdict
from datetime import datetime

from liga_predictor.database import get_db
from liga_predictor.models import HeadToHead


class HeadToHeadBuilder:
    """Builds head-to-head statistics from historical matches"""

    def __init__(self):
        self.db = get_db()

    def compute_h2h(self):
        """Compute head-to-head statistics for all team pairs"""
        logger.info("Loading finished matches with results...")
        rows = self.db.execute_query("""
            SELECT match_id, home_team_id, away_team_id, result
            FROM matches
            WHERE is_finished = 1 AND home_goals IS NOT NULL AND away_goals IS NOT NULL
        """)
        logger.info(f"Processing {len(rows)} matches")

        # Aggregation maps
        key_to_counts = defaultdict(lambda: {
            'total': 0, 'a_wins': 0, 'b_wins': 0, 'draws': 0, 'a_home_wins': 0, 'a_away_wins': 0
        })

        for r in rows:
            home_id = r['home_team_id']
            away_id = r['away_team_id']
            result = r['result']  # 'H','D','A'

            # Canonicalize unordered pair
            a_id, b_id = (home_id, away_id) if home_id < away_id else (away_id, home_id)
            key = (a_id, b_id)

            counts = key_to_counts[key]
            counts['total'] += 1

            if result == 'D':
                counts['draws'] += 1
            elif result == 'H':
                # Home team won
                if home_id == a_id:
                    counts['a_wins'] += 1
                    counts['a_home_wins'] += 1
                else:
                    counts['b_wins'] += 1
                    counts.setdefault('b_home_wins', 0)
                    counts['b_home_wins'] += 1
            elif result == 'A':
                # Away team won
                if away_id == a_id:
                    counts['a_wins'] += 1
                    counts['a_away_wins'] += 1
                else:
                    counts['b_wins'] += 1
                    counts.setdefault('b_away_wins', 0)
                    counts['b_away_wins'] += 1

        logger.info(f"Computed H2H for {len(key_to_counts)} team pairs")

        # Upsert into head_to_head using ORM helper
        inserted = 0
        for (a_id, b_id), ct in key_to_counts.items():
            self.db.merge_or_create(
                HeadToHead,
                filter_dict={'team_a_id': a_id, 'team_b_id': b_id},
                defaults={
                    'total_matches': ct['total'],
                    'team_a_wins': ct['a_wins'],
                    'draws': ct['draws'],
                    'team_b_wins': ct['b_wins'],
                    'team_a_home_wins': ct.get('a_home_wins', 0),
                    'team_a_away_wins': ct.get('a_away_wins', 0)
                }
            )
            inserted += 1

        logger.success(f"Inserted/updated {inserted} head_to_head rows")


if __name__ == "__main__":
    print("Use CLI instead: liga-predictor build-h2h")
    import sys
    sys.exit(1)
