"""
Database Quality Check Script
Identifies duplicates, orphaned records, and data integrity issues
"""

import pandas as pd
import sqlite3
from pathlib import Path
import json
from datetime import datetime

class DatabaseQualityChecker:
    def __init__(self, db_path='database/3liga.db', output_dir='docs/data'):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.issues = {
            'duplicates': {},
            'orphaned': {},
            'integrity': {},
            'anomalies': {}
        }

    def connect(self):
        """Connect to database"""
        self.conn = sqlite3.connect(self.db_path)

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def check_duplicate_matches(self):
        """Check for duplicate match records"""
        print("\n" + "="*80)
        print("CHECKING FOR DUPLICATE MATCHES")
        print("="*80)

        # Check duplicate openligadb IDs
        query = """
        SELECT openligadb_match_id, COUNT(*) as cnt
        FROM matches
        WHERE openligadb_match_id IS NOT NULL
        GROUP BY openligadb_match_id
        HAVING cnt > 1
        """
        duplicates = pd.read_sql(query, self.conn)

        if len(duplicates) > 0:
            print(f"\n⚠️  Found {len(duplicates)} duplicate openligadb_match_id values:")
            print(duplicates.head(10))
            self.issues['duplicates']['openligadb_match_id'] = len(duplicates)
        else:
            print("\n✅ No duplicate openligadb_match_id values")

        # Check duplicate match combinations (same teams, same date)
        query = """
        SELECT
            home_team_id,
            away_team_id,
            DATE(match_datetime) as match_date,
            COUNT(*) as cnt
        FROM matches
        GROUP BY home_team_id, away_team_id, match_date
        HAVING cnt > 1
        """
        duplicates = pd.read_sql(query, self.conn)

        if len(duplicates) > 0:
            print(f"\n⚠️  Found {len(duplicates)} duplicate match combinations:")
            print(duplicates.head(10))
            self.issues['duplicates']['match_combinations'] = len(duplicates)
        else:
            print("\n✅ No duplicate match combinations")

        return len(self.issues['duplicates']) == 0

    def check_duplicate_teams(self):
        """Check for duplicate team records"""
        print("\n" + "="*80)
        print("CHECKING FOR DUPLICATE TEAMS")
        print("="*80)

        # Check duplicate team names
        query = """
        SELECT team_name, COUNT(*) as cnt
        FROM teams
        GROUP BY team_name
        HAVING cnt > 1
        """
        duplicates = pd.read_sql(query, self.conn)

        if len(duplicates) > 0:
            print(f"\n⚠️  Found {len(duplicates)} duplicate team names:")
            print(duplicates)
            self.issues['duplicates']['team_names'] = len(duplicates)
        else:
            print("\n✅ No duplicate team names")

        # Check similar team names (potential duplicates)
        teams = pd.read_sql("SELECT team_name FROM teams ORDER BY team_name", self.conn)
        similar = []
        for i in range(len(teams) - 1):
            name1 = teams.iloc[i]['team_name'].lower()
            name2 = teams.iloc[i + 1]['team_name'].lower()
            if name1 in name2 or name2 in name1:
                if name1 != name2:
                    similar.append((teams.iloc[i]['team_name'], teams.iloc[i + 1]['team_name']))

        if similar:
            print(f"\nℹ️  Found {len(similar)} potentially similar team names:")
            for t1, t2 in similar[:10]:
                print(f"  - {t1} <-> {t2}")

        return len(self.issues['duplicates']) == 0

    def check_duplicate_statistics(self):
        """Check for duplicate match statistics"""
        print("\n" + "="*80)
        print("CHECKING FOR DUPLICATE STATISTICS")
        print("="*80)

        query = """
        SELECT match_id, team_id, COUNT(*) as cnt
        FROM match_statistics
        GROUP BY match_id, team_id
        HAVING cnt > 1
        """
        duplicates = pd.read_sql(query, self.conn)

        if len(duplicates) > 0:
            print(f"\n⚠️  Found {len(duplicates)} duplicate match statistics:")
            print(duplicates.head(10))
            self.issues['duplicates']['match_statistics'] = len(duplicates)
        else:
            print("\n✅ No duplicate match statistics")

    def check_duplicate_ratings(self):
        """Check for duplicate team ratings"""
        print("\n" + "="*80)
        print("CHECKING FOR DUPLICATE RATINGS")
        print("="*80)

        query = """
        SELECT match_id, team_id, COUNT(*) as cnt
        FROM team_ratings
        WHERE match_id IS NOT NULL
        GROUP BY match_id, team_id
        HAVING cnt > 1
        """
        duplicates = pd.read_sql(query, self.conn)

        if len(duplicates) > 0:
            print(f"\n⚠️  Found {len(duplicates)} duplicate team ratings:")
            print(duplicates.head(10))
            self.issues['duplicates']['team_ratings'] = len(duplicates)
        else:
            print("\n✅ No duplicate team ratings")

    def check_duplicate_odds(self):
        """Check for duplicate betting odds"""
        print("\n" + "="*80)
        print("CHECKING FOR DUPLICATE ODDS")
        print("="*80)

        query = """
        SELECT match_id, bookmaker, COUNT(*) as cnt
        FROM betting_odds
        GROUP BY match_id, bookmaker
        HAVING cnt > 1
        """
        duplicates = pd.read_sql(query, self.conn)

        if len(duplicates) > 0:
            print(f"\n⚠️  Found {len(duplicates)} duplicate betting odds:")
            print(duplicates.head(10))
            self.issues['duplicates']['betting_odds'] = len(duplicates)
        else:
            print("\n✅ No duplicate betting odds")

    def check_orphaned_data(self):
        """Check for orphaned records"""
        print("\n" + "="*80)
        print("CHECKING FOR ORPHANED DATA")
        print("="*80)

        # Teams not in any match
        query = """
        SELECT COUNT(*) as cnt
        FROM teams t
        WHERE NOT EXISTS (
            SELECT 1 FROM matches m
            WHERE m.home_team_id = t.team_id OR m.away_team_id = t.team_id
        )
        """
        count = pd.read_sql(query, self.conn).iloc[0]['cnt']
        if count > 0:
            print(f"\n⚠️  Found {count} teams not in any match")
            self.issues['orphaned']['unused_teams'] = count
        else:
            print("\n✅ All teams appear in at least one match")

        # Match statistics without match
        query = """
        SELECT COUNT(*) as cnt
        FROM match_statistics ms
        WHERE NOT EXISTS (
            SELECT 1 FROM matches m WHERE m.match_id = ms.match_id
        )
        """
        count = pd.read_sql(query, self.conn).iloc[0]['cnt']
        if count > 0:
            print(f"⚠️  Found {count} orphaned match_statistics records")
            self.issues['orphaned']['match_statistics'] = count
        else:
            print("✅ All match_statistics have corresponding matches")

        # Team ratings without match
        query = """
        SELECT COUNT(*) as cnt
        FROM team_ratings tr
        WHERE tr.match_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM matches m WHERE m.match_id = tr.match_id
        )
        """
        count = pd.read_sql(query, self.conn).iloc[0]['cnt']
        if count > 0:
            print(f"⚠️  Found {count} orphaned team_ratings records")
            self.issues['orphaned']['team_ratings'] = count
        else:
            print("✅ All team_ratings have corresponding matches")

        # Betting odds without match
        query = """
        SELECT COUNT(*) as cnt
        FROM betting_odds bo
        WHERE NOT EXISTS (
            SELECT 1 FROM matches m WHERE m.match_id = bo.match_id
        )
        """
        count = pd.read_sql(query, self.conn).iloc[0]['cnt']
        if count > 0:
            print(f"⚠️  Found {count} orphaned betting_odds records")
            self.issues['orphaned']['betting_odds'] = count
        else:
            print("✅ All betting_odds have corresponding matches")

    def check_data_integrity(self):
        """Check data integrity and consistency"""
        print("\n" + "="*80)
        print("CHECKING DATA INTEGRITY")
        print("="*80)

        # Matches with missing teams
        query = """
        SELECT COUNT(*) as cnt
        FROM matches m
        WHERE NOT EXISTS (SELECT 1 FROM teams WHERE team_id = m.home_team_id)
           OR NOT EXISTS (SELECT 1 FROM teams WHERE team_id = m.away_team_id)
        """
        count = pd.read_sql(query, self.conn).iloc[0]['cnt']
        if count > 0:
            print(f"\n⚠️  Found {count} matches with missing team references")
            self.issues['integrity']['missing_team_refs'] = count
        else:
            print("\n✅ All matches have valid team references")

        # Finished matches without result
        query = """
        SELECT COUNT(*) as cnt
        FROM matches
        WHERE is_finished = 1 AND (result IS NULL OR result = '')
        """
        count = pd.read_sql(query, self.conn).iloc[0]['cnt']
        if count > 0:
            print(f"⚠️  Found {count} finished matches without result")
            self.issues['integrity']['missing_results'] = count
        else:
            print("✅ All finished matches have results")

        # Finished matches without goals
        query = """
        SELECT COUNT(*) as cnt
        FROM matches
        WHERE is_finished = 1 AND (home_goals IS NULL OR away_goals IS NULL)
        """
        count = pd.read_sql(query, self.conn).iloc[0]['cnt']
        if count > 0:
            print(f"⚠️  Found {count} finished matches without goals")
            self.issues['integrity']['missing_goals'] = count
        else:
            print("✅ All finished matches have goal data")

        # Matches with inconsistent results
        query = """
        SELECT COUNT(*) as cnt
        FROM matches
        WHERE is_finished = 1
          AND ((result = 'H' AND home_goals <= away_goals)
           OR (result = 'D' AND home_goals != away_goals)
           OR (result = 'A' AND away_goals <= home_goals))
        """
        count = pd.read_sql(query, self.conn).iloc[0]['cnt']
        if count > 0:
            print(f"⚠️  Found {count} matches with inconsistent results")
            self.issues['integrity']['inconsistent_results'] = count
        else:
            print("✅ All match results are consistent with scores")

    def check_anomalies(self):
        """Check for data anomalies"""
        print("\n" + "="*80)
        print("CHECKING FOR ANOMALIES")
        print("="*80)

        # Unusually high goal counts
        query = """
        SELECT match_id, home_goals, away_goals, home_goals + away_goals as total_goals
        FROM matches
        WHERE home_goals + away_goals > 10
        ORDER BY total_goals DESC
        """
        high_scoring = pd.read_sql(query, self.conn)
        if len(high_scoring) > 0:
            print(f"\nℹ️  Found {len(high_scoring)} matches with >10 total goals:")
            print(high_scoring)
            self.issues['anomalies']['high_scoring_matches'] = len(high_scoring)
        else:
            print("\n✅ No extremely high-scoring matches")

        # Matches with same home and away team
        query = """
        SELECT COUNT(*) as cnt
        FROM matches
        WHERE home_team_id = away_team_id
        """
        count = pd.read_sql(query, self.conn).iloc[0]['cnt']
        if count > 0:
            print(f"\n⚠️  Found {count} matches where team plays against itself!")
            self.issues['anomalies']['self_matches'] = count
        else:
            print("\n✅ No matches where team plays against itself")

        # Negative goals
        query = """
        SELECT COUNT(*) as cnt
        FROM matches
        WHERE home_goals < 0 OR away_goals < 0
        """
        count = pd.read_sql(query, self.conn).iloc[0]['cnt']
        if count > 0:
            print(f"⚠️  Found {count} matches with negative goals!")
            self.issues['anomalies']['negative_goals'] = count
        else:
            print("✅ No matches with negative goals")

        # Check rating ranges
        query = """
        SELECT MIN(elo_rating) as min_elo, MAX(elo_rating) as max_elo
        FROM team_ratings
        """
        rating_range = pd.read_sql(query, self.conn).iloc[0]
        print(f"\nℹ️  Elo rating range: {rating_range['min_elo']:.0f} - {rating_range['max_elo']:.0f}")

        if rating_range['min_elo'] < 1000 or rating_range['max_elo'] > 2000:
            print("⚠️  Elo ratings outside typical range (1000-2000)")
            self.issues['anomalies']['elo_out_of_range'] = True

        # Check possession percentages
        query = """
        SELECT match_id, SUM(possession_percent) as total_possession
        FROM match_statistics
        GROUP BY match_id
        HAVING ABS(total_possession - 100) > 2
        """
        possession_issues = pd.read_sql(query, self.conn)
        if len(possession_issues) > 0:
            print(f"\nℹ️  Found {len(possession_issues)} matches where possession doesn't sum to 100%:")
            print(possession_issues.head(10))
            self.issues['anomalies']['possession_mismatch'] = len(possession_issues)

    def check_temporal_consistency(self):
        """Check temporal consistency"""
        print("\n" + "="*80)
        print("CHECKING TEMPORAL CONSISTENCY")
        print("="*80)

        # Matches in the future
        query = """
        SELECT COUNT(*) as cnt
        FROM matches
        WHERE match_datetime > datetime('now')
          AND is_finished = 1
        """
        count = pd.read_sql(query, self.conn).iloc[0]['cnt']
        if count > 0:
            print(f"\n⚠️  Found {count} finished matches in the future!")
            self.issues['integrity']['future_finished_matches'] = count
        else:
            print("\n✅ No finished matches in the future")

        # Check season-matchday consistency
        query = """
        SELECT season, COUNT(DISTINCT matchday) as num_matchdays
        FROM matches
        WHERE is_finished = 1
        GROUP BY season
        HAVING num_matchdays > 40
        """
        inconsistent = pd.read_sql(query, self.conn)
        if len(inconsistent) > 0:
            print(f"\nℹ️  Found {len(inconsistent)} seasons with >40 matchdays:")
            print(inconsistent)

    def generate_quality_report(self):
        """Generate comprehensive quality report"""
        print("\n" + "="*80)
        print("GENERATING QUALITY REPORT")
        print("="*80)

        report_path = self.output_dir / 'DATA_QUALITY_REPORT.md'

        with open(report_path, 'w') as f:
            f.write("# 3. Liga Dataset - Data Quality Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            f.write("## Executive Summary\n\n")

            total_issues = sum(len(v) for v in self.issues.values())
            if total_issues == 0:
                f.write("✅ **No significant data quality issues found!**\n\n")
                f.write("The database appears to be in excellent condition with no duplicates, ")
                f.write("orphaned records, or integrity violations.\n\n")
            else:
                f.write(f"⚠️  **Found {total_issues} data quality issues** that require attention.\n\n")

            # Duplicates section
            f.write("## 1. Duplicate Records\n\n")
            if self.issues['duplicates']:
                f.write("The following duplicate records were found:\n\n")
                f.write("| Type | Count |\n")
                f.write("|------|-------|\n")
                for key, value in self.issues['duplicates'].items():
                    f.write(f"| {key} | {value} |\n")
                f.write("\n**Action Required:** Review and remove duplicate records.\n\n")
            else:
                f.write("✅ No duplicate records found.\n\n")

            # Orphaned data section
            f.write("## 2. Orphaned Data\n\n")
            if self.issues['orphaned']:
                f.write("The following orphaned records were found:\n\n")
                f.write("| Type | Count |\n")
                f.write("|------|-------|\n")
                for key, value in self.issues['orphaned'].items():
                    f.write(f"| {key} | {value} |\n")
                f.write("\n**Action Required:** Clean up orphaned records or fix relationships.\n\n")
            else:
                f.write("✅ No orphaned records found.\n\n")

            # Integrity issues section
            f.write("## 3. Data Integrity Issues\n\n")
            if self.issues['integrity']:
                f.write("The following integrity issues were found:\n\n")
                f.write("| Type | Count |\n")
                f.write("|------|-------|\n")
                for key, value in self.issues['integrity'].items():
                    f.write(f"| {key} | {value} |\n")
                f.write("\n**Action Required:** Fix data integrity violations.\n\n")
            else:
                f.write("✅ No data integrity issues found.\n\n")

            # Anomalies section
            f.write("## 4. Data Anomalies\n\n")
            if self.issues['anomalies']:
                f.write("The following anomalies were found:\n\n")
                f.write("| Type | Count/Value |\n")
                f.write("|------|-------------|\n")
                for key, value in self.issues['anomalies'].items():
                    f.write(f"| {key} | {value} |\n")
                f.write("\n**Note:** Some anomalies may be legitimate data points.\n\n")
            else:
                f.write("✅ No significant data anomalies found.\n\n")

            # Recommendations
            f.write("## 5. Recommendations\n\n")

            if self.issues['duplicates']:
                f.write("### Immediate Actions (Critical)\n\n")
                f.write("1. **Remove duplicate records** to ensure data integrity\n")
                f.write("2. **Add unique constraints** to prevent future duplicates\n")
                f.write("3. **Review data import processes** to identify duplication source\n\n")

            if self.issues['orphaned']:
                f.write("### Data Cleanup\n\n")
                f.write("1. **Clean up orphaned records** or establish proper relationships\n")
                f.write("2. **Add foreign key constraints** for referential integrity\n")
                f.write("3. **Implement cascade deletes** where appropriate\n\n")

            if self.issues['integrity']:
                f.write("### Data Validation\n\n")
                f.write("1. **Fix integrity violations** before they propagate\n")
                f.write("2. **Add database constraints** to prevent invalid data\n")
                f.write("3. **Implement validation** in data collection scripts\n\n")

            f.write("### Ongoing Maintenance\n\n")
            f.write("1. **Run quality checks regularly** (e.g., weekly)\n")
            f.write("2. **Monitor data collection processes** for errors\n")
            f.write("3. **Document data quality standards** and validation rules\n")
            f.write("4. **Maintain changelog** of data corrections\n\n")

            f.write("---\n\n")
            f.write("*This report was automatically generated by the data quality checker.*\n")

        print(f"\n✓ Quality report saved: {report_path}")

        # Save issues as JSON
        issues_file = self.output_dir / 'data_quality_issues.json'
        with open(issues_file, 'w') as f:
            json.dump(self.issues, f, indent=2, default=str)
        print(f"✓ Issues saved: {issues_file}")

        return report_path

    def run_all_checks(self):
        """Run all quality checks"""
        print("\n" + "="*80)
        print("3. LIGA DATABASE QUALITY CHECK")
        print("="*80)

        self.check_duplicate_matches()
        self.check_duplicate_teams()
        self.check_duplicate_statistics()
        self.check_duplicate_ratings()
        self.check_duplicate_odds()
        self.check_orphaned_data()
        self.check_data_integrity()
        self.check_anomalies()
        self.check_temporal_consistency()

        self.generate_quality_report()

        print("\n" + "="*80)
        print("QUALITY CHECK COMPLETE!")
        print("="*80)

        total_issues = sum(len(v) for v in self.issues.values() if isinstance(v, dict))
        if total_issues == 0:
            print("\n✅ Database is in excellent condition!")
        else:
            print(f"\n⚠️  Found {total_issues} issue categories requiring attention")
            print("    See DATA_QUALITY_REPORT.md for details")

def main():
    checker = DatabaseQualityChecker()

    try:
        checker.connect()
        checker.run_all_checks()
    finally:
        checker.close()

if __name__ == "__main__":
    main()
