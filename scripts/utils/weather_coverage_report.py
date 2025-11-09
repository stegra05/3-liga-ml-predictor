"""
Weather coverage report utility.
Generates detailed reports on weather data coverage by source, season, etc.
"""
from pathlib import Path
import sys
from typing import Dict, List
import pandas as pd
from loguru import logger
import argparse

sys.path.append(str(Path(__file__).parent.parent.parent))
from database.db_manager import get_db


def get_overall_coverage(db) -> Dict:
    """Get overall weather coverage statistics"""
    result = db.execute_query("""
        SELECT 
            COUNT(*) as total_matches,
            COUNT(temperature_celsius) as with_weather,
            COUNT(CASE WHEN weather_source IS NOT NULL THEN 1 END) as with_source
        FROM matches
        WHERE is_finished = 1
    """)[0]
    
    total = result['total_matches']
    with_weather = result['with_weather']
    coverage = (with_weather / total * 100) if total > 0 else 0.0
    
    return {
        'total_matches': total,
        'with_weather': with_weather,
        'coverage_pct': coverage,
        'missing': total - with_weather,
        'with_source': result['with_source'],
    }


def get_coverage_by_source(db) -> List[Dict]:
    """Get coverage breakdown by weather source"""
    results = db.execute_query("""
        SELECT 
            weather_source,
            COUNT(*) as count,
            AVG(weather_confidence) as avg_confidence,
            MIN(weather_confidence) as min_confidence,
            MAX(weather_confidence) as max_confidence
        FROM matches
        WHERE weather_source IS NOT NULL
        GROUP BY weather_source
        ORDER BY count DESC
    """)
    
    return [dict(r) for r in results]


def get_coverage_by_season(db) -> List[Dict]:
    """Get coverage breakdown by season"""
    results = db.execute_query("""
        SELECT 
            season,
            COUNT(*) as total_matches,
            COUNT(temperature_celsius) as with_weather,
            COUNT(CASE WHEN weather_source = 'meteostat' THEN 1 END) as meteostat_count,
            COUNT(CASE WHEN weather_source = 'open_meteo' THEN 1 END) as open_meteo_count,
            COUNT(CASE WHEN weather_source = 'dwd' THEN 1 END) as dwd_count
        FROM matches
        WHERE is_finished = 1
        GROUP BY season
        ORDER BY season DESC
    """)
    
    coverage_list = []
    for r in results:
        total = r['total_matches']
        with_weather = r['with_weather']
        coverage_list.append({
            'season': r['season'],
            'total_matches': total,
            'with_weather': with_weather,
            'coverage_pct': (with_weather / total * 100) if total > 0 else 0.0,
            'meteostat': r['meteostat_count'],
            'open_meteo': r['open_meteo_count'],
            'dwd': r['dwd_count'],
        })
    
    return coverage_list


def get_missing_weather_analysis(db) -> Dict:
    """Analyze matches still missing weather"""
    results = db.execute_query("""
        SELECT 
            COUNT(*) as total_missing,
            COUNT(CASE WHEN t.lat IS NULL OR t.lon IS NULL THEN 1 END) as missing_location,
            COUNT(CASE WHEN m.match_datetime IS NULL THEN 1 END) as missing_datetime,
            COUNT(CASE WHEN t.lat IS NOT NULL AND t.lon IS NOT NULL AND m.match_datetime IS NOT NULL THEN 1 END) as fetchable
        FROM matches m
        LEFT JOIN team_locations t ON t.team_id = m.home_team_id
        WHERE m.temperature_celsius IS NULL
          AND m.is_finished = 1
    """)[0]
    
    return dict(results)


def print_report(db, detailed: bool = False):
    """Print comprehensive coverage report"""
    print("\n" + "="*80)
    print("WEATHER COVERAGE REPORT")
    print("="*80)
    
    # Overall coverage
    overall = get_overall_coverage(db)
    print(f"\n📊 Overall Coverage:")
    print(f"  Total matches: {overall['total_matches']:,}")
    print(f"  With weather: {overall['with_weather']:,} ({overall['coverage_pct']:.2f}%)")
    print(f"  Missing: {overall['missing']:,}")
    print(f"  With source tracking: {overall['with_source']:,}")
    
    # Source breakdown
    sources = get_coverage_by_source(db)
    if sources:
        print(f"\n🌦️  Coverage by Source:")
        for src in sources:
            print(f"  {src['weather_source']:15s}: {src['count']:6,} matches")
            print(f"    Avg confidence: {src['avg_confidence']:.3f} "
                  f"(min: {src['min_confidence']:.3f}, max: {src['max_confidence']:.3f})")
    
    # Season breakdown
    if detailed:
        seasons = get_coverage_by_season(db)
        if seasons:
            print(f"\n📅 Coverage by Season:")
            print(f"  {'Season':<12} {'Total':>8} {'With Weather':>12} {'Coverage':>10} {'Meteostat':>10} {'Open-Meteo':>12} {'DWD':>6}")
            print("  " + "-"*70)
            for s in seasons:
                print(f"  {s['season']:<12} {s['total_matches']:>8,} {s['with_weather']:>12,} "
                      f"{s['coverage_pct']:>9.1f}% {s['meteostat']:>10,} {s['open_meteo']:>12,} {s['dwd']:>6,}")
    
    # Missing analysis
    missing = get_missing_weather_analysis(db)
    print(f"\n❌ Missing Weather Analysis:")
    print(f"  Total missing: {missing['total_missing']:,}")
    print(f"  Missing location data: {missing['missing_location']:,}")
    print(f"  Missing datetime: {missing['missing_datetime']:,}")
    print(f"  Fetchable (has location + datetime): {missing['fetchable']:,}")
    
    print("\n" + "="*80)


def assert_coverage(db, min_coverage: float = 95.0):
    """Assert that coverage meets minimum threshold"""
    overall = get_overall_coverage(db)
    coverage = overall['coverage_pct']
    
    if coverage >= min_coverage:
        logger.success(f"✓ Coverage assertion passed: {coverage:.2f}% >= {min_coverage}%")
        return True
    else:
        logger.error(f"✗ Coverage assertion failed: {coverage:.2f}% < {min_coverage}%")
        logger.info(f"  Missing: {overall['missing']:,} matches")
        return False


def main():
    parser = argparse.ArgumentParser(description="Weather coverage report and verification")
    parser.add_argument("--detailed", action="store_true", help="Show detailed season breakdown")
    parser.add_argument("--assert", type=float, default=None, metavar="PERCENT", 
                       help="Assert minimum coverage percentage (e.g., 95.0)")
    parser.add_argument("--csv", type=str, default=None, help="Export report to CSV file")
    args = parser.parse_args()
    
    db = get_db()
    
    # Print report
    print_report(db, detailed=args.detailed)
    
    # Assertion check
    if args.assert is not None:
        success = assert_coverage(db, args.assert)
        if not success:
            sys.exit(1)
    
    # CSV export
    if args.csv:
        overall = get_overall_coverage(db)
        sources = get_coverage_by_source(db)
        seasons = get_coverage_by_season(db)
        
        # Create summary DataFrame
        summary_data = {
            'metric': ['total_matches', 'with_weather', 'coverage_pct', 'missing'],
            'value': [overall['total_matches'], overall['with_weather'], 
                     overall['coverage_pct'], overall['missing']]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # Export
        with pd.ExcelWriter(args.csv, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            if sources:
                pd.DataFrame(sources).to_excel(writer, sheet_name='By Source', index=False)
            if seasons:
                pd.DataFrame(seasons).to_excel(writer, sheet_name='By Season', index=False)
        
        logger.info(f"Report exported to {args.csv}")


if __name__ == "__main__":
    main()

