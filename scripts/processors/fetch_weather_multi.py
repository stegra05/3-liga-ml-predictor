"""
Multi-source weather fetching orchestrator.
Runs staged pipeline: Meteostat → Open-Meteo → DWD to achieve 95%+ coverage.
"""
from pathlib import Path
import sys
from datetime import datetime
from loguru import logger
import argparse
import subprocess

sys.path.append(str(Path(__file__).parent.parent.parent))
from database.db_manager import get_db


def get_weather_coverage(db) -> float:
    """Calculate current weather coverage percentage"""
    result = db.execute_query("""
        SELECT 
            COUNT(*) as total_matches,
            COUNT(temperature_celsius) as with_weather
        FROM matches
        WHERE is_finished = 1
    """)[0]
    
    if result['total_matches'] == 0:
        return 0.0
    
    coverage = (result['with_weather'] / result['total_matches']) * 100
    return coverage


def get_matches_needing_weather(db) -> int:
    """Get count of matches still needing weather"""
    result = db.execute_query("""
        SELECT COUNT(*) as count
        FROM matches m
        JOIN team_locations t ON t.team_id = m.home_team_id
        WHERE m.temperature_celsius IS NULL
          AND m.match_datetime IS NOT NULL
          AND t.lat IS NOT NULL AND t.lon IS NOT NULL
          AND m.is_finished = 1
    """)[0]
    return result['count']


def run_stage(stage_name: str, script_path: str, limit: int = None, sleep: float = None, **kwargs) -> bool:
    """Run a weather fetching stage"""
    logger.info(f"=== Stage: {stage_name} ===")
    
    cmd = [sys.executable, str(script_path)]
    if limit:
        cmd.extend(["--limit", str(limit)])
    if sleep is not None:
        cmd.extend(["--sleep", str(sleep)])
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Stage {stage_name} completed successfully")
        if result.stdout:
            logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Stage {stage_name} failed: {e}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Multi-source weather fetching pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit matches per stage (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without running")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep between API calls (seconds)")
    parser.add_argument("--target-coverage", type=float, default=95.0, help="Target coverage percentage")
    parser.add_argument("--skip-meteostat", action="store_true", help="Skip Meteostat stage")
    parser.add_argument("--skip-open-meteo", action="store_true", help="Skip Open-Meteo stage")
    parser.add_argument("--skip-dwd", action="store_true", help="Skip DWD stage")
    args = parser.parse_args()
    
    db = get_db()
    
    logger.info("=== Multi-Source Weather Fetching Pipeline ===")
    
    # Initial coverage check
    initial_coverage = get_weather_coverage(db)
    matches_needing = get_matches_needing_weather(db)
    
    logger.info(f"Initial coverage: {initial_coverage:.2f}%")
    logger.info(f"Matches needing weather: {matches_needing}")
    
    if initial_coverage >= args.target_coverage:
        logger.success(f"Coverage already at {initial_coverage:.2f}% (target: {args.target_coverage}%)")
        return
    
    if matches_needing == 0:
        logger.info("No matches need weather data")
        return
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info(f"Would run stages to improve coverage from {initial_coverage:.2f}% to {args.target_coverage}%")
        return
    
    # Stage 1: Meteostat
    stage1_coverage = initial_coverage
    if not args.skip_meteostat:
        script_path = Path(__file__).parent / "fetch_weather_meteostat.py"
        if run_stage("Meteostat", script_path, limit=args.limit):
            stage1_coverage = get_weather_coverage(db)
            logger.info(f"After Meteostat: {stage1_coverage:.2f}% coverage")
            
            if stage1_coverage >= args.target_coverage:
                logger.success(f"Target coverage reached after Meteostat: {stage1_coverage:.2f}%")
                return
        else:
            logger.warning("Meteostat stage failed, continuing to next stage")
    
    # Stage 2: Open-Meteo
    stage2_coverage = stage1_coverage
    if not args.skip_open_meteo:
        script_path = Path(__file__).parent / "fetch_weather.py"
        if run_stage("Open-Meteo", script_path, limit=args.limit, sleep=args.sleep):
            stage2_coverage = get_weather_coverage(db)
            logger.info(f"After Open-Meteo: {stage2_coverage:.2f}% coverage")
            
            if stage2_coverage >= args.target_coverage:
                logger.success(f"Target coverage reached after Open-Meteo: {stage2_coverage:.2f}%")
                return
        else:
            logger.warning("Open-Meteo stage failed, continuing to next stage")
    
    # Stage 3: DWD
    stage3_coverage = stage2_coverage
    if not args.skip_dwd:
        script_path = Path(__file__).parent / "fetch_weather_dwd.py"
        if run_stage("DWD", script_path, limit=args.limit, sleep=args.sleep):
            stage3_coverage = get_weather_coverage(db)
            logger.info(f"After DWD: {stage3_coverage:.2f}% coverage")
        else:
            logger.warning("DWD stage failed")
    
    # Final summary
    final_coverage = get_weather_coverage(db)
    final_needing = get_matches_needing_weather(db)
    
    logger.info("=== Pipeline Summary ===")
    logger.info(f"Initial coverage: {initial_coverage:.2f}%")
    logger.info(f"Final coverage: {final_coverage:.2f}%")
    logger.info(f"Improvement: {final_coverage - initial_coverage:.2f}%")
    logger.info(f"Matches still needing weather: {final_needing}")
    
    # Source breakdown
    source_stats = db.execute_query("""
        SELECT 
            weather_source,
            COUNT(*) as count,
            AVG(weather_confidence) as avg_confidence
        FROM matches
        WHERE weather_source IS NOT NULL
        GROUP BY weather_source
        ORDER BY count DESC
    """)
    
    if source_stats:
        logger.info("\nWeather Source Breakdown:")
        for stat in source_stats:
            logger.info(f"  {stat['weather_source']}: {stat['count']} matches, avg confidence: {stat['avg_confidence']:.3f}")
    
    if final_coverage >= args.target_coverage:
        logger.success(f"✓ Target coverage achieved: {final_coverage:.2f}% >= {args.target_coverage}%")
    else:
        logger.warning(f"⚠ Target coverage not reached: {final_coverage:.2f}% < {args.target_coverage}%")
        logger.info(f"  Remaining gaps: {final_needing} matches")
    
    # Log to collection_logs
    db.log_collection(
        source='weather_multi',
        collection_type='weather_data',
        status='success' if final_coverage >= args.target_coverage else 'partial',
        records_collected=int(final_coverage - initial_coverage),
        error_message=None if final_coverage >= args.target_coverage else f"Coverage {final_coverage:.2f}% < target {args.target_coverage}%",
        started_at=datetime.now()
    )


if __name__ == "__main__":
    import sys
    print("This script is deprecated. Use: python main.py fetch-weather-multi [args]", file=sys.stderr)
    sys.exit(2)

