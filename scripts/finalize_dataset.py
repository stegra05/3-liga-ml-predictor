"""
Finalize 3. Liga Dataset
Waits for weather collection to complete, then validates and exports ML datasets
"""

import time
import subprocess
import sys
from pathlib import Path
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from database.db_manager import get_db
from scripts.processors.ml_data_exporter import MLDataExporter


def is_weather_collector_running():
    """Check if weather collector is still running"""
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    return 'weather_collector.py' in result.stdout


def wait_for_weather_collection():
    """Wait for weather collection to complete"""
    logger.info("Waiting for weather collection to complete...")

    check_interval = 60  # Check every minute
    while is_weather_collector_running():
        time.sleep(check_interval)

    logger.success("Weather collection completed!")


def validate_data_quality():
    """Validate final data quality"""
    logger.info("=== Validating Data Quality ===")

    db = get_db()

    # Get comprehensive stats
    query = """
    SELECT
        COUNT(*) as total_matches,
        COUNT(CASE WHEN is_finished = 1 THEN 1 END) as finished_matches,
        COUNT(CASE WHEN temperature_celsius IS NOT NULL THEN 1 END) as with_weather,
        COUNT(CASE WHEN bo.match_id IS NOT NULL THEN 1 END) as with_odds,
        COUNT(CASE WHEN ms.match_id IS NOT NULL THEN 1 END) as with_stats,
        COUNT(CASE WHEN tr.match_id IS NOT NULL THEN 1 END) as with_ratings
    FROM matches m
    LEFT JOIN betting_odds bo ON m.match_id = bo.match_id
    LEFT JOIN match_statistics ms ON m.match_id = ms.match_id AND m.home_team_id = ms.team_id
    LEFT JOIN team_ratings tr ON m.match_id = tr.match_id AND m.home_team_id = tr.team_id
    WHERE m.is_finished = 1
    """

    result = db.execute_query(query)[0]

    total = result['finished_matches']
    stats = {
        'Total Matches': result['total_matches'],
        'Finished Matches': total,
        'With Weather': f"{result['with_weather']} ({result['with_weather']/total*100:.1f}%)",
        'With Betting Odds': f"{result['with_odds']} ({result['with_odds']/total*100:.1f}%)",
        'With Statistics': f"{result['with_stats']/2} ({result['with_stats']/2/total*100:.1f}%)",  # Divide by 2 (home+away)
        'With Ratings': f"{result['with_ratings']/2} ({result['with_ratings']/2/total*100:.1f}%)"
    }

    logger.info("Data Quality Summary:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    return stats


def export_ml_datasets():
    """Export final ML-ready datasets"""
    logger.info("=== Exporting ML Datasets ===")

    exporter = MLDataExporter(output_dir="data/processed")

    # Export comprehensive dataset
    df = exporter.export_comprehensive_dataset(min_season="2009-2010")

    # Create temporal splits
    train, val, test = exporter.create_train_test_split(
        df,
        test_size=0.2,
        val_size=0.1,
        temporal=True
    )

    # Save datasets
    logger.info("Saving datasets...")
    df.to_csv("data/processed/3liga_full.csv", index=False)
    train.to_csv("data/processed/3liga_train.csv", index=False)
    val.to_csv("data/processed/3liga_val.csv", index=False)
    test.to_csv("data/processed/3liga_test.csv", index=False)

    logger.success(f"Exported datasets:")
    logger.success(f"  Full: {len(df)} matches, {len(df.columns)} features")
    logger.success(f"  Train: {len(train)} matches")
    logger.success(f"  Validation: {len(val)} matches")
    logger.success(f"  Test: {len(test)} matches")

    # Print feature counts by category
    logger.info("\nFeature counts:")
    logger.info(f"  Total features: {len(df.columns)}")
    logger.info(f"  Rating features: {len([c for c in df.columns if 'elo' in c or 'pi' in c or 'points' in c or 'goals_' in c])}")
    logger.info(f"  Statistics features: {len([c for c in df.columns if 'possession' in c or 'shots' in c or 'passes' in c])}")
    logger.info(f"  Weather features: {len([c for c in df.columns if 'temperature' in c or 'humidity' in c or 'precipitation' in c or 'wind' in c or 'weather' in c])}")
    logger.info(f"  Betting features: {len([c for c in df.columns if 'odds' in c or 'prob' in c])}")

    return df


def main():
    """Main execution"""
    logger.info("=== 3. Liga Dataset Finalization ===\n")

    # Step 1: Wait for weather collection
    wait_for_weather_collection()

    # Step 2: Validate data quality
    stats = validate_data_quality()

    # Step 3: Export ML datasets
    df = export_ml_datasets()

    logger.success("\n=== Dataset Finalization Complete! ===")
    logger.success(f"Final dataset: {len(df)} matches with {len(df.columns)} features")
    logger.success("Ready for machine learning training!")

    print("\n" + "="*60)
    print("3. LIGA DATASET READY!")
    print("="*60)
    print(f"Matches: {len(df)}")
    print(f"Features: {len(df.columns)}")
    print(f"Output: data/processed/3liga_*.csv")
    print("="*60)


if __name__ == "__main__":
    main()
