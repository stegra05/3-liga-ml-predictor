#!/usr/bin/env python3
"""
3. Liga Match Predictor - Modern Typer-based CLI

This is the main CLI entry point for all functionality. Running without arguments
defaults to predicting matches.

Usage:
    liga-predictor                    # Predict next matchday (default)
    liga-predictor predict            # Explicit prediction
    liga-predictor predict --help     # Show prediction options
    liga-predictor collect-fbref      # Collect FBref data
    liga-predictor --help             # List all available commands
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated
from loguru import logger

# Ensure parent directory is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = typer.Typer(
    name="liga-predictor",
    help="3. Liga Match Prediction System",
    no_args_is_help=False,  # Allow running without args (defaults to predict)
    add_completion=False,
)


@app.command()
def predict(
    season: Annotated[
        Optional[str],
        typer.Option(
            help="Season to predict (e.g., 2025-2026 or 2025). Default: current season"
        ),
    ] = None,
    matchday: Annotated[
        Optional[int],
        typer.Option(help="Specific matchday to predict. Default: next upcoming matchday"),
    ] = None,
    update_data: Annotated[
        bool,
        typer.Option("--update-data", help="Update/fetch data for the matchday before predicting"),
    ] = False,
    output: Annotated[
        Optional[str],
        typer.Option(help="Save predictions to CSV file"),
    ] = None,
    weather_mode: Annotated[
        str,
        typer.Option(
            help="Weather fetching mode: live (forecast API), estimate (historical), or off (defaults)"
        ),
    ] = "live",
    ext_data: Annotated[
        bool,
        typer.Option(
            "--ext-data",
            help="Include heavy external data collection (FBref, matchday-level standings)",
        ),
    ] = False,
):
    """
    Predict match results (default command).

    Uses the trained Random Forest Classifier model to predict upcoming 3. Liga match results.
    By default, predicts the next upcoming matchday of the current season.
    """
    from liga_predictor.predictor import MatchPredictor

    # Validate weather mode
    if weather_mode not in ["live", "estimate", "off"]:
        logger.error(f"Invalid weather mode: {weather_mode}. Must be 'live', 'estimate', or 'off'")
        raise typer.Exit(code=1)

    # Initialize predictor
    predictor = MatchPredictor(weather_mode=weather_mode, ext_data=ext_data)

    # Normalize season format if provided as start year (e.g., "2025" -> "2025-2026")
    season_arg = season
    if season_arg and season_arg.isdigit() and len(season_arg) == 4:
        start_year = int(season_arg)
        season_arg = f"{start_year}-{start_year+1}"

    # Determine which matchday to predict
    if matchday and season_arg:
        target_season = season_arg
        target_matchday = matchday
    elif matchday:
        target_season = predictor.get_current_season()
        target_matchday = matchday
    else:
        target_season, target_matchday, has_data = predictor.find_next_matchday(season_arg)

    logger.info(f"Target: {target_season} Matchday {target_matchday}")

    # Check data availability
    data_status = predictor.check_data_availability(target_season, target_matchday)
    logger.info(
        f"Data availability: exists={data_status['exists']}, can_predict={data_status['can_predict']}"
    )

    # Update data if requested
    if update_data:
        logger.info("--update-data flag set, running data acquisition pipeline...")
        success = predictor.update_matchday_data(target_season, target_matchday)
        if not success:
            logger.error("Data acquisition failed, but continuing with prediction attempt...")
        # Re-check availability after update
        data_status = predictor.check_data_availability(target_season, target_matchday)

    # If no data available, acquire it
    if not data_status["exists"] or not data_status["can_predict"]:
        logger.warning(f"No data available for {target_season} MD {target_matchday}")

        if not update_data:
            logger.info(
                f"To acquire data, run with --update-data flag:\n"
                f"  liga-predictor predict --season {target_season} --matchday {target_matchday} --update-data"
            )
            raise typer.Exit(code=1)

    # Make predictions
    predictions = predictor.predict_matches(target_season, target_matchday)

    if predictions is None or len(predictions) == 0:
        logger.error(f"No predictions generated for {target_season} MD {target_matchday}")
        raise typer.Exit(code=1)

    # Display predictions
    predictor.print_predictions(predictions)

    # Save to file if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        logger.success(f"Predictions saved to: {output_path}")


@app.command()
def evaluate(
    mode: Annotated[
        str,
        typer.Option(
            help="Evaluation mode: expanding-season, sliding-season, rolling-matchday, static-preseason"
        ),
    ] = "expanding-season",
    start_season: Annotated[
        int,
        typer.Option(help="First season to use in the training set. 2014+ has richer data."),
    ] = 2014,
    test_season: Annotated[
        int,
        typer.Option(
            help="The single season to test for 'rolling-matchday' or 'static-preseason' modes."
        ),
    ] = 2025,
    window_size: Annotated[
        int,
        typer.Option(help="Number of seasons to use in 'sliding-season' mode."),
    ] = 4,
    log_mlflow: Annotated[
        bool,
        typer.Option(help="Log results to MLflow."),
    ] = True,
):
    """
    Run a backtest evaluation of the prediction model.

    This command runs comprehensive evaluations to test model performance across
    different time periods and training strategies. Use this to validate your
    model and understand how it performs in realistic scenarios.

    Evaluation modes:
    - expanding-season: Train on all past data, test on each future season
    - sliding-season: Train on fixed window of recent seasons
    - rolling-matchday: Retrain after each matchday within a season
    - static-preseason: Single model trained once, tested throughout season
    """
    from rich.console import Console
    from liga_predictor import evaluation

    console = Console()

    # Validate mode
    valid_modes = ["expanding-season", "sliding-season", "rolling-matchday", "static-preseason"]
    if mode not in valid_modes:
        console.print(f"[red]Invalid mode: {mode}[/red]")
        console.print(f"Valid modes: {', '.join(valid_modes)}")
        raise typer.Exit(code=1)

    console.print(f"[bold green]Starting evaluation in {mode} mode...[/bold green]")

    # Run evaluation
    evaluation.run_evaluation(
        mode=mode,
        start_season=start_season,
        test_season=test_season,
        window_size=window_size,
        log_mlflow=log_mlflow,
        console=console,
    )

    console.print("\n[bold green]✓ Evaluation complete![/bold green]")

    if log_mlflow:
        console.print("\n[cyan]To view results in MLflow UI, run:[/cyan]")
        console.print("  mlflow ui")
        console.print("  Then open http://localhost:5000 in your browser")


@app.command("collect-fbref")
def collect_fbref(
    ctx: typer.Context,
):
    """
    Collect data from FBref.

    Collects team standings and player statistics from FBref for 3. Liga seasons.
    Pass additional arguments after the command for collector-specific options.
    """
    from liga_predictor.collection.fbref import FBrefCollector, main

    # Call the module's main function
    main()


@app.command("collect-openligadb")
def collect_openligadb(
    ctx: typer.Context,
):
    """
    Collect data from OpenLigaDB API.

    Collects match data, scores, and fixtures from the OpenLigaDB API for 3. Liga.
    Pass additional arguments after the command for collector-specific options.
    """
    from liga_predictor.collection.openligadb import OpenLigaDBCollector, main

    # Call the module's main function
    main()


@app.command("collect-transfermarkt")
def collect_transfermarkt(
    ctx: typer.Context,
):
    """
    Collect referee data from Transfermarkt.

    Collects referee information and statistics from Transfermarkt for 3. Liga matches.
    Pass additional arguments after the command for collector-specific options.
    """
    from liga_predictor.collection.transfermarkt import TransfermarktRefereeCollector, main

    # Call the module's main function
    main()


@app.command("collect-oddsportal")
def collect_oddsportal(
    ctx: typer.Context,
):
    """
    Collect betting odds from OddsPortal.

    Collects 1X2 betting odds from OddsPortal.com for 3. Liga matches.
    Pass additional arguments after the command for collector-specific options.
    """
    from liga_predictor.collection.oddsportal import OddsPortalCollector, main

    # Call the module's main function
    main()


@app.command("export-ml-data")
def export_ml_data(
    ctx: typer.Context,
):
    """
    Export ML training data.

    Exports processed match data with features for machine learning model training.
    Creates a comprehensive dataset with all features needed for prediction.
    Pass additional arguments after the command for exporter-specific options.
    """
    from liga_predictor.processing.ml_export import MLDataExporter, main

    # Call the module's main function
    main()


@app.command("fetch-weather")
def fetch_weather(
    limit: Annotated[
        Optional[int],
        typer.Option(help="Limit matches per stage (for testing)"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be done without running"),
    ] = False,
    sleep: Annotated[
        float,
        typer.Option(help="Sleep between API calls (seconds)"),
    ] = 1.0,
    target_coverage: Annotated[
        float,
        typer.Option(help="Target coverage percentage"),
    ] = 95.0,
    skip_meteostat: Annotated[
        bool,
        typer.Option("--skip-meteostat", help="Skip Meteostat stage"),
    ] = False,
    skip_open_meteo: Annotated[
        bool,
        typer.Option("--skip-open-meteo", help="Skip Open-Meteo stage"),
    ] = False,
    skip_dwd: Annotated[
        bool,
        typer.Option("--skip-dwd", help="Skip DWD stage"),
    ] = False,
):
    """
    Fetch weather data from multiple sources.

    Runs a multi-source weather fetching pipeline to collect historical weather data
    for matches. Uses Meteostat, Open-Meteo, and DWD sources to maximize coverage.
    """
    from liga_predictor.processing.weather import get_db, get_weather_coverage, get_matches_needing_weather, run_weather_stage
    from datetime import datetime

    db = get_db()

    logger.info("=== Multi-Source Weather Fetching Pipeline ===")

    # Initial coverage check
    initial_coverage = get_weather_coverage(db)
    matches_needing = get_matches_needing_weather(db)

    logger.info(f"Initial coverage: {initial_coverage:.2f}%")
    logger.info(f"Matches needing weather: {matches_needing}")

    if dry_run:
        logger.info("Dry run mode - showing stages that would be executed:")
        if not skip_meteostat:
            logger.info("  1. Meteostat")
        if not skip_open_meteo:
            logger.info("  2. Open-Meteo")
        if not skip_dwd:
            logger.info("  3. DWD")
        return

    # Run stages
    if not skip_meteostat:
        logger.info("\n=== Stage 1: Meteostat ===")
        run_weather_stage("meteostat", "meteostat", limit=limit, sleep=sleep)

    if not skip_open_meteo:
        logger.info("\n=== Stage 2: Open-Meteo ===")
        run_weather_stage("open_meteo", "open-meteo", limit=limit, sleep=sleep)

    if not skip_dwd:
        logger.info("\n=== Stage 3: DWD ===")
        run_weather_stage("dwd", "dwd", limit=limit, sleep=sleep)

    # Final coverage check
    final_coverage = get_weather_coverage(db)
    final_needing = get_matches_needing_weather(db)

    logger.info(f"\n=== Final Results ===")
    logger.info(f"Initial coverage: {initial_coverage:.2f}%")
    logger.info(f"Final coverage: {final_coverage:.2f}%")
    logger.info(f"Improvement: +{final_coverage - initial_coverage:.2f}%")
    logger.info(f"Matches still needing weather: {final_needing}")

    if final_coverage >= target_coverage:
        logger.success(
            f"✓ Target coverage achieved: {final_coverage:.2f}% >= {target_coverage}%"
        )
    else:
        logger.warning(
            f"⚠ Target coverage not reached: {final_coverage:.2f}% < {target_coverage}%"
        )
        logger.info(f"  Remaining gaps: {final_needing} matches")


@app.command("build-locations")
def build_locations(
    ctx: typer.Context,
):
    """
    Build team location mappings.

    Builds and updates team location data including coordinates for travel distance
    calculations and weather lookups.
    """
    from liga_predictor.processing.locations import main

    # Call the module's main function
    main()


@app.command("build-h2h")
def build_h2h(
    ctx: typer.Context,
):
    """
    Build head-to-head statistics.

    Computes head-to-head match statistics between all team pairs from historical
    match data. Creates a comprehensive H2H table for prediction features.
    """
    from liga_predictor.processing.h2h import HeadToHeadBuilder

    logger.info("=== Building Head-to-Head table ===")
    builder = HeadToHeadBuilder()
    builder.compute_h2h()
    logger.success("Head-to-head statistics built successfully")


@app.command("calculate-ratings")
def calculate_ratings(
    ctx: typer.Context,
):
    """
    Calculate team ratings.

    Calculates and updates team rating statistics based on recent performance.
    Computes Elo-style ratings and other performance metrics for prediction features.
    """
    from liga_predictor.processing.ratings import RatingCalculator, main

    # Call the module's main function
    main()


@app.command("db-init")
def db_init():
    """
    Initialize database schema.

    Creates the SQLite database schema and all required tables. Run this command
    when setting up the system for the first time or after major schema changes.
    """
    from liga_predictor.database import DatabaseManager

    logger.info("Initializing database schema...")
    db = DatabaseManager()
    db.initialize_schema()
    logger.success("Database initialized successfully!")

    # Print stats
    stats = db.get_database_stats()
    logger.info("\nDatabase Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    3. Liga Match Predictor - Modern CLI

    A comprehensive machine learning system for predicting 3. Liga match results.
    Includes data collection from multiple sources, feature engineering, and
    prediction using Random Forest Classifier.

    Running without a command defaults to predicting the next matchday.
    """
    if ctx.invoked_subcommand is None:
        # Default behavior: run predict with no arguments
        ctx.invoke(predict)


def cli():
    """Entry point for the CLI (used by setup.py console_scripts)"""
    app()


if __name__ == "__main__":
    app()
