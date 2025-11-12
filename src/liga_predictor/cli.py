#!/usr/bin/env python3
"""
3. Liga Match Predictor - Modern Typer-based CLI

This is the main CLI entry point for all functionality. Running without arguments
defaults to predicting matches.

Usage:
    liga-predictor                    # Predict next matchday (default)
    liga-predictor predict            # Explicit prediction
    liga-predictor collect fbref      # Collect FBref data
    liga-predictor process h2h        # Build head-to-head stats
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


@app.command()
def collect(
    source: Annotated[
        str,
        typer.Argument(help="Data source: fbref, openligadb, transfermarkt, oddsportal, fotmob"),
    ],
):
    """
    Collect data from specified source.

    Collects data from various sources for 3. Liga matches, teams, and statistics.
    """
    match source.lower():
        case "fbref":
            from liga_predictor.collection.fbref import FBrefCollector

            collector = FBrefCollector(use_selenium=True)
            collector.collect_all_seasons()
        case "openligadb":
            from liga_predictor.collection.openligadb import OpenLigaDBCollector
            from datetime import datetime

            collector = OpenLigaDBCollector()
            collector.collect_all_historical_data(start_year=2009, end_year=2024)
            current_year = datetime.now().year
            if datetime.now().month >= 7:  # Season starts around July
                collector.collect_season(str(current_year))
        case "transfermarkt":
            from liga_predictor.collection.transfermarkt import TransfermarktRefereeCollector

            collector = TransfermarktRefereeCollector()
            collector.collect_all_seasons()
        case "oddsportal":
            from liga_predictor.collection.oddsportal import OddsPortalCollector

            collector = OddsPortalCollector(use_selenium=True)
            collector.collect_recent_matches(days=7)
        case "fotmob":
            from liga_predictor.collection.fotmob import FotMobCollector

            collector = FotMobCollector()
            # Collect recent seasons (similar to other collectors)
            for year in range(2021, 2026):
                collector.collect_season(f"{year}-{year+1}")
        case _:
            logger.error(f"Unknown source: {source}")
            logger.info("Valid sources: fbref, openligadb, transfermarkt, oddsportal, fotmob")
            raise typer.Exit(code=1)


@app.command()
def process(
    step: Annotated[
        str,
        typer.Argument(help="Processing step: ml-export, weather, locations, h2h, ratings, unify"),
    ],
):
    """
    Run data processing step.

    Executes various data processing and feature engineering steps for the prediction system.
    """
    match step.lower():
        case "ml-export":
            from liga_predictor.processing.ml_export import MLDataExporter

            exporter = MLDataExporter()
            exporter.export_to_csv(save_splits=True)
        case "weather":
            from liga_predictor.processing.weather import get_db, run_weather_stage

            db = get_db()
            logger.info("=== Simplified Weather Fetching ===")
            run_weather_stage("meteostat", "meteostat")
            run_weather_stage("open_meteo", "open-meteo")
            run_weather_stage("dwd", "dwd")
            logger.success("Weather fetching complete")
        case "locations":
            from liga_predictor.processing.locations import TeamLocationBuilder

            builder = TeamLocationBuilder()
            builder.build_locations()
        case "h2h":
            from liga_predictor.processing.h2h import HeadToHeadBuilder

            logger.info("=== Building Head-to-Head table ===")
            builder = HeadToHeadBuilder()
            builder.compute_h2h()
            logger.success("Head-to-head statistics built successfully")
        case "ratings":
            from liga_predictor.processing.ratings import RatingCalculator

            calculator = RatingCalculator(initial_elo=1500.0, k_factor=32.0)
            calculator.calculate_all_ratings()
            stats = calculator.db.get_database_stats()
            logger.info("\n=== Database Statistics ===")
            for key, value in stats.items():
                logger.info(f"{key}: {value}")
        case "unify":
            from liga_predictor.processing.unify import TeamUnifier

            unifier = TeamUnifier()
            unifier.unify_all()
        case _:
            logger.error(f"Unknown processing step: {step}")
            logger.info("Valid steps: ml-export, weather, locations, h2h, ratings, unify")
            raise typer.Exit(code=1)


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
def main(
    ctx: typer.Context,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Enable debug logging with verbose output",
            rich_help_panel="General options",
        ),
    ] = False,
):
    """
    3. Liga Match Predictor - Modern CLI

    A comprehensive machine learning system for predicting 3. Liga match results.
    Includes data collection from multiple sources, feature engineering, and
    prediction using Random Forest Classifier.

    Running without a command defaults to predicting the next matchday.
    """
    # Configure logging: INFO by default, DEBUG only with --debug
    from sys import stderr

    logger.remove()
    if debug:
        logger.add(
            stderr,
            level="DEBUG",
            colorize=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )
        logger.debug("Debug logging enabled")
    else:
        logger.add(
            stderr,
            level="INFO",
            colorize=True,
            format="<level>{message}</level>",
        )

    if ctx.invoked_subcommand is None:
        # Default behavior: run predict with no arguments
        ctx.invoke(predict)


def cli():
    """Entry point for the CLI (used by setup.py console_scripts)"""
    app()


if __name__ == "__main__":
    app()
