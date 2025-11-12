"""
Model evaluation framework for the 3. Liga Match Predictor.

This module implements multiple evaluation modes for backtesting:
1. Expanding Season: Train on all past data, test on each future season
2. Sliding Season: Train on fixed window of recent seasons
3. Rolling Matchday: Retrain after each matchday within a season
4. Static Pre-Season: Single model trained once, tested throughout season

Each mode simulates realistic prediction scenarios and tracks model performance.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from rich.console import Console
from rich.table import Table
from rich.progress import track

from . import metrics


def load_master_data(data_path: str = "data/processed/3liga_ml_dataset_full.csv") -> pd.DataFrame:
    """
    Load the master ML dataset.

    Args:
        data_path: Path to the full dataset CSV

    Returns:
        DataFrame sorted by match_datetime
    """
    df = pd.read_csv(data_path, parse_dates=["match_datetime"])

    # Sort by date to ensure temporal ordering
    df = df.sort_values("match_datetime").reset_index(drop=True)

    return df


def get_model(model_type: str = "rf_classifier") -> RandomForestClassifier:
    """
    Get a new model instance with production hyperparameters.

    This function returns a *new* untrained model with the same hyperparameters
    as the production model. This ensures we're testing the training strategy,
    not just loading a pre-trained model.

    Args:
        model_type: Type of model (currently only 'rf_classifier' supported)

    Returns:
        Untrained RandomForestClassifier with production hyperparameters
    """
    if model_type == "rf_classifier":
        # These hyperparameters should match your production model
        # Adjust if your model uses different parameters
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class CompatibleUnpickler(pickle.Unpickler):
    """Custom unpickler that handles module name changes."""
    
    def find_class(self, module, name):
        # Map old module name to new module name
        if module.startswith('kicktipp_predictor'):
            module = module.replace('kicktipp_predictor', 'liga_predictor')
        # Map old models module to new modeling module
        if module.startswith('liga_predictor.models'):
            module = module.replace('liga_predictor.models', 'liga_predictor.modeling')
        return super().find_class(module, name)


def load_production_model(model_path: str = "models/rf_classifier.pkl") -> Tuple[RandomForestClassifier, Dict]:
    """
    Load the pre-trained production model.

    This is used only for the 'static-preseason' evaluation mode.
    Handles module name changes from kicktipp_predictor to liga_predictor.

    Args:
        model_path: Path to the pickled model file

    Returns:
        Tuple of (RandomForestClassifier model, saved metadata dict)
    """
    with open(model_path, "rb") as f:
        unpickler = CompatibleUnpickler(f)
        saved = unpickler.load()

    # The saved file contains a dictionary with 'model', 'features', etc.
    if isinstance(saved, dict) and 'model' in saved:
        model_obj = saved['model']
        # Check if it's a ClassifierExperiment wrapper
        if hasattr(model_obj, 'rf_model'):
            # Extract the actual RandomForestClassifier from ClassifierExperiment
            return model_obj.rf_model, saved
        else:
            # Assume it's already a scikit-learn model
            return model_obj, saved
    else:
        # Fallback: assume it's already the model object
        return saved, {}


def prepare_features_target(
    df: pd.DataFrame, feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prepare features and target from dataset.

    Args:
        df: Input DataFrame
        feature_cols: List of feature column names. If None, automatically detected.

    Returns:
        Tuple of (X_features, y_target)
    """
    # Target column
    target_col = "target_multiclass"

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    y = df[target_col].values

    # Auto-detect feature columns if not provided
    if feature_cols is None:
        # Exclude non-feature columns
        exclude_cols = [
            "match_id",
            "season",
            "matchday",
            "match_datetime",
            "home_team",
            "away_team",
            "home_team_id",
            "away_team_id",
            "home_goals",
            "away_goals",
            "result",
            "venue",
            # All target columns
            "target_home_win",
            "target_draw",
            "target_away_win",
            "target_multiclass",
            "target_home_goals",
            "target_away_goals",
            "target_total_goals",
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].copy()

    # Only keep numeric columns (float64, int64, bool)
    numeric_cols = X.select_dtypes(include=["float64", "int64", "bool"]).columns.tolist()
    X = X[numeric_cols].copy()

    # Handle missing values (fill with median)
    # Suppress numpy warnings for empty slices
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
        for col in X.columns:
            if X[col].isna().any():
                # Check if column has any non-NaN values before computing median
                non_null_values = X[col].dropna()
                if len(non_null_values) == 0:
                    # All values are NaN, fill with 0
                    X[col] = X[col].fillna(0)
                else:
                    median_val = non_null_values.median()
                    if pd.isna(median_val):
                        # If median is still NaN (shouldn't happen), fill with 0
                        X[col] = X[col].fillna(0)
                    else:
                        X[col] = X[col].fillna(median_val)

    return X, y


def calculate_all_metrics(
    df_fold: pd.DataFrame, y_true: np.ndarray, y_pred_classes: np.ndarray, y_pred_probs: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics for a fold.

    Args:
        df_fold: DataFrame for this fold (needed for P&L calculation)
        y_true: True class labels
        y_pred_classes: Predicted class labels
        y_pred_probs: Predicted probabilities

    Returns:
        Dictionary of all metrics
    """
    results = {}

    # Classification metrics
    results["accuracy"] = metrics.calc_accuracy(y_true, y_pred_classes)
    results["logloss"] = metrics.calc_logloss(y_true, y_pred_probs)
    results["brier"] = metrics.calc_brier_score(y_true, y_pred_probs)
    results["rps"] = metrics.calc_rps(y_true, y_pred_probs)

    # P&L simulation
    pnl_results = metrics.calc_pnl(df_fold, y_pred_probs)
    results.update(
        {
            "pnl_total": pnl_results["total_pnl"],
            "pnl_num_bets": pnl_results["num_bets"],
            "pnl_win_rate": pnl_results["win_rate"],
            "pnl_roi": pnl_results["roi"],
            "pnl_avg_odds": pnl_results["avg_odds"],
        }
    )

    # Per-class metrics
    per_class = metrics.get_per_class_metrics(y_true, y_pred_classes)
    for class_name, class_metrics in per_class.items():
        for metric_name, value in class_metrics.items():
            results[f"{class_name}_{metric_name}"] = value

    return results


def run_expanding_season(
    df: pd.DataFrame,
    start_season: int,
    console: Console,
    log_mlflow: bool = False,
) -> List[Dict]:
    """
    Mode 1: Expanding Season Evaluation.

    Train on all data from start_season up to (test_season - 1),
    test on each future season.

    Args:
        df: Full dataset
        start_season: First season to use in training (integer year, e.g., 2014)
        console: Rich console for output
        log_mlflow: Whether to log to MLflow

    Returns:
        List of result dictionaries for each test season
    """
    console.print("\n[bold cyan]Mode 1: Expanding Season[/bold cyan]")
    console.print(f"Training starts from season {start_season}")

    # Convert start_season to string format "YYYY-YYYY+1"
    start_season_str = f"{start_season}-{start_season+1}"

    # Get unique seasons
    seasons = sorted(df["season"].unique())

    # Filter test seasons (must be after start_season + 1 year for training)
    test_seasons = [s for s in seasons if s > start_season_str]

    results = []

    for test_season in track(test_seasons, description="Evaluating seasons"):
        # Split data - only train on data from start_season onwards
        train_df = df[(df["season"] >= start_season_str) & (df["season"] < test_season)].copy()
        test_df = df[df["season"] == test_season].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            console.print(f"[yellow]Skipping season {test_season}: insufficient data[/yellow]")
            continue

        # Prepare features and target
        X_train, y_train = prepare_features_target(train_df)
        X_test, y_test = prepare_features_target(test_df)

        # Train model
        model = get_model()
        model.fit(X_train, y_train)

        # Predict
        y_pred_classes = model.predict(X_test)
        y_pred_probs = model.predict_proba(X_test)

        # Calculate metrics
        fold_metrics = calculate_all_metrics(test_df, y_test, y_pred_classes, y_pred_probs)
        fold_metrics["fold"] = f"Season {test_season}"
        fold_metrics["train_size"] = len(train_df)
        fold_metrics["test_size"] = len(test_df)

        results.append(fold_metrics)

        # Log to MLflow if requested
        if log_mlflow:
            log_to_mlflow(
                run_name=f"expanding_season_{test_season}",
                params={"mode": "expanding-season", "test_season": test_season, "start_season": start_season},
                metrics=fold_metrics,
            )

    return results


def run_sliding_season(
    df: pd.DataFrame,
    start_season: int,
    window_size: int,
    console: Console,
    log_mlflow: bool = False,
) -> List[Dict]:
    """
    Mode 2: Sliding Season Evaluation.

    Train on a fixed window of recent seasons, test on the next season.
    Tests if older data becomes harmful (concept drift).

    Args:
        df: Full dataset
        start_season: First season to use (integer year)
        window_size: Number of seasons in training window
        console: Rich console for output
        log_mlflow: Whether to log to MLflow

    Returns:
        List of result dictionaries for each test season
    """
    console.print("\n[bold cyan]Mode 2: Sliding Season[/bold cyan]")
    console.print(f"Window size: {window_size} seasons")

    # Convert integer season years to string format for comparison
    seasons = sorted(df["season"].unique())

    # Parse start year from season strings for window calculation
    def get_start_year(season_str):
        return int(season_str.split("-")[0])

    # Filter test seasons that have enough history for the window
    test_seasons = [s for s in seasons if get_start_year(s) >= start_season + window_size]

    results = []

    for test_season in track(test_seasons, description="Evaluating seasons"):
        # Train on previous window_size seasons
        test_year = get_start_year(test_season)
        train_start_year = test_year - window_size
        train_start_str = f"{train_start_year}-{train_start_year+1}"

        train_df = df[(df["season"] >= train_start_str) & (df["season"] < test_season)].copy()
        test_df = df[df["season"] == test_season].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        # Prepare and train
        X_train, y_train = prepare_features_target(train_df)
        X_test, y_test = prepare_features_target(test_df)

        model = get_model()
        model.fit(X_train, y_train)

        # Predict
        y_pred_classes = model.predict(X_test)
        y_pred_probs = model.predict_proba(X_test)

        # Calculate metrics
        fold_metrics = calculate_all_metrics(test_df, y_test, y_pred_classes, y_pred_probs)
        fold_metrics["fold"] = f"Season {test_season}"
        fold_metrics["train_size"] = len(train_df)
        fold_metrics["test_size"] = len(test_df)

        results.append(fold_metrics)

        if log_mlflow:
            log_to_mlflow(
                run_name=f"sliding_season_{test_season}",
                params={
                    "mode": "sliding-season",
                    "test_season": test_season,
                    "window_size": window_size,
                },
                metrics=fold_metrics,
            )

    return results


def run_rolling_matchday(
    df: pd.DataFrame,
    start_season: int,
    test_season: int,
    console: Console,
    log_mlflow: bool = False,
) -> List[Dict]:
    """
    Mode 3: Rolling Matchday Evaluation.

    Train on all past data, test on each matchday of the test season,
    then add that matchday to training data and retrain for the next matchday.

    Simulates the most realistic scenario where model is retrained regularly.

    Args:
        df: Full dataset
        start_season: First season to use in initial training (integer year)
        test_season: Season to test on (integer year)
        console: Rich console for output
        log_mlflow: Whether to log to MLflow

    Returns:
        List of result dictionaries for each matchday
    """
    console.print("\n[bold cyan]Mode 3: Rolling Matchday[/bold cyan]")
    console.print(f"Testing season {test_season}, retraining after each matchday")

    # Convert to season string format
    start_season_str = f"{start_season}-{start_season+1}"
    test_season_str = f"{test_season}-{test_season+1}"

    # Initial training data: all data from start_season before test season
    train_df = df[(df["season"] >= start_season_str) & (df["season"] < test_season_str)].copy()

    # Get all matchdays in test season
    test_season_df = df[df["season"] == test_season_str].copy()
    matchdays = sorted(test_season_df["matchday"].unique())

    results = []

    for matchday in track(matchdays, description="Evaluating matchdays"):
        # Test on current matchday
        test_df = test_season_df[test_season_df["matchday"] == matchday].copy()

        if len(test_df) == 0:
            continue

        # Prepare and train
        X_train, y_train = prepare_features_target(train_df)
        X_test, y_test = prepare_features_target(test_df)

        model = get_model()
        model.fit(X_train, y_train)

        # Predict
        y_pred_classes = model.predict(X_test)
        y_pred_probs = model.predict_proba(X_test)

        # Calculate metrics
        fold_metrics = calculate_all_metrics(test_df, y_test, y_pred_classes, y_pred_probs)
        fold_metrics["fold"] = f"MD {matchday}"
        fold_metrics["train_size"] = len(train_df)
        fold_metrics["test_size"] = len(test_df)

        results.append(fold_metrics)

        # Add current matchday to training data for next iteration
        train_df = pd.concat([train_df, test_df], ignore_index=True)

        if log_mlflow:
            log_to_mlflow(
                run_name=f"rolling_matchday_{test_season}_MD{matchday}",
                params={
                    "mode": "rolling-matchday",
                    "test_season": test_season,
                    "matchday": matchday,
                },
                metrics=fold_metrics,
            )

    return results


def run_static_preseason(
    df: pd.DataFrame,
    start_season: int,
    test_season: int,
    console: Console,
    use_production_model: bool = True,
    log_mlflow: bool = False,
) -> List[Dict]:
    """
    Mode 4: Static Pre-Season Evaluation.

    Train a single model on all past data, then test on each matchday of the
    test season WITHOUT retraining. Shows how model performance decays during
    the season as it becomes "stale".

    Args:
        df: Full dataset
        start_season: First season to use in training (integer year)
        test_season: Season to test on (integer year)
        console: Rich console for output
        use_production_model: If True, load models/rf_classifier.pkl instead of training
        log_mlflow: Whether to log to MLflow

    Returns:
        List of result dictionaries for each matchday
    """
    console.print("\n[bold cyan]Mode 4: Static Pre-Season[/bold cyan]")
    console.print(f"Testing season {test_season} with a single static model")

    # Convert to season string format
    start_season_str = f"{start_season}-{start_season+1}"
    test_season_str = f"{test_season}-{test_season+1}"

    # Train once on all data from start_season before test season
    train_df = df[(df["season"] >= start_season_str) & (df["season"] < test_season_str)].copy()

    if use_production_model:
        console.print("[yellow]Loading production model from models/rf_classifier.pkl[/yellow]")
        model, saved_metadata = load_production_model()
    else:
        console.print("[yellow]Training new model on historical data[/yellow]")
        X_train, y_train = prepare_features_target(train_df)
        model = get_model()
        model.fit(X_train, y_train)
        saved_metadata = {}

    # Test on each matchday WITHOUT retraining
    test_season_df = df[df["season"] == test_season_str].copy()
    matchdays = sorted(test_season_df["matchday"].unique())

    results = []

    for matchday in track(matchdays, description="Evaluating matchdays (static model)"):
        test_df = test_season_df[test_season_df["matchday"] == matchday].copy()

        if len(test_df) == 0:
            continue

        # Prepare and predict (no training!)
        X_test, y_test = prepare_features_target(test_df)

        # Filter features to match what the model was trained on
        # If we have saved metadata with features list, use that for proper ordering
        if saved_metadata and 'features' in saved_metadata:
            # Use saved features list to ensure proper ordering
            saved_features = saved_metadata['features']
            # Only use features that exist in X_test
            available_features = [f for f in saved_features if f in X_test.columns]
            X_test = X_test[available_features]
        else:
            # Fallback: filter to NUMERICAL_FEATURES (same as ClassifierExperiment does)
            from liga_predictor import config
            numerical_features = [f for f in X_test.columns if f in config.NUMERICAL_FEATURES]
            X_test = X_test[numerical_features]

        y_pred_classes = model.predict(X_test)
        y_pred_probs = model.predict_proba(X_test)

        # Calculate metrics
        fold_metrics = calculate_all_metrics(test_df, y_test, y_pred_classes, y_pred_probs)
        fold_metrics["fold"] = f"MD {matchday}"
        fold_metrics["train_size"] = len(train_df)
        fold_metrics["test_size"] = len(test_df)

        results.append(fold_metrics)

        if log_mlflow:
            log_to_mlflow(
                run_name=f"static_preseason_{test_season}_MD{matchday}",
                params={
                    "mode": "static-preseason",
                    "test_season": test_season,
                    "matchday": matchday,
                    "use_production_model": use_production_model,
                },
                metrics=fold_metrics,
            )

    return results


def format_results_table(results: List[Dict], console: Console) -> None:
    """
    Format and display results as a Rich table.

    Args:
        results: List of result dictionaries from evaluation
        console: Rich console for output
    """
    if not results:
        console.print("[red]No results to display[/red]")
        return

    table = Table(title="Evaluation Results")

    # Add columns
    table.add_column("Fold", style="cyan", no_wrap=True)
    table.add_column("Samples", style="magenta")
    table.add_column("Accuracy", style="green")
    table.add_column("RPS", style="green")
    table.add_column("Brier", style="green")
    table.add_column("LogLoss", style="green")
    table.add_column("P&L (units)", style="yellow")
    table.add_column("# Bets", style="yellow")
    table.add_column("ROI %", style="yellow")

    # Add rows
    for result in results:
        table.add_row(
            result.get("fold", "N/A"),
            str(result.get("test_size", 0)),
            f"{result.get('accuracy', 0):.3f}",
            f"{result.get('rps', 0):.4f}",
            f"{result.get('brier', 0):.4f}",
            f"{result.get('logloss', 0):.4f}",
            f"{result.get('pnl_total', 0):+.2f}",
            str(result.get("pnl_num_bets", 0)),
            f"{result.get('pnl_roi', 0):+.2f}",
        )

    # Add summary row
    if len(results) > 1:
        avg_accuracy = np.mean([r["accuracy"] for r in results])
        avg_rps = np.mean([r["rps"] for r in results])
        avg_brier = np.mean([r["brier"] for r in results])
        avg_logloss = np.mean([r["logloss"] for r in results])
        total_pnl = sum([r["pnl_total"] for r in results])
        total_bets = sum([r["pnl_num_bets"] for r in results])
        total_samples = sum([r["test_size"] for r in results])

        # Calculate weighted average ROI
        total_staked = total_bets  # Assuming 1 unit stake
        avg_roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0.0

        table.add_row(
            "[bold]AVERAGE[/bold]",
            f"[bold]{total_samples}[/bold]",
            f"[bold]{avg_accuracy:.3f}[/bold]",
            f"[bold]{avg_rps:.4f}[/bold]",
            f"[bold]{avg_brier:.4f}[/bold]",
            f"[bold]{avg_logloss:.4f}[/bold]",
            f"[bold]{total_pnl:+.2f}[/bold]",
            f"[bold]{total_bets}[/bold]",
            f"[bold]{avg_roi:+.2f}[/bold]",
        )

    console.print(table)


def log_to_mlflow(run_name: str, params: Dict, metrics: Dict) -> None:
    """
    Log a run to MLflow.

    Args:
        run_name: Name for this run
        params: Dictionary of parameters to log
        metrics: Dictionary of metrics to log (may contain non-numeric values)
    """
    try:
        import mlflow
        import warnings
        
        # Suppress MLflow filesystem backend deprecation warning
        warnings.filterwarnings('ignore', category=FutureWarning, module='mlflow')

        # Separate numeric metrics from non-numeric metadata
        numeric_metrics = {}
        tags = {}
        
        for key, value in metrics.items():
            # Try to convert to float - if it works, it's numeric
            try:
                # Handle numpy types by converting to Python native type
                if hasattr(value, 'item'):
                    numeric_value = float(value.item())
                else:
                    numeric_value = float(value)
                numeric_metrics[key] = numeric_value
            except (ValueError, TypeError, AttributeError):
                # Non-numeric values go to tags
                tags[key] = str(value)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(numeric_metrics)
            if tags:
                mlflow.set_tags(tags)
    except ImportError:
        pass  # MLflow not installed, skip logging


def run_evaluation(
    mode: str,
    start_season: int,
    test_season: int,
    window_size: int,
    log_mlflow: bool,
    console: Console,
) -> None:
    """
    Main entry point for running evaluations.

    Args:
        mode: Evaluation mode (expanding-season, sliding-season, rolling-matchday, static-preseason)
        start_season: First season to use in training
        test_season: Season to test for rolling/static modes
        window_size: Window size for sliding mode
        log_mlflow: Whether to log to MLflow
        console: Rich console for output
    """
    # Load data
    console.print("[cyan]Loading master dataset...[/cyan]")
    df = load_master_data()

    console.print(f"Loaded {len(df)} matches from {df['season'].min()} to {df['season'].max()}")

    # Run selected mode
    if mode == "expanding-season":
        results = run_expanding_season(df, start_season, console, log_mlflow)
    elif mode == "sliding-season":
        results = run_sliding_season(df, start_season, window_size, console, log_mlflow)
    elif mode == "rolling-matchday":
        results = run_rolling_matchday(df, start_season, test_season, console, log_mlflow)
    elif mode == "static-preseason":
        results = run_static_preseason(df, start_season, test_season, console, log_mlflow=log_mlflow)
    else:
        console.print(f"[red]Unknown mode: {mode}[/red]")
        return

    # Display results
    format_results_table(results, console)

    # Baselines
    console.print("\n[bold cyan]Baseline Comparisons[/bold cyan]")
    start_season_str = f"{start_season}-{start_season+1}"
    baselines = metrics.run_all_baselines(df[df["season"] >= start_season_str])
    for name, baseline in baselines.items():
        console.print(f"  {baseline['description']}: {baseline['accuracy']:.3f}")
