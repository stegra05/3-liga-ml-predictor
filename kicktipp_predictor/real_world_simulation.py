"""
Real-World Simulation Module
Simulates actual Kicktipp usage: predict each matchday sequentially
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .evaluation import calculate_average_kicktipp_points


def simulate_season_predictions(
    full_data: pd.DataFrame,
    season: str,
    model_trainer,
    model_name: str,
    verbose: bool = True
) -> Dict:
    """
    Simulate real-world predictions for a season, matchday by matchday

    For each matchday:
    1. Train on all data BEFORE this matchday
    2. Predict this matchday's matches
    3. Evaluate predictions
    4. Move to next matchday

    Args:
        full_data: Complete dataset (train + val + test)
        season: Season to simulate (e.g., "2025-2026")
        model_trainer: Function that trains and returns a model
        model_name: Name for reporting
        verbose: Print progress

    Returns:
        Dictionary with simulation results
    """
    # Filter to target season
    season_mask = full_data['season'] == season
    season_data = full_data[season_mask].copy()

    if len(season_data) == 0:
        print(f"Warning: No data found for season {season}")
        return {}

    # Sort by match datetime to ensure correct order
    season_data = season_data.sort_values('match_datetime')

    # Get unique matchdays
    matchdays = sorted(season_data['matchday'].unique())

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"REAL-WORLD SIMULATION: {season}")
        print(f"Model: {model_name}")
        print(f"{'=' * 70}")
        print(f"\nSimulating {len(matchdays)} matchdays...")

    # Store results for each matchday
    matchday_results = []
    cumulative_points = []

    for matchday in matchdays:
        # Get training data: everything BEFORE this matchday
        # (includes all previous seasons and previous matchdays of current season)
        train_mask = (
            (full_data['season'] < season) |  # All previous seasons
            ((full_data['season'] == season) & (full_data['matchday'] < matchday))  # Previous matchdays
        )
        train_data = full_data[train_mask]

        # Get prediction data: just this matchday
        pred_mask = (full_data['season'] == season) & (full_data['matchday'] == matchday)
        pred_data = full_data[pred_mask]

        if len(pred_data) == 0:
            continue

        if len(train_data) < 100:
            # Not enough training data
            if verbose:
                print(f"  MD {matchday:2d}: Skipped (insufficient training data)")
            continue

        # Train model on all data before this matchday
        try:
            predictions = model_trainer(train_data, pred_data)

            if predictions is None:
                if verbose:
                    print(f"  MD {matchday:2d}: Failed (training error)")
                continue

            pred_home, pred_away = predictions

            # Get true values
            true_home = pred_data['target_home_goals'].values
            true_away = pred_data['target_away_goals'].values

            # Calculate points for this matchday
            avg_points = calculate_average_kicktipp_points(
                true_home, true_away, pred_home, pred_away
            )

            # Outcome accuracy
            def get_outcome(h, a):
                diff = h - a
                if diff > 0:
                    return 'H'
                elif diff < 0:
                    return 'A'
                else:
                    return 'D'

            true_outcomes = [get_outcome(h, a) for h, a in zip(true_home, true_away)]
            pred_outcomes = [get_outcome(h, a) for h, a in zip(pred_home, pred_away)]
            outcome_acc = np.mean([t == p for t, p in zip(true_outcomes, pred_outcomes)])

            matchday_results.append({
                'matchday': matchday,
                'n_matches': len(pred_data),
                'n_train': len(train_data),
                'kicktipp_points': avg_points,
                'outcome_accuracy': outcome_acc
            })

            cumulative_points.append(avg_points)

            if verbose:
                cumulative_avg = np.mean(cumulative_points)
                print(f"  MD {matchday:2d}: {avg_points:.3f} pts "
                      f"({outcome_acc*100:4.1f}% acc) "
                      f"[{len(pred_data):2d} matches, {len(train_data):5d} train] "
                      f"→ Cumulative: {cumulative_avg:.3f}")

        except Exception as e:
            if verbose:
                print(f"  MD {matchday:2d}: Failed ({str(e)[:30]})")
            continue

    if len(matchday_results) == 0:
        return {}

    # Aggregate results
    df = pd.DataFrame(matchday_results)

    results = {
        'season': season,
        'model': model_name,
        'n_matchdays': len(df),
        'total_matches': df['n_matches'].sum(),
        'avg_kicktipp_points': df['kicktipp_points'].mean(),
        'std_kicktipp_points': df['kicktipp_points'].std(),
        'avg_outcome_accuracy': df['outcome_accuracy'].mean(),
        'matchday_details': df
    }

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"SIMULATION SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Season:              {season}")
        print(f"  Model:               {model_name}")
        print(f"  Matchdays Simulated: {results['n_matchdays']}")
        print(f"  Total Matches:       {results['total_matches']}")
        print(f"  Avg Kicktipp Points: {results['avg_kicktipp_points']:.4f} ± {results['std_kicktipp_points']:.4f}")
        print(f"  Avg Outcome Acc:     {results['avg_outcome_accuracy']*100:.2f}%")

    return results


def create_classifier_trainer(model_class, features, default_scores, use_catboost=True):
    """
    Create a trainer function for classifier models

    Args:
        model_class: ClassifierExperiment class
        features: List of feature names
        default_scores: Score heuristics dict
        use_catboost: Use CatBoost (True) or Random Forest (False)

    Returns:
        Trainer function
    """
    def trainer(train_data: pd.DataFrame, pred_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Train classifier and return score predictions"""
        try:
            # Prepare data - create copies to avoid SettingWithCopyWarning
            X_train = train_data[features].copy()
            y_train = train_data['target_multiclass'].copy()
            X_pred = pred_data[features].copy()

            # Handle missing values
            from . import config
            for feat in features:
                if feat in config.CATEGORICAL_FEATURES:
                    X_train[feat] = X_train[feat].fillna('MISSING')
                    X_pred[feat] = X_pred[feat].fillna('MISSING')
                else:
                    median_val = X_train[feat].median()
                    X_train[feat] = X_train[feat].fillna(median_val)
                    X_pred[feat] = X_pred[feat].fillna(median_val)

            # Create experiment
            exp = model_class(default_scores=default_scores)

            # Train
            if use_catboost:
                # Use a small validation set from end of training data
                val_size = min(100, len(train_data) // 10)
                X_val = X_train.iloc[-val_size:]
                y_val = y_train.iloc[-val_size:]
                X_train_sub = X_train.iloc[:-val_size]
                y_train_sub = y_train.iloc[:-val_size]

                exp.train_catboost(X_train_sub, y_train_sub, X_val, y_val, verbose=False)
                pred_home, pred_away = exp.predict_catboost(X_pred)
            else:
                exp.train_random_forest(X_train, y_train, verbose=False)
                pred_home, pred_away = exp.predict_random_forest(X_pred)

            return pred_home, pred_away

        except Exception as e:
            print(f"Training error: {e}")
            return None

    return trainer


def create_regressor_trainer(model_class, features, use_catboost=True):
    """
    Create a trainer function for regressor models

    Args:
        model_class: RegressorExperiment class
        features: List of feature names
        use_catboost: Use CatBoost (True) or Random Forest (False)

    Returns:
        Trainer function
    """
    def trainer(train_data: pd.DataFrame, pred_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Train regressors and return score predictions"""
        try:
            # Prepare data - create copies to avoid SettingWithCopyWarning
            X_train = train_data[features].copy()
            y_train_home = train_data['target_home_goals'].copy()
            y_train_away = train_data['target_away_goals'].copy()
            X_pred = pred_data[features].copy()

            # Handle missing values
            from . import config
            for feat in features:
                if feat in config.CATEGORICAL_FEATURES:
                    X_train[feat] = X_train[feat].fillna('MISSING')
                    X_pred[feat] = X_pred[feat].fillna('MISSING')
                else:
                    median_val = X_train[feat].median()
                    X_train[feat] = X_train[feat].fillna(median_val)
                    X_pred[feat] = X_pred[feat].fillna(median_val)

            # Create experiment
            exp = model_class()

            # Train
            if use_catboost:
                # Use a small validation set
                val_size = min(100, len(train_data) // 10)
                X_val = X_train.iloc[-val_size:]
                y_val_home = y_train_home.iloc[-val_size:]
                y_val_away = y_train_away.iloc[-val_size:]
                X_train_sub = X_train.iloc[:-val_size]
                y_train_home_sub = y_train_home.iloc[:-val_size]
                y_train_away_sub = y_train_away.iloc[:-val_size]

                exp.train_catboost(
                    X_train_sub, y_train_home_sub, y_train_away_sub,
                    X_val, y_val_home, y_val_away,
                    verbose=False
                )
                pred_home, pred_away = exp.predict_catboost(X_pred)
            else:
                exp.train_random_forest(X_train, y_train_home, y_train_away, verbose=False)
                pred_home, pred_away = exp.predict_random_forest(X_pred)

            # Round to integers
            pred_home = np.round(pred_home).astype(int)
            pred_away = np.round(pred_away).astype(int)
            pred_home = np.maximum(pred_home, 0)
            pred_away = np.maximum(pred_away, 0)

            return pred_home, pred_away

        except Exception as e:
            print(f"Training error: {e}")
            return None

    return trainer
