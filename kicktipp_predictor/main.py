"""
Main Execution Script for Kicktipp Prediction System

Runs all three experiments and compares results
"""

import pandas as pd
import numpy as np
from typing import List
import sys

from . import config
from .data_loader import load_datasets, prepare_features, prepare_regression_features, combine_train_val
from .evaluation import print_results, compare_models
from .models.classifiers import ClassifierExperiment
from .models.regressors import RegressorExperiment
from .models.ensemble import EnsembleExperiment


def print_header(title: str):
    """Print section header"""
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)


def run_experiment_1(
    X_train, y_train, X_val, y_val, X_test, y_test,
    y_test_home, y_test_away
) -> List[dict]:
    """
    Experiment 1: Classifiers

    Returns:
        List of results dictionaries
    """
    print_header("EXPERIMENT 1: CLASSIFIERS")

    exp = ClassifierExperiment()

    # Train CatBoost
    exp.train_catboost(X_train, y_train, X_val, y_val, verbose=False)
    results_cb = exp.evaluate_catboost(X_test, y_test, y_test_home, y_test_away)
    print_results(results_cb)

    # Train Random Forest
    exp.train_random_forest(X_train, y_train, verbose=False)
    results_rf = exp.evaluate_random_forest(X_test, y_test, y_test_home, y_test_away)
    print_results(results_rf)

    return [results_cb, results_rf]


def run_experiment_2(
    X_train, y_train_home, y_train_away,
    X_val, y_val_home, y_val_away,
    X_test, y_test_home, y_test_away
) -> List[dict]:
    """
    Experiment 2: Regressors

    Returns:
        List of results dictionaries
    """
    print_header("EXPERIMENT 2: REGRESSORS")

    exp = RegressorExperiment()

    # Train CatBoost
    exp.train_catboost(
        X_train, y_train_home, y_train_away,
        X_val, y_val_home, y_val_away,
        verbose=False
    )
    results_cb = exp.evaluate_catboost(X_test, y_test_home, y_test_away)
    print_results(results_cb)

    # Train Random Forest
    exp.train_random_forest(X_train, y_train_home, y_train_away, verbose=False)
    results_rf = exp.evaluate_random_forest(X_test, y_test_home, y_test_away)
    print_results(results_rf)

    return [results_cb, results_rf]


def run_experiment_3(
    X_train, y_train, X_val, y_val, X_test, y_test,
    y_test_home, y_test_away
) -> List[dict]:
    """
    Experiment 3: Stacked Ensemble

    Returns:
        List with single results dictionary
    """
    print_header("EXPERIMENT 3: STACKED ENSEMBLE")

    exp = EnsembleExperiment()

    # Train ensemble
    exp.train(X_train, y_train, X_val, y_val, verbose=False)

    # Evaluate
    results = exp.evaluate(X_test, y_test, y_test_home, y_test_away)
    print_results(results)

    return [results]


def main():
    """
    Main execution function

    Steps:
    1. Load data
    2. Prepare features
    3. Run all three experiments
    4. Compare results
    5. Identify best model
    """
    print_header("KICKTIPP PREDICTION SYSTEM")
    print("\nThis system implements three approaches to predict football match scores:")
    print("  1. Classifiers (predict outcome → map to score)")
    print("  2. Regressors (predict exact goals)")
    print("  3. Stacked Ensemble (combine multiple models)")
    print("\nEvaluation metric: Average Kicktipp points per match")

    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print_header("STEP 1: LOADING DATA")

    train, val, test = load_datasets()

    # ========================================================================
    # STEP 2: Prepare Features
    # ========================================================================
    print_header("STEP 2: PREPARING FEATURES")

    # For classification experiments (1 and 3)
    X_train_cls, y_train_cls, X_val_cls, y_val_cls, X_test_cls, y_test_cls, features_cls = prepare_features(
        train, val, test, use_categorical=True
    )

    # For regression experiment (2)
    (X_train_reg, y_train_home, y_train_away,
     X_val_reg, y_val_home, y_val_away,
     X_test_reg, y_test_home, y_test_away, features_reg) = prepare_regression_features(
        train, val, test, use_categorical=True
    )

    # Get true goals for test set (needed for Kicktipp scoring in Exp 1 & 3)
    y_test_home_cls = test[config.TARGET_HOME_GOALS]
    y_test_away_cls = test[config.TARGET_AWAY_GOALS]

    print(f"\nDatasets prepared successfully.")
    print(f"  Classification features: {len(features_cls)}")
    print(f"  Regression features: {len(features_reg)}")

    # ========================================================================
    # STEP 3: Run Experiments
    # ========================================================================

    all_results = []

    # Experiment 1: Classifiers
    try:
        results_1 = run_experiment_1(
            X_train_cls, y_train_cls, X_val_cls, y_val_cls,
            X_test_cls, y_test_cls, y_test_home_cls, y_test_away_cls
        )
        all_results.extend(results_1)
    except Exception as e:
        print(f"\n[ERROR] Experiment 1 failed: {e}")
        import traceback
        traceback.print_exc()

    # Experiment 2: Regressors
    try:
        results_2 = run_experiment_2(
            X_train_reg, y_train_home, y_train_away,
            X_val_reg, y_val_home, y_val_away,
            X_test_reg, y_test_home, y_test_away
        )
        all_results.extend(results_2)
    except Exception as e:
        print(f"\n[ERROR] Experiment 2 failed: {e}")
        import traceback
        traceback.print_exc()

    # Experiment 3: Stacked Ensemble
    try:
        results_3 = run_experiment_3(
            X_train_cls, y_train_cls, X_val_cls, y_val_cls,
            X_test_cls, y_test_cls, y_test_home_cls, y_test_away_cls
        )
        all_results.extend(results_3)
    except Exception as e:
        print(f"\n[ERROR] Experiment 3 failed: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # STEP 4: Compare Results
    # ========================================================================
    print_header("FINAL RESULTS COMPARISON")

    if len(all_results) == 0:
        print("\n[ERROR] No experiments completed successfully!")
        sys.exit(1)

    leaderboard = compare_models(all_results)
    print("\nLEADERBOARD (sorted by Kicktipp points):")
    print(leaderboard.to_string(index=True))

    # ========================================================================
    # STEP 5: Winner
    # ========================================================================
    winner = leaderboard.iloc[0]

    print_header("WINNER")
    print(f"\nBest Model: {winner['model']}")
    print(f"Kicktipp Score: {winner['avg_kicktipp_points']:.4f} points/match")

    if 'accuracy' in winner:
        print(f"Accuracy: {winner['accuracy']:.4f}")
    if 'rmse_avg' in winner:
        print(f"RMSE: {winner['rmse_avg']:.4f}")

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Retrain the winning model on train+val data")
    print("2. Save the model for production use")
    print("3. Optional: Tune hyperparameters for further improvement")
    print("4. Optional: Analyze error patterns and feature importance")

    return leaderboard


if __name__ == "__main__":
    leaderboard = main()
