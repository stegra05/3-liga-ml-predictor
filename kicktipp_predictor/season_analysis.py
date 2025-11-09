"""
Per-Season Analysis Module
Evaluates models on individual seasons to detect temporal degradation
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from .evaluation import calculate_average_kicktipp_points


def evaluate_per_season(
    test_df: pd.DataFrame,
    y_pred_home: np.ndarray,
    y_pred_away: np.ndarray,
    model_name: str
) -> pd.DataFrame:
    """
    Evaluate predictions broken down by season

    Args:
        test_df: Test dataframe with 'season' column
        y_pred_home: Predicted home goals
        y_pred_away: Predicted away goals
        model_name: Name of model

    Returns:
        DataFrame with per-season metrics
    """
    if 'season' not in test_df.columns:
        print("Warning: 'season' column not found in test data")
        return pd.DataFrame()

    # Get true values
    y_true_home = test_df['target_home_goals'].values
    y_true_away = test_df['target_away_goals'].values

    seasons = test_df['season'].unique()
    results = []

    for season in sorted(seasons):
        mask = test_df['season'] == season
        n_matches = mask.sum()

        if n_matches == 0:
            continue

        # Filter predictions for this season
        y_true_h_season = y_true_home[mask]
        y_true_a_season = y_true_away[mask]
        y_pred_h_season = y_pred_home[mask]
        y_pred_a_season = y_pred_away[mask]

        # Calculate metrics
        avg_points = calculate_average_kicktipp_points(
            y_true_h_season, y_true_a_season,
            y_pred_h_season, y_pred_a_season
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

        true_outcomes = [get_outcome(h, a) for h, a in zip(y_true_h_season, y_true_a_season)]
        pred_outcomes = [get_outcome(h, a) for h, a in zip(y_pred_h_season, y_pred_a_season)]
        outcome_acc = np.mean([t == p for t, p in zip(true_outcomes, pred_outcomes)])

        # Exact score accuracy
        exact_acc = np.mean((y_true_h_season == y_pred_h_season) & (y_true_a_season == y_pred_a_season))

        results.append({
            'season': season,
            'n_matches': n_matches,
            'kicktipp_points': avg_points,
            'outcome_accuracy': outcome_acc,
            'exact_accuracy': exact_acc
        })

    df = pd.DataFrame(results)
    return df


def print_season_analysis(season_df: pd.DataFrame, model_name: str):
    """
    Pretty print per-season analysis

    Args:
        season_df: DataFrame from evaluate_per_season
        model_name: Name of model
    """
    if len(season_df) == 0:
        return

    print(f"\n{'=' * 70}")
    print(f"PER-SEASON ANALYSIS: {model_name}")
    print(f"{'=' * 70}")

    print(f"\n{'Season':<15} {'Matches':>8} {'Kicktipp':>10} {'Outcome':>10} {'Exact':>10}")
    print("-" * 70)

    for _, row in season_df.iterrows():
        print(f"{row['season']:<15} {row['n_matches']:>8} "
              f"{row['kicktipp_points']:>10.4f} "
              f"{row['outcome_accuracy']*100:>9.1f}% "
              f"{row['exact_accuracy']*100:>9.1f}%")

    # Summary statistics
    print("-" * 70)
    print(f"{'AVERAGE':<15} {season_df['n_matches'].sum():>8} "
          f"{season_df['kicktipp_points'].mean():>10.4f} "
          f"{season_df['outcome_accuracy'].mean()*100:>9.1f}% "
          f"{season_df['exact_accuracy'].mean()*100:>9.1f}%")
    print(f"{'STD DEV':<15} {'':<8} "
          f"{season_df['kicktipp_points'].std():>10.4f} "
          f"{season_df['outcome_accuracy'].std()*100:>9.1f}% "
          f"{season_df['exact_accuracy'].std()*100:>9.1f}%")


def evaluate_baseline(
    y_true_home: np.ndarray,
    y_true_away: np.ndarray,
    pred_home: int = 2,
    pred_away: int = 1
) -> Dict:
    """
    Evaluate a naive baseline that always predicts the same score

    Args:
        y_true_home: True home goals
        y_true_away: True away goals
        pred_home: Predicted home goals (default 2)
        pred_away: Predicted away goals (default 1)

    Returns:
        Dictionary with baseline metrics
    """
    n_samples = len(y_true_home)

    # Create constant predictions
    y_pred_home = np.full(n_samples, pred_home)
    y_pred_away = np.full(n_samples, pred_away)

    # Calculate Kicktipp points
    avg_points = calculate_average_kicktipp_points(
        y_true_home, y_true_away, y_pred_home, y_pred_away
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

    true_outcomes = [get_outcome(h, a) for h, a in zip(y_true_home, y_true_away)]
    pred_outcomes = [get_outcome(h, a) for h, a in zip(y_pred_home, y_pred_away)]
    outcome_acc = np.mean([t == p for t, p in zip(true_outcomes, pred_outcomes)])

    # Exact score accuracy
    exact_acc = np.mean((y_true_home == y_pred_home) & (y_true_away == y_pred_away))

    return {
        'model': f'Baseline ({pred_home}-{pred_away} always)',
        'avg_kicktipp_points': avg_points,
        'outcome_accuracy': outcome_acc,
        'exact_accuracy': exact_acc,
        'n_samples': n_samples
    }


def print_baseline_comparison(baseline_results: Dict):
    """
    Print baseline results

    Args:
        baseline_results: Dictionary from evaluate_baseline
    """
    print(f"\n{'=' * 70}")
    print(f"BASELINE COMPARISON")
    print(f"{'=' * 70}")

    print(f"\nModel: {baseline_results['model']}")
    print(f"  Samples:          {baseline_results['n_samples']}")
    print(f"  Kicktipp Points:  {baseline_results['avg_kicktipp_points']:.4f}")
    print(f"  Outcome Accuracy: {baseline_results['outcome_accuracy']*100:.2f}%")
    print(f"  Exact Accuracy:   {baseline_results['exact_accuracy']*100:.2f}%")


def check_data_leakage(train_df: pd.DataFrame) -> Dict:
    """
    Check for potential data leakage issues

    Args:
        train_df: Training dataframe

    Returns:
        Dictionary with leakage warnings
    """
    warnings = []

    # Check 1: Do odds probabilities perfectly predict outcomes?
    if 'implied_prob_home' in train_df.columns and 'target_multiclass' in train_df.columns:
        # For each row, check if max prob matches outcome
        mask = train_df['implied_prob_home'].notna()
        subset = train_df[mask]

        if len(subset) > 0:
            # Get max prob class
            probs = subset[['implied_prob_away', 'implied_prob_draw', 'implied_prob_home']].values
            max_class = probs.argmax(axis=1)  # 0=Away, 1=Draw, 2=Home
            true_class = subset['target_multiclass'].values

            # Check accuracy
            odds_accuracy = (max_class == true_class).mean()

            if odds_accuracy > 0.8:
                warnings.append({
                    'feature': 'implied_prob_*',
                    'issue': f'Odds perfectly predict {odds_accuracy*100:.1f}% of outcomes - possible leakage or overfitting',
                    'severity': 'HIGH'
                })

    # Check 2: Are there any post-match statistics?
    post_match_features = [
        'home_possession', 'home_shots', 'home_passes',
        'away_possession', 'away_shots', 'away_passes',
        'home_fouls', 'away_fouls', 'home_corners', 'away_corners'
    ]

    found_post_match = [f for f in post_match_features if f in train_df.columns]
    if found_post_match:
        warnings.append({
            'feature': ', '.join(found_post_match[:3]),
            'issue': f'{len(found_post_match)} post-match features found in dataset',
            'severity': 'CRITICAL'
        })

    # Check 3: Are result columns in features?
    result_cols = ['result', 'home_goals', 'away_goals']
    found_results = [c for c in result_cols if c in train_df.columns]
    if found_results:
        warnings.append({
            'feature': ', '.join(found_results),
            'issue': 'Result columns found - ensure these are NOT used as features',
            'severity': 'CRITICAL'
        })

    return {'warnings': warnings, 'clean': len(warnings) == 0}


def print_leakage_check(leakage_report: Dict):
    """
    Print data leakage check results

    Args:
        leakage_report: Dictionary from check_data_leakage
    """
    print(f"\n{'=' * 70}")
    print(f"DATA LEAKAGE CHECK")
    print(f"{'=' * 70}")

    if leakage_report['clean']:
        print("\n✓ No obvious data leakage detected")
    else:
        print(f"\n⚠ Found {len(leakage_report['warnings'])} potential issues:\n")
        for i, warning in enumerate(leakage_report['warnings'], 1):
            severity_icon = '🔴' if warning['severity'] == 'CRITICAL' else '🟡'
            print(f"{i}. {severity_icon} [{warning['severity']}] {warning['feature']}")
            print(f"   → {warning['issue']}\n")
