"""
Detailed Analysis Module
Provides debugging and analysis functions for predictions
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import Tuple, Dict


def analyze_predictions(
    y_true_home: np.ndarray,
    y_true_away: np.ndarray,
    y_pred_home: np.ndarray,
    y_pred_away: np.ndarray,
    model_name: str = "Model"
) -> Dict:
    """
    Detailed analysis of predictions including:
    - Points breakdown (0, 2, 3, 4 point predictions)
    - Most common predicted vs actual scorelines
    - Score distribution comparison

    Args:
        y_true_home: True home goals
        y_true_away: True away goals
        y_pred_home: Predicted home goals
        y_pred_away: Predicted away goals
        model_name: Name for display

    Returns:
        Dictionary with analysis results
    """
    from .evaluation import calculate_kicktipp_points

    n_samples = len(y_true_home)

    # Calculate points for each prediction
    points = []
    for i in range(n_samples):
        true_score = [int(y_true_home[i]), int(y_true_away[i])]
        pred_score = [int(y_pred_home[i]), int(y_pred_away[i])]
        pts = calculate_kicktipp_points(true_score, pred_score)
        points.append(pts)

    points = np.array(points)

    # Points breakdown
    points_breakdown = {
        0: np.sum(points == 0),
        2: np.sum(points == 2),
        3: np.sum(points == 3),
        4: np.sum(points == 4)
    }

    points_pct = {k: v / n_samples * 100 for k, v in points_breakdown.items()}

    # Predicted scorelines distribution
    pred_scores = [f"{int(h)}-{int(a)}" for h, a in zip(y_pred_home, y_pred_away)]
    pred_counter = Counter(pred_scores)

    # True scorelines distribution
    true_scores = [f"{int(h)}-{int(a)}" for h, a in zip(y_true_home, y_true_away)]
    true_counter = Counter(true_scores)

    # Outcome accuracy
    def get_outcome(home, away):
        diff = home - away
        if diff > 0:
            return 'H'
        elif diff < 0:
            return 'A'
        else:
            return 'D'

    true_outcomes = [get_outcome(h, a) for h, a in zip(y_true_home, y_true_away)]
    pred_outcomes = [get_outcome(h, a) for h, a in zip(y_pred_home, y_pred_away)]

    outcome_accuracy = np.mean([t == p for t, p in zip(true_outcomes, pred_outcomes)])

    # Goal difference accuracy
    true_diff = y_true_home - y_true_away
    pred_diff = y_pred_home - y_pred_away
    diff_accuracy = np.mean(true_diff == pred_diff)

    # Exact score accuracy
    exact_accuracy = np.mean((y_true_home == y_pred_home) & (y_true_away == y_pred_away))

    return {
        'model_name': model_name,
        'n_samples': n_samples,
        'avg_points': np.mean(points),
        'points_breakdown': points_breakdown,
        'points_pct': points_pct,
        'outcome_accuracy': outcome_accuracy,
        'diff_accuracy': diff_accuracy,
        'exact_accuracy': exact_accuracy,
        'pred_scorelines': pred_counter.most_common(10),
        'true_scorelines': true_counter.most_common(10),
        'pred_avg_home': np.mean(y_pred_home),
        'pred_avg_away': np.mean(y_pred_away),
        'true_avg_home': np.mean(y_true_home),
        'true_avg_away': np.mean(y_true_away)
    }


def print_detailed_analysis(analysis: Dict):
    """
    Pretty print detailed analysis

    Args:
        analysis: Dictionary from analyze_predictions
    """
    print(f"\n{'=' * 70}")
    print(f"DETAILED ANALYSIS: {analysis['model_name']}")
    print(f"{'=' * 70}")

    print(f"\n📊 OVERALL METRICS:")
    print(f"  Samples:              {analysis['n_samples']}")
    print(f"  Avg Kicktipp Points:  {analysis['avg_points']:.4f}")
    print(f"  Outcome Accuracy:     {analysis['outcome_accuracy']*100:.2f}%")
    print(f"  Goal Diff Accuracy:   {analysis['diff_accuracy']*100:.2f}%")
    print(f"  Exact Score Accuracy: {analysis['exact_accuracy']*100:.2f}%")

    print(f"\n🎯 POINTS BREAKDOWN:")
    for points, count in sorted(analysis['points_breakdown'].items()):
        pct = analysis['points_pct'][points]
        print(f"  {points} points: {count:4d} predictions ({pct:5.2f}%)")

    print(f"\n⚽ AVERAGE GOALS:")
    print(f"  Predicted: Home {analysis['pred_avg_home']:.2f}, Away {analysis['pred_avg_away']:.2f}")
    print(f"  Actual:    Home {analysis['true_avg_home']:.2f}, Away {analysis['true_avg_away']:.2f}")

    print(f"\n🔮 TOP 10 PREDICTED SCORELINES:")
    for score, count in analysis['pred_scorelines']:
        pct = count / analysis['n_samples'] * 100
        print(f"  {score}: {count:4d} times ({pct:5.2f}%)")

    print(f"\n📈 TOP 10 ACTUAL SCORELINES:")
    for score, count in analysis['true_scorelines']:
        pct = count / analysis['n_samples'] * 100
        print(f"  {score}: {count:4d} times ({pct:5.2f}%)")


def compute_optimal_default_scores(
    train: pd.DataFrame,
    target_home_col: str = 'target_home_goals',
    target_away_col: str = 'target_away_goals',
    target_class_col: str = 'target_multiclass'
) -> Dict[int, Tuple[int, int]]:
    """
    Compute optimal default scores for each outcome class
    based on training data distribution

    Strategy: For each outcome, find the most common scoreline

    Args:
        train: Training dataframe
        target_home_col: Column name for home goals
        target_away_col: Column name for away goals
        target_class_col: Column name for class (0=Away, 1=Draw, 2=Home)

    Returns:
        Dictionary mapping class to [home_goals, away_goals]
    """
    default_scores = {}

    for outcome_class in [0, 1, 2]:
        # Filter matches with this outcome
        mask = train[target_class_col] == outcome_class
        matches = train[mask]

        if len(matches) == 0:
            # Fallback
            if outcome_class == 0:  # Away win
                default_scores[outcome_class] = [0, 1]
            elif outcome_class == 1:  # Draw
                default_scores[outcome_class] = [1, 1]
            else:  # Home win
                default_scores[outcome_class] = [1, 0]
            continue

        # Find most common scoreline for this outcome
        scorelines = matches[[target_home_col, target_away_col]].values
        scoreline_strs = [f"{int(h)}-{int(a)}" for h, a in scorelines]
        counter = Counter(scoreline_strs)
        most_common = counter.most_common(1)[0][0]

        # Parse back to [home, away]
        home, away = map(int, most_common.split('-'))
        default_scores[outcome_class] = [home, away]

    return default_scores


def print_default_scores_analysis(train: pd.DataFrame, default_scores: Dict):
    """
    Print analysis of default score choices

    Args:
        train: Training dataframe
        default_scores: Dictionary of default scores
    """
    print(f"\n{'=' * 70}")
    print(f"DEFAULT SCORE HEURISTICS")
    print(f"{'=' * 70}")

    outcome_names = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}

    for outcome_class in [2, 1, 0]:  # Home, Draw, Away
        mask = train['target_multiclass'] == outcome_class
        n_matches = mask.sum()

        # Get scoreline distribution for this outcome
        matches = train[mask]
        scorelines = matches[['target_home_goals', 'target_away_goals']].values
        scoreline_strs = [f"{int(h)}-{int(a)}" for h, a in scorelines]
        counter = Counter(scoreline_strs)

        chosen_score = default_scores[outcome_class]
        chosen_str = f"{chosen_score[0]}-{chosen_score[1]}"
        chosen_count = counter.get(chosen_str, 0)
        chosen_pct = chosen_count / n_matches * 100 if n_matches > 0 else 0

        print(f"\n{outcome_names[outcome_class]} (Class {outcome_class}):")
        print(f"  Total matches: {n_matches}")
        print(f"  Chosen default: {chosen_str} (appears {chosen_count} times, {chosen_pct:.1f}%)")
        print(f"  Top 5 scorelines:")
        for score, count in counter.most_common(5):
            pct = count / n_matches * 100
            marker = " ← CHOSEN" if score == chosen_str else ""
            print(f"    {score}: {count:4d} ({pct:5.1f}%){marker}")
