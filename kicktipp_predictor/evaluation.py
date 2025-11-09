"""
Evaluation Module
Implements Kicktipp scoring system and other metrics
"""

import numpy as np
from sklearn.metrics import log_loss, mean_squared_error, accuracy_score
from typing import List, Tuple
import pandas as pd


def calculate_kicktipp_points(y_true_score: List[int], y_pred_score: List[int]) -> int:
    """
    Calculate Kicktipp points for a single prediction

    Rules:
    - Exact score match: 4 points
    - Correct goal difference: 3 points
    - Correct outcome (H/D/A): 2 points
    - Otherwise: 0 points

    Args:
        y_true_score: Actual score [home_goals, away_goals]
        y_pred_score: Predicted score [home_goals, away_goals]

    Returns:
        Points (0, 2, 3, or 4)
    """
    true_home, true_away = y_true_score
    pred_home, pred_away = y_pred_score

    # Exact match
    if true_home == pred_home and true_away == pred_away:
        return 4

    true_diff = true_home - true_away
    pred_diff = pred_home - pred_away

    # Correct goal difference
    if true_diff == pred_diff:
        return 3

    # Correct outcome (sign of difference)
    true_sign = np.sign(true_diff)
    pred_sign = np.sign(pred_diff)

    if true_sign == pred_sign:
        return 2

    return 0


def calculate_average_kicktipp_points(
    y_true_home: np.ndarray,
    y_true_away: np.ndarray,
    y_pred_home: np.ndarray,
    y_pred_away: np.ndarray
) -> float:
    """
    Calculate average Kicktipp points across all predictions

    Args:
        y_true_home: True home goals
        y_true_away: True away goals
        y_pred_home: Predicted home goals
        y_pred_away: Predicted away goals

    Returns:
        Average points per match
    """
    total_points = 0
    n_matches = len(y_true_home)

    for i in range(n_matches):
        true_score = [int(y_true_home[i]), int(y_true_away[i])]
        pred_score = [int(y_pred_home[i]), int(y_pred_away[i])]
        total_points += calculate_kicktipp_points(true_score, pred_score)

    return total_points / n_matches


def scores_from_class_predictions(
    y_pred_class: np.ndarray,
    default_scores: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert class predictions (0/1/2) to score predictions using heuristics

    Args:
        y_pred_class: Predicted classes (0=Away, 1=Draw, 2=Home)
        default_scores: Dict mapping class to [home_goals, away_goals]

    Returns:
        pred_home, pred_away arrays
    """
    pred_home = []
    pred_away = []

    for cls in y_pred_class:
        score = default_scores[int(cls)]
        pred_home.append(score[0])
        pred_away.append(score[1])

    return np.array(pred_home), np.array(pred_away)


def evaluate_classifier(
    y_true_class: np.ndarray,
    y_pred_class: np.ndarray,
    y_pred_proba: np.ndarray,
    y_true_home: np.ndarray,
    y_true_away: np.ndarray,
    default_scores: dict,
    model_name: str
) -> dict:
    """
    Evaluate a classifier model

    Args:
        y_true_class: True classes
        y_pred_class: Predicted classes
        y_pred_proba: Predicted probabilities
        y_true_home: True home goals
        y_true_away: True away goals
        default_scores: Score heuristics
        model_name: Name of the model

    Returns:
        Dictionary of metrics
    """
    # Classification metrics
    accuracy = accuracy_score(y_true_class, y_pred_class)
    logloss = log_loss(y_true_class, y_pred_proba)

    # Rank Probability Score (RPS) for ordered outcomes
    rps = calculate_rps(y_true_class, y_pred_proba)

    # Convert predictions to scores
    pred_home, pred_away = scores_from_class_predictions(y_pred_class, default_scores)

    # Kicktipp points
    avg_points = calculate_average_kicktipp_points(
        y_true_home, y_true_away, pred_home, pred_away
    )

    results = {
        'model': model_name,
        'accuracy': accuracy,
        'log_loss': logloss,
        'rps': rps,
        'avg_kicktipp_points': avg_points
    }

    return results


def evaluate_regressor(
    y_true_home: np.ndarray,
    y_true_away: np.ndarray,
    y_pred_home_raw: np.ndarray,
    y_pred_away_raw: np.ndarray,
    model_name: str
) -> dict:
    """
    Evaluate a regressor model

    Args:
        y_true_home: True home goals
        y_true_away: True away goals
        y_pred_home_raw: Predicted home goals (continuous)
        y_pred_away_raw: Predicted away goals (continuous)
        model_name: Name of the model

    Returns:
        Dictionary of metrics
    """
    # Round predictions to integers
    y_pred_home = np.round(y_pred_home_raw).astype(int)
    y_pred_away = np.round(y_pred_away_raw).astype(int)

    # Ensure non-negative
    y_pred_home = np.maximum(y_pred_home, 0)
    y_pred_away = np.maximum(y_pred_away, 0)

    # RMSE on raw predictions
    rmse_home = np.sqrt(mean_squared_error(y_true_home, y_pred_home_raw))
    rmse_away = np.sqrt(mean_squared_error(y_true_away, y_pred_away_raw))
    rmse_avg = (rmse_home + rmse_away) / 2

    # Kicktipp points
    avg_points = calculate_average_kicktipp_points(
        y_true_home, y_true_away, y_pred_home, y_pred_away
    )

    results = {
        'model': model_name,
        'rmse_home': rmse_home,
        'rmse_away': rmse_away,
        'rmse_avg': rmse_avg,
        'avg_kicktipp_points': avg_points
    }

    return results


def calculate_rps(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Calculate Rank Probability Score for ordered outcomes

    RPS is particularly appropriate for football because outcomes have an
    inherent order (Away < Draw < Home)

    Args:
        y_true: True classes (0, 1, 2)
        y_pred_proba: Predicted probabilities (N x 3)

    Returns:
        Average RPS
    """
    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1]

    rps_sum = 0

    for i in range(n_samples):
        # Create one-hot encoding of true class
        true_cumsum = np.zeros(n_classes)
        true_cumsum[int(y_true[i]):] = 1

        # Cumulative probabilities
        pred_cumsum = np.cumsum(y_pred_proba[i])

        # RPS is sum of squared differences
        rps_sum += np.sum((pred_cumsum - true_cumsum) ** 2)

    return rps_sum / n_samples


def print_results(results: dict):
    """
    Pretty print evaluation results

    Args:
        results: Dictionary from evaluate_classifier or evaluate_regressor
    """
    print(f"\nResults for {results['model']}:")
    print("=" * 60)

    for key, value in results.items():
        if key == 'model':
            continue
        if isinstance(value, float):
            print(f"  {key:25s}: {value:.4f}")
        else:
            print(f"  {key:25s}: {value}")

    # Highlight Kicktipp score
    if 'avg_kicktipp_points' in results:
        print("\n" + "=" * 60)
        print(f"  KICKTIPP SCORE: {results['avg_kicktipp_points']:.4f} points/match")
        print("=" * 60)


def compare_models(results_list: List[dict]) -> pd.DataFrame:
    """
    Compare multiple models and return sorted leaderboard

    Args:
        results_list: List of result dictionaries

    Returns:
        DataFrame sorted by Kicktipp points
    """
    df = pd.DataFrame(results_list)
    df = df.sort_values('avg_kicktipp_points', ascending=False)
    df = df.reset_index(drop=True)
    df.index = df.index + 1  # Start ranking from 1

    return df
