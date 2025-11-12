"""
Evaluation metrics for the 3. Liga Match Predictor.

This module contains all metric calculations for model evaluation including:
- Classification metrics (accuracy, log loss, Brier score)
- Ranked Probability Score (RPS)
- P&L simulation for betting strategies
- Per-class metrics (precision, recall, F1)
- Baseline comparisons
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve


def calc_accuracy(y_true: np.ndarray, y_pred_classes: np.ndarray) -> float:
    """Calculate classification accuracy."""
    return accuracy_score(y_true, y_pred_classes)


def calc_logloss(y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
    """
    Calculate log loss (cross-entropy loss).

    Args:
        y_true: True class labels (0, 1, 2 for Away, Draw, Home)
        y_pred_probs: Predicted probabilities, shape (n_samples, 3)

    Returns:
        Log loss value (lower is better)
    """
    # Explicitly provide labels to handle cases where not all classes are present
    return log_loss(y_true, y_pred_probs, labels=[0, 1, 2])


def calc_brier_score(y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
    """
    Calculate multi-class Brier score.

    Brier score measures the mean squared difference between predicted
    probabilities and actual outcomes.

    Args:
        y_true: True class labels (0, 1, 2)
        y_pred_probs: Predicted probabilities, shape (n_samples, 3)

    Returns:
        Brier score (lower is better, ranges 0-2 for 3 classes)
    """
    n_samples, n_classes = y_pred_probs.shape

    # Convert y_true to one-hot encoding
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1

    # Calculate Brier score: mean squared error between probabilities
    brier = np.mean(np.sum((y_pred_probs - y_true_onehot) ** 2, axis=1))

    return brier


def calc_rps(y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
    """
    Calculate Ranked Probability Score (RPS) for ordered outcomes.

    RPS is particularly suitable for football predictions where outcomes have
    a natural order: Away Win < Draw < Home Win

    The RPS is the sum of squared differences between cumulative probabilities:
    RPS = sum((cumsum(pred) - cumsum(actual))^2)

    For ordered outcomes (Away=0, Draw=1, Home=2), this penalizes predictions
    that are "far" from the actual outcome more than Brier score.

    Args:
        y_true: True class labels (0=Away, 1=Draw, 2=Home)
        y_pred_probs: Predicted probabilities, shape (n_samples, 3)

    Returns:
        Mean RPS across all samples (lower is better, ranges 0-2)
    """
    n_samples, n_classes = y_pred_probs.shape

    # Convert y_true to one-hot encoding
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1

    # Calculate cumulative probabilities
    pred_cumsum = np.cumsum(y_pred_probs, axis=1)
    true_cumsum = np.cumsum(y_true_onehot, axis=1)

    # RPS is the sum of squared differences of cumulative probabilities
    rps = np.mean(np.sum((pred_cumsum - true_cumsum) ** 2, axis=1))

    return rps


def calc_pnl(
    df_fold: pd.DataFrame,
    y_pred_probs: np.ndarray,
    confidence_threshold: float = 0.05,
    stake: float = 1.0,
) -> Dict[str, float]:
    """
    Calculate P&L from a simple betting strategy.

    Strategy: Bet on an outcome if model confidence exceeds market implied
    probability by at least `confidence_threshold`.

    Args:
        df_fold: DataFrame with columns: odds_home, odds_draw, odds_away,
                 implied_prob_home, implied_prob_draw, implied_prob_away,
                 target_multiclass (actual outcome: 0=Away, 1=Draw, 2=Home)
        y_pred_probs: Model predicted probabilities, shape (n_samples, 3)
                      [P(Away), P(Draw), P(Home)]
        confidence_threshold: Minimum edge over market odds to bet (default 5%)
        stake: Bet size per bet (default 1 unit)

    Returns:
        Dictionary with:
        - total_pnl: Total profit/loss in units
        - num_bets: Number of bets placed
        - win_rate: Percentage of winning bets
        - roi: Return on investment (%)
        - avg_odds: Average odds of bets placed
    """
    if "odds_home" not in df_fold.columns or "target_multiclass" not in df_fold.columns:
        return {
            "total_pnl": 0.0,
            "num_bets": 0,
            "win_rate": 0.0,
            "roi": 0.0,
            "avg_odds": 0.0,
        }

    # Ensure we have the same number of rows
    if len(df_fold) != len(y_pred_probs):
        raise ValueError(f"Mismatch: df_fold has {len(df_fold)} rows, y_pred_probs has {len(y_pred_probs)}")

    total_pnl = 0.0
    num_bets = 0
    num_wins = 0
    total_odds = 0.0

    # Extract odds and implied probabilities
    odds = df_fold[["odds_away", "odds_draw", "odds_home"]].values
    implied_probs = df_fold[["implied_prob_away", "implied_prob_draw", "implied_prob_home"]].values
    y_true = df_fold["target_multiclass"].values

    for i in range(len(df_fold)):
        # Find outcomes where model confidence exceeds market by threshold
        edge = y_pred_probs[i] - implied_probs[i]

        # Find best edge
        best_outcome_idx = np.argmax(edge)
        best_edge = edge[best_outcome_idx]

        # Only bet if edge exceeds threshold and odds are valid
        if best_edge >= confidence_threshold and odds[i, best_outcome_idx] > 1.0:
            num_bets += 1
            bet_odds = odds[i, best_outcome_idx]
            total_odds += bet_odds

            # Check if bet won
            if y_true[i] == best_outcome_idx:
                num_wins += 1
                total_pnl += (bet_odds - 1) * stake  # Profit
            else:
                total_pnl -= stake  # Loss

    # Calculate metrics
    win_rate = (num_wins / num_bets * 100) if num_bets > 0 else 0.0
    total_staked = num_bets * stake
    roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0.0
    avg_odds = (total_odds / num_bets) if num_bets > 0 else 0.0

    return {
        "total_pnl": total_pnl,
        "num_bets": num_bets,
        "win_rate": win_rate,
        "roi": roi,
        "avg_odds": avg_odds,
    }


def get_per_class_metrics(y_true: np.ndarray, y_pred_classes: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate per-class precision, recall, and F1 score.

    Args:
        y_true: True class labels (0=Away, 1=Draw, 2=Home)
        y_pred_classes: Predicted class labels

    Returns:
        Dictionary with metrics for each class:
        {
            'away': {'precision': ..., 'recall': ..., 'f1': ...},
            'draw': {'precision': ..., 'recall': ..., 'f1': ...},
            'home': {'precision': ..., 'recall': ..., 'f1': ...}
        }
    """
    # Explicitly provide labels to handle cases where not all classes are present
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_classes, average=None, zero_division=0, labels=[0, 1, 2]
    )

    class_names = ["away", "draw", "home"]

    per_class = {}
    for i, class_name in enumerate(class_names):
        per_class[class_name] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
        }

    return per_class


def run_all_baselines(df_fold: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for baseline strategies.

    Baselines:
    1. Always predict Home Win
    2. Always predict Draw
    3. Always predict Away Win
    4. Always predict the favorite (based on odds)

    Args:
        df_fold: DataFrame with target_multiclass and odds columns

    Returns:
        Dictionary of baseline results with accuracy for each strategy
    """
    if "target_multiclass" not in df_fold.columns:
        return {}

    y_true = df_fold["target_multiclass"].values
    n_samples = len(y_true)

    baselines = {}

    # Baseline 1: Always Home Win (class 2)
    y_pred_home = np.full(n_samples, 2)
    baselines["always_home"] = {
        "accuracy": accuracy_score(y_true, y_pred_home),
        "description": "Always predict Home Win",
    }

    # Baseline 2: Always Draw (class 1)
    y_pred_draw = np.full(n_samples, 1)
    baselines["always_draw"] = {
        "accuracy": accuracy_score(y_true, y_pred_draw),
        "description": "Always predict Draw",
    }

    # Baseline 3: Always Away Win (class 0)
    y_pred_away = np.full(n_samples, 0)
    baselines["always_away"] = {
        "accuracy": accuracy_score(y_true, y_pred_away),
        "description": "Always predict Away Win",
    }

    # Baseline 4: Always predict favorite (lowest odds)
    if "odds_home" in df_fold.columns and "odds_draw" in df_fold.columns and "odds_away" in df_fold.columns:
        odds = df_fold[["odds_away", "odds_draw", "odds_home"]].values

        # Favorite is the outcome with lowest odds (highest implied probability)
        y_pred_favorite = np.argmin(odds, axis=1)

        baselines["always_favorite"] = {
            "accuracy": accuracy_score(y_true, y_pred_favorite),
            "description": "Always predict the favorite (lowest odds)",
        }

    return baselines


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred_classes: np.ndarray,
    save_path: str,
    class_names: Optional[list] = None,
) -> None:
    """
    Plot and save confusion matrix.

    Args:
        y_true: True class labels
        y_pred_classes: Predicted class labels
        save_path: Path to save the plot
        class_names: List of class names for labels (default: ['Away', 'Draw', 'Home'])
    """
    if class_names is None:
        class_names = ["Away", "Draw", "Home"]

    # Explicitly provide labels to handle missing classes
    cm = confusion_matrix(y_true, y_pred_classes, labels=[0, 1, 2])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    save_path: str,
    n_bins: int = 10,
) -> None:
    """
    Plot calibration curves for each class.

    A calibration curve shows how well predicted probabilities match actual
    frequencies. A perfectly calibrated model has points on the diagonal.

    Args:
        y_true: True class labels (0, 1, 2)
        y_pred_probs: Predicted probabilities, shape (n_samples, 3)
        save_path: Path to save the plot
        n_bins: Number of bins for calibration curve
    """
    class_names = ["Away", "Draw", "Home"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        # Convert to binary: class i vs rest
        y_binary = (y_true == i).astype(int)
        prob_class = y_pred_probs[:, i]

        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_binary, prob_class, n_bins=n_bins, strategy="uniform"
        )

        # Plot
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=class_name)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Calibration: {class_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
