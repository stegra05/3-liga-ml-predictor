"""
Experiment 1: Classifier Models
Predict match outcome (H/D/A) and convert to scores
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import Tuple, Dict
from .. import config


def scores_from_class_predictions(y_pred_class: np.ndarray, default_scores: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert class predictions to score predictions using default scores
    
    Args:
        y_pred_class: Predicted classes (0=Away, 1=Draw, 2=Home)
        default_scores: Dictionary mapping class to [home_goals, away_goals]
    
    Returns:
        (home_goals, away_goals) predictions
    """
    home_goals = np.array([default_scores[cls][0] for cls in y_pred_class])
    away_goals = np.array([default_scores[cls][1] for cls in y_pred_class])
    return home_goals, away_goals


def calculate_kicktipp_points(pred_home: int, pred_away: int, actual_home: int, actual_away: int) -> int:
    """
    Calculate Kicktipp points for a single prediction
    
    Scoring:
    - 4 points: Exact score
    - 3 points: Correct result (home/away/draw) + correct goal difference
    - 2 points: Correct result only
    - 1 point: Correct goal difference only
    - 0 points: Nothing correct
    
    Args:
        pred_home: Predicted home goals
        pred_away: Predicted away goals
        actual_home: Actual home goals
        actual_away: Actual away goals
    
    Returns:
        Points awarded
    """
    # Exact score
    if pred_home == actual_home and pred_away == actual_away:
        return 4
    
    pred_diff = pred_home - pred_away
    actual_diff = actual_home - actual_away
    pred_result = 2 if pred_diff > 0 else (1 if pred_diff == 0 else 0)  # 2=Home, 1=Draw, 0=Away
    actual_result = 2 if actual_diff > 0 else (1 if actual_diff == 0 else 0)
    
    correct_result = (pred_result == actual_result)
    correct_diff = (pred_diff == actual_diff)
    
    if correct_result and correct_diff:
        return 3
    elif correct_result:
        return 2
    elif correct_diff:
        return 1
    else:
        return 0


def evaluate_classifier(
    y_test_class: np.ndarray,
    y_pred_class: np.ndarray,
    y_pred_proba: np.ndarray,
    y_test_home: np.ndarray,
    y_test_away: np.ndarray,
    default_scores: dict,
    model_name: str
) -> Dict:
    """
    Evaluate classifier performance
    
    Args:
        y_test_class: True classes
        y_pred_class: Predicted classes
        y_pred_proba: Predicted class probabilities
        y_test_home: True home goals
        y_test_away: True away goals
        default_scores: Score heuristics for each class
        model_name: Name of the model
    
    Returns:
        Results dictionary
    """
    # Calculate accuracy
    accuracy = accuracy_score(y_test_class, y_pred_class)
    
    # Convert predictions to scores
    pred_home, pred_away = scores_from_class_predictions(y_pred_class, default_scores)
    
    # Calculate Kicktipp points
    kicktipp_points = [
        calculate_kicktipp_points(int(pred_home[i]), int(pred_away[i]), 
                                  int(y_test_home[i]), int(y_test_away[i]))
        for i in range(len(y_test_class))
    ]
    avg_kicktipp_points = np.mean(kicktipp_points)
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'avg_kicktipp_points': avg_kicktipp_points,
        'total_matches': len(y_test_class)
    }


class ClassifierExperiment:
    """
    Experiment 1: Classification approach

    Random Forest classifier predicts outcome class (0=Away, 1=Draw, 2=Home)
    Then convert to score using heuristics
    """

    def __init__(self, default_scores: dict = None):
        """
        Args:
            default_scores: Score heuristics for each class
        """
        self.default_scores = default_scores or config.DEFAULT_SCORES
        self.rf_model = None

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        verbose: bool = False
    ) -> RandomForestClassifier:
        """
        Train Random Forest classifier

        Args:
            X_train: Training features (numerical only)
            y_train: Training target (class)
            verbose: Print training progress

        Returns:
            Trained model
        """
        print("\nTraining Random Forest Classifier...")

        # Random Forest doesn't handle categorical features
        # Use only numerical features
        numerical_features = [f for f in X_train.columns if f in config.NUMERICAL_FEATURES]
        X_train_num = X_train[numerical_features]

        model = RandomForestClassifier(**config.RF_CLASSIFIER_PARAMS)
        if verbose:
            model.set_params(verbose=1)

        model.fit(X_train_num, y_train)

        self.rf_model = model
        print("Random Forest trained.")

        return model

    def evaluate_random_forest(
        self,
        X_test: pd.DataFrame,
        y_test_class: pd.Series,
        y_test_home: pd.Series,
        y_test_away: pd.Series
    ) -> dict:
        """
        Evaluate Random Forest model on test set

        Args:
            X_test: Test features
            y_test_class: Test classes
            y_test_home: Test home goals (for Kicktipp scoring)
            y_test_away: Test away goals (for Kicktipp scoring)

        Returns:
            Results dictionary
        """
        if self.rf_model is None:
            raise ValueError("Random Forest model not trained yet")

        # Use only numerical features
        numerical_features = [f for f in X_test.columns if f in config.NUMERICAL_FEATURES]
        X_test_num = X_test[numerical_features]

        y_pred_class = self.rf_model.predict(X_test_num)
        y_pred_proba = self.rf_model.predict_proba(X_test_num)

        results = evaluate_classifier(
            y_test_class.values,
            y_pred_class,
            y_pred_proba,
            y_test_home.values,
            y_test_away.values,
            self.default_scores,
            'Random Forest Classifier'
        )

        return results

    def predict_random_forest(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict scores using Random Forest

        Args:
            X: Features

        Returns:
            (home_goals, away_goals) predictions
        """
        if self.rf_model is None:
            raise ValueError("Random Forest model not trained yet")

        # Use only numerical features
        numerical_features = [f for f in X.columns if f in config.NUMERICAL_FEATURES]
        X_num = X[numerical_features]

        y_pred_class = self.rf_model.predict(X_num)

        # Convert to scores
        home_goals, away_goals = scores_from_class_predictions(y_pred_class, self.default_scores)

        return home_goals, away_goals
