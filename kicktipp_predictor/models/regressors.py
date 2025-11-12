"""
Experiment 2: Regressor Models
Predict exact goal counts for home and away teams
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple
from .. import config
from ..evaluation import evaluate_regressor


class RegressorExperiment:
    """
    Experiment 2: Regression approach

    Two separate Random Forest regressors:
    - Model 1: Predict home team goals
    - Model 2: Predict away team goals
    """

    def __init__(self):
        self.rf_home = None
        self.rf_away = None

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train_home: pd.Series,
        y_train_away: pd.Series,
        verbose: bool = False
    ) -> Tuple[RandomForestRegressor, RandomForestRegressor]:
        """
        Train two Random Forest regressors (home and away goals)

        Args:
            X_train: Training features (numerical only)
            y_train_home: Training home goals
            y_train_away: Training away goals
            verbose: Print training progress

        Returns:
            (model_home, model_away)
        """
        print("\nTraining Random Forest Regressors...")

        # Random Forest doesn't handle categorical features
        numerical_features = [f for f in X_train.columns if f in config.NUMERICAL_FEATURES]
        X_train_num = X_train[numerical_features]

        params = config.RF_REGRESSOR_PARAMS.copy()
        if verbose:
            params['verbose'] = 1

        # Train home goals model
        print("  - Training HOME goals model...")
        model_home = RandomForestRegressor(**params)
        model_home.fit(X_train_num, y_train_home)

        # Train away goals model
        print("  - Training AWAY goals model...")
        model_away = RandomForestRegressor(**params)
        model_away.fit(X_train_num, y_train_away)

        self.rf_home = model_home
        self.rf_away = model_away

        print("Random Forest Regressors trained.")

        return model_home, model_away

    def evaluate_random_forest(
        self,
        X_test: pd.DataFrame,
        y_test_home: pd.Series,
        y_test_away: pd.Series
    ) -> dict:
        """
        Evaluate Random Forest regressors on test set

        Args:
            X_test: Test features
            y_test_home: Test home goals
            y_test_away: Test away goals

        Returns:
            Results dictionary
        """
        if self.rf_home is None or self.rf_away is None:
            raise ValueError("Random Forest regressors not trained yet")

        # Use only numerical features
        numerical_features = [f for f in X_test.columns if f in config.NUMERICAL_FEATURES]
        X_test_num = X_test[numerical_features]

        y_pred_home = self.rf_home.predict(X_test_num)
        y_pred_away = self.rf_away.predict(X_test_num)

        results = evaluate_regressor(
            y_test_home.values,
            y_test_away.values,
            y_pred_home,
            y_pred_away,
            'Random Forest Regressor'
        )

        return results

    def predict_random_forest(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict scores using Random Forest regressors

        Args:
            X: Features

        Returns:
            (home_goals, away_goals) predictions (continuous)
        """
        if self.rf_home is None or self.rf_away is None:
            raise ValueError("Random Forest regressors not trained yet")

        # Use only numerical features
        numerical_features = [f for f in X.columns if f in config.NUMERICAL_FEATURES]
        X_num = X[numerical_features]

        home_goals = self.rf_home.predict(X_num)
        away_goals = self.rf_away.predict(X_num)

        return home_goals, away_goals
