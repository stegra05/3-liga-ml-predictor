"""
Experiment 2: Regressor Models
Predict exact goal counts for home and away teams
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple
from .. import config
from ..evaluation import evaluate_regressor


class RegressorExperiment:
    """
    Experiment 2: Regression approach

    Two separate models:
    - Model 1: Predict home team goals
    - Model 2: Predict away team goals
    """

    def __init__(self):
        self.catboost_home = None
        self.catboost_away = None
        self.rf_home = None
        self.rf_away = None

    def train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train_home: pd.Series,
        y_train_away: pd.Series,
        X_val: pd.DataFrame,
        y_val_home: pd.Series,
        y_val_away: pd.Series,
        verbose: bool = False
    ) -> Tuple[CatBoostRegressor, CatBoostRegressor]:
        """
        Train two CatBoost regressors (home and away goals)

        Args:
            X_train: Training features
            y_train_home: Training home goals
            y_train_away: Training away goals
            X_val: Validation features
            y_val_home: Validation home goals
            y_val_away: Validation away goals
            verbose: Print training progress

        Returns:
            (model_home, model_away)
        """
        print("\nTraining CatBoost Regressors...")

        # Get parameters and update verbosity
        params = config.CATBOOST_REGRESSOR_PARAMS.copy()
        if verbose:
            params['verbose'] = 100

        # Train home goals model
        print("  - Training HOME goals model...")
        train_pool_home = Pool(X_train, y_train_home, cat_features=config.CATEGORICAL_FEATURES)
        val_pool_home = Pool(X_val, y_val_home, cat_features=config.CATEGORICAL_FEATURES)

        model_home = CatBoostRegressor(**params)
        model_home.fit(train_pool_home, eval_set=val_pool_home, plot=False)

        # Train away goals model
        print("  - Training AWAY goals model...")
        train_pool_away = Pool(X_train, y_train_away, cat_features=config.CATEGORICAL_FEATURES)
        val_pool_away = Pool(X_val, y_val_away, cat_features=config.CATEGORICAL_FEATURES)

        model_away = CatBoostRegressor(**params)
        model_away.fit(train_pool_away, eval_set=val_pool_away, plot=False)

        self.catboost_home = model_home
        self.catboost_away = model_away

        print(f"CatBoost Regressors trained.")
        print(f"  Home model best iteration: {model_home.get_best_iteration()}")
        print(f"  Away model best iteration: {model_away.get_best_iteration()}")

        return model_home, model_away

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

    def evaluate_catboost(
        self,
        X_test: pd.DataFrame,
        y_test_home: pd.Series,
        y_test_away: pd.Series
    ) -> dict:
        """
        Evaluate CatBoost regressors on test set

        Args:
            X_test: Test features
            y_test_home: Test home goals
            y_test_away: Test away goals

        Returns:
            Results dictionary
        """
        if self.catboost_home is None or self.catboost_away is None:
            raise ValueError("CatBoost regressors not trained yet")

        y_pred_home = self.catboost_home.predict(X_test)
        y_pred_away = self.catboost_away.predict(X_test)

        results = evaluate_regressor(
            y_test_home.values,
            y_test_away.values,
            y_pred_home,
            y_pred_away,
            'CatBoost Regressor'
        )

        return results

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

    def predict_catboost(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict scores using CatBoost regressors

        Args:
            X: Features

        Returns:
            (home_goals, away_goals) predictions (continuous)
        """
        if self.catboost_home is None or self.catboost_away is None:
            raise ValueError("CatBoost regressors not trained yet")

        home_goals = self.catboost_home.predict(X)
        away_goals = self.catboost_away.predict(X)

        return home_goals, away_goals

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
