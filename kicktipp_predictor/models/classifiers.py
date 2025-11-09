"""
Experiment 1: Classifier Models
Predict match outcome (H/D/A) and convert to scores
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple
from .. import config
from ..evaluation import evaluate_classifier


class ClassifierExperiment:
    """
    Experiment 1: Classification approach

    Models predict outcome class (0=Away, 1=Draw, 2=Home)
    Then convert to score using heuristics
    """

    def __init__(self, default_scores: dict = None):
        """
        Args:
            default_scores: Score heuristics for each class
        """
        self.default_scores = default_scores or config.DEFAULT_SCORES
        self.catboost_model = None
        self.rf_model = None

    def train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        verbose: bool = False
    ) -> CatBoostClassifier:
        """
        Train CatBoost classifier

        Args:
            X_train: Training features
            y_train: Training target (class)
            X_val: Validation features
            y_val: Validation target (class)
            verbose: Print training progress

        Returns:
            Trained model
        """
        print("\nTraining CatBoost Classifier...")

        # Create pools
        train_pool = Pool(X_train, y_train, cat_features=config.CATEGORICAL_FEATURES)
        val_pool = Pool(X_val, y_val, cat_features=config.CATEGORICAL_FEATURES)

        # Get parameters and update verbosity
        params = config.CATBOOST_CLASSIFIER_PARAMS.copy()
        if verbose:
            params['verbose'] = 100

        # Train
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, plot=False)

        self.catboost_model = model
        print(f"CatBoost trained. Best iteration: {model.get_best_iteration()}")

        return model

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

    def evaluate_catboost(
        self,
        X_test: pd.DataFrame,
        y_test_class: pd.Series,
        y_test_home: pd.Series,
        y_test_away: pd.Series
    ) -> dict:
        """
        Evaluate CatBoost model on test set

        Args:
            X_test: Test features
            y_test_class: Test classes
            y_test_home: Test home goals (for Kicktipp scoring)
            y_test_away: Test away goals (for Kicktipp scoring)

        Returns:
            Results dictionary
        """
        if self.catboost_model is None:
            raise ValueError("CatBoost model not trained yet")

        y_pred_class = self.catboost_model.predict(X_test)
        y_pred_proba = self.catboost_model.predict_proba(X_test)

        results = evaluate_classifier(
            y_test_class.values,
            y_pred_class,
            y_pred_proba,
            y_test_home.values,
            y_test_away.values,
            self.default_scores,
            'CatBoost Classifier'
        )

        return results

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

    def predict_catboost(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict scores using CatBoost

        Args:
            X: Features

        Returns:
            (home_goals, away_goals) predictions
        """
        if self.catboost_model is None:
            raise ValueError("CatBoost model not trained yet")

        y_pred_class = self.catboost_model.predict(X)

        # Convert to scores
        from ..evaluation import scores_from_class_predictions
        home_goals, away_goals = scores_from_class_predictions(y_pred_class, self.default_scores)

        return home_goals, away_goals

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
        from ..evaluation import scores_from_class_predictions
        home_goals, away_goals = scores_from_class_predictions(y_pred_class, self.default_scores)

        return home_goals, away_goals
