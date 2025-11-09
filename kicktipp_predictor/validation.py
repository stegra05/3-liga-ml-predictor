"""
Walk-Forward Validation Module
Implements time-series cross-validation for temporal data
"""

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, List


class WalkForwardValidator:
    """
    Walk-forward cross-validation for time-series data

    Ensures we always train on past data and validate on future data
    """

    def __init__(self, n_splits: int = 5):
        """
        Args:
            n_splits: Number of temporal splits
        """
        self.n_splits = n_splits
        self.splitter = TimeSeriesSplit(n_splits=n_splits)

    def get_splits(self, X) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get train/validation index splits

        Args:
            X: Feature matrix (just need the size)

        Returns:
            List of (train_idx, val_idx) tuples
        """
        return list(self.splitter.split(X))

    def validate_model(self, model_class, X, y, fit_params=None):
        """
        Perform walk-forward validation and return out-of-fold predictions

        Args:
            model_class: Model class with fit/predict methods
            X: Features
            y: Target
            fit_params: Additional parameters for fit() method

        Returns:
            out_of_fold_predictions: Predictions for all samples (in validation order)
            out_of_fold_indices: Indices corresponding to predictions
        """
        if fit_params is None:
            fit_params = {}

        oof_predictions = []
        oof_indices = []

        for fold, (train_idx, val_idx) in enumerate(self.get_splits(X), 1):
            # Split data
            X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]

            # Train model
            model = model_class()
            model.fit(X_train_fold, y_train_fold, **fit_params)

            # Predict
            if hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X_val_fold)
            else:
                preds = model.predict(X_val_fold)

            oof_predictions.append(preds)
            oof_indices.append(val_idx)

        # Concatenate all folds
        oof_predictions = np.vstack(oof_predictions)
        oof_indices = np.concatenate(oof_indices)

        return oof_predictions, oof_indices

    def validate_two_regressors(self, model_class, X, y_home, y_away, fit_params=None):
        """
        Perform walk-forward validation for two regressor models (home and away goals)

        Args:
            model_class: Model class with fit/predict methods
            X: Features
            y_home: Home goals target
            y_away: Away goals target
            fit_params: Additional parameters for fit() method

        Returns:
            oof_home: Out-of-fold predictions for home goals
            oof_away: Out-of-fold predictions for away goals
            oof_indices: Indices corresponding to predictions
        """
        if fit_params is None:
            fit_params = {}

        oof_home = []
        oof_away = []
        oof_indices = []

        for fold, (train_idx, val_idx) in enumerate(self.get_splits(X), 1):
            # Split data
            X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]

            y_home_train = y_home.iloc[train_idx] if hasattr(y_home, 'iloc') else y_home[train_idx]
            y_away_train = y_away.iloc[train_idx] if hasattr(y_away, 'iloc') else y_away[train_idx]

            # Train home goals model
            model_home = model_class()
            model_home.fit(X_train_fold, y_home_train, **fit_params)
            pred_home = model_home.predict(X_val_fold)

            # Train away goals model
            model_away = model_class()
            model_away.fit(X_train_fold, y_away_train, **fit_params)
            pred_away = model_away.predict(X_val_fold)

            oof_home.append(pred_home)
            oof_away.append(pred_away)
            oof_indices.append(val_idx)

        # Concatenate all folds
        oof_home = np.concatenate(oof_home)
        oof_away = np.concatenate(oof_away)
        oof_indices = np.concatenate(oof_indices)

        return oof_home, oof_away, oof_indices
