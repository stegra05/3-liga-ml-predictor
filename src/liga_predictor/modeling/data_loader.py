"""
Data Loading and Preprocessing Module
Handles loading datasets and preparing features
"""

import pandas as pd
import numpy as np
from typing import Tuple
from liga_predictor import config


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the pre-split train, validation, and test datasets

    Returns:
        train, val, test DataFrames
    """
    train = pd.read_csv(config.TRAIN_FILE)
    val = pd.read_csv(config.VAL_FILE)
    test = pd.read_csv(config.TEST_FILE)

    print(f"Loaded datasets:")
    print(f"  Train: {len(train)} matches")
    print(f"  Val:   {len(val)} matches")
    print(f"  Test:  {len(test)} matches")

    return train, val, test


def prepare_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    use_categorical: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list]:
    """
    Prepare features for modeling

    Args:
        train: Training dataset
        val: Validation dataset
        test: Test dataset
        use_categorical: Whether to use categorical features (False for Random Forest models)

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    """
    # Choose feature set
    if use_categorical:
        features = config.ALL_FEATURES
    else:
        features = config.NUMERICAL_FEATURES

    # Check which features are actually available in the data
    available_features = [f for f in features if f in train.columns]
    missing_features = [f for f in features if f not in train.columns]

    if missing_features:
        print(f"Warning: {len(missing_features)} features not found in data:")
        print(f"  {missing_features[:5]}...")  # Show first 5

    features = available_features

    # Handle missing values
    # For numerical features: fill with median
    # For categorical features: fill with 'MISSING'
    for feat in features:
        if feat in config.CATEGORICAL_FEATURES:
            train[feat] = train[feat].fillna('MISSING')
            val[feat] = val[feat].fillna('MISSING')
            test[feat] = test[feat].fillna('MISSING')
        else:
            # Use train median for all datasets
            median_val = train[feat].median()
            train[feat] = train[feat].fillna(median_val)
            val[feat] = val[feat].fillna(median_val)
            test[feat] = test[feat].fillna(median_val)

    # Prepare X and y
    X_train = train[features]
    y_train = train[config.TARGET_CLASSIFICATION]

    X_val = val[features]
    y_val = val[config.TARGET_CLASSIFICATION]

    X_test = test[features]
    y_test = test[config.TARGET_CLASSIFICATION]

    print(f"\nUsing {len(features)} features for modeling")

    return X_train, y_train, X_val, y_val, X_test, y_test, features


def prepare_regression_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    use_categorical: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series, list]:
    """
    Prepare features for regression (predicting goals)

    Returns:
        X_train, y_train_home, y_train_away, X_val, y_val_home, y_val_away, X_test, y_test_home, y_test_away, features
    """
    # Choose feature set
    if use_categorical:
        features = config.ALL_FEATURES
    else:
        features = config.NUMERICAL_FEATURES

    # Check availability
    available_features = [f for f in features if f in train.columns]
    features = available_features

    # Handle missing values
    for feat in features:
        if feat in config.CATEGORICAL_FEATURES:
            train[feat] = train[feat].fillna('MISSING')
            val[feat] = val[feat].fillna('MISSING')
            test[feat] = test[feat].fillna('MISSING')
        else:
            median_val = train[feat].median()
            train[feat] = train[feat].fillna(median_val)
            val[feat] = val[feat].fillna(median_val)
            test[feat] = test[feat].fillna(median_val)

    # Prepare X and y
    X_train = train[features]
    y_train_home = train[config.TARGET_HOME_GOALS]
    y_train_away = train[config.TARGET_AWAY_GOALS]

    X_val = val[features]
    y_val_home = val[config.TARGET_HOME_GOALS]
    y_val_away = val[config.TARGET_AWAY_GOALS]

    X_test = test[features]
    y_test_home = test[config.TARGET_HOME_GOALS]
    y_test_away = test[config.TARGET_AWAY_GOALS]

    return X_train, y_train_home, y_train_away, X_val, y_val_home, y_val_away, X_test, y_test_home, y_test_away, features


def combine_train_val(
    train: pd.DataFrame,
    val: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine train and validation sets for final model training

    Args:
        train: Training dataset
        val: Validation dataset

    Returns:
        Combined dataset
    """
    combined = pd.concat([train, val], ignore_index=True)
    print(f"Combined train+val: {len(combined)} matches")
    return combined
