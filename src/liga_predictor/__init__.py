"""
Liga Predictor - 3. Liga Match Prediction System

A comprehensive machine learning system for predicting German 3. Liga football matches.
Includes data collection, processing, feature engineering, and ML model training/prediction.
"""

__version__ = "1.0.0"

# Core modules
from .database import get_db, DatabaseManager
from .config import (
    FEATURE_COLUMNS,
    RF_CLASSIFIER_PARAMS,
    DEFAULT_SCORES,
    CATEGORICAL_FEATURES,
    RATING_FEATURES,
    FORM_FEATURES,
    ODDS_FEATURES,
    CONTEXT_FEATURES,
    H2H_FEATURES
)
from .predictor import MatchPredictor

# Subpackages are available via explicit imports
# from liga_predictor.collection import OpenLigaDBCollector
# from liga_predictor.processing import MLDataExporter
# from liga_predictor.modeling import ClassifierExperiment
# from liga_predictor.utils import TeamMapper

__all__ = [
    '__version__',
    'get_db',
    'DatabaseManager',
    'MatchPredictor',
    'FEATURE_COLUMNS',
    'RF_CLASSIFIER_PARAMS',
    'DEFAULT_SCORES',
    'CATEGORICAL_FEATURES',
    'RATING_FEATURES',
    'FORM_FEATURES',
    'ODDS_FEATURES',
    'CONTEXT_FEATURES',
    'H2H_FEATURES'
]
