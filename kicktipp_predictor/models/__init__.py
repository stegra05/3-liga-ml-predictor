"""
Model implementations for Kicktipp prediction
"""

from .classifiers import ClassifierExperiment
from .regressors import RegressorExperiment
from .ensemble import EnsembleExperiment

__all__ = ['ClassifierExperiment', 'RegressorExperiment', 'EnsembleExperiment']
