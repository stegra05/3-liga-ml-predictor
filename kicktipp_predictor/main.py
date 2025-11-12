"""
Main Execution Script for Kicktipp Prediction System

Runs all three experiments and compares results
"""

import pandas as pd
import numpy as np
from typing import List
import sys

# This module is deprecated - all analysis/evaluation modules have been removed
# For predictions, use: python main.py predict
from . import config
from .data_loader import load_datasets, prepare_features, prepare_regression_features, combine_train_val
from .models.classifiers import ClassifierExperiment


def print_header(title: str):
    """Print section header"""
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)


# Analysis/evaluation functions removed - this module is deprecated


def main():
    """
    Main execution function

    Steps:
    1. Load data
    2. Prepare features
    3. Run all three experiments
    4. Compare results
    5. Identify best model
    """
    # This module is deprecated - all analysis/evaluation functionality has been removed
    # For predictions, use: python main.py predict
    print("This module is deprecated. All analysis/evaluation modules have been removed.")
    print("For predictions, use: python main.py predict")
    sys.exit(2)


if __name__ == "__main__":
    print("This script is deprecated. This module is for internal use only.", file=sys.stderr)
    print("For predictions, use: python main.py predict", file=sys.stderr)
    sys.exit(2)
