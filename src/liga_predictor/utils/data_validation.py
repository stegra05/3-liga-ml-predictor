"""
Data Validation Utilities for Temporal Leakage Detection
Ensures no future information leaks into training/prediction features
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Tuple, Optional


class TemporalLeakageDetector:
    """Detects potential temporal leakage in ML datasets"""

    def __init__(self, df: pd.DataFrame, datetime_col: str = 'match_datetime'):
        """
        Initialize detector

        Args:
            df: DataFrame to validate
            datetime_col: Name of the datetime column
        """
        self.df = df.copy()
        self.datetime_col = datetime_col
        self.issues = []

    def validate_all(self) -> Tuple[bool, List[str]]:
        """
        Run all validation checks

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        logger.info("Running temporal leakage validation checks...")

        self.check_h2h_monotonicity()
        self.check_rating_consistency()
        self.check_future_data_presence()
        self.check_post_match_features()

        if self.issues:
            logger.error(f"Found {len(self.issues)} potential data leakage issues:")
            for issue in self.issues:
                logger.error(f"  - {issue}")
            return False, self.issues
        else:
            logger.success("✓ No temporal leakage detected")
            return True, []

    def check_h2h_monotonicity(self) -> None:
        """
        Verify H2H counts are monotonic (never decrease over time)
        For each team pair, H2H total should increase or stay constant
        """
        if 'h2h_total_matches' not in self.df.columns:
            return

        # Create team pair identifier
        self.df['_team_pair'] = self.df.apply(
            lambda row: tuple(sorted([row['home_team_id'], row['away_team_id']])),
            axis=1
        )

        # Sort by datetime
        df_sorted = self.df.sort_values(self.datetime_col)

        # Check for each team pair
        violations = []
        for pair, group in df_sorted.groupby('_team_pair'):
            h2h_counts = group['h2h_total_matches'].values
            if len(h2h_counts) > 1:
                # Check if any count decreases
                if np.any(np.diff(h2h_counts) < 0):
                    violations.append(f"Team pair {pair}: H2H count decreases over time")

        if violations:
            self.issues.append(f"H2H monotonicity violation: {len(violations)} team pairs have decreasing H2H counts")
            for v in violations[:5]:  # Show first 5
                self.issues.append(f"  {v}")

    def check_rating_consistency(self) -> None:
        """
        Verify ratings exist and are reasonable
        Ratings should not have impossible jumps
        """
        rating_cols = ['home_elo', 'away_elo', 'home_pi', 'away_pi']
        missing_cols = [col for col in rating_cols if col not in self.df.columns]

        if missing_cols:
            return

        # Check for impossible Elo values (typical range: 800-2200)
        for col in ['home_elo', 'away_elo']:
            if col in self.df.columns:
                invalid = self.df[(self.df[col] < 500) | (self.df[col] > 2500)][col].notna()
                if invalid.any():
                    self.issues.append(
                        f"{col}: Found {invalid.sum()} values outside realistic range [500, 2500]"
                    )

    def check_future_data_presence(self) -> None:
        """
        Check if any match has features from matches with later dates
        This is a sophisticated check for the specific H2H case
        """
        if 'h2h_total_matches' not in self.df.columns:
            return

        # For each match, verify its H2H count is consistent with historical data
        df_sorted = self.df.sort_values(self.datetime_col).reset_index(drop=True)

        for idx, row in df_sorted.iterrows():
            current_datetime = row[self.datetime_col]
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            current_h2h = row['h2h_total_matches']

            # Count actual historical matches between these teams
            historical_matches = df_sorted[
                (df_sorted[self.datetime_col] < current_datetime) &
                (((df_sorted['home_team_id'] == home_id) & (df_sorted['away_team_id'] == away_id)) |
                 ((df_sorted['home_team_id'] == away_id) & (df_sorted['away_team_id'] == home_id)))
            ]

            actual_count = len(historical_matches)

            # Allow for matches from before the dataset start
            if current_h2h > actual_count + 20:  # 20 = reasonable buffer for pre-dataset history
                self.issues.append(
                    f"Match {row['match_id']} ({current_datetime}): "
                    f"H2H count ({current_h2h}) suspiciously higher than dataset history ({actual_count})"
                )

    def check_post_match_features(self) -> None:
        """
        Identify features that are only available after the match
        These should be flagged for training but not prediction
        """
        post_match_indicators = [
            'possession', 'shots', 'corners', 'pass_accuracy',
            'tackles', 'fouls', 'yellow_cards', 'red_cards'
        ]

        # Check if any post-match features have suspiciously high coverage
        post_match_cols = [
            col for col in self.df.columns
            if any(indicator in col.lower() for indicator in post_match_indicators)
        ]

        if post_match_cols:
            coverage = {}
            for col in post_match_cols:
                pct_available = (self.df[col].notna().sum() / len(self.df)) * 100
                coverage[col] = pct_available

            high_coverage = {k: v for k, v in coverage.items() if v > 50}
            if high_coverage:
                logger.warning(
                    "Warning: Found post-match features with high coverage. "
                    "Ensure these are excluded from prediction pipeline:"
                )
                for col, pct in high_coverage.items():
                    logger.warning(f"  {col}: {pct:.1f}% coverage")


class DatasetValidator:
    """Validates ML dataset quality and integrity"""

    @staticmethod
    def validate_train_test_split(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        datetime_col: str = 'match_datetime'
    ) -> bool:
        """
        Validate temporal train/test split

        Args:
            train_df: Training set
            test_df: Test set
            datetime_col: DateTime column name

        Returns:
            True if valid, False otherwise
        """
        train_max = train_df[datetime_col].max()
        test_min = test_df[datetime_col].min()

        if train_max >= test_min:
            logger.error(
                f"TEMPORAL SPLIT VIOLATION: Train set contains matches from {train_max}, "
                f"but test set starts at {test_min}. This creates data leakage!"
            )
            return False

        logger.success(
            f"✓ Valid temporal split: train ends {train_max}, test starts {test_min}"
        )
        return True

    @staticmethod
    def check_feature_availability(
        df: pd.DataFrame,
        required_features: List[str],
        threshold: float = 0.8
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if required features have sufficient coverage

        Args:
            df: DataFrame to check
            required_features: List of feature names
            threshold: Minimum required coverage (0-1)

        Returns:
            Tuple of (is_valid, coverage_dict)
        """
        coverage = {}
        issues = []

        for feat in required_features:
            if feat not in df.columns:
                coverage[feat] = 0.0
                issues.append(f"Missing feature: {feat}")
            else:
                pct = df[feat].notna().sum() / len(df)
                coverage[feat] = pct
                if pct < threshold:
                    issues.append(f"Low coverage for {feat}: {pct*100:.1f}%")

        if issues:
            logger.warning(f"Feature availability issues ({len(issues)}):")
            for issue in issues:
                logger.warning(f"  - {issue}")
            return False, coverage

        logger.success(f"✓ All {len(required_features)} features have sufficient coverage")
        return True, coverage


def validate_ml_export(df: pd.DataFrame, datetime_col: str = 'match_datetime') -> bool:
    """
    Convenience function to run all validations on ML export

    Args:
        df: Exported ML dataset
        datetime_col: DateTime column name

    Returns:
        True if all validations pass
    """
    logger.info("=" * 60)
    logger.info("VALIDATING ML DATASET FOR TEMPORAL LEAKAGE")
    logger.info("=" * 60)

    detector = TemporalLeakageDetector(df, datetime_col)
    is_valid, issues = detector.validate_all()

    logger.info("=" * 60)
    if is_valid:
        logger.success("✓ Dataset passed all validation checks")
    else:
        logger.error("✗ Dataset has validation issues - review before training")

    return is_valid


if __name__ == "__main__":
    print("Use CLI instead: liga-predictor validate-data")
