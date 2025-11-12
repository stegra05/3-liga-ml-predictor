"""
Configuration for Kicktipp Prediction System
Defines the 39 predictive features and model parameters
"""

# ============================================================================
# FEATURE DEFINITIONS (39 Core Predictive Features)
# ============================================================================

CATEGORICAL_FEATURES = [
    'home_team',
    'away_team',
    'day_of_week',
    'month',
    'is_midweek'
]

RATING_FEATURES = [
    'home_elo',
    'away_elo',
    'elo_diff',
    'home_pi',
    'away_pi',
    'pi_diff'
]

FORM_FEATURES = [
    'home_points_l5',
    'away_points_l5',
    'form_diff_l5',
    'home_points_l10',
    'away_points_l10',
    'home_goals_scored_l5',
    'home_goals_conceded_l5',
    'away_goals_scored_l5',
    'away_goals_conceded_l5',
    'goal_diff_l5'
]

ODDS_FEATURES = [
    'odds_home',
    'odds_draw',
    'odds_away',
    'implied_prob_home',
    'implied_prob_draw',
    'implied_prob_away'
]

CONTEXT_FEATURES = [
    'rest_days_home',
    'rest_days_away',
    'rest_days_diff',
    'travel_distance_km',
    'temperature_celsius',
    'humidity_percent',
    'wind_speed_kmh',
    'precipitation_mm'
]

H2H_FEATURES = [
    'h2h_total_matches',
    'h2h_home_win_rate',
    'h2h_draw_rate',
    'h2h_match_count'
]

# All 40 predictive features
ALL_FEATURES = (
    CATEGORICAL_FEATURES +
    RATING_FEATURES +
    FORM_FEATURES +
    ODDS_FEATURES +
    CONTEXT_FEATURES +
    H2H_FEATURES
)

# Numerical features only (for models that can't handle categorical)
NUMERICAL_FEATURES = (
    RATING_FEATURES +
    FORM_FEATURES +
    ODDS_FEATURES +
    CONTEXT_FEATURES +
    H2H_FEATURES +
    ['day_of_week', 'month', 'is_midweek']  # These can be treated as numerical
)

# ============================================================================
# TARGET VARIABLES
# ============================================================================

TARGET_CLASSIFICATION = 'target_multiclass'  # 0=Away, 1=Draw, 2=Home
TARGET_HOME_GOALS = 'target_home_goals'
TARGET_AWAY_GOALS = 'target_away_goals'

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Walk-Forward Validation
N_SPLITS = 5  # Number of temporal splits

# Random Forest Parameters
RF_CLASSIFIER_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'random_state': 42,
    'n_jobs': -1
}

RF_REGRESSOR_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'random_state': 42,
    'n_jobs': -1
}

# ============================================================================
# DATA PATHS
# ============================================================================

DATA_DIR = './data/processed'
TRAIN_FILE = f'{DATA_DIR}/3liga_ml_dataset_train.csv'
VAL_FILE = f'{DATA_DIR}/3liga_ml_dataset_val.csv'
TEST_FILE = f'{DATA_DIR}/3liga_ml_dataset_test.csv'

# ============================================================================
# SCORE HEURISTICS FOR CLASSIFIERS
# ============================================================================

# Default scores for each outcome based on most common scorelines
DEFAULT_SCORES = {
    2: [1, 0],  # Home win: 1-0
    1: [1, 1],  # Draw: 1-1
    0: [0, 1]   # Away win: 0-1
}
