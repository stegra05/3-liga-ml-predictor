# Getting Started with 3. Liga Football Prediction Dataset

Welcome! This guide will help you get up and running with the 3. Liga comprehensive football dataset for machine learning.

## Table of Contents

1. [What is This Project?](#what-is-this-project)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Quick Start: Using Pre-Built Datasets](#quick-start-using-pre-built-datasets)
5. [Understanding the Data](#understanding-the-data)
6. [Building Your First Model](#building-your-first-model)
7. [Next Steps](#next-steps)
8. [Getting Help](#getting-help)

---

## What is This Project?

This project provides a **comprehensive, ML-ready dataset** for predicting German 3. Liga football match outcomes. It's perfect for:

- 🎓 **Students & Researchers**: Learn machine learning with real-world sports data
- 🤖 **ML Practitioners**: Build and test prediction models
- ⚽ **Football Analytics**: Analyze team performance and trends
- 📊 **Data Scientists**: Explore feature engineering and model optimization

### What Makes This Dataset Special?

✅ **ML-Ready**: Pre-split into train/validation/test sets
✅ **Rich Features**: 103 features including ratings, form, odds, weather, and more
✅ **Long History**: 17 seasons (2009-2025) with 6,290+ matches
✅ **Research-Backed**: Includes Pi-ratings proven effective in academic research
✅ **No Setup Required**: Use pre-exported CSV files immediately

---

## Prerequisites

### Required Knowledge

- **Basic Python**: You should know variables, functions, and basic data structures
- **Pandas Basics**: Understanding DataFrames is helpful (but not required)
- **ML Fundamentals**: Basic understanding of classification/regression (optional)

### System Requirements

- **Python**: 3.8 or higher
- **Disk Space**: ~500 MB for dataset and dependencies
- **RAM**: 2 GB minimum (4 GB recommended)
- **Operating System**: Windows, macOS, or Linux

---

## Installation

### Step 1: Clone the Repository

```bash
# Using HTTPS
git clone https://github.com/yourusername/catboost-predictor.git
cd catboost-predictor

# Or using SSH
git clone git@github.com:yourusername/catboost-predictor.git
cd catboost-predictor
```

### Step 2: Set Up Python Environment

We recommend using a virtual environment to avoid dependency conflicts.

**Using venv (Standard Python):**
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**Using conda (Anaconda):**
```bash
# Create conda environment
conda create -n 3liga python=3.10
conda activate 3liga
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# For ML modeling, scikit-learn is already included in requirements.txt
```

### Step 4: Verify Installation

```bash
# Test that everything works
python -c "import pandas; import numpy; print('✅ Installation successful!')"
```

---

## Quick Start: Using Pre-Built Datasets

**Good news!** You don't need to run any collection scripts. The dataset is already prepared for you.

### Step 1: Load the Data

Create a new Python file (e.g., `my_first_analysis.py`):

```python
import pandas as pd

# Load the training data
train = pd.read_csv('data/processed/3liga_ml_dataset_train.csv')

# Display basic information
print(f"✅ Loaded {len(train)} training matches")
print(f"✅ Features available: {len(train.columns)}")
print(f"✅ Date range: {train['match_datetime'].min()} to {train['match_datetime'].max()}")

# Preview the data
print("\n📊 First few rows:")
print(train.head())
```

Run it:
```bash
python my_first_analysis.py
```

### Step 2: Explore the Data

```python
import pandas as pd

train = pd.read_csv('data/processed/3liga_ml_dataset_train.csv')

# Check target distribution
print("\n🎯 Match Outcomes:")
print(train['result'].value_counts())
print(f"\nHome wins: {(train['result'] == 'H').sum()} ({(train['result'] == 'H').mean()*100:.1f}%)")
print(f"Draws: {(train['result'] == 'D').sum()} ({(train['result'] == 'D').mean()*100:.1f}%)")
print(f"Away wins: {(train['result'] == 'A').sum()} ({(train['result'] == 'A').mean()*100:.1f}%)")

# Check feature availability
print("\n📈 Key Features Availability:")
for feature in ['home_elo', 'home_pi', 'home_points_l5', 'odds_home']:
    available = train[feature].notna().sum()
    print(f"{feature}: {available}/{len(train)} ({available/len(train)*100:.1f}%)")
```

### Step 3: Understand the Features

**Essential reading**: Check `docs/data/DATA_DICTIONARY.md` for complete feature descriptions.

**Key feature categories:**

| Category | Examples | Coverage | Use For |
|----------|----------|----------|---------|
| **Ratings** | `home_elo`, `pi_diff` | 100% | Primary predictors |
| **Form** | `home_points_l5`, `form_diff_l5` | 100% | Recent performance |
| **Odds** | `odds_home`, `implied_prob_home` | 98.6% | Market baseline |
| **Weather** | `temperature_celsius`, `is_rainy` | 81.9% | Environmental factors |
| **Stats** | `home_possession`, `home_shots` | 37.6% | Post-match analysis only |

---

## Understanding the Data

### What's in the Dataset?

```
data/processed/
├── 3liga_ml_dataset_full.csv      # Complete dataset (5,970 matches)
├── 3liga_ml_dataset_train.csv     # Training set (72%)
├── 3liga_ml_dataset_val.csv       # Validation set (8%)
├── 3liga_ml_dataset_test.csv      # Test set (20%)
├── feature_documentation.txt      # Feature descriptions
└── dataset_summary.txt            # Statistics overview
```

### Important Concepts

#### 1. **Temporal Splits**
The data is split chronologically (not randomly) because football is time-series data:
- **Train**: 2009-2021 matches (older data)
- **Validation**: 2021-2022 season
- **Test**: 2022-2025 seasons (most recent)

**Why?** This simulates real prediction: training on past data, predicting future matches.

#### 2. **Target Variables**

For **classification** (predicting match outcome):
```python
# Three-class classification
y = train['target_multiclass']  # 0=Away win, 1=Draw, 2=Home win

# Or binary classification
y = train['target_home_win']  # 1=Home win, 0=Not home win
```

For **regression** (predicting goals):
```python
y_home = train['target_home_goals']  # Home team goals
y_away = train['target_away_goals']  # Away team goals
y_total = train['target_total_goals']  # Total goals
```

#### 3. **Feature Types**

**✅ Use for prediction** (available BEFORE match):
- Ratings: `home_elo`, `away_pi`, `elo_diff`
- Form: `home_points_l5`, `away_goals_scored_l5`
- Odds: `odds_home`, `odds_draw`, `odds_away`
- Context: `rest_days_home`, `temperature_celsius`

**❌ Do NOT use for prediction** (only available AFTER match):
- Match stats: `home_possession`, `home_shots`, `home_passes`
- Advanced stats: `home_big_chances`, `away_interceptions`
- Results: `home_goals`, `away_goals`, `result`

---

## Building Your First Model

### Example 1: Simple Classification

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load data
train = pd.read_csv('data/processed/3liga_ml_dataset_train.csv')
test = pd.read_csv('data/processed/3liga_ml_dataset_test.csv')

# Select features (start with the most important ones, numerical only for Random Forest)
features = [
    'elo_diff',       # Elo rating difference
    'pi_diff',        # Pi rating difference
    'form_diff_l5',   # Form difference (last 5 matches)
    'home_elo',       # Home team Elo
    'away_elo',       # Away team Elo
    'home_points_l5', # Home team points (last 5)
    'away_points_l5', # Away team points (last 5)
]

# Prepare data
X_train = train[features]
y_train = train['target_multiclass']
X_test = test[features]
y_test = test['target_multiclass']

# Train model
print("🚀 Training model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Test Accuracy: {accuracy:.3f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred,
                          target_names=['Away Win', 'Draw', 'Home Win']))

# Show feature importance
print("\n🎯 Feature Importance:")
for feature, importance in zip(features, model.feature_importances_):
    print(f"{feature:20s}: {importance:.3f}")
```

**Expected Output:**
```
✅ Test Accuracy: 0.545
```

### Example 2: Using More Features

```python
# Add betting odds and form features
extended_features = [
    # Ratings
    'home_elo', 'away_elo', 'elo_diff',
    'home_pi', 'away_pi', 'pi_diff',

    # Form (last 5 matches)
    'home_points_l5', 'away_points_l5', 'form_diff_l5',
    'home_goals_scored_l5', 'home_goals_conceded_l5',
    'away_goals_scored_l5', 'away_goals_conceded_l5',

    # Betting odds
    'odds_home', 'odds_draw', 'odds_away',

    # Context
    'rest_days_home', 'rest_days_away',
]

# Filter out rows with missing values in key features
train_clean = train.dropna(subset=extended_features)
test_clean = test.dropna(subset=extended_features)

print(f"Dataset after cleaning: {len(train_clean)} train, {len(test_clean)} test")

# Train with extended features
X_train = train_clean[extended_features]
y_train = train_clean['target_multiclass']
X_test = test_clean[extended_features]
y_test = test_clean['target_multiclass']

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"\n✅ Test Accuracy with Extended Features: {accuracy:.3f}")
```

### Example 3: Goal Prediction (Regression)

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Same features as before
features = [
    'elo_diff', 'pi_diff', 'form_diff_l5',
    'home_goals_scored_l5', 'home_goals_conceded_l5',
    'away_goals_scored_l5', 'away_goals_conceded_l5',
]

X_train = train[features]
X_test = test[features]

# Predict home team goals
y_train_home = train['target_home_goals']
y_test_home = test['target_home_goals']

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train_home)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test_home, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_home, y_pred))

print(f"\n✅ Home Goals Prediction:")
print(f"   MAE: {mae:.3f} goals")
print(f"   RMSE: {rmse:.3f} goals")
```

---

## Next Steps

### 1. Explore Advanced Features

- Read `docs/data/DATA_DICTIONARY.md` for all 103 features
- Try adding weather features: `temperature_celsius`, `is_rainy`
- Experiment with head-to-head features: `h2h_home_win_rate`

### 2. Feature Engineering

```python
# Create new features
train['home_advantage'] = train['home_elo'] - train['away_elo']
train['form_ratio'] = train['home_points_l5'] / (train['away_points_l5'] + 1)
train['odds_ratio'] = train['odds_away'] / train['odds_home']
```

### 3. Try Different Hyperparameters

```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest with different hyperparameters
rf1 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf1.fit(X_train, y_train)

rf2 = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
rf2.fit(X_train, y_train)
```

### 4. Advanced Topics

- **Hyperparameter tuning**: Use `GridSearchCV` or `Optuna`
- **Ensemble methods**: Combine multiple models
- **Probability calibration**: Improve predicted probabilities
- **Custom loss functions**: Optimize for specific metrics

### 5. Update the Data

Want the latest matches? Use the unified CLI:

```bash
# Collect latest matches
python main.py collect-openligadb

# Recalculate ratings
python main.py rating-calculator

# Re-export ML datasets
python main.py export-ml-data
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.

---

## Getting Help

### Documentation

📖 **Full documentation** in the `docs/` directory:
- `docs/data/DATA_DICTIONARY.md` - Complete feature reference
- `docs/data/FBREF_INTEGRATION.md` - FBref data source details
- `docs/data/README.md` - Dataset overview

### Example Code

💡 **Working examples** in `examples/`:
- `examples/train_model_example.py` - Complete training script

### Common Issues

**Issue**: `FileNotFoundError` when loading data
```python
# Make sure you're in the project root directory
import os
print(os.getcwd())  # Should show .../catboost-predictor

# Or use absolute paths
df = pd.read_csv('/full/path/to/3liga_ml_dataset_train.csv')
```

**Issue**: Missing values in features
```python
# Check for missing data
print(train.isnull().sum())

# Option 1: Drop rows with missing values
train_clean = train.dropna(subset=['odds_home', 'odds_draw', 'odds_away'])

# Option 2: Fill missing values (be careful with this!)
train_filled = train.fillna({'odds_home': train['odds_home'].median()})
```

**Issue**: Low accuracy
- Start with **ratings + form** features (most predictive)
- Check your train/test split (should be temporal, not random)
- Try more iterations: `iterations=1000` or `2000`
- Tune hyperparameters: `depth`, `learning_rate`

### Community & Support

- **GitHub Issues**: Report bugs or ask questions
- **Discussions**: Share your models and findings
- **Pull Requests**: Contribute improvements (see [CONTRIBUTING.md](CONTRIBUTING.md))

---

## Quick Reference Card

### Load Data
```python
import pandas as pd
train = pd.read_csv('data/processed/3liga_ml_dataset_train.csv')
test = pd.read_csv('data/processed/3liga_ml_dataset_test.csv')
```

### Best Features to Start With
```python
features = ['elo_diff', 'pi_diff', 'form_diff_l5',
            'home_points_l5', 'away_points_l5']
```

### Train Model
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
model.fit(train[features], train['target_multiclass'])
```

### Evaluate
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test['target_multiclass'],
                          model.predict(test[features]))
print(f"Accuracy: {accuracy:.3f}")
```

---

**Ready to start?** Jump to [Building Your First Model](#building-your-first-model) and run the first example!

**Questions?** Open an issue on GitHub or check the [documentation](docs/).

**Happy predicting! ⚽🤖**
