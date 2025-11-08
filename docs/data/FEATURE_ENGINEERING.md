# Feature Engineering Guide for 3. Liga Dataset

**Version:** 1.0  
**Last Updated:** 2025-11-08

This document provides a comprehensive guide to feature engineering opportunities for the 3. Liga dataset, including implemented features, recommended transformations, and advanced techniques.

---

## Table of Contents

1. [Current Features Overview](#current-features-overview)
2. [Feature Importance Hierarchy](#feature-importance-hierarchy)
3. [Recommended Feature Engineering](#recommended-feature-engineering)
4. [Advanced Techniques](#advanced-techniques)
5. [Feature Selection Strategies](#feature-selection-strategies)
6. [Temporal Considerations](#temporal-considerations)

---

## Current Features Overview

### Available Feature Groups

| Group | Count | Avg Coverage | Predictive Power |
|-------|-------|--------------|------------------|
| **Ratings** | 6 | 100% | ⭐⭐⭐⭐⭐ Very High |
| **Form Metrics** | 10 | 100% | ⭐⭐⭐⭐⭐ Very High |
| **Match Statistics** | 10 | 21.5% | ⭐⭐⭐⭐ High |
| **Betting Odds** | 4 | 39.0% | ⭐⭐⭐⭐ High |
| **Context** | 43 | 51.0% | ⭐⭐⭐ Medium |

---

## Feature Importance Hierarchy

Based on research and empirical testing, features can be ranked by predictive power:

### Tier 1: Essential Features (Must Have) 🔴

**Team Ratings** - *Single most important feature group*

```python
essential_features = [
    'home_elo',          # Elo rating home team
    'away_elo',          # Elo rating away team
    'elo_diff',          # Difference (home - away)
    'home_pi',           # Pi-rating home team
    'away_pi',           # Pi-rating away team
    'pi_diff',           # Difference (home - away)
]
```

**Why These Matter:**
- Elo ratings capture long-term team strength
- Pi-ratings weight recent performance heavily
- Difference features provide relative strength directly
- Research shows 55.82% accuracy with just these + form

**Expected Feature Importance:** ~40-50% of model decisions

### Tier 2: Core Features (Highly Recommended) 🟡

**Recent Form Indicators**

```python
core_features = [
    'home_points_l5',           # Last 5 matches points
    'away_points_l5',
    'form_diff_l5',             # Form difference
    'home_goals_scored_l5',     # Offensive form
    'home_goals_conceded_l5',   # Defensive form
    'away_goals_scored_l5',
    'away_goals_conceded_l5',
]
```

**Why These Matter:**
- Capture momentum and current form
- Goals scored/conceded = offensive/defensive strength
- Short-term trends complement long-term ratings
- Handle form changes (injuries, coaching changes, etc.)

**Expected Feature Importance:** ~25-30% of model decisions

### Tier 3: Valuable Features (Recommended) 🟢

**Match Statistics & Odds**

```python
valuable_features = [
    # Betting odds (market consensus)
    'odds_home',
    'odds_draw', 
    'odds_away',
    
    # Match statistics (when available)
    'home_possession',
    'away_possession',
    'home_shots_on_target',
    'away_shots_on_target',
    'home_corners',
    'away_corners',
    
    # League context
    'home_position',
    'away_position',
    'position_diff',
]
```

**Why These Matter:**
- Odds encode market wisdom and inside information
- Possession/shots indicate playing style
- League position captures season trajectory
- Provides additional signal beyond ratings

**Expected Feature Importance:** ~15-20% of model decisions

### Tier 4: Contextual Features (Optional) ⚪

**Temporal & Environmental**

```python
contextual_features = [
    'matchday',          # Season progression
    'is_weekend',        # Match timing
    'temperature',       # Weather (minor impact)
    'is_home',           # Home advantage (already in ratings)
]
```

**Why These Matter:**
- Season phase (early vs late) affects motivation
- Weather has marginal impact in football
- Useful for ensemble diversity, less for single models

**Expected Feature Importance:** ~5-10% of model decisions

---

## Recommended Feature Engineering

### 1. Ratio Features

Transform absolute values into ratios for better model learning:

```python
def create_ratio_features(df):
    """Create ratio-based features"""
    
    # Elo ratio (avoids negative values, emphasizes relative strength)
    df['elo_ratio'] = df['home_elo'] / df['away_elo']
    
    # Form ratio
    df['form_ratio_l5'] = (df['home_points_l5'] + 1) / (df['away_points_l5'] + 1)
    
    # Goal scoring ratio
    df['goals_ratio_l5'] = (df['home_goals_scored_l5'] + 0.1) / (df['away_goals_scored_l5'] + 0.1)
    
    # Defensive ratio (lower is better for defense)
    df['defense_ratio_l5'] = (df['away_goals_conceded_l5'] + 0.1) / (df['home_goals_conceded_l5'] + 0.1)
    
    # Odds ratio
    df['odds_ratio_home_away'] = df['odds_away'] / df['odds_home']
    
    return df
```

**Benefits:**
- Tree models handle ratios better than differences
- Emphasizes relative strength over absolute
- Reduces feature space dimensionality

### 2. Interaction Features

Combine features to capture non-linear relationships:

```python
def create_interaction_features(df):
    """Create interaction features"""
    
    # Rating × Form (strong team in good form = very dangerous)
    df['home_rating_form'] = df['home_elo'] * (df['home_points_l5'] / 15.0)
    df['away_rating_form'] = df['away_elo'] * (df['away_points_l5'] / 15.0)
    
    # Offensive strength (goals × position)
    df['home_attack_strength'] = df['home_goals_scored_l5'] * (21 - df['home_position'])
    df['away_attack_strength'] = df['away_goals_scored_l5'] * (21 - df['away_position'])
    
    # Momentum indicator (recent form × rating change)
    # Requires tracking rating changes over time
    
    # Home advantage amplifier
    df['home_advantage_strength'] = df['elo_diff'] * df['is_home']
    
    return df
```

**Use Cases:**
- Capture "hot teams" (good rating + good form)
- Identify mismatches (strong attack vs weak defense)
- Add complexity without too many features

### 3. Temporal Features

Extract time-based patterns:

```python
def create_temporal_features(df):
    """Create temporal features"""
    
    # Season phase
    df['season_phase'] = pd.cut(df['matchday'], 
                                 bins=[0, 10, 28, 38],
                                 labels=['early', 'mid', 'late'])
    
    # Matches since season start
    df['matches_into_season'] = df['matchday']
    
    # Is crucial match (late season + close positions)
    df['is_crucial'] = ((df['matchday'] > 30) & 
                        (abs(df['position_diff']) <= 3)).astype(int)
    
    # Days since last match (if available)
    # df['rest_days_home'] = ...
    # df['rest_days_away'] = ...
    
    return df
```

**Insights:**
- Early season: teams finding form, ratings less reliable
- Late season: relegation/promotion pressure affects performance
- Crucial matches: higher motivation, less predictable

### 4. Aggregation Features

Create features from historical aggregations:

```python
def create_aggregation_features(df, matches_history):
    """Create features from match history"""
    
    # Home/away split performance
    df['home_elo_home_only'] = df['team'].map(
        matches_history[matches_history['is_home'] == 1]
        .groupby('team')['elo_rating'].mean()
    )
    
    df['away_elo_away_only'] = df['opponent'].map(
        matches_history[matches_history['is_home'] == 0]
        .groupby('team')['elo_rating'].mean()
    )
    
    # Scoring consistency (std dev of goals)
    df['home_goals_consistency'] = df['team'].map(
        matches_history.groupby('team')['goals_scored'].std()
    )
    
    # Win rate in last 10
    df['home_win_rate_l10'] = df['team'].map(
        matches_history.groupby('team')['result'].apply(
            lambda x: (x.tail(10) == 'W').sum() / 10
        )
    )
    
    return df
```

### 5. Normalization & Scaling

```python
def normalize_features(df):
    """Normalize features for better model training"""
    
    # Z-score normalization for continuous features
    from sklearn.preprocessing import StandardScaler
    
    continuous_features = [
        'elo_diff', 'pi_diff', 'form_diff_l5',
        'home_goals_scored_l5', 'home_goals_conceded_l5'
    ]
    
    scaler = StandardScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])
    
    # Min-max scaling for bounded features
    df['possession_scaled'] = df['home_possession'] / 100.0
    
    return df
```

**Note:** CatBoost and LightGBM generally don't require normalization, but it can help with:
- Interpretation of feature importance
- Regularization effectiveness
- Faster convergence

---

## Advanced Techniques

### 1. Rolling Window Features

Capture trends over different time windows:

```python
def create_rolling_features(df, windows=[3, 5, 10]):
    """Create rolling window statistics"""
    
    for window in windows:
        # Rolling average Elo
        df[f'home_elo_ma{window}'] = (
            df.groupby('home_team')['home_elo']
            .rolling(window, min_periods=1).mean()
            .reset_index(drop=True)
        )
        
        # Rolling win rate
        df[f'home_win_rate_l{window}'] = (
            df.groupby('home_team')['target_home_win']
            .rolling(window, min_periods=1).mean()
            .reset_index(drop=True)
        )
        
        # Rolling goals scored
        df[f'home_goals_ma{window}'] = (
            df.groupby('home_team')['target_home_goals']
            .rolling(window, min_periods=1).mean()
            .reset_index(drop=True)
        )
    
    return df
```

**Benefits:**
- Capture trends at multiple time scales
- Smooth out variance in performance
- Identify momentum shifts

### 2. Head-to-Head Features

Historical matchup patterns:

```python
def create_h2h_features(df, match_history):
    """Create head-to-head features"""
    
    def get_h2h_stats(row):
        home = row['home_team']
        away = row['away_team']
        
        # Get historical H2H matches
        h2h = match_history[
            ((match_history['home_team'] == home) & (match_history['away_team'] == away)) |
            ((match_history['home_team'] == away) & (match_history['away_team'] == home))
        ]
        
        if len(h2h) == 0:
            return {'h2h_matches': 0, 'h2h_home_wins': 0, 'h2h_draws': 0}
        
        # Calculate H2H stats
        home_wins = (h2h['home_team'] == home) & (h2h['result'] == 'H')
        away_wins = (h2h['away_team'] == home) & (h2h['result'] == 'A')
        
        return {
            'h2h_matches': len(h2h),
            'h2h_home_wins': (home_wins | away_wins).sum(),
            'h2h_draws': (h2h['result'] == 'D').sum(),
            'h2h_avg_goals': h2h['total_goals'].mean()
        }
    
    h2h_stats = df.apply(get_h2h_stats, axis=1, result_type='expand')
    return pd.concat([df, h2h_stats], axis=1)
```

### 3. Streak Features

Identify hot and cold streaks:

```python
def create_streak_features(df):
    """Create winning/losing streak features"""
    
    def calculate_streak(results):
        """Calculate current streak length"""
        if len(results) == 0:
            return 0
        
        current = results.iloc[-1]
        streak = 1
        
        for i in range(len(results) - 2, -1, -1):
            if results.iloc[i] == current:
                streak += 1
            else:
                break
        
        return streak if current == 'W' else -streak if current == 'L' else 0
    
    # Calculate streaks for each team
    df['home_streak'] = (
        df.groupby('home_team')['result']
        .rolling(window=38, min_periods=1)
        .apply(calculate_streak, raw=False)
        .reset_index(drop=True)
    )
    
    return df
```

### 4. Style Compatibility Features

Identify playing style matchups:

```python
def create_style_features(df):
    """Create playing style features"""
    
    # Possession style
    df['possession_diff'] = df['home_possession'] - df['away_possession']
    
    # Attacking style (shots per match)
    df['attacking_style_home'] = df['home_shots_on_target'] / (df['home_shots'] + 0.1)
    df['attacking_style_away'] = df['away_shots_on_target'] / (df['away_shots'] + 0.1)
    
    # Defensive style (fouls as proxy for pressing)
    df['pressing_home'] = df['home_fouls'] / (df['home_possession'] + 1)
    df['pressing_away'] = df['away_fouls'] / (df['away_possession'] + 1)
    
    # Style mismatch
    df['style_mismatch'] = abs(df['attacking_style_home'] - df['attacking_style_away'])
    
    return df
```

---

## Feature Selection Strategies

### 1. Correlation Analysis

```python
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_feature_correlations(df, target='target_multiclass'):
    """Analyze feature correlations with target"""
    
    # Calculate correlations
    correlations = df.corr()[target].sort_values(ascending=False)
    
    # Plot top features
    plt.figure(figsize=(10, 8))
    correlations.head(20).plot(kind='barh')
    plt.title('Top 20 Feature Correlations with Target')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig('feature_correlations.png')
    
    return correlations
```

### 2. Tree-Based Feature Importance

```python
from catboost import CatBoostClassifier
import pandas as pd

def get_feature_importance(X_train, y_train):
    """Get feature importance from CatBoost"""
    
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance
```

### 3. Permutation Importance

```python
from sklearn.inspection import permutation_importance

def get_permutation_importance(model, X_test, y_test):
    """Calculate permutation importance"""
    
    perm_importance = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=42
    )
    
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    return importance_df
```

### 4. Recursive Feature Elimination

```python
from sklearn.feature_selection import RFE

def recursive_feature_elimination(model, X, y, n_features=20):
    """Select top N features using RFE"""
    
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(X, y)
    
    selected_features = X.columns[rfe.support_]
    
    return selected_features
```

---

## Temporal Considerations

### Critical: Avoid Data Leakage

**Never use future information for past predictions:**

```python
# ❌ WRONG: Uses future match results
df['season_points'] = df.groupby('team')['points'].cumsum()

# ✅ CORRECT: Uses only past results
df['season_points'] = df.groupby('team')['points'].shift(1).fillna(0).cumsum()
```

### Train/Validation/Test Splits

**Use temporal splits, not random splits:**

```python
def temporal_split(df, train_end, val_end):
    """Split data temporally"""
    
    train = df[df['match_datetime'] < train_end]
    val = df[(df['match_datetime'] >= train_end) & (df['match_datetime'] < val_end)]
    test = df[df['match_datetime'] >= val_end]
    
    return train, val, test

# Example: 70% train, 10% validation, 20% test
train_end = '2022-06-01'
val_end = '2023-06-01'

train, val, test = temporal_split(df, train_end, val_end)
```

### Walk-Forward Validation

```python
def walk_forward_validation(df, n_splits=5):
    """Perform walk-forward validation"""
    
    df = df.sort_values('match_datetime')
    fold_size = len(df) // n_splits
    
    for i in range(1, n_splits):
        train = df[:fold_size * i]
        test = df[fold_size * i:fold_size * (i + 1)]
        
        # Train and evaluate
        # ...
```

---

## Implementation Checklist

### Phase 1: Basic Features
- [ ] Implement Elo ratings
- [ ] Implement Pi-ratings
- [ ] Calculate form metrics (L5, L10)
- [ ] Add betting odds
- [ ] Create basic difference features

### Phase 2: Enhanced Features
- [ ] Add ratio features
- [ ] Create interaction features
- [ ] Implement rolling windows
- [ ] Add temporal features
- [ ] Calculate H2H statistics

### Phase 3: Advanced Features
- [ ] Style compatibility metrics
- [ ] Streak features
- [ ] Season phase indicators
- [ ] Motivation factors
- [ ] Rest days / fatigue

### Phase 4: Feature Selection
- [ ] Correlation analysis
- [ ] Feature importance from models
- [ ] Permutation importance
- [ ] Remove redundant features
- [ ] Select optimal feature set

---

## Best Practices

1. **Start Simple:** Begin with Tier 1 features only
2. **Add Incrementally:** Add feature groups one at a time and measure impact
3. **Avoid Leakage:** Always use `shift()` for historical features
4. **Handle Missing:** Decide strategy (drop, impute, or flag)
5. **Document Everything:** Keep track of feature definitions
6. **Version Control:** Track feature engineering code changes
7. **Test Thoroughly:** Validate on holdout test set
8. **Monitor Drift:** Check if feature distributions change over time

---

## Expected Performance by Feature Set

| Feature Set | Expected Accuracy | RPS Score |
|-------------|------------------|-----------|
| **Baseline (Ratings only)** | 50-52% | 0.210 |
| **Ratings + Form** | 53-55% | 0.195 |
| **Ratings + Form + Odds** | 54-56% | 0.190 |
| **All Features** | 55-57% | 0.185 |
| **All + Advanced Engineering** | 56-58% | 0.180 |

*Based on research and empirical testing on similar leagues*

---

## References

- Dixon & Coles (1997) - Modelling Association Football Scores
- Baio & Blangiardo (2010) - Bayesian hierarchical model for prediction
- Research on Pi-ratings for gradient boosting (2023)
- State-of-the-art: 55.82% accuracy with CatBoost + Pi-ratings

---

*Part of the 3. Liga Dataset Documentation*
