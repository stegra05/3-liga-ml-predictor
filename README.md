# 3. Liga Comprehensive Football Dataset

**Extensive dataset for machine learning match prediction in German 3. Liga (2009-2025)**

## 📊 Dataset Overview

This project provides the most comprehensive publicly available dataset for German 3. Liga football, specifically designed for machine learning match prediction models using gradient boosting algorithms (CatBoost, LightGBM).

### Current Database Statistics

```
Total Matches: 6,290 (2009-2025)
Teams: 70
Match Statistics: 4,446 detailed records
League Standings: 320 records
Betting Odds: 1,247 records
Team Ratings: 9,646 (Elo, Pi-ratings for all finished matches)
Seasons: 17 complete seasons
```

### ML-Ready Export Statistics

```
Total ML Dataset: 4,063 matches (2014-2025, when detailed stats available)
Training Set: 2,925 matches (72%)
Validation Set: 325 matches (8%)
Test Set: 813 matches (20%)

Features: 73 total
  - Rating features (Elo, Pi-ratings): 100% coverage
  - Form metrics (last 5/10 games): 100% coverage
  - Detailed match statistics: 53.6% coverage
  - Betting odds: 18.7% coverage
```

## 🎯 Key Features

### 1. **Match Results & Basic Data** (100% coverage, 2009-2025)
- Match outcomes, scores, dates, venues
- Home/away team information
- Season and matchday details
- Source: OpenLigaDB API

### 2. **Rating Systems** (100% coverage for finished matches)
Research shows these are the **most predictive features** for gradient boosting models:

- **Elo Ratings**: Dynamic skill ratings updated after each match
- **Pi-Ratings**: Research-proven best features for tree-based models
  - Based on weighted recent performance
  - Shown to achieve state-of-the-art accuracy in football prediction
- **Form Metrics**: Points and goals in last 5/10 matches
- All ratings calculated **BEFORE** each match for proper prediction

### 3. **Detailed Match Statistics** (53.6% coverage, 2014-2025)
From FotMob:
- Possession percentages
- Shots (total, on target, big chances)
- Passing stats (total passes, accuracy, crosses)
- Defensive actions (tackles, interceptions, clearances)
- Duels and aerial battles
- Fouls, cards, corners, offsides

### 4. **Attendance Data** (Collection in progress, 2009-2025)
From Transfermarkt:
- Match attendance figures for all seasons
- Automated collection system
- See `docs/ATTENDANCE_COLLECTION.md` for details

### 5. **Betting Odds** (18.7% coverage, 2009-2025)
From OddsPortal:
- Closing odds for Home/Draw/Away
- Implied probabilities
- Market expectations baseline

### 6. **League Standings** (2009-2025)
### 5. **League Standings** (2009-2025)
- Historical standings after each matchday
- Position, points, wins/draws/losses
- Goals for/against, goal difference

## 📁 Project Structure

```
catboost-predictor/
├── database/
│   ├── 3liga.db                 # SQLite database
│   ├── schema.sql               # Database schema
│   └── db_manager.py            # Database operations
│
├── scripts/
│   ├── collectors/
│   │   └── openligadb_collector.py    # API data collection
│   ├── processors/
│   │   ├── import_existing_data.py    # Import CSV files
│   │   ├── rating_calculator.py       # Elo/Pi-ratings
│   │   ├── ml_data_exporter.py        # Export ML datasets
│   │   └── clean_team_mappings.py     # Team name standardization
│   └── utils/
│       └── team_mapper.py             # Team name mapping
│
├── config/
│   └── team_mappings.json       # Team name standardization
│
├── data/
│   ├── raw/                     # Original CSV files
│   └── processed/               # ML-ready exports
│       ├── 3liga_ml_dataset_full.csv
│       ├── 3liga_ml_dataset_train.csv
│       ├── 3liga_ml_dataset_val.csv
│       ├── 3liga_ml_dataset_test.csv
│       ├── feature_documentation.txt
│       └── dataset_summary.txt
│
└── requirements.txt             # Python dependencies
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database (if not already done)
python database/db_manager.py
```

### Load ML-Ready Data

```python
import pandas as pd

# Load pre-split datasets
train = pd.read_csv('data/processed/3liga_ml_dataset_train.csv')
val = pd.read_csv('data/processed/3liga_ml_dataset_val.csv')
test = pd.read_csv('data/processed/3liga_ml_dataset_test.csv')

# Features for prediction
predictive_features = [
    'home_elo', 'away_elo', 'elo_diff',
    'home_pi', 'away_pi', 'pi_diff',
    'home_points_l5', 'away_points_l5', 'form_diff_l5',
    'home_goals_scored_l5', 'home_goals_conceded_l5',
    'away_goals_scored_l5', 'away_goals_conceded_l5',
    'odds_home', 'odds_draw', 'odds_away',
    # ... see feature_documentation.txt for full list
]

# Targets
# Classification: target_multiclass (0=Away, 1=Draw, 2=Home)
# Regression: target_home_goals, target_away_goals
```

### Example: Train CatBoost Model

```python
from catboost import CatBoostClassifier
import pandas as pd

# Load data
train = pd.read_csv('data/processed/3liga_ml_dataset_train.csv')
test = pd.read_csv('data/processed/3liga_ml_dataset_test.csv')

# Define features (ratings + form are most important)
features = [
    'elo_diff', 'pi_diff', 'form_diff_l5',
    'home_elo', 'away_elo',
    'home_pi', 'away_pi',
    'home_points_l5', 'away_points_l5',
    'home_goals_scored_l5', 'home_goals_conceded_l5',
    'away_goals_scored_l5', 'away_goals_conceded_l5',
]

X_train = train[features]
y_train = train['target_multiclass']
X_test = test[features]
y_test = test['target_multiclass']

# Train model
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    random_seed=42,
    verbose=100
)

model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f'Test Accuracy: {accuracy:.3f}')
```

## 📊 Target Distribution

Based on 4,063 matches (2014-2025):
- **Home Wins**: 42.5% (1,726 matches)
- **Draws**: 27.3% (1,109 matches)
- **Away Wins**: 30.2% (1,228 matches)

This shows typical home advantage in football (~42% vs 30%).

## 🔄 Data Collection & Updates

### Automated Data Collection

```bash
# Collect latest data from OpenLigaDB
python scripts/collectors/openligadb_collector.py

# Recalculate ratings
python scripts/processors/rating_calculator.py

# Export updated ML datasets
python scripts/processors/ml_data_exporter.py
```

### Manual Data Import

```bash
# Import additional CSV data
python scripts/processors/import_existing_data.py
```

## 📈 Research-Backed Features

### Pi-Ratings: State-of-the-Art Performance

Research (Dixon & Coles, 1997; Baio & Blangiardo, 2010) shows that **Pi-ratings** combined with gradient boosting achieve:
- **55.82% accuracy** on match outcome prediction
- **Rank Probability Score (RPS): 0.1925** (lower is better)
- Superior performance vs. raw statistics or Elo alone

### Recommended Feature Set

Based on research for gradient-boosted tree models:

**Tier 1 (Essential):**
- Pi-ratings (home/away/diff)
- Elo ratings (home/away/diff)
- Form metrics (points last 5/10)

**Tier 2 (Important):**
- Goal scoring/conceding trends
- Betting odds (market baseline)
- Match context (home/away, matchday)

**Tier 3 (Supplementary):**
- Detailed match statistics (when available)
- League position
- Head-to-head history

## 🎯 Use Cases

### 1. Match Outcome Prediction
```python
# Binary classification (win/not win)
target = 'target_home_win'

# 3-way classification (home/draw/away)
target = 'target_multiclass'
```

### 2. Goal Prediction
```python
# Regression for goals
target_home = 'target_home_goals'
target_away = 'target_away_goals'
target_total = 'target_total_goals'
```

### 3. Research & Analysis
- Team performance analysis
- Rating system comparison
- Feature importance studies
- Betting strategy development

## 📝 Data Quality Notes

### Coverage by Period

**2009-2014**: Basic match results + standings only
- Good for: Historical analysis, basic models
- Missing: Detailed statistics

**2014-2018**: Results + partial statistics + ratings
- Good for: Training basic models
- Statistics coverage: ~40-50%

**2018-2025**: Results + detailed statistics + ratings + odds
- Good for: Full-featured models
- Statistics coverage: ~70-80%

### Recommended Usage

For **best model performance**, use data from **2014-2025** (included in exported datasets):
- 100% coverage of rating features (Elo, Pi)
- 53.6% coverage of detailed statistics
- Sufficient data for robust training (4,063 matches)

## 🔮 Future Enhancements

Potential additions (not currently implemented):
- ❌ Player-level statistics
- ❌ Transfer market data
- ❌ Weather conditions
- ❌ xG (Expected Goals) - not available for 3. Liga
- ❌ Event-level tracking data - only available for top leagues

## 📄 License

Data sources:
- **OpenLigaDB**: Free API, no authentication required
- **FotMob**: Publicly scraped data (use responsibly)
- **OddsPortal**: Historical odds data

**Usage**: This dataset is for research and educational purposes. Please cite appropriately if used in publications.

## 🤝 Contributing

To update or extend the dataset:

1. **Add new data sources**: Create collector in `scripts/collectors/`
2. **Improve features**: Modify `ml_data_exporter.py`
3. **Fix data quality**: Update `import_existing_data.py`

## 📚 References

Key research papers for football prediction:
- Dixon & Coles (1997) - Modelling Association Football Scores
- Baio & Blangiardo (2010) - Bayesian hierarchical model for prediction
- Research on Pi-ratings for gradient boosting (2023)
- State-of-the-art: 55.82% accuracy with CatBoost + Pi-ratings

## 💡 Tips for ML Models

1. **Use temporal splits**: Football is time-series data
2. **Feature importance**: Ratings > Form > Stats > Odds
3. **Handle missing data**: Use only available features
4. **Class imbalance**: Consider class weights for draws
5. **Ensemble models**: Combine predictions from multiple models
6. **Validation**: Use recent seasons for testing

---

**Dataset Statistics**: 6,290 matches · 70 teams · 17 seasons · 73 ML features

**Last Updated**: November 2025
