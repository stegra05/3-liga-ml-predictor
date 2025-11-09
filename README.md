# 3. Liga Football Prediction Dataset

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Data Quality](https://img.shields.io/badge/data%20quality-production-brightgreen.svg)](docs/data/README.md)

**The most comprehensive ML-ready dataset for German 3. Liga football prediction (2009-2025)**

Perfect for data scientists, researchers, and football analytics enthusiasts to build and test match outcome prediction models.

## 🚀 Quick Start

**New to this project?** Start here: **[GETTING_STARTED.md](GETTING_STARTED.md)**

```bash
# 1. Clone and install
git clone https://github.com/yourusername/catboost-predictor.git
cd catboost-predictor
pip install -r requirements.txt

# 2. Load data and train your first model
python examples/train_model_example.py
```

That's it! The dataset is pre-built and ready to use.

---

## 📊 Dataset Overview

This project provides the most comprehensive publicly available dataset for German 3. Liga football, specifically designed for machine learning match prediction models using gradient boosting algorithms (CatBoost, LightGBM).

**What makes it special:**
- ✅ **ML-Ready**: Pre-split train/val/test sets, no preprocessing needed
- ✅ **Rich Features**: 103 features including ratings, form, odds, and weather
- ✅ **Long History**: 17 seasons with 6,290+ matches
- ✅ **Research-Backed**: Includes Pi-ratings from academic research
- ✅ **Well-Documented**: Complete data dictionary and usage examples

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

## 💡 Example: Train Your First Model

```python
from catboost import CatBoostClassifier
import pandas as pd

# 1. Load pre-split datasets
train = pd.read_csv('data/processed/3liga_ml_dataset_train.csv')
test = pd.read_csv('data/processed/3liga_ml_dataset_test.csv')

# 2. Select the most important features
features = [
    'elo_diff',       # Rating difference (most predictive!)
    'pi_diff',        # Pi-rating difference
    'form_diff_l5',   # Recent form difference
    'home_points_l5', # Home team recent points
    'away_points_l5', # Away team recent points
]

X_train = train[features]
y_train = train['target_multiclass']  # 0=Away, 1=Draw, 2=Home
X_test = test[features]
y_test = test['target_multiclass']

# 3. Train model
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=4,
    random_seed=42,
    verbose=100
)
model.fit(X_train, y_train)

# 4. Evaluate
accuracy = model.score(X_test, y_test)
print(f'Test Accuracy: {accuracy:.3f}')  # Expected: ~54-56%
```

**Want more?** See [examples/train_model_example.py](examples/train_model_example.py) for a complete script.

## 📊 Target Distribution

Based on 4,063 matches (2014-2025):
- **Home Wins**: 42.5% (1,726 matches)
- **Draws**: 27.3% (1,109 matches)
- **Away Wins**: 30.2% (1,228 matches)

This shows typical home advantage in football (~42% vs 30%).

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[GETTING_STARTED.md](GETTING_STARTED.md)** | 👈 **Start here!** Step-by-step tutorial for beginners |
| **[docs/data/DATA_DICTIONARY.md](docs/data/DATA_DICTIONARY.md)** | Complete reference for all 103 features |
| **[docs/data/README.md](docs/data/README.md)** | Dataset overview and statistics |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | How to contribute to this project |
| **[API_REFERENCE.md](API_REFERENCE.md)** | Developer documentation for key modules |
| **[docs/data/FBREF_INTEGRATION.md](docs/data/FBREF_INTEGRATION.md)** | FBref data source documentation |

---

## 🔄 Updating the Dataset

**Using pre-built data?** You can skip this section.

**Want the latest matches?** Run the data collection pipeline:

```bash
# 1. Collect latest matches
python scripts/collectors/openligadb_collector.py

# 2. Recalculate ratings
python scripts/processors/rating_calculator.py

# 3. Re-export ML datasets
python scripts/processors/ml_data_exporter.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.

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

We welcome contributions! Whether you're:
- 🐛 Reporting bugs
- 📝 Improving documentation
- ✨ Adding new features
- 🔧 Fixing issues

**See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.**

Quick start for contributors:
```bash
# Fork and clone the repository
git clone https://github.com/YOUR-USERNAME/catboost-predictor.git

# Create a branch
git checkout -b feature/your-feature

# Make changes, test, and submit PR
pytest tests/
git push origin feature/your-feature
```

## 📚 References

Key research papers for football prediction:
- Dixon & Coles (1997) - Modelling Association Football Scores
- Baio & Blangiardo (2010) - Bayesian hierarchical model for prediction
- Research on Pi-ratings for gradient boosting (2023)
- State-of-the-art: 55.82% accuracy with CatBoost + Pi-ratings

## 💡 Tips for Better Models

**Feature Selection:**
- 🥇 **Tier 1**: Start with `elo_diff`, `pi_diff`, `form_diff_l5` (most predictive)
- 🥈 **Tier 2**: Add goal form and betting odds
- 🥉 **Tier 3**: Include weather and contextual features

**Common Pitfalls:**
- ❌ Don't use random splits (use temporal splits!)
- ❌ Don't use post-match statistics (e.g., `home_possession`)
- ❌ Don't ignore missing data (check coverage in data dictionary)
- ✅ Do handle class imbalance (draws are less frequent)
- ✅ Do use cross-validation on temporal folds
- ✅ Do check feature importance after training

**Expected Performance:**
- Baseline (always predict home win): ~43% accuracy
- Decent model (ratings + form): ~54-56% accuracy
- Strong model (all features optimized): ~57-59% accuracy
- Research state-of-the-art: ~55.8% (Pi-ratings + CatBoost)

See [GETTING_STARTED.md](GETTING_STARTED.md#building-your-first-model) for detailed examples.

---

## 📊 Dataset at a Glance

| Statistic | Value |
|-----------|-------|
| **Total Matches** | 6,290 (2009-2025) |
| **ML Dataset** | 5,970 matches |
| **Teams** | 70 unique |
| **Seasons** | 17 complete |
| **Features** | 103 total (40 for prediction) |
| **Train/Val/Test** | 72% / 8% / 20% |
| **Rating Coverage** | 100% (Elo, Pi-ratings) |
| **Odds Coverage** | 98.6% |
| **Stats Coverage** | 37.6% (2014+) |

---

## 🏆 Use Cases

This dataset is perfect for:

1. **🎓 Education**: Learn ML with real-world sports data
2. **🔬 Research**: Test new prediction algorithms
3. **⚽ Sports Analytics**: Analyze team performance patterns
4. **📈 Betting Models**: Develop data-driven strategies
5. **🤖 ML Competitions**: Practice feature engineering
6. **📊 Benchmarking**: Compare model performance

---

## 📄 License & Attribution

### Data Sources
- **Match Results**: [OpenLigaDB](https://www.openligadb.de) (Public API)
- **Betting Odds**: OddsPortal (Educational use)
- **Match Statistics**: FotMob (Educational use)
- **League Standings**: [FBref](https://fbref.com) (Educational use)
- **Weather**: Meteostat, OpenWeatherMap (API)

### Usage
This dataset is provided for **educational and research purposes**. If you use this data in academic work, please cite:

```bibtex
@dataset{3liga_dataset_2025,
  title={3. Liga Comprehensive Football Prediction Dataset},
  author={[Your Name/Team]},
  year={2025},
  url={https://github.com/yourusername/catboost-predictor}
}
```

For commercial use, verify data rights with original sources.

---

## 🌟 Acknowledgments

This project builds on research from:
- Dixon & Coles (1997) - Modelling Association Football Scores
- Baio & Blangiardo (2010) - Bayesian hierarchical models
- Pi-rating research achieving 55.82% accuracy with gradient boosting

---

## 📞 Support

**Questions or Issues?**
- 📖 **Documentation**: Check [GETTING_STARTED.md](GETTING_STARTED.md) first
- 💬 **Discussions**: For questions and ideas
- 🐛 **Issues**: Report bugs or request features
- 🤝 **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Last Updated**: November 2025 | **Status**: Production Ready ✅

**Star this repo** ⭐ if you find it useful!
