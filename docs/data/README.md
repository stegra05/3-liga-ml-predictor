# 3. Liga Dataset Documentation

**Generated:** 2025-11-08
**Dataset Quality:** Production Ready
**Total Matches:** 6,290 (database) / 5,970 (ML dataset)
**Period:** 2009-2026 (17 seasons)

---

## Documentation Files

### 📖 [DATA_DICTIONARY.md](DATA_DICTIONARY.md)
**Complete feature reference** - Essential for modeling
- All features documented
- Type, description, range, coverage for each
- Categorized by use case
- Missing data patterns
- Feature engineering recommendations

### 🔗 [FBREF_INTEGRATION.md](FBREF_INTEGRATION.md)
**FBref data integration** - New data source (2025-11-09)
- Integration overview and architecture
- Technical implementation details
- Coverage: 2018-2019 onwards (8 seasons)
- Available data: League standings, player stats, team aggregates
- Usage instructions and maintenance procedures

---

## Dataset Statistics

### Database Overview
- **Total matches:** 6,290
- **Teams:** 68 unique
- **Ratings:** 11,940 Elo/Pi records
- **Statistics:** 4,574 detailed match stats
- **Betting odds:** 40,055 records
- **Head-to-head:** 1,283 matchup records
- **Players:** 500+ individual records (FBref, 2018+)
- **League standings:** 160+ season records (FBref, 2018+)

### ML Dataset
- **Total matches:** 5,970 (filtered, cleaned)
- **Features:** 103 total
- **Train set:** 4,298 matches (72%)
- **Validation set:** 478 matches (8%)
- **Test set:** 1,194 matches (20%)

---

## Usage Examples

### Loading Dataset
```python
import pandas as pd

# Load full dataset
df = pd.read_csv('data/processed/3liga_ml_dataset_full.csv')

# Load splits
train = pd.read_csv('data/processed/3liga_ml_dataset_train.csv')
val = pd.read_csv('data/processed/3liga_ml_dataset_val.csv')
test = pd.read_csv('data/processed/3liga_ml_dataset_test.csv')
```

### Filtering by Quality
```python
# Use only matches with all core features
core_features = [
    'home_elo', 'away_elo', 'home_pi', 'away_pi',
    'home_points_l5', 'away_points_l5',
    'odds_home', 'odds_draw', 'odds_away'
]

df_high_quality = df[df[core_features].notna().all(axis=1)]
print(f"High quality subset: {len(df_high_quality)} matches")
```

---

## License & Attribution

**Data Sources:**
- **Match results:** OpenLigaDB (public API)
- **Betting odds:** OddsPortal (scraped, educational use)
- **Match statistics:** FotMob (scraped, educational use)
- **Weather:** Meteostat, OpenWeatherMap (API access)
- **Player & standings:** FBref (scraped, educational use)

**Usage:**
- This dataset is for educational and research purposes
- Commercial use requires verification of data rights
- Attribution recommended for derived works
