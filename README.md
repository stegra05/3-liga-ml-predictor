# 3. Liga Match Predictor

## Backtested Performance (eye-catchers)

- **+142.13% ROI** in 2025 static pre-season backtest (MD 1–14, 119 bets, **+169.13 units**)
- **+103.85% ROI** in 2025 rolling matchday backtest (117 bets, **+121.50 units**)
- **+67.84% ROI** average across 2018–2026 (sliding-season, 2017 bets, **+1368.30 units**)
- **+61.15% ROI** average across 2015–2026 (expanding-season, 2887 bets, **+1765.48 units**)
- Strong accuracy vs baselines: up to **0.714** (2025–26 sliding), baselines ≤ **0.471**

| Mode | Horizon | Accuracy | ROI % | P&L (units) | # Bets |
|---|---|---:|---:|---:|---:|
| Static Pre-Season | 2025 MD 1–14 | 0.836 | +142.13 | +169.13 | 119 |
| Rolling Matchday | 2025 | 0.736 | +103.85 | +121.50 | 117 |
| Sliding Season | 2018–2026 | 0.596 | +67.84 | +1368.30 | 2017 |
| Expanding Season | 2015–2026 | 0.587 | +61.15 | +1765.48 | 2887 |

Baselines for comparison (accuracy): Home 0.425, Draw 0.273, Away 0.303, Favorite 0.471.

To inspect runs locally, log results to MLflow and open the UI:
- Run: `mlflow ui` then open `http://localhost:5000`

A comprehensive machine learning system for predicting German 3. Liga football match results. The system aggregates historical data from multiple sources, engineers predictive features (ratings, form, odds, weather, head-to-head), and uses a Random Forest Classifier model to generate match predictions with detailed explanations.

## Features

- **Live Weather Integration**: Fetches real-time weather forecasts via Open-Meteo API, with fallback to historical estimates
- **Comprehensive Feature Engineering**: Includes Elo ratings, Pi-ratings, recent form, head-to-head statistics, betting odds, and contextual factors
- **SQLite Database**: Robust database schema storing match data, statistics, ratings, and more
- **Modern CLI**: Typer-based command-line interface (`liga-predictor`) with intuitive commands
- **ML-Ready Datasets**: Export feature-engineered datasets to CSV for your own modeling experiments
- **Model Evaluation**: Comprehensive backtesting framework with multiple evaluation modes and metrics

## Requirements

- **Python**: 3.9 to 3.14
- **Poetry**: For dependency management
- **SQLite 3**: Database engine (usually pre-installed)
- **Optional**: Chrome + chromedriver for Selenium-based data collectors (FBref, OddsPortal, Transfermarkt)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd 3-liga-ml-predictor

# Install dependencies using Poetry
poetry install
```

## Quick Start

### Predict Next Matchday (Default)

Simply run without arguments to predict the next upcoming matchday:

```bash
poetry run liga-predictor
```

### Explicit Prediction with Options

```bash
# Predict specific season/matchday, update data first, and save to CSV
poetry run liga-predictor predict \
  --season 2025 \
  --matchday 15 \
  --update-data \
  --weather-mode live \
  --output outputs/predictions_2025_MD15.csv
```

**Weather Modes:**
- `live`: Fetch live weather forecast from Open-Meteo API (default)
- `estimate`: Use historical weather estimates based on similar time periods
- `off`: Use default weather values

## Database Setup

Before making predictions, initialize the SQLite database schema:

```bash
poetry run liga-predictor db-init
```

This creates `database/3liga.db` with all required tables based on `database/schema.sql`.

## Data Collection

The system supports collecting data from multiple sources. Run these commands as needed to update your database:

```bash
# Matches & fixtures from OpenLigaDB API
poetry run liga-predictor collect-openligadb

# Team standings & player statistics from FBref
poetry run liga-predictor collect-fbref

# Betting odds from OddsPortal
poetry run liga-predictor collect-oddsportal

# Referee data from Transfermarkt
poetry run liga-predictor collect-transfermarkt
```

## Export ML Datasets

Export feature-engineered datasets for machine learning model training:

```bash
poetry run liga-predictor export-ml-data
```

This creates the following files in `data/processed/`:

- `3liga_ml_dataset_full.csv` - Complete dataset with all matches
- `3liga_ml_dataset_train.csv` - Training set (temporal split)
- `3liga_ml_dataset_val.csv` - Validation set
- `3liga_ml_dataset_test.csv` - Test set

Each dataset includes 100+ engineered features including ratings, form, odds, weather, and head-to-head statistics.

## Model Evaluation

The system includes comprehensive backtesting capabilities to evaluate model performance across different time periods and training strategies. Use the `evaluate` command to run evaluations:

```bash
# Expanding season evaluation (default)
poetry run liga-predictor evaluate

# Sliding season window evaluation
poetry run liga-predictor evaluate \
  --mode sliding-season \
  --start-season 2014 \
  --window-size 4

# Rolling matchday evaluation (retrain after each matchday)
poetry run liga-predictor evaluate \
  --mode rolling-matchday \
  --start-season 2014 \
  --test-season 2025

# Static pre-season evaluation (single model, no retraining)
poetry run liga-predictor evaluate \
  --mode static-preseason \
  --start-season 2014 \
  --test-season 2025
```

**Evaluation Modes:**
- `expanding-season`: Train on all past data, test on each future season (simulates growing dataset)
- `sliding-season`: Train on a fixed window of recent seasons, test on the next season (tests concept drift)
- `rolling-matchday`: Retrain after each matchday within a season (most realistic scenario)
- `static-preseason`: Single model trained once, tested throughout season (shows performance decay)

**Metrics Calculated:**
- Classification accuracy
- Log loss (cross-entropy)
- Brier score
- Ranked Probability Score (RPS)
- P&L simulation with betting strategies
- Per-class precision, recall, and F1 scores
- Baseline comparisons (always home/draw/away, favorite based on odds)

Results are displayed in a formatted table and can optionally be logged to MLflow for tracking and comparison.

## Stadium Locations

The predictor resolves stadium coordinates for weather lookups using the following priority:

1. **JSON Config** (preferred): `src/liga_predictor/config/stadium_locations.json`
2. **Database Fallback**: `team_locations` table

If you see warnings about missing `stadium_locations.json`, you can either:

- Copy `config/stadium_locations.json` to `src/liga_predictor/config/stadium_locations.json`, or
- Populate the database with locations:
  ```bash
  poetry run liga-predictor build-locations
  ```

## Project Structure

Key paths in the project:

- **Database**: `database/3liga.db` (SQLite database file)
- **Schema**: `database/schema.sql` (database schema definition)
- **Processed Data**: `data/processed/` (ML-ready CSV exports)
- **Models**: `models/rf_classifier.pkl` (default model path)
- **CLI**: `src/liga_predictor/cli.py` (command-line interface)
- **Evaluation**: `src/liga_predictor/evaluation.py` (backtesting framework)
- **Metrics**: `src/liga_predictor/metrics.py` (evaluation metrics)
- **Config**: `src/liga_predictor/config/` (configuration files)

## CLI Reference

Get help for any command:

```bash
poetry run liga-predictor --help
poetry run liga-predictor predict --help
```

### Common Commands

**Prediction & Evaluation:**
- `predict` - Run match predictions (default command when run without arguments)
- `evaluate` - Run backtest evaluations to test model performance across different scenarios

**Database & Setup:**
- `db-init` - Initialize database schema

**Data Collection:**
- `collect-openligadb` - Collect matches and fixtures from OpenLigaDB API
- `collect-fbref` - Collect team standings and player stats from FBref
- `collect-oddsportal` - Collect 1X2 betting odds from OddsPortal
- `collect-transfermarkt` - Collect referee data from Transfermarkt

**Data Processing:**
- `export-ml-data` - Export ML-ready datasets to `data/processed/`
- `build-locations` - Build team location mappings for travel distance calculations
- `build-h2h` - Compute head-to-head statistics table
- `calculate-ratings` - Calculate and update team ratings (Elo, Pi, etc.)
- `fetch-weather` - Fetch historical weather data from multiple sources

## Troubleshooting

### Stadium Coordinates Missing

If you see warnings about missing stadium coordinates:

- Ensure `src/liga_predictor/config/stadium_locations.json` exists, or
- Run `poetry run liga-predictor build-locations` to populate the database

### Weather Unavailable

If weather data is unavailable for a specific venue:

- Use `--weather-mode estimate` to fall back to historical weather estimates
- The system will automatically use seasonal defaults if no historical data is available

### No Data for Matchday

If predictions fail due to missing data:

- Run with `--update-data` flag to fetch latest data before predicting:
  ```bash
  poetry run liga-predictor predict --season 2025 --matchday 15 --update-data
  ```
- Or manually run the appropriate data collectors listed above

## Development

### Running Tests

```bash
poetry run pytest
```

### Code Style

The project uses Black and isort for code formatting (configured in `pyproject.toml`).

## License

This project is intended for research and educational use. Please check the repository for license details.

## Acknowledgments

- **OpenLigaDB** - Match data and fixtures API
- **FBref** - Team standings and player statistics
- **OddsPortal** - Betting odds data
- **Transfermarkt** - Referee information
- **Open-Meteo** - Weather forecast data provider

## Additional Documentation

For more detailed information about:

- **Using the ML datasets**: See [GETTING_STARTED.md](GETTING_STARTED.md)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) (if available)
- **Data dictionary**: Check `docs/data/DATA_DICTIONARY.md` (if available)

## Detailed Backtest Results

- **Dataset coverage**: 2009-2010 to 2025-2026, 5,973 matches
- **Baselines (accuracy)**: Home 0.425, Draw 0.273, Away 0.303, Favorite 0.471
- **Notes**:
  - Sliding-season uses a 4-season training window
  - Rolling-matchday retrains after each matchday within the season
  - Static pre-season trains once and stays fixed through the season

### Expanding Season (train grows across seasons)

| Fold | Samples | Accuracy | RPS | Brier | LogLoss | P&L (units) | # Bets | ROI % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Season 2015-2016 | 380 | 0.500 | 0.3961 | 0.5985 | 0.9986 | +124.51 | 303 | +41.09 |
| Season 2016-2017 | 380 | 0.566 | 0.3760 | 0.5497 | 0.9222 | +172.53 | 288 | +59.91 |
| Season 2017-2018 | 380 | 0.576 | 0.3743 | 0.5456 | 0.9153 | +94.68 | 290 | +32.65 |
| Season 2018-2019 | 380 | 0.555 | 0.3889 | 0.5599 | 0.9309 | +169.88 | 282 | +60.24 |
| Season 2019-2020 | 380 | 0.555 | 0.3708 | 0.5575 | 0.9250 | +177.11 | 273 | +64.88 |
| Season 2020-2021 | 380 | 0.537 | 0.3844 | 0.5565 | 0.9262 | +196.67 | 306 | +64.27 |
| Season 2021-2022 | 374 | 0.594 | 0.3581 | 0.5134 | 0.8522 | +225.22 | 308 | +73.12 |
| Season 2022-2023 | 140 | 0.686 | 0.3392 | 0.4677 | 0.8063 | +74.50 | 111 | +67.12 |
| Season 2023-2024 | 380 | 0.584 | 0.3702 | 0.5296 | 0.8826 | +221.94 | 316 | +70.23 |
| Season 2024-2025 | 379 | 0.609 | 0.3534 | 0.5048 | 0.8488 | +198.22 | 297 | +66.74 |
| Season 2025-2026 | 140 | 0.693 | 0.2836 | 0.4221 | 0.7348 | +110.22 | 113 | +97.54 |
| AVERAGE | 3693 | 0.587 | 0.3632 | 0.5277 | 0.8857 | +1765.48 | 2887 | +61.15 |

### Static Pre-Season 2025 (single model, no retraining)

| Fold | Samples | Accuracy | RPS | Brier | LogLoss | P&L (units) | # Bets | ROI % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MD 1 | 10 | 0.800 | 0.1567 | 0.3831 | 0.6798 | +13.90 | 8 | +173.75 |
| MD 2 | 10 | 0.900 | 0.2603 | 0.3513 | 0.6347 | +9.98 | 5 | +199.60 |
| MD 3 | 10 | 0.800 | 0.2004 | 0.2748 | 0.5180 | +8.37 | 9 | +93.00 |
| MD 4 | 10 | 0.700 | 0.2739 | 0.3504 | 0.6151 | +12.80 | 10 | +128.00 |
| MD 5 | 10 | 0.900 | 0.2227 | 0.2858 | 0.5372 | +13.38 | 9 | +148.67 |
| MD 6 | 10 | 1.000 | 0.1221 | 0.1832 | 0.3969 | +13.56 | 10 | +135.60 |
| MD 7 | 10 | 0.800 | 0.2133 | 0.3400 | 0.6064 | +9.45 | 9 | +105.00 |
| MD 8 | 10 | 0.800 | 0.1899 | 0.2954 | 0.5541 | +18.93 | 9 | +210.33 |
| MD 9 | 10 | 0.900 | 0.1109 | 0.2334 | 0.4701 | +15.74 | 9 | +174.89 |
| MD 10 | 10 | 0.800 | 0.2010 | 0.3320 | 0.6140 | +11.65 | 8 | +145.62 |
| MD 11 | 10 | 0.900 | 0.2038 | 0.2192 | 0.4421 | +10.98 | 10 | +109.80 |
| MD 12 | 10 | 0.900 | 0.1928 | 0.2418 | 0.4816 | +9.91 | 10 | +99.10 |
| MD 13 | 10 | 0.900 | 0.0825 | 0.1686 | 0.3643 | +13.02 | 9 | +144.67 |
| MD 14 | 10 | 0.600 | 0.3914 | 0.5528 | 0.9374 | +7.46 | 4 | +186.50 |
| AVERAGE | 140 | 0.836 | 0.2016 | 0.3008 | 0.5608 | +169.13 | 119 | +142.13 |

### Rolling Matchday 2025 (retrain after each matchday)

| Fold | Samples | Accuracy | RPS | Brier | LogLoss | P&L (units) | # Bets | ROI % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MD 1 | 10 | 0.600 | 0.2017 | 0.4658 | 0.7933 | +15.52 | 10 | +155.20 |
| MD 2 | 10 | 0.800 | 0.3297 | 0.4380 | 0.7690 | +5.60 | 10 | +56.00 |
| MD 3 | 10 | 0.800 | 0.2461 | 0.3561 | 0.6324 | +5.81 | 9 | +64.56 |
| MD 4 | 10 | 0.800 | 0.2837 | 0.3850 | 0.6709 | +11.91 | 9 | +132.33 |
| MD 5 | 10 | 0.800 | 0.2829 | 0.3672 | 0.6632 | +10.12 | 8 | +126.50 |
| MD 6 | 10 | 0.800 | 0.1968 | 0.3009 | 0.5510 | +6.77 | 9 | +75.22 |
| MD 7 | 10 | 0.400 | 0.4115 | 0.5979 | 0.9640 | +1.21 | 9 | +13.44 |
| MD 8 | 10 | 0.800 | 0.2790 | 0.4006 | 0.7089 | +17.15 | 9 | +190.56 |
| MD 9 | 10 | 0.800 | 0.2011 | 0.4014 | 0.7165 | +10.78 | 7 | +154.00 |
| MD 10 | 10 | 0.500 | 0.2962 | 0.5072 | 0.8677 | +3.63 | 9 | +40.33 |
| MD 11 | 10 | 0.800 | 0.3026 | 0.3439 | 0.6308 | +5.10 | 7 | +72.86 |
| MD 12 | 10 | 0.800 | 0.3114 | 0.3941 | 0.6996 | +4.63 | 8 | +57.88 |
| MD 13 | 10 | 0.900 | 0.1463 | 0.2863 | 0.5498 | +13.02 | 9 | +144.67 |
| MD 14 | 10 | 0.700 | 0.3642 | 0.5353 | 0.8916 | +10.25 | 4 | +256.25 |
| AVERAGE | 140 | 0.736 | 0.2752 | 0.4128 | 0.7221 | +121.50 | 117 | +103.85 |

### Sliding Season (window size = 4 seasons)

| Fold | Samples | Accuracy | RPS | Brier | LogLoss | P&L (units) | # Bets | ROI % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Season 2018-2019 | 380 | 0.555 | 0.3889 | 0.5599 | 0.9309 | +169.88 | 282 | +60.24 |
| Season 2019-2020 | 380 | 0.537 | 0.3737 | 0.5606 | 0.9267 | +139.36 | 294 | +47.40 |
| Season 2020-2021 | 380 | 0.545 | 0.3836 | 0.5516 | 0.9219 | +210.40 | 296 | +71.08 |
| Season 2021-2022 | 374 | 0.594 | 0.3586 | 0.5131 | 0.8581 | +234.96 | 304 | +77.29 |
| Season 2022-2023 | 140 | 0.650 | 0.3514 | 0.4843 | 0.8310 | +99.50 | 110 | +90.45 |
| Season 2023-2024 | 380 | 0.582 | 0.3710 | 0.5309 | 0.8867 | +209.67 | 310 | +67.64 |
| Season 2024-2025 | 379 | 0.594 | 0.3562 | 0.5100 | 0.8594 | +195.65 | 309 | +63.32 |
| Season 2025-2026 | 140 | 0.714 | 0.2844 | 0.4207 | 0.7368 | +108.88 | 112 | +97.21 |
| AVERAGE | 2553 | 0.596 | 0.3585 | 0.5164 | 0.8689 | +1368.30 | 2017 | +67.84 |

### How ROI and P&L are computed

- **P&L (units)**: cumulative profit/loss using a flat stake of 1 unit per bet
- **ROI %**: `P&L / # Bets * 100`
- **# Bets**: number of qualifying bets placed by the strategy during the fold

### Reproduce these runs

```bash
# Static Pre-Season (single model for entire season)
poetry run liga-predictor evaluate --mode static-preseason

# Rolling Matchday (retrain after each matchday)
poetry run liga-predictor evaluate --mode rolling-matchday

# Sliding Season (4-season training window)
poetry run liga-predictor evaluate --mode sliding-season --window-size 4

# Visualize with MLflow
mlflow ui
# then open http://localhost:5000
```

