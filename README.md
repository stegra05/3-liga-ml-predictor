# 3. Liga Match Predictor

## Backtested Performance (eye-catchers)

- **Realistic Performance**: After fixing data leakage, the model achieves **~48-50% accuracy** and **~-2% ROI**.
- **Baseline Comparison**: Performs comparably to the "Always Predict Favorite" baseline (47% accuracy).
- **Valid Backtesting**: The previous +142% ROI was due to training on post-match statistics. The current pipeline enforces strict temporal separation.

| Mode | Horizon | Accuracy | ROI % |
|---|---|---:|---:|
| Expanding Season | 2015–2026 | 0.437 | -2.20 |
| Sliding Season | 2018–2026 | 0.441 | +4.72 |
| Baseline (Favorite) | - | 0.471 | - |

*Note: These results reflect a basic Random Forest model without extensive hyperparameter tuning. The primary goal of this project is to provide a robust, leak-free framework for further experimentation.*

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

| Fold | Samples | Accuracy | ROI % |
|---|---|---:|---:|
| Average (2015-2026) | 3600+ | 0.437 | -2.20 |

### Sliding Season (4-season window)

| Fold | Samples | Accuracy | ROI % |
|---|---|---:|---:|
| Average (2018-2026) | 2500+ | 0.441 | +4.72 |

### Rolling Matchday (Season 2025)

| Fold | Samples | Accuracy | ROI % |
|---|---|---:|---:|
| Average | 140 | 0.457 | -10.82 |

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

