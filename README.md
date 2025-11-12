# 3. Liga Match Predictor

A comprehensive machine learning system for predicting German 3. Liga football match results. The system aggregates historical data from multiple sources, engineers predictive features (ratings, form, odds, weather, head-to-head), and uses a Random Forest Classifier model to generate match predictions with detailed explanations.

## Features

- **Live Weather Integration**: Fetches real-time weather forecasts via Open-Meteo API, with fallback to historical estimates
- **Comprehensive Feature Engineering**: Includes Elo ratings, Pi-ratings, recent form, head-to-head statistics, betting odds, and contextual factors
- **SQLite Database**: Robust database schema storing match data, statistics, ratings, and more
- **Modern CLI**: Typer-based command-line interface (`liga-predictor`) with intuitive commands
- **ML-Ready Datasets**: Export feature-engineered datasets to CSV for your own modeling experiments

## Requirements

- **Python**: 3.9 to 3.14
- **Poetry**: For dependency management
- **SQLite 3**: Database engine (usually pre-installed)
- **Optional**: Chrome + chromedriver for Selenium-based data collectors (FBref, OddsPortal, Transfermarkt)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd catboost-predictor

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
- **Config**: `src/liga_predictor/config/` (configuration files)

## CLI Reference

Get help for any command:

```bash
poetry run liga-predictor --help
poetry run liga-predictor predict --help
```

### Common Commands

- `predict` - Run match predictions (default command when run without arguments)
- `db-init` - Initialize database schema
- `collect-openligadb` - Collect matches and fixtures from OpenLigaDB API
- `collect-fbref` - Collect team standings and player stats from FBref
- `collect-oddsportal` - Collect 1X2 betting odds from OddsPortal
- `collect-transfermarkt` - Collect referee data from Transfermarkt
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

