# Gemini Project Context: catboost-predictor

## Project Overview

This project is a comprehensive data science pipeline for creating a machine learning dataset to predict football match outcomes in the German 3. Liga. It gathers data from multiple sources, processes it, engineers features, and exports a final dataset ready for use with ML models like CatBoost and LightGBM.

**Core Technologies:**

*   **Language:** Python
*   **Data Manipulation:** Pandas
*   **Database:** SQLite
*   **Machine Learning:** CatBoost, LightGBM (as per `README.md`)
*   **Web Scraping/Data Collection:** Requests, BeautifulSoup, Selenium
*   **Logging:** Loguru

**Architecture:**

The project is structured around a central SQLite database (`database/3liga.db`) and a series of Python scripts organized into three main categories:

1.  **`scripts/collectors`**: Scripts responsible for gathering raw data from various sources like OpenLigaDB, FotMob, and Transfermarkt.
2.  **`scripts/processors`**: Scripts that clean, transform, and enrich the data. This includes calculating team ratings (Elo, Pi-ratings), cleaning team names, and exporting the final ML-ready dataset.
3.  **`database`**: Contains the database schema, and a `db_manager.py` that provides a centralized, consistent interface for all database operations.

## Building and Running

### 1. Initial Setup

To set up the project environment and initialize the database:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Initialize the SQLite database and create the schema
python database/db_manager.py
```

### 2. Data Pipeline

The core workflow consists of running a series of scripts to collect, process, and export the data.

```bash
# 1. Collect the latest match data
python scripts/collectors/openligadb_collector.py

# 2. Calculate team ratings (Elo, Pi-ratings)
python scripts/processors/rating_calculator.py

# 3. Export the final ML dataset
python scripts/processors/ml_data_exporter.py
```

The final datasets are saved in the `data/processed/` directory.

### 3. Training an Example Model

The `README.md` provides an example of how to train a CatBoost model using the exported data. See the `examples/train_model_example.py` for a runnable example.

## Development Conventions

*   **Modular Structure:** The project is organized into distinct modules for data collection, processing, and database management. This separation of concerns makes the codebase easier to maintain and extend.
*   **Database Abstraction:** All database interactions are handled through the `DatabaseManager` class in `database/db_manager.py`. This ensures consistent connection handling and query execution.
*   **Type Hinting:** The code uses Python's type hints for improved readability and static analysis.
*   **Logging:** `loguru` is used for structured and informative logging throughout the application.
*   **Configuration:** Project-specific configurations, like team name mappings, are stored in the `config/` directory.
*   **Data Splits:** The `ml_data_exporter.py` script handles the temporal train-validation-test split, which is crucial for time-series data like sports matches.
