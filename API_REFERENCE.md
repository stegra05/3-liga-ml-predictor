# API Reference

Developer documentation for the 3. Liga Football Dataset codebase.

## Table of Contents

1. [Core Modules](#core-modules)
   - [DatabaseManager](#databasemanager)
   - [TeamMapper](#teammapper)
2. [Data Collectors](#data-collectors)
   - [OpenLigaDBCollector](#openligadbcollector)
   - [FBrefCollector](#fbrefcollector)
   - [TransfermarktRefereeCollector](#transfermarktreferee collector)
3. [Data Processors](#data-processors)
   - [RatingCalculator](#ratingcalculator)
   - [MLDataExporter](#mldataexporter)
   - [WeatherFetcher](#weatherfetcher)
4. [Utilities](#utilities)
5. [Database Schema](#database-schema)

---

## Core Modules

### DatabaseManager

**Location**: `database/db_manager.py`

The central database interface for all operations. Handles SQLite connections, CRUD operations, and data validation.

#### Initialization

```python
from database.db_manager import DatabaseManager

# Create instance (auto-initializes schema)
db = DatabaseManager()

# Or specify custom database path
db = DatabaseManager(db_path="/path/to/custom.db")
```

#### Key Methods

##### Team Operations

```python
# Get or create a team
team_id = db.get_or_create_team(
    name="FC Bayern München",
    short_name="Bayern",
    icon_url="https://..."
)
# Returns: int (team_id)

# Get team by name
team = db.get_team_by_name("FC Bayern München")
# Returns: dict or None
# Example: {'id': 1, 'name': 'FC Bayern München', 'short_name': 'Bayern'}

# List all teams
teams = db.get_all_teams()
# Returns: list of dicts
```

##### Match Operations

```python
# Insert a match
match_id = db.insert_match(
    season="2023-2024",
    matchday=15,
    match_datetime="2024-01-15 15:30:00",
    home_team_id=1,
    away_team_id=2,
    home_goals=2,
    away_goals=1,
    is_finished=True,
    venue="Allianz Arena"
)
# Returns: int (match_id)

# Get match by ID
match = db.get_match_by_id(match_id)
# Returns: dict or None

# Get matches by season
matches = db.get_matches_by_season("2023-2024")
# Returns: list of dicts

# Check if match exists
exists = db.match_exists(
    season="2023-2024",
    matchday=15,
    home_team_id=1,
    away_team_id=2
)
# Returns: bool
```

##### Rating Operations

```python
# Insert team ratings
db.insert_team_rating(
    match_id=123,
    team_id=1,
    elo_rating=1650.5,
    pi_rating=1.523,
    is_home=True
)

# Get ratings for a match
ratings = db.get_ratings_by_match(match_id=123)
# Returns: list of dicts
```

##### Query Execution

```python
# Execute custom query
results = db.execute_query(
    "SELECT * FROM matches WHERE season = ?",
    params=("2023-2024",)
)
# Returns: list of tuples

# Execute with named parameters
results = db.execute_query(
    "SELECT * FROM teams WHERE name LIKE :pattern",
    params={'pattern': '%Bayern%'}
)
```

#### Context Manager

```python
# Use as context manager for auto-cleanup
with DatabaseManager() as db:
    teams = db.get_all_teams()
    # Connection automatically closed when done
```

---

### TeamMapper

**Location**: `scripts/utils/team_mapper.py`

Handles team name standardization across different data sources.

#### Initialization

```python
from scripts.utils.team_mapper import TeamMapper

# Load default mappings
mapper = TeamMapper()

# Or specify custom mapping file
mapper = TeamMapper(mapping_file="config/custom_mappings.json")
```

#### Methods

```python
# Map external name to standard name
standard_name = mapper.map_team_name(
    external_name="Munich 1860",
    source="fbref"
)
# Returns: "TSV 1860 München" or None if no mapping

# Get all mappings for a source
mappings = mapper.get_source_mappings("fbref")
# Returns: dict

# Add new mapping
mapper.add_mapping(
    source="fbref",
    external_name="Jahn R'burg",
    standard_name="Jahn Regensburg"
)
mapper.save_mappings()  # Persist to disk
```

#### Mapping File Format

```json
{
  "source_to_standard": {
    "fbref": {
      "Munich 1860": "TSV 1860 München",
      "Viktoria Köln": "Viktoria Köln"
    },
    "transfermarkt": {
      "Bayern Munich": "FC Bayern München"
    }
  }
}
```

---

## Data Collectors

### OpenLigaDBCollector

**Location**: `scripts/collectors/openligadb_collector.py`

Collects match results, scores, and basic data from the OpenLigaDB API.

#### Initialization

```python
from scripts.collectors.openligadb_collector import OpenLigaDBCollector

collector = OpenLigaDBCollector()
```

#### Key Methods

```python
# Collect all data for a season
results = collector.collect_season(season="2023-2024")
# Returns: dict with counts and status
# Example: {
#     'season': '2023-2024',
#     'matches_collected': 380,
#     'matches_updated': 15,
#     'teams_added': 2
# }

# Collect specific matchday
results = collector.collect_matchday(
    season="2023-2024",
    matchday=15
)

# Collect all available seasons
results = collector.collect_all_seasons()
# Returns: list of dicts (results per season)

# Update recent matches only
results = collector.update_recent_matches(days=7)
# Returns: dict with update counts
```

#### Configuration

```python
# API settings
collector.base_url = "https://api.openligadb.de"
collector.league_shortcut = "bl3"  # 3. Liga
collector.league_season = "2023"   # Season start year

# Rate limiting (built-in)
# Automatic 1-second delay between requests
```

#### Example Usage

```python
from scripts.collectors.openligadb_collector import OpenLigaDBCollector

# Initialize
collector = OpenLigaDBCollector()

# Collect current season
print("Collecting 2024-2025 season...")
results = collector.collect_season("2024-2025")

print(f"Collected {results['matches_collected']} matches")
print(f"Added {results['teams_added']} new teams")
```

---

### FBrefCollector

**Location**: `scripts/collectors/fbref_collector.py`

Scrapes league standings and team statistics from FBref.

#### Initialization

```python
from scripts.collectors.fbref_collector import FBrefCollector

# With Selenium (recommended for avoiding bot detection)
collector = FBrefCollector(use_selenium=True)

# Without Selenium (faster but may be blocked)
collector = FBrefCollector(use_selenium=False)
```

#### Key Methods

```python
# Collect all data for a season
results = collector.collect_season_data(season="2023-2024")
# Returns: dict with collection results
# Example: {
#     'season': '2023-2024',
#     'standings': {'teams_collected': 20, 'status': 'success'},
#     'team_stats': {'teams_collected': 20, 'status': 'success'}
# }

# Collect league standings only
standings_df = collector.collect_season_standings(season="2023-2024")
# Returns: pandas DataFrame

# Collect all available seasons
results = collector.collect_all_seasons()
# Returns: list of dicts
```

#### Configuration

```python
# Seasons available on FBref for 3. Liga
collector.AVAILABLE_SEASONS = [
    "2018-2019", "2019-2020", "2020-2021", "2021-2022",
    "2022-2023", "2023-2024", "2024-2025", "2025-2026"
]

# Rate limiting
collector.delay = 3.0  # Seconds between requests
```

#### Example Usage

```python
from scripts.collectors.fbref_collector import FBrefCollector

# Initialize with Selenium
collector = FBrefCollector(use_selenium=True)

# Collect all available seasons
print("Starting FBref collection...")
results = collector.collect_all_seasons()

for season_result in results:
    print(f"{season_result['season']}: "
          f"{season_result['standings']['teams_collected']} teams")
```

**See also**: [docs/data/FBREF_INTEGRATION.md](docs/data/FBREF_INTEGRATION.md)

---

### TransfermarktRefereeCollector

**Location**: `scripts/collectors/transfermarkt_referee_collector.py`

Collects referee and attendance data from Transfermarkt.

#### Initialization

```python
from scripts.collectors.transfermarkt_referee_collector import (
    TransfermarktRefereeCollector
)

collector = TransfermarktRefereeCollector()
```

#### Key Methods

```python
# Collect referee and attendance data for a season
results = collector.collect_season(season="2023-2024")
# Returns: dict with collection results

# Collect specific matchday
results = collector.collect_matchday(
    season="2023-2024",
    matchday=15
)
```

**Note**: This collector uses web scraping and may need updates if Transfermarkt changes their HTML structure.

---

## Data Processors

### RatingCalculator

**Location**: `scripts/processors/rating_calculator.py`

Calculates Elo and Pi-ratings for all teams based on match history.

#### Initialization

```python
from scripts.processors.rating_calculator import RatingCalculator

# Initialize with default settings
calculator = RatingCalculator()

# Or customize parameters
calculator = RatingCalculator(
    initial_elo=1500,
    k_factor=32,
    home_advantage=100
)
```

#### Key Methods

```python
# Calculate all ratings for all matches
results = calculator.calculate_all_ratings()
# Returns: dict with counts
# Example: {
#     'matches_processed': 6290,
#     'ratings_calculated': 12580,
#     'errors': 0
# }

# Calculate ratings for specific season
results = calculator.calculate_season_ratings(season="2023-2024")

# Recalculate from scratch (clears existing)
results = calculator.recalculate_all()
```

#### Configuration

```python
# Elo parameters
calculator.INITIAL_ELO = 1500        # Starting rating for new teams
calculator.K_FACTOR = 32              # K-factor for updates
calculator.HOME_ADVANTAGE = 100       # Home team advantage in Elo

# Pi-rating parameters
calculator.INITIAL_PI = 1.5           # Starting pi-rating
calculator.PI_ALPHA = 0.3             # Weight for recent matches
calculator.PI_BETA = 0.1              # Decay factor
```

#### Algorithm Details

**Elo Rating:**
```python
# Expected score
expected_home = 1 / (1 + 10 ** ((away_elo - home_elo - home_advantage) / 400))

# Actual score (1 = win, 0.5 = draw, 0 = loss)
actual = 1 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0)

# New rating
new_home_elo = home_elo + K * (actual - expected_home)
```

**Pi-Rating:**
Based on weighted recent performance:
```python
pi_rating = attack_strength / defense_strength
# Where attack/defense are calculated from recent matches
# with exponential decay for older matches
```

#### Example Usage

```python
from scripts.processors.rating_calculator import RatingCalculator

# Initialize
calc = RatingCalculator()

# Calculate all ratings
print("Calculating ratings...")
results = calc.calculate_all_ratings()

print(f"Processed {results['matches_processed']} matches")
print(f"Calculated {results['ratings_calculated']} ratings")
```

---

### MLDataExporter

**Location**: `scripts/processors/ml_data_exporter.py`

Exports ML-ready datasets from the database.

#### Initialization

```python
from scripts.processors.ml_data_exporter import MLDataExporter

# Initialize
exporter = MLDataExporter()

# Or customize output directory
exporter = MLDataExporter(output_dir="data/custom_exports/")
```

#### Key Methods

```python
# Export full dataset with automatic train/val/test splits
results = exporter.export_ml_datasets()
# Returns: dict with export info
# Example: {
#     'total_matches': 5970,
#     'train_size': 4298,
#     'val_size': 478,
#     'test_size': 1194,
#     'features': 103,
#     'output_files': [...]
# }

# Export specific season
df = exporter.export_season(season="2023-2024")
# Returns: pandas DataFrame

# Export with custom split ratios
results = exporter.export_ml_datasets(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

#### Configuration

```python
# Split configuration
exporter.TRAIN_RATIO = 0.72
exporter.VAL_RATIO = 0.08
exporter.TEST_RATIO = 0.20

# Feature configuration
exporter.INCLUDE_WEATHER = True
exporter.INCLUDE_ODDS = True
exporter.INCLUDE_H2H = True

# Output format
exporter.OUTPUT_FORMAT = 'csv'  # or 'parquet'
```

#### Output Files

```
data/processed/
├── 3liga_ml_dataset_full.csv      # Complete dataset
├── 3liga_ml_dataset_train.csv     # Training set (72%)
├── 3liga_ml_dataset_val.csv       # Validation set (8%)
├── 3liga_ml_dataset_test.csv      # Test set (20%)
├── feature_documentation.txt      # Feature descriptions
└── dataset_summary.txt            # Statistics
```

#### Example Usage

```python
from scripts.processors.ml_data_exporter import MLDataExporter

# Initialize
exporter = MLDataExporter()

# Export all datasets
print("Exporting ML datasets...")
results = exporter.export_ml_datasets()

print(f"Total matches: {results['total_matches']}")
print(f"Features: {results['features']}")
print(f"Files created: {len(results['output_files'])}")
```

---

### WeatherFetcher

**Location**: `scripts/processors/fetch_weather_multi.py`

Fetches historical weather data for matches.

#### Initialization

```python
from scripts.processors.fetch_weather_multi import WeatherFetcher

# Initialize with API keys (optional for Meteostat)
fetcher = WeatherFetcher(
    openweather_api_key="your_key_here"  # Optional
)
```

#### Key Methods

```python
# Fetch weather for all matches
results = fetcher.fetch_all_weather()
# Returns: dict with fetch results

# Fetch for specific season
results = fetcher.fetch_season_weather(season="2023-2024")

# Fetch for specific match
weather = fetcher.fetch_match_weather(
    match_id=123,
    latitude=48.2188,
    longitude=11.6244,
    match_datetime="2024-01-15 15:30:00"
)
# Returns: dict with weather data
```

#### Weather Data Structure

```python
{
    'temperature_celsius': 12.5,
    'humidity_percent': 65.0,
    'wind_speed_kmh': 15.2,
    'precipitation_mm': 0.0,
    'is_cold': False,
    'is_hot': False,
    'is_rainy': False,
    'is_windy': False
}
```

---

## Utilities

### Command-Line Interface

**All commands are accessed through the single entry point: `main.py`**

#### Run Predictions (Default)

```bash
# Predict next matchday (default command)
python main.py

# Predict specific season/matchday
python main.py predict --season 2025 --matchday 15

# Update data before predicting
python main.py predict --update-data
```

#### Run Data Collection

```bash
# Collect latest matches
python main.py collect-openligadb

# Collect FBref data
python main.py collect-fbref

# Collect referee data
python main.py collect-transfermarkt-referees

# Fetch weather data
python main.py fetch-weather-multi -- --limit 50
```

#### Run Data Processing

```bash
# Calculate ratings
python main.py rating-calculator

# Export ML datasets
python main.py export-ml-data

# Build head-to-head statistics
python main.py build-head-to-head

# Build team locations
python main.py build-team-locations
```

#### Database Management

```bash
# Initialize database schema
python main.py db-init
```

**Note**: All subcommands support `--help` for detailed usage information:
```bash
python main.py predict --help
python main.py collect-openligadb --help
```

### Helper Functions

#### Team Name Utilities

```python
from scripts.utils.team_mapper import normalize_team_name

# Normalize team name (remove prefixes, standardize)
name = normalize_team_name("1. FC Bayern München")
# Returns: "Bayern München"
```

#### Date Utilities

```python
from datetime import datetime

# Parse match datetime
dt = datetime.fromisoformat("2024-01-15T15:30:00")

# Format for database
db_datetime = dt.strftime("%Y-%m-%d %H:%M:%S")
```

---

## Database Schema

### Core Tables

#### `teams`

```sql
CREATE TABLE teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    short_name TEXT,
    icon_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### `matches`

```sql
CREATE TABLE matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season TEXT NOT NULL,
    matchday INTEGER NOT NULL,
    match_datetime TIMESTAMP NOT NULL,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    home_goals INTEGER,
    away_goals INTEGER,
    is_finished BOOLEAN DEFAULT 0,
    venue TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (home_team_id) REFERENCES teams(id),
    FOREIGN KEY (away_team_id) REFERENCES teams(id),
    UNIQUE(season, matchday, home_team_id, away_team_id)
);
```

#### `team_ratings`

```sql
CREATE TABLE team_ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    elo_rating REAL,
    pi_rating REAL,
    is_home BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (match_id) REFERENCES matches(id),
    FOREIGN KEY (team_id) REFERENCES teams(id),
    UNIQUE(match_id, team_id)
);
```

### Supporting Tables

#### `match_statistics`

```sql
CREATE TABLE match_statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL UNIQUE,
    home_possession INTEGER,
    away_possession INTEGER,
    home_shots INTEGER,
    away_shots INTEGER,
    home_shots_on_target INTEGER,
    away_shots_on_target INTEGER,
    -- ... many more fields
    FOREIGN KEY (match_id) REFERENCES matches(id)
);
```

#### `betting_odds`

```sql
CREATE TABLE betting_odds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    bookmaker TEXT NOT NULL,
    odds_home REAL,
    odds_draw REAL,
    odds_away REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (match_id) REFERENCES matches(id)
);
```

#### `weather_data`

```sql
CREATE TABLE weather_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL UNIQUE,
    temperature_celsius REAL,
    humidity_percent REAL,
    wind_speed_kmh REAL,
    precipitation_mm REAL,
    source TEXT,
    FOREIGN KEY (match_id) REFERENCES matches(id)
);
```

### Indexes

```sql
CREATE INDEX idx_matches_season ON matches(season);
CREATE INDEX idx_matches_datetime ON matches(match_datetime);
CREATE INDEX idx_matches_home_team ON matches(home_team_id);
CREATE INDEX idx_matches_away_team ON matches(away_team_id);
CREATE INDEX idx_ratings_match ON team_ratings(match_id);
CREATE INDEX idx_ratings_team ON team_ratings(team_id);
```

---

## Error Handling

### Common Exceptions

```python
from database.db_manager import DatabaseError

try:
    db = DatabaseManager()
    match_id = db.insert_match(...)
except DatabaseError as e:
    print(f"Database error: {e}")
except ValueError as e:
    print(f"Invalid data: {e}")
```

### Validation

```python
# All insert methods validate data before insertion
db.insert_match(
    season="invalid",  # Will raise ValueError
    ...
)
# ValueError: Invalid season format. Expected 'YYYY-YYYY'
```

---

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_db_manager.py

# Run with coverage
pytest tests/ --cov=database --cov=scripts
```

### Test Database

Tests use a separate test database to avoid corrupting production data:

```python
import pytest
from database.db_manager import DatabaseManager

@pytest.fixture
def test_db():
    """Create test database"""
    db = DatabaseManager(db_path="test_3liga.db")
    yield db
    db.close()
    os.remove("test_3liga.db")

def test_insert_team(test_db):
    team_id = test_db.get_or_create_team(name="Test Team")
    assert team_id is not None
```

---

## Performance Considerations

### Bulk Operations

```python
# Use transactions for bulk inserts
db.conn.execute("BEGIN TRANSACTION")
try:
    for match in matches:
        db.insert_match(**match)
    db.conn.commit()
except Exception as e:
    db.conn.rollback()
    raise
```

### Query Optimization

```python
# Use indexes
db.execute_query(
    "SELECT * FROM matches WHERE season = ? AND matchday = ?",
    params=("2023-2024", 15)
)
# Uses idx_matches_season

# Avoid N+1 queries
# Bad: Multiple queries
for match in matches:
    home_team = db.get_team_by_id(match['home_team_id'])
    away_team = db.get_team_by_id(match['away_team_id'])

# Good: Single JOIN query
results = db.execute_query("""
    SELECT m.*, ht.name as home_name, at.name as away_name
    FROM matches m
    JOIN teams ht ON m.home_team_id = ht.id
    JOIN teams at ON m.away_team_id = at.id
""")
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-01 | Initial codebase |
| 1.1 | 2025-11-08 | Added FBref integration |
| 1.2 | 2025-11-09 | Added weather data collection |

---

## See Also

- [GETTING_STARTED.md](GETTING_STARTED.md) - User guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contributing guidelines
- [docs/data/DATA_DICTIONARY.md](docs/data/DATA_DICTIONARY.md) - Feature reference

---

**Questions?** Open an issue on GitHub or check the [documentation](docs/).
