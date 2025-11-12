# Contributing to 3. Liga Football Dataset

Thank you for your interest in contributing! This project welcomes contributions from everyone, whether you're fixing a typo, adding a new data source, or implementing a new feature.

## Table of Contents

1. [Ways to Contribute](#ways-to-contribute)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Project Structure](#project-structure)
5. [Adding New Data Sources](#adding-new-data-sources)
6. [Improving Existing Features](#improving-existing-features)
7. [Testing Your Changes](#testing-your-changes)
8. [Submitting Your Contribution](#submitting-your-contribution)
9. [Code Style Guidelines](#code-style-guidelines)
10. [Getting Help](#getting-help)

---

## Ways to Contribute

### For Everyone

- 🐛 **Report bugs** - Found an error? Let us know!
- 📝 **Improve documentation** - Fix typos, add examples, clarify instructions
- 💡 **Suggest features** - Have an idea? Open an issue to discuss it
- ❓ **Ask questions** - Your questions help improve our docs

### For Developers

- 🔧 **Fix bugs** - Check our [issues](../../issues) for bugs to fix
- ✨ **Add features** - Implement new data sources, features, or improvements
- 🧪 **Write tests** - Help us maintain code quality
- 📊 **Add examples** - Create tutorials or example models

### For Data Scientists

- 📈 **Feature engineering** - Create new predictive features
- 🤖 **Model improvements** - Share better modeling approaches
- 📉 **Analysis** - Contribute insights about the data
- 🎯 **Benchmarks** - Establish baseline model performance

---

## Getting Started

### Before You Begin

1. **Check existing issues/PRs** - Someone might already be working on it
2. **Open an issue first** - For major changes, discuss before coding
3. **Read the docs** - Familiarize yourself with the project structure

### Quick Contribution Workflow

```bash
# 1. Fork the repository (click "Fork" on GitHub)

# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/catboost-predictor.git
cd catboost-predictor

# 3. Create a branch
git checkout -b feature/your-feature-name

# 4. Make your changes
# ... edit files ...

# 5. Test your changes
pytest tests/

# 6. Commit and push
git add .
git commit -m "Add: brief description of changes"
git push origin feature/your-feature-name

# 7. Open a Pull Request on GitHub
```

---

## Development Setup

### 1. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies
pip install pytest pytest-cov black flake8 mypy

# ML libraries (for testing)
pip install scikit-learn
```

### 2. Set Up the Database

```bash
# Initialize the database (creates 3liga.db)
python main.py db-init

# Optional: Import existing data
python main.py import-existing-data
```

### 3. Verify Setup

```bash
# Run tests to ensure everything works
pytest tests/ -v

# Check code style
black --check .
flake8 .
```

---

## Project Structure

Understanding the codebase:

```
catboost-predictor/
│
├── database/                    # Database layer
│   ├── db_manager.py           # Database operations (core module)
│   ├── schema.sql              # Database schema
│   └── 3liga.db                # SQLite database (created on init)
│
├── scripts/
│   ├── collectors/             # Data collection scripts
│   │   ├── openligadb_collector.py   # Match results from API
│   │   ├── fbref_collector.py        # FBref standings/stats
│   │   └── [your_collector].py       # Add new collectors here!
│   │
│   ├── processors/             # Data processing scripts
│   │   ├── rating_calculator.py      # Elo/Pi ratings
│   │   ├── ml_data_exporter.py       # Export to CSV
│   │   └── import_existing_data.py   # Batch imports
│   │
│   └── utils/                  # Utility modules
│       └── team_mapper.py      # Team name standardization
│
├── config/                     # Configuration files
│   ├── team_mappings.json      # Team name mappings
│   └── fbref_team_mapping.json # FBref-specific mappings
│
├── data/
│   ├── raw/                    # Original data files
│   └── processed/              # ML-ready exports
│       └── 3liga_ml_dataset_*.csv
│
├── docs/                       # Documentation
│   └── data/
│       ├── DATA_DICTIONARY.md  # Feature reference
│       ├── FBREF_INTEGRATION.md # FBref documentation
│       └── README.md
│
├── examples/                   # Example code
│   └── train_model_example.py
│
├── tests/                      # Test suite (add tests here!)
│   ├── test_collectors.py
│   ├── test_processors.py
│   └── test_db_manager.py
│
├── README.md                   # Main documentation
├── GETTING_STARTED.md         # User guide
├── CONTRIBUTING.md            # This file!
└── requirements.txt           # Dependencies
```

---

## Adding New Data Sources

Want to add a new data source (e.g., transfer market data, social media sentiment)?

### Step 1: Create a Collector

Create a new file in `scripts/collectors/`:

```python
# scripts/collectors/your_source_collector.py

import requests
from database.db_manager import DatabaseManager
from scripts.utils.team_mapper import TeamMapper

class YourSourceCollector:
    """
    Collector for [Your Data Source]

    Purpose: Collects [what data] from [source]
    Coverage: [date range or scope]
    Rate Limit: [requests per minute/hour]
    """

    def __init__(self):
        self.db = DatabaseManager()
        self.team_mapper = TeamMapper()
        self.base_url = "https://api.example.com"

    def collect_season_data(self, season: str):
        """
        Collect data for a specific season.

        Args:
            season: Season string (e.g., "2023-2024")

        Returns:
            dict: Collection results with counts and status
        """
        print(f"Collecting data for {season}...")

        # 1. Fetch data from source
        data = self._fetch_from_api(season)

        # 2. Standardize team names
        data = self._standardize_teams(data)

        # 3. Store in database
        self._save_to_database(data, season)

        return {"season": season, "items": len(data)}

    def _fetch_from_api(self, season):
        """Fetch data from external API"""
        # Your implementation here
        pass

    def _standardize_teams(self, data):
        """Map team names to database standard"""
        # Use TeamMapper to standardize names
        pass

    def _save_to_database(self, data, season):
        """Save to database"""
        # Use DatabaseManager to insert data
        pass
```

### Step 2: Update Database Schema

If you need new tables, create a migration:

```sql
-- database/migrations/add_your_feature.sql

CREATE TABLE IF NOT EXISTS your_new_table (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER,
    your_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (match_id) REFERENCES matches(id)
);

CREATE INDEX idx_your_table_match ON your_new_table(match_id);
```

Apply the migration:
```python
# In db_manager.py, add to _initialize_schema()
with open('database/migrations/add_your_feature.sql') as f:
    self.conn.executescript(f.read())
```

### Step 3: Add Team Mappings

If your source uses different team names, add mappings:

```json
// config/your_source_team_mapping.json
{
  "source_to_standard": {
    "Munich 1860": "TSV 1860 München",
    "Viktoria Koln": "Viktoria Köln"
  }
}
```

### Step 4: Write Tests

```python
# tests/test_your_collector.py

import pytest
from scripts.collectors.your_source_collector import YourSourceCollector

def test_collector_initialization():
    collector = YourSourceCollector()
    assert collector is not None

def test_collect_season_data():
    collector = YourSourceCollector()
    results = collector.collect_season_data("2023-2024")
    assert results['season'] == "2023-2024"
    assert results['items'] > 0

def test_team_name_standardization():
    collector = YourSourceCollector()
    # Test that your team mapping works
    assert collector._standardize_teams([...]) == expected_output
```

### Step 5: Document Your Source

Create documentation in `docs/data/`:

```markdown
# YOUR_SOURCE_INTEGRATION.md

## Overview
Description of your data source...

## Coverage
- Date range: 2018 onwards
- Data types: [list]

## Usage
```python
from scripts.collectors.your_source_collector import YourSourceCollector
collector = YourSourceCollector()
collector.collect_season_data("2023-2024")
```

## Data Quality
[Any known issues or limitations]
```

---

## Improving Existing Features

### Enhancing Feature Engineering

Want to add new calculated features? Modify `ml_data_exporter.py`:

```python
# scripts/processors/ml_data_exporter.py

def _calculate_additional_features(self, df):
    """Calculate additional features from base data"""

    # Your new feature
    df['your_new_feature'] = (
        df['some_column'] / (df['another_column'] + 1)
    )

    # Document it!
    """
    your_new_feature: [description]
        Type: float
        Range: [min, max]
        Coverage: [%]
        Use: [when to use this feature]
    """

    return df
```

### Improving Data Quality

Found a data quality issue? Here's how to fix it:

1. **Identify the issue** - Document what's wrong
2. **Trace the source** - Which collector/processor creates it?
3. **Fix at the source** - Modify the collector/processor
4. **Add validation** - Prevent future occurrences
5. **Document the fix** - Update relevant docs

Example fix:

```python
# In rating_calculator.py

def calculate_elo_rating(self, ...):
    # Before: Could crash on missing data
    new_rating = base_rating + k_factor * (result - expected)

    # After: Handle edge cases
    if pd.isna(base_rating):
        base_rating = 1500  # Default for new teams

    new_rating = base_rating + k_factor * (result - expected)

    # Validate result
    assert 1000 <= new_rating <= 2500, f"Invalid Elo: {new_rating}"

    return new_rating
```

---

## Testing Your Changes

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_collectors.py

# Run with coverage report
pytest tests/ --cov=scripts --cov-report=html

# Run verbose
pytest tests/ -v
```

### Writing Good Tests

```python
import pytest
from scripts.collectors.your_module import YourClass

class TestYourClass:
    """Test suite for YourClass"""

    def setup_method(self):
        """Run before each test"""
        self.instance = YourClass()

    def test_basic_functionality(self):
        """Test basic feature works"""
        result = self.instance.method()
        assert result is not None

    def test_edge_case_empty_input(self):
        """Test handling of empty input"""
        result = self.instance.method([])
        assert result == expected_for_empty

    def test_invalid_input_raises_error(self):
        """Test that invalid input raises appropriate error"""
        with pytest.raises(ValueError):
            self.instance.method(invalid_input)

    def teardown_method(self):
        """Cleanup after each test"""
        # Clean up any test data
        pass
```

### Manual Testing Checklist

Before submitting:

- [ ] Code runs without errors
- [ ] New features are documented
- [ ] Tests pass (`pytest tests/`)
- [ ] Code style follows guidelines (`black .` and `flake8 .`)
- [ ] No data corruption (check database integrity)
- [ ] Documentation is updated
- [ ] Example code works

---

## Submitting Your Contribution

### Pull Request Process

1. **Update your branch**
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-feature-branch
   git rebase main
   ```

2. **Clean your commits**
   ```bash
   # Squash multiple small commits if needed
   git rebase -i HEAD~3  # For last 3 commits
   ```

3. **Write a good commit message**
   ```
   Add: FBref player statistics collector

   - Implements collector for player season stats
   - Adds team name mappings for FBref
   - Includes tests and documentation
   - Closes #123
   ```

4. **Push and create PR**
   ```bash
   git push origin your-feature-branch
   # Then open Pull Request on GitHub
   ```

### PR Description Template

```markdown
## Description
[What does this PR do?]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement

## Testing
[How did you test this?]

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or documented if unavoidable)

## Related Issues
Closes #[issue number]
```

### Review Process

- Maintainers will review within 3-5 days
- Address feedback by pushing new commits
- Once approved, we'll merge your PR
- Your contribution will be credited in the changelog!

---

## Code Style Guidelines

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Good: Clear, descriptive names
def calculate_elo_rating(home_team_id, away_team_id, result):
    """
    Calculate updated Elo ratings after a match.

    Args:
        home_team_id: Database ID of home team
        away_team_id: Database ID of away team
        result: Match result (0=away win, 0.5=draw, 1=home win)

    Returns:
        tuple: (new_home_elo, new_away_elo)
    """
    pass

# Bad: Unclear names, no documentation
def calc(h, a, r):
    pass
```

### Use Black for Formatting

```bash
# Format all Python files
black .

# Check without modifying
black --check .
```

### Import Order

```python
# 1. Standard library
import os
import sys
from datetime import datetime

# 2. Third-party libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# 3. Local modules
from database.db_manager import DatabaseManager
from scripts.utils.team_mapper import TeamMapper
```

### Documentation

Every module, class, and public function needs a docstring:

```python
def process_match_data(match_data: dict, season: str) -> pd.DataFrame:
    """
    Process raw match data into standardized format.

    Takes raw match data from an external source and converts it
    into the standard format used by the database. Handles team
    name mapping and data validation.

    Args:
        match_data: Raw match data dictionary with keys:
            - 'home_team': Home team name (str)
            - 'away_team': Away team name (str)
            - 'score': Match score (str, format: "2-1")
            - 'date': Match date (str, ISO format)
        season: Season identifier (str, format: "2023-2024")

    Returns:
        DataFrame with columns: home_team_id, away_team_id,
        home_goals, away_goals, match_date

    Raises:
        ValueError: If match_data is missing required keys
        ValueError: If team names cannot be mapped to database

    Example:
        >>> data = {'home_team': 'Team A', 'away_team': 'Team B',
        ...         'score': '2-1', 'date': '2024-01-15'}
        >>> df = process_match_data(data, "2023-2024")
        >>> print(df.head())
    """
    pass
```

### SQL Style

```sql
-- Good: Readable, well-formatted
SELECT
    m.id,
    m.match_datetime,
    ht.name AS home_team,
    at.name AS away_team,
    m.home_goals,
    m.away_goals
FROM matches m
INNER JOIN teams ht ON m.home_team_id = ht.id
INNER JOIN teams at ON m.away_team_id = at.id
WHERE m.season = '2023-2024'
ORDER BY m.match_datetime DESC;

-- Bad: Hard to read
SELECT m.id,m.match_datetime,ht.name AS home_team,at.name AS away_team FROM matches m INNER JOIN teams ht ON m.home_team_id=ht.id WHERE m.season='2023-2024';
```

---

## Getting Help

### Questions?

- **GitHub Discussions**: For general questions and ideas
- **GitHub Issues**: For bugs and feature requests
- **Documentation**: Check `docs/` directory first

### Common Issues

**Issue**: "Import error when running collector"
```bash
# Make sure you're in the project root
cd /path/to/catboost-predictor

# Run as module
python -m scripts.collectors.your_collector
```

**Issue**: "Database locked" error
```python
# Close other connections to the database
# Or use write-ahead logging (WAL) mode
conn = sqlite3.connect('database/3liga.db')
conn.execute('PRAGMA journal_mode=WAL')
```

**Issue**: "Test fails locally but worked before"
```bash
# Reset test database
rm database/test_3liga.db
pytest tests/ -v
```

### Development Tips

1. **Use virtual environments** - Avoid dependency conflicts
2. **Commit often** - Small, focused commits are easier to review
3. **Write tests first** - TDD helps catch bugs early
4. **Ask questions** - Better to ask than to guess
5. **Start small** - Fix a typo, then work up to bigger changes

---

## Recognition

All contributors will be:
- Listed in our `CONTRIBUTORS.md` file
- Mentioned in release notes
- Credited in any academic publications using this data

---

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers warmly
- Provide constructive feedback
- Focus on the code, not the person
- Respect different viewpoints

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Personal attacks
- Publishing others' private information

### Enforcement

Violations should be reported to [project maintainer email]. All reports will be reviewed and investigated promptly.

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

**Thank you for contributing! 🎉**

Questions? Open an issue or start a discussion. We're here to help!
