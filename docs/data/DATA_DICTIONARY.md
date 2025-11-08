# 3. Liga Dataset - Data Dictionary

**Last Updated:** 2025-11-08

This document provides detailed descriptions of all tables, columns, and data points in the 3. Liga dataset.

---

## Table of Contents

1. [Core Entities](#core-entities)
2. [Match Data](#match-data)
3. [Derived Features](#derived-features)
4. [ML Features](#ml-features)
5. [Data Types & Formats](#data-types--formats)

---

## Core Entities

### `teams`

Master table for all teams that have participated in 3. Liga.

| Column | Type | Description | Completeness |
|--------|------|-------------|--------------|
| `team_id` | INTEGER | Primary key, auto-increment | 100% |
| `openligadb_id` | INTEGER | External ID from OpenLigaDB API | ~90% |
| `team_name` | VARCHAR(100) | Official team name | 100% |
| `team_name_short` | VARCHAR(50) | Abbreviated team name | ~80% |
| `team_name_alt` | VARCHAR(100) | Alternative names for matching | ~60% |
| `founded_year` | INTEGER | Year the club was founded | ~70% |
| `stadium_name` | VARCHAR(100) | Home stadium name | ~75% |
| `stadium_capacity` | INTEGER | Stadium capacity | ~70% |
| `city` | VARCHAR(100) | City location | ~85% |

**Total Records:** 70 teams

**Notes:**
- Includes all teams that participated since 2009-2010 season
- Some teams changed names or merged (e.g., RW Erfurt → FC Rot-Weiß Erfurt)
- Team mappings are handled in `config/team_mappings.json`

---

## Match Data

### `matches`

Core match results and metadata.

| Column | Type | Description | Completeness |
|--------|------|-------------|--------------|
| `match_id` | INTEGER | Primary key | 100% |
| `openligadb_match_id` | INTEGER | External match ID | ~95% |
| `season` | VARCHAR(9) | Season (e.g., "2024-2025") | 100% |
| `matchday` | INTEGER | Matchday number (1-38) | 100% |
| `match_datetime` | TIMESTAMP | Date and time of match | 100% |
| `home_team_id` | INTEGER | Home team foreign key | 100% |
| `away_team_id` | INTEGER | Away team foreign key | 100% |
| `home_goals` | INTEGER | Goals scored by home team | 95% |
| `away_goals` | INTEGER | Goals scored by away team | 95% |
| `result` | CHAR(1) | Match result: H/D/A | 95% |
| `is_finished` | BOOLEAN | Match completion status | 100% |
| `venue` | VARCHAR(100) | Stadium name | ~30% |
| `attendance` | INTEGER | Match attendance | 0% ⚠️ |
| `referee` | VARCHAR(100) | Referee name | ~25% |
| `temperature_celsius` | REAL | Temperature during match | 95% |
| `humidity_percent` | REAL | Humidity percentage | 95% |
| `wind_speed_kmh` | REAL | Wind speed | 95% |
| `precipitation_mm` | REAL | Precipitation amount | 95% |
| `weather_condition` | VARCHAR(50) | Weather description | 95% |
| `is_midweek` | BOOLEAN | Midweek match flag | 0% ⚠️ |
| `is_derby` | BOOLEAN | Local derby flag | 0% ⚠️ |

**Total Records:** 6,290 matches (2009-2026)
**Finished Matches:** 5,969 (94.9%)

**Notes:**
- Weather data added from external APIs
- Attendance data not yet collected (priority improvement)
- Derby detection not yet implemented

### `match_statistics`

Detailed in-match statistics (possession, shots, passes, etc.).

| Column | Type | Description | Completeness |
|--------|------|-------------|--------------|
| `stat_id` | INTEGER | Primary key | 100% |
| `match_id` | INTEGER | Foreign key to matches | 100% |
| `team_id` | INTEGER | Foreign key to teams | 100% |
| `is_home` | BOOLEAN | Home team indicator | 100% |
| `possession_percent` | REAL | Ball possession % | 100% |
| `shots_total` | INTEGER | Total shots | 0% ⚠️ |
| `shots_on_target` | INTEGER | Shots on target | 100% |
| `shots_off_target` | INTEGER | Shots off target | 0% |
| `shots_blocked` | INTEGER | Blocked shots | 0% |
| `big_chances` | INTEGER | Clear scoring chances | 0% |
| `big_chances_missed` | INTEGER | Missed clear chances | 0% |
| `passes_total` | INTEGER | Total passes | 0% ⚠️ |
| `passes_accurate` | INTEGER | Accurate passes | 0% |
| `pass_accuracy_percent` | REAL | Pass accuracy % | 0% ⚠️ |
| `key_passes` | INTEGER | Passes leading to shots | 0% |
| `crosses_total` | INTEGER | Total crosses | 0% |
| `crosses_accurate` | INTEGER | Accurate crosses | 0% |
| `long_balls_total` | INTEGER | Long passes | 0% |
| `long_balls_accurate` | INTEGER | Accurate long passes | 0% |
| `tackles_total` | INTEGER | Total tackles | 0% ⚠️ |
| `tackles_won` | INTEGER | Won tackles | 0% |
| `interceptions` | INTEGER | Interceptions made | 0% |
| `clearances` | INTEGER | Defensive clearances | 0% |
| `blocked_shots` | INTEGER | Shots blocked | 0% |
| `duels_total` | INTEGER | Total duels | 0% |
| `duels_won` | INTEGER | Won duels | 0% |
| `aerials_total` | INTEGER | Aerial duels | 0% |
| `aerials_won` | INTEGER | Won aerial duels | 0% |
| `fouls_committed` | INTEGER | Fouls committed | 0% ⚠️ |
| `fouls_won` | INTEGER | Fouls won | 0% |
| `yellow_cards` | INTEGER | Yellow cards | 98% |
| `red_cards` | INTEGER | Red cards | 98% |
| `corners` | INTEGER | Corner kicks | 100% |
| `offsides` | INTEGER | Offsides | 0% |
| `touches` | INTEGER | Total touches | 0% |
| `dribbles_attempted` | INTEGER | Dribble attempts | 0% |
| `dribbles_successful` | INTEGER | Successful dribbles | 0% |
| `source` | VARCHAR(50) | Data source | 100% |
| `has_complete_stats` | BOOLEAN | Full stats available | 100% |

**Total Records:** 4,446 (2,223 unique matches)
**Coverage:** 35% of all matches
**Primary Source:** FotMob

**Notes:**
- Statistics available primarily from 2014 onwards
- Many detailed fields still at 0% due to data collection limitations
- Possession, shots on target, corners, and cards have best coverage

### `match_events`

Goal, card, and substitution events.

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | INTEGER | Primary key |
| `match_id` | INTEGER | Foreign key to matches |
| `team_id` | INTEGER | Team involved |
| `event_type` | VARCHAR(20) | Type: goal/yellow_card/red_card/substitution |
| `minute` | INTEGER | Match minute |
| `minute_extra` | INTEGER | Extra/injury time |
| `player_id` | INTEGER | Player involved |
| `player_name` | VARCHAR(100) | Player name |
| `is_penalty` | BOOLEAN | Penalty goal flag |
| `is_own_goal` | BOOLEAN | Own goal flag |
| `assist_player_id` | INTEGER | Assisting player |
| `assist_player_name` | VARCHAR(100) | Assisting player name |
| `player_out_id` | INTEGER | Substituted player |
| `player_out_name` | VARCHAR(100) | Substituted player name |

**Total Records:** 0 ⚠️
**Status:** Table exists but not yet populated

---

## Derived Features

### `team_ratings`

Team strength ratings calculated before each match.

| Column | Type | Description | Completeness |
|--------|------|-------------|--------------|
| `rating_id` | INTEGER | Primary key | 100% |
| `team_id` | INTEGER | Foreign key to teams | 100% |
| `match_id` | INTEGER | Match after which rating updated | 100% |
| `season` | VARCHAR(9) | Season | 100% |
| `matchday` | INTEGER | Matchday number | 100% |
| `elo_rating` | REAL | Elo rating (1200-1800 range) | 100% |
| `pi_rating` | REAL | Pi-rating (0.0-1.0 range) | 100% |
| `points_last_5` | INTEGER | Points in last 5 matches | 100% |
| `points_last_10` | INTEGER | Points in last 10 matches | 100% |
| `goals_scored_last_5` | REAL | Avg goals scored (L5) | 100% |
| `goals_conceded_last_5` | REAL | Avg goals conceded (L5) | 100% |
| `current_win_streak` | INTEGER | Current win streak | 100% |
| `current_unbeaten_streak` | INTEGER | Current unbeaten streak | 100% |
| `current_loss_streak` | INTEGER | Current loss streak | 100% |

**Total Records:** 9,646
**Coverage:** 100% for finished matches
**Seasons:** 14 complete seasons

**Rating Descriptions:**

**Elo Rating:**
- Dynamic rating system that updates after each match
- Higher = stronger team
- Range: ~1300-1750 (mean: 1509)
- Standard deviation: ~61 points
- K-factor: 40 for 3. Liga

**Pi-Rating:**
- Weighted performance indicator
- Based on recent match results
- Range: 0.0 (worst) to 1.0 (best)
- Mean: 0.454
- Research shows high predictive value for tree-based models

### `league_standings`

Historical league table after each matchday.

| Column | Type | Description | Completeness |
|--------|------|-------------|--------------|
| `standing_id` | INTEGER | Primary key | 100% |
| `season` | VARCHAR(9) | Season | 100% |
| `matchday` | INTEGER | Matchday number | 100% |
| `team_id` | INTEGER | Team foreign key | 100% |
| `position` | INTEGER | League position (1-20) | 100% |
| `matches_played` | INTEGER | Matches played | 100% |
| `wins` | INTEGER | Wins | 100% |
| `draws` | INTEGER | Draws | 100% |
| `losses` | INTEGER | Losses | 100% |
| `goals_for` | INTEGER | Goals scored | 100% |
| `goals_against` | INTEGER | Goals conceded | 100% |
| `goal_difference` | INTEGER | Goal difference | 100% |
| `points` | INTEGER | Total points | 100% |
| `form_last_5` | VARCHAR(5) | Last 5 results (e.g., "WDLWW") | ~60% |

**Total Records:** 320
**Coverage:** Historical tables available

### `betting_odds`

Betting market odds for matches.

| Column | Type | Description | Completeness |
|--------|------|-------------|--------------|
| `odds_id` | INTEGER | Primary key | 100% |
| `match_id` | INTEGER | Foreign key to matches | 100% |
| `bookmaker` | VARCHAR(50) | Bookmaker name | 100% |
| `odds_home` | REAL | Home win odds | 99% |
| `odds_draw` | REAL | Draw odds | 99% |
| `odds_away` | REAL | Away win odds | 99% |
| `implied_prob_home` | REAL | Implied home win probability | 99% |
| `implied_prob_draw` | REAL | Implied draw probability | 99% |
| `implied_prob_away` | REAL | Implied away win probability | 99% |
| `odds_type` | VARCHAR(20) | Opening/closing | 100% |
| `collected_at` | TIMESTAMP | Collection timestamp | 100% |

**Total Records:** 4,319
**Coverage:** 68.7% of all matches
**Average Odds:** Home 2.31, Draw 3.48, Away 3.50

**Notes:**
- Primarily closing odds
- Average odds show home advantage (2.31 vs 3.50)
- Best coverage in recent seasons

### `head_to_head`

Historical head-to-head records between teams.

**Status:** Table exists but not yet populated ⚠️

---

## ML Features

### ML Dataset Feature Groups

The ML-ready datasets (`3liga_ml_dataset_*.csv`) contain 73 features grouped into:

#### 1. Rating Features (6 features) - **100% Coverage**
- `home_elo`: Home team Elo rating
- `away_elo`: Away team Elo rating
- `elo_diff`: Elo difference (home - away)
- `home_pi`: Home team Pi-rating
- `away_pi`: Away team Pi-rating
- `pi_diff`: Pi-rating difference

#### 2. Form Metrics (10 features) - **100% Coverage**
- `home_points_l5`: Home team points in last 5 matches
- `away_points_l5`: Away team points in last 5 matches
- `form_diff_l5`: Form difference (last 5)
- `home_points_l10`: Home team points in last 10 matches
- `away_points_l10`: Away team points in last 10 matches
- `form_diff_l10`: Form difference (last 10)
- `home_goals_scored_l5`: Home goals scored (last 5)
- `home_goals_conceded_l5`: Home goals conceded (last 5)
- `away_goals_scored_l5`: Away goals scored (last 5)
- `away_goals_conceded_l5`: Away goals conceded (last 5)

#### 3. Match Statistics (10 features) - **21.5% Average Coverage**
- `home_possession`: Home possession %
- `away_possession`: Away possession %
- `home_shots`: Home shots
- `away_shots`: Away shots
- `home_shots_on_target`: Home shots on target
- `away_shots_on_target`: Away shots on target
- `home_corners`: Home corners
- `away_corners`: Away corners
- `home_fouls`: Home fouls
- `away_fouls`: Away fouls

#### 4. Betting Odds (4 features) - **39.0% Coverage**
- `odds_home`: Home win odds
- `odds_draw`: Draw odds
- `odds_away`: Away win odds
- `odds_margin`: Bookmaker margin

#### 5. Context Features (43 features) - **51.0% Average Coverage**

**Temporal:**
- `matchday`: Matchday number
- `day_of_week`: Day of week (0=Monday)
- `month`: Month
- `year`: Year
- `is_weekend`: Weekend match flag

**Team Context:**
- `is_home`: Home team indicator
- `home_position`: League position
- `away_position`: League position
- `position_diff`: Position difference

**Match Context:**
- `temperature`: Temperature (°C)
- `humidity`: Humidity (%)
- `wind_speed`: Wind speed (km/h)
- `precipitation`: Precipitation (mm)

**Flags:**
- `has_detailed_stats`: Detailed stats available
- `has_odds`: Odds available
- `has_ratings`: Ratings available

### Target Variables

**Classification Targets:**
- `target_multiclass`: 0=Away, 1=Draw, 2=Home
- `target_home_win`: Binary (1 if home wins)
- `target_away_win`: Binary (1 if away wins)
- `target_draw`: Binary (1 if draw)

**Regression Targets:**
- `target_home_goals`: Home team goals
- `target_away_goals`: Away team goals
- `target_total_goals`: Total goals in match
- `target_goal_difference`: Goal difference (home - away)

---

## Data Types & Formats

### Date/Time Formats

- **Date:** YYYY-MM-DD
- **Timestamp:** YYYY-MM-DD HH:MM:SS (UTC)
- **Season:** "YYYY-YYYY" (e.g., "2024-2025")

### Encoding Conventions

**Match Result:**
- `H`: Home win
- `D`: Draw
- `A`: Away win

**Target Multiclass:**
- `0`: Away win
- `1`: Draw
- `2`: Home win

**Boolean Fields:**
- `0`: False/No
- `1`: True/Yes

### Missing Data Handling

| Symbol | Meaning |
|--------|---------|
| `NULL` | Data not available |
| `0.0` | Zero value (not missing) |
| `-1` | Not applicable |

### Value Ranges

| Feature | Typical Range | Unit |
|---------|---------------|------|
| Elo Rating | 1300-1750 | points |
| Pi-Rating | 0.0-1.0 | normalized |
| Possession | 0-100 | % |
| Odds | 1.01-50.0 | decimal |
| Temperature | -10 to 35 | °C |
| Humidity | 0-100 | % |
| Wind Speed | 0-50 | km/h |

---

## Data Quality Notes

### High Quality (>90% completeness)
✅ Match results (home/away goals, date, teams)
✅ Team ratings (Elo, Pi-ratings)
✅ Form metrics (last 5/10 matches)
✅ Weather data
✅ Basic match info

### Medium Quality (50-90% completeness)
⚠️ Match statistics (35% of matches)
⚠️ Betting odds (69% of matches)
⚠️ League positions

### Low Quality (<50% completeness)
❌ Attendance (0%)
❌ Player events (0%)
❌ Detailed passing/shooting stats
❌ Player-level data

---

## Updates & Versioning

**Current Version:** 1.0
**Last Database Update:** 2025-11-08
**Update Frequency:** Weekly during season

**Changelog:**
- 2025-11-08: Added comprehensive data dictionary
- 2025-11-07: Added ML dataset exports
- 2025-11-06: Integrated weather data
- 2025-11-05: Calculated team ratings

---

## References

For more information:
- See `DATA_EXPLORATION_REPORT.md` for detailed analysis
- See `DATABASE_SCHEMA.md` for technical schema details
- See `FEATURE_ENGINEERING.md` for ML feature descriptions
- See `README.md` for usage examples

---

*Generated by 3. Liga Dataset Project*
