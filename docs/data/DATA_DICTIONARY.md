# 3. Liga ML Dataset - Data Dictionary

**Generated:** 2025-11-08
**Total Features:** 103
**Dataset Version:** Full (2009-2026)

---

## Table of Contents
1. [Identifiers & Metadata](#identifiers--metadata) (6 features)
2. [Team Identifiers](#team-identifiers) (4 features)
3. [Match Results](#match-results) (3 features)
4. [Rating Features](#rating-features) (7 features)
5. [Form Features](#form-features) (10 features)
6. [Betting Odds](#betting-odds) (7 features)
7. [Match Statistics](#match-statistics) (18 features)
8. [Advanced Statistics](#advanced-statistics) (8 features)
9. [Weather Data](#weather-data) (11 features)
10. [Contextual Features](#contextual-features) (10 features)
11. [Head-to-Head](#head-to-head) (10 features)
12. [Target Variables](#target-variables) (7 features)
13. [Data Quality Flags](#data-quality-flags) (4 features)

---

## 1. Identifiers & Metadata

| Feature | Type | Description | Usage |
|---------|------|-------------|-------|
| `match_id` | int | Unique identifier for each match in database | Index, merge key |
| `season` | str | Season in format "YYYY-YYYY" (e.g., "2023-2024") | Grouping, temporal splits |
| `matchday` | int | Matchday number within season (1-38) | Ordering, season progress |
| `match_datetime` | datetime | Date and time of match kickoff | Temporal features, sorting |
| `day_of_week` | int | Day of week (0=Monday, 6=Sunday) | Temporal pattern |
| `month` | int | Month of match (1-12) | Seasonal patterns |

**Coverage:** 100%
**Use in Modeling:** Identifiers only, except `day_of_week`, `month` as categorical features

---

## 2. Team Identifiers

| Feature | Type | Description | Usage |
|---------|------|-------------|-------|
| `home_team` | str | Name of home team | Categorical feature, team effects |
| `away_team` | str | Name of away team | Categorical feature, team effects |
| `home_team_id` | int | Database ID of home team | Joins, lookups |
| `away_team_id` | int | Database ID of away team | Joins, lookups |

**Coverage:** 100%
**Unique Teams:** 68 teams across 17 seasons
**Encoding:** Team names should be encoded numerically for Random Forest models

---

## 3. Match Results

| Feature | Type | Description | Range/Values | Coverage |
|---------|------|-------------|--------------|----------|
| `result` | str | Match outcome: H (Home win), D (Draw), A (Away win) | H, D, A | 100% |
| `home_goals` | int | Goals scored by home team | 0-9 | 100% |
| `away_goals` | int | Goals scored by away team | 0-8 | 100% |

**Note:** These are TARGET variables for prediction, not available before match.

---

## 4. Rating Features

### Elo Rating System
[Elo rating](https://en.wikipedia.org/wiki/Elo_rating_system) - Dynamic skill rating system originating from chess

| Feature | Type | Description | Mean | Range | Coverage |
|---------|------|-------------|------|-------|----------|
| `home_elo` | float | Home team's Elo rating before match | 1,500 | 1,318-1,674 | 100% |
| `away_elo` | float | Away team's Elo rating before match | 1,500 | 1,318-1,674 | 100% |
| `elo_diff` | float | home_elo - away_elo | 0 | -356 to +356 | 100% |

**Calculation:** Dynamic rating updated after each match based on result vs expectation
- **K-factor:** 32 (standard for football)
- **Starting rating:** 1,500 for new teams
- **Home advantage:** Included in calculation

### Pi Rating System
Research-backed rating system optimized for football prediction

| Feature | Type | Description | Mean | Range | Coverage |
|---------|------|-------------|------|-------|----------|
| `home_pi` | float | Home team's Pi rating before match | 1.487 | 1.124-1.892 | 100% |
| `away_pi` | float | Away team's Pi rating before match | 1.487 | 1.124-1.892 | 100% |
| `pi_diff` | float | home_pi - away_pi | 0 | -0.768 to +0.768 | 100% |

**Source:** Based on academic research (Pi-rating paper)
**Advantage:** Specifically calibrated for football match prediction

**Correlation:** Elo vs Pi = 0.847 (very strong agreement)

---

## 5. Form Features

Form metrics based on recent match performance. **L5** = last 5 matches, **L10** = last 10 matches.

### Points Form
| Feature | Type | Description | Range | Coverage |
|---------|------|-------------|-------|----------|
| `home_points_l5` | int | Home team points from last 5 matches | 0-15 | 100% |
| `away_points_l5` | int | Away team points from last 5 matches | 0-15 | 100% |
| `form_diff_l5` | int | home_points_l5 - away_points_l5 | -15 to +15 | 100% |
| `home_points_l10` | int | Home team points from last 10 matches | 0-30 | 100% |
| `away_points_l10` | int | Away team points from last 10 matches | 0-30 | 100% |

**Interpretation:**
- 15 points (L5) = 5 wins = Excellent form
- 0 points (L5) = 5 losses = Poor form
- ~7-8 points (L5) = Average form

### Goals Form
| Feature | Type | Description | Mean | Coverage |
|---------|------|-------------|------|----------|
| `home_goals_scored_l5` | float | Home team's goals scored in last 5 | 1.68 | 100% |
| `home_goals_conceded_l5` | float | Home team's goals conceded in last 5 | 1.32 | 100% |
| `away_goals_scored_l5` | float | Away team's goals scored in last 5 | 1.32 | 100% |
| `away_goals_conceded_l5` | float | Away team's goals conceded in last 5 | 1.68 | 100% |
| `goal_diff_l5` | float | (home_scored - home_conceded) - (away_scored - away_conceded) | 0 | 100% |

**Use:** Captures offensive and defensive strength in recent matches

---

## 6. Betting Odds

Market odds from OddsPortal (average across bookmakers).

### Raw Odds
| Feature | Type | Description | Mean | Range | Coverage |
|---------|------|-------------|------|-------|----------|
| `odds_home` | float | Decimal odds for home win | 2.18 | 1.10-15.00 | 98.6% |
| `odds_draw` | float | Decimal odds for draw | 3.42 | 2.50-5.50 | 98.6% |
| `odds_away` | float | Decimal odds for away win | 3.65 | 1.15-20.00 | 98.6% |

**Interpretation:**
- Odds of 2.00 = 50% implied probability
- Odds of 1.50 = 66.7% implied probability
- Lower odds = more likely outcome (favorite)

### Implied Probabilities
| Feature | Type | Description | Mean | Coverage |
|---------|------|-------------|------|----------|
| `implied_prob_home` | float | 100 / odds_home | 45.9% | 98.6% |
| `implied_prob_draw` | float | 100 / odds_draw | 29.2% | 98.6% |
| `implied_prob_away` | float | 100 / odds_away | 27.4% | 98.6% |

**Note:** Sum > 100% due to bookmaker overround (~7%)

### Flags
| Feature | Type | Description | Coverage |
|---------|------|-------------|----------|
| `has_odds` | bool | Whether betting odds available for match | 100% |

---

## 7. Match Statistics

**Coverage:** 37.6% overall (varies by season: 0% pre-2014, ~40-80% from 2014+)
**Source:** FotMob scraped data
**Note:** POST-MATCH DATA - Not available before kickoff, for analysis only

### Possession & Territory
| Feature | Type | Description | Mean (when available) | Unit |
|---------|------|-------------|----------------------|------|
| `home_possession` | float | Home team possession percentage | 51.2% | % |
| `away_possession` | float | Away team possession percentage | 48.8% | % |

### Shooting
| Feature | Type | Description | Home Mean | Away Mean |
|---------|------|-------------|-----------|-----------|
| `home_shots` | int | Total shots by home team | 13.8 | - |
| `away_shots` | int | Total shots by away team | - | 11.4 |
| `home_shots_on_target` | int | Shots on target by home | 5.2 | - |
| `away_shots_on_target` | int | Shots on target by away | - | 4.3 |

### Passing
| Feature | Type | Description | Home Mean | Away Mean |
|---------|------|-------------|-----------|-----------|
| `home_passes` | int | Total passes attempted | 342 | - |
| `away_passes` | int | Total passes attempted | - | 318 |
| `home_pass_accuracy` | float | Pass completion rate | 78.4% | - |
| `away_pass_accuracy` | float | Pass completion rate | - | 78.0% |

### Defense & Physicality
| Feature | Type | Description | Home Mean | Away Mean |
|---------|------|-------------|-----------|-----------|
| `home_tackles` | int | Tackles made | - | - |
| `away_tackles` | int | Tackles made | - | - |
| `home_fouls` | int | Fouls committed | 13.2 | - |
| `away_fouls` | int | Fouls committed | - | 14.1 |

### Set Pieces & Discipline
| Feature | Type | Description | Home Mean | Away Mean |
|---------|------|-------------|-----------|-----------|
| `home_corners` | int | Corner kicks won | - | - |
| `away_corners` | int | Corner kicks won | - | - |
| `home_yellow_cards` | int | Yellow cards received | 2.1 | - |
| `away_yellow_cards` | int | Yellow cards received | - | 2.3 |
| `home_red_cards` | int | Red cards received | 0.09 | - |
| `away_red_cards` | int | Red cards received | - | 0.11 |

**WARNING:** Do NOT use these for prediction - only available after match ends!

---

## 8. Advanced Statistics

**Coverage:** 36.5-36.6% (subset of match statistics)

| Feature | Type | Description | Coverage |
|---------|------|-------------|----------|
| `home_big_chances` | int | Clear goalscoring opportunities created | 36.6% |
| `away_big_chances` | int | Clear goalscoring opportunities created | 36.6% |
| `home_interceptions` | int | Ball interceptions | 36.6% |
| `away_interceptions` | int | Ball interceptions | 36.6% |

**Note:** POST-MATCH DATA - For analysis only

---

## 9. Weather Data

**Coverage:** 81.9% overall (recently added November 2025)
**Source:** Historical weather APIs (Meteostat, OpenWeatherMap)
**Location:** Stadium coordinates from team_locations table
**Note:** Weather data provides environmental context for matches

### Raw Meteorological Data
| Feature | Type | Description | Mean | Range | Coverage |
|---------|------|-------------|------|-------|----------|
| `temperature_celsius` | float | Temperature at kickoff | 12.5°C | -11.6 to 33.9 | 81.9% |
| `humidity_percent` | float | Relative humidity | 66.8% | 20.5 to 100 | 81.9% |
| `wind_speed_kmh` | float | Wind speed | 13.2 km/h | 0 to 46.8 | 81.5% |
| `precipitation_mm` | float | Rainfall amount | 0.09 mm | 0 to 4.5 | 82.1% |

### Weather Categories
| Feature | Type | Description | Threshold | Coverage |
|---------|------|-------------|-----------|----------|
| `is_cold` | bool | Cold temperature | < 5°C | 100% |
| `is_hot` | bool | Hot temperature | > 20°C | 100% |
| `is_rainy` | bool | Raining | > 0.1mm | 100% |
| `is_heavy_rain` | bool | Heavy rain | > 2.0mm | 100% |
| `is_windy` | bool | Windy conditions | > 25 km/h | 100% |
| `is_extreme_weather` | bool | Any extreme condition | Combined | 100% |

**Note:** Category flags are 100% because they're computed from measurements (0=False when missing)

### Weather Impact on Performance
Based on literature and preliminary analysis:
- **Cold (<5°C):** Minimal impact on results
- **Hot (>20°C):** Slightly fewer goals scored
- **Rainy:** Minimal impact (modern pitches drain well)
- **Windy (>25km/h):** More unpredictable play, slightly more draws

---

## 10. Contextual Features

### Venue & Location
| Feature | Type | Description | Coverage | Notes |
|---------|------|-------------|----------|-------|
| `venue` | str | Stadium name | 16.8% | Low coverage, not critical |
| `is_home` | bool | Always 1 for home team perspective | 100% | Constant in dataset |
| `travel_distance_km` | float | Distance away team traveled | 67.1% | Calculated from team locations |

**Travel Distance Calculation:**
- Source: team_locations table with stadium coordinates
- Method: Haversine formula (great circle distance)
- Missing: Teams without location data or historical teams

### Rest Days
| Feature | Type | Description | Mean | Coverage |
|---------|------|-------------|------|----------|
| `rest_days_home` | int | Days since home team's last match | 6.8 | 99.5% |
| `rest_days_away` | int | Days since away team's last match | 6.9 | 99.4% |
| `rest_days_diff` | int | home_rest_days - away_rest_days | 0.1 | 99.1% |

**Typical Values:**
- 3-4 days: Midweek then weekend
- 7 days: Normal weekly schedule
- 14+ days: International break or bye week

### Rest Advantage Flags
| Feature | Type | Description | Coverage |
|---------|------|-------------|----------|
| `home_rest_advantage` | bool | Home team has 2+ days more rest | 100% |
| `away_rest_advantage` | bool | Away team has 2+ days more rest | 100% |
| `home_rest_category` | str | "short"/"normal"/"long" | 99.5% |
| `away_rest_category` | str | "short"/"normal"/"long" | 99.4% |

### Temporal Context
| Feature | Type | Description | Values | Coverage |
|---------|------|-------------|--------|----------|
| `year` | int | Year of match | 2009-2026 | 100% |
| `is_midweek` | bool | Match on Tuesday/Wednesday/Thursday | 5.5% True | 100% |

---

## 11. Head-to-Head Features

**Coverage:** 100%
**Lookback:** All historical matches between teams in database

| Feature | Type | Description | Notes |
|---------|------|-------------|-------|
| `h2h_total_matches` | int | Total previous meetings | Includes all history |
| `h2h_team_a_wins` | int | Wins for alphabetically first team | Neutral ordering |
| `h2h_draws` | int | Drawn matches | - |
| `h2h_team_b_wins` | int | Wins for alphabetically second team | Neutral ordering |
| `h2h_team_a_id` | int | Database ID of team A | For reference |
| `h2h_home_wins` | int | Home wins in this fixture | Home/away specific |
| `h2h_away_wins` | int | Away wins in this fixture | Home/away specific |
| `h2h_home_win_rate` | float | % of home wins | 0.0 to 1.0 |
| `h2h_draw_rate` | float | % of draws | 0.0 to 1.0 |
| `h2h_match_count` | int | Same as h2h_total_matches | Duplicate for convenience |

**Interpretation:**
- New matchups: h2h_total_matches = 0, rates = 0.0
- Derby matches: Often 20+ historical meetings
- Promoted teams: Few or no head-to-head history

**Use:** Captures fixture-specific dynamics and rivalries

---

## 12. Target Variables

The prediction targets - what we're trying to predict.

### Classification Targets
| Feature | Type | Description | Distribution |
|---------|------|-------------|--------------|
| `target_home_win` | bool | Did home team win? | 43.4% |
| `target_draw` | bool | Was it a draw? | 27.5% |
| `target_away_win` | bool | Did away team win? | 29.1% |
| `target_multiclass` | int | 0=Away win, 1=Draw, 2=Home win | Class-coded |

**Use:** For classification models predicting match outcome

### Regression Targets
| Feature | Type | Description | Mean | Range |
|---------|------|-------------|------|-------|
| `target_home_goals` | int | Goals scored by home team | 1.68 | 0-9 |
| `target_away_goals` | int | Goals scored by away team | 1.32 | 0-8 |
| `target_total_goals` | int | Total goals in match | 3.00 | 0-12 |

**Use:** For regression models predicting goal counts

---

## 13. Data Quality Flags

Internal flags indicating data availability.

| Feature | Type | Description | Coverage |
|---------|------|-------------|----------|
| `has_detailed_stats` | bool | Match statistics available | 37.6% |
| `has_odds` | bool | Betting odds available | 98.6% |
| `has_ratings` | bool | Elo/Pi ratings available | 100% |
| `has_weather` | bool | Weather data available | 81.9% |

**Use:** For filtering, data validation, or feature availability indicators

---

## Feature Categorization for ML

### For Prediction (40 core features)
**Use ONLY these features for training predictive models:**

#### Categorical (6)
- home_team, away_team, venue, day_of_week, month, is_midweek

#### Numerical Ratings (6)
- home_elo, away_elo, elo_diff, home_pi, away_pi, pi_diff

#### Numerical Form (10)
- home_points_l5, away_points_l5, form_diff_l5
- home_points_l10, away_points_l10
- home_goals_scored_l5, home_goals_conceded_l5
- away_goals_scored_l5, away_goals_conceded_l5, goal_diff_l5

#### Numerical Odds (6)
- odds_home, odds_draw, odds_away
- implied_prob_home, implied_prob_draw, implied_prob_away

#### Numerical Context (8)
- rest_days_home, rest_days_away, rest_days_diff
- travel_distance_km
- temperature_celsius, humidity_percent, wind_speed_kmh, precipitation_mm

#### Numerical H2H (4)
- h2h_total_matches, h2h_home_win_rate, h2h_draw_rate, h2h_match_count

### For Analysis Only (POST-MATCH)
**Do NOT use for prediction - only available after match:**
- All match_statistics features (possession, shots, passes, etc.)
- All advanced_statistics features
- Match results (result, home_goals, away_goals)

### For Indexing/Grouping
**Identifiers, not features:**
- match_id, season, matchday, match_datetime, home_team_id, away_team_id

---

## Missing Data Patterns

| Feature Category | Typical Coverage | Missing Pattern |
|------------------|------------------|-----------------|
| **Core features** | 100% | None |
| **Ratings** | 100% | None |
| **Form** | 100% | None (0 for first matches) |
| **Odds** | 98.6% | Missing for some early/obscure matches |
| **Match stats** | 37.6% | Pre-2014: 0%, Post-2014: 40-80% |
| **Weather** | 81.9% | Random gaps in API backfill |
| **Travel distance** | 67.1% | Missing for teams without location data |
| **Venue** | 16.8% | Not systematically collected |
| **Attendance** | 2.4% | Rarely available |

---

## Data Quality Notes

### Reliable Features (100% coverage, high quality)
✅ Team identifiers, match results, dates
✅ Elo and Pi ratings (calculated for all matches)
✅ Form metrics (calculated for all matches)
✅ Head-to-head statistics (comprehensive)

### Good Coverage (95%+, occasional gaps)
⚠️ Betting odds (98.6% - missing for some matches)
⚠️ Rest days (99%+ - occasional edge cases)

### Variable Coverage (use with caution)
⚠️ Match statistics (37.6% - systematic gaps)
⚠️ Weather data (82% - recently added, some gaps)
⚠️ Travel distance (67% - depends on location data)

### Poor Coverage (not reliable)
❌ Venue names (16.8% - not systematically collected)
❌ Attendance (2.4% - rarely available)
❌ Player statistics (0% - not collected)

---

## Feature Engineering Recommendations

### High-Value Additions
1. **Rolling averages** of form over different windows (3, 7, 15 matches)
2. **Interaction features** between ratings and form
3. **Recent h2h results** (weighted by recency)
4. **Position in table** (though form captures this)
5. **Home/away split form** (separate home and away form)

### Transformations to Consider
1. **Log-transform** extreme odds values
2. **Normalize** rating differences by league std dev
3. **Polynomial features** for rating differences (capture non-linearity)
4. **One-hot encode** day_of_week, month
5. **Bin** continuous features (temperature, rest days)

### Avoid
- Leakage from post-match statistics
- Future information (results after match date)
- Over-complicated interactions (Random Forest can capture these through tree splits)

---

## Data Sources

This dataset integrates data from multiple sources to provide comprehensive match prediction features:

### Primary Sources

| Source | Data Types | Coverage | Integration Status |
|--------|-----------|----------|-------------------|
| **OpenLigaDB** | Match results, scores, dates | 100% (2009-2026) | ✅ Complete |
| **OddsPortal** | Betting odds (home/draw/away) | 98.6% | ✅ Complete |
| **FotMob** | Match statistics (shots, possession, etc.) | 37.6% (2014+) | ✅ Complete |
| **Meteostat/OpenWeather** | Weather data | 81.9% | ✅ Complete |
| **FBref** | League standings, player stats | 100% (2018+) | 🔄 In Progress |

### Calculated Features

| Feature Type | Source | Notes |
|-------------|---------|-------|
| Elo Ratings | Calculated | Dynamic rating system (K=32) |
| Pi Ratings | Calculated | Research-backed football rating |
| Form Metrics | Calculated | Rolling windows (L5, L10) |
| Head-to-Head | Calculated | Historical matchup statistics |
| Rest Days | Calculated | From match schedules |
| Travel Distance | Calculated | Haversine formula from coordinates |

### FBref Integration (New - 2025-11-09)

**Coverage:** 2018-2019 season onwards (8 seasons)

**Available Data:**
- ✅ **League Standings:** Final season tables with position, points, W/D/L, GF/GA
- ✅ **Player Season Stats:** Individual player statistics (goals, assists, minutes, etc.)
- ✅ **Team Season Stats:** Aggregated team performance metrics

**Limitations for 3. Liga:**
- ❌ No advanced metrics (xG, xA, progressive passes) - requires Opta data
- ❌ No match-by-match player stats - only season totals
- ⚠️ Coverage starts from 2018-2019 (cannot backfill 2009-2017)

**Integration Status:**
- Data collection: In progress (background process)
- Database storage: Ready (tables created)
- ML export: Not yet integrated (planned for Phase 8)

**Potential Features:**
- Final season position/points (league finish)
- Historical performance patterns
- Squad depth metrics

**See:** [FBREF_INTEGRATION.md](FBREF_INTEGRATION.md) for technical details

---

## Related Documentation

- [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) - Dataset quality overview
- [DATA_EXPLORATION_REPORT.md](DATA_EXPLORATION_REPORT.md) - Statistical analysis
- [COMPLETENESS_ANALYSIS.md](COMPLETENESS_ANALYSIS.md) - Coverage by season
- [RECOMMENDATIONS.md](RECOMMENDATIONS.md) - Improvement roadmap
- [FBREF_INTEGRATION.md](FBREF_INTEGRATION.md) - FBref data source details

---

*Generated from 3liga_ml_dataset_full.csv on 2025-11-08*
*Updated with FBref integration on 2025-11-09*
*Total Features: 103 | Predictive Features: 40 | Analysis Features: 18 | Targets: 7*
