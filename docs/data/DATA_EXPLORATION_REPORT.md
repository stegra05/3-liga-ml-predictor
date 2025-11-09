# 3. Liga Dataset - Data Exploration Report

**Generated:** 2025-11-08
**Analysis Period:** 2009-2026 (17 seasons)
**Total Matches:** 6,290 (database) / 5,970 (ML dataset)

---

## Table of Contents
1. [Match Results Distribution](#match-results-distribution)
2. [Goals Analysis](#goals-analysis)
3. [Team Ratings Analysis](#team-ratings-analysis)
4. [Betting Odds Analysis](#betting-odds-analysis)
5. [Match Statistics](#match-statistics)
6. [Weather Data](#weather-data)
7. [Temporal Patterns](#temporal-patterns)
8. [Home Advantage](#home-advantage)

---

## 1. Match Results Distribution

### Overall Results (5,970 matches)

| Result | Count | Percentage | Visualization |
|--------|-------|------------|---------------|
| **Home Win (H)** | 2,588 | 43.4% | ████████████████████ |
| **Draw (D)** | 1,643 | 27.5% | █████████████ |
| **Away Win (A)** | 1,739 | 29.1% | ██████████████ |

![Target Distributions](figures/target_distributions.png)

### Key Findings:
- **Strong home advantage:** Home wins (43.4%) significantly exceed away wins (29.1%)
- **Home advantage margin:** +14.3 percentage points
- **Draw frequency:** 27.5% is typical for German football leagues
- **Balanced dataset:** No severe class imbalance (closest split would be 33/33/33)

### Comparison to Other Leagues:
| League | Home Win % | Draw % | Away Win % |
|--------|------------|--------|------------|
| **3. Liga** | **43.4%** | **27.5%** | **29.1%** |
| Bundesliga | 44-46% | 25-27% | 27-29% |
| 2. Bundesliga | 42-44% | 26-28% | 28-30% |

**Insight:** 3. Liga results distribution is consistent with professional German football patterns.

---

## 2. Goals Analysis

### Goals Per Match Statistics

| Statistic | Home Goals | Away Goals | Total Goals |
|-----------|------------|------------|-------------|
| **Mean** | 1.68 | 1.32 | 3.00 |
| **Median** | 2.0 | 1.0 | 3.0 |
| **Std Dev** | 1.29 | 1.14 | 1.71 |
| **Min** | 0 | 0 | 0 |
| **Max** | 9 | 8 | 12 |
| **Mode** | 1 | 1 | 3 |

### Total Goals Distribution

| Total Goals | Frequency | Percentage | Visualization |
|-------------|-----------|------------|---------------|
| 0 goals | 42 | 0.7% | ▌ |
| 1 goal | 487 | 8.2% | ████ |
| 2 goals | 1,203 | 20.2% | ██████████ |
| 3 goals | 1,685 | 28.2% | ██████████████ |
| 4 goals | 1,346 | 22.5% | ███████████ |
| 5 goals | 738 | 12.4% | ██████ |
| 6 goals | 315 | 5.3% | ██▌ |
| 7+ goals | 154 | 2.6% | █ |

### Key Findings:
- **Average goals per match:** 3.00 (Poisson-like distribution)
- **Most common scoreline:** 1-1, 2-1, 1-0 (typical for this level)
- **High-scoring matches:** 2.6% with 7+ total goals
- **Goalless draws:** Very rare (0.7%)
- **Home goal advantage:** +0.36 goals per match (1.68 vs 1.32)

### Goals by Result Type

| Result | Avg Home Goals | Avg Away Goals | Avg Total |
|--------|----------------|----------------|-----------|
| Home Win | 2.45 | 0.95 | 3.40 |
| Draw | 1.32 | 1.32 | 2.64 |
| Away Win | 0.78 | 2.18 | 2.96 |

---

## 3. Team Ratings Analysis

![Rating Distributions](figures/rating_distributions.png)

### Elo Rating System

| Statistic | Value |
|-----------|-------|
| **Mean** | 1,500 (by design) |
| **Median** | 1,496 |
| **Std Dev** | 68.4 |
| **Min** | 1,318 |
| **Max** | 1,674 |
| **Range** | 356 points |

### Pi Rating System

| Statistic | Value |
|-----------|-------|
| **Mean** | 1.487 |
| **Median** | 1.485 |
| **Std Dev** | 0.143 |
| **Min** | 1.124 |
| **Max** | 1.892 |
| **Range** | 0.768 |

### Rating Correlation
- **Elo vs Pi correlation:** 0.847 (very strong)
- Both systems capture similar team strength patterns
- Pi-ratings show slightly more variance

### Key Findings:
- **Well-calibrated:** Elo range (1,318-1,674) appropriate for single league
- **Stable distribution:** Ratings centered around league average
- **No extreme outliers:** No teams with ratings far outside expected range
- **Temporal stability:** Ratings evolve smoothly over time (see evolution chart)

### Rating Differences and Prediction Power

| Elo Difference | Home Win % | Draw % | Away Win % | Sample Size |
|----------------|------------|--------|------------|-------------|
| -100 to -50 | 28.3% | 29.1% | 42.6% | 847 |
| -50 to 0 | 38.7% | 28.4% | 32.9% | 1,423 |
| 0 to +50 | 47.2% | 27.1% | 25.7% | 1,689 |
| +50 to +100 | 55.8% | 26.3% | 17.9% | 1,134 |
| +100+ | 64.1% | 23.7% | 12.2% | 877 |

**Insight:** Clear relationship between rating difference and outcome probability.

---

## 4. Betting Odds Analysis

![Odds Analysis](figures/odds_analysis.png)

### Betting Odds Summary (5,882 matches with odds)

| Outcome | Mean Odds | Median Odds | Implied Prob |
|---------|-----------|-------------|--------------|
| **Home Win** | 2.18 | 2.05 | 45.9% |
| **Draw** | 3.42 | 3.35 | 29.2% |
| **Away Win** | 3.65 | 3.25 | 27.4% |

### Market Efficiency Analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Overround** | 7.0% | Typical for football |
| **Market Accuracy** | 47.5% | Near-random for 3-way |
| **Favorite Wins** | 47.5% | Efficient market |
| **Underdog Wins** | 22.3% | Within expectations |

### Odds Distribution Characteristics:
- **Home odds range:** 1.10 to 15.00
- **Draw odds range:** 2.50 to 5.50 (narrow, predictable)
- **Away odds range:** 1.15 to 20.00
- **Most common home odds:** 1.80-2.50 (moderate favorites)

### Overround Distribution:
- **Mean overround:** 7.0%
- **Median overround:** 6.9%
- **Std deviation:** 1.1%
- **Range:** 4.2% to 12.8%

**Insight:** Consistent bookmaker margins indicate stable, efficient market.

### Implied Probability vs Actual Results

| Market Favorite | Implied % | Actual % | Calibration |
|-----------------|-----------|----------|-------------|
| Strong Home (<1.80) | 55-60% | 58.3% | ✅ Well-calibrated |
| Moderate Home (1.80-2.50) | 40-55% | 46.2% | ✅ Well-calibrated |
| Even (2.50-3.00) | 33-40% | 35.7% | ✅ Well-calibrated |
| Moderate Away (3.00-4.00) | 25-33% | 28.1% | ✅ Well-calibrated |
| Strong Away (>4.00) | <25% | 22.4% | ✅ Well-calibrated |

**Key Finding:** Betting markets are well-calibrated predictors of match outcomes.

---

## 5. Match Statistics

![Statistics Coverage](figures/stats_coverage.png)
![Match Statistics](figures/shots_analysis.png)

### Coverage by Season

| Era | Seasons | Avg Coverage | Best Season | Worst Season |
|-----|---------|--------------|-------------|--------------|
| **Pre-Statistics** | 2009-2014 | 0% | - | - |
| **Early Coverage** | 2014-2018 | 43.2% | 2017-18 (52%) | 2014-15 (34%) |
| **Improved** | 2018-2022 | 64.7% | 2020-21 (79%) | 2018-19 (55%) |
| **Recent** | 2022-2025 | 58.3% | 2023-24 (70%) | 2022-23 (45%) |

### When Statistics Are Available:

#### Possession Statistics (2,244 matches)
| Team | Mean | Median | Std Dev | Min | Max |
|------|------|--------|---------|-----|-----|
| **Home** | 51.2% | 51.0% | 9.3% | 18% | 82% |
| **Away** | 48.8% | 49.0% | 9.3% | 18% | 82% |

**Insight:** Slight home possession advantage (51.2% vs 48.8%).

#### Shots Statistics
| Metric | Home Mean | Away Mean | Home Advantage |
|--------|-----------|-----------|----------------|
| **Total Shots** | 13.8 | 11.4 | +2.4 (+21%) |
| **Shots on Target** | 5.2 | 4.3 | +0.9 (+21%) |
| **Shot Accuracy** | 37.7% | 37.7% | 0% |

**Insight:** Home teams take more shots but similar accuracy.

#### Passing Statistics
| Metric | Home Mean | Away Mean |
|--------|-----------|-----------|
| **Total Passes** | 342 | 318 |
| **Accurate Passes** | 268 | 248 |
| **Pass Accuracy** | 78.4% | 78.0% |

#### Disciplinary Statistics
| Metric | Home Mean | Away Mean |
|--------|-----------|-----------|
| **Yellow Cards** | 2.1 | 2.3 |
| **Red Cards** | 0.09 | 0.11 |
| **Fouls** | 13.2 | 14.1 |

**Insight:** Away teams commit more fouls and receive more cards (referee bias?).

---

## 6. Weather Data

### Coverage Analysis
- **Overall coverage:** 81.9% of matches
- **Recently added:** November 2025
- **Backfill quality:** Good historical coverage

### Weather Conditions Distribution

| Metric | Mean | Median | Std Dev | Min | Max |
|--------|------|--------|---------|-----|-----|
| **Temperature (°C)** | 12.5 | 12.8 | 8.7 | -11.6 | 33.9 |
| **Humidity (%)** | 66.8 | 67.0 | 15.4 | 20.5 | 100.0 |
| **Wind Speed (km/h)** | 13.2 | 12.5 | 7.8 | 0.0 | 46.8 |
| **Precipitation (mm)** | 0.09 | 0.0 | 0.34 | 0.0 | 4.5 |

### Weather Categories

| Category | Count | Percentage | Definition |
|----------|-------|------------|------------|
| **Cold (<5°C)** | 1,247 | 25.5% | Winter matches |
| **Moderate (5-20°C)** | 2,896 | 59.2% | Spring/Fall |
| **Hot (>20°C)** | 745 | 15.2% | Summer matches |
| **Rainy (>0.1mm)** | 412 | 8.4% | Wet conditions |
| **Windy (>25km/h)** | 63 | 1.3% | High wind |

### Weather Impact on Results (Preliminary)

| Condition | Home Win % | Draw % | Away Win % |
|-----------|------------|--------|------------|
| **Cold** | 44.1% | 27.8% | 28.1% |
| **Moderate** | 43.2% | 27.3% | 29.5% |
| **Hot** | 43.7% | 27.6% | 28.7% |
| **Rainy** | 42.5% | 28.2% | 29.3% |
| **Windy** | 41.3% | 31.7% | 27.0% |

**Preliminary Insight:** Minimal weather impact on results. Windy conditions show slightly more draws.

---

## 7. Temporal Patterns

### Matches by Day of Week

| Day | Matches | Percentage | Avg Goals |
|-----|---------|------------|-----------|
| **Friday** | 734 | 12.3% | 2.97 |
| **Saturday** | 3,567 | 59.7% | 3.01 |
| **Sunday** | 1,342 | 22.5% | 2.99 |
| **Midweek** | 327 | 5.5% | 3.04 |

**Insight:** Saturday is primary match day. No significant goal difference by day.

### Matches by Month

| Month | Matches | Avg Temp (°C) | Home Win % |
|-------|---------|---------------|------------|
| **Jul** | 178 | 21.3 | 44.9% |
| **Aug** | 643 | 19.2 | 43.1% |
| **Sep** | 658 | 15.7 | 43.8% |
| **Oct** | 671 | 11.4 | 43.2% |
| **Nov** | 598 | 6.8 | 43.7% |
| **Dec** | 443 | 3.2 | 42.9% |
| **Feb** | 512 | 4.1 | 43.9% |
| **Mar** | 687 | 7.9 | 43.6% |
| **Apr** | 718 | 12.1 | 43.1% |
| **May** | 862 | 16.3 | 43.4% |

**Insight:** Consistent home advantage across all months and temperatures.

---

## 8. Home Advantage

### Overall Home Advantage Metrics

| Metric | Home | Away | Advantage |
|--------|------|------|-----------|
| **Win %** | 43.4% | 29.1% | +14.3 pp |
| **Goals per match** | 1.68 | 1.32 | +0.36 |
| **Points per match** | 1.56 | 1.17 | +0.39 |
| **Shots per match** | 13.8 | 11.4 | +2.4 |
| **Possession %** | 51.2% | 48.8% | +2.4 pp |

### Home Advantage by Elo Difference

| Scenario | Home Win % | Draw % | Away Win % |
|----------|------------|--------|------------|
| **Home Much Better** (+100 Elo) | 64.1% | 23.7% | 12.2% |
| **Home Better** (+50 Elo) | 55.8% | 26.3% | 17.9% |
| **Even** (±0 Elo) | 47.2% | 27.1% | 25.7% |
| **Away Better** (-50 Elo) | 38.7% | 28.4% | 32.9% |
| **Away Much Better** (-100 Elo) | 28.3% | 29.1% | 42.6% |

**Key Finding:** Home advantage persists across all team quality levels, adding ~10-15% to win probability.

---

## Key Statistical Insights

### 1. Predictive Feature Hierarchy
Based on correlation with match outcomes:

**Tier 1 (Strongest):**
- Pi-rating difference (r = 0.42)
- Elo rating difference (r = 0.41)
- Implied probability from odds (r = 0.39)

**Tier 2 (Strong):**
- Points last 5 matches (r = 0.28)
- Goals for/against ratio (r = 0.24)
- Head-to-head record (r = 0.19)

**Tier 3 (Moderate):**
- Home advantage (r = 0.14)
- Rest days (r = 0.08)
- Weather conditions (r = 0.03-0.05)

**Tier 4 (Weak):**
- Attendance (insufficient data)
- Venue name (not predictive)
- Match statistics (post-match only)

### 2. Class Balance for ML
- **Home wins:** 43.4% (2,588 matches) - Majority class
- **Draws:** 27.5% (1,643 matches) - Minority class
- **Away wins:** 29.1% (1,739 matches) - Between

**Recommendation:** Consider class weights or stratified sampling for balanced training.

### 3. Data Completeness Timeline

![Feature Completeness](figures/feature_completeness.png)

**Era 1 (2009-2014):** Basic data only (results, ratings, odds)
**Era 2 (2014-2018):** Added match statistics (40-50% coverage)
**Era 3 (2018-2024):** Improved statistics (60-80% coverage)
**Era 4 (2024+):** Added weather data (82% coverage)

---

## Conclusions

1. **Data Quality:** Excellent for core prediction features, adequate for supplementary
2. **Home Advantage:** Strong and consistent (+14.3 pp)
3. **Market Efficiency:** Betting odds are well-calibrated predictors
4. **Feature Richness:** 103 features span all relevant categories
5. **Temporal Coverage:** 17 seasons provide substantial training data
6. **Readiness:** Dataset ready for CatBoost model training

---

**Related Documentation:**
- [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) - Overall quality assessment
- [DATA_DICTIONARY.md](DATA_DICTIONARY.md) - Feature definitions
- [COMPLETENESS_ANALYSIS.md](COMPLETENESS_ANALYSIS.md) - Coverage details
- [RECOMMENDATIONS.md](RECOMMENDATIONS.md) - Improvement roadmap

---

*Generated by comprehensive_data_analysis.py on 2025-11-08*
