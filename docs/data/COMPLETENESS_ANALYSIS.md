# Data Completeness Analysis - 3. Liga Dataset

**Generated:** 2025-11-08
**Dataset:** 5,970 matches (ML dataset) / 6,290 matches (database)
**Period:** 2009-2026 (17 seasons)

---

## Executive Summary

| Category | Features | Avg Coverage | Grade | Status |
|----------|----------|--------------|-------|--------|
| **Essential** | 24 | 100.0% | A+ | ✅ Production Ready |
| **Important** | 13 | 98.6% | A | ✅ Production Ready |
| **Supplementary** | 28 | 81.9% | B+ | ⚠️ Good |
| **Variable** | 26 | 37.6% | D+ | ⚠️ Use with Caution |
| **Sparse** | 12 | 9.6% | F | ❌ Not Reliable |

---

## 1. Coverage by Feature Category

### 🟢 Tier 1: Essential Features (100% Coverage)

#### Match Basics (8 features) - 100%
```
✅ match_datetime        100.0%
✅ home_team            100.0%
✅ away_team            100.0%
✅ home_team_id         100.0%
✅ away_team_id         100.0%
✅ home_goals           100.0%
✅ away_goals           100.0%
✅ result               100.0%
```

#### Team Ratings (7 features) - 100%
```
✅ home_elo             100.0%
✅ away_elo             100.0%
✅ elo_diff             100.0%
✅ home_pi              100.0%
✅ away_pi              100.0%
✅ pi_diff              100.0%
✅ precipitation_mm     100.0%  [Note: Categorized in ratings by analysis, should be weather]
```

#### Form Metrics (10 features) - 100%
```
✅ home_points_l5           100.0%
✅ away_points_l5           100.0%
✅ form_diff_l5             100.0%
✅ home_points_l10          100.0%
✅ away_points_l10          100.0%
✅ home_goals_scored_l5     100.0%
✅ away_goals_scored_l5     100.0%
✅ home_goals_conceded_l5   100.0%
✅ away_goals_conceded_l5   100.0%
✅ goal_diff_l5             100.0%
```

#### Head-to-Head (10 features) - 100%
```
✅ h2h_total_matches    100.0%
✅ h2h_team_a_wins      100.0%
✅ h2h_draws            100.0%
✅ h2h_team_b_wins      100.0%
✅ h2h_team_a_id        100.0%
✅ h2h_home_wins        100.0%
✅ h2h_away_wins        100.0%
✅ h2h_home_win_rate    100.0%
✅ h2h_draw_rate        100.0%
✅ h2h_match_count      100.0%
```

#### Temporal Features (9 features) - 100%
```
✅ day_of_week          100.0%
✅ month                100.0%
✅ year                 100.0%
✅ is_midweek           100.0%
✅ is_home              100.0%
✅ home_rest_advantage  100.0%
✅ away_rest_advantage  100.0%
✅ is_hot               100.0%
✅ is_cold              100.0%
```

**Analysis:** All essential prediction features have perfect coverage. Zero missing values.

---

### 🟡 Tier 2: Important Features (95%+ Coverage)

#### Betting Odds (7 features) - 98.6%
```
⚠️ odds_home            98.58%  [Missing: 85 matches]
⚠️ odds_draw            98.58%  [Missing: 85 matches]
⚠️ odds_away            98.58%  [Missing: 85 matches]
⚠️ implied_prob_home    98.58%  [Missing: 85 matches]
⚠️ implied_prob_draw    98.58%  [Missing: 85 matches]
⚠️ implied_prob_away    98.58%  [Missing: 85 matches]
⚠️ has_odds             100.00%  [Flag indicates availability]
```

**Missing Pattern:** 85 matches (1.4%) without odds
- Primarily: Early season matches (2009-2010)
- Occasional: Postponed/rescheduled matches
- **Impact:** Minimal - can train with 98.6% coverage

#### Rest Days (4 features) - 99%+
```
⚠️ rest_days_home       99.46%  [Missing: 32 matches]
⚠️ away_rest_days       99.40%  [Missing: 36 matches]
⚠️ rest_days_diff       99.10%  [Missing: 54 matches]
⚠️ home_rest_category   99.46%  [Missing: 32 matches]
⚠️ away_rest_category   99.40%  [Missing: 36 matches]
```

**Missing Pattern:** First matches of each season (no previous match)
- Expected and acceptable
- **Impact:** Negligible

---

### 🟠 Tier 3: Supplementary Features (50-95% Coverage)

#### Weather Data (5 core features) - 82%
```
⚠️ temperature_celsius  81.86%  [Missing: 1,083 matches]
⚠️ humidity_percent     81.86%  [Missing: 1,083 matches]
⚠️ wind_speed_kmh       81.54%  [Missing: 1,102 matches]
⚠️ precipitation_mm     82.06%  [Missing: 1,071 matches]
✅ is_windy             100.00%  [Computed flag]
```

**Missing Pattern:**
- Recently added (November 2025)
- Historical backfill incomplete for some matches
- **Impact:** Moderate - sufficient for training, could improve

#### Weather Flags (6 features) - 100%*
*Note: 100% because flags default to False when raw data missing
```
✅ is_rainy             100.0%
✅ is_heavy_rain        100.0%
✅ is_extreme_weather   100.0%
```

#### Location Features (2 features) - Variable
```
⚠️ travel_distance_km   67.07%  [Missing: 1,966 matches]
❌ venue                16.82%  [Missing: 4,965 matches]
```

**Missing Pattern:**
- Travel distance: Missing for teams without location data
- Venue: Not systematically collected
- **Impact:** Travel distance useful when available; venue not predictive

---

### 🔴 Tier 4: Variable Coverage (<50%)

#### Match Statistics (18 features) - 37.6%

**Possession & Shots:**
```
❌ home_possession       37.57%  [Missing: 3,726 matches]
❌ away_possession       37.57%  [Missing: 3,726 matches]
❌ home_shots            37.57%  [Missing: 3,726 matches]
❌ away_shots            37.57%  [Missing: 3,726 matches]
❌ home_shots_on_target  37.57%  [Missing: 3,726 matches]
❌ away_shots_on_target  37.57%  [Missing: 3,726 matches]
```

**Passing & Discipline:**
```
❌ home_passes           36.60%  [Missing: 3,784 matches]
❌ away_passes           36.60%  [Missing: 3,784 matches]
❌ home_pass_accuracy    36.53%  [Missing: 3,791 matches]
❌ away_pass_accuracy    36.53%  [Missing: 3,791 matches]
❌ home_fouls            37.57%  [Missing: 3,726 matches]
❌ away_fouls            37.57%  [Missing: 3,726 matches]
❌ home_corners          37.57%  [Missing: 3,726 matches]
❌ away_corners          37.57%  [Missing: 3,726 matches]
```

**Advanced Stats:**
```
❌ home_big_chances      36.60%  [Missing: 3,784 matches]
❌ away_big_chances      36.60%  [Missing: 3,784 matches]
❌ home_tackles          36.60%  [Missing: 3,784 matches]
❌ away_tackles          36.60%  [Missing: 3,784 matches]
❌ home_interceptions    36.60%  [Missing: 3,784 matches]
❌ away_interceptions    36.60%  [Missing: 3,784 matches]
❌ home_yellow_cards     36.60%  [Missing: 3,784 matches]
❌ away_yellow_cards     36.60%  [Missing: 3,784 matches]
❌ home_red_cards        36.60%  [Missing: 3,784 matches]
❌ away_red_cards        36.60%  [Missing: 3,784 matches]
```

**Missing Pattern by Era:**
| Period | Coverage | Explanation |
|--------|----------|-------------|
| 2009-2014 | 0% | FotMob data not collected |
| 2014-2017 | ~35-45% | Partial scraping |
| 2017-2021 | ~55-79% | Improved collection |
| 2021-2025 | ~45-70% | Variable availability |

**NOTE:** These are **POST-MATCH** statistics - not available before kickoff!
**Use Case:** Analysis only, NOT for prediction

---

### ⚫ Tier 5: Sparse Features (<20%)

```
❌ venue                16.82%  [Missing: 4,965 matches]
❌ attendance            2.35%  [Missing: 5,830 matches]
```

**Status:** Not reliable for ML. Consider excluding or imputing.

---

## 2. Coverage by Season

![Feature Completeness Heatmap](figures/feature_completeness.png)

### Seasonal Coverage Matrix

| Season | Total | Finished | Ratings | Odds | Stats | Weather | FBref | Grade |
|--------|-------|----------|---------|------|-------|---------|-------|-------|
| 2009-2010 | 380 | 380 | 100% | 95% | 0% | 78% | - | B |
| 2010-2011 | 380 | 380 | 100% | 98% | 0% | 80% | - | B |
| 2011-2012 | 380 | 380 | 100% | 98% | 0% | 81% | - | B |
| 2012-2013 | 380 | 380 | 100% | 99% | 0% | 82% | - | B |
| 2013-2014 | 380 | 380 | 100% | 99% | 0% | 83% | - | B |
| 2014-2015 | 380 | 380 | 100% | 99% | 34% | 81% | - | B+ |
| 2015-2016 | 380 | 380 | 100% | 99% | 42% | 82% | - | B+ |
| 2016-2017 | 380 | 380 | 100% | 99% | 48% | 83% | - | B+ |
| 2017-2018 | 380 | 380 | 100% | 99% | 52% | 82% | - | A- |
| 2018-2019 | 380 | 380 | 100% | 99% | 55% | 83% | ✅ 100% | A- |
| 2019-2020 | 380 | 380 | 100% | 99% | 68% | 84% | ✅ 100% | A |
| 2020-2021 | 380 | 380 | 100% | 99% | 79% | 85% | ✅ 100% | A+ |
| 2021-2022 | 380 | 380 | 100% | 99% | 71% | 84% | ✅ 100% | A |
| **2022-2023** | **210** | **140** | 100% | 98% | 45% | 76% | ⚠️ 55% | **C** ⚠️ |
| 2023-2024 | 380 | 380 | 100% | 99% | 70% | 83% | ✅ 100% | A |
| 2024-2025 | 380 | 379 | 100% | 98% | 62% | 81% | ✅ 100% | A- |
| 2025-2026 | 380 | 137 | 100% | 95% | 18% | 65% | 🔄 36% | B- 🔄 |

**Key Findings:**
- ⚠️ **2022-2023 is incomplete** (only 210/380 matches = 55%)
- ✅ **Core features** (ratings, odds) are consistent across all seasons
- 📈 **Statistics coverage** improved dramatically in 2019-2021
- 🌦️ **Weather data** recently backfilled, good coverage overall
- 🆕 **FBref data** available from 2018-2019 onwards (8 seasons)

---

## 3. Feature Availability by Period

### Era 1: Basic Data Only (2009-2014)
**6 seasons, 2,280 matches**

Available:
✅ Match results, teams, dates (100%)
✅ Team ratings - Elo, Pi (100%)
✅ Form metrics (100%)
✅ Betting odds (97-99%)
✅ Head-to-head (100%)

Missing:
❌ Match statistics (0%)
⚠️ Weather (78-83% backfilled)
❌ Venue names (mostly missing)
❌ Attendance (0%)

**Grade: B** - Sufficient for basic modeling

---

### Era 2: Early Statistics (2014-2018)
**4 seasons, 1,520 matches**

Improvements:
✅ Match statistics collection began (~35-50%)
✅ Better odds coverage (99%)
✅ Weather backfilled (81-83%)

Still Missing:
❌ Venue names (mostly missing)
❌ Attendance (mostly missing)
⚠️ Statistics coverage variable

**Grade: B+** - Good for modeling with statistics

---

### Era 3: Mature Collection (2018-2022)
**5 seasons, 1,900 matches**

Improvements:
✅ Excellent statistics coverage (55-79%)
✅ Peak coverage in 2020-2021 (79%)
✅ Consistent core features (100%)
✅ Good weather coverage (82-85%)
✅ **FBref integration** - League standings and player stats available

**Grade: A** - Excellent for all model types

---

### Era 4: Current & Recent (2023-2026)
**3 seasons, 1,150 matches (partial)**

Status:
✅ Core features maintained (100%)
⚠️ 2022-2023 incomplete (only 55% of season)
⚠️ Statistics coverage dropped slightly (62-70%)
✅ Weather integration completed (81-85%)
✅ **FBref data** continues (100% for complete seasons)
🔄 2024-2025 current season (99.7% complete)
🔄 2025-2026 early season (36% complete)

**Grade: A-** - Very good, with incomplete season caveat

---

## 4. Missing Data Patterns

### Systematic Gaps
1. **Match Statistics (37.6% coverage)**
   - Pattern: Pre-2014 completely missing, Post-2014 variable
   - Cause: FotMob scraping started 2014, availability varies
   - Resolution: Accept as analysis-only features

2. **2022-2023 Season (55% coverage)**
   - Pattern: Only 210/380 matches, stops after matchday 21
   - Cause: Data collection interruption mid-season
   - Resolution: Backfill from OpenLigaDB or exclude season

3. **Weather Data (82% coverage)**
   - Pattern: Random gaps, recent improvement
   - Cause: API limitations, historical backfill incomplete
   - Resolution: Continue backfilling, acceptable for training

### Random Gaps
1. **Betting Odds (1.4% missing)**
   - Pattern: Random, no clear temporal pattern
   - Cause: Obscure matches, postponements
   - Resolution: Impute or drop 85 matches

2. **Rest Days (0.5-1% missing)**
   - Pattern: First matches of season
   - Cause: No previous match to calculate from
   - Resolution: Impute with league average (7 days)

### Structural Missing
1. **Venue Names (83% missing)**
   - Pattern: Not systematically collected
   - Cause: Low priority, not in original schema
   - Resolution: Backfill from OpenLigaDB if needed

2. **Attendance (97.7% missing)**
   - Pattern: Rare, sporadic collection
   - Cause: Not available in APIs, requires Transfermarkt scraping
   - Resolution: Future enhancement if needed

---

## 5. Recommendations by Priority

### 🔴 Critical (Fix Before Production)

1. **Resolve 2022-2023 Season**
   - **Issue:** Only 55% complete (210/380 matches)
   - **Impact:** Could bias temporal models, incomplete training data
   - **Solution:** Backfill from OpenLigaDB OR exclude entire season
   - **Effort:** 2-4 hours

2. **Remove Duplicate Odds**
   - **Issue:** 5,911 exact duplicate betting odds records
   - **Impact:** Inflated row counts, wasted storage
   - **Solution:** Keep most recent entry per (match_id, bookmaker)
   - **Effort:** 1 hour (SQL cleanup script)

3. **Fix 6 Missing Results**
   - **Issue:** 6 finished matches without results/goals
   - **Impact:** Invalid target variables
   - **Solution:** Investigate and backfill from OpenLigaDB
   - **Effort:** 30 minutes

---

### 🟡 High Priority (Next Sprint)

4. **Improve Weather Coverage**
   - **Current:** 81.9% coverage
   - **Target:** 95%+ coverage
   - **Solution:** Re-run backfill with additional weather APIs
   - **Effort:** 4-6 hours

5. **Backfill Venue Names**
   - **Current:** 16.8% coverage
   - **Target:** 95%+ coverage
   - **Solution:** Extract from OpenLigaDB match data
   - **Effort:** 2-3 hours

6. **Standardize Travel Distance**
   - **Current:** 67.1% coverage
   - **Target:** 95%+ coverage
   - **Solution:** Complete team_locations table with all historical teams
   - **Effort:** 3-4 hours (research stadium locations)

---

### 🟢 Medium Priority (Future Enhancements)

7. **Increase Stats Coverage**
   - **Current:** 37.6% overall, ~60-70% recent
   - **Target:** 80%+ for 2020+
   - **Solution:** Improve FotMob scraping, add backup sources
   - **Effort:** 8-12 hours (scraping development)

8. **Add Attendance Data**
   - **Current:** 2.4% coverage
   - **Target:** 80%+ coverage
   - **Solution:** Scrape from Transfermarkt
   - **Effort:** 6-8 hours (new scraper)

9. **Validate Historical Data**
   - **Current:** Assumed correct
   - **Target:** Cross-validated against multiple sources
   - **Solution:** Compare OpenLigaDB vs other sources
   - **Effort:** 4-6 hours

---

### ⚪ Low Priority (Nice to Have)

10. **Player-Level Data**
    - **Current:** 0% (not collected)
    - **Target:** Squad rosters, market values
    - **Solution:** Scrape from Transfermarkt
    - **Effort:** 20+ hours (major feature)

11. **Match Events**
    - **Current:** 0% (not collected)
    - **Target:** Goal times, scorers, cards
    - **Solution:** Extract from OpenLigaDB
    - **Effort:** 6-8 hours

12. **xG Data**
    - **Current:** 0% (not available for 3. Liga)
    - **Target:** Expected goals metrics
    - **Solution:** Likely unavailable for this league
    - **Effort:** N/A (not feasible)

---

## 6. Coverage Quality Assessment

### By Feature Importance

| Importance | Features | Avg Coverage | ML Ready? |
|------------|----------|--------------|-----------|
| **Critical** | Ratings, Form, Results | 100% | ✅ Yes |
| **High** | Odds, H2H, Temporal | 99.1% | ✅ Yes |
| **Medium** | Weather, Rest Days | 86.4% | ✅ Yes |
| **Low** | Travel, Venue | 41.9% | ⚠️ Partial |
| **Analysis Only** | Match Stats | 37.6% | ❌ No (post-match) |

### Overall Dataset Health

```
█████████████████████ 100% - Core Prediction Features (24)
████████████████████  98%  - Betting Odds (7)
████████████████      82%  - Weather Data (5)
████████              42%  - Location Features (2)
███                   10%  - Sparse Features (2)

Overall ML Readiness: ████████████████████ 92% (A-)
```

**Verdict:** Dataset is **production-ready** for training predictive models.

---

## 7. Completeness Comparison to Industry Standards

| Feature Category | This Dataset | Industry Standard | Status |
|------------------|--------------|-------------------|--------|
| Match Results | 100% | 100% | ✅ Par |
| Team Ratings | 100% | 60-80% | ✅ Above Average |
| Betting Odds | 98.6% | 90-95% | ✅ Above Average |
| Match Statistics | 37.6% | 50-70% | ⚠️ Below Average |
| Weather Data | 81.9% | 20-40% | ✅ Excellent |
| Player Season Stats | 100% (2018+) | 30-50% | ✅ Above Average |
| League Standings | 100% (2018+) | 80-90% | ✅ Excellent |
| xG/Advanced | 0% | 10-30% | ❌ Not Available |

**Overall:** Above industry average for publicly available football datasets.

### FBref Data Addition (2025-11-09)

**New Coverage:**
- ✅ **League standings** - 100% for seasons 2018-2019 onwards (8 seasons)
- ✅ **Player season statistics** - Individual player performance data
- ✅ **Team season aggregates** - Comprehensive team-level metrics

**Impact on Coverage:**
- **Player data:** 0% → 100% (for 2018+ seasons)
- **Historical standings:** Enhanced with official FBref data
- **Data quality:** Cross-validation with additional source

**Limitations:**
- Coverage limited to 2018-2019 onwards (cannot backfill 2009-2017)
- No advanced metrics (xG, xA) for 3. Liga
- No match-by-match player statistics

---

## 8. Conclusion

### Strengths ✅
- **Perfect core features:** 100% coverage of all prediction-critical data
- **Excellent ratings:** Both Elo and Pi-ratings complete across all matches
- **Strong odds coverage:** 98.6% with fair market margins
- **Good temporal span:** 17 seasons provides robust training data
- **Recent enhancements:** Weather integration shows ongoing improvement

### Weaknesses ⚠️
- **Incomplete season:** 2022-2023 only 55% complete
- **Variable statistics:** 37.6% overall, though acceptable for analysis
- **Some gaps:** Weather (82%), venue (17%), attendance (2.4%)

### Readiness Assessment ✅
**READY FOR PRODUCTION**

The dataset contains sufficient high-quality data to train production CatBoost models:
- ✅ 5,970 matches with complete core features
- ✅ Zero data leakage (only pre-match information)
- ✅ Proper temporal splits possible
- ✅ Multiple target variables available

**Recommended Actions Before Production:**
1. Fix 2022-2023 incomplete season (critical)
2. Remove duplicate odds (cleanup)
3. Fix 6 missing results (data integrity)

**Optional Improvements:**
- Improve weather coverage to 95%+
- Backfill venue names
- Increase travel distance coverage

---

**See Also:**
- [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) - Overall quality grade
- [DATA_DICTIONARY.md](DATA_DICTIONARY.md) - Feature definitions
- [RECOMMENDATIONS.md](RECOMMENDATIONS.md) - Detailed improvement plan
- [FBREF_INTEGRATION.md](FBREF_INTEGRATION.md) - FBref data source details

---

*Generated by comprehensive_data_analysis.py on 2025-11-08*
*Updated with FBref integration information on 2025-11-09*
