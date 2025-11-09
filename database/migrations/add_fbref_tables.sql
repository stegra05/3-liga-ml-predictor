-- Migration: Add FBref-specific tables
-- Date: 2025-11-08
-- Description: Adds player match statistics table for detailed player performance per match

-- =============================================
-- PLAYER MATCH STATISTICS (NEW)
-- =============================================

-- Detailed player performance statistics for each match
-- This complements player_season_stats with match-by-match granularity
CREATE TABLE IF NOT EXISTS player_match_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,

    -- Basic info
    started BOOLEAN DEFAULT 0,
    minutes_played INTEGER DEFAULT 0,
    shirt_number INTEGER,
    position VARCHAR(20), -- Position played in this match

    -- Scoring
    goals INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    penalties_scored INTEGER DEFAULT 0,
    penalties_attempted INTEGER DEFAULT 0,

    -- Shooting
    shots_total INTEGER DEFAULT 0,
    shots_on_target INTEGER DEFAULT 0,

    -- Passing
    passes_total INTEGER DEFAULT 0,
    passes_completed INTEGER DEFAULT 0,
    pass_accuracy_percent REAL,
    key_passes INTEGER DEFAULT 0,

    -- Defensive
    tackles_total INTEGER DEFAULT 0,
    tackles_won INTEGER DEFAULT 0,
    interceptions INTEGER DEFAULT 0,
    clearances INTEGER DEFAULT 0,

    -- Discipline
    yellow_cards INTEGER DEFAULT 0,
    red_cards INTEGER DEFAULT 0,
    fouls_committed INTEGER DEFAULT 0,
    fouls_drawn INTEGER DEFAULT 0,

    -- Other
    offsides INTEGER DEFAULT 0,
    crosses INTEGER DEFAULT 0,
    dribbles_attempted INTEGER DEFAULT 0,
    dribbles_successful INTEGER DEFAULT 0,

    -- Goalkeeper-specific stats (if applicable)
    saves INTEGER DEFAULT 0,
    goals_conceded INTEGER DEFAULT 0,
    clean_sheet BOOLEAN DEFAULT 0,

    source VARCHAR(50) DEFAULT 'fbref',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    UNIQUE(match_id, player_id, source)
);

CREATE INDEX IF NOT EXISTS idx_player_match_stats_match ON player_match_stats(match_id);
CREATE INDEX IF NOT EXISTS idx_player_match_stats_player ON player_match_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_player_match_stats_team ON player_match_stats(team_id);

-- =============================================
-- ADVANCED MATCH STATISTICS (OPTIONAL)
-- =============================================

-- Advanced analytics metrics (xG, progressive actions, etc.)
-- Note: These are unlikely to be available for 3. Liga on FBref
-- but this table is created for completeness
CREATE TABLE IF NOT EXISTS advanced_match_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    is_home BOOLEAN NOT NULL,

    -- Expected metrics
    xg REAL, -- Expected goals
    xg_against REAL,
    xa REAL, -- Expected assists
    npxg REAL, -- Non-penalty expected goals

    -- Progressive actions
    progressive_passes INTEGER,
    progressive_carries INTEGER,
    progressive_pass_distance REAL,

    -- Pressure and possession
    pressures INTEGER,
    pressure_success_rate REAL,
    pressures_att_third INTEGER,
    pressures_mid_third INTEGER,
    pressures_def_third INTEGER,

    -- Pass types
    through_balls INTEGER,
    switches INTEGER,
    ground_passes INTEGER,
    low_passes INTEGER,
    high_passes INTEGER,

    -- Shot creation
    shot_creating_actions INTEGER,
    goal_creating_actions INTEGER,

    source VARCHAR(50) DEFAULT 'fbref',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    UNIQUE(match_id, team_id, source)
);

CREATE INDEX IF NOT EXISTS idx_advanced_stats_match ON advanced_match_stats(match_id);
CREATE INDEX IF NOT EXISTS idx_advanced_stats_team ON advanced_match_stats(team_id);

-- =============================================
-- DATA QUALITY TRACKING
-- =============================================

-- Track FBref scraping sessions and data quality
CREATE TABLE IF NOT EXISTS fbref_collection_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    season VARCHAR(9),
    collection_type VARCHAR(50), -- 'season_stats', 'match_stats', 'player_stats'

    -- Statistics
    items_attempted INTEGER DEFAULT 0,
    items_collected INTEGER DEFAULT 0,
    items_failed INTEGER DEFAULT 0,

    -- Timing
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds INTEGER,

    -- Status
    status VARCHAR(20), -- 'success', 'partial', 'failed'
    error_message TEXT,

    -- Details
    teams_processed INTEGER,
    matches_processed INTEGER,
    players_processed INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fbref_log_season ON fbref_collection_log(season);
CREATE INDEX IF NOT EXISTS idx_fbref_log_status ON fbref_collection_log(status);

-- =============================================
-- VIEWS FOR CONVENIENCE
-- =============================================

-- View: Player performance with team and match context
CREATE VIEW IF NOT EXISTS v_player_match_performance AS
SELECT
    pms.*,
    p.full_name AS player_name,
    p.position AS primary_position,
    t.team_name,
    m.season,
    m.matchday,
    m.match_datetime,
    m.home_team_id,
    m.away_team_id,
    CASE WHEN m.home_team_id = pms.team_id THEN 1 ELSE 0 END AS is_home_team
FROM player_match_stats pms
JOIN players p ON pms.player_id = p.player_id
JOIN teams t ON pms.team_id = t.team_id
JOIN matches m ON pms.match_id = m.match_id;

-- View: Complete match statistics (combining standard and advanced)
CREATE VIEW IF NOT EXISTS v_complete_match_stats AS
SELECT
    ms.*,
    ams.xg,
    ams.xa,
    ams.progressive_passes,
    ams.progressive_carries,
    ams.pressures,
    ams.shot_creating_actions,
    ams.goal_creating_actions
FROM match_statistics ms
LEFT JOIN advanced_match_stats ams
    ON ms.match_id = ams.match_id
    AND ms.team_id = ams.team_id
    AND ams.source = 'fbref';

-- =============================================
-- MIGRATION NOTES
-- =============================================

-- To apply this migration:
-- sqlite3 database/3liga.db < database/migrations/add_fbref_tables.sql

-- To verify:
-- sqlite3 database/3liga.db "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%match_stats%';"
