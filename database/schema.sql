-- 3. Liga Comprehensive Database Schema
-- Supports ML match prediction with extensive feature engineering

-- =============================================
-- CORE ENTITIES
-- =============================================

-- Teams master table with standardized IDs
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
    openligadb_id INTEGER UNIQUE,
    team_name VARCHAR(100) NOT NULL,
    team_name_short VARCHAR(50),
    team_name_alt VARCHAR(100), -- Alternative names for matching
    founded_year INTEGER,
    stadium_name VARCHAR(100),
    stadium_capacity INTEGER,
    city VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for name-based lookups
CREATE INDEX IF NOT EXISTS idx_teams_name ON teams(team_name);
CREATE INDEX IF NOT EXISTS idx_teams_openligadb ON teams(openligadb_id);

-- =============================================
-- MATCH DATA
-- =============================================

-- Core match results and metadata
CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    openligadb_match_id INTEGER UNIQUE,
    season VARCHAR(9) NOT NULL, -- Format: "2024-2025"
    matchday INTEGER NOT NULL,
    match_datetime TIMESTAMP NOT NULL,

    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,

    home_goals INTEGER,
    away_goals INTEGER,
    result CHAR(1), -- 'H', 'D', 'A'

    -- Match context
    is_finished BOOLEAN DEFAULT 0,
    venue VARCHAR(100),
    attendance INTEGER,
    referee VARCHAR(100),

    -- Weather conditions
    temperature_celsius REAL,
    humidity_percent REAL,
    wind_speed_kmh REAL,
    precipitation_mm REAL,
    weather_condition VARCHAR(50),
    weather_source TEXT, -- Source: 'meteostat', 'open_meteo', 'dwd'
    weather_confidence REAL, -- Confidence score 0.0-1.0

    -- Time context
    is_midweek BOOLEAN,
    is_derby BOOLEAN,
    competition_phase VARCHAR(50), -- e.g., "regular", "playoff"

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);

CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season);
CREATE INDEX IF NOT EXISTS idx_matches_datetime ON matches(match_datetime);
CREATE INDEX IF NOT EXISTS idx_matches_home_team ON matches(home_team_id);
CREATE INDEX IF NOT EXISTS idx_matches_away_team ON matches(away_team_id);

-- =============================================
-- DETAILED MATCH STATISTICS
-- =============================================

-- Team performance statistics per match
CREATE TABLE IF NOT EXISTS match_statistics (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    is_home BOOLEAN NOT NULL,

    -- Possession
    possession_percent REAL,

    -- Shooting
    shots_total INTEGER,
    shots_on_target INTEGER,
    shots_off_target INTEGER,
    shots_blocked INTEGER,
    big_chances INTEGER,
    big_chances_missed INTEGER,

    -- Passing
    passes_total INTEGER,
    passes_accurate INTEGER,
    pass_accuracy_percent REAL,
    key_passes INTEGER,
    crosses_total INTEGER,
    crosses_accurate INTEGER,
    long_balls_total INTEGER,
    long_balls_accurate INTEGER,

    -- Defensive actions
    tackles_total INTEGER,
    tackles_won INTEGER,
    interceptions INTEGER,
    clearances INTEGER,
    blocked_shots INTEGER,

    -- Duels and aerial
    duels_total INTEGER,
    duels_won INTEGER,
    aerials_total INTEGER,
    aerials_won INTEGER,

    -- Discipline
    fouls_committed INTEGER,
    fouls_won INTEGER,
    yellow_cards INTEGER,
    red_cards INTEGER,

    -- Set pieces
    corners INTEGER,
    offsides INTEGER,

    -- Other
    touches INTEGER,
    dribbles_attempted INTEGER,
    dribbles_successful INTEGER,

    source VARCHAR(50), -- 'fotmob', 'fbref', etc.
    has_complete_stats BOOLEAN DEFAULT 0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    UNIQUE(match_id, team_id)
);

CREATE INDEX IF NOT EXISTS idx_match_stats_match ON match_statistics(match_id);
CREATE INDEX IF NOT EXISTS idx_match_stats_team ON match_statistics(team_id);

-- =============================================
-- MATCH EVENTS
-- =============================================

-- Goals, cards, substitutions
CREATE TABLE IF NOT EXISTS match_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,

    event_type VARCHAR(20) NOT NULL, -- 'goal', 'yellow_card', 'red_card', 'substitution'
    minute INTEGER,
    minute_extra INTEGER, -- injury time

    player_id INTEGER, -- Link to players table
    player_name VARCHAR(100),

    -- For goals
    is_penalty BOOLEAN,
    is_own_goal BOOLEAN,
    assist_player_id INTEGER,
    assist_player_name VARCHAR(100),

    -- For substitutions
    player_out_id INTEGER,
    player_out_name VARCHAR(100),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (match_id) REFERENCES matches(match_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE INDEX IF NOT EXISTS idx_events_match ON match_events(match_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON match_events(event_type);

-- =============================================
-- LEAGUE STANDINGS
-- =============================================

-- Historical league tables after each matchday
CREATE TABLE IF NOT EXISTS league_standings (
    standing_id INTEGER PRIMARY KEY AUTOINCREMENT,
    season VARCHAR(9) NOT NULL,
    matchday INTEGER NOT NULL,
    team_id INTEGER NOT NULL,

    position INTEGER NOT NULL,
    matches_played INTEGER NOT NULL,
    wins INTEGER NOT NULL,
    draws INTEGER NOT NULL,
    losses INTEGER NOT NULL,
    goals_for INTEGER NOT NULL,
    goals_against INTEGER NOT NULL,
    goal_difference INTEGER NOT NULL,
    points INTEGER NOT NULL,

    -- Additional context
    form_last_5 VARCHAR(5), -- e.g., "WDLWW"

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    UNIQUE(season, matchday, team_id)
);

CREATE INDEX IF NOT EXISTS idx_standings_season ON league_standings(season);
CREATE INDEX IF NOT EXISTS idx_standings_team ON league_standings(team_id);

-- =============================================
-- BETTING ODDS
-- =============================================

-- Betting market data
CREATE TABLE IF NOT EXISTS betting_odds (
    odds_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,

    bookmaker VARCHAR(50) DEFAULT 'average',
    odds_home REAL,
    odds_draw REAL,
    odds_away REAL,

    -- Derived probabilities
    implied_prob_home REAL,
    implied_prob_draw REAL,
    implied_prob_away REAL,

    -- Timing
    odds_type VARCHAR(20) DEFAULT 'closing', -- 'opening', 'closing'
    collected_at TIMESTAMP,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

CREATE INDEX IF NOT EXISTS idx_odds_match ON betting_odds(match_id);

-- Unique constraint to prevent duplicate odds for same match/bookmaker
CREATE UNIQUE INDEX IF NOT EXISTS idx_betting_odds_unique 
ON betting_odds(match_id, bookmaker);

-- =============================================
-- PLAYER DATA
-- =============================================

-- Player master table
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,

    full_name VARCHAR(100) NOT NULL,
    known_name VARCHAR(100),
    date_of_birth DATE,
    nationality VARCHAR(50),

    position VARCHAR(20), -- GK, DF, MF, FW
    position_detail VARCHAR(50),

    height_cm INTEGER,
    foot VARCHAR(10), -- 'left', 'right', 'both'

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_players_name ON players(full_name);

-- Player season statistics
CREATE TABLE IF NOT EXISTS player_season_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    season VARCHAR(9) NOT NULL,

    -- Appearance data
    matches_played INTEGER DEFAULT 0,
    minutes_played INTEGER DEFAULT 0,
    starts INTEGER DEFAULT 0,

    -- Scoring
    goals INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    penalties_scored INTEGER DEFAULT 0,

    -- Discipline
    yellow_cards INTEGER DEFAULT 0,
    red_cards INTEGER DEFAULT 0,

    -- Market value
    market_value_eur INTEGER,

    -- Additional stats (from FBref if available)
    shots_total INTEGER,
    shots_on_target INTEGER,
    pass_accuracy_percent REAL,
    tackles_won INTEGER,

    source VARCHAR(50), -- 'fbref', etc.

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    UNIQUE(player_id, team_id, season)
);

CREATE INDEX IF NOT EXISTS idx_player_stats_season ON player_season_stats(season);
CREATE INDEX IF NOT EXISTS idx_player_stats_team ON player_season_stats(team_id);

-- Squad compositions (which players were on which team in which season)
CREATE TABLE IF NOT EXISTS squad_memberships (
    membership_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    season VARCHAR(9) NOT NULL,

    jersey_number INTEGER,
    joined_date DATE,
    left_date DATE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    UNIQUE(player_id, team_id, season)
);

-- =============================================
-- TRANSFERS
-- =============================================

-- Transfer market data
CREATE TABLE IF NOT EXISTS transfers (
    transfer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    from_team_id INTEGER,
    to_team_id INTEGER,

    season VARCHAR(9) NOT NULL,
    transfer_date DATE,
    transfer_fee_eur INTEGER, -- NULL for free transfers
    transfer_type VARCHAR(20), -- 'permanent', 'loan', 'free'

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (from_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (to_team_id) REFERENCES teams(team_id)
);

-- =============================================
-- DERIVED FEATURES & RATINGS
-- =============================================

-- Team rating systems (Elo, Pi-ratings, etc.)
CREATE TABLE IF NOT EXISTS team_ratings (
    rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    match_id INTEGER, -- Rating after this match (NULL for season start)

    season VARCHAR(9) NOT NULL,
    matchday INTEGER NOT NULL,

    -- Rating systems
    elo_rating REAL,
    pi_rating REAL,

    -- Form metrics
    points_last_5 INTEGER,
    points_last_10 INTEGER,
    goals_scored_last_5 REAL,
    goals_conceded_last_5 REAL,

    -- Streak data
    current_win_streak INTEGER DEFAULT 0,
    current_unbeaten_streak INTEGER DEFAULT 0,
    current_loss_streak INTEGER DEFAULT 0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

CREATE INDEX IF NOT EXISTS idx_ratings_team_season ON team_ratings(team_id, season);
CREATE INDEX IF NOT EXISTS idx_ratings_matchday ON team_ratings(season, matchday);

-- Head-to-head historical records
CREATE TABLE IF NOT EXISTS head_to_head (
    h2h_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_a_id INTEGER NOT NULL,
    team_b_id INTEGER NOT NULL,

    -- Overall record (team_a perspective)
    total_matches INTEGER DEFAULT 0,
    team_a_wins INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    team_b_wins INTEGER DEFAULT 0,

    -- Home/away splits
    team_a_home_wins INTEGER DEFAULT 0,
    team_a_away_wins INTEGER DEFAULT 0,

    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (team_a_id) REFERENCES teams(team_id),
    FOREIGN KEY (team_b_id) REFERENCES teams(team_id),
    UNIQUE(team_a_id, team_b_id)
);

-- =============================================
-- DATA COLLECTION METADATA
-- =============================================

-- Track data collection runs
CREATE TABLE IF NOT EXISTS collection_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source VARCHAR(50) NOT NULL, -- 'openligadb', 'fbref', etc.
    collection_type VARCHAR(50) NOT NULL, -- 'match_data', 'player_stats', etc.

    season VARCHAR(9),
    matchday INTEGER,

    status VARCHAR(20) NOT NULL, -- 'success', 'partial', 'failed'
    records_collected INTEGER DEFAULT 0,
    error_message TEXT,

    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    duration_seconds REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_logs_source ON collection_logs(source);
CREATE INDEX IF NOT EXISTS idx_logs_status ON collection_logs(status);

-- =============================================
-- VIEWS FOR COMMON QUERIES
-- =============================================

-- Comprehensive match view with team names
CREATE VIEW IF NOT EXISTS v_matches_detailed AS
SELECT
    m.match_id,
    m.openligadb_match_id,
    m.season,
    m.matchday,
    m.match_datetime,
    ht.team_name as home_team,
    at.team_name as away_team,
    m.home_goals,
    m.away_goals,
    m.result,
    m.venue,
    m.attendance,
    m.temperature_celsius,
    m.humidity_percent,
    m.is_midweek,
    m.is_derby
FROM matches m
JOIN teams ht ON m.home_team_id = ht.team_id
JOIN teams at ON m.away_team_id = at.team_id;

-- Team performance summary per season
CREATE VIEW IF NOT EXISTS v_team_season_summary AS
SELECT
    t.team_name,
    ls.season,
    MAX(ls.position) as final_position,
    SUM(ls.matches_played) / COUNT(DISTINCT ls.matchday) as avg_matches,
    SUM(ls.wins) as total_wins,
    SUM(ls.draws) as total_draws,
    SUM(ls.losses) as total_losses,
    SUM(ls.goals_for) as total_goals_for,
    SUM(ls.goals_against) as total_goals_against,
    SUM(ls.points) as total_points
FROM league_standings ls
JOIN teams t ON ls.team_id = t.team_id
GROUP BY t.team_name, ls.season;
