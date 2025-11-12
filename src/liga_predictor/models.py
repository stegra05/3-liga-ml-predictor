"""
SQLAlchemy ORM Models for 3. Liga Database
All tables defined as declarative models for type-safe database operations
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Boolean, Float, Date, DateTime, Text,
    ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


# =============================================
# CORE ENTITIES
# =============================================

class Team(Base):
    """Team master table with standardized IDs"""
    __tablename__ = 'teams'

    team_id = Column(Integer, primary_key=True, autoincrement=True)
    openligadb_id = Column(Integer, unique=True, nullable=True)
    team_name = Column(String(100), nullable=False)
    team_name_short = Column(String(50), nullable=True)
    team_name_alt = Column(String(100), nullable=True)
    founded_year = Column(Integer, nullable=True)
    stadium_name = Column(String(100), nullable=True)
    stadium_capacity = Column(Integer, nullable=True)
    city = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    # Relationships
    home_matches = relationship('Match', foreign_keys='Match.home_team_id', back_populates='home_team')
    away_matches = relationship('Match', foreign_keys='Match.away_team_id', back_populates='away_team')
    match_statistics = relationship('MatchStatistic', back_populates='team')
    match_events = relationship('MatchEvent', back_populates='team')
    league_standings = relationship('LeagueStanding', back_populates='team')
    player_season_stats = relationship('PlayerSeasonStats', back_populates='team')
    team_ratings = relationship('TeamRating', back_populates='team')
    head_to_head_a = relationship('HeadToHead', foreign_keys='HeadToHead.team_a_id', back_populates='team_a')
    head_to_head_b = relationship('HeadToHead', foreign_keys='HeadToHead.team_b_id', back_populates='team_b')
    team_location = relationship('TeamLocation', uselist=False, back_populates='team')

    __table_args__ = (
        Index('idx_teams_name', 'team_name'),
        Index('idx_teams_openligadb', 'openligadb_id'),
    )


class Player(Base):
    """Player master table"""
    __tablename__ = 'players'

    player_id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String(100), nullable=False)
    known_name = Column(String(100), nullable=True)
    date_of_birth = Column(Date, nullable=True)
    nationality = Column(String(50), nullable=True)
    position = Column(String(20), nullable=True)  # GK, DF, MF, FW
    position_detail = Column(String(50), nullable=True)
    height_cm = Column(Integer, nullable=True)
    foot = Column(String(10), nullable=True)  # 'left', 'right', 'both'
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    # Relationships
    match_events = relationship('MatchEvent', back_populates='player')
    player_season_stats = relationship('PlayerSeasonStats', back_populates='player')
    player_match_stats = relationship('PlayerMatchStats', back_populates='player')

    __table_args__ = (
        Index('idx_players_name', 'full_name'),
    )


# =============================================
# MATCH DATA
# =============================================

class Match(Base):
    """Core match results and metadata"""
    __tablename__ = 'matches'

    match_id = Column(Integer, primary_key=True, autoincrement=True)
    openligadb_match_id = Column(Integer, unique=True, nullable=True)
    fotmob_match_id = Column(Integer, nullable=True)
    season = Column(String(9), nullable=False)  # Format: "2024-2025"
    matchday = Column(Integer, nullable=False)
    match_datetime = Column(DateTime, nullable=False)

    home_team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)
    away_team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)

    home_goals = Column(Integer, nullable=True)
    away_goals = Column(Integer, nullable=True)
    result = Column(String(1), nullable=True)  # 'H', 'D', 'A'

    # Match context
    is_finished = Column(Boolean, default=False)
    venue = Column(String(100), nullable=True)
    attendance = Column(Integer, nullable=True)
    referee = Column(String(100), nullable=True)

    # Weather conditions
    temperature_celsius = Column(Float, nullable=True)
    humidity_percent = Column(Float, nullable=True)
    wind_speed_kmh = Column(Float, nullable=True)
    precipitation_mm = Column(Float, nullable=True)
    weather_condition = Column(String(50), nullable=True)
    weather_source = Column(Text, nullable=True)  # 'meteostat', 'open_meteo', 'dwd'
    weather_confidence = Column(Float, nullable=True)  # Confidence score 0.0-1.0

    # Time context
    is_midweek = Column(Boolean, nullable=True)
    is_derby = Column(Boolean, nullable=True)
    competition_phase = Column(String(50), nullable=True)  # e.g., "regular", "playoff"

    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    # Relationships
    home_team = relationship('Team', foreign_keys=[home_team_id], back_populates='home_matches')
    away_team = relationship('Team', foreign_keys=[away_team_id], back_populates='away_matches')
    match_statistics = relationship('MatchStatistic', back_populates='match')
    match_events = relationship('MatchEvent', back_populates='match')
    betting_odds = relationship('BettingOdds', back_populates='match')
    team_ratings = relationship('TeamRating', back_populates='match')
    player_match_stats = relationship('PlayerMatchStats', back_populates='match')

    __table_args__ = (
        Index('idx_matches_season', 'season'),
        Index('idx_matches_datetime', 'match_datetime'),
        Index('idx_matches_home_team', 'home_team_id'),
        Index('idx_matches_away_team', 'away_team_id'),
        Index('idx_matches_fotmob', 'fotmob_match_id'),
    )


class MatchStatistic(Base):
    """Team performance statistics per match"""
    __tablename__ = 'match_statistics'

    stat_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey('matches.match_id'), nullable=False)
    team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)
    is_home = Column(Boolean, nullable=False)

    # Possession
    possession_percent = Column(Float, nullable=True)

    # Shooting
    shots_total = Column(Integer, nullable=True)
    shots_on_target = Column(Integer, nullable=True)
    shots_off_target = Column(Integer, nullable=True)
    shots_blocked = Column(Integer, nullable=True)
    big_chances = Column(Integer, nullable=True)
    big_chances_missed = Column(Integer, nullable=True)

    # Passing
    passes_total = Column(Integer, nullable=True)
    passes_accurate = Column(Integer, nullable=True)
    pass_accuracy_percent = Column(Float, nullable=True)
    key_passes = Column(Integer, nullable=True)
    crosses_total = Column(Integer, nullable=True)
    crosses_accurate = Column(Integer, nullable=True)
    long_balls_total = Column(Integer, nullable=True)
    long_balls_accurate = Column(Integer, nullable=True)

    # Defensive actions
    tackles_total = Column(Integer, nullable=True)
    tackles_won = Column(Integer, nullable=True)
    interceptions = Column(Integer, nullable=True)
    clearances = Column(Integer, nullable=True)
    blocked_shots = Column(Integer, nullable=True)

    # Duels and aerial
    duels_total = Column(Integer, nullable=True)
    duels_won = Column(Integer, nullable=True)
    aerials_total = Column(Integer, nullable=True)
    aerials_won = Column(Integer, nullable=True)

    # Discipline
    fouls_committed = Column(Integer, nullable=True)
    fouls_won = Column(Integer, nullable=True)
    yellow_cards = Column(Integer, nullable=True)
    red_cards = Column(Integer, nullable=True)

    # Set pieces
    corners = Column(Integer, nullable=True)
    offsides = Column(Integer, nullable=True)

    # Other
    touches = Column(Integer, nullable=True)
    dribbles_attempted = Column(Integer, nullable=True)
    dribbles_successful = Column(Integer, nullable=True)

    source = Column(String(50), nullable=True)  # 'fotmob', 'fbref', etc.
    has_complete_stats = Column(Boolean, default=False)

    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    # Relationships
    match = relationship('Match', back_populates='match_statistics')
    team = relationship('Team', back_populates='match_statistics')

    __table_args__ = (
        UniqueConstraint('match_id', 'team_id', name='uq_match_statistics_match_team'),
        Index('idx_match_stats_match', 'match_id'),
        Index('idx_match_stats_team', 'team_id'),
    )


class MatchEvent(Base):
    """Goals, cards, substitutions"""
    __tablename__ = 'match_events'

    event_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey('matches.match_id'), nullable=False)
    team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)

    event_type = Column(String(20), nullable=False)  # 'goal', 'yellow_card', 'red_card', 'substitution'
    minute = Column(Integer, nullable=True)
    minute_extra = Column(Integer, nullable=True)  # injury time

    player_id = Column(Integer, ForeignKey('players.player_id'), nullable=True)
    player_name = Column(String(100), nullable=True)

    # For goals
    is_penalty = Column(Boolean, nullable=True)
    is_own_goal = Column(Boolean, nullable=True)
    assist_player_id = Column(Integer, nullable=True)
    assist_player_name = Column(String(100), nullable=True)

    # For substitutions
    player_out_id = Column(Integer, nullable=True)
    player_out_name = Column(String(100), nullable=True)

    created_at = Column(DateTime, default=func.current_timestamp())

    # Relationships
    match = relationship('Match', back_populates='match_events')
    team = relationship('Team', back_populates='match_events')
    player = relationship('Player', back_populates='match_events')

    __table_args__ = (
        Index('idx_events_match', 'match_id'),
        Index('idx_events_type', 'event_type'),
    )


# =============================================
# LEAGUE DATA
# =============================================

class LeagueStanding(Base):
    """Historical league tables after each matchday"""
    __tablename__ = 'league_standings'

    standing_id = Column(Integer, primary_key=True, autoincrement=True)
    season = Column(String(9), nullable=False)
    matchday = Column(Integer, nullable=False)
    team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)

    position = Column(Integer, nullable=False)
    matches_played = Column(Integer, nullable=False)
    wins = Column(Integer, nullable=False)
    draws = Column(Integer, nullable=False)
    losses = Column(Integer, nullable=False)
    goals_for = Column(Integer, nullable=False)
    goals_against = Column(Integer, nullable=False)
    goal_difference = Column(Integer, nullable=False)
    points = Column(Integer, nullable=False)

    # Additional context
    form_last_5 = Column(String(5), nullable=True)  # e.g., "WDLWW"

    created_at = Column(DateTime, default=func.current_timestamp())

    # Relationships
    team = relationship('Team', back_populates='league_standings')

    __table_args__ = (
        UniqueConstraint('season', 'matchday', 'team_id', name='uq_league_standings_season_matchday_team'),
        Index('idx_standings_season', 'season'),
        Index('idx_standings_team', 'team_id'),
    )


class BettingOdds(Base):
    """Betting market data"""
    __tablename__ = 'betting_odds'

    odds_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey('matches.match_id'), nullable=False)

    bookmaker = Column(String(50), default='average')
    odds_home = Column(Float, nullable=True)
    odds_draw = Column(Float, nullable=True)
    odds_away = Column(Float, nullable=True)

    # Derived probabilities
    implied_prob_home = Column(Float, nullable=True)
    implied_prob_draw = Column(Float, nullable=True)
    implied_prob_away = Column(Float, nullable=True)

    # Timing
    odds_type = Column(String(20), default='closing')  # 'opening', 'closing'
    collected_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=func.current_timestamp())

    # Relationships
    match = relationship('Match', back_populates='betting_odds')

    __table_args__ = (
        UniqueConstraint('match_id', 'bookmaker', name='uq_betting_odds_match_bookmaker'),
        Index('idx_odds_match', 'match_id'),
    )


# =============================================
# PLAYER DATA
# =============================================

class PlayerSeasonStats(Base):
    """Player season statistics"""
    __tablename__ = 'player_season_stats'

    stat_id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey('players.player_id'), nullable=False)
    team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)
    season = Column(String(9), nullable=False)

    # Appearance data
    matches_played = Column(Integer, default=0)
    minutes_played = Column(Integer, default=0)
    starts = Column(Integer, default=0)

    # Scoring
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    penalties_scored = Column(Integer, default=0)

    # Discipline
    yellow_cards = Column(Integer, default=0)
    red_cards = Column(Integer, default=0)

    # Market value
    market_value_eur = Column(Integer, nullable=True)

    # Additional stats (from FBref if available)
    shots_total = Column(Integer, nullable=True)
    shots_on_target = Column(Integer, nullable=True)
    pass_accuracy_percent = Column(Float, nullable=True)
    tackles_won = Column(Integer, nullable=True)

    source = Column(String(50), nullable=True)  # 'fbref', etc.

    created_at = Column(DateTime, default=func.current_timestamp())

    # Relationships
    player = relationship('Player', back_populates='player_season_stats')
    team = relationship('Team', back_populates='player_season_stats')

    __table_args__ = (
        UniqueConstraint('player_id', 'team_id', 'season', name='uq_player_season_stats_player_team_season'),
        Index('idx_player_stats_season', 'season'),
        Index('idx_player_stats_team', 'team_id'),
    )


class PlayerMatchStats(Base):
    """Detailed player performance statistics for each match"""
    __tablename__ = 'player_match_stats'

    stat_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey('matches.match_id'), nullable=False)
    player_id = Column(Integer, ForeignKey('players.player_id'), nullable=False)
    team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)

    # Basic info
    started = Column(Boolean, default=False)
    minutes_played = Column(Integer, default=0)
    shirt_number = Column(Integer, nullable=True)
    position = Column(String(20), nullable=True)  # Position played in this match

    # Scoring
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    penalties_scored = Column(Integer, default=0)
    penalties_attempted = Column(Integer, default=0)

    # Shooting
    shots_total = Column(Integer, default=0)
    shots_on_target = Column(Integer, default=0)

    # Passing
    passes_total = Column(Integer, default=0)
    passes_completed = Column(Integer, default=0)
    pass_accuracy_percent = Column(Float, nullable=True)
    key_passes = Column(Integer, default=0)

    # Defensive
    tackles_total = Column(Integer, default=0)
    tackles_won = Column(Integer, default=0)
    interceptions = Column(Integer, default=0)
    clearances = Column(Integer, default=0)

    # Discipline
    yellow_cards = Column(Integer, default=0)
    red_cards = Column(Integer, default=0)
    fouls_committed = Column(Integer, default=0)
    fouls_drawn = Column(Integer, default=0)

    # Other
    offsides = Column(Integer, default=0)
    crosses = Column(Integer, default=0)
    dribbles_attempted = Column(Integer, default=0)
    dribbles_successful = Column(Integer, default=0)

    # Goalkeeper-specific stats
    saves = Column(Integer, default=0)
    goals_conceded = Column(Integer, default=0)
    clean_sheet = Column(Boolean, default=False)

    source = Column(String(50), default='fbref')
    created_at = Column(DateTime, default=func.current_timestamp())

    # Relationships
    match = relationship('Match', back_populates='player_match_stats')
    player = relationship('Player', back_populates='player_match_stats')

    __table_args__ = (
        UniqueConstraint('match_id', 'player_id', 'source', name='uq_player_match_stats_match_player_source'),
        Index('idx_player_match_stats_match', 'match_id'),
        Index('idx_player_match_stats_player', 'player_id'),
        Index('idx_player_match_stats_team', 'team_id'),
    )


class AdvancedMatchStats(Base):
    """Advanced analytics metrics (xG, progressive actions, etc.)"""
    __tablename__ = 'advanced_match_stats'

    stat_id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey('matches.match_id'), nullable=False)
    team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)
    is_home = Column(Boolean, nullable=False)

    # Expected metrics
    xg = Column(Float, nullable=True)  # Expected goals
    xg_against = Column(Float, nullable=True)
    xa = Column(Float, nullable=True)  # Expected assists
    npxg = Column(Float, nullable=True)  # Non-penalty expected goals

    # Progressive actions
    progressive_passes = Column(Integer, nullable=True)
    progressive_carries = Column(Integer, nullable=True)
    progressive_pass_distance = Column(Float, nullable=True)

    # Pressure and possession
    pressures = Column(Integer, nullable=True)
    pressure_success_rate = Column(Float, nullable=True)
    pressures_att_third = Column(Integer, nullable=True)
    pressures_mid_third = Column(Integer, nullable=True)
    pressures_def_third = Column(Integer, nullable=True)

    # Pass types
    through_balls = Column(Integer, nullable=True)
    switches = Column(Integer, nullable=True)
    ground_passes = Column(Integer, nullable=True)
    low_passes = Column(Integer, nullable=True)
    high_passes = Column(Integer, nullable=True)

    # Shot creation
    shot_creating_actions = Column(Integer, nullable=True)
    goal_creating_actions = Column(Integer, nullable=True)

    source = Column(String(50), default='fbref')
    created_at = Column(DateTime, default=func.current_timestamp())

    __table_args__ = (
        UniqueConstraint('match_id', 'team_id', 'source', name='uq_advanced_match_stats_match_team_source'),
        Index('idx_advanced_stats_match', 'match_id'),
        Index('idx_advanced_stats_team', 'team_id'),
    )


# =============================================
# DERIVED FEATURES & RATINGS
# =============================================

class TeamRating(Base):
    """Team rating systems (Elo, Pi-ratings, etc.)"""
    __tablename__ = 'team_ratings'

    rating_id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)
    match_id = Column(Integer, ForeignKey('matches.match_id'), nullable=True)  # NULL for season start

    season = Column(String(9), nullable=False)
    matchday = Column(Integer, nullable=False)

    # Rating systems
    elo_rating = Column(Float, nullable=True)
    pi_rating = Column(Float, nullable=True)

    # Form metrics
    points_last_5 = Column(Integer, nullable=True)
    points_last_10 = Column(Integer, nullable=True)
    goals_scored_last_5 = Column(Float, nullable=True)
    goals_conceded_last_5 = Column(Float, nullable=True)

    # Streak data
    current_win_streak = Column(Integer, default=0)
    current_unbeaten_streak = Column(Integer, default=0)
    current_loss_streak = Column(Integer, default=0)

    created_at = Column(DateTime, default=func.current_timestamp())

    # Relationships
    team = relationship('Team', back_populates='team_ratings')
    match = relationship('Match', back_populates='team_ratings')

    __table_args__ = (
        Index('idx_ratings_team_season', 'team_id', 'season'),
        Index('idx_ratings_matchday', 'season', 'matchday'),
    )


class HeadToHead(Base):
    """Head-to-head historical records"""
    __tablename__ = 'head_to_head'

    h2h_id = Column(Integer, primary_key=True, autoincrement=True)
    team_a_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)
    team_b_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)

    # Overall record (team_a perspective)
    total_matches = Column(Integer, default=0)
    team_a_wins = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    team_b_wins = Column(Integer, default=0)

    # Home/away splits
    team_a_home_wins = Column(Integer, default=0)
    team_a_away_wins = Column(Integer, default=0)

    last_updated = Column(DateTime, default=func.current_timestamp())

    # Relationships
    team_a = relationship('Team', foreign_keys=[team_a_id], back_populates='head_to_head_a')
    team_b = relationship('Team', foreign_keys=[team_b_id], back_populates='head_to_head_b')

    __table_args__ = (
        UniqueConstraint('team_a_id', 'team_b_id', name='uq_head_to_head_team_a_team_b'),
    )


class TeamLocation(Base):
    """Team location data (lat/lon for geocoding)"""
    __tablename__ = 'team_locations'

    team_id = Column(Integer, ForeignKey('teams.team_id'), primary_key=True)
    team_name = Column(Text, nullable=True)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    source = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    # Relationships
    team = relationship('Team', back_populates='team_location')


# =============================================
# DATA COLLECTION METADATA
# =============================================

class CollectionLog(Base):
    """Track data collection runs"""
    __tablename__ = 'collection_logs'

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False)  # 'openligadb', 'fbref', etc.
    collection_type = Column(String(50), nullable=False)  # 'match_data', 'player_stats', etc.

    season = Column(String(9), nullable=True)
    matchday = Column(Integer, nullable=True)

    status = Column(String(20), nullable=False)  # 'success', 'partial', 'failed'
    records_collected = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)

    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)

    created_at = Column(DateTime, default=func.current_timestamp())

    __table_args__ = (
        Index('idx_logs_source', 'source'),
        Index('idx_logs_status', 'status'),
    )


class FBrefCollectionLog(Base):
    """Track FBref scraping sessions and data quality"""
    __tablename__ = 'fbref_collection_log'

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    season = Column(String(9), nullable=True)
    collection_type = Column(String(50), nullable=True)  # 'season_stats', 'match_stats', 'player_stats'

    # Statistics
    items_attempted = Column(Integer, default=0)
    items_collected = Column(Integer, default=0)
    items_failed = Column(Integer, default=0)

    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)

    # Status
    status = Column(String(20), nullable=True)  # 'success', 'partial', 'failed'
    error_message = Column(Text, nullable=True)

    # Details
    teams_processed = Column(Integer, nullable=True)
    matches_processed = Column(Integer, nullable=True)
    players_processed = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=func.current_timestamp())

    __table_args__ = (
        Index('idx_fbref_log_season', 'season'),
        Index('idx_fbref_log_status', 'status'),
    )

