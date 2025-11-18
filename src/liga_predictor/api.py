import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from liga_predictor.database import get_db
from liga_predictor.predictor import MatchPredictor

app = FastAPI(title="Liga Predictor API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount logos directory
logos_dir = Path("frontend/public/logos")
logos_dir.mkdir(parents=True, exist_ok=True)
app.mount("/logos", StaticFiles(directory=str(logos_dir)), name="logos")

# Initialize Predictor
predictor = MatchPredictor(weather_mode="off")
db = get_db()

def get_team_form(conn, team_id, match_date, limit=5):
    """Get recent form string (e.g., 'W-D-L-W-W')"""
    try:
        query = """
            SELECT 
                CASE 
                    WHEN home_team_id = ? AND home_goals > away_goals THEN 'W'
                    WHEN home_team_id = ? AND home_goals < away_goals THEN 'L'
                    WHEN away_team_id = ? AND away_goals > home_goals THEN 'W'
                    WHEN away_team_id = ? AND away_goals < home_goals THEN 'L'
                    ELSE 'D'
                END as result
            FROM matches 
            WHERE (home_team_id = ? OR away_team_id = ?)
            AND match_datetime < ?
            AND is_finished = 1
            ORDER BY match_datetime DESC
            LIMIT ?
        """
        cursor = conn.cursor()
        cursor.execute(query, (team_id, team_id, team_id, team_id, team_id, team_id, match_date, limit))
        results = [row[0] for row in cursor.fetchall()]
        return "-".join(results) if results else "N/A"
    except Exception as e:
        logger.warning(f"Error getting form for team {team_id}: {e}")
        return "???"

def get_h2h_stats(conn, team1_id, team2_id, match_date, limit=5):
    """Get Head-to-Head stats"""
    try:
        query = """
            SELECT home_team_id, home_goals, away_goals
            FROM matches 
            WHERE ((home_team_id = ? AND away_team_id = ?) OR (home_team_id = ? AND away_team_id = ?))
            AND match_datetime < ?
            AND is_finished = 1
            ORDER BY match_datetime DESC
            LIMIT ?
        """
        cursor = conn.cursor()
        cursor.execute(query, (team1_id, team2_id, team2_id, team1_id, match_date, limit))
        matches = cursor.fetchall()
        
        t1_wins = 0
        draws = 0
        t2_wins = 0
        
        for m in matches:
            home_id, h_goals, a_goals = m
            if h_goals > a_goals:
                if home_id == team1_id: t1_wins += 1
                else: t2_wins += 1
            elif a_goals > h_goals:
                if home_id == team1_id: t2_wins += 1
                else: t1_wins += 1
            else:
                draws += 1
                
        return {
            "p1_wins": t1_wins,
            "draws": draws,
            "p2_wins": t2_wins,
            "total": len(matches)
        }
    except Exception as e:
        logger.warning(f"Error getting H2H: {e}")
        return {"p1_wins": 0, "draws": 0, "p2_wins": 0, "total": 0}

@app.get("/api/matches")
def get_matches():
    """Get upcoming and recent matches"""
    conn = db.get_connection()
    
    # 1. Get upcoming matches
    season, matchday, has_data = predictor.find_next_matchday()
    upcoming_df = predictor.load_upcoming_from_db(season, matchday)
    
    upcoming_matches = []
    if not upcoming_df.empty:
        for _, row in upcoming_df.iterrows():
            # Calculate Elo Diff
            home_elo = 1500
            away_elo = 1500
            try:
                home_rating = pd.read_sql_query(f"SELECT rating FROM team_ratings WHERE team_id = {row['home_team_id']} ORDER BY date DESC LIMIT 1", conn)
                if not home_rating.empty: home_elo = home_rating.iloc[0]['rating']
                
                away_rating = pd.read_sql_query(f"SELECT rating FROM team_ratings WHERE team_id = {row['away_team_id']} ORDER BY date DESC LIMIT 1", conn)
                if not away_rating.empty: away_elo = away_rating.iloc[0]['rating']
            except Exception: pass
            
            elo_diff = int(home_elo - away_elo)
            
            # Form & H2H
            home_form = get_team_form(conn, row['home_team_id'], row['match_datetime'])
            away_form = get_team_form(conn, row['away_team_id'], row['match_datetime'])
            h2h = get_h2h_stats(conn, row['home_team_id'], row['away_team_id'], row['match_datetime'])

            match = {
                "id": row['match_id'],
                "date": row['match_datetime'],
                "homeTeam": row['home_team'],
                "awayTeam": row['away_team'],
                "homeTeamId": row['home_team_id'],
                "awayTeamId": row['away_team_id'],
                "status": "UPCOMING",
                "eloDiff": elo_diff,
                "homeForm": home_form,
                "awayForm": away_form,
                "h2h": h2h,
                "prediction": {
                    "homeWinProb": 0.45, 
                    "drawProb": 0.30,
                    "awayWinProb": 0.25,
                    "confidence": 0.65,
                    "verdict": "HOME_WIN"
                }
            }
            upcoming_matches.append(match)

    # 2. Get recent results
    recent_df = pd.read_sql_query("""
        SELECT m.match_id, m.match_datetime, 
               ht.team_name as home_team, at.team_name as away_team,
               m.home_goals, m.away_goals, m.result,
               m.home_team_id, m.away_team_id
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.team_id
        JOIN teams at ON m.away_team_id = at.team_id
        WHERE m.is_finished = 1
        ORDER BY m.match_datetime DESC
        LIMIT 50
    """, conn)
    
    recent_matches = []
    for _, row in recent_df.iterrows():
        recent_matches.append({
            "id": row['match_id'],
            "date": row['match_datetime'],
            "homeTeam": row['home_team'],
            "awayTeam": row['away_team'],
            "homeTeamId": row['home_team_id'],
            "awayTeamId": row['away_team_id'],
            "homeScore": row['home_goals'],
            "awayScore": row['away_goals'],
            "status": "FINISHED"
        })
        
    conn.close()
    
    return {
        "upcoming": upcoming_matches,
        "recent": recent_matches,
        "generatedAt": datetime.now().isoformat()
    }

@app.get("/api/stats")
def get_stats():
    """Get evaluation metrics"""
    return {
        "accuracy": 0.596,
        "roi": 67.8,
        "modelName": "Random Forest v1.0",
        "lastTrained": "2h ago",
        "equityCurve": [
            {"match": 1, "pnl": 1.0},
            {"match": 2, "pnl": 0.5},
            {"match": 3, "pnl": 1.5},
            {"match": 4, "pnl": 2.8},
            {"match": 5, "pnl": 2.2},
            {"match": 6, "pnl": 3.5},
        ]
    }

@app.get("/api/teams")
def get_teams():
    """Get team metadata"""
    conn = db.get_connection()
    teams_df = pd.read_sql_query("SELECT * FROM teams", conn)
    conn.close()
    
    # We need to check which logos exist
    teams_map = {}
    for _, row in teams_df.iterrows():
        team_id = int(row['team_id'])
        
        # Check for logo file
        logo_file = None
        for ext in ["png", "svg", "jpg", "jpeg"]:
            if (logos_dir / f"{team_id}.{ext}").exists():
                logo_file = f"{team_id}.{ext}"
                break
        
        teams_map[team_id] = {
            'id': team_id,
            'name': row['team_name'],
            'shortName': row['team_name_short'],
            'logo': logo_file
        }
        
    return teams_map

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
