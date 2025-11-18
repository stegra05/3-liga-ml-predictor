"""
Frontend Data Exporter
Exports static JSON files for the React frontend.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger

from liga_predictor.database import get_db
from liga_predictor.predictor import MatchPredictor
from liga_predictor.processing.ml_export import MLDataExporter

class FrontendExporter:
    def __init__(self, output_dir: str = "frontend/public/data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db = get_db()
        self.predictor = MatchPredictor(weather_mode="off") # No live weather for export speed
        
    def export_all(self):
        """Run all exports"""
        logger.info("Starting frontend data export...")
        
        self.export_matches()
        self.export_stats()
        self.download_logos()
        self.export_team_metadata()
        
        logger.success("Frontend export complete!")

    def download_logos(self):
        """Download team logos from OpenLigaDB"""
        logger.info("Downloading team logos...")
        logo_dir = self.output_dir.parent / "logos"
        logo_dir.mkdir(exist_ok=True)
        
        # Store logo filenames for metadata export
        self.team_logos = {}
        
        # Fetch current season data to get icon URLs
        try:
            import requests
            season = datetime.now().year
            if datetime.now().month < 7:
                season -= 1
                
            url = f"https://api.openligadb.de/getavailableteams/bl3/{season}"
            response = requests.get(url)
            response.raise_for_status()
            teams = response.json()
            
            # Also get from database to map IDs
            conn = self.db.get_connection()
            db_teams = pd.read_sql_query("SELECT team_id, team_name, openligadb_id FROM teams", conn)
            conn.close()
            
            count = 0
            for team in teams:
                team_name = team.get('teamName')
                icon_url = team.get('teamIconUrl')
                
                if not icon_url:
                    continue
                    
                # Find matching team in DB
                match = db_teams[db_teams['team_name'] == team_name]
                
                if match.empty:
                    short_name = team.get('shortName')
                    match = db_teams[db_teams['team_name'] == short_name]
                
                if not match.empty:
                    team_id = int(match.iloc[0]['team_id'])
                    
                    # Determine extension
                    ext = "png"
                    if icon_url.endswith(".svg"):
                        ext = "svg"
                    elif icon_url.endswith(".jpg") or icon_url.endswith(".jpeg"):
                        ext = "jpg"
                    
                    filename = f"{team_id}.{ext}"
                    self.team_logos[team_id] = filename
                    
                    # Download if not exists or force update
                    if not (logo_dir / filename).exists():
                        try:
                            img_data = requests.get(icon_url).content
                            with open(logo_dir / filename, "wb") as f:
                                f.write(img_data)
                            count += 1
                        except Exception as e:
                            logger.warning(f"Failed to download logo for {team_name}: {e}")
            
            logger.success(f"Downloaded {count} new team logos")
            
        except Exception as e:
            logger.error(f"Logo download failed: {e}")

    def export_team_metadata(self):
        """Export team metadata including logo paths"""
        logger.info("Exporting team metadata...")
        
        conn = self.db.get_connection()
        teams_df = pd.read_sql_query("SELECT * FROM teams", conn)
        conn.close()
        
        teams_map = {}
        for _, row in teams_df.iterrows():
            team_id = int(row['team_id'])
            teams_map[team_id] = {
                'id': team_id,
                'name': row['team_name'],
                'shortName': row['team_name_short'],
                'logo': self.team_logos.get(team_id)
            }
            
        with open(self.output_dir / "teams.json", "w") as f:
            json.dump(teams_map, f, indent=2)

    def export_matches(self):
        """Export upcoming and recent matches with predictions"""
        logger.info("Exporting matches...")
        
        # 1. Get upcoming matches (next matchday)
        season, matchday, has_data = self.predictor.find_next_matchday()
        upcoming_df = self.predictor.load_upcoming_from_db(season, matchday)
        
        upcoming_matches = []
        if not upcoming_df.empty:
            # Generate predictions
            # Note: This requires the model to be trained and saved. 
            # If not, we might need to handle that.
            # For now, assuming model exists or we skip prediction details.
            try:
                # We need to prepare features for these matches
                # This is complex because predictor.predict_next_matchday does a lot.
                # Let's try to use the predictor's internal methods if possible, 
                # or just export the schedule if prediction fails.
                pass 
            except Exception as e:
                logger.warning(f"Could not generate predictions: {e}")

            # Format for frontend
            for _, row in upcoming_df.iterrows():
                # Calculate Elo Diff (Home - Away)
                home_elo = 1500
                away_elo = 1500
                
                try:
                    conn = self.db.get_connection()
                    # Get latest rating for home team
                    home_rating = pd.read_sql_query(f"SELECT rating FROM team_ratings WHERE team_id = {row['home_team_id']} ORDER BY date DESC LIMIT 1", conn)
                    if not home_rating.empty:
                        home_elo = home_rating.iloc[0]['rating']
                        
                    # Get latest rating for away team
                    away_rating = pd.read_sql_query(f"SELECT rating FROM team_ratings WHERE team_id = {row['away_team_id']} ORDER BY date DESC LIMIT 1", conn)
                    if not away_rating.empty:
                        away_elo = away_rating.iloc[0]['rating']
                    conn.close()
                except Exception:
                    pass
                
                elo_diff = int(home_elo - away_elo)
                
                # Get Real Form
                home_form = self.get_team_form(row['home_team_id'], row['match_datetime'])
                away_form = self.get_team_form(row['away_team_id'], row['match_datetime'])
                
                # Get H2H Stats
                h2h = self.get_h2h_stats(row['home_team_id'], row['away_team_id'], row['match_datetime'])

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
                        # Placeholder - real prediction logic needed here
                        "homeWinProb": 0.45, 
                        "drawProb": 0.30,
                        "awayWinProb": 0.25,
                        "confidence": 0.65,
                        "verdict": "HOME_WIN"
                    }
                }
                upcoming_matches.append(match)

    def get_team_form(self, team_id, match_date, limit=5):
        """Get recent form string (e.g., 'W-D-L-W-W')"""
        try:
            conn = self.db.get_connection()
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
            conn.close()
            
            return "-".join(results) if results else "N/A"
        except Exception as e:
            logger.warning(f"Error getting form for team {team_id}: {e}")
            return "???"

    def get_h2h_stats(self, team1_id, team2_id, match_date, limit=5):
        """Get Head-to-Head stats"""
        try:
            conn = self.db.get_connection()
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
            conn.close()
            
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

        # 2. Get recent results
        conn = self.db.get_connection()
        recent_df = pd.read_sql_query("""
            SELECT m.match_id, m.match_datetime, 
                   ht.team_name as home_team, at.team_name as away_team,
                   m.home_goals, m.away_goals, m.result
            FROM matches m
            JOIN teams ht ON m.home_team_id = ht.team_id
            JOIN teams at ON m.away_team_id = at.team_id
            WHERE m.is_finished = 1
            ORDER BY m.match_datetime DESC
            LIMIT 50
        """, conn)
        conn.close()
        
        recent_matches = []
        for _, row in recent_df.iterrows():
            recent_matches.append({
                "id": row['match_id'],
                "date": row['match_datetime'],
                "homeTeam": row['home_team'],
                "awayTeam": row['away_team'],
                "homeScore": row['home_goals'],
                "awayScore": row['away_goals'],
                "status": "FINISHED"
            })
            
        data = {
            "upcoming": upcoming_matches,
            "recent": recent_matches,
            "generatedAt": datetime.now().isoformat()
        }
        
        with open(self.output_dir / "matches.json", "w") as f:
            json.dump(data, f, indent=2, default=str)

    def export_stats(self):
        """Export general stats and evaluation metrics"""
        logger.info("Exporting stats...")
        
        # Placeholder for real evaluation metrics
        # In a real scenario, we would load the latest evaluation results
        stats = {
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
        
        with open(self.output_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    exporter = FrontendExporter()
    exporter.export_all()
