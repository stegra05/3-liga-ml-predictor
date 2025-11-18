import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import MatchesPage from './pages/MatchesPage';
import BacktestPage from './pages/BacktestPage';
import AnalysisPanel from './sections/AnalysisPanel';
import './App.css';

function App() {
  const [matches, setMatches] = useState({ upcoming: [], recent: [] });
  const [stats, setStats] = useState(null);
  const [teams, setTeams] = useState({});
  const [loading, setLoading] = useState(true);
  const [selectedMatch, setSelectedMatch] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [matchesRes, statsRes, teamsRes] = await Promise.all([
          fetch('/data/matches.json'),
          fetch('/data/stats.json'),
          fetch('/data/teams.json')
        ]);

        const matchesData = await matchesRes.json();
        const statsData = await statsRes.json();
        const teamsData = await teamsRes.json();

        // Enrich matches with team metadata (logos)
        const enrichMatch = (match) => ({
          ...match,
          homeTeamLogo: teamsData[match.homeTeamId]?.logo,
          awayTeamLogo: teamsData[match.awayTeamId]?.logo
        });

        setMatches({
          upcoming: matchesData.upcoming.map(enrichMatch),
          recent: matchesData.recent.map(enrichMatch)
        });
        setStats(statsData);
        setTeams(teamsData);
      } catch (error) {
        console.error("Failed to fetch data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner"></div>
        <p>Initializing Surgical Analytics...</p>
      </div>
    );
  }

  return (
    <BrowserRouter>
      <div className="app-container">
        <Navbar stats={stats} />

        <Routes>
          <Route path="/" element={<Dashboard matches={matches} onMatchClick={setSelectedMatch} />} />
          <Route path="/matches" element={<MatchesPage matches={matches} onMatchClick={setSelectedMatch} />} />
          <Route path="/backtest" element={<BacktestPage stats={stats} />} />
        </Routes>

        {selectedMatch && (
          <AnalysisPanel
            match={selectedMatch}
            onClose={() => setSelectedMatch(null)}
          />
        )}

        <footer className="app-footer">
          <div className="container">
            <p>&copy; 2025 3. Liga ML Predictor. Powered by Surgical Sports Analytics.</p>
          </div>
        </footer>
      </div>
    </BrowserRouter>
  );
}

export default App;
