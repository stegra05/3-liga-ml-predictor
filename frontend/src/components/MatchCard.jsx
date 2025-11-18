import React from 'react';
import './MatchCard.css';

const MatchCard = ({ match, onClick }) => {
    // Calculate probabilities for the bar chart
    const homeProb = match.prediction?.homeWinProb * 100 || 33;
    const drawProb = match.prediction?.drawProb * 100 || 33;
    const awayProb = match.prediction?.awayWinProb * 100 || 34;

    const date = new Date(match.date);
    const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const dateStr = date.toLocaleDateString([], { weekday: 'short', day: 'numeric', month: 'short' });

    return (
        <div className="match-card" onClick={() => onClick(match)}>
            <div className="match-card-header">
                <span className="match-date">{dateStr} • {timeStr}</span>
                {match.status === 'FINISHED' && <span className="match-status">FT</span>}
            </div>

            <div className="match-teams">
                <div className="team-row">
                    <div className="team-info">
                        {match.homeTeamLogo ? (
                            <img src={`/logos/${match.homeTeamLogo}`} alt={match.homeTeam} className="team-logo-small" />
                        ) : (
                            <div className="team-logo-small placeholder-logo">{match.homeTeam?.substring(0, 2)}</div>
                        )}
                        <span className="team-name-small">{match.homeTeam}</span>
                    </div>
                    <span className="team-score">{match.homeScore !== undefined ? match.homeScore : '-'}</span>
                </div>
                <div className="team-row">
                    <div className="team-info">
                        {match.awayTeamLogo ? (
                            <img src={`/logos/${match.awayTeamLogo}`} alt={match.awayTeam} className="team-logo-small" />
                        ) : (
                            <div className="team-logo-small placeholder-logo">{match.awayTeam?.substring(0, 2)}</div>
                        )}
                        <span className="team-name-small">{match.awayTeam}</span>
                    </div>
                    <span className="team-score">{match.awayScore !== undefined ? match.awayScore : '-'}</span>
                </div>
            </div>

            {match.status === 'UPCOMING' && (
                <div className="match-probabilities">
                    <div className="prob-bar">
                        <div className="prob-segment home" style={{ width: `${homeProb}%` }}></div>
                        <div className="prob-segment draw" style={{ width: `${drawProb}%` }}></div>
                        <div className="prob-segment away" style={{ width: `${awayProb}%` }}></div>
                    </div>
                    <div className="prob-labels">
                        <span>{(homeProb).toFixed(0)}%</span>
                        <span>{(drawProb).toFixed(0)}%</span>
                        <span>{(awayProb).toFixed(0)}%</span>
                    </div>
                </div>
            )}

            <div className="match-card-hover">
                <div className="hover-stat">
                    <span className="hover-label">Elo Diff</span>
                    <span className={`hover-value ${match.eloDiff > 0 ? 'text-green' : 'text-orange'}`}>
                        {match.eloDiff > 0 ? '+' : ''}{match.eloDiff}
                    </span>
                </div>
                <div className="hover-stat">
                    <span className="hover-label">Form</span>
                    <span className="hover-value">{match.homeForm} vs {match.awayForm}</span>
                </div>
            </div>
        </div>
    );
};

export default MatchCard;
