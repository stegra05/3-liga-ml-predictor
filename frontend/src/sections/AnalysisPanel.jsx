import React from 'react';
import TeamRadarChart from '../components/RadarChart';
import FormGuide from '../components/FormGuide';
import './AnalysisPanel.css';

const AnalysisPanel = ({ match, onClose }) => {
    if (!match) return null;

    return (
        <div className="analysis-panel-overlay" onClick={onClose}>
            <div className="analysis-panel" onClick={e => e.stopPropagation()}>
                <button className="close-btn" onClick={onClose}>&times;</button>

                <div className="panel-header">
                    <h2 className="panel-title">Match Analysis</h2>
                    <div className="stat-row">
                        <span>Form (Last 5)</span>
                        <div className="form-comparison">
                            <FormGuide formString={match.homeForm} />
                            <span className="vs-divider">vs</span>
                            <FormGuide formString={match.awayForm} />
                        </div>
                    </div>
                    <div className="stat-row">
                        <span>Head-to-Head</span>
                        <span>
                            {match.h2h ? `${match.h2h.p1_wins}W - ${match.h2h.draws}D - ${match.h2h.p2_wins}L` : 'N/A'}
                        </span>
                    </div>

                    <div className="panel-content">
                        <div className="analysis-section">
                            <h3 className="analysis-subtitle">Team Comparison</h3>
                            <span className="radar-label left-bottom">Fatigue</span>
                            <span className="radar-label left-top">H2H</span>
                        </div>
                    </div>

                    <div className="analysis-section">
                        <h3 className="analysis-subtitle">Travel & Conditions</h3>
                        <div className="conditions-grid">
                            <div className="condition-card">
                                <span className="condition-icon">📍</span>
                                <span className="condition-value">450km</span>
                                <span className="condition-label">Travel Distance</span>
                            </div>
                            <div className="condition-card">
                                <span className="condition-icon">🌧️</span>
                                <span className="condition-value">8°C</span>
                                <span className="condition-label">Heavy Rain</span>
                            </div>
                            <div className="condition-card">
                                <span className="condition-icon">💨</span>
                                <span className="condition-value">25km/h</span>
                                <span className="condition-label">Wind Speed</span>
                            </div>
                        </div>
                    </div>

                    <div className="analysis-section">
                        <h3 className="analysis-subtitle">AI Explanation</h3>
                        <p className="ai-text">
                            The model heavily weighted <span className="highlight">Home Form (Last 5)</span> and <span className="highlight">Travel Distance</span> for this prediction.
                            {match.homeTeam} has a significant advantage in recent momentum.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AnalysisPanel;
