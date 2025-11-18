import React from 'react';
import Gauge from '../components/Gauge';
import './HeroSection.css';

const HeroSection = ({ match }) => {
    if (!match) return null;

    // Determine confidence color
    const confidence = match.prediction?.confidence ? Math.round(match.prediction.confidence * 100) : 0;

    return (
        <section className="hero-section">
            <div className="hero-bg-video">
                {/* Placeholder for video loop */}
                <div className="hero-noise-overlay"></div>
            </div>

            <div className="container hero-content">
                <div className="hero-header">
                    <span className="hero-badge">Match of the Week</span>
                    <h1 className="hero-title">Prediction Insight</h1>
                </div>

                <div className="match-faceoff">
                    <div className="team home-team">
                        <div className="team-logo-large placeholder-logo">{match.homeTeam?.substring(0, 2)}</div>
                        <h2 className="team-name">{match.homeTeam}</h2>
                    </div>

                    <div className="match-center">
                        <Gauge value={confidence} label="Confidence" />
                        <div className="prediction-verdict">
                            <span className="verdict-label">Projected Winner</span>
                            <span className="verdict-value text-green">
                                {match.prediction?.verdict === 'HOME_WIN' ? match.homeTeam :
                                    match.prediction?.verdict === 'AWAY_WIN' ? match.awayTeam : 'Draw'}
                            </span>
                        </div>
                    </div>

                    <div className="team away-team">
                        <div className="team-logo-large placeholder-logo">{match.awayTeam?.substring(0, 2)}</div>
                        <h2 className="team-name">{match.awayTeam}</h2>
                    </div>
                </div>

                <div className="hero-footer">
                    <p className="hero-insight">
                        <span className="insight-icon">⚡</span>
                        {match.homeTeam} has a +140 Elo advantage and {match.awayTeam} is traveling 450km on 3 days rest.
                    </p>
                </div>
            </div>
        </section>
    );
};

export default HeroSection;
