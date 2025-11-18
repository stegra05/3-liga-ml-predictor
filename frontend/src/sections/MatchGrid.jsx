import React from 'react';
import MatchCard from '../components/MatchCard';
import './MatchGrid.css';

const MatchGrid = ({ title, matches, onMatchClick }) => {
    if (!matches || matches.length === 0) return null;

    return (
        <section className="match-grid-section">
            <h3 className="section-title">{title}</h3>
            <div className="match-grid">
                {matches.map(match => (
                    <MatchCard key={match.id} match={match} onClick={onMatchClick} />
                ))}
            </div>
        </section>
    );
};

export default MatchGrid;
