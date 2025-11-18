import React, { useState } from 'react';
import MatchGrid from '../sections/MatchGrid';
import './MatchesPage.css';

const MatchesPage = ({ matches, onMatchClick }) => {
    const [filter, setFilter] = useState('all'); // all, upcoming, finished

    const filteredMatches = () => {
        if (filter === 'upcoming') return matches.upcoming;
        if (filter === 'finished') return matches.recent;
        return [...matches.upcoming, ...matches.recent];
    };

    return (
        <div className="matches-page container">
            <div className="page-header">
                <h1 className="page-title">Match Center</h1>
                <div className="filter-controls">
                    <button
                        className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
                        onClick={() => setFilter('all')}
                    >
                        All Matches
                    </button>
                    <button
                        className={`filter-btn ${filter === 'upcoming' ? 'active' : ''}`}
                        onClick={() => setFilter('upcoming')}
                    >
                        Upcoming
                    </button>
                    <button
                        className={`filter-btn ${filter === 'finished' ? 'active' : ''}`}
                        onClick={() => setFilter('finished')}
                    >
                        Finished
                    </button>
                </div>
            </div>

            <MatchGrid
                title={filter === 'all' ? 'All Matches' : filter === 'upcoming' ? 'Upcoming Fixtures' : 'Match Results'}
                matches={filteredMatches()}
                onMatchClick={onMatchClick}
            />
        </div>
    );
};

export default MatchesPage;
