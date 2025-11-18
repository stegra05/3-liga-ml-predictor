import React from 'react';
import HeroSection from '../sections/HeroSection';
import MatchGrid from '../sections/MatchGrid';

const Dashboard = ({ matches, onMatchClick }) => {
    // Select the most interesting match for the hero (highest confidence or first upcoming)
    const heroMatch = matches.upcoming.length > 0 ? matches.upcoming[0] : null;

    return (
        <main>
            <HeroSection match={heroMatch} />

            <div className="container">
                <MatchGrid
                    title="Upcoming Fixtures"
                    matches={matches.upcoming.slice(0, 6)}
                    onMatchClick={onMatchClick}
                />

                <MatchGrid
                    title="Recent Results"
                    matches={matches.recent.slice(0, 6)}
                    onMatchClick={onMatchClick}
                />
            </div>
        </main>
    );
};

export default Dashboard;
