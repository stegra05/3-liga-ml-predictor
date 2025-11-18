import React from 'react';
import { NavLink } from 'react-router-dom';
import './Navbar.css';

const Navbar = ({ stats }) => {
    return (
        <header className="app-header">
            <div className="container header-content">
                <div className="logo">
                    <span className="logo-icon">⚡</span>
                    <span className="logo-text">LIGA<span className="text-green">PREDICTOR</span></span>
                </div>
                <nav className="main-nav">
                    <NavLink to="/" className={({ isActive }) => isActive ? "active" : ""}>Dashboard</NavLink>
                    <NavLink to="/matches" className={({ isActive }) => isActive ? "active" : ""}>Matches</NavLink>
                    <NavLink to="/backtest" className={({ isActive }) => isActive ? "active" : ""}>Backtest Lab</NavLink>
                </nav>
                <div className="header-stats">
                    {stats && (
                        <>
                            <span className="stat-item">ACC: <span className="text-green">{(stats.accuracy * 100).toFixed(1)}%</span></span>
                            <span className="stat-item">ROI: <span className="text-green">+{stats.roi}%</span></span>
                        </>
                    )}
                </div>
            </div>
        </header>
    );
};

export default Navbar;
