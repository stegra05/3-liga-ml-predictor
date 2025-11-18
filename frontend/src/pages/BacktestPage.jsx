import React from 'react';
import './BacktestPage.css';

const BacktestPage = ({ stats }) => {
    if (!stats) return null;

    return (
        <div className="backtest-page container">
            <div className="page-header">
                <h1 className="page-title">Backtest Lab</h1>
                <div className="model-info">
                    <span className="model-badge">{stats.modelName}</span>
                    <span className="last-trained">Last trained: {stats.lastTrained}</span>
                </div>
            </div>

            <div className="kpi-grid">
                <div className="kpi-card">
                    <span className="kpi-label">Accuracy</span>
                    <span className="kpi-value text-green">{(stats.accuracy * 100).toFixed(1)}%</span>
                </div>
                <div className="kpi-card">
                    <span className="kpi-label">ROI</span>
                    <span className="kpi-value text-green">+{stats.roi}%</span>
                </div>
                <div className="kpi-card">
                    <span className="kpi-label">Total Bets</span>
                    <span className="kpi-value">142</span>
                </div>
                <div className="kpi-card">
                    <span className="kpi-label">Win Rate</span>
                    <span className="kpi-value">48%</span>
                </div>
            </div>

            <div className="chart-section">
                <h3 className="chart-title">Equity Curve (Season 2024/25)</h3>
                <div className="equity-chart-placeholder">
                    {/* Placeholder for Equity Curve */}
                    <svg viewBox="0 0 800 300" className="chart-svg">
                        <defs>
                            <linearGradient id="gradientGreen" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="var(--color-accent-green)" stopOpacity="0.2" />
                                <stop offset="100%" stopColor="var(--color-accent-green)" stopOpacity="0" />
                            </linearGradient>
                        </defs>
                        <path
                            d="M0,250 L50,240 L100,200 L150,210 L200,150 L250,160 L300,100 L350,110 L400,80 L450,90 L500,40 L550,50 L600,20 L700,10 L800,5"
                            fill="url(#gradientGreen)"
                            stroke="var(--color-accent-green)"
                            strokeWidth="3"
                        />
                        <line x1="0" y1="250" x2="800" y2="250" stroke="var(--color-border)" strokeDasharray="5,5" />
                    </svg>
                </div>
            </div>

            <div className="confusion-matrix-section">
                <h3 className="chart-title">Confusion Matrix</h3>
                <div className="matrix-grid">
                    <div className="matrix-cell header"></div>
                    <div className="matrix-cell header">Pred Home</div>
                    <div className="matrix-cell header">Pred Draw</div>
                    <div className="matrix-cell header">Pred Away</div>

                    <div className="matrix-cell header">True Home</div>
                    <div className="matrix-cell value high">45</div>
                    <div className="matrix-cell value low">12</div>
                    <div className="matrix-cell value low">8</div>

                    <div className="matrix-cell header">True Draw</div>
                    <div className="matrix-cell value medium">15</div>
                    <div className="matrix-cell value medium">22</div>
                    <div className="matrix-cell value low">10</div>

                    <div className="matrix-cell header">True Away</div>
                    <div className="matrix-cell value low">5</div>
                    <div className="matrix-cell value low">11</div>
                    <div className="matrix-cell value high">38</div>
                </div>
            </div>
        </div>
    );
};

export default BacktestPage;
