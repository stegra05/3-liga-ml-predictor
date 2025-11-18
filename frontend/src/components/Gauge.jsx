import React from 'react';
import './Gauge.css';

const Gauge = ({ value, label }) => {
    // Value should be 0-100
    const radius = 40;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (value / 100) * circumference;

    let color = 'var(--color-text-muted)';
    if (value >= 70) color = 'var(--color-accent-green)';
    else if (value >= 40) color = 'var(--color-accent-orange)';
    else color = 'var(--color-accent-blue)';

    return (
        <div className="gauge-container">
            <svg className="gauge-ring" width="120" height="120">
                <circle
                    className="gauge-circle-bg"
                    stroke="var(--color-bg-gunmetal)"
                    strokeWidth="8"
                    fill="transparent"
                    r={radius}
                    cx="60"
                    cy="60"
                />
                <circle
                    className="gauge-circle-progress"
                    stroke={color}
                    strokeWidth="8"
                    fill="transparent"
                    r={radius}
                    cx="60"
                    cy="60"
                    style={{
                        strokeDasharray: `${circumference} ${circumference}`,
                        strokeDashoffset: offset,
                        transform: 'rotate(-90deg)',
                        transformOrigin: '50% 50%',
                        transition: 'stroke-dashoffset 1s ease-out'
                    }}
                />
            </svg>
            <div className="gauge-content">
                <span className="gauge-value" style={{ color }}>{value}%</span>
                <span className="gauge-label">{label}</span>
            </div>
        </div>
    );
};

export default Gauge;
