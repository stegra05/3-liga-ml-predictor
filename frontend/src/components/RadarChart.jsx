import React from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Legend } from 'recharts';

const TeamRadarChart = ({ homeTeam, awayTeam, stats }) => {
    // Mock data if real stats are not available yet
    // In a real scenario, we would pass these stats from the backend
    const data = [
        { subject: 'Attack', A: 120, B: 110, fullMark: 150 },
        { subject: 'Defense', A: 98, B: 130, fullMark: 150 },
        { subject: 'Possession', A: 86, B: 130, fullMark: 150 },
        { subject: 'Form', A: 99, B: 100, fullMark: 150 },
        { subject: 'H2H', A: 85, B: 90, fullMark: 150 },
        { subject: 'Motivation', A: 65, B: 85, fullMark: 150 },
    ];

    return (
        <div style={{ width: '100%', height: 300 }}>
            <ResponsiveContainer>
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
                    <PolarGrid stroke="var(--color-border)" />
                    <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--color-text-muted)', fontSize: 12 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 150]} tick={false} axisLine={false} />
                    <Radar
                        name={homeTeam}
                        dataKey="A"
                        stroke="var(--color-accent-green)"
                        fill="var(--color-accent-green)"
                        fillOpacity={0.3}
                    />
                    <Radar
                        name={awayTeam}
                        dataKey="B"
                        stroke="var(--color-accent-blue)"
                        fill="var(--color-accent-blue)"
                        fillOpacity={0.3}
                    />
                    <Legend wrapperStyle={{ color: 'var(--color-text-primary)' }} />
                </RadarChart>
            </ResponsiveContainer>
        </div>
    );
};

export default TeamRadarChart;
