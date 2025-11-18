import React from 'react';
import './FormGuide.css';

const FormGuide = ({ formString }) => {
    // formString format: "W-D-L-W-W"
    const results = formString ? formString.split('-') : [];

    return (
        <div className="form-guide">
            {results.map((result, index) => (
                <div
                    key={index}
                    className={`form-badge ${result === 'W' ? 'win' : result === 'D' ? 'draw' : 'loss'}`}
                    title={result === 'W' ? 'Win' : result === 'D' ? 'Draw' : 'Loss'}
                >
                    {result}
                </div>
            ))}
        </div>
    );
};

export default FormGuide;
