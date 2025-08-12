import React, { useState, useEffect } from 'react';

const BiomarkerData = () => {
    const [data, setData] = useState([]);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetch('http://localhost:8000/dashboard/data');
                const jsonData = await response.json();
                setData(jsonData.recent_biomarkers);
            } catch (error) {
                console.error('Error fetching biomarker data:', error);
            }
        };

        fetchData();
    }, []);

    return (
        <div>
            <h2>Biomarker Data</h2>
            <ul>
                {data.map((biomarker) => (
                    <li key={biomarker.user_id}>
                        <strong>User ID:</strong> {biomarker.user_id}
                        <br />
                        <strong>Glucose:</strong> {biomarker.glucose}
                        <br />
                        <strong>HRV:</strong> {biomarker.hrv}
                        <br />
                        <strong>Cortisol:</strong> {biomarker.cortisol}
                        <br />
                        <strong>Sleep Quality:</strong> {biomarker.sleep_quality}
                        <br />
                        <strong>Neuro Fatigue:</strong> {biomarker.neuro_fatigue}
                    </li>
                ))}
            </ul>
        </div>
    );
};