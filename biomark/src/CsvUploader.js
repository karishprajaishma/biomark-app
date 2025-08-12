import React, { useState } from 'react';

const CsvUploader = () => {
    const [file, setFile] = useState(null);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleUpload = async () => {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8000/upload-csv', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                console.log('CSV uploaded successfully');
            } else {
                console.error('Error uploading CSV:', response.statusText);
            }
        } catch (error) {
            console.error('Error uploading CSV:', error);
        }
    };

    return (
        <div>
            <input type="file" onChange={handleFileChange} />
            <button onClick={handleUpload}>Upload CSV</button>
        </div>
    );
};

export default CsvUploader;