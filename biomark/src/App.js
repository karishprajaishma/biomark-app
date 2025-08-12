import React, { useState, useEffect } from 'react';
import { Upload, Activity, Brain, Heart, Droplet, Moon, AlertTriangle, CheckCircle, Play, Square, Download, RefreshCw, Zap, TrendingUp, Shield } from 'lucide-react';
import "./App.css"


function App() {
  const [apiStatus, setApiStatus] = useState('disconnected');
  const [dashboardData, setDashboardData] = useState({});
  const [biomarkerHistory, setBiomarkerHistory] = useState([]);
  const [activeInterventions, setActiveInterventions] = useState([]);
  const [csvFile, setCsvFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [biomarkerData, setBiomarkerData] = useState({
    user_id: 'user_001',
    glucose: '',
    hrv: '',
    cortisol: '',
    sleep_quality: '',
    neuro_fatigue: ''
  });

  const API_BASE = 'http://localhost:8000';

  // Check API status
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
          setApiStatus('connected');
          fetchDashboardData();
        }
      } catch (error) {
        setApiStatus('disconnected');
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const response = await fetch(`${API_BASE}/dashboard/data`);
      if (response.ok) {
        const data = await response.json();
        setDashboardData(data);
        if (data.biomarker_history && Array.isArray(data.biomarker_history)) {
          setBiomarkerHistory(prev => {
            // Merge API data with local data, avoiding duplicates
            const combined = [...prev];
            data.biomarker_history.forEach(apiEntry => {
              if (!combined.find(localEntry => 
                localEntry.timestamp === apiEntry.timestamp && 
                localEntry.user_id === apiEntry.user_id
              )) {
                combined.push(apiEntry);
              }
            });
            return combined.slice(-10); // Keep last 10 entries
          });
        }
        if (data.active_interventions) {
          setActiveInterventions(prev => {
            // Merge new interventions with existing ones
            const combined = [...prev];
            data.active_interventions.forEach(apiIntervention => {
              if (!combined.find(localIntervention => 
                localIntervention.timestamp === apiIntervention.timestamp &&
                localIntervention.biomarker === apiIntervention.biomarker
              )) {
                combined.unshift(apiIntervention); // Add to beginning
              }
            });
            return combined.slice(0, 20); // Keep max 20 interventions
          });
        }
      }
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    }
  };

  const handleCsvUpload = async () => {
    if (!csvFile) return;

    const formData = new FormData();
    formData.append('file', csvFile);

    try {
      setUploadStatus('uploading');
      const response = await fetch(`${API_BASE}/upload-csv`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        setUploadStatus(`success: ${result.message}`);
        fetchDashboardData();
      } else {
        setUploadStatus('error: Upload failed');
      }
    } catch (error) {
      setUploadStatus('error: Connection failed');
    }
  };

  const handleBiomarkerSubmit = async () => {
    try {
      console.log("biomark : ", biomarkerData);
      
      // Create entry with timestamp
      const newEntry = {
        ...biomarkerData,
        timestamp: new Date().toLocaleString(),
        glucose: parseFloat(biomarkerData.glucose) || 0,
        hrv: parseFloat(biomarkerData.hrv) || 0,
        cortisol: parseFloat(biomarkerData.cortisol) || 0,
        sleep_quality: parseFloat(biomarkerData.sleep_quality) || 0,
        neuro_fatigue: parseFloat(biomarkerData.neuro_fatigue) || 0
      };

      // Add to biomarker history immediately
      setBiomarkerHistory(prev => [...prev, newEntry].slice(-5));

      const response = await fetch(`${API_BASE}/process-biomarkers`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(biomarkerData)
      });

      if (response.ok) {
        const result = await response.json();
        console.log("result : ", result);
        if (result.interventions) {
          // Add timestamp to interventions and show latest first
          const timestampedInterventions = result.interventions.map(intervention => ({
            ...intervention,
            timestamp: intervention.timestamp || new Date().toLocaleString()
          }));
          setActiveInterventions(prev => [...timestampedInterventions, ...prev]);
        }
        fetchDashboardData();
      } else {
        // Even if API fails, keep the local entry
        console.log("API call failed, but data added locally");
      }

      // Clear form
      setBiomarkerData({
        ...biomarkerData,
        glucose: '',
        hrv: '',
        cortisol: '',
        sleep_quality: '',
        neuro_fatigue: ''
      });
    } catch (error) {
      console.error('Error processing biomarkers:', error);
      // Even on error, keep the local entry
      const newEntry = {
        ...biomarkerData,
        timestamp: new Date().toLocaleString(),
        glucose: parseFloat(biomarkerData.glucose) || 0,
        hrv: parseFloat(biomarkerData.hrv) || 0,
        cortisol: parseFloat(biomarkerData.cortisol) || 0,
        sleep_quality: parseFloat(biomarkerData.sleep_quality) || 0,
        neuro_fatigue: parseFloat(biomarkerData.neuro_fatigue) || 0
      };
      setBiomarkerHistory(prev => [...prev, newEntry].slice(-5));
    }
  };

  const generateSampleData = async () => {
    try {
      const response = await fetch(`${API_BASE}/generate-sample-data`, {
        method: 'POST'
      });
      if (response.ok) {
        const result = await response.json();
        setUploadStatus(`Generated sample data: ${result.message}`);
        fetchDashboardData();
      }
    } catch (error) {
      setUploadStatus('Error generating sample data');
    }
  };

  const startSimulation = async () => {
    try {
      const response = await fetch(`${API_BASE}/simulate-realtime`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ duration_minutes: 5 })
      });

      if (response.ok) {
        setSimulationRunning(true);
        const interval = setInterval(() => {
          fetchDashboardData();
        }, 2000);

        setTimeout(() => {
          setSimulationRunning(false);
          clearInterval(interval);
        }, 300000);
      }
    } catch (error) {
      console.error('Error starting simulation:', error);
    }
  };

  const stopSimulation = async () => {
    try {
      await fetch(`${API_BASE}/stop-simulation`, { method: 'POST' });
      setSimulationRunning(false);
    } catch (error) {
      console.error('Error stopping simulation:', error);
    }
  };

  const getBiomarkerIcon = (biomarker) => {
    switch (biomarker) {
      case 'glucose': return <Droplet size={20} />;
      case 'hrv': return <Heart size={20} />;
      case 'cortisol': return <Activity size={20} />;
      case 'sleep_quality': return <Moon size={20} />;
      case 'neuro_fatigue': return <Brain size={20} />;
      default: return <Activity size={20} />;
    }
  };

  const handleSystemReset = async () => {
    try {
      await fetch(`${API_BASE}/reset`, { method: 'DELETE' });
      setActiveInterventions([]);
      setDashboardData({});
      setBiomarkerHistory([]);
      setUploadStatus('System reset successfully');
      fetchDashboardData();
    } catch (error) {
      setUploadStatus('Error resetting system');
    }
  };

  return (
    <div className="app-container">
      {/* Animated Background */}
      <div className="background-animation">
        <div className="floating-orb orb-1"></div>
        <div className="floating-orb orb-2"></div>
        <div className="floating-orb orb-3"></div>
      </div>

      <div className="main-content">
        {/* Header */}
        <div className="header-card">
          <div className="header-content">
            <div className="header-left">
              <div className="logo-icon">
                <Zap size={32} />
              </div>
              <div className="header-text">
                <h1 className="main-title">BioSyncDEX™</h1>
                <p className="subtitle">Real-time Biomarker Monitoring & ML Intervention System</p>
              </div>
            </div>
            <div className={`status-indicator ${apiStatus === 'connected' ? 'connected' : 'disconnected'}`}>
              <div className="status-dot"></div>
              <span className="status-text">
                {apiStatus === 'connected' ? 'SYSTEM ONLINE' : 'DISCONNECTED'}
              </span>
            </div>
          </div>
        </div>

        <div className="dashboard-grid">
          {/* Left Column - Data Input */}
          <div className="left-column">
            {/* CSV Upload */}
            <div className="card">
              <h2 className="card-title">
                <Upload size={24} />
                Data Training Center
              </h2>

              <div className="card-content">
                <div className="input-group">
                  <label className="input-label">Upload Training Dataset</label>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => setCsvFile(e.target.files[0])}
                    className="file-input"
                  />
                </div>

                <div className="button-group">
                  <button
                    onClick={handleCsvUpload}
                    disabled={!csvFile}
                    className="btn btn-primary"
                  >
                    Train Models
                  </button>
                  <button
                    onClick={generateSampleData}
                    className="btn btn-secondary"
                  >
                    Generate Sample
                  </button>
                </div>

                {uploadStatus && (
                  <div className={`status-message ${uploadStatus.includes('success') ? 'success' : 'error'}`}>
                    {uploadStatus}
                  </div>
                )}
              </div>
            </div>

            {/* Manual Data Input */}
            <div className="card">
              <h2 className="card-title">
                <TrendingUp size={24} />
                Live Data Input
              </h2>

              <div className="card-content">
                <div className="biomarker-inputs">
                  {[
                    { key: 'glucose', label: 'Glucose', unit: 'mg/dL', placeholder: '70-180', icon: Droplet },
                    { key: 'hrv', label: 'Heart Rate Variability', unit: 'ms', placeholder: '20-80', icon: Heart },
                    { key: 'cortisol', label: 'Cortisol', unit: 'μg/dL', placeholder: '6-23', icon: Activity },
                    { key: 'sleep_quality', label: 'Sleep Quality', unit: '0-100', placeholder: '0-100', icon: Moon },
                    { key: 'neuro_fatigue', label: 'Neuro Fatigue', unit: '0-100', placeholder: '0-100', icon: Brain }
                  ].map((field) => {
                    const IconComponent = field.icon;
                    return (
                      <div key={field.key} className="input-group">
                        <label className="input-label">
                          <IconComponent size={16} />
                          {field.label} ({field.unit})
                        </label>
                        <input
                          type="number"
                          step="0.01"
                          value={biomarkerData[field.key]}
                          onChange={(e) => setBiomarkerData({ ...biomarkerData, [field.key]: e.target.value })}
                          className="biomarker-input"
                          placeholder={field.placeholder}
                        />
                      </div>
                    );
                  })}
                </div>

                <button
                  onClick={handleBiomarkerSubmit}
                  disabled={apiStatus !== 'connected'}
                  className="btn btn-success btn-large"
                >
                  <Activity size={20} />
                  ANALYZE BIOMARKERS
                </button>
              </div>
            </div>
          </div>

          {/* Center Column - Dashboard */}
          <div className="center-column">
            {/* Status Cards */}
            <div className="status-cards">
              <div className="stat-card blue">
                <div className="stat-content">
                  <div className="stat-text">
                    <p className="stat-label">Total Processed</p>
                    <p className="stat-value">{dashboardData?.total_processed || 0}</p>
                  </div>
                  <div className="stat-icon blue">
                    <Activity size={32} />
                  </div>
                </div>
              </div>

              <div className="stat-card purple">
                <div className="stat-content">
                  <div className="stat-text">
                    <p className="stat-label">ML Models</p>
                    <p className="stat-value">{dashboardData?.models_available?.length || 0}</p>
                  </div>
                  <div className="stat-icon purple">
                    <Brain size={32} />
                  </div>
                </div>
              </div>
            </div>

            {/* Recent Biomarkers */}
            <div className="card biomarkers-card">
              <h2 className="card-title">
                <TrendingUp size={24} />
                Live Biomarker Feed
              </h2>
                
              <div className="biomarker-feed">
                {biomarkerHistory.length > 0 && biomarkerHistory.slice(-5).map((data, index) => (
                  <div key={index} className="biomarker-entry">
                    <div className="entry-header">
                      <div className="user-info">
                        <div className="user-avatar">{data.user_id?.slice(-3) || 'USR'}</div>
                        <div className="user-details">
                          <span className="user-id">{data.user_id || 'Unknown User'}</span>
                          <p className="timestamp">{data.timestamp || 'No timestamp'}</p>
                        </div>
                      </div>
                    </div>
                    <div className="biomarker-grid">
                      {[
                        { key: 'glucose', icon: Droplet, value: data.glucose, color: '#ef4444' },
                        { key: 'hrv', icon: Heart, value: data.hrv, color: '#ec4899' },
                        { key: 'cortisol', icon: Activity, value: data.cortisol, color: '#eab308' },
                        { key: 'sleep_quality', icon: Moon, value: data.sleep_quality, color: '#3b82f6' },
                        { key: 'neuro_fatigue', icon: Brain, value: data.neuro_fatigue, color: '#8b5cf6' }
                      ].map((biomarker) => {
                        const IconComponent = biomarker.icon;
                        return (
                          <div key={biomarker.key} className="biomarker-item">
                            <div className="biomarker-icon" style={{ backgroundColor: biomarker.color }}>
                              <IconComponent size={20} />
                            </div>
                            <div className="biomarker-value">
                              {biomarker.value ? Number(biomarker.value).toFixed(1) : 'N/A'}
                            </div>
                            <div className="biomarker-label">{biomarker.key}</div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}

                {(!biomarkerHistory || biomarkerHistory.length === 0) && (
                  <div className="empty-state">
                    <Activity size={64} />
                    <p className="empty-title">No biomarker data available</p>
                    <p className="empty-subtitle">Upload CSV or enter manual data to begin</p>
                  </div>
                )}
              </div>
            </div>

            {/* Simulation Controls */}
            <div className="card">
              <h2 className="card-title">
                <Play size={24} />
                Real-time Simulation
              </h2>

              <div className="simulation-controls">
                <button
                  onClick={startSimulation}
                  disabled={simulationRunning || apiStatus !== 'connected'}
                  className="btn btn-success"
                >
                  <Play size={20} />
                  START SIM
                </button>

                <button
                  onClick={stopSimulation}
                  disabled={!simulationRunning}
                  className="btn btn-danger"
                >
                  <Square size={20} />
                  STOP
                </button>

                <button
                  onClick={fetchDashboardData}
                  className="btn btn-secondary btn-icon"
                >
                  <RefreshCw size={20} />
                </button>
              </div>

              {simulationRunning && (
                <div className="simulation-status">
                  <div className="simulation-indicator">
                    <div className="pulse-dot"></div>
                    <span>SIMULATION ACTIVE</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right Column - Interventions & Status */}
          <div className="right-column">
            {/* Active Interventions */}
            <div className="card interventions-card">
              <h2 className="card-title">
                <Shield size={24} />
                Active Interventions
                {activeInterventions.length > 0 && (
                  <span className="intervention-count">{activeInterventions.length}</span>
                )}
              </h2>

              <div className="interventions-list">
                {activeInterventions.slice().reverse().map((intervention, index) => (
                  <div key={index} className={`intervention-card ${intervention.priority || 'medium'}`}>
                    <div className="intervention-header">
                      <div className="intervention-info">
                        <div className="intervention-icon">
                          {getBiomarkerIcon(intervention.biomarker)}
                        </div>
                        <div className="intervention-details">
                          <div className="intervention-biomarker">
                            {intervention.biomarker ? intervention.biomarker.replace('_', ' ').toUpperCase() : 'UNKNOWN'}
                          </div>
                          <div className="intervention-meta">
                            Value: {intervention.value || 'N/A'} | {intervention.status || 'Pending'}
                          </div>
                        </div>
                      </div>
                      <span className={`priority-badge ${intervention.priority || 'medium'}`}>
                        {intervention.priority || 'medium'}
                      </span>
                    </div>
                    <div className="intervention-content">
                      <p className="intervention-text">{intervention.intervention || 'No intervention specified'}</p>
                      <p className="intervention-time">
                        <Activity size={12} />
                        {intervention.timestamp || 'No timestamp'}
                      </p>
                    </div>
                  </div>
                ))}

                {activeInterventions.length === 0 && (
                  <div className="no-interventions">
                    <div className="success-icon">
                      <CheckCircle size={48} />
                    </div>
                    <p className="success-title">All Systems Optimal</p>
                    <p className="success-subtitle">All biomarkers within normal ranges</p>
                  </div>
                )}
              </div>
            </div>

            {/* System Status & Export */}
            <div className="bottom-cards">
              <div className="card">
                <h2 className="card-title">
                  <Shield size={20} />
                  System Status
                </h2>

                <div className="status-items">
                  <div className="status-item">
                    <span>Engine Status</span>
                    <span className={`status-value ${dashboardData?.engine_status === 'running' ? 'running' : 'idle'}`}>
                      {dashboardData?.engine_status || 'idle'}
                    </span>
                  </div>

                  <div className="status-item">
                    <span>Active Alerts</span>
                    <span className="status-value alert">{activeInterventions.length}</span>
                  </div>

                  <div className="status-item">
                    <span>ML Models</span>
                    <span className="status-value models">{dashboardData?.models_available?.length || 0}</span>
                  </div>

                  {dashboardData?.models_available && dashboardData.models_available.length > 0 && (
                    <div className="models-list">
                      <p className="models-label">Trained Models:</p>
                      <div className="model-tags">
                        {dashboardData.models_available.map((model, index) => (
                          <span key={index} className="model-tag">{model}</span>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="last-update">
                    <span>Last Update:</span>
                    <span className="update-time">{dashboardData?.current_time || 'Never'}</span>
                  </div>
                </div>
              </div>

              {/* Export Controls */}
              <div className="card">
                <h2 className="card-title">
                  <Download size={20} />
                  Export Center
                </h2>

                <div className="export-controls">
                  <a
                    href={`${API_BASE}/export-results`}
                    className="btn btn-primary btn-export"
                  >
                    <Download size={16} />
                    Export Interventions
                  </a>

                  <a
                    href={`${API_BASE}/download-sample-csv`}
                    className="btn btn-secondary btn-export"
                  >
                    <Download size={16} />
                    Download Sample CSV
                  </a>

                  <button
                    onClick={handleSystemReset}
                    className="btn btn-danger btn-export"
                  >
                    <RefreshCw size={16} />
                    Reset System
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Floating Refresh Button */}
        <button
          onClick={fetchDashboardData}
          className="floating-refresh"
        >
          <RefreshCw size={24} />
        </button>
      </div>
    </div>
  );
}

export default App;