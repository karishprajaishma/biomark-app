from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import json
import time
import random
import asyncio
import threading
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import csv
import io
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Pydantic Models
class BiometricDataModel(BaseModel):
    user_id: str
    timestamp: Optional[str] = None
    glucose: float = Field(..., description="Blood glucose in mg/dL")
    hrv: float = Field(..., description="Heart rate variability in ms")
    cortisol: float = Field(..., description="Cortisol level in μg/dL")
    sleep_quality: int = Field(..., ge=0, le=100, description="Sleep quality score 0-100")
    neuro_fatigue: int = Field(..., ge=0, le=100, description="Neurological fatigue score 0-100")

class InterventionModel(BaseModel):
    user_id: str
    biomarker: str
    value: float
    status: str
    intervention: str
    timestamp: str
    priority: str = "medium"

class ProcessingResponse(BaseModel):
    success: bool
    message: str
    interventions: List[InterventionModel]
    processed_data: Dict[str, Any]

class StatusResponse(BaseModel):
    engine_status: str
    total_processed: int
    active_interventions: int
    models_trained: List[str]
    last_update: str

# Core Classes (adapted from original)
class BiomarkerMLEngine:
    """ML-based prediction system for biomarker status"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.thresholds = {
            'glucose': {'normal': (70, 140), 'warning': (140, 170), 'critical': 170},
            'hrv': {'normal': (30, 60), 'warning': (20, 30), 'critical': 20},
            'cortisol': {'normal': (6, 23), 'warning': (23, 30), 'critical': 30},
            'sleep_quality': {'normal': (70, 100), 'warning': (50, 70), 'critical': 50},
            'neuro_fatigue': {'normal': (0, 40), 'warning': (40, 60), 'critical': 60}
        }
        self.is_trained = False

    def prepare_user_data(self, csv_data: List[Dict]) -> Dict[str, pd.DataFrame]:
        """Prepare user-specific biomarker data for ML training"""
        user_data = {}
        
        for biomarker in ['glucose', 'hrv', 'cortisol', 'sleep_quality', 'neuro_fatigue']:
            biomarker_records = []
            
            for record in csv_data:
                if biomarker in record:
                    biomarker_records.append({
                        'user_id': record['user_id'],
                        'biomarker': biomarker,
                        'value': float(record[biomarker]),
                        'timestamp': pd.to_datetime(record['timestamp'])
                    })
            
            if biomarker_records:
                df = pd.DataFrame(biomarker_records)
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['days_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.days
                df['status'] = df['value'].apply(lambda x: self._get_status_label(biomarker, x))
                user_data[biomarker] = df
                
        return user_data
    
    def _get_status_label(self, biomarker: str, value: float) -> int:
        """Convert biomarker value to status label (0=normal, 1=warning, 2=critical)"""
        thresholds = self.thresholds[biomarker]
        
        if biomarker in ['glucose', 'cortisol', 'neuro_fatigue']:
            if value > thresholds['critical']:
                return 2
            elif value > thresholds['normal'][1]:
                return 1
            else:
                return 0
        else:
            if value < thresholds['critical']:
                return 2
            elif value < thresholds['normal'][0]:
                return 1
            else:
                return 0
    
    async def train_models(self, csv_data: List[Dict]):
        """Train logistic regression models for each biomarker"""
        user_data = self.prepare_user_data(csv_data)
        
        for biomarker, df in user_data.items():
            if len(df) < 10:
                continue
                
            features = ['value', 'hour', 'day_of_week', 'days_since_start']
            X = df[features].values
            y = df['status'].values
            
            if len(np.unique(y)) < 2:
                continue
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            if len(np.unique(y)) > 2:
                model = LogisticRegression(multi_class='ovr', random_state=42, max_iter=1000)
            else:
                model = LogisticRegression(random_state=42, max_iter=1000)
            
            model.fit(X_scaled, y)
            
            self.models[biomarker] = model
            self.scalers[biomarker] = scaler
        
        self.is_trained = True
        return len(self.models)
    
    def predict_status(self, biomarker: str, value: float, timestamp: datetime) -> tuple:
        """Predict biomarker status using ML model"""
        if not self.is_trained or biomarker not in self.models:
            return self._fallback_rule_prediction(biomarker, value)
        
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        days_since_start = 1
        
        features = np.array([[value, hour, day_of_week, days_since_start]])
        
        scaler = self.scalers[biomarker]
        features_scaled = scaler.transform(features)
        
        model = self.models[biomarker]
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        status_map = {0: "normal", 1: "warning", 2: "critical"}
        predicted_status = status_map.get(prediction, "normal")
        confidence = max(probability)
        
        return predicted_status, confidence
    
    def _fallback_rule_prediction(self, biomarker: str, value: float) -> tuple:
        """Fallback rule-based prediction when ML model unavailable"""
        status_label = self._get_status_label(biomarker, value)
        status_map = {0: "normal", 1: "warning", 2: "critical"}
        return status_map[status_label], 0.8

class BiomarkerRuleEngine:
    """Enhanced rule-based intervention system with ML predictions"""
    
    def __init__(self):
        self.ml_engine = BiomarkerMLEngine()
        self.intervention_rules = {
            'glucose': {
                'high': "Reduce carb load + 5g berberine supplement",
                'moderate': "Light exercise + hydration boost",
                'critical': "URGENT: Contact healthcare provider - severe hyperglycemia"
            },
            'hrv': {
                'low': "Breathing exercises + magnesium supplement",
                'critical': "Stress management protocol + immediate rest"
            },
            'cortisol': {
                'high': "Adaptogenic herbs + meditation protocol",
                'critical': "URGENT: Severe stress response - seek medical attention"
            },
            'sleep_quality': {
                'low': "Melatonin + blue light blocking",
                'critical': "Sleep hygiene overhaul + possible sleep study"
            },
            'neuro_fatigue': {
                'high': "B-complex vitamins + cognitive break",
                'critical': "Complete cognitive rest + neurological assessment"
            }
        }
    
    async def train_ml_models(self, csv_data: List[Dict]):
        """Train ML models for prediction"""
        return await self.ml_engine.train_models(csv_data)
    
    def evaluate_biomarker(self, biomarker: str, value: float, timestamp: datetime = None) -> Optional[InterventionModel]:
        """Apply ML prediction + rules to determine intervention"""
        if biomarker not in self.ml_engine.thresholds:
            return None
        
        if timestamp is None:
            timestamp = datetime.now()
            
        predicted_status, confidence = self.ml_engine.predict_status(biomarker, value, timestamp)
        
        intervention = None
        priority = "low"
        
        if predicted_status == "critical":
            intervention = self.intervention_rules[biomarker].get('critical')
            priority = "critical"
        elif predicted_status == "warning":
            if biomarker in ['glucose']:
                intervention = self.intervention_rules[biomarker].get('high')
                priority = "high"
            else:
                intervention = self.intervention_rules[biomarker].get('low', 
                             self.intervention_rules[biomarker].get('high'))
                priority = "medium"
        
        if intervention:
            return InterventionModel(
                user_id="user_001",
                biomarker=biomarker,
                value=value,
                status=f"{predicted_status} (ML: {confidence:.2f})",
                intervention=intervention,
                timestamp=timestamp.strftime("%H:%M:%S"),
                priority=priority
            )
        return None

class BioSyncDataProcessor:
    """Data normalization and processing layer"""
    
    def __init__(self):
        self.data_buffer = []
        self.processed_data = pd.DataFrame()
    
    def normalize_data(self, raw_data: Dict) -> BiometricDataModel:
        """Convert raw sensor data to standardized format using scikit-learn"""
        try:
            numeric_fields = ['glucose', 'hrv', 'cortisol', 'sleep_quality', 'neuro_fatigue']
            values = []
            
            for field in numeric_fields:
                if field in raw_data and raw_data[field] is not None:
                    values.append(float(raw_data[field]))
                else:
                    raise ValueError(f"Missing or invalid value for {field}")
            
            data_array = np.array(values).reshape(1, -1)
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(data_array).flatten()
            
            return BiometricDataModel(
                user_id=str(raw_data.get('user_id', 'unknown')).strip(),
                timestamp=raw_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                glucose=float(normalized_data[0]),
                hrv=float(normalized_data[1]),
                cortisol=float(normalized_data[2]),
                sleep_quality=float(normalized_data[3]),
                neuro_fatigue=float(normalized_data[4])
            )
            
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error normalizing data: {e}")
    
    def store_data(self, biometric_data: BiometricDataModel):
        """Store normalized data for analysis"""
        self.data_buffer.append(biometric_data.dict())
        if len(self.data_buffer) > 100:
            self.data_buffer.pop(0)

# Global engine instance
biosync_engine = None

class BioSyncEngine:
    """Main biofeedback engine orchestrating all components"""
    
    def __init__(self):
        self.rule_engine = BiomarkerRuleEngine()
        self.processor = BioSyncDataProcessor()
        self.active_interventions = []
        self.is_running = False
        self.csv_data = []
        self.csv_index = 0
        self.intervention_log = []
    
    async def load_data_from_csv(self, csv_content: str):
        """Load biomarker data from CSV content and train ML models"""
        try:
            # Parse CSV content
            csv_reader = csv.DictReader(io.StringIO(csv_content))
            self.csv_data = list(csv_reader)
            self.csv_index = 0
            
            if self.csv_data:
                models_count = await self.rule_engine.train_ml_models(self.csv_data)
                return {
                    "success": True,
                    "message": f"Loaded {len(self.csv_data)} records and trained {models_count} ML models",
                    "records_count": len(self.csv_data),
                    "models_trained": models_count
                }
            else:
                return {"success": False, "message": "No valid data found in CSV"}
                
        except Exception as e:
            return {"success": False, "message": f"Error loading CSV: {str(e)}"}
    
    async def process_biomarker_data(self, raw_data: Dict):
        """Main processing pipeline"""
        try:
            # Normalize data
            biometric_data = self.processor.normalize_data(raw_data)
            self.processor.store_data(biometric_data)
            
            interventions = []
            
            # Evaluate each biomarker
            for biomarker in ['glucose', 'hrv', 'cortisol', 'sleep_quality', 'neuro_fatigue']:
                if biomarker in raw_data:
                    value = float(raw_data[biomarker])
                    
                    # Parse timestamp
                    timestamp = datetime.now()
                    if raw_data.get('timestamp'):
                        try:
                            timestamp = pd.to_datetime(raw_data['timestamp'])
                        except:
                            pass
                    
                    intervention = self.rule_engine.evaluate_biomarker(biomarker, value, timestamp)
                    if intervention:
                        self.active_interventions.append(intervention)
                        self.intervention_log.append(intervention.dict())
                        interventions.append(intervention)
            
            # Keep only recent interventions
            if len(self.active_interventions) > 10:
                self.active_interventions = self.active_interventions[-10:]
            
            return ProcessingResponse(
                success=True,
                message="Data processed successfully",
                interventions=interventions,
                processed_data=biometric_data.dict()
            )
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing data: {str(e)}")
    
    def get_status(self) -> StatusResponse:
        """Get current engine status"""
        return StatusResponse(
            engine_status="running" if self.is_running else "idle",
            total_processed=len(self.processor.data_buffer),
            active_interventions=len(self.active_interventions),
            models_trained=list(self.rule_engine.ml_engine.models.keys()),
            last_update=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

# Initialize FastAPI app
app = FastAPI(
    title="BioSyncDEX™ API",
    description="Real-time biomarker monitoring and ML-based intervention system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global engine
@app.on_event("startup")
async def startup_event():
    global biosync_engine
    biosync_engine = BioSyncEngine()

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "BioSyncDEX™ API - Real-time Biomarker Monitoring",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current engine status"""
    return biosync_engine.get_status()

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV file for ML model training"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        result = await biosync_engine.load_data_from_csv(csv_content)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@app.post("/process-biomarkers", response_model=ProcessingResponse)
async def process_biomarkers(data: BiometricDataModel):
    """Process biomarker data and get interventions"""
    raw_data = data.dict()
    
    # Add current timestamp if not provided
    if not raw_data.get('timestamp'):
        raw_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return await biosync_engine.process_biomarker_data(raw_data)

@app.post("/process-raw-data", response_model=ProcessingResponse)
async def process_raw_data(data: Dict[str, Any]):
    """Process raw biomarker data (flexible format)"""
    return await biosync_engine.process_biomarker_data(data)

@app.get("/interventions", response_model=List[InterventionModel])
async def get_active_interventions():
    """Get currently active interventions"""
    return biosync_engine.active_interventions

@app.get("/interventions/history")
async def get_intervention_history(limit: int = 50):
    """Get intervention history"""
    return {
        "total_interventions": len(biosync_engine.intervention_log),
        "interventions": biosync_engine.intervention_log[-limit:] if biosync_engine.intervention_log else []
    }

@app.get("/biomarkers/thresholds")
async def get_biomarker_thresholds():
    """Get biomarker thresholds"""
    return biosync_engine.rule_engine.ml_engine.thresholds

@app.post("/predict-biomarker")
async def predict_biomarker(
    biomarker: str,
    value: float,
    timestamp: Optional[str] = None
):
    """Predict biomarker status using ML model"""
    try:
        if timestamp:
            ts = pd.to_datetime(timestamp)
        else:
            ts = datetime.now()
        
        status, confidence = biosync_engine.rule_engine.ml_engine.predict_status(
            biomarker, value, ts
        )
        
        return {
            "biomarker": biomarker,
            "value": value,
            "predicted_status": status,
            "confidence": confidence,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/dashboard/data")
async def get_dashboard_data():
    """Get data for dashboard visualization"""
    recent_data = biosync_engine.processor.data_buffer[-20:] if biosync_engine.processor.data_buffer else []
    
    return {
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "engine_status": "running" if biosync_engine.is_running else "idle",
        "recent_biomarkers": recent_data,
        "active_interventions": [i.dict() for i in biosync_engine.active_interventions],
        "models_available": list(biosync_engine.rule_engine.ml_engine.models.keys()),
        "total_processed": len(biosync_engine.processor.data_buffer)
    }

@app.post("/generate-sample-data")
async def generate_sample_data(num_records: int = 50):
    """Generate sample biomarker data for testing"""
    sample_data = []
    base_time = datetime.now() - timedelta(hours=24)
    
    for i in range(num_records):
        timestamp = base_time + timedelta(minutes=i*30)
        
        glucose_base = 95 + random.uniform(-20, 40)
        hrv_base = 42 + random.uniform(-15, 15)
        cortisol_base = 12 + random.uniform(-5, 8)
        sleep_base = 78 + random.uniform(-25, 20)
        fatigue_base = 25 + random.uniform(-20, 30)
        
        # Create occasional outliers
        if random.random() < 0.1:
            glucose_base += random.choice([-30, 80])
        if random.random() < 0.15:
            hrv_base = max(15, hrv_base - 20)
        
        sample_data.append({
            'user_id': f'user_{random.randint(1, 3):03d}',
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'glucose': max(50, min(300, glucose_base)),
            'hrv': max(10, min(80, hrv_base)),
            'cortisol': max(2, min(40, cortisol_base)),
            'sleep_quality': max(0, min(100, int(sleep_base))),
            'neuro_fatigue': max(0, min(100, int(fatigue_base)))
        })
    
    # Convert to CSV format
    output = io.StringIO()
    fieldnames = ['user_id', 'timestamp', 'glucose', 'hrv', 'cortisol', 'sleep_quality', 'neuro_fatigue']
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(sample_data)
    
    csv_content = output.getvalue()
    
    # Train models with generated data
    result = await biosync_engine.load_data_from_csv(csv_content)
    
    return {
        "message": f"Generated {num_records} sample records and trained ML models",
        "csv_data": csv_content,
        "training_result": result
    }

@app.get("/download-sample-csv")
async def download_sample_csv():
    """Download sample CSV data"""
    # Generate sample data
    sample_result = await generate_sample_data(50)
    csv_content = sample_result["csv_data"]
    
    # Return as downloadable file
    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=biosync_sample_data.csv"}
    )

@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time data streaming"""
    await websocket.accept()
    try:
        while True:
            # Send current dashboard data
            dashboard_data = await get_dashboard_data()
            await websocket.send_json(dashboard_data)
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.post("/simulate-realtime")
async def simulate_realtime(background_tasks: BackgroundTasks, duration_minutes: int = 5):
    """Simulate real-time biomarker processing using CSV data"""
    if not biosync_engine.csv_data:
        raise HTTPException(status_code=400, detail="No CSV data loaded. Upload CSV first.")
    
    async def simulation_task():
        biosync_engine.is_running = True
        processed = 0
        start_time = datetime.now()
        
        while (biosync_engine.is_running and 
               biosync_engine.csv_index < len(biosync_engine.csv_data) and
               (datetime.now() - start_time).seconds < duration_minutes * 60):
            
            # Get next data point
            if biosync_engine.csv_index < len(biosync_engine.csv_data):
                raw_data = biosync_engine.csv_data[biosync_engine.csv_index]
                biosync_engine.csv_index += 1
                
                # Process data
                await biosync_engine.process_biomarker_data(raw_data)
                processed += 1
                
                await asyncio.sleep(1.5)  # 1.5 second delay
        
        biosync_engine.is_running = False
        print(f"Simulation completed: {processed} records processed")
    
    background_tasks.add_task(simulation_task)
    
    return {
        "message": f"Started real-time simulation for {duration_minutes} minutes",
        "data_points_available": len(biosync_engine.csv_data),
        "simulation_duration": duration_minutes
    }

@app.post("/stop-simulation")
async def stop_simulation():
    """Stop the real-time simulation"""
    biosync_engine.is_running = False
    return {"message": "Simulation stopped"}

@app.get("/export-results")
async def export_results():
    """Export intervention results as JSON"""
    if not biosync_engine.intervention_log:
        raise HTTPException(status_code=404, detail="No interventions recorded")
    
    results = {
        "export_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_interventions": len(biosync_engine.intervention_log),
        "interventions": biosync_engine.intervention_log
    }
    
    return StreamingResponse(
        io.StringIO(json.dumps(results, indent=2)),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=biosync_interventions.json"}
    )

@app.delete("/reset")
async def reset_engine():
    """Reset the engine and clear all data"""
    global biosync_engine
    biosync_engine = BioSyncEngine()
    return {"message": "Engine reset successfully"}

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "engine_initialized": biosync_engine is not None
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🧪 Starting BioSyncDEX™ FastAPI Server...")
    print("📋 Available endpoints:")
    print("  • POST /upload-csv - Upload training data")
    print("  • POST /process-biomarkers - Process biomarker data")
    print("  • GET /interventions - Get active interventions")
    print("  • POST /simulate-realtime - Start real-time simulation")
    print("  • GET /dashboard/data - Get dashboard data")
    print("  • GET /docs - Interactive API documentation")
    
    port = int(os.environ.get("PORT", 8000))  # use Render's PORT if provided

    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False,
        log_level="info"
    )