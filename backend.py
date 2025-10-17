# backend.py - FastAPI Backend for Dynamic Landmine Risk Intelligence System
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json

from model_and_utils import train_rf, predict_grid
from simulate_data import generate_synthetic_geodata

# Initialize FastAPI app
app = FastAPI(title="Dynamic Landmine Risk Intelligence API", version="1.0.0")

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store model and data
model = None
df_data = None
df_predictions = None

# Pydantic models for API requests/responses
class PointRequest(BaseModel):
    lat: float
    lon: float

class ShapExplanation(BaseModel):
    feature: str
    value: float
    contribution: float

class PointExplanation(BaseModel):
    coordinates: Dict[str, float]
    risk_probability: float
    features: Dict[str, float]
    shap_explanations: List[ShapExplanation]

class ModelMetrics(BaseModel):
    test_auc: float
    test_accuracy: float
    cv_auc_mean: float
    cv_auc_std: float
    n_train: int
    n_test: int

@app.on_event("startup")
async def startup_event():
    """Initialize the model and data on startup"""
    global model, df_data, df_predictions
    
    print("Initializing model and data...")
    
    # Generate synthetic data
    df_data = generate_synthetic_geodata(n_points=1200, seed=42)
    
    # Train model
    model, metrics, feature_importances = train_rf(df_data)
    
    # Generate predictions
    df_predictions = predict_grid(model, df_data)
    
    print("Model initialized successfully!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Dynamic Landmine Risk Intelligence API", "status": "running"}

@app.get("/api/data", response_model=List[Dict[str, Any]])
async def get_data():
    """Get all data points with predictions"""
    if df_predictions is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Convert to list of dictionaries for JSON serialization
    data = df_predictions.to_dict('records')
    return data

@app.get("/api/metrics", response_model=ModelMetrics)
async def get_metrics():
    """Get model performance metrics"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Train model to get fresh metrics
    _, metrics, _ = train_rf(df_data)
    
    return ModelMetrics(**metrics)

@app.post("/api/explain", response_model=PointExplanation)
async def explain_point(request: PointRequest):
    """Get feature explanation for a specific point (simplified without SHAP)"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Find the closest point in our dataset
    distances = np.sqrt(
        (df_data['lat'] - request.lat)**2 + 
        (df_data['lon'] - request.lon)**2
    )
    closest_idx = distances.idxmin()
    closest_point = df_data.iloc[closest_idx]
    
    # Get prediction for this point
    X = closest_point[['vegetation', 'soil_moisture', 'distance_to_road', 'conflict_intensity', 'elevation']].values.reshape(1, -1)
    risk_prob = model.predict_proba(X)[0, 1]
    
    # Get feature importances as simplified explanations
    feature_importances = model.feature_importances_
    
    # Create explanation
    features = {
        'vegetation': float(closest_point['vegetation']),
        'soil_moisture': float(closest_point['soil_moisture']),
        'distance_to_road': float(closest_point['distance_to_road']),
        'conflict_intensity': int(closest_point['conflict_intensity']),
        'elevation': float(closest_point['elevation'])
    }
    
    feature_names = ['vegetation', 'soil_moisture', 'distance_to_road', 'conflict_intensity', 'elevation']
    
    # Create simplified explanations based on feature values and importances
    shap_explanations = []
    for i, feature in enumerate(feature_names):
        # Simple heuristic: high feature value * high importance = positive contribution
        # This is a simplified version without actual SHAP values
        base_contribution = feature_importances[i] * 0.1  # Scale down
        if feature == 'distance_to_road':
            # Distance to road: closer = higher risk
            contribution = -base_contribution * features[feature]
        else:
            # Other features: higher value = higher risk
            contribution = base_contribution * features[feature]
        
        shap_explanations.append(ShapExplanation(
            feature=feature,
            value=features[feature],
            contribution=float(contribution)
        ))
    
    return PointExplanation(
        coordinates={'lat': float(closest_point['lat']), 'lon': float(closest_point['lon'])},
        risk_probability=float(risk_prob),
        features=features,
        shap_explanations=shap_explanations
    )

@app.get("/api/feature-importance")
async def get_feature_importance():
    """Get feature importance rankings"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Get feature importances
    feature_names = ['vegetation', 'soil_moisture', 'distance_to_road', 'conflict_intensity', 'elevation']
    importances = model.feature_importances_
    
    # Create sorted list
    feature_importance = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(feature_names, importances)
    ]
    
    # Sort by importance
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)
    
    return feature_importance

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
