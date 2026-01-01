"""
FastAPI Server for v2 DL Route Optimizer
Provides endpoints for DL-based route optimization
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dl_predict import DLRoutePredictor

# Initialize FastAPI app
app = FastAPI(
    title="AI-Driven Route Optimization v2 (DL)",
    description="Deep Learning-based route optimization API",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None
data_df = None


# Pydantic models
class StopData(BaseModel):
    stop_id: str
    driver_id: str
    country: str
    day_of_week: str
    indexp: int
    indexa: int
    distancep: float
    distancea: float
    depot: int
    delivery: int
    earliest_time: str
    latest_time: str
    delay_flag: Optional[int] = 0
    delay_minutes: Optional[float] = 0.0


class RouteData(BaseModel):
    route_id: int
    stops: List[StopData]


class PredictionResponse(BaseModel):
    success: bool
    route_id: int
    num_stops: int
    predicted_sequence: List[int]
    planned_sequence: List[int]
    actual_sequence: List[int]
    stop_ids: List[str]
    confidence_scores: List[float]
    metrics: Dict[str, float]
    error: Optional[str] = None


class EvaluationResponse(BaseModel):
    success: bool
    num_routes_evaluated: int
    avg_sequence_accuracy: float
    avg_kendall_tau_predicted: float
    avg_kendall_tau_planned: float
    avg_improvement: float
    pct_better_than_planned: float
    error: Optional[str] = None


class RouteListResponse(BaseModel):
    routes: List[Dict[str, Any]]
    total_routes: int


@app.on_event("startup")
async def startup_event():
    """Initialize predictor and load data on startup."""
    global predictor, data_df
    
    try:
        # Load predictor
        model_path = Path("outputs_v2/dl_models/best_model.pt")
        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}")
            print("Please train the model first using: python train_dl_model.py")
            predictor = None
        else:
            predictor = DLRoutePredictor(str(model_path))
            print("✓ DL Route Predictor loaded successfully")
        
        # Load data
        data_path = Path("data/synthetic_delivery_data.csv")
        if data_path.exists():
            data_df = pd.read_csv(data_path)
            print(f"✓ Loaded {len(data_df):,} stops from {data_df['route_id'].nunique():,} routes")
        else:
            print(f"Warning: Data not found at {data_path}")
            data_df = None
    
    except Exception as e:
        print(f"Error during startup: {e}")
        predictor = None
        data_df = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI-Driven Route Optimization v2 (DL)",
        "version": "2.0.0",
        "status": "running",
        "model_loaded": predictor is not None,
        "data_loaded": data_df is not None
    }


@app.get("/api/v2/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "data_loaded": data_df is not None,
        "num_routes": int(data_df['route_id'].nunique()) if data_df is not None else 0
    }


@app.get("/api/v2/routes", response_model=RouteListResponse)
async def get_routes(limit: int = 50, offset: int = 0):
    """Get list of available routes."""
    if data_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    route_ids = data_df['route_id'].unique()[offset:offset + limit]
    
    routes = []
    for route_id in route_ids:
        route_data = data_df[data_df['route_id'] == route_id]
        routes.append({
            'route_id': int(route_id),
            'num_stops': len(route_data),
            'driver_id': str(route_data['driver_id'].iloc[0]),
            'country': str(route_data['country'].iloc[0]),
            'day_of_week': str(route_data['day_of_week'].iloc[0]),
            'total_distance_planned': float(route_data['distancep'].sum()),
            'total_distance_actual': float(route_data['distancea'].sum())
        })
    
    return RouteListResponse(
        routes=routes,
        total_routes=int(data_df['route_id'].nunique())
    )


@app.get("/api/v2/route/{route_id}")
async def get_route_details(route_id: int):
    """Get detailed information for a specific route."""
    if data_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    route_data = data_df[data_df['route_id'] == route_id]
    
    if len(route_data) == 0:
        raise HTTPException(status_code=404, detail=f"Route {route_id} not found")
    
    route_data_sorted = route_data.sort_values('indexp').reset_index(drop=True)
    
    stops = []
    for _, row in route_data_sorted.iterrows():
        stops.append({
            'stop_id': str(row['stop_id']),
            'planned_position': int(row['indexp']),
            'actual_position': int(row['indexa']),
            'distance_planned': float(row['distancep']),
            'distance_actual': float(row['distancea']),
            'is_depot': int(row['depot']) == 1,
            'is_delivery': int(row['delivery']) == 1,
            'earliest_time': str(row['earliest_time']),
            'latest_time': str(row['latest_time']),
            'delay_flag': int(row.get('delay_flag', 0)),
            'delay_minutes': float(row.get('delay_minutes', 0))
        })
    
    return {
        'route_id': int(route_id),
        'num_stops': len(stops),
        'driver_id': str(route_data['driver_id'].iloc[0]),
        'country': str(route_data['country'].iloc[0]),
        'day_of_week': str(route_data['day_of_week'].iloc[0]),
        'stops': stops,
        'total_distance_planned': float(route_data['distancep'].sum()),
        'total_distance_actual': float(route_data['distancea'].sum())
    }


@app.post("/api/v2/predict", response_model=PredictionResponse)
async def predict_route(route_data: RouteData):
    """
    Predict optimal route sequence using DL model.
    
    Args:
        route_data: Route information with stops
    
    Returns:
        Prediction with optimal sequence and metrics
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Convert to DataFrame
        stops_data = []
        for stop in route_data.stops:
            stops_data.append(stop.dict())
        
        route_df = pd.DataFrame(stops_data)
        route_df['route_id'] = route_data.route_id
        
        # Predict
        result = predictor.predict_single_route(route_df)
        
        if not result.get('success', False):
            raise HTTPException(status_code=500, detail=result.get('error', 'Prediction failed'))
        
        return PredictionResponse(
            success=True,
            route_id=result['route_id'],
            num_stops=result['num_stops'],
            predicted_sequence=result['predicted_sequence'],
            planned_sequence=result['planned_sequence'],
            actual_sequence=result['actual_sequence'],
            stop_ids=result['stop_ids'],
            confidence_scores=result['confidence_scores'],
            metrics={
                'sequence_accuracy': result['sequence_accuracy'],
                'kendall_tau_predicted': result['kendall_tau_predicted'],
                'kendall_tau_planned': result['kendall_tau_planned'],
                'improvement_over_planned': result['improvement_over_planned'],
                'planned_total_distance': result['planned_total_distance'],
                'actual_total_distance': result['actual_total_distance']
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/predict/{route_id}", response_model=PredictionResponse)
async def predict_route_by_id(route_id: int):
    """Predict optimal sequence for a route by ID."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if data_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    route_data = data_df[data_df['route_id'] == route_id]
    
    if len(route_data) == 0:
        raise HTTPException(status_code=404, detail=f"Route {route_id} not found")
    
    try:
        result = predictor.predict_single_route(route_data)
        
        if not result.get('success', False):
            raise HTTPException(status_code=500, detail=result.get('error', 'Prediction failed'))
        
        return PredictionResponse(
            success=True,
            route_id=result['route_id'],
            num_stops=result['num_stops'],
            predicted_sequence=result['predicted_sequence'],
            planned_sequence=result['planned_sequence'],
            actual_sequence=result['actual_sequence'],
            stop_ids=result['stop_ids'],
            confidence_scores=result['confidence_scores'],
            metrics={
                'sequence_accuracy': result['sequence_accuracy'],
                'kendall_tau_predicted': result['kendall_tau_predicted'],
                'kendall_tau_planned': result['kendall_tau_planned'],
                'improvement_over_planned': result['improvement_over_planned'],
                'planned_total_distance': result['planned_total_distance'],
                'actual_total_distance': result['actual_total_distance']
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/evaluate", response_model=EvaluationResponse)
async def evaluate_model(sample_size: int = 100):
    """
    Evaluate model performance on random sample of routes.
    
    Args:
        sample_size: Number of routes to evaluate
    
    Returns:
        Aggregate evaluation metrics
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if data_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    try:
        metrics = predictor.evaluate_on_test_set(data_df, sample_size)
        
        if 'error' in metrics:
            raise HTTPException(status_code=500, detail=metrics['error'])
        
        return EvaluationResponse(
            success=True,
            num_routes_evaluated=metrics['num_routes_evaluated'],
            avg_sequence_accuracy=metrics['avg_sequence_accuracy'],
            avg_kendall_tau_predicted=metrics['avg_kendall_tau_predicted'],
            avg_kendall_tau_planned=metrics['avg_kendall_tau_planned'],
            avg_improvement=metrics['avg_improvement'],
            pct_better_than_planned=metrics['pct_better_than_planned']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/visualization/{route_id}")
async def get_visualization_data(route_id: int):
    """Get data formatted for visualization in UI."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if data_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    route_data = data_df[data_df['route_id'] == route_id]
    
    if len(route_data) == 0:
        raise HTTPException(status_code=404, detail=f"Route {route_id} not found")
    
    try:
        # Get prediction
        result = predictor.predict_single_route(route_data)
        
        if not result.get('success', False):
            raise HTTPException(status_code=500, detail=result.get('error', 'Prediction failed'))
        
        # Create visualization data
        viz_data = predictor.create_route_visualization_data(result, route_data)
        
        return viz_data
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/random-routes")
async def get_random_routes(count: int = 10):
    """Get random routes for testing."""
    if data_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    route_ids = data_df['route_id'].unique()
    random_routes = np.random.choice(route_ids, min(count, len(route_ids)), replace=False)
    
    routes = []
    for route_id in random_routes:
        route_data = data_df[data_df['route_id'] == route_id]
        routes.append({
            'route_id': int(route_id),
            'num_stops': len(route_data),
            'driver_id': str(route_data['driver_id'].iloc[0]),
            'country': str(route_data['country'].iloc[0]),
            'day_of_week': str(route_data['day_of_week'].iloc[0])
        })
    
    return {'routes': routes}


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("AI-DRIVEN ROUTE OPTIMIZATION API v2 (DL)")
    print("=" * 80)
    print("Starting server on http://localhost:8001")
    print("API documentation: http://localhost:8001/docs")
    print("=" * 80)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)

