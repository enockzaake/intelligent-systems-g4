from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from predict import ModelPredictor


app = FastAPI(
    title="Delivery Route Optimization and Delay Prediction System",
    description="Delivery Route Optimization and Delay Prediction System",
    version="1.0.0"
)

predictor = None


@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        models_dir = Path(__file__).parent.parent / "outputs" / "models"
        preprocessor_dir = Path(__file__).parent.parent / "outputs" / "preprocessor"
        
        predictor = ModelPredictor(
            models_dir=str(models_dir),
            preprocessor_dir=str(preprocessor_dir)
        )
        predictor.load_all_models()
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load models: {e}")
        print("Run training first: python main.py")


class StopData(BaseModel):
    route_id: int
    driver_id: int
    stop_id: int
    address_id: int
    week_id: int
    country: int
    day_of_week: str
    indexp: int
    indexa: int
    arrived_time: str
    earliest_time: str
    latest_time: str
    distancep: float
    distancea: float
    depot: int
    delivery: int


class PredictionRequest(BaseModel):
    stops: List[StopData]
    model_type: Optional[str] = "random_forest"


class PredictionResponse(BaseModel):
    route_id: int
    stop_id: int
    driver_id: int
    delayed_flag_pred: int
    delay_probability: Optional[float]
    delay_minutes_pred: float


class RouteAggregate(BaseModel):
    route_id: int
    route_delay_rate: float
    avg_delay_probability: float
    total_delay_minutes: float
    avg_delay_minutes: float
    max_delay_minutes: float


@app.get("/")
async def root():
    return {
        "name": "Fleet Management API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": predictor is not None
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_available": list(predictor.models.keys()) if predictor else []
    }


@app.post("/predict/delays", response_model=List[PredictionResponse])
async def predict_delays(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run training first."
        )
    
    try:
        stops_data = [stop.dict() for stop in request.stops]
        df = pd.DataFrame(stops_data)
        
        predictions = predictor.predict_route_delays(df)
        
        results = []
        for _, row in predictions.iterrows():
            results.append(PredictionResponse(
                route_id=int(row["route_id"]),
                stop_id=int(row["stop_id"]),
                driver_id=int(row["driver_id"]),
                delayed_flag_pred=int(row["delayed_flag_pred"]),
                delay_probability=float(row["delay_probability"]) if row["delay_probability"] is not None else None,
                delay_minutes_pred=float(row["delay_minutes_pred"])
            ))
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/routes", response_model=List[RouteAggregate])
async def predict_route_aggregates(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run training first."
        )
    
    try:
        stops_data = [stop.dict() for stop in request.stops]
        df = pd.DataFrame(stops_data)
        
        route_aggregates = predictor.predict_and_aggregate_routes(df)
        
        results = []
        for _, row in route_aggregates.iterrows():
            results.append(RouteAggregate(
                route_id=int(row["route_id"]),
                route_delay_rate=float(row["route_delay_rate"]),
                avg_delay_probability=float(row["avg_delay_probability"]),
                total_delay_minutes=float(row["total_delay_minutes"]),
                avg_delay_minutes=float(row["avg_delay_minutes"]),
                max_delay_minutes=float(row["max_delay_minutes"])
            ))
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    if predictor is None:
        return {"models": [], "status": "Models not loaded"}
    
    return {
        "models": list(predictor.models.keys()),
        "status": "Models loaded successfully"
    }
