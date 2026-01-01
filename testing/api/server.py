from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from predict import ModelPredictor
from integrated_optimizer import IntegratedRouteOptimizer
from simulation_engine import SimulationEngine


app = FastAPI(
    title="Fleet Management & Route Optimization API",
    description="AI-Driven Route Optimization and Delay Prediction System",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


predictor = None
optimizer = None
simulation_engine = None


@app.on_event("startup")
async def startup_event():
    global predictor, optimizer, simulation_engine
    try:
        models_dir = Path(__file__).parent.parent / "outputs" / "models"
        preprocessor_dir = Path(__file__).parent.parent / "outputs" / "preprocessor"
        
        predictor = ModelPredictor(
            models_dir=str(models_dir),
            preprocessor_dir=str(preprocessor_dir)
        )
        predictor.load_all_models()
        
        optimizer = IntegratedRouteOptimizer(
            models_dir=str(models_dir),
            preprocessor_dir=str(preprocessor_dir)
        )
        
        simulation_engine = SimulationEngine(
            predictor=predictor,
            optimizer=optimizer
        )
        
        print("Models, optimizer, and simulation engine loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load models: {e}")
        print("Run training first: python main.py")


class StopData(BaseModel):
    route_id: int
    driver_id: Union[int, str]  # Accept both int and string (preprocessing will encode)
    stop_id: Union[int, str]  # Accept both int and string
    address_id: int
    week_id: int
    country: Union[int, str]  # Accept both int and string (preprocessing will encode)
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
    risk_level: Optional[str] = None
    contributing_factors: Optional[List[Dict[str, Any]]] = None


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


@app.get("/data/sample")
async def get_sample_data(limit: int = 100):
    """Get sample data from the cleaned dataset for dashboard display"""
    try:
        data_path = Path(__file__).parent.parent / "data" / "cleaned_delivery_data.csv"
        
        if not data_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Data file not found at {data_path}"
            )
        
        df = pd.read_csv(data_path)
        
        # Limit the number of records
        sample_df = df.head(min(limit, len(df)))
        
        # Convert to list of dictionaries
        records = sample_df.to_dict('records')
        
        return {
            "total_records": len(df),
            "sample_size": len(records),
            "data": records
        }
    
    except Exception as e:
        import traceback
        print(f"Error fetching sample data: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/routes")
async def get_routes_data(limit: int = 50):
    """Get routes data grouped by route_id"""
    try:
        data_path = Path(__file__).parent.parent / "data" / "synthetic_delivery_data.csv"
        
        if not data_path.exists():
            data_path = Path(__file__).parent.parent / "data" / "cleaned_delivery_data.csv"
        
        if not data_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Data file not found"
            )
        
        df = pd.read_csv(data_path)
        
        # Fix negative delays - clamp to 0
        if 'delay_minutes' in df.columns:
            df['delay_minutes'] = df['delay_minutes'].clip(lower=0)
        
        # Count actual delays (delay_minutes > 0)
        df['has_delay'] = (df['delay_minutes'] > 0).astype(int)
        
        # Get unique routes with more details
        routes = df.groupby('route_id').agg({
            'driver_id': 'first',
            'stop_id': 'count',
            'distancep': 'sum',
            'day_of_week': 'first',
            'country': 'first',
            'has_delay': 'sum'
        }).reset_index()
        
        routes.columns = ['route_id', 'driver_id', 'total_stops', 'total_distance', 'day_of_week', 'country', 'total_delays']
        routes['delay_rate'] = (routes['total_delays'] / routes['total_stops'] * 100).round(1)
        routes = routes.head(limit)
        
        return {
            "total_routes": len(df['route_id'].unique()),
            "sample_size": len(routes),
            "routes": routes.to_dict('records')
        }
    
    except Exception as e:
        print(f"Error fetching routes data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/route/{route_id}")
async def get_route_details(route_id: int):
    """Get detailed data for a specific route"""
    try:
        data_path = Path(__file__).parent.parent / "data" / "synthetic_delivery_data.csv"
        
        if not data_path.exists():
            data_path = Path(__file__).parent.parent / "data" / "cleaned_delivery_data.csv"
        
        if not data_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Data file not found"
            )
        
        df = pd.read_csv(data_path)
        
        # Filter by route_id
        route_df = df[df['route_id'] == route_id].copy()
        
        if len(route_df) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Route {route_id} not found"
            )
        
        # Sort by planned index
        route_df = route_df.sort_values('indexp')
        
        # Calculate route summary - fix negative delays and proper counting
        # Clamp negative delays to 0 (negative means early arrival, not a delay)
        route_df['delay_minutes_clamped'] = route_df['delay_minutes'].clip(lower=0)
        
        # Count actual delays (delay_flag = 1 AND delay_minutes > 0)
        actual_delays = ((route_df['delay_flag'] == 1) & (route_df['delay_minutes_clamped'] > 0)).sum()
        
        summary = {
            "route_id": int(route_id),
            "driver_id": route_df['driver_id'].iloc[0],
            "country": route_df['country'].iloc[0],
            "day_of_week": route_df['day_of_week'].iloc[0],
            "total_stops": len(route_df),
            "total_distance": float(route_df['distancep'].sum()),
            "total_delays": int(actual_delays),
            "delay_rate": float((actual_delays / len(route_df)) * 100) if len(route_df) > 0 else 0.0,
            "avg_delay_minutes": float(route_df['delay_minutes_clamped'].mean()),
            "max_delay_minutes": float(route_df['delay_minutes_clamped'].max())
        }
        
        # Get stops data - fix negative delays in stop data
        stops = route_df.copy()
        stops['delay_minutes'] = stops['delay_minutes_clamped']
        stops = stops.drop(columns=['delay_minutes_clamped'])
        stops = stops.to_dict('records')
        
        return {
            "summary": summary,
            "stops": stops
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching route details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/dataset-structure")
async def get_dataset_structure():
    """Get complete dataset structure with parameter descriptions"""
    
    # Define all parameters in the dataset
    parameters = {
        "raw_data_columns": [
            {
                "name": "route_id",
                "type": "integer",
                "description": "Unique identifier for each delivery route",
                "used": True,
                "usage": "Route grouping, train/test splitting, sequence aggregation"
            },
            {
                "name": "stop_id",
                "type": "string",
                "description": "Unique identifier for each stop (format: S{route}_{stop})",
                "used": True,
                "usage": "Stop identification, sequence ordering"
            },
            {
                "name": "driver_id",
                "type": "string",
                "description": "Driver identifier (format: D{number})",
                "used": True,
                "usage": "Encoded as categorical feature for ML models"
            },
            {
                "name": "country",
                "type": "string",
                "description": "Country where delivery occurs",
                "used": True,
                "usage": "Encoded as categorical feature (regional patterns)"
            },
            {
                "name": "day_of_week",
                "type": "string",
                "description": "Day of the week (Monday-Sunday)",
                "used": True,
                "usage": "Encoded + weekday_flag feature (weekday vs weekend)"
            },
            {
                "name": "indexp",
                "type": "integer",
                "description": "Planned stop index/position in route",
                "used": True,
                "usage": "Sequence ordering, stop_deviation calculation"
            },
            {
                "name": "indexa",
                "type": "integer",
                "description": "Actual stop index/position executed",
                "used": True,
                "usage": "Stop_deviation = indexa - indexp (sequence changes)"
            },
            {
                "name": "distancep",
                "type": "float",
                "description": "Planned cumulative distance to stop (km)",
                "used": True,
                "usage": "Direct feature + route aggregates + distance_deviation"
            },
            {
                "name": "distancea",
                "type": "float",
                "description": "Actual cumulative distance traveled (km)",
                "used": True,
                "usage": "Distance_deviation = (distancea - distancep) / distancep"
            },
            {
                "name": "depot",
                "type": "binary (0/1)",
                "description": "Whether stop is the depot (starting point)",
                "used": True,
                "usage": "Binary feature indicating route start"
            },
            {
                "name": "delivery",
                "type": "binary (0/1)",
                "description": "Whether stop is a delivery location",
                "used": True,
                "usage": "Binary feature distinguishing deliveries from depot"
            },
            {
                "name": "arrived_time",
                "type": "time (HH:MM:SS)",
                "description": "Actual arrival time at stop",
                "used": True,
                "usage": "Extracted hour_of_arrival feature, delay calculation"
            },
            {
                "name": "earliest_time",
                "type": "time (HH:MM:SS)",
                "description": "Earliest acceptable arrival time (time window start)",
                "used": True,
                "usage": "Time_window_length calculation"
            },
            {
                "name": "latest_time",
                "type": "time (HH:MM:SS)",
                "description": "Latest acceptable arrival time (time window end)",
                "used": True,
                "usage": "Time_window_length + delay detection"
            },
            {
                "name": "actual_arrival_delay",
                "type": "float",
                "description": "Normalized delay value (fraction of day)",
                "used": True,
                "usage": "Converted to delay_minutes (target variable)"
            },
            {
                "name": "delay_flag",
                "type": "binary (0/1)",
                "description": "Binary flag: 1 if delayed, 0 if on-time",
                "used": True,
                "usage": "Primary target variable (classification)"
            },
            {
                "name": "delay_minutes",
                "type": "float",
                "description": "Delay duration in minutes",
                "used": True,
                "usage": "Secondary target variable (regression)"
            }
        ],
        "engineered_features": [
            {
                "name": "hour_of_arrival",
                "type": "integer (0-23)",
                "description": "Hour extracted from arrived_time",
                "calculation": "arrived_time.hour",
                "purpose": "Capture time-of-day patterns (rush hour, off-peak)"
            },
            {
                "name": "time_window_length",
                "type": "float (minutes)",
                "description": "Length of acceptable time window",
                "calculation": "(latest_time - earliest_time) in minutes",
                "purpose": "Tighter windows = higher delay risk"
            },
            {
                "name": "delay_ratio",
                "type": "float",
                "description": "Delay relative to time window",
                "calculation": "delay_minutes / (time_window_length + epsilon)",
                "purpose": "Normalized delay severity"
            },
            {
                "name": "weekday_flag",
                "type": "binary (0/1)",
                "description": "1 for weekdays, 0 for weekends",
                "calculation": "Map day_of_week to 1 (Mon-Fri) or 0 (Sat-Sun)",
                "purpose": "Weekday vs weekend traffic patterns"
            },
            {
                "name": "stop_deviation",
                "type": "integer",
                "description": "Difference between actual and planned stop order",
                "calculation": "indexa - indexp",
                "purpose": "Captures route reordering (deviation from plan)"
            },
            {
                "name": "distance_deviation",
                "type": "float (percentage)",
                "description": "Proportional distance deviation",
                "calculation": "(distancea - distancep) / (distancep + epsilon)",
                "purpose": "Captures unexpected detours/route changes"
            },
            {
                "name": "stop_position_norm",
                "type": "float (0-1)",
                "description": "Normalized position of stop in route",
                "calculation": "stop_index / total_stops_in_route",
                "purpose": "Later stops accumulate delays"
            },
            {
                "name": "prev_stop_delay",
                "type": "float (minutes)",
                "description": "Delay at previous stop in same route",
                "calculation": "delay_minutes.shift(1) by route",
                "purpose": "Delay propagation modeling"
            },
            {
                "name": "cumulative_delay",
                "type": "float (minutes)",
                "description": "Total delay accumulated up to current stop",
                "calculation": "delay_minutes.cumsum() by route",
                "purpose": "Cumulative delay impact"
            },
            {
                "name": "route_total_stops",
                "type": "integer",
                "description": "Total number of stops in the route",
                "calculation": "Count stops per route_id",
                "purpose": "Route complexity indicator"
            },
            {
                "name": "route_avg_distance",
                "type": "float (km)",
                "description": "Average distance per stop in route",
                "calculation": "Mean distancep per route_id",
                "purpose": "Route density/sparsity"
            },
            {
                "name": "route_total_distance",
                "type": "float (km)",
                "description": "Total distance of entire route",
                "calculation": "Sum distancep per route_id",
                "purpose": "Overall route length"
            }
        ],
        "target_variables": [
            {
                "name": "delayed_flag",
                "type": "binary (0/1)",
                "description": "Primary classification target",
                "usage": "Used by Logistic Regression, Random Forest Classifier, LSTM Classifier",
                "note": "Predicted by all classification models"
            },
            {
                "name": "delay_minutes",
                "type": "float",
                "description": "Regression target (delay duration)",
                "usage": "Used by Random Forest Regressor, LSTM Regressor",
                "note": "Predicts actual delay magnitude"
            }
        ],
        "unused_columns": [],
        "summary": {
            "total_raw_columns": 17,
            "used_raw_columns": 17,
            "unused_raw_columns": 0,
            "engineered_features": 12,
            "total_features_for_ml": 29,
            "categorical_encoded": 3,
            "note": "All raw data columns are used either directly as features or to engineer new features. No data is wasted."
        }
    }
    
    return parameters


@app.get("/data/drivers")
async def get_drivers_data():
    """Get drivers data grouped by driver_id"""
    try:
        data_path = Path(__file__).parent.parent / "data" / "synthetic_delivery_data.csv"
        
        if not data_path.exists():
            data_path = Path(__file__).parent.parent / "data" / "cleaned_delivery_data.csv"
        
        if not data_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Data file not found"
            )
        
        df = pd.read_csv(data_path)
        
        # Get driver statistics
        drivers = df.groupby('driver_id').agg({
            'stop_id': 'count',
            'route_id': 'nunique',
            'distancep': 'sum'
        }).reset_index()
        
        drivers.columns = ['driver_id', 'completed_stops', 'total_routes', 'total_distance']
        
        return {
            "total_drivers": len(df['driver_id'].unique()),
            "drivers": drivers.to_dict('records')
        }
    
    except Exception as e:
        print(f"Error fetching drivers data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/delays", response_model=List[PredictionResponse])
async def predict_delays(request: PredictionRequest, include_factors: bool = False):
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run training first."
        )
    
    try:
        stops_data = [stop.dict() for stop in request.stops]
        df = pd.DataFrame(stops_data)
        
        # Log sample data for debugging
        print(f"Received {len(df)} stops for prediction")
        if len(df) > 0:
            print(f"Sample stop: {df.iloc[0].to_dict()}")
        
        predictions = predictor.predict_route_delays(df, include_factors=include_factors)
        
        results = []
        for _, row in predictions.iterrows():
            results.append(PredictionResponse(
                route_id=int(row["route_id"]),
                stop_id=int(row["stop_id"]),
                driver_id=int(row["driver_id"]),
                delayed_flag_pred=int(row["delayed_flag_pred"]),
                delay_probability=float(row["delay_probability"]) if row["delay_probability"] is not None else None,
                delay_minutes_pred=float(row["delay_minutes_pred"]),
                risk_level=row.get("risk_level"),
                contributing_factors=row.get("contributing_factors") if include_factors else None
            ))
        
        print(f"Successfully predicted {len(results)} stops")
        return results
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in predict_delays: {str(e)}")
        print(f"Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


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
        "optimizer_ready": optimizer is not None,
        "status": "Models loaded successfully"
    }


@app.post("/optimize/routes")
async def optimize_routes(request: PredictionRequest):
    if optimizer is None:
        raise HTTPException(
            status_code=503,
            detail="Optimizer not loaded. Please run training first."
        )
    
    try:
        stops_data = [stop.dict() for stop in request.stops]
        df = pd.DataFrame(stops_data)
        
        result = optimizer.predict_and_optimize(df, time_limit=30)
        
        if result is None:
            raise HTTPException(status_code=500, detail="Optimization failed")
        
        return {
            "solution": result['solution'],
            "num_reassignments": len(result['reassignments']),
            "reassignments": result['reassignments'][:20]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reassign/stops")
async def reassign_stops(request: PredictionRequest, delay_threshold: float = 0.7):
    if optimizer is None:
        raise HTTPException(
            status_code=503,
            detail="Optimizer not loaded. Please run training first."
        )
    
    try:
        stops_data = [stop.dict() for stop in request.stops]
        df = pd.DataFrame(stops_data)
        
        predictions = predictor.predict_route_delays(df)
        
        available_drivers = df['driver_id'].unique().tolist()
        
        routes_reassigned, reassignments = optimizer.reassignment.reassign_stops(
            df,
            predictions,
            available_drivers
        )
        
        return {
            "num_reassignments": len(reassignments),
            "reassignments": reassignments,
            "message": f"Identified {len(reassignments)} reassignment opportunities"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/realtime/adjust")
async def realtime_adjustment(
    current_routes: List[StopData],
    live_delays: List[Dict[str, Any]]
):
    if optimizer is None:
        raise HTTPException(
            status_code=503,
            detail="Optimizer not loaded."
        )
    
    try:
        current_df = pd.DataFrame([stop.dict() for stop in current_routes])
        delays_df = pd.DataFrame(live_delays)
        
        adjusted_solution = optimizer.real_time_adjustment(current_df, delays_df)
        
        return {
            "adjusted_routes": adjusted_solution,
            "message": "Routes adjusted based on live delays"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/optimization/status")
async def optimization_status():
    return {
        "predictor_loaded": predictor is not None,
        "optimizer_loaded": optimizer is not None,
        "available_models": list(predictor.models.keys()) if predictor else [],
        "optimization_ready": optimizer is not None
    }


class ScenarioModifications(BaseModel):
    stop_modifications: Optional[Dict[str, Any]] = None
    driver_reassignments: Optional[Dict[int, int]] = None  # stop_id -> new_driver_id
    traffic_multiplier: Optional[float] = 1.0
    distance_multiplier: Optional[float] = 1.0
    time_shift_minutes: Optional[Dict[int, int]] = None  # stop_id -> minutes_shift
    stops_to_remove: Optional[List[int]] = None


class ScenarioComparison(BaseModel):
    metric: str
    original: float
    modified: float
    improvement_percent: float


@app.post("/simulate/scenario")
async def simulate_scenario(
    request: PredictionRequest,
    modifications: Optional[ScenarioModifications] = None
):
    """
    Simulate what-if scenarios with parameter modifications
    
    This endpoint allows you to:
    - Reassign drivers to specific stops
    - Apply traffic multipliers to distances
    - Shift arrival times
    - Remove stops from routes
    - Compare before/after metrics
    """
    if predictor is None or optimizer is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run training first."
        )
    
    try:
        # Original stops
        original_stops_data = [stop.dict() for stop in request.stops]
        original_df = pd.DataFrame(original_stops_data)
        
        # Make predictions on original
        print("Predicting delays for original scenario...")
        original_predictions = predictor.predict_route_delays(original_df, include_factors=True)
        
        # Apply modifications if provided
        modified_df = original_df.copy()
        
        if modifications:
            # Apply driver reassignments
            if modifications.driver_reassignments:
                for stop_id, new_driver_id in modifications.driver_reassignments.items():
                    modified_df.loc[modified_df['stop_id'] == stop_id, 'driver_id'] = new_driver_id
                print(f"Applied {len(modifications.driver_reassignments)} driver reassignments")
            
            # Apply traffic multiplier to distances
            if modifications.traffic_multiplier and modifications.traffic_multiplier != 1.0:
                modified_df['distancep'] = modified_df['distancep'] * modifications.traffic_multiplier
                modified_df['distancea'] = modified_df['distancea'] * modifications.traffic_multiplier
                print(f"Applied traffic multiplier: {modifications.traffic_multiplier}")
            
            # Apply distance multiplier
            if modifications.distance_multiplier and modifications.distance_multiplier != 1.0:
                modified_df['distancep'] = modified_df['distancep'] * modifications.distance_multiplier
                modified_df['distancea'] = modified_df['distancea'] * modifications.distance_multiplier
                print(f"Applied distance multiplier: {modifications.distance_multiplier}")
            
            # Apply time shifts
            if modifications.time_shift_minutes:
                for stop_id, minutes_shift in modifications.time_shift_minutes.items():
                    mask = modified_df['stop_id'] == stop_id
                    if mask.any():
                        # Parse time strings and add minutes
                        for time_col in ['arrived_time', 'earliest_time', 'latest_time']:
                            if time_col in modified_df.columns:
                                time_vals = pd.to_datetime(modified_df.loc[mask, time_col])
                                modified_df.loc[mask, time_col] = (
                                    time_vals + pd.Timedelta(minutes=minutes_shift)
                                ).dt.strftime('%H:%M')
                print(f"Applied time shifts to {len(modifications.time_shift_minutes)} stops")
            
            # Remove stops
            if modifications.stops_to_remove:
                modified_df = modified_df[~modified_df['stop_id'].isin(modifications.stops_to_remove)]
                print(f"Removed {len(modifications.stops_to_remove)} stops")
        
        # Make predictions on modified scenario
        print("Predicting delays for modified scenario...")
        modified_predictions = predictor.predict_route_delays(modified_df, include_factors=True)
        
        # Run optimization on both
        print("Optimizing original routes...")
        original_optimization = optimizer.predict_and_optimize(original_df, time_limit=30)
        
        print("Optimizing modified routes...")
        modified_optimization = optimizer.predict_and_optimize(modified_df, time_limit=30)
        
        # Calculate comparisons
        comparisons = []
        
        # Total delay comparison
        orig_total_delay = float(original_predictions['delay_minutes_pred'].sum())
        mod_total_delay = float(modified_predictions['delay_minutes_pred'].sum())
        comparisons.append({
            "metric": "Total Delay (minutes)",
            "original": orig_total_delay,
            "modified": mod_total_delay,
            "improvement_percent": ((orig_total_delay - mod_total_delay) / max(orig_total_delay, 1)) * 100
        })
        
        # Average delay probability
        orig_avg_prob = float(original_predictions['delay_probability'].mean())
        mod_avg_prob = float(modified_predictions['delay_probability'].mean())
        comparisons.append({
            "metric": "Average Delay Probability",
            "original": orig_avg_prob,
            "modified": mod_avg_prob,
            "improvement_percent": ((orig_avg_prob - mod_avg_prob) / max(orig_avg_prob, 0.01)) * 100
        })
        
        # High risk stops
        orig_high_risk = int((original_predictions['risk_level'] == 'HIGH').sum())
        mod_high_risk = int((modified_predictions['risk_level'] == 'HIGH').sum())
        comparisons.append({
            "metric": "High Risk Stops",
            "original": float(orig_high_risk),
            "modified": float(mod_high_risk),
            "improvement_percent": ((orig_high_risk - mod_high_risk) / max(orig_high_risk, 1)) * 100
        })
        
        # Distance comparison (if optimization succeeded)
        if original_optimization and modified_optimization:
            orig_distance = original_optimization['solution']['total_distance']
            mod_distance = modified_optimization['solution']['total_distance']
            comparisons.append({
                "metric": "Total Distance (km)",
                "original": orig_distance,
                "modified": mod_distance,
                "improvement_percent": ((orig_distance - mod_distance) / max(orig_distance, 1)) * 100
            })
        
        # Convert predictions to list of dicts
        original_pred_list = original_predictions.to_dict('records')
        modified_pred_list = modified_predictions.to_dict('records')
        
        return {
            "original": {
                "predictions": original_pred_list,
                "total_stops": len(original_df),
                "total_delay_minutes": orig_total_delay,
                "avg_delay_probability": orig_avg_prob,
                "high_risk_count": orig_high_risk,
                "optimization": original_optimization['solution'] if original_optimization else None
            },
            "modified": {
                "predictions": modified_pred_list,
                "total_stops": len(modified_df),
                "total_delay_minutes": mod_total_delay,
                "avg_delay_probability": mod_avg_prob,
                "high_risk_count": mod_high_risk,
                "optimization": modified_optimization['solution'] if modified_optimization else None
            },
            "comparisons": comparisons,
            "modifications_applied": {
                "driver_reassignments": len(modifications.driver_reassignments) if modifications and modifications.driver_reassignments else 0,
                "traffic_multiplier": modifications.traffic_multiplier if modifications else 1.0,
                "distance_multiplier": modifications.distance_multiplier if modifications else 1.0,
                "time_shifts": len(modifications.time_shift_minutes) if modifications and modifications.time_shift_minutes else 0,
                "stops_removed": len(modifications.stops_to_remove) if modifications and modifications.stops_to_remove else 0
            }
        }
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in simulate_scenario: {str(e)}")
        print(f"Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


# ============================================================================
# NEW SIMULATION ENDPOINTS - Enhanced prediction and optimization with explanations
# ============================================================================

class SimulationScenarioRequest(BaseModel):
    """Request model for simulation/predict endpoint"""
    route_id: Optional[str] = None
    stops: List[StopData]
    scenario_type: str = "normal"  # normal, traffic_congestion, delay_at_stop, etc.
    custom_params: Optional[Dict[str, Any]] = None


class StopPrediction(BaseModel):
    """Detailed prediction for a single stop"""
    stop_id: Union[int, str]  # Can be string like 'S8_7' or int
    route_id: int
    delay_minutes: float
    delay_probability: float
    risk_level: str
    reason: str
    contributing_factors: Optional[List[Dict[str, Any]]] = None


class RouteSummary(BaseModel):
    """Summary statistics for the entire route"""
    total_expected_delay: float
    delay_risk: str  # low, medium, high
    expected_total_distance: float
    expected_total_duration: float
    high_risk_stops: int
    on_time_probability: float


class SimulationPredictionResponse(BaseModel):
    """Response from simulation/predict endpoint"""
    route_id: str
    predictions: List[StopPrediction]
    route_summary: RouteSummary
    scenario_applied: Dict[str, Any]


class OptimizationAction(BaseModel):
    """A single optimization action with explanation"""
    action: str
    reason: str
    expected_benefit: str
    details: Optional[Dict[str, Any]] = None


class OptimizationImpact(BaseModel):
    """Expected impact of optimization"""
    delay_reduction_minutes: float
    energy_saving_km: float
    on_time_rate_improvement: float


class SimulationOptimizationResponse(BaseModel):
    """Response from simulation/optimize endpoint"""
    old_sequence: List[int]
    new_sequence: List[int]
    actions: List[OptimizationAction]
    impact: OptimizationImpact


@app.post("/simulation/predict", response_model=SimulationPredictionResponse)
async def simulate_and_predict(request: SimulationScenarioRequest):
    """
    Simulate a specific scenario and predict delays with explanations
    
    This endpoint:
    - Applies scenario modifications (traffic, delays, faults, etc.)
    - Runs ML predictions on the modified data
    - Returns detailed predictions with reasons for each stop
    - Provides route-level summary with risk assessment
    """
    if simulation_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Simulation engine not loaded. Please run training first."
        )
    
    try:
        # Convert stops to DataFrame
        stops_data = [stop.dict() for stop in request.stops]
        original_df = pd.DataFrame(stops_data)
        
        print(f"Received simulation request: {request.scenario_type}")
        print(f"Processing {len(original_df)} stops")
        
        # Apply scenario modifications
        modified_df, modifications = simulation_engine.apply_scenario_modifications(
            original_df,
            request.scenario_type,
            request.custom_params
        )
        
        print(f"Applied modifications: {len(modifications['applied_changes'])} changes")
        
        # Generate predictions with explanations
        prediction_result = simulation_engine.predict_with_explanations(
            modified_df,
            route_id=request.route_id
        )
        
        # Build response
        response = {
            "route_id": prediction_result["route_id"],
            "predictions": prediction_result["predictions"],
            "route_summary": prediction_result["route_summary"],
            "scenario_applied": modifications
        }
        
        print(f"Prediction complete. Total expected delay: {prediction_result['route_summary']['total_expected_delay']:.1f} min")
        
        return response
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in simulation/predict: {str(e)}")
        print(f"Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Simulation prediction error: {str(e)}")


@app.post("/simulation/optimize", response_model=SimulationOptimizationResponse)
async def simulate_and_optimize(request: SimulationScenarioRequest):
    """
    Generate optimization recommendations with clear explanations
    
    This endpoint:
    - Takes predicted delays and route data
    - Generates actionable optimization recommendations
    - Explains WHY each optimization is suggested
    - Calculates expected impact of each action
    """
    if simulation_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Simulation engine not loaded. Please run training first."
        )
    
    try:
        # Convert stops to DataFrame
        stops_data = [stop.dict() for stop in request.stops]
        original_df = pd.DataFrame(stops_data)
        
        print(f"Generating optimization recommendations for {len(original_df)} stops")
        
        # Apply scenario modifications if any
        modified_df, modifications = simulation_engine.apply_scenario_modifications(
            original_df,
            request.scenario_type,
            request.custom_params
        )
        
        # Generate predictions
        predictions_df = simulation_engine.predictor.predict_route_delays(
            modified_df, 
            include_factors=True
        )
        
        # Run optimizer
        optimization_result = None
        try:
            opt_result = simulation_engine.optimizer.predict_and_optimize(
                modified_df, 
                time_limit=30
            )
            if opt_result:
                optimization_result = opt_result
        except Exception as opt_error:
            print(f"Warning: Optimization failed: {opt_error}")
        
        # Generate recommendations with explanations
        recommendations = simulation_engine.generate_optimization_recommendations(
            modified_df,
            predictions_df,
            optimization_result
        )
        
        # Format actions for response
        formatted_actions = []
        for action in recommendations["actions"]:
            formatted_actions.append({
                "action": action["action"],
                "reason": action["reason"],
                "expected_benefit": action.get("expected_benefit", "Improved efficiency"),
                "details": {k: v for k, v in action.items() if k not in ["action", "reason", "expected_benefit"]}
            })
        
        response = {
            "old_sequence": recommendations["old_sequence"],
            "new_sequence": recommendations.get("new_sequence", recommendations["old_sequence"]),
            "actions": formatted_actions,
            "impact": recommendations["impact"]
        }
        
        print(f"Generated {len(formatted_actions)} optimization recommendations")
        print(f"Expected delay reduction: {recommendations['impact']['delay_reduction_minutes']:.1f} min")
        
        return response
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in simulation/optimize: {str(e)}")
        print(f"Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Simulation optimization error: {str(e)}")


@app.post("/simulation/full-analysis")
async def full_simulation_analysis(request: SimulationScenarioRequest):
    """
    Complete simulation analysis: predict + optimize in one call
    
    Returns both prediction results and optimization recommendations together.
    This is useful for the frontend to get all simulation data in a single request.
    """
    if simulation_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Simulation engine not loaded."
        )
    
    try:
        # Get predictions
        prediction_response = await simulate_and_predict(request)
        
        # Get optimizations
        optimization_response = await simulate_and_optimize(request)
        
        return {
            "predictions": prediction_response if isinstance(prediction_response, dict) else prediction_response.dict(),
            "optimizations": optimization_response if isinstance(optimization_response, dict) else optimization_response.dict(),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in full simulation analysis: {str(e)}")
        print(f"Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Full analysis error: {str(e)}")
