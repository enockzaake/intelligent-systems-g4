# AI-Driven Route Optimization and Delay Prediction System - Core Module

This is the core implementation of the AI-driven route optimization and delay prediction system for fleet operations.

## Project Structure

```
core/
├── api/                          # FastAPI REST API
│   └── main.py                   # API endpoints for predictions
├── data/                         # Data directory
│   ├── cleaned_delivery_data.csv # Cleaned dataset
│   └── raw.xlsx                  # Raw dataset
├── models/                       # Model implementations
│   ├── __init__.py
│   ├── baseline_models.py        # Logistic Regression & Random Forest
│   └── lstm_model.py             # LSTM implementation with PyTorch
├── notebooks/                    # Jupyter notebooks
│   └── eda.ipynb                 # Exploratory Data Analysis
├── outputs/                      # Generated outputs (created during training)
│   ├── models/                   # Trained model files
│   ├── preprocessor/             # Data preprocessor artifacts
│   └── results/                  # Evaluation results and plots
├── data_preprocessing.py         # Feature engineering & data preparation
├── train.py                      # Training pipeline
├── predict.py                    # Inference module
├── evaluate.py                   # Evaluation and metrics
├── main.py                       # Main entry point
└── pyproject.toml                # Dependencies

```

## Features

### Data Processing
- Comprehensive feature engineering (time-based, sequence-based, route-based)
- Route-level data splitting to prevent leakage
- Sequence preparation for LSTM models
- Scalable preprocessing pipeline

### Models Implemented

1. **Logistic Regression** (Classification)
   - Binary classification: on-time vs delayed
   - Interpretable baseline model
   - Feature importance analysis

2. **Random Forest Classifier** (Classification)
   - Non-linear pattern detection
   - Handles complex feature interactions
   - Feature importance ranking

3. **Random Forest Regressor** (Regression)
   - Predicts delay duration in minutes
   - Robust to outliers
   - Feature importance analysis

4. **LSTM Classifier** (Sequential Classification)
   - Captures temporal dependencies
   - Route sequence learning
   - Previous stop delay propagation

5. **LSTM Regressor** (Sequential Regression)
   - Predicts delay duration with sequence context
   - Temporal pattern recognition
   - Route-level delay accumulation

### Evaluation Metrics

**Classification:**
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion Matrix

**Regression:**
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

## Installation

```bash
cd core
pip install -e .
```

Or using uv:

```bash
cd core
uv sync
```

## Usage

### 1. Run Full Training Pipeline

```bash
cd core
python main.py
```

This will:
- Load and preprocess data
- Engineer features
- Train all 5 models
- Generate evaluation reports
- Save models and results

### 2. Run Training Module Separately

```bash
python train.py
```

### 3. Make Predictions

```bash
python predict.py
```

### 4. Start API Server

```bash
cd api
fastapi dev main.py
```

Then access:
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 5. API Endpoints

**Predict Stop-Level Delays:**
```bash
POST /predict/delays
```

**Predict Route-Level Aggregates:**
```bash
POST /predict/routes
```

**List Available Models:**
```bash
GET /models
```

## Output Structure

After training, the `outputs/` directory will contain:

```
outputs/
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest_classifier.pkl
│   ├── random_forest_regressor.pkl
│   ├── lstm_classifier.pth
│   └── lstm_regressor.pth
├── preprocessor/
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── feature_columns.pkl
└── results/
    ├── evaluation_results_*.json
    ├── evaluation_report.txt
    ├── classification_comparison.csv
    ├── regression_comparison.csv
    ├── classification_comparison.png
    ├── regression_comparison.png
    └── feature_importance_*.png
```

## Model Performance

Models are evaluated on:
- Separate test routes (20% split)
- No data leakage between train and test
- Comprehensive metrics for classification and regression tasks

## Feature Engineering

### Time-based Features
- `hour_of_arrival`: Hour extracted from arrival time
- `time_window_length`: Latest - Earliest time
- `delay_ratio`: Delay relative to time window

### Sequence Features
- `stop_deviation`: Actual vs planned stop index
- `distance_deviation`: Actual vs planned distance ratio
- `stop_position_norm`: Normalized stop position in route
- `prev_stop_delay`: Previous stop's delay
- `cumulative_delay`: Cumulative delay within route

### Route Aggregates
- `route_total_stops`: Total stops in route
- `route_avg_distance`: Average distance per stop
- `route_total_distance`: Total route distance

### Categorical Encodings
- Day of week, Country, Driver ID (label encoded)

## Dependencies

Core libraries:
- pandas, numpy: Data processing
- scikit-learn: Baseline models and preprocessing
- torch: LSTM implementation
- fastapi: REST API
- matplotlib, seaborn: Visualization
- pydantic: Data validation

## Notes

- All models use balanced class weights for classification
- LSTM models support GPU acceleration (automatic detection)
- Feature scaling is applied consistently across train/test
- Preprocessor artifacts are saved for consistent inference

