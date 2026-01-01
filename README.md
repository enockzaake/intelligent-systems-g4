# AI-Driven Route Optimization and Delay Prediction System

**Team:** Enock Zaake, Nour Ashraf Attia Mohamed, Akmenli Permanova  
**Supervised by:** Prof. Hamidreza Heidari

---

## Problem Statement

### The Challenge

Last-mile delivery logistics companies face a critical challenge: traditional route optimization algorithms (OR-Tools, genetic algorithms) plan routes based purely on mathematical constraints—minimizing distance, respecting time windows, and optimizing capacity.

**However, real-world drivers frequently deviate from these planned routes** based on their experience and local knowledge. These deviations often result in **better performance** than the planned routes.

### Why This Matters

- Last-mile delivery accounts for **53% of total shipping costs**
- Drivers deviate from planned routes **40% of the time**
- When drivers deviate, **73% of the time they improve performance**
- Drivers have implicit knowledge that algorithms lack:
  - Traffic patterns at different times of day
  - Time window urgency (which stops can't be missed)
  - Local geography and shortcuts
  - Stop clustering strategies

### Research Question

**Can we train AI models to learn from driver behavior and predict better routes than traditional optimization algorithms?**

---

## Solution Approach

We propose a **two-phase hybrid approach** that combines machine learning and deep learning:

### Phase 1: Delay Prediction (ML)

**Problem:** Predict which stops are likely to be delayed before they happen.

**Solution:** Train traditional ML models (Logistic Regression, Random Forest, LSTM) on historical route data to predict delay probabilities.

**Key Features:**
- Temporal features (hour of day, time windows, urgency)
- Sequential features (position in route, previous delays, cumulative delays)
- Route aggregates (total stops, average distance, driver performance)

**Result:** Random Forest achieves **95% accuracy** and **88% recall** in predicting delays.

### Phase 2: Route Sequence Learning (DL)

**Problem:** Learn the optimal stop visit order from experienced drivers.

**Solution:** Train a Transformer neural network with attention mechanism on planned vs. actual route sequences to learn driver decision patterns.

**Key Innovation:**
- **First application of Transformer architecture to learning from driver behavior**
- Attention mechanism naturally captures:
  - Time window constraints (urgency)
  - Geographic clustering (nearby stops)
  - Sequence dependencies (previous stop affects next)

**Result:** DL model achieves **71% Kendall Tau correlation** with actual driver sequences, **36% better** than OR-Tools planned routes.

---

## Codebase Walkthrough

This codebase is organized to clearly show the step-by-step problem-solving process:

```
Start → Data → Cleaning → EDA → Solution 1 (ML) → Solution 2 (DL) → Testing
```

### Step 1: Data Preparation
**Location:** `data/` and `notebooks/01_data_cleaning.ipynb`

- Raw data: `data/raw.xlsx` (original dataset)
- Cleaning notebook: `notebooks/01_data_cleaning.ipynb`
- Output: `data/cleaned_delivery_data.csv`

**What happens here:** Load raw data, handle missing values, convert timestamps, create target variables (delay flags and delay minutes).

### Step 2: Exploratory Data Analysis
**Location:** `notebooks/02_eda.ipynb`

- Analyze data distributions
- Identify patterns and relationships
- Visualize delay patterns, route characteristics, driver behavior

**What happens here:** Understand the data before modeling—when do delays occur? Which routes are problematic? How do drivers deviate?

### Step 3: Solution 1 - ML Delay Prediction
**Location:** `solution_1_ml/`

**Training:** `solution_1_ml/train.py`
- Trains Logistic Regression, Random Forest, and LSTM models
- Feature engineering and preprocessing
- Model evaluation and comparison

**Prediction:** `solution_1_ml/predict.py`
- Load trained models
- Predict delays for new routes

**Models:** `solution_1_ml/models/`
- Baseline models (Logistic Regression)
- Advanced models (Random Forest, LSTM)
- Ensemble models

**Validation:** `solution_1_ml/validation/`
- Baseline comparison
- Cross-validation
- Temporal validation
- Statistical significance tests

**Output:** Delay predictions with probabilities and risk classifications (HIGH/MEDIUM/LOW).

### Step 4: Solution 2 - DL Route Sequence Learning
**Location:** `solution_2_dl/`

**Training:** `solution_2_dl/train_dl_model.py`
- Trains Transformer model on route sequences
- Learns from planned vs. actual sequences
- Attention mechanism captures stop relationships

**Prediction:** `solution_2_dl/dl_predict.py`
- Predicts optimal route sequence for new routes
- Compares with planned and actual sequences

**Model:** `solution_2_dl/dl_route_optimizer.py`
- Transformer architecture (128-dim embeddings, 8 attention heads, 3 layers)
- Sequence decoder that predicts visit order

**Output:** Optimal route sequences that match driver behavior patterns.

### Step 5: Testing & UI Interface
**Location:** `testing/`

**API:** `testing/api/`
- `server.py` - ML + OR-Tools approach
- `server_v2.py` - Pure DL approach
- REST endpoints for predictions and optimization

**Dashboard:** `testing/dashboard/`
- Next.js frontend for interactive testing
- Visualize predictions and route sequences
- Compare planned vs. predicted vs. actual routes

**Testing:** `testing/test_models.py`
- Unit tests and integration tests
- Model performance validation

---

## Key Results

### Delay Prediction (Solution 1)
- **Random Forest:** 95% accuracy, 88% recall, 0.98 ROC-AUC
- Catches 88% of all delays before they happen
- Identifies high-risk stops for proactive intervention

### Route Sequence Learning (Solution 2)
- **DL Transformer:** 71% Kendall Tau correlation with driver sequences
- **36% improvement** over OR-Tools planned routes
- **10-60x faster** inference than OR-Tools (0.5s vs. 5-30s)

### Business Impact
- **$700K-$800K annual savings** for medium-sized fleet (100 vehicles)
- **5-10% distance reduction** through better route sequences
- **88% delay detection rate** enables proactive mitigation

---

## Documentation

Detailed documentation is available in 4 main files:

1. **01_PROPOSAL.md** - Problem statement, literature review, method outline, baseline code
2. **02_METHODOLOGY.md** - Detailed methodology, demo walkthrough, experiments & analysis
3. **03_FINAL_REPORT.md** - Complete final report with all results
4. **04_BIBLIOGRAPHY.md** - Bibliography

---

## Quick Start

**Install dependencies:**
```bash
pip install -r requirements.txt
cd testing/dashboard && npm install
```

**Train models:**
```bash
# ML models
python solution_1_ml/train.py --dataset data/cleaned_delivery_data.csv

# DL model
python solution_2_dl/train_dl_model.py --data data/prepared_raw_data.csv --epochs 50
```

**Run system:**
```bash
# API server
python testing/api/server_v2.py

# Dashboard (in another terminal)
cd testing/dashboard && npm run dev
```

---

**Last Updated:** December 2024
