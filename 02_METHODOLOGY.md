# Methodology: Step-by-Step Problem Solving & Results Evaluation

**Team:** Enock Zaake, Nour Ashraf Attia Mohamed, Akmenli Permanova  
**Supervised by:** Prof. Hamidreza Heidari

---

## Overview

This document describes the step-by-step methodology for solving the route optimization problem, from data preparation through model training to results evaluation. For implementation details, refer to the codebase locations mentioned in each section.

---

## Step 1: Data Preparation

### 1.1 Data Source

**Dataset:** Last-mile delivery route deviations dataset (Konovalenko et al., 2024)
- **Size:** 240,184 stops from 1,043 routes
- **Coverage:** Multiple countries (Netherlands, Spain, Italy, Germany, UK)
- **Key Feature:** Contains both planned routes (from OR-Tools) and actual routes (from drivers)

**Location:** `data/raw.xlsx`

### 1.2 Data Cleaning

**Process:**
1. Load raw Excel data
2. Handle missing values
3. Convert timestamps to datetime format
4. Create target variables:
   - `delay_flag`: Binary (1 if delayed, 0 if on-time)
   - `delay_minutes`: Continuous (minutes of delay)

**Implementation:** `notebooks/01_data_cleaning.ipynb`

**Output:** `data/cleaned_delivery_data.csv`

### 1.3 Exploratory Data Analysis

**Analysis Performed:**
- Distribution of delays (when do delays occur?)
- Route characteristics (average stops per route, route lengths)
- Driver behavior patterns (deviation rates, performance differences)
- Time patterns (hourly, daily, weekly patterns)

**Key Findings:**
- Drivers deviate from planned routes 40% of the time
- 73% of deviations improve performance
- Delays cluster at certain times of day
- Some drivers consistently perform better than others

**Implementation:** `notebooks/02_eda.ipynb`

---

## Step 2: Feature Engineering

### 2.1 Feature Categories

**Temporal Features:**
- Hour of arrival (0-23)
- Time window width (latest_time - earliest_time)
- Time window urgency (how close to deadline)
- Day of week (weekday vs. weekend)

**Sequential Features:**
- Position in route (normalized 0-1)
- Previous stop's delay (lag feature)
- Cumulative delay (running sum in route)
- Sequence deviation (planned vs. actual position)

**Route-Level Features:**
- Total stops in route
- Average distance per stop
- Route delay rate (historical)
- Driver's historical delay rate

**Implementation:** `utils/data_preprocessing.py`

### 2.2 Data Splitting

**Critical Requirement:** Route-aware splitting to prevent data leakage

**Method:**
- Split by `route_id` (not by individual stops)
- 80% of routes → training set
- 20% of routes → test set
- No route appears in both sets

**Why This Matters:** Routes have internal structure. If stops from the same route are in both train and test, the model "sees" the test route during training.

**Implementation:** `solution_1_ml/train.py` (see `_split_by_route` function)

---

## Step 3: Solution 1 - ML Delay Prediction

### 3.1 Problem Formulation

**Goal:** Predict which stops will be delayed before they happen.

**Input:** Stop features (temporal, sequential, route-level)
**Output:** 
- Binary classification: Will this stop be delayed? (0 or 1)
- Regression: How many minutes of delay? (continuous)

### 3.2 Models Evaluated

**Model 1: Logistic Regression**
- Purpose: Interpretable baseline
- Implementation: `solution_1_ml/models/baseline_models.py`
- Result: 78.5% accuracy, 79.6% recall

**Model 2: Random Forest Classifier**
- Purpose: Capture non-linear interactions
- Implementation: `solution_1_ml/models/baseline_models.py`
- Hyperparameters: 200 trees, max depth 20
- Result: **94.8% accuracy, 88.2% recall, 76.8% F1-score** (Best performer)

**Model 3: LSTM Network**
- Purpose: Capture temporal dependencies in route sequences
- Implementation: `solution_1_ml/models/lstm_model.py`
- Architecture: Bidirectional LSTM with 2 layers, 64 hidden units
- Result: 65.7% accuracy, 66.7% recall

### 3.3 Training Process

**Steps:**
1. Load cleaned data
2. Engineer features
3. Split data (route-aware)
4. Train each model
5. Evaluate on test set
6. Compare models

**Implementation:** `solution_1_ml/train.py`

**Output:** Trained models saved to `solution_1_ml/outputs/models/`

### 3.4 Key Insights

**Top Features (Random Forest):**
1. Cumulative delay (0.142) - Delays accumulate along route
2. Route average distance (0.098) - Longer routes more prone to delays
3. Time window urgency (0.087) - Tight windows increase risk
4. Previous stop delay (0.076) - Cascading effect

**Finding:** Temporal dependencies are crucial. Early delays affect later stops in the same route.

---

## Step 4: Solution 2 - DL Route Sequence Learning

### 4.1 Problem Formulation

**Goal:** Learn the optimal stop visit order from experienced drivers.

**Input:** Set of n stops with features (no inherent order)
**Output:** Predicted visit sequence (permutation of stops)

**Learning Signal:** Actual sequences from drivers (supervised learning)

### 4.2 Model Architecture

**Transformer with Attention Mechanism**

**Components:**
1. **Feature Embedding:** Maps stop features to 128-dimensional vectors
2. **Positional Encoding:** Learnable embeddings for stop positions
3. **Transformer Encoder:** 3 layers, 8 attention heads
   - Attention mechanism learns relationships between stops
   - Captures time window constraints, geographic clustering, dependencies
4. **Sequence Decoder:** Predicts position for each stop

**Implementation:** `solution_2_dl/dl_route_optimizer.py`

**Why Transformer?**
- Attention naturally captures routing logic
- No hand-crafted heuristics needed
- Learns which stops should be visited together

### 4.3 Training Process

**Steps:**
1. Prepare sequence data (planned vs. actual sequences)
2. Create batches of routes
3. Train Transformer model (50 epochs)
4. Validate on held-out routes
5. Evaluate sequence predictions

**Implementation:** `solution_2_dl/train_dl_model.py`

**Output:** Trained model saved to `solution_2_dl/outputs/dl_models/best_model.pt`

### 4.4 Key Results

**Performance Metrics:**
- Sequence Accuracy: 48.9% (nearly 1 in 2 stops in correct position)
- Kendall Tau: 0.712 (strong correlation with driver sequences)
- Improvement over OR-Tools: +36% correlation

**Finding:** DL model successfully learns driver strategies (time window urgency, geographic clustering).

---

## Step 5: Results Evaluation

### 5.1 Delay Prediction Evaluation

**Metrics Used:**
- **Accuracy:** Overall correctness (94.8% for Random Forest)
- **Recall:** % of delays caught (88.2% - most important for this application)
- **Precision:** % of predicted delays that are correct (68.1%)
- **F1-Score:** Harmonic mean of precision and recall (76.8%)
- **ROC-AUC:** Discrimination ability (0.982)

**Confusion Matrix (Random Forest):**
- True Negatives: 42,940 (86.1%) - Correctly identified on-time stops
- True Positives: 4,323 (8.7%) - Correctly predicted delays
- False Positives: 2,025 (4.1%) - False alarms (acceptable)
- False Negatives: 581 (1.2%) - Missed delays (acceptable)

**Implementation:** `solution_1_ml/validation/evaluate.py`

### 5.2 Route Sequence Evaluation

**Metrics Used:**
- **Kendall Tau:** Rank correlation between predicted and actual sequences (0.712)
- **Sequence Accuracy:** % of stops in exact correct position (48.9%)
- **Edit Distance:** Number of swaps needed to match actual (6.1 swaps)
- **Spearman Correlation:** Alternative rank correlation (0.748)

**Comparison:**
- OR-Tools (Planned) vs. Actual: 0.523 Kendall Tau
- DL Model vs. Actual: 0.712 Kendall Tau
- **Improvement: +36%**

**Implementation:** `solution_2_dl/dl_predict.py`

### 5.3 Validation Framework

**Baseline Comparison:**
- Compare ML models against simple baselines (majority class, route mean)
- **Result:** Random Forest outperforms best baseline by 81%
- **Implementation:** `solution_1_ml/validation/baseline_comparison.py`

**Cross-Validation:**
- 5-fold cross-validation to assess model stability
- **Result:** F1 = 0.7684 ± 0.0298 (very stable)
- **Implementation:** `solution_1_ml/validation/cross_validation.py`

**Temporal Validation:**
- Train on early weeks, test on later weeks
- **Result:** Only 2.53% performance drop (good generalization)
- **Implementation:** `solution_1_ml/validation/temporal_validation.py`

**Statistical Tests:**
- McNemar's test for classification models
- Paired t-test for continuous metrics
- **Result:** P-value < 0.001 (highly significant improvements)
- **Implementation:** `solution_1_ml/validation/statistical_tests.py`

**Run All Validations:**
```bash
python solution_1_ml/validation/comprehensive_validation.py
```

---

## Step 6: System Integration & Testing

### 6.1 API Development

**Backend API (FastAPI):**
- Endpoints for delay prediction
- Endpoints for route sequence prediction
- Endpoints for model evaluation

**Implementation:** 
- ML + OR-Tools: `testing/api/server.py`
- Pure DL: `testing/api/server_v2.py`

### 6.2 Dashboard Interface

**Frontend (Next.js/React):**
- Route selection from test dataset
- Real-time prediction visualization
- Sequence comparison (planned vs. actual vs. predicted)
- Performance metrics display

**Implementation:** `testing/dashboard/`

### 6.3 Testing

**Model Testing:**
- Unit tests for individual components
- Integration tests for full pipeline
- Performance benchmarking

**Implementation:** `testing/test_models.py`

---

## Experimental Results

### ML Delay Prediction Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Majority Class (Baseline) | 90.3% | 0.0% | 0.0% | 0.0% | 0.500 |
| Route Mean | 74.2% | 18.5% | 62.1% | 28.4% | 0.712 |
| Logistic Regression | 78.5% | 28.6% | 79.6% | 42.1% | 0.872 |
| **Random Forest** | **94.8%** | **68.1%** | **88.2%** | **76.8%** | **0.982** |
| LSTM | 65.7% | 17.5% | 66.7% | 27.7% | 0.719 |

**Key Finding:** Random Forest is the clear winner, achieving 95% accuracy and 88% recall.

### DL Route Sequence Results

| Metric | OR-Tools (Planned) | DL Transformer | Improvement |
|--------|-------------------|----------------|-------------|
| Kendall Tau | 0.523 | **0.712** | **+36%** |
| Spearman ρ | 0.587 | **0.748** | **+27%** |
| Sequence Accuracy | 31.2% | **48.9%** | **+57%** |
| Edit Distance | 8.3 swaps | **6.1 swaps** | **-26%** |

**Key Finding:** DL model significantly outperforms traditional OR-Tools optimization.

### Cross-Validation Results

**Random Forest (5-fold CV):**
- Accuracy: 94.8% ± 0.2%
- Recall: 88.1% ± 0.5%
- F1-Score: 76.7% ± 0.5%
- ROC-AUC: 0.982 ± 0.002

**Interpretation:** Very low variance across folds → model is stable and reliable.

### Temporal Validation Results

**Train on Weeks 1-3, Test on Week 4:**
- Random Forest: 93.9% accuracy, 86.5% recall
- Performance drop: Only 0.9% (excellent generalization)

**Interpretation:** Model generalizes well to future routes.

### Statistical Significance

**McNemar's Test (Random Forest vs. Logistic Regression):**
- P-value: < 0.001
- Conclusion: Random Forest is significantly better

**Paired T-Test (DL vs. Planned Kendall Tau):**
- T-statistic: 12.47
- P-value: < 0.001
- Conclusion: DL model is significantly better than planned routes

---

## Key Insights & Findings

### What We Learned

1. **Drivers are smarter than algorithms**
   - 40% deviation rate, 73% improve performance
   - Implicit knowledge (traffic, geography, urgency) can be learned

2. **Traditional ML excels at delay prediction**
   - Random Forest beats LSTM
   - Feature engineering > model complexity
   - 95% accuracy achievable with proper features

3. **Deep learning captures routing logic**
   - Transformer learns driver strategies naturally
   - Attention mechanism captures time windows, clustering, dependencies
   - 36% improvement over OR-Tools

4. **Hybrid approach is optimal**
   - Phase 1 (ML): Identify problematic stops
   - Phase 2 (DL): Optimize entire sequence
   - Combined system leverages strengths of both

### Business Impact

**For medium-sized fleet (100 vehicles):**
- Delay reduction: $577,500/year
- Route efficiency: $93,750-$187,500/year
- Driver productivity: $62,500/year
- **Total: $733,750-$827,500/year**
- **ROI: 734% first year**

---

## Codebase Reference

### Data Processing
- Data cleaning: `notebooks/01_data_cleaning.ipynb`
- EDA: `notebooks/02_eda.ipynb`
- Feature engineering: `utils/data_preprocessing.py`

### Solution 1 (ML)
- Training: `solution_1_ml/train.py`
- Prediction: `solution_1_ml/predict.py`
- Models: `solution_1_ml/models/`
- Validation: `solution_1_ml/validation/`

### Solution 2 (DL)
- Training: `solution_2_dl/train_dl_model.py`
- Prediction: `solution_2_dl/dl_predict.py`
- Model: `solution_2_dl/dl_route_optimizer.py`

### Testing & UI
- API: `testing/api/`
- Dashboard: `testing/dashboard/`
- Testing: `testing/test_models.py`

---

## Summary

This methodology demonstrates a complete pipeline from raw data to deployed system:

1. **Data Preparation:** Clean and understand the data
2. **Feature Engineering:** Create meaningful features
3. **ML Delay Prediction:** Predict which stops will be delayed (95% accuracy)
4. **DL Route Learning:** Learn optimal sequences from drivers (71% correlation)
5. **Evaluation:** Comprehensive validation with statistical significance
6. **Integration:** API and dashboard for interactive testing

The results show that learning from driver behavior significantly outperforms traditional optimization algorithms, with both statistical significance and practical business value.

---

**Last Updated:** December 2024
