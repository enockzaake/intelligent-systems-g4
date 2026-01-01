# Proposal: Literature Map & Method Outline

**Team:** Enock Zaake, Nour Ashraf Attia Mohamed, Akmenli Permanova  
**Supervised by:** Prof. Hamidreza Heidari

---

## 1. Problem Statement

### 1.1 Industrial Challenge

Last-mile delivery logistics companies face a critical challenge:

**The Problem:**
- Traditional route optimization algorithms (OR-Tools, genetic algorithms) plan routes based purely on mathematical constraints
- **Real-world drivers frequently deviate from these planned routes** based on their experience and local knowledge
- These deviations often result in **better performance** than the planned routes
- **Key Question:** Can we learn from driver behavior to improve route optimization?

### 1.2 Why This Matters

**Business Impact:**
- Last-mile delivery accounts for **53% of total shipping costs**
- Route inefficiencies lead to:
  - Increased fuel consumption
  - Delayed deliveries (customer dissatisfaction)
  - Driver overtime costs
  - Missed time windows (failed deliveries)

**Current Gap:**
- Existing optimization systems ignore valuable driver knowledge
- Drivers have implicit understanding of:
  - Traffic patterns at different times
  - Time window urgency
  - Local geography and shortcuts
  - Stop clustering and density

---

## 2. Research Objectives

### 2.1 Primary Objective

**Learn optimal route sequences from experienced drivers using Deep Learning**

Instead of relying solely on mathematical optimization, we will:
1. Analyze historical data of planned vs. actual routes
2. Train a neural network to predict the sequences drivers actually follow
3. Compare performance: Traditional Optimization vs. Deep Learning vs. Actual Driver Routes

### 2.2 Specific Goals

1. **Delay Prediction:** Predict which stops are likely to be delayed
   - Input: Stop characteristics, time windows, route features
   - Output: Delay probability and expected delay minutes

2. **Route Sequence Optimization:** Learn optimal visit order
   - Input: Set of stops with features
   - Output: Predicted optimal sequence (learned from drivers)

3. **Performance Comparison:** Quantify improvements
   - Measure: Distance reduction, time savings, sequence correlation
   - Compare: OR-Tools (planned) vs. DL Model vs. Actual (drivers)

---

## 3. Literature Review Summary

### 3.1 Classical Vehicle Routing Problem (VRP)

**Foundational Work:**
- Dantzig & Ramser (1959): First formulation of VRP
- NP-hard combinatorial optimization problem
- Variants: VRPTW (time windows), CVRP (capacity), DVRP (dynamic)

**Modern Approaches:**
- OR-Tools (Google): Constraint programming + local search
- Genetic Algorithms & Simulated Annealing: Population-based optimization
- **Limitation:** None learn from historical data or driver behavior

### 3.2 Machine Learning for Routing

**Traditional ML:**
- Feature-based prediction (delays, travel times)
- Models: Random Forest, Gradient Boosting
- **Strength:** Good for single-point predictions
- **Weakness:** Cannot optimize entire route sequence

**Reinforcement Learning:**
- Q-learning for dynamic route selection
- **Strength:** Adapts to dynamic conditions
- **Weakness:** Requires extensive simulation, slow convergence

### 3.3 Deep Learning for Combinatorial Optimization

**Key References:**
- **Kool et al. (2019):** "Attention, Learn to Solve Routing Problems!"
  - Transformer architecture for VRP
  - Matches OR-Tools performance, faster inference
  - **Relevance:** Demonstrates feasibility of learning routing from data

- **Vinyals et al. (2015):** "Pointer Networks"
  - First neural approach to combinatorial problems
  - Sequence-to-sequence with attention

**Our Innovation:**
- Apply supervised learning from expert demonstrations (drivers)
- Learn implicit knowledge (traffic, geography, urgency)
- First application of Transformer to learning from driver behavior

### 3.4 Research Gap

**Gap 1: Learning from Driver Behavior**
- Existing: Algorithms optimize based on mathematical models
- Missing: Learning from experienced drivers' decisions
- **Our Approach:** Train DL model on actual vs. planned sequences

**Gap 2: End-to-End Route Optimization with Delays**
- Existing: Delay prediction separate from route optimization
- Missing: Integrated system that predicts delays AND optimizes routes
- **Our Approach:** Two-phase ML+DL pipeline

**Gap 3: Validation Against Human Performance**
- Existing: Models compared only to other algorithms
- Missing: Comparison with actual driver performance
- **Our Approach:** Three-way comparison (OR-Tools vs. DL vs. Actual)

---

## 4. Proposed Solution Overview

### 4.1 Two-Phase Approach

#### Phase 1: Delay Prediction (ML)

**Models:** Logistic Regression, Random Forest, LSTM

**Purpose:** Identify high-risk stops that may cause delays

**Output:** Delay probabilities for each stop

**Key Features:**
- Temporal: Hour of day, time window width, urgency
- Sequential: Position in route, previous stop delay, cumulative delay
- Route aggregates: Total stops, average distance, driver patterns

**Implementation:** `solution_1_ml/`

#### Phase 2: Route Sequence Learning (DL)

**Model:** Transformer with Attention Mechanism

**Purpose:** Learn optimal stop visit order from driver behavior

**Output:** Predicted sequence for new routes

**Architecture:**
- Feature embedding (128 dimensions)
- Transformer encoder (3 layers, 8 attention heads)
- Sequence decoder

**Why Transformer?**
- Attention mechanism captures dependencies between stops
- Learns which stops should be grouped together
- Handles variable route sizes
- State-of-the-art for sequence tasks

**Implementation:** `solution_2_dl/`

### 4.2 Key Innovation

**Learning from Human Expertise:**
- Traditional algorithms: Rules + constraints → optimal route
- **Our approach:** Historical driver behavior → learned patterns → optimal route

**Advantages:**
- Captures implicit knowledge (traffic, geography, time-of-day effects)
- Adapts to local conditions automatically
- Continuous improvement as more data becomes available
- Fast inference (<1 second) after training

---

## 5. Data Description

### 5.1 Dataset Source

**Last-mile delivery route deviations dataset** (Konovalenko et al., 2024)
- Published: Mendeley Data (DOI: 10.17632/kkwgfvmtxn.1)
- Coverage: Multiple countries, multiple weeks
- Routes: 1,000+ delivery routes with planned vs. actual sequences

### 5.2 Data Structure

Each row = one stop within a route

**Key Features:**
- **Identifiers:** Route ID, Driver ID, Stop ID, Address ID
- **Sequences:** 
  - `IndexP` (Planned position in route)
  - `IndexA` (Actual position driver visited)
- **Spatial:** Distances between stops (planned vs. actual)
- **Temporal:** Arrival times, time windows (earliest/latest)
- **Context:** Country, day of week, depot/delivery flags

**Target Variables:**
1. **Classification:** Delayed or on-time (binary)
2. **Regression:** Delay minutes (continuous)
3. **Sequence:** Optimal visit order (permutation)

**Data Location:** `data/raw.xlsx`

---

## 6. Methodology Preview

### 6.1 Overall Pipeline

```
Raw Data → Data Cleaning → Feature Engineering → 
Solution 1 (ML Delay Prediction) → Solution 2 (DL Route Learning) → 
Evaluation & Validation
```

### 6.2 Phase 1: ML Delay Prediction

**Process:**
1. Load and clean data
2. Engineer features (temporal, sequential, route-level)
3. Split data (route-aware to prevent leakage)
4. Train models (Logistic Regression, Random Forest, LSTM)
5. Evaluate and compare models

**Implementation:** `solution_1_ml/train.py`

### 6.3 Phase 2: DL Route Sequence Learning

**Process:**
1. Prepare sequence data (planned vs. actual)
2. Train Transformer model on route sequences
3. Evaluate sequence predictions
4. Compare with OR-Tools and actual driver sequences

**Implementation:** `solution_2_dl/train_dl_model.py`

### 6.4 Evaluation Framework

**Baseline Comparisons:**
- Simple heuristics (majority class, route mean)
- Traditional OR-Tools optimization

**Validation Methods:**
- 5-fold cross-validation
- Temporal validation (train on early weeks, test on later weeks)
- Statistical tests (McNemar's test, paired t-test)

**Implementation:** `solution_1_ml/validation/`

---

## 7. Expected Outcomes

### 7.1 Quantifiable Deliverables

1. **Trained ML Models:**
   - Delay prediction accuracy: Target >80%
   - Recall for high-risk stops: Target >75%

2. **DL Route Optimizer:**
   - Sequence correlation (Kendall Tau): Target >0.7
   - Improvement over planned routes: Target >10%

3. **Interactive Dashboard:**
   - Real-time delay prediction
   - Route visualization (planned vs. predicted vs. actual)
   - Scenario testing interface

4. **Performance Metrics:**
   - Classification: Accuracy, Precision, Recall, F1, ROC-AUC
   - Sequence: Kendall Tau, Edit Distance, Spearman Correlation
   - Business: Distance reduction, time savings

### 7.2 Validation Strategy

1. **Baseline Comparison:**
   - Compare against simple heuristics (majority class, route mean)
   - Compare against traditional OR-Tools optimization

2. **Cross-Validation:**
   - 5-fold cross-validation for ML models
   - Temporal validation (train on early weeks, test on later weeks)

3. **Statistical Testing:**
   - McNemar's test (paired predictions)
   - Paired t-test (continuous metrics)

---

## 8. Project Timeline

### Week 1-2: Data Preparation & EDA
- Load and clean dataset
- Exploratory data analysis
- Feature engineering
- Train/test split (route-aware)

**Deliverables:**
- Cleaned dataset
- EDA report
- Feature engineering pipeline

### Week 3-4: ML Model Development
- Train baseline models
- Train classification models (LR, RF, LSTM)
- Train regression models
- Hyperparameter tuning

**Deliverables:**
- Trained ML models
- Model comparison report
- Feature importance analysis

### Week 5-6: DL Model Development
- Implement Transformer architecture
- Create dataset for sequence learning
- Train DL route optimizer
- Evaluate sequence predictions

**Deliverables:**
- Trained DL model
- Sequence prediction results
- Comparison with OR-Tools

### Week 7-8: Integration & Evaluation
- Build comparison framework
- Statistical testing
- Dashboard development
- Final experiments & analysis

**Deliverables:**
- Complete validation report
- Interactive dashboard
- Final results and analysis

---

## 9. Success Criteria

### Minimum Viable Product (MVP)

✓ ML models trained and evaluated (accuracy >75%)  
✓ DL model trained (sequence correlation >0.6)  
✓ Comparison with baselines completed  
✓ Basic visualization dashboard

### Stretch Goals

✓ Real-time prediction API  
✓ Interactive scenario testing  
✓ Integration with existing routing systems  
✓ Multi-objective optimization (distance + time + delays)

---

## 10. Risk Assessment & Mitigation

### Risk 1: Model Overfitting

**Risk:** Model learns training routes too well, poor generalization

**Mitigation:**
- Route-aware splitting (no route in both train and test)
- Cross-validation
- Regularization techniques

### Risk 2: Sequence Prediction Complexity

**Risk:** Combinatorial explosion (n! possible sequences)

**Mitigation:**
- Use attention mechanism (handles variable sizes)
- Limit max route size if needed
- Focus on learning patterns, not exact sequences

### Risk 3: Data Imbalance

**Risk:** More on-time stops than delayed stops

**Mitigation:**
- Class weighting in loss function
- Focus on recall metric (detect delays)
- SMOTE for synthetic minority samples (if needed)

### Risk 4: Computational Resources

**Risk:** DL training requires GPUs

**Mitigation:**
- Use smaller models initially
- Batch training
- Cloud resources if needed

---

## 11. Novel Contributions

1. **First application of Transformer architecture to learning from driver behavior**
   - Not just solving VRP, but learning driver strategies

2. **Hybrid ML+DL architecture**
   - Phase 1 (ML): Delay prediction → identifies problematic stops
   - Phase 2 (DL): Route optimization → learns optimal sequences

3. **Comprehensive validation framework**
   - Baseline comparisons (majority class, route mean)
   - Statistical tests (McNemar, paired t-test)
   - Temporal validation (realistic test setup)

4. **Interactive demonstration system**
   - Real-time route visualization
   - Scenario testing with actual dataset routes

---

## 12. Practical Impact

### For Logistics Companies
- Reduce delivery costs by 5-15%
- Improve on-time delivery rates
- Better driver route assignments
- Reduced fuel consumption (environmental benefit)

### For Drivers
- Routes that match their intuition
- Less stressful delivery schedules
- Fewer failed deliveries

---

## 13. Baseline Code & Simulation

### 13.1 Baseline Models

**Baseline 1: Majority Class**
- Always predict "no delay"
- Establishes minimum performance

**Baseline 2: Route Mean**
- Predict average delay rate for each route
- Simple but route-aware

**Baseline 3: OR-Tools (Planned Sequence)**
- Traditional optimization algorithm
- Our target to beat

**Implementation:** `solution_1_ml/validation/baseline_comparison.py`

### 13.2 Simulation System

**Purpose:** Test scenarios before deployment

**Capabilities:**
- Traffic level adjustments
- Weather conditions
- Driver reassignments
- Time window modifications
- Stop removals

**Output:**
- Delay predictions for original vs. modified scenarios
- Route optimization results
- Comparison metrics (improvement percentages)

**Implementation:** `testing/simulation_engine.py`

---

## 14. References

For complete bibliography, see `04_BIBLIOGRAPHY.md`

**Key References:**
1. Konovalenko, A., et al. (2024). "Last-mile delivery route deviations dataset." *Mendeley Data*.
2. Kool, W., et al. (2019). "Attention, Learn to Solve Routing Problems!" *ICLR*.
3. Psaraftis, H. N., et al. (2016). "Dynamic vehicle routing problems: Three decades and counting." *Networks*.

---
