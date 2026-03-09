# ICAIF ML Contribution Enhancement Plan

## Current State (Observational)
- Finding: "SMB-target predictability decays fastest"
- Method: HMM + Granger (classic stats from 1960s)
- Application: Diagnostic only
- **Problem**: Looks like Applied Econometrics, not "AI in Finance"

## Target State (ML Contribution)
Transform from observational study to actionable ML framework

---

## Proposed ML Contributions

### 1. **Decay Prediction Model** (Primary ML Contribution)
**Goal**: Predict WHEN a factor relationship will decay

**Approach**:
- Input features: Rolling Granger F-stat, regime volatility, factor crowding proxies
- Target: Time to significance loss (survival analysis)
- Models: Cox PH, Random Survival Forest, DeepSurv

**Deliverable**:
- Predict decay onset for NEW factor pairs
- Out-of-sample validation on held-out pairs

### 2. **Online Regime Detection** (Real-time ML)
**Goal**: Detect regime changes in real-time for trading systems

**Approach**:
- Online HMM with Bayesian updating
- Compare: Batch HMM vs Online HMM vs BOCPD (Bayesian Online Changepoint Detection)

**Deliverable**:
- Streaming regime classifier
- Latency vs accuracy tradeoff analysis

### 3. **Neural Granger with Regime Conditioning** (Deep Learning)
**Goal**: Learn nonlinear regime-conditional causal structure

**Approach**:
- Extend Tank et al. (2022) Neural Granger to regime-conditional setting
- Architecture: LSTM encoder → regime embedding → causal graph decoder

**Deliverable**:
- Compare linear Granger vs Neural Granger per regime
- Show where nonlinearity matters

### 4. **Factor Crowding Detector** (Practical ML Tool)
**Goal**: Predict which factor pairs are at risk of decay

**Approach**:
- Features: AUM flows, factor momentum, correlation clustering
- Target: Binary (will decay in next 5 years?)
- Model: Gradient boosting, calibrated probabilities

**Deliverable**:
- Risk score for each factor pair
- Backtested on historical decays

---

## Implementation Priority

| Contribution | Effort | ICAIF Impact | Priority |
|--------------|--------|--------------|----------|
| Decay Prediction | Medium | High | **1** |
| Online Regime | Medium | High | **2** |
| Neural Granger | High | Medium | 3 |
| Crowding Detector | Medium | Medium | 4 |

---

## Plan: Implement #1 and #2 first

### Step 1: Decay Prediction Model
- Use 30 factor pairs
- Train on 20 pairs, test on 10 pairs
- Predict: Will this pair show decay? When?
- Evaluate: C-index, calibration

### Step 2: Online Regime Detection
- Implement online HMM
- Compare to batch HMM regime assignments
- Measure detection delay

### Step 3: Update Paper
- New section: "Predictive Models"
- Algorithm 2: Decay prediction
- Algorithm 3: Online regime detection
- Emphasize ML contribution in abstract/title

---

## Success Criteria
- [ ] At least one model with OOS predictive accuracy
- [ ] Reusable algorithm (not just empirical observation)
- [ ] Clear ML contribution beyond classic econometrics
- [ ] Actionable output (risk score, real-time signal)
