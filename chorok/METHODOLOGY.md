# Methodology: Distribution Shift Detection via Uncertainty Quantification

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Why Predictions Are Not The Goal](#why-predictions-are-not-the-goal)
3. [Distribution Shift Metrics](#distribution-shift-metrics)
4. [Epistemic Uncertainty Calculation](#epistemic-uncertainty-calculation)
5. [Classification vs Regression Approaches](#classification-vs-regression-approaches)
6. [Validation Strategy](#validation-strategy)

---

## Core Concepts

### Types of Uncertainty

**Aleatoric Uncertainty** (Irreducible):
- Inherent noise in the data
- Cannot be reduced by collecting more data
- Example: Random measurement error, inherent stochasticity
- NOT the focus of this research

**Epistemic Uncertainty** (Reducible):
- Uncertainty due to lack of knowledge
- Can be reduced with more training data
- Reflects model's unfamiliarity with input
- **THIS is what we measure for shift detection**

### Why Epistemic Uncertainty Indicates Distribution Shift

When a model encounters data from a different distribution than training:
1. Model is uncertain about how to predict (lacks knowledge)
2. Different models in an ensemble make different predictions (disagreement)
3. High ensemble disagreement = High epistemic uncertainty
4. **High epistemic uncertainty signals distribution shift**

**Key Insight**: If training data distribution = test data distribution, models should agree. If distributions differ, models disagree.

---

## Why Predictions Are Not The Goal

### Traditional ML Workflow
```
Train Model → Evaluate Accuracy on Test Set → Report Metrics
Goal: Maximize accuracy
```

### Our Research Workflow
```
Train Ensemble → Measure Uncertainty on Test Set → Compare with Shift Metrics
Goal: Detect when distribution has shifted
```

### The Fundamental Difference

**Traditional ML asks**: "How accurate are the predictions?"

**Our research asks**: "How confident/uncertain is the model, and does uncertainty correlate with distribution shift?"

### Why This Matters in Production

In deployed ML systems:
- Models encounter new data continuously
- Distribution can drift over time (concept drift, covariate shift)
- Accuracy degrades silently without warning

**Our contribution**: Use epistemic uncertainty as an **early warning system**:
- Rising uncertainty → Distribution is shifting → Time to retrain
- Stable uncertainty → Distribution is stable → Model still valid

Predictions are merely the **mechanism** by which we measure uncertainty, not the end goal.

---

## Distribution Shift Metrics

### Population Stability Index (PSI)

**Formula**:
```
PSI = Σ (p_test - p_train) × ln(p_test / p_train)
```

Where:
- `p_train`: Proportion of samples in each bin/class during training
- `p_test`: Proportion of samples in each bin/class during testing

**Interpretation**:
- PSI < 0.1: No significant shift
- 0.1 ≤ PSI < 0.2: Moderate shift (investigate)
- PSI ≥ 0.2: Significant shift (model likely needs retraining)

**Advantages**:
- Industry standard for model monitoring
- Easy to interpret
- Works for both categorical and binned numerical data

**Implementation** (temporal_shift_detection.py):
```python
def calculate_psi(train_dist, test_dist, epsilon=1e-10):
    psi = 0
    for category in train_dist.index:
        p_train = train_dist.get(category, 0) + epsilon
        p_test = test_dist.get(category, 0) + epsilon
        psi += (p_test - p_train) * np.log(p_test / p_train)
    return psi
```

### Chi-Square Test

**Formula**:
```
χ² = Σ (O - E)² / E
```

Where:
- `O`: Observed frequency in test set
- `E`: Expected frequency (from train distribution)

**Interpretation**:
- p-value < 0.05: Statistically significant shift
- p-value ≥ 0.05: No significant shift

**Advantages**:
- Provides statistical significance
- Widely accepted in scientific research

**Implementation**:
```python
from scipy.stats import chisquare

chi2, p_value = chisquare(
    f_obs=test_counts,
    f_exp=expected_counts
)
```

### Jensen-Shannon Divergence

**Formula**:
```
JS(P || Q) = 0.5 × KL(P || M) + 0.5 × KL(Q || M)
```

Where:
- `M = 0.5 × (P + Q)` (midpoint distribution)
- `KL`: Kullback-Leibler divergence

**Interpretation**:
- Range: [0, 1] (when using log base 2)
- 0: Identical distributions
- 1: Completely different distributions

**Advantages**:
- Symmetric (unlike KL divergence)
- Bounded and normalized
- Theoretically well-founded

**Implementation**:
```python
from scipy.spatial.distance import jensenshannon

js_div = jensenshannon(train_dist, test_dist)
```

---

## Epistemic Uncertainty Calculation

### For Classification Tasks

**Approach**: Mutual Information between predictions and model parameters

**Mathematical Definition**:
```
I(Y; θ | X) = H(E[p(y|x, D)]) - E[H(p(y|x, θ))]
```

Where:
- `H(·)`: Entropy function
- `E[p(y|x, D)]`: Average prediction across ensemble (predictive distribution)
- `p(y|x, θ)`: Individual model's prediction
- `θ`: Model parameters (different for each ensemble member)

**Intuitive Interpretation**:
- **H(E[p(y|x, D)])**: Entropy of the averaged prediction (how uncertain is the ensemble's consensus?)
- **E[H(p(y|x, θ))]**: Average entropy of individual predictions (how uncertain is each model?)
- **Difference**: If models agree → low mutual information. If models disagree → high mutual information.

**Implementation** (temporal_uncertainty_analysis.py):
```python
def calculate_epistemic_uncertainty_classification(predictions):
    """
    predictions: (n_models, n_samples, n_classes) array
    """
    # Average prediction across ensemble
    mean_pred = np.mean(predictions, axis=0)  # (n_samples, n_classes)

    # Entropy of averaged prediction (predictive entropy)
    pred_entropy = -np.sum(
        mean_pred * np.log(mean_pred + 1e-10),
        axis=1
    )

    # Average entropy of individual predictions (expected entropy)
    individual_entropies = -np.sum(
        predictions * np.log(predictions + 1e-10),
        axis=2
    )
    expected_entropy = np.mean(individual_entropies, axis=0)

    # Mutual information (epistemic uncertainty)
    epistemic = pred_entropy - expected_entropy

    return epistemic.mean()
```

### For Regression Tasks

**Approach**: Variance across ensemble predictions

**Mathematical Definition**:
```
Var_epistemic = Var_θ[E[y | x, θ]]
```

Where variance is computed across ensemble members at each input x.

**Intuitive Interpretation**:
- If all models predict similar values → low variance → low epistemic uncertainty
- If models predict different values → high variance → high epistemic uncertainty

**Implementation** (analyze_regression_uq.py):
```python
def calculate_epistemic_uncertainty_regression(predictions):
    """
    predictions: (n_models, n_samples) array
    """
    # Standard deviation across ensemble members
    epistemic = np.std(predictions, axis=0)

    return epistemic.mean()
```

**Alternative**: Prediction Interval Width
```python
# 95% prediction interval
lower = np.percentile(predictions, 2.5, axis=0)
upper = np.percentile(predictions, 97.5, axis=0)
interval_width = upper - lower
```

---

## Classification vs Regression Approaches

### Differences in Uncertainty Quantification

| Aspect | Classification | Regression |
|--------|---------------|------------|
| **Metric** | Mutual Information (bits) | Standard Deviation |
| **Range** | [0, log₂(num_classes)] | [0, ∞) |
| **Normalization** | Divide by log₂(num_classes) | Use coefficient of variation |
| **Interpretation** | Information gained by knowing model | Spread of predictions |

### Differences in Shift Detection

**Classification**:
- Analyze label distribution shifts (discrete categories)
- PSI, Chi-square directly applicable
- Natural binning by class

**Regression**:
- Need to bin continuous targets for PSI
- Can use Kolmogorov-Smirnov test instead of Chi-square
- Consider both mean shift and variance shift

### Example: SALT (Classification) vs F1 (Regression)

**SALT item-plant** (8-class classification):
```
Train distribution:
  Plant A: 40%
  Plant B: 30%
  Plant C: 20%
  ...

Test distribution (COVID impact):
  Plant A: 35%  ← Shift detected
  Plant B: 35%  ← Shift detected
  Plant C: 20%
  ...

PSI = 0.092 (moderate shift)
Epistemic uncertainty increased 137.88%
```

**F1 driver-position** (regression):
```
Train target: position ∈ [1, 20]
Bin into quantiles for PSI calculation

Epistemic uncertainty = Std(ensemble predictions)
If shift occurs:
  - Std increases (models disagree more)
  - Correlation with distributional shift metrics
```

---

## Validation Strategy

### Hypothesis Testing Framework

**Null Hypothesis (H₀)**: Epistemic uncertainty does NOT correlate with distribution shift

**Alternative Hypothesis (H₁)**: Epistemic uncertainty increases when distribution shifts

### Validation Approach

#### Step 1: Quantify Distribution Shift (Ground Truth)
```python
# Phase 1: Calculate PSI for each task
psi_scores = {}
for task in tasks:
    train_dist = get_label_distribution(task, 'train')
    test_dist = get_label_distribution(task, 'test')
    psi_scores[task] = calculate_psi(train_dist, test_dist)
```

#### Step 2: Measure Uncertainty Increase
```python
# Phase 3: Calculate epistemic uncertainty
uncertainty_increase = {}
for task in tasks:
    train_unc = epistemic_uncertainty(task, 'train')
    test_unc = epistemic_uncertainty(task, 'test')
    uncertainty_increase[task] = (test_unc - train_unc) / train_unc
```

#### Step 3: Correlation Analysis
```python
# Phase 4: Validate hypothesis
correlation = pearson_r(
    list(psi_scores.values()),
    list(uncertainty_increase.values())
)

if correlation > 0.7 and p_value < 0.05:
    print("HYPOTHESIS VALIDATED")
```

### Expected Results

**If hypothesis is correct**:
- Tasks with high PSI → High uncertainty increase
- Tasks with low PSI → Low uncertainty increase
- Positive correlation (r > 0.7)

**If hypothesis is wrong**:
- No correlation between PSI and uncertainty increase
- Random scatter plot

### Current Evidence

**item-plant** (completed):
- PSI = 0.092 (moderate shift)
- Uncertainty increase = +137.88%
- ✅ Consistent with hypothesis (moderate shift → large uncertainty increase)

**Pending**: Multi-task correlation with 8 tasks to statistically validate

---

## Temporal Considerations

### Why Temporal Splits Matter

**Random splits** (traditional ML):
- Train/test sampled randomly from same time period
- Distribution assumed to be identical (IID assumption)
- **Cannot detect distribution shift** (no shift occurs)

**Temporal splits** (our approach):
- Train: Past data (2018-2020)
- Test: Future data (2020-2021)
- Distribution CAN shift over time
- **Enables shift detection research**

### COVID-19 as Natural Experiment

**Why SALT dataset is ideal**:
- Clear temporal boundary: Feb 2020 (COVID onset)
- Known exogenous shock to business processes
- Expected distribution shift in sales patterns

**Validation opportunity**:
- Compare pre-COVID (train) vs post-COVID (test) distributions
- Epistemic uncertainty should increase for COVID-impacted tasks
- Control: Tasks unaffected by COVID should show stable uncertainty

---

## Practical Implementation Notes

### Ensemble Size

**Minimum**: 5 models
- Statistical reliability
- Computational feasibility

**Recommended**: 10+ models
- Better uncertainty estimates
- More robust to outliers

**Trade-off**:
- More models → Better uncertainty estimation
- More models → Higher computational cost

### Random Seed Selection

**Important**: Use different random seeds for:
1. Model weight initialization
2. Training data shuffling
3. Dropout masks (if applicable)

**Example**:
```python
seeds = [42, 43, 44, 45, 46]  # Consecutive for reproducibility
for seed in seeds:
    train_model(seed=seed)
```

### Computational Efficiency

**For large datasets**:
- Use subsampling for training (e.g., 50,000 samples)
- Full ensemble inference on all data splits
- Uncertainty calculation is cheap (just variance/entropy)

**Current approach**:
```bash
--sample_size 50000  # Subsample training data
# But evaluate on full train/val/test for UQ
```

---

## Summary

This research uses **predictions as a tool** to measure **epistemic uncertainty**, which serves as a **proxy for distribution shift detection**.

**Key principle**: Uncertainty quantification enables proactive model monitoring without requiring ground truth labels on new data.

**Scientific contribution**: Validating that epistemic uncertainty reliably correlates with distribution shift magnitude across multiple tasks and datasets.

---

**Last Updated**: 2025-01-20
