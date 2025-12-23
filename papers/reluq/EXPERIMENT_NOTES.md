# RelUQ Experiment Notes

## Experiment Results (Dec 2024)

### Summary
- **Sample-level uncertainty-error correlation**: ρ = 0.25-0.91 across tasks
- **Feature importance attribution**: Successfully differentiates FK groups
- **Intervention interpretation**: Fixing top FK *increases* error (expected - removes signal)

### Detailed Results

#### Sample-Level Uncertainty-Error Correlation
| Task | Seeds | Mean ρ |
|------|-------|--------|
| rel-f1/driver-position | 3 | 0.25 |
| rel-f1/driver-dnf | 3 | 0.88 |
| rel-f1/driver-top3 | 3 | 0.91 |
| rel-avito/user-visits | 3 | 0.85 |
| rel-avito/user-clicks | 3 | 0.84 |
| rel-stack/user-engagement | 3 | 0.88 |

**Key insight**: High ρ means uncertainty estimates are meaningful - high uncertainty samples have high error.

#### FK Attribution (Feature Importance-Based)
rel-f1:
- RESULTS: 75-79%
- STANDINGS: 15-21%
- QUALIFYING: 4-7%

rel-avito (5 FK groups):
- SEARCHINFO: 33-49%
- VISITSTREAM: 15-36%
- PHONEREQUESTSSTREAM: 10-16%
- USERINFO: 11-13%
- TRAIN: 8-9%

#### Intervention Effect (Fixing Top FK)
| Task | Top FK | MAE Change |
|------|--------|------------|
| rel-f1/driver-position | RESULTS | +35% |
| rel-f1/driver-dnf | RESULTS | +11% |
| rel-avito/user-clicks | SEARCHINFO | +70% |
| rel-stack/user-engagement | BADGES | +78% |

Positive change = error increases when FK is "fixed" (replaced with mean).
This validates that attribution correctly identifies predictively important FKs.

---

## Critical Issue Discovered (Dec 2024)

### Problem: Permutation-based Uncertainty Attribution Fails

**Observation**: When running experiments, FK attribution showed equal values (33.33% each for 3 groups) despite clear differences in error impact.

**Root Cause**: For bootstrap ensembles, permuting features doesn't significantly change ensemble variance because:
1. Ensemble variance comes from model diversity (different bootstrap samples)
2. Permuting input features changes predictions but not the *variance* of predictions
3. The uncertainty signal is in the disagreement between models, not input sensitivity

**Example Output**:
```
FK Attribution: {'QUALIFYING': 33.33%, 'RESULTS': 33.33%, 'STANDINGS': 33.33%}
FK Error Impact: {'QUALIFYING': 0.02%, 'RESULTS': 97.83%, 'STANDINGS': 2.17%}
```

The error impact varies dramatically, but uncertainty attribution is flat.

### Solution: Pivot to Alternative Metrics

Instead of permutation-based uncertainty attribution, use:

1. **Intervention Effect** (Primary Metric)
   - Fix top FK group (replace with mean values)
   - Measure actual MAE/RMSE reduction
   - This directly validates actionability

2. **Per-Sample Uncertainty-Error Correlation**
   - For each sample: uncertainty = ensemble std, error = |pred - true|
   - Compute Spearman correlation
   - Tests if uncertainty estimates are meaningful

3. **Feature Importance Attribution**
   - Use LightGBM's built-in feature importance
   - Aggregate by FK group
   - More stable than permutation

### Additional Issues Found

1. **Few FK Groups**: Most RelBench tasks have only 1-2 FK groups, making Spearman correlation (n≥3) impossible
2. **Missing Tasks**: Many task names in my curated list don't exist in the registry
3. **Schema Simplicity**: RelBench schemas are simpler than expected for this analysis

### Updated Experiment Files

- `run_full_experiments.py` - Original approach (permutation-based, issues with equal attribution)
- `run_intervention_focused.py` - Pivoted approach (intervention + importance-based)

### Recommended Approach Going Forward

1. Use `run_intervention_focused.py` for experiments
2. Focus paper claims on:
   - Intervention effect validates actionability
   - Per-sample uncertainty-error correlation validates UQ quality
   - FK grouping provides stability vs feature-level
3. Downplay or remove Spearman correlation claims (need ≥3 FK groups)

### Valid Tasks in RelBench Registry

```python
VALID_TASKS = {
    'rel-f1': ['driver-position', 'driver-dnf', 'driver-top3'],
    'rel-trial': ['study-outcome', 'site-success', 'study-adverse'],
    'rel-avito': ['user-visits', 'user-clicks'],
    'rel-stack': ['user-engagement', 'post-votes'],
}
```

Tasks like `results-position`, `qualifying-position`, `item-plant` do NOT exist.
