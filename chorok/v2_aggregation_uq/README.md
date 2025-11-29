# V2: Aggregation-Aware UQ (PAUSED)

**Status**: PAUSED (2025-11-29)
**Reason**: Problem exists but contribution is weak

---

## Hypothesis

ML models are miscalibrated for relational data because they don't account for aggregation uncertainty (1/√n).

## Empirical Evidence

| Cardinality | Error Rate | Model Uncertainty |
|-------------|------------|-------------------|
| 1 row | 64.8% | 0.560 |
| 30 rows | 52.5% | 0.566 |

Model makes 12% more errors on sparse data, **but doesn't know it**.

## Statistical Validation

- Cardinality vs Error: ρ = -0.053, **p = 0.039** (significant)
- SE of Mean vs Error: ρ = 0.053, **p = 0.039** (significant)
- Model uncertainty does NOT correlate with cardinality (p = 0.29)

## Why Paused

1. **Problem is real** but contribution is weak
2. "Feature 추가로 해결 안 됨" is known (Guo 2017)
3. Specific to relational data = too niche
4. No novel solution proposed yet

## Files

| File | Description |
|------|-------------|
| `aggregation_uncertainty.py` | Hypothesis testing experiments |

## Potential Future Work

1. Design cardinality-aware calibration method
2. Validate ECE improvement on RelBench
3. Compare with standard calibration methods

---

*Paused: 2025-11-29*
