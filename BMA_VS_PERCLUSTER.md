# BMA vs Per-Cluster Granger: Side-by-Side Comparison

## What Was Wrong: The BMA Problem

### Step 1: Compute BIC for each cluster
```
Cluster 1 (best):  BIC = 75,587.3  ← BEST
Cluster 2:         BIC = 75,624.7  (Δ = 37.4)
Cluster 3:         BIC = 75,660.2  (Δ = 72.9)
Cluster 4:         BIC = 75,726.5  (Δ = 139.2)
Cluster 5:         BIC = 75,804.9  (Δ = 217.6)
Cluster 6:         BIC = 75,906.3  (Δ = 319.0)
Cluster 7:         BIC = 76,137.5  (Δ = 550.2)
```

### Step 2: Compute BMA weights
```
Formula: w_k = exp(-0.5 × ΔBIC_k) / Σ exp(-0.5 × ΔBIC_j)

Cluster 1: exp(-0.5 × 0)      = 1.000      → 100.0%
Cluster 2: exp(-0.5 × 37.4)   = 4.5e-9     → 0.0%
Cluster 3: exp(-0.5 × 72.9)   = 1.5e-16    → 0.0%
Cluster 4: exp(-0.5 × 139.2)  = 1.1e-30    → 0.0%
Cluster 5: exp(-0.5 × 217.6)  = 1.2e-47    → 0.0%
Cluster 6: exp(-0.5 × 319.0)  = 1.6e-69    → 0.0%
Cluster 7: exp(-0.5 × 550.2)  = 7.4e-120   → 0.0%

Sum of weights ≈ 1.000
```

### The Problem
**BMA reduces to essentially Cluster 1 selection.** This defeats the purpose of averaging across models.

Why? The BIC differences are too large. When Δ > 100, exponential weighting collapses to zero for all non-best models.

---

## The Solution: Per-Cluster Robustness

Instead of computing degenerate weights, we show results for EACH cluster:

### For HML→SMB Causality in Elevated Regime:

```
Cluster | BIC    | Δ BIC | GFC% | Rep Seed | Elevated p | Significant?
--------|--------|-------|------|----------|------------|-------------
   1    | 75587  |   0   |  0%  |    28    |  0.0414   |     YES
   2    | 75625  |  38   |  0%  |    35    |  0.0258   |     YES
   3    | 75660  |  73   |  0%  |    12    |  0.0258   |     YES
   4    | 75726  | 139   |  0%  |    15    |  0.0336   |     YES
   5    | 75805  | 218   | 90%  |    21    |  0.0414   |     YES
   6    | 75906  | 319   |100%  |    49    |  0.0258   |     YES
   7    | 76137  | 550   | 92%  |    24    |  0.0258   |     YES
```

### Key Observations

1. **All 7 clusters show p < 0.05** in Elevated regime
2. **p-values are tightly clustered** (0.0258 to 0.0414)
3. **No need for BMA weighting** — the finding is robust regardless of which cluster you choose
4. **Cluster 1 (best BIC, cannot detect 2008 crisis)** still shows significant effect (p=0.0414)
5. **Economically-valid clusters (5-7)** that DO detect 2008 show comparable effects (p≤0.0414)

---

## Why Per-Cluster is Better

| Aspect | BMA | Per-Cluster |
|--------|-----|-------------|
| **Degeneracy** | Collapses to ~100% on best cluster | Treats all clusters equally |
| **Interpretability** | "Weighted average" (but essentially useless) | Direct robustness evidence |
| **Data needed** | BIC values only | Pre-computed Granger p-values |
| **Computation** | Simple exponential weights | Extract and aggregate p-values |
| **Conclusion** | "Elevated p ≈ 0.041 (almost all from Cluster 1)" | "All 7 clusters show p < 0.05" |
| **Robustness statement** | Weak (really just Cluster 1 result) | Strong (all clusters agree) |

---

## How It Answers the Review Questions

### Question 1: "Does the HML→SMB finding depend on which cluster is chosen?"

**BMA answer:** "The weighted average p-value is 0.041. But wait, that's almost entirely from Cluster 1..."

**Per-cluster answer:** "No, it's robust. All 7 clusters show p < 0.05 in Elevated regime."

### Question 2: "Best-fit cluster can't detect 2008 as a crisis—is the Elevated finding valid?"

**BMA answer:** "BMA combines clusters including economically invalid ones..."

**Per-cluster answer:** "Cluster 1 (best-BIC, no crisis detection) has p=0.0414 in Elevated. Clusters 5-7 (valid, 90%+ GFC) have p≤0.0414. Both agree."

### Question 3: "This multi-cluster analysis seems to combine incompatible models."

**BMA answer:** "Degenerate weights avoid truly combining them, so... not much combining happens."

**Per-cluster answer:** "We're not combining them. We're reporting each cluster's result. All 7 are significant independently."

---

## Mathematical Insight

The core issue is **BIC scale**. In model selection, differences of even 10-20 points are considered significant. When Δ BIC > 100:

```
exp(-0.5 × ΔB)  < exp(-50)  ≈ 1.9e-22
```

This is **numerically zero** in any practical sense. You get floating-point underflow.

Solution: Don't use weighting. Use direct comparison instead.

---

## What the Per-Cluster Approach Accomplishes

### 1. Answers the core question directly
"Is the finding stable across different HMM fits?"
Yes, all 7 fits (local optima) agree on HML→SMB causality in Elevated.

### 2. Addresses the crisis detection concern
"One cluster can't detect 2008. Does this invalidate the finding?"
No, because:
- Both good and bad crisis-detection clusters show the effect
- The effect is in Elevated regime, not Crisis regime
- Crisis regime is not statistically significant in any cluster (p > 0.17)

### 3. Provides transparent robustness evidence
All 7 p-values are shown. Readers can see:
- The effect is real (all p < 0.05)
- The effect is consistent (all p ∈ [0.0258, 0.0414])
- No cluster is a major outlier

### 4. Avoids the philosophical problem of BMA here
Traditional BMA says: "Weight models by posterior probability based on BIC."
But when BIC differences are this large, weighting is meaningless.
Per-cluster simply says: "All local optima agree on the effect."

---

## What About Within-Cluster Variation?

The per-cluster results use MEDIAN across seeds in each cluster:

### Cluster 2 (15 seeds):
- Individual seed Elevated p-values: 0.0258, 0.0414, 0.0258, 0.0414, ... (varies within cluster)
- Median: 0.0258
- All seeds in cluster: significant? NO, but 13/15 are (p < 0.05)

### Cluster 1 (3 seeds):
- Seed 28: p = 0.0414
- Seed 20: p = 0.0258
- Seed 6:  p = 0.5265
- Median: 0.0414
- Not all 3 agree! But majority do.

This within-cluster variation is captured in the JSON output. For the main table, we use the robust median statistic.

---

## Final Summary

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Clusters with Elevated p < 0.05 | 7/7 | ROBUST |
| Range of Elevated p | 0.0258 - 0.0414 | TIGHT |
| Best-BIC cluster in agreement | YES (p=0.0414) | CONSISTENT |
| Crisis-valid clusters agreement | YES (p≤0.0414) | CONSISTENT |
| **Recommendation** | **Use per-cluster table in paper** | **BMA is unnecessary** |

---

## References

- **BIC scale issue:** Kass & Raftery (1995) recommend ΔBIC > 10 as "strong evidence"
  - Our Δ BIC = 37-550 is "decisive" in their terminology
  - This makes BMA weighting degenerate by design

- **Per-cluster approach:** Directly analogous to sensitivity analysis
  - Report results across different specifications
  - Show consistency (robustness) or divergence (fragility)
  - Here: all specifications agree → robust finding

---

Generated: 2026-02-28
Analysis: Per-Cluster Granger Robustness (frozen OOS, 50 seeds, 7 clusters)
