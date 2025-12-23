# Validation Results Summary

**Date**: 2025-12-23
**Experiments Run**: Causal Validation Suite (corruption, learning curve, EP detection)
**Domains Tested**: 6 domain/task pairs, 31 FK data points

---

## Executive Summary

| Experiment | Result | Implication |
|------------|--------|-------------|
| **Corruption** | ❌ FAILED (ρ = -0.04) | Uncertainty does NOT predict data quality impact |
| **Learning Curve** | ✅ 4/6 PASSED | Uncertainty IS epistemic (reducible) |
| **EP Detection** | ⚠️ 2/6 EP domains | Most domains are non-EP |

**Critical Finding**: The core claim "FK uncertainty identifies data investment targets" is NOT validated by the causal experiment.

---

## Detailed Results

### 1. Corruption Experiment (Causal Validation)

**Goal**: Show that high-uncertainty FKs are more sensitive to data corruption

**Aggregate Results** (n=31 FK data points):
```
Uncertainty vs Corruption: ρ = -0.04 (p = 0.82) ❌
Error vs Corruption:       ρ = +0.60 (p = 0.0003) ✅
```

**Per-Domain Results**:

| Domain | Task | ρ (unc vs corr) | Verdict |
|--------|------|-----------------|---------|
| rel-f1 | driver-position | +0.50 | FAIL |
| rel-f1 | driver-dnf | -1.00 | FAIL |
| rel-f1 | driver-top3 | -0.87 | FAIL |
| rel-trial | study-outcome | +0.58 | MARGINAL |
| rel-avito | ad-ctr | -0.37 | FAIL |
| rel-event | user-attendance | +0.77 | PASS |

**Key Insight**: Only 1/6 domains show significant correlation between uncertainty and corruption sensitivity. The aggregate correlation is essentially zero.

**However**: Error attribution (permutation importance) DOES predict corruption (ρ = 0.60, p < 0.001). This suggests FK importance is valid, but FK uncertainty is not.

---

### 2. Learning Curve Experiment (Epistemic Validation)

**Goal**: Show that high-uncertainty FKs have uncertainty that decreases with more data

**Results**:

| Domain | Task | Verdict |
|--------|------|---------|
| rel-f1 | driver-position | ✅ PASS |
| rel-f1 | driver-dnf | ❌ FAIL |
| rel-f1 | driver-top3 | ✅ PASS |
| rel-trial | study-outcome | ✅ PASS |
| rel-avito | ad-ctr | ✅ PASS |
| rel-event | user-attendance | ❌ FAIL |

**Overall**: 4/6 passed - uncertainty IS generally epistemic (reducible with more data)

---

### 3. EP Domain Detection

**Goal**: Determine if domains satisfy Error Propagation property

**Results**:

| Domain | Task | ρ (unc vs error) | Type |
|--------|------|------------------|------|
| rel-f1 | driver-position | +0.50 | MIXED |
| rel-f1 | driver-dnf | +0.50 | MIXED |
| rel-f1 | driver-top3 | +1.00 | EP ✅ |
| rel-trial | study-outcome | +0.78 | EP ✅ |
| rel-avito | ad-ctr | -0.68 | NON-EP |
| rel-event | user-attendance | -0.54 | NON-EP |

**Overall**: Only 2/6 domains are EP domains

---

## What This Means for the Paper

### ❌ Claims We CANNOT Make

1. **"FK uncertainty predicts data quality impact"** - FAILED (ρ = -0.04)
2. **"High-uncertainty FKs should be prioritized for data investment"** - NOT VALIDATED
3. **"EP property holds generally"** - Only 2/6 domains

### ✅ Claims We CAN Make

1. **"FK-level feature importance predicts corruption sensitivity"** (ρ = 0.60, p < 0.001)
2. **"Schema-guided grouping provides stable attributions"** (std < 3% across seeds)
3. **"Uncertainty is epistemic in most cases"** (4/6 learning curve experiments pass)
4. **"EP detection can identify when FK attribution is valid"** (diagnostic tool)

---

## Revised Paper Strategy

### Option A: Pivot to "FK Importance" (Recommended)

**New Core Claim**: "Schema-guided FK grouping provides stable, actionable feature importance"

**Drop**: Uncertainty attribution as main contribution
**Keep**: FK grouping, stability analysis, intervention experiments

**Probability**: This is a smaller contribution but more defensible (KDD 70%, NeurIPS 50%)

### Option B: Focus on EP Detection as Contribution

**New Core Claim**: "We provide an EP detection criterion that identifies when FK attribution is valid"

**Narrative**:
- FK attribution works in EP domains (ρ = 0.78-1.00)
- Fails in non-EP domains (ρ < 0)
- We provide a diagnostic tool (compute EP criterion first)

**Probability**: Novel diagnostic tool, honest about limitations (NeurIPS 55-60%)

### Option C: Acknowledge Negative Result (Risky)

**Narrative**: "We investigated FK uncertainty attribution and found it does NOT predict data quality impact, in contrast to FK importance which does"

**Risk**: Negative results are harder to publish
**Benefit**: Honest, could generate discussion

---

## Recommended Next Steps

1. **Update paper outline** to focus on FK importance (not uncertainty)
2. **Emphasize error attribution** (ρ = 0.60 with corruption)
3. **Keep EP detection** as a diagnostic contribution
4. **Acknowledge limitation** that uncertainty ≠ importance
5. **Target KDD 2026** instead of NeurIPS (more applied venue)

---

## Raw Numbers for Paper

### Aggregate Statistics
```
Total FK data points:        31
Domains tested:              4 (6 tasks)
FK groups per domain:        3-11

Aggregate correlations:
- Uncertainty vs Error:      ρ = 0.09 (n.s.)
- Uncertainty vs Corruption: ρ = -0.04 (n.s.)
- Error vs Corruption:       ρ = 0.60 (p < 0.001) ***
```

### Best Domain (rel-trial/study-outcome)
```
FK groups:                   11
EP correlation (ρ):          0.78 (p < 0.01)
Top uncertainty FK:          OUTCOME_ANALYSES (+14%)
Top corruption FK:           OUTCOME_ANALYSES (+126%)
Match:                       ✓ YES
```

### Worst Domain (rel-f1/driver-dnf)
```
FK groups:                   3
Corruption correlation (ρ):  -1.00
Learning curve:              FAIL
EP correlation:              0.50
```

---

## Key Takeaway

**The data shows FK importance works, but FK uncertainty does not.**

Error attribution (permutation importance) correlates with corruption sensitivity (ρ = 0.60), but uncertainty attribution does not (ρ = -0.04).

This is actually an interesting finding - it suggests that:
1. Ensemble variance is NOT a good proxy for data quality importance
2. Permutation importance IS a good proxy
3. The two measure fundamentally different things

**Recommendation**: Pivot the paper to "FK-level feature importance" and be honest that uncertainty attribution doesn't work as well as we hoped.

---

*Analysis completed: 2025-12-23*
*Recommendation: Pivot to FK importance, target KDD 2026*
