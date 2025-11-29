# FK-Level Uncertainty Attribution: Experiment Results

## Overview

This document summarizes experiments on **SAP SALT** (ERP sales data) demonstrating:
1. FK-level uncertainty attribution
2. Entity-level optimization recommendations
3. Simulation vs optimization framework
4. Verification of recommendations

---

## Dataset: SAP SALT (rel-salt)

| Table | Rows | Description |
|-------|------|-------------|
| salesdocumentitem | 1.9M | Main entity - line items |
| salesdocument | 411K | Sales headers |
| customer | 140K | Customer master data |
| address | 1.8M | Address information |

**FK Structure:**
```
salesdocumentitem → SALESDOCUMENT (header)
salesdocumentitem → SOLDTOPARTY (ordering customer)
salesdocumentitem → SHIPTOPARTY (receiving customer)
customer → ADDRESSID
```

**Target:** PLANT (which plant fulfills the order) - regression task

---

## Experiment 1: FK-Level Attribution

**File:** `experiment_salt_entity_optimization.py`

### Results (n=10,000 samples)

| FK Group | Attribution | Features |
|----------|-------------|----------|
| ITEM | 37.3% | SHIPPINGPOINT, ITEMINCOTERMSCLASSIFICATION |
| SALESDOCUMENT | 24.3% | SALESOFFICE, CUSTOMERPAYMENTTERMS, SHIPPINGCONDITION |
| SALESGROUP | 20.7% | SALESGROUP |
| SHIPTOPARTY | 9.2% | SHIPTOPARTY, SHIPTO_ADDRESSID |
| SOLDTOPARTY | 8.6% | SOLDTOPARTY, SOLDTO_ADDRESSID |

**Key Finding:** ITEM FK (shipping point) is the biggest lever for uncertainty reduction.

---

## Experiment 2: Entity-Level Drill-Down

Within each FK group, specific entities vary widely:

### SOLDTOPARTY (Customer) Analysis

| Entity | Uncertainty | Interpretation |
|--------|-------------|----------------|
| Customer 29806 | 0.000064 | Very predictable |
| Customer 100300 | 0.000077 | Very predictable |
| Customer 86919 | 0.288514 | **4,500x higher** - hard to predict |
| Customer 74524 | 0.242159 | High uncertainty |

**Business Insight:** Some customers are 4,500x more predictable than others. Orders from Customer 29806 can be fulfilled with high confidence; Customer 86919 requires safety margins.

### SHIPPINGPOINT Analysis

| Entity | Uncertainty | Sample Size |
|--------|-------------|-------------|
| Shipping Point 2 | 0.002659 | 201 orders |
| Shipping Point 6 | 0.005672 | 5,145 orders |
| Shipping Point 40 | 0.171389 | 7 orders |

---

## Experiment 3: Simulation vs Optimization

**File:** `experiment_salt_simulation_vs_optimization.py`

### SIMULATION Questions (What IF?)

| Question | Scenario | Result |
|----------|----------|--------|
| Q1: Entity Switch | Switch from Customer 61397 to 25950 | -0.60% uncertainty |
| Q2: Remove Entity | Remove Customer 61397 | -0.58% uncertainty |
| Q3: Data Quality | ITEM data degrades 10% | **+246% uncertainty** |
| Q4: Rebalance | Shift to low-unc customers only | **-99.58% uncertainty** |

### OPTIMIZATION Questions (What SHOULD?)

| Question | Result |
|----------|--------|
| Q1: Best Entity | Customer 25950 (unc=0.000029) |
| Q2: Optimal Allocation | 45.6% to Entity 92652, 15.3% to 89832, 13.3% to 31001, ... |
| Q3: Constrained | If must include high-unc entity, minimize others |
| Q4: Threshold | 47.7% of entities meet uncertainty threshold |

### Key Distinction

```
SIMULATION: User provides scenario → System evaluates uncertainty
OPTIMIZATION: User provides objective → System finds best scenario
```

---

## Experiment 4: Verification

**File:** `experiment_salt_verification.py`

All 4 verification tests **PASSED**:

| Test | Metric | Result | Interpretation |
|------|--------|--------|----------------|
| Held-Out Consistency | Train-test correlation | 0.707 | Rankings stable across splits |
| Counterfactual Simulation | Uncertainty reduction | 0.6% | Simulation confirms reduction |
| Bootstrap Stability | Rank stability score | 0.992 | Rankings robust to resampling |
| Uncertainty Calibration | Unc-error correlation | 0.541 | Low uncertainty = low error |

### Calibration Detail

| Uncertainty Level | MAE |
|-------------------|-----|
| Very Low | 0.019 |
| Low | 0.044 |
| Medium | 0.067 |
| High | 0.120 |
| Very High | 0.453 |

**Conclusion:** Uncertainty is well-calibrated. Low-uncertainty predictions have lower errors.

---

## Experiment 5: Temporal + Entity Analysis

**File:** `experiment_salt_temporal_entity.py`

### Entity Stability Over Time

**STABLE entities** (low uncertainty variance):
- Entity 341: CV=0.00, mean_unc=0.0023
- Entity 1346: CV=0.00, mean_unc=0.0047

**VOLATILE entities** (high uncertainty variance):
- Entity 116989: CV=11.00, mean_unc=0.0034
- Entity 80279: CV=9.28, mean_unc=0.0163

**Recommendation:** Prefer stable entities for critical orders; use volatile entities with larger safety margins.

---

## Actionable Recommendations Framework

### Level 1: FK Attribution
"Which type of relationship drives uncertainty?"
```
Answer: ITEM (shipping point) → 37.3%
        SALESDOCUMENT (header) → 24.3%
```

### Level 2: Entity Identification
"Which specific entities are problematic?"
```
Answer: Customer 86919 (unc=0.29)
        vs Customer 29806 (unc=0.00006)
```

### Level 3: Optimization
"How should I allocate orders?"
```
Answer: 45.6% to Entity 92652
        15.3% to Entity 89832
        13.3% to Entity 31001
        ...
```

### Level 4: Simulation
"What if I change to this scenario?"
```
Answer: Shifting to low-unc entities → 99.58% uncertainty reduction
```

---

## Files Created

| File | Purpose |
|------|---------|
| `data_loader_salt.py` | Load SAP SALT with FK structure |
| `experiment_salt_entity_optimization.py` | FK attribution + entity recommendations |
| `experiment_salt_temporal_entity.py` | Temporal stability analysis |
| `experiment_salt_verification.py` | 4 verification tests |
| `experiment_salt_simulation_vs_optimization.py` | 8 question types (4 sim + 4 opt) |

---

## Comparison: Feature-Level vs FK-Level Attribution

### Without FK Structure (Feature-Level)
```
"SHIPPINGPOINT contributes 15% to uncertainty"
→ Which shipping point? All of them? Some?
→ Not actionable
```

### With FK Structure (RelUQ)
```
Level 1: ITEM FK contributes 37.3%
Level 2: Shipping Point 40 has high uncertainty (0.17)
Level 3: Recommendation: Route through Shipping Point 2 instead
→ Actionable decision
```

---

## Business Value

1. **Risk Quantification**: Know which customers/shipping points have unpredictable outcomes
2. **Portfolio Optimization**: Allocate orders to minimize overall uncertainty
3. **Early Warning**: Monitor entity uncertainty trends over time
4. **Constraint Handling**: Optimize even when forced to use certain entities

---

---

## Scalability Analysis

### Computational Complexity

| Component | Complexity | SALT (1.9M rows) | Notes |
|-----------|------------|------------------|-------|
| Ensemble Training | O(n × k models) | ~10 sec/model | Parallelizable |
| FK Attribution | O(P × F × n) | ~30 permutations × 5 FKs | 150 forward passes |
| Entity Analysis | O(E × inference) | ~4000 entities | One pass per entity |
| Optimization | O(E log E) | Sorting 4000 entities | Very fast |

Where: n=samples, k=ensemble size, P=permutations, F=FK groups, E=unique entities

### Data Scale Tested

| Experiment | Sample Size | Entities | Runtime |
|------------|-------------|----------|---------|
| Entity Optimization | 10,000 | 3,986 customers | ~45 sec |
| Temporal Analysis | 50,000 | 4,000+ customers | ~2 min |
| Verification Suite | 20,000 | 3,986 customers | ~3 min |

### Scalability Strengths

1. **FK Structure is Fixed**: Number of FK groups is small (5-10 typically), regardless of data size
2. **Entity Aggregation**: Compute once, aggregate by entity - scales O(E) not O(n)
3. **Sampling Works**: 10-50K samples sufficient for stable rankings (verified by bootstrap)
4. **Parallelizable**: Ensemble models train independently; FK permutations independent

### Scalability Limitations

1. **Rare Entities**: Entities with <5 samples have unstable uncertainty estimates
2. **High Cardinality**: Millions of unique entities would require hierarchical grouping
3. **Real-time**: Permutation-based attribution not suitable for real-time (use cached results)
4. **Memory**: Full ensemble predictions on 1M+ rows requires batching

### Production Recommendations

```
For 1M+ transactions:
1. Train ensemble once (offline, ~1 hour)
2. Cache entity uncertainty rankings (update daily)
3. Real-time lookup: O(1) per entity
4. Optimization: Pre-compute optimal allocations
```

### Business Decision Scalability

| Decision Type | Entities | Feasibility |
|---------------|----------|-------------|
| Supplier selection | 100-1000 | Fully tractable |
| Customer prioritization | 10K-100K | Tractable with sampling |
| Product-level | 100K+ | Requires hierarchical approach |
| SKU-level | 1M+ | Aggregate to category first |

### Verified on SAP SALT

- **1.9M transactions**: Sampled 50K, results stable
- **140K customers**: Analyzed 4,000 with sufficient volume
- **Rankings stable**: Bootstrap stability = 0.992
- **Calibrated**: Low uncertainty → low error (confirmed)

---

## Next Steps

1. Apply to other datasets (Amazon, Stack Overflow)
2. Add temporal forecasting (predict future uncertainty spikes)
3. Build interactive dashboard for entity-level exploration
4. Integrate with SAP S/4HANA process traces for richer context
5. Test on 1M+ samples with distributed computing
