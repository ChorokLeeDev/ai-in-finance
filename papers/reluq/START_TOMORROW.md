# Start Tomorrow: Active Learning Implementation

**Date:** 2025-12-24
**Goal:** Begin implementing REAL FK-guided active learning
**Timeline:** 4 weeks to validation

---

## ✅ What We Validated Today

1. **SHAP Baseline:** ✅ Works (ρ = 1.000)
2. **Decomposition:** ✅ Works (36% reduction)
3. **Active Learning:** ⚠️ Only simulated - NEEDS REAL IMPLEMENTATION
4. **Causal:** ❌ Doesn't work - DROP

**Score: 2/4 validated, 1/4 needs work**

---

## 🎯 Tomorrow's Goal

**Implement FK-level uncertainty computation (the real version)**

### Current (Fake):
```python
# This is what test_2 did - WRONG!
fk_acquire = random_acquire  # Just random!
simulated_improvement = np.random.uniform(15, 30)  # Fake number!
```

### Tomorrow (Real):
```python
def compute_fk_uncertainty(ensemble, X, fk_groups):
    """
    Compute ACTUAL uncertainty contribution from each FK.
    """
    base_unc = ensemble_variance(ensemble, X)

    fk_uncertainties = {}
    for fk_name, col_indices in fk_groups.items():
        # Permute this FK
        X_perm = X.copy()
        X_perm[:, col_indices] = np.random.permutation(X_perm[:, col_indices])

        # Measure uncertainty increase
        perm_unc = ensemble_variance(ensemble, X_perm)

        # FK uncertainty = increase
        fk_uncertainties[fk_name] = perm_unc - base_unc

    return fk_uncertainties
```

---

## 📋 Week 1 Schedule

### Day 1 (Tomorrow):
- ✅ Read existing code: `examples/run_regression_ensemble.py`
- ✅ Implement `compute_fk_uncertainty()` function
- ✅ Test on rel-f1 (verify not all equal!)
- ✅ Compare with simulated version

**Success:** Get different uncertainty values for different FKs (not all equal)

### Day 2:
- Implement FK-targeted sample selection
- Within top FK, select highest-uncertainty samples
- Test: Does it select different samples than random?

**Success:** Selected samples differ from random selection

### Day 3:
- Implement full active learning loop
- Random acquisition baseline
- FK-guided acquisition
- Compare MAE over iterations

**Success:** Learning curves differ (maybe better, maybe not - that's OK for now)

### Day 4-5:
- Generate proper learning curves
- Plot: MAE vs samples acquired
- Measure: Samples needed to reach 90% of final accuracy

**Success:** Can measure actual improvement (might be 5%, 15%, or 25% - we'll see!)

### Day 6-7:
- Test on multiple datasets (rel-salt, rel-trial)
- Verify: Does FK-guided consistently beat random?
- Document results

**Success:** Know whether active learning actually works

---

## 🚦 Week 1 Checkpoints

### Mid-week (Day 3):
**Question:** Does FK-guided select different samples than random?
- If YES → continue
- If NO → debug, something is wrong

### End-week (Day 7):
**Question:** Does FK-guided beat random by >10%?
- If YES → continue to Week 2
- If MARGINAL (5-10%) → decide whether to invest more
- If NO (<5%) → drop active learning, proceed with 2 directions

---

## 🎓 Learning from Today's Tests

### What Worked:
- Quick validation approach (run all 4 tests in 20 min)
- Honest assessment (caught that Test 2 was fake)
- Rapid iteration (fixed decomposition same day)

### What Didn't:
- Over-optimism (claiming 23% gain when it was simulated)
- Assuming tests were real without checking code
- Not reading the actual implementation

### Lesson:
**Always read the actual code.** Don't trust summaries (even mine).

---

## 💡 Implementation Tips

### 1. Start Simple
Don't try to implement the perfect version. Get something working first.

**Version 1 (Tomorrow):**
- Permutation-based FK uncertainty
- Top FK selection
- Random samples within top FK

**Version 2 (Later):**
- Uncertainty-weighted sampling within FK
- Multi-FK acquisition
- Adaptive budgets

### 2. Test on Small Data First
Use rel-f1 (3K samples, 3 FK groups). Fast iteration.

Once working, scale to rel-salt (ERP, 5 FK groups).

### 3. Expect Negative Results
Active learning might not work. That's research.

**If it shows 5% improvement:** OK, mention in discussion
**If it shows 0% improvement:** That's valuable negative result, report honestly
**If it shows 25% improvement:** Jackpot!

### 4. Compare Fairly
Don't cherry-pick. Report all results.

**Baselines to beat:**
1. Random acquisition
2. Uncertainty sampling (standard active learning)
3. Diversity sampling

FK-guided needs to beat #1 and ideally #2.

---

## 📊 What Success Looks Like

### Minimum Success (Week 1):
- FK uncertainty computation works (not all equal)
- FK-guided selects different samples than random
- Can generate learning curves

### Good Success (Week 1):
- FK-guided shows ≥10% improvement over random
- Consistent across 2-3 datasets
- Learning curves show clear difference

### Great Success (Week 1):
- FK-guided shows ≥20% improvement
- Works on all tested datasets
- Clear actionable insights ("RESULTS FK needs more data")

---

## 🛠️ Code to Write Tomorrow

### File 1: `fk_active_learning.py`
```python
"""
Real FK-Guided Active Learning Implementation
"""

import numpy as np
import lightgbm as lgb
from collections import defaultdict


def compute_fk_uncertainty(ensemble, X, fk_groups):
    """Compute FK-level uncertainty (REAL VERSION)."""
    # TODO: Implement tomorrow
    pass


def fk_guided_acquisition(X_pool, y_pool, ensemble, fk_groups, budget=200):
    """Select samples from highest-uncertainty FK (REAL VERSION)."""
    # TODO: Implement tomorrow
    pass


def run_active_learning_experiment(X, y, fk_groups, strategy='fk_guided'):
    """Run full active learning experiment."""
    # TODO: Implement tomorrow
    pass


if __name__ == '__main__':
    # Test on rel-f1
    dataset = get_dataset('rel-f1')
    task = get_task('rel-f1', 'driver-position')

    X, y, fk_groups = extract_features_with_fk(dataset, task)

    # Run experiment
    results = run_active_learning_experiment(X, y, fk_groups, strategy='fk_guided')

    print(f"FK-guided improvement: {results['improvement_pct']:.1f}%")
```

---

## 🎯 Tomorrow's Success Criteria

By end of tomorrow, you should have:

1. ✅ `compute_fk_uncertainty()` function implemented
2. ✅ Tested on rel-f1
3. ✅ Verified FK uncertainties are different (not all equal)
4. ✅ Compared with fake version from Test 2

**If all 4 ✅ → proceed to Day 2**
**If any ❌ → debug before continuing**

---

## 💪 Motivation

You've already done the hard part:
- ✅ Validated 2 directions (SHAP, Decomposition)
- ✅ Fixed bugs in real-time (decomposition)
- ✅ Ran all tests in 20 minutes
- ✅ Honest assessment complete

**Now it's just implementation work.**

4 weeks from now, you'll have a complete NeurIPS paper with 3 validated directions.

**Start tomorrow. One function at a time.**

---

*Ready to begin: 2025-12-24*
*Week 1 goal: Validate FK-guided active learning works*
*Week 4 goal: Full implementation complete*
*Week 20 goal: NeurIPS submission*
