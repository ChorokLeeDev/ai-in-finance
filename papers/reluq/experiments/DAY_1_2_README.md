# Day 1-2 Active Learning Experiments

**Status:** Day 2 Complete - Multi-seed validation next
**Date:** 2025-12-24

---

## 📊 Current Status

### Validated (2/4)
1. ✅ **SHAP Baseline** - ρ = 1.000
2. ✅ **Decomposition** - 36% reduction

### Uncertain (1/4)
3. ⚠️ **Active Learning** - +50% (unvalidated, needs multi-seed)

### Failed (1/4)
4. ❌ **Causal Regimes** - Dropped

**NeurIPS Probability:** 69% (expected value of scenarios)

---

## 📁 Key Files

### Implementation
- `fk_active_learning.py` - Real FK-guided active learning (v1 + v2)

### Documentation
- `DAY_2_RESULTS.md` - Technical analysis
- `DAY_2_SUMMARY.md` - Executive summary
- `FINAL_TEST_RESULTS.md` - Day 1 results

### Quick Tests
- `test_1_shap_baseline.py` - ✅ PASS
- `test_2_active_learning.py` - ⚠️ SIMULATED (fake)
- `test_3_decomposition.py` - ✅ PASS
- `test_4_causal.py` - ❌ FAIL

---

## ⏭️ Next: Multi-Seed Validation

**Goal:** Determine if +50% is real or random variation

**Expected outcome:** ~15-25% (more realistic than 50%)

See DAY_2_SUMMARY.md for full details.
