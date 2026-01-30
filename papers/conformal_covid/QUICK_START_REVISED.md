# Quick Start - REVISED Task Selection

**Date**: 2025-12-27
**Issue**: Original plan included 2 link prediction tasks (incompatible)
**Solution**: Revised to use only classification/regression tasks

---

## ❌ ORIGINAL PLAN (Had Issues)

```
1. rel-trial/study-outcome         ✓ Binary classification
2. rel-trial/condition-sponsor-run  ✗ Link prediction (incompatible)
3. rel-trial/site-sponsor-run       ✗ Link prediction (incompatible)
4. rel-f1/driver-dnf                ✓ Binary classification
```

**Problem**: Link prediction tasks have different API/structure, not comparable to our classification tasks.

---

## ✅ REVISED PLAN (All Compatible)

### Target: n=8 → n=12 (Add 4 Tasks)

| # | Dataset | Task | Type | Compatible |
|---|---------|------|------|------------|
| 1 | rel-trial | study-outcome | Binary Classification | ✓ |
| 2 | rel-trial | study-adverse | Regression | ✓ |
| 3 | rel-trial | site-success | Regression | ✓ |
| 4 | rel-f1 | driver-dnf | Binary Classification | ✓ |

**All tasks**: Standard classification/regression (no link prediction)
**Total**: n=8 (rel-salt) + 4 (new) = **n=12** ✓

---

## 📊 TASK DETAILS

### 1. rel-trial/study-outcome
- **Type**: Binary classification
- **Entity**: Study (nct_id)
- **Target**: Whether study had successful outcome
- **Train**: 11,994 rows
- **Val/Test**: 960 / 825 rows
- **Shift**: COVID-19 (2020-01-01 / 2021-01-01)
- **Features**: Need engineering from DB
- **Status**: ✅ Verified

### 2. rel-trial/study-adverse
- **Type**: Regression
- **Entity**: Study (nct_id)
- **Target**: Number of adverse events
- **Shift**: COVID-19
- **Features**: Need engineering
- **Status**: ✅ Already in paper (Table 8)

### 3. rel-trial/site-success
- **Type**: Regression
- **Entity**: Clinical trial site (facility_id)
- **Target**: Success rate
- **Shift**: COVID-19
- **Features**: Need engineering
- **Status**: ✅ Already in paper (Table 8)

### 4. rel-f1/driver-dnf
- **Type**: Binary classification
- **Entity**: Driver (driverId)
- **Target**: Did not finish race
- **Train**: 11,411 rows
- **Val/Test**: 566 / 702 rows
- **Shift**: Temporal (seasons: 2005 / 2010)
- **Features**: Need engineering from DB
- **Status**: ✅ Verified

---

## 🎯 ADVANTAGES OF REVISED PLAN

1. ✅ **All compatible** - Same task types as rel-salt (classification/regression)
2. ✅ **Paper already uses 2/4** - study-adverse and site-success in Table 8
3. ✅ **Cross-domain** - Supply chain (8) + Clinical trials (3) + Motorsports (1)
4. ✅ **COVID-19 focus** - 3/4 new tasks have COVID-19 shift
5. ✅ **Clean n=12** - Exactly our target

---

## 📋 UPDATED WEEK PLAN

### Day 1: ✅ COMPLETED
- [x] Verified task availability
- [x] Identified incompatible link prediction tasks
- [x] Revised to 4 compatible tasks
- [x] All 4 tasks load successfully

### Day 2-3: Feature Engineering (3 rel-trial tasks)

**study-outcome** (NEW - needs engineering):
- Study characteristics from DB
- Intervention types
- Sponsor information
- Estimated: 4 hours

**study-adverse** (PARTIALLY DONE - paper has this):
- Check if features already engineered
- If yes: Use existing pipeline
- If no: Engineer from scratch
- Estimated: 2-4 hours

**site-success** (PARTIALLY DONE - paper has this):
- Check existing pipeline
- Reuse or adapt
- Estimated: 2-4 hours

### Day 4: Feature Engineering (1 rel-f1 task)

**driver-dnf** (NEW - needs engineering):
- Driver characteristics
- Constructor (team) info
- Circuit characteristics
- Recent performance history
- Estimated: 4 hours

### Day 5-6: Experiments + SHAP (4 tasks)

**Conformal Prediction** (Day 5):
- 4 tasks × 50 seeds = 200 models
- Parallelized across tasks
- Estimated: 4-6 hours runtime

**SHAP Analysis** (Day 6):
- Compute SHAP values
- Compute concentration
- Estimated: 4-6 hours runtime

### Day 7: Analysis + Paper Updates

**Analysis**:
- Combine n=8 + n=4 = n=12
- Compute Spearman correlation
- Target: ρ≥0.70, p<0.02

**Paper Updates**:
- Update abstract: "12 tasks across 3 domains"
- Expand Table 3 (SHAP concentration) with 4 rows
- Update statistics throughout
- Optional: Add cross-domain validation subsection

---

## 🚀 NEXT STEPS (Start Day 2)

**Immediate**:
1. Check if study-adverse / site-success features already exist
2. If yes: Huge time savings!
3. If no: Set up feature engineering pipeline

**Feature Engineering Strategy**:
- Use RelBench's automated feature generation if available
- Otherwise, manual DB joins for entity characteristics
- Keep features simple (don't over-engineer)
- Target: Baseline performance, not SOTA

---

## 📊 SUCCESS METRICS (Unchanged)

**Minimum**: p < 0.05, ρ ≥ 0.65
**Target**: p < 0.02, ρ ≥ 0.70
**Stretch**: p < 0.01, ρ ≥ 0.75

---

**Status**: ✅ Day 1 Complete, Ready for Day 2
**Next**: Start feature engineering for rel-trial tasks
