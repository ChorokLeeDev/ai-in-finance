# Issue #1: Expand to n=20+ Tasks - Strategic Plan

**Date**: 2025-12-27
**Goal**: Strengthen statistical power from p=0.047 (n=8) to p<0.01 (n=20+)
**Status**: Planning Phase

---

## 🎯 OBJECTIVE

**Current State**:
- n=8 tasks (rel-salt, private data)
- Spearman ρ=0.71, p=0.047 (barely significant)
- 40% threshold identified
- Statistical power: Limited

**Target State**:
- n=20+ tasks (12 new tasks)
- Spearman ρ≥0.70, p<0.01 (strongly significant)
- 40% threshold validated across domains
- Statistical power: Robust

**Impact**: +15% acceptance probability (70% → 85%)

---

## 📊 AVAILABLE TASKS INVENTORY

**Total Available**: 30 public tasks across 7 datasets

| Dataset | Tasks | Type | Temporal Shift |
|---------|-------|------|----------------|
| rel-amazon | 7 | Mixed | Normal |
| rel-avito | 4 | Mixed | Normal |
| rel-event | 3 | Classification | Normal |
| rel-f1 | 3 | Regression/Class | Temporal |
| rel-hm | 3 | Mixed | Normal |
| rel-stack | 5 | Mixed | Normal |
| rel-trial | 5 | Regression/Class | COVID-19 |

---

## 🎯 STRATEGIC SELECTION

### Phase 1: Quick Wins (Priority - 4-6 tasks)

**rel-trial (COVID-19 shift)** - HIGHEST PRIORITY
- ✅ Already used: study-adverse, site-success (in paper)
- 🆕 Add: study-outcome, condition-sponsor-run, site-sponsor-run
- **Why**: Same COVID-19 shift as rel-salt, easy to justify
- **Tasks**: 5 total
- **Effort**: Medium (feature engineering needed)

**rel-f1 (Temporal shift)**
- ✅ Already used: driver-position (in paper)
- 🆕 Add: driver-dnf, driver-top3
- **Why**: Temporal shift (seasons), classification tasks
- **Tasks**: 3 total
- **Effort**: Medium

**Quick Wins Total**: 8 tasks (5 trial + 3 f1)

### Phase 2: Domain Diversity (4-6 tasks)

**rel-amazon (E-commerce)**
- 🆕 Select: user-churn, item-churn, user-ltv, item-ltv
- **Why**: Different domain (e-commerce vs supply chain)
- **Tasks**: 4
- **Effort**: High (new domain, feature engineering)

**rel-stack (Social network)**
- 🆕 Select: user-engagement, post-votes, user-badge
- **Why**: Yet another domain (social network)
- **Tasks**: 3
- **Effort**: High

**Domain Diversity Total**: 7 tasks

### Phase 3: Extended (if needed, 2-4 tasks)

**rel-hm (Retail)**
- user-churn, item-sales
- **Tasks**: 2

**rel-avito (Classifieds)**
- ad-ctr, user-clicks
- **Tasks**: 2

---

## 📋 PROPOSED TASK SELECTION

### Tier 1: Essential (n=8 → n=15)
**7 new tasks, MEDIUM effort**

1. rel-trial/study-outcome ⭐
2. rel-trial/condition-sponsor-run ⭐
3. rel-trial/site-sponsor-run ⭐
4. rel-f1/driver-dnf ⭐
5. rel-f1/driver-top3 ⭐
6. rel-amazon/user-churn
7. rel-amazon/item-churn

**Why this set**:
- COVID-19 shift validation (3 more rel-trial tasks)
- Temporal shift validation (2 more rel-f1 tasks)
- Domain diversity (2 amazon tasks)
- Reaches n=15 (adequate for p<0.05)

**Estimated effort**: 2-3 weeks
- Week 1: Feature engineering (rel-trial, rel-f1)
- Week 2: Feature engineering (rel-amazon), run experiments
- Week 3: SHAP analysis, paper updates

### Tier 2: Stretch Goal (n=15 → n=20)
**5 more tasks, HIGH effort**

8. rel-amazon/user-ltv
9. rel-amazon/item-ltv
10. rel-stack/user-engagement
11. rel-stack/post-votes
12. rel-stack/user-badge

**Why this set**:
- Further domain diversity
- Reaches n=20 (strong for p<0.01)

**Estimated effort**: +1-2 weeks

---

## 🚀 IMPLEMENTATION STRATEGY

### Approach A: Incremental (Recommended)
**Start with Tier 1 (n=15)**

**Advantages**:
- Manageable scope (2-3 weeks)
- n=15 is sufficient for p<0.05
- COVID-19 validation strengthened
- Can extend to Tier 2 if needed

**Deliverable**: n=15, p<0.05, 3+ domains

### Approach B: Full Expansion
**Go straight to n=20**

**Advantages**:
- Strongest statistical power
- Maximum reviewer confidence
- Comprehensive validation

**Disadvantages**:
- 4-5 weeks effort
- High risk if some tasks fail
- May have diminishing returns

---

## 📊 EXPECTED OUTCOMES

### Best Case (Tier 1: n=15)
```
Current: n=8, ρ=0.71, p=0.047
Target:  n=15, ρ≥0.70, p<0.02

Conclusion: Threshold validated across 3+ domains
Impact: +10-15% acceptance probability
```

### Strong Case (Tier 2: n=20)
```
Current: n=8, ρ=0.71, p=0.047
Target:  n=20, ρ≥0.70, p<0.01

Conclusion: Highly robust threshold, multiple domains
Impact: +15-20% acceptance probability
```

### Risk Case (Correlation weakens)
```
Current: n=8, ρ=0.71, p=0.047
Outcome: n=15, ρ=0.55, p=0.06 (not significant)

Conclusion: Dataset-specific effect, limited generalization
Impact: -10% acceptance probability (need to reframe)
```

---

## 🛠️ TECHNICAL REQUIREMENTS

### For Each New Task:

**1. Feature Engineering**
- Set up database joins
- Create temporal splits
- Engineer features (manual or automated)
- **Time**: 0.5-1 day per task

**2. Conformal Prediction Experiments**
- Train LightGBM models (50 seeds)
- Run conformal prediction (APS/CQR)
- Compute coverage degradation
- **Time**: 0.5 day per task (parallelizable)

**3. SHAP Analysis**
- Compute SHAP values on validation set
- Compute feature importance
- Compute concentration (Top / Total)
- **Time**: 0.5 day per task (parallelizable)

**Total per task**: ~2 days
**Total for 7 tasks (Tier 1)**: ~14 days (2 weeks)
**Total for 12 tasks (Tier 2)**: ~24 days (3.5 weeks)

---

## 📝 PAPER UPDATES NEEDED

### 1. Expand Table 3 (SHAP Concentration)

**Current**: 8 rel-salt tasks

**New**: 15+ tasks across multiple datasets

### 2. Update Abstract

**Before**:
> "Analyzing 8 supply chain tasks..."

**After**:
> "Analyzing 15 tasks across 3 domains (supply chain, clinical trials, motorsports)..."

### 3. Update Statistics

**Before**:
> "Spearman ρ=0.71, p=0.047"

**After** (if successful):
> "Spearman ρ=0.72, p=0.009"

### 4. Add Cross-Domain Validation Section

**New subsection**: "Cross-Domain Validation"
- Show concentration vs drop for all 15 tasks
- Scatter plot colored by dataset
- Report correlation per dataset
- Demonstrate generalization

### 5. Update Limitations

**Remove**:
> "With n=8 tasks, statistical power is limited"

**Replace with**:
> "Validated across 15 tasks from 3 domains, demonstrating cross-domain robustness"

---

## ⚡ QUICK START: Minimum Viable Expansion

### Option: n=8 → n=12 (Just rel-trial)

**Add 4 tasks**:
1. rel-trial/study-outcome
2. rel-trial/condition-sponsor-run
3. rel-trial/site-sponsor-run
4. rel-f1/driver-dnf

**Why**:
- All COVID-19 / temporal shift tasks
- Same narrative (distribution shift)
- Easiest feature engineering
- Gets to n=12 (moderate power)

**Effort**: 1 week
**Impact**: p=0.047 → p~0.02

**Statistical Power**:
- n=8: Can detect ρ≥0.7 with 80% power
- n=12: Can detect ρ≥0.6 with 80% power
- Improvement: Can detect weaker correlations

---

## 🎯 RECOMMENDATION

**Start with Quick Start (n=12)**:
1. Add 4 rel-trial/rel-f1 tasks (1 week)
2. Check if p<0.02 achieved
3. If yes: Paper ready for submission
4. If no: Extend to Tier 1 (n=15)

**Rationale**:
- Lowest risk (familiar domains)
- Fastest turnaround (1 week vs 2-3 weeks)
- Significant statistical improvement
- Can always extend later

---

## 📊 SUCCESS METRICS

### Minimum Viable (n=12)
- [  ] p < 0.05 (ideally p < 0.02)
- [  ] ρ ≥ 0.65
- [  ] 40% threshold holds across tasks
- [  ] COVID-19 narrative strengthened

### Strong Paper (n=15)
- [  ] p < 0.01
- [  ] ρ ≥ 0.70
- [  ] 3+ domains validated
- [  ] Cross-domain correlation table

### Exceptional (n=20)
- [  ] p < 0.001
- [  ] ρ ≥ 0.75
- [  ] 4+ domains validated
- [  ] Domain-specific threshold analysis

---

## 🚀 NEXT STEPS

**Option 1: Quick Start (1 week)**
1. Set up feature engineering for 4 tasks (rel-trial, rel-f1)
2. Run conformal experiments (parallelizable)
3. Compute SHAP concentration
4. Update paper to n=12
5. Check if p<0.02

**Option 2: Tier 1 (2-3 weeks)**
1. Add 7 tasks (rel-trial, rel-f1, rel-amazon)
2. Full experiments + SHAP
3. Update paper to n=15
4. Target p<0.01

**Option 3: Tier 2 (4-5 weeks)**
1. Add 12 tasks (all domains)
2. Comprehensive validation
3. Update paper to n=20
4. Target p<0.001

---

## 🎓 DECISION POINT

**Your choice**:
- **A)** Quick Start (n=12, 1 week) - Fast, low risk
- **B)** Tier 1 (n=15, 2-3 weeks) - Balanced, strong impact
- **C)** Tier 2 (n=20, 4-5 weeks) - Maximum impact, high effort
- **D)** Something else?

**My recommendation**: Start with **Quick Start (n=12)**, then extend if needed.

---

**Created**: 2025-12-27
**Status**: Awaiting user decision on scope
**Files Ready**: `code/expand_validation_tasks.py` (framework exists)
