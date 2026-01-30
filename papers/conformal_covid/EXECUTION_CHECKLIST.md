# Execution Checklist: Addressing Critical Issues
## Week-by-Week Action Items

**Last Updated**: 2025-12-27
**Target Completion**: 6 weeks from today (2025-02-07)

---

## Quick Wins (This Week) ⚡

### ✅ Issue #3: Reframe "Mechanism" Language (2-3 days)
**Status**: ✅ READY TO IMPLEMENT (text prepared)
**Impact**: HIGH (scientific rigor)
**Effort**: LOW (search-replace)

- [x] Document created: `REFRAMING_MECHANISM_DISCOVERY.md`
- [ ] Edit abstract (Line 34)
- [ ] Edit introduction (Lines 67-69)
- [ ] Edit Section 4.4 title
- [ ] Edit Section 4.4 mechanism insight (Line 274)
- [ ] Edit conclusion (Line 515)
- [ ] Add causal validation section to Discussion
- [ ] Regenerate PDF and verify changes

**Files to edit**: `main.tex`

**Success criteria**:
- No instances of "mechanism discovery" without caveats
- All causal language changed to correlational
- Causal validation added to future work

---

### ✅ Issue #2: Document 2D Framework (1 day)
**Status**: ✅ CODE WORKING (87.5% accuracy)
**Impact**: HIGH (fixes critical flaw)
**Effort**: LOW (LaTeX only)

- [x] 2D framework code validated: `revised_decision_framework.py`
- [x] LaTeX section generated: `revised_decision_framework.tex`
- [ ] Copy LaTeX to Section 6 (Decision Framework)
- [ ] Create 2D decision flowchart figure
- [ ] Update abstract with 2D framework mention
- [ ] Add sales-office example to Section 4.4

**Files to edit**: `main.tex`, `figures/decision_framework_2d.pdf` (new)

**Success criteria**:
- Section 6 contains full 2D algorithm
- Flowchart clearly shows protective factor check
- sales-office explained as validation case

---

## Week 1-2: Issue #2 - Validate 2D Framework 🔧

### TODO: Compute Secondary Features for All Tasks
**Status**: ⏳ NOT STARTED
**Blockers**: Need SHAP analysis results

**Tasks**:
- [ ] Re-run SHAP analysis for all 8 tasks
- [ ] For each task, identify top-5 features
- [ ] Compute Jaccard similarity for each top-5 feature
- [ ] Identify protective factors (Jaccard > 0.5, Importance > 15%)
- [ ] Update Table 3 with secondary feature data
- [ ] Re-run 2D framework validation
- [ ] Update paper with results

**Script to run**:
```bash
# Need to create this script
python code/compute_secondary_features.py --dataset rel-salt --all-tasks
```

**Output**: Updated `results/shap/secondary_features.csv`

**Success criteria**:
- All 8 tasks have secondary feature data
- i-shippoint correctly classified (currently misclassified)
- 2D framework accuracy ≥ 87.5%

**Estimated effort**: 3-4 days
- Day 1: SHAP computation
- Day 2: Jaccard computation
- Day 3: Framework validation
- Day 4: Paper updates

---

## Week 3-4: Issue #4 - Continuous Feature Validation 🧪

### TODO: Test Concentration on Regression Tasks
**Status**: ⏳ NOT STARTED
**Blockers**: Need feature type identification

**Tasks**:
- [ ] Load regression task data (study-adverse, site-success, driver-position)
- [ ] Identify which features are continuous vs categorical
- [ ] Compute SHAP concentration separately by feature type
- [ ] Test if 40% threshold holds for continuous features
- [ ] Update Table 8 with concentration data
- [ ] Add continuous feature section to paper

**Script to run**:
```bash
python code/regression_shap_concentration.py --datasets rel-f1 rel-trial
```

**Expected outcomes**:
1. **Best case**: 40% threshold holds for both categorical and continuous ✓
2. **Likely case**: Need different thresholds (e.g., 50% for continuous)
3. **Worst case**: No correlation for continuous (categorical-specific)

**Success criteria**:
- At least 3 continuous-dominant tasks tested
- Correlation computed for continuous features
- Threshold recommendation for continuous features
- Updated decision framework with feature-type-specific guidance

**Estimated effort**: 5-7 days
- Day 1-2: Feature type identification
- Day 3-4: SHAP computation
- Day 5-6: Analysis and correlation tests
- Day 7: Paper updates

---

## Week 5-6: Issue #1 - Expand to n=20+ Tasks 📊

### TODO: Feature Engineering for New Datasets
**Status**: ⏳ NOT STARTED
**Blockers**: Need feature engineering pipeline

**Tasks**:
- [ ] Set up feature engineering for rel-trial (3 new tasks)
- [ ] Set up feature engineering for rel-f1 (4 new tasks)
- [ ] Set up feature engineering for rel-amazon (4 tasks)
- [ ] Set up feature engineering for rel-stack (3 tasks)
- [ ] Run conformal prediction experiments (50 seeds each)
- [ ] Compute SHAP concentration for all new tasks
- [ ] Update Table 3 with all tasks
- [ ] Re-compute correlation statistics

**Script to run**:
```bash
python code/expand_validation_tasks.py --num_seeds 50
```

**Target**:
- Total tasks: n ≥ 20
- Statistical power: Detect ρ ≥ 0.5 with 80% power
- Correlation p-value: p < 0.01 (currently p=0.047)

**Expected outcomes**:
1. **Best case**: ρ improves to 0.75-0.80, p<0.001 ✓✓✓
2. **Good case**: ρ stays 0.70-0.75, p<0.01 ✓✓
3. **Acceptable case**: ρ stays 0.65-0.70, p<0.05 ✓
4. **Problematic**: ρ drops below 0.6 or p>0.05 ⚠️

**Success criteria**:
- At least 20 tasks total
- Spearman ρ with p < 0.01
- 40% threshold validated (or adjusted) across all tasks
- Cross-domain validation (4+ datasets)

**Estimated effort**: 10-14 days
- Week 1: Feature engineering (4 datasets)
- Week 2: Experiments + analysis

---

## Parallel Track: Paper Revisions 📝

### Minor Edits (Ongoing)
- [ ] Fix all TODO comments in LaTeX
- [ ] Update figure captions with 2D framework
- [ ] Proofread for consistency
- [ ] Spell check

### Major Additions (After experiments)
- [ ] New section: "Cross-Domain Validation" (n=20 results)
- [ ] New section: "Feature Type Analysis" (continuous vs categorical)
- [ ] Updated abstract with n=20 and continuous validation
- [ ] Updated conclusion with strengthened claims

---

## Progress Tracking

### Week 1 (Dec 27 - Jan 2)
- [x] ✅ Ultrathink review completed
- [x] ✅ 2D framework implemented and validated
- [x] ✅ Action plan documented
- [ ] ⏳ Issue #3: Reframe mechanism language
- [ ] ⏳ Issue #2: Integrate 2D framework into paper

**Target**: Language reframed, 2D framework in paper

### Week 2 (Jan 3 - Jan 9)
- [ ] ⏳ Issue #2: Compute secondary features
- [ ] ⏳ Issue #2: Validate 2D framework on all tasks

**Target**: 2D framework fully validated

### Week 3-4 (Jan 10 - Jan 23)
- [ ] ⏳ Issue #4: Continuous feature validation
- [ ] ⏳ Issue #4: Update paper with findings

**Target**: Continuous features validated

### Week 5-6 (Jan 24 - Feb 7)
- [ ] ⏳ Issue #1: Feature engineering for 12 new tasks
- [ ] ⏳ Issue #1: Run experiments
- [ ] ⏳ Issue #1: Update paper with n=20+ results

**Target**: Full paper ready for submission

---

## Risk Management

### High Risk Items
🔴 **Feature engineering for new datasets**
- Risk: May take longer than expected
- Mitigation: Start with easiest datasets (rel-trial, rel-f1)
- Fallback: Aim for n=15 instead of n=20

🔴 **Continuous feature validation may fail**
- Risk: 40% threshold may not work for continuous
- Mitigation: Accept different thresholds as finding
- Fallback: Limit scope to "categorical feature tasks"

### Medium Risk Items
🟡 **2D framework may not improve i-shippoint**
- Risk: Still misclassified even with protective factors
- Mitigation: Deeper analysis of i-shippoint
- Fallback: Acknowledge as limitation (87.5% still good)

🟡 **n=20 correlation may weaken**
- Risk: ρ could drop with more tasks
- Mitigation: Analyze domain-specific effects
- Fallback: Report heterogeneous effects across domains

---

## Daily Standup Questions

**What did I complete yesterday?**
- [ ] _Fill in_

**What am I working on today?**
- [ ] _Fill in_

**What blockers do I have?**
- [ ] _Fill in_

---

## Next Session Priorities

**Right now (next 2 hours)**:
1. ✅ Issue #3: Edit main.tex to reframe "mechanism" → "signal"
2. ✅ Issue #2: Copy 2D framework LaTeX to Section 6
3. ✅ Regenerate PDF and verify changes

**This week**:
1. Issue #2: Compute secondary features for all tasks
2. Issue #2: Create 2D framework flowchart

**Next week**:
1. Issue #4: Start continuous feature validation

---

## Success Metrics

**Minimum viable (conference acceptance)**:
- [x] Issue #3: Language reframed ✓
- [ ] Issue #2: 2D framework integrated ⏳
- [ ] Issue #4: ≥3 continuous tasks validated ⏳
- [ ] Issue #1: n ≥ 15, p < 0.05 ⏳

**Strong paper (high confidence)**:
- All minimum criteria ✓
- [ ] Issue #1: n ≥ 20, p < 0.01
- [ ] Issue #4: Separate thresholds for categorical/continuous

**Exceptional (best paper track)**:
- All strong criteria ✓
- [ ] n ≥ 25 across 5+ domains
- [ ] Deep learning validation
- [ ] Theoretical contribution

---

## Contact & Coordination

**Lead Author**: Chorok Lee
**Collaborators**: [Add if any]
**Target Venue**: UAI 2026 / AISTATS 2026
**Submission Deadline**: [Add deadline]

---

**Last Updated**: 2025-12-27
**Next Review**: Weekly (every Monday)
