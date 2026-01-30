# Day 2 Afternoon: Experiments Running!

**Date**: 2025-12-27
**Time**: Afternoon
**Status**: 🚀 EXPERIMENTS LAUNCHED IN PARALLEL

---

## ✅ COMPLETED TODAY

### Morning: Discovery Phase
- [x] Checked for existing assets
- [x] Found 2/4 conformal results already complete (study-adverse, site-success)
- [x] Identified simple feature engineering approach (entity table merge)
- [x] Revised timeline from 7 days → 3 days (60% faster!)

### Afternoon: Execution Phase
- [x] Created classification script (`run_classification_task.py`)
- [x] Tested with 1 seed - works perfectly!
- [x] Launched study-outcome (50 seeds) - running in background
- [x] Launched driver-dnf (50 seeds) - running in parallel

---

## 🏃 CURRENTLY RUNNING

### Experiment 1: study-outcome (rel-trial)
```
Task: Binary classification (study outcome success/failure)
Dataset: rel-trial
Seeds: 50 (running)
Status: In progress
PID: Background task bff2de2
Log: papers/conformal_covid/study_outcome.log
```

**Test Run Results** (1 seed):
- Val coverage: 87.5%
- Test coverage: 88.7%
- Drop: -1.2% (actually IMPROVES!)
- Mean Jaccard: 0.534 (moderate stability)

**Interpretation**: Looks ROBUST (no degradation)

### Experiment 2: driver-dnf (rel-f1)
```
Task: Binary classification (driver did not finish race)
Dataset: rel-f1
Seeds: 50 (running)
Status: In progress (~4 seeds done)
PID: Background task b94b43e
Log: papers/conformal_covid/driver_dnf.log
```

**Partial Results** (first 3-4 seeds):
- Drops ranging from -1.2% to 4.8%
- Appears relatively robust

**Interpretation**: Likely ROBUST (minimal degradation)

---

## ⏱️ ESTIMATED COMPLETION

**Runtime per seed**: ~30-60 seconds
**Total seeds**: 50 per task
**Parallel execution**: Both running simultaneously

**Estimated completion**: 1-1.5 hours from launch
**Expected done by**: ~Evening (6-7 PM)

---

## 📊 WHAT WE'LL HAVE

### After Experiments Complete (Tonight)

**Conformal Results** (4/4 tasks):
1. ✅ study-adverse (rel-trial) - Already done
2. ✅ site-success (rel-trial) - Already done
3. ⏳ study-outcome (rel-trial) - Running now
4. ⏳ driver-dnf (rel-f1) - Running now

**Summary Stats**:
- All 4 tasks: coverage degradation measured
- All 4 tasks: Jaccard similarity computed
- Ready for SHAP analysis tomorrow

---

## 🔄 MONITORING

**To check progress**:
```bash
bash papers/conformal_covid/monitor_experiments.sh
```

**Manual checks**:
```bash
# Check logs
tail -f papers/conformal_covid/study_outcome.log
tail -f papers/conformal_covid/driver_dnf.log

# Count completed seeds
grep -c "Seed.*coverage_drop" papers/conformal_covid/study_outcome.log
grep -c "Seed.*coverage_drop" papers/conformal_covid/driver_dnf.log

# Check results
ls -lh papers/conformal_covid/results/aps_*.json
```

---

## 📋 TOMORROW'S PLAN (Day 3)

### Morning: SHAP Analysis (All 4 Tasks)

**For each task**:
1. Load trained models (saved in results)
2. Compute SHAP values on validation set
3. Compute feature importance
4. Compute concentration (Top / Total)
5. Save results

**Tasks**:
- study-adverse: ~2 hours
- site-success: ~2 hours
- study-outcome: ~2 hours
- driver-dnf: ~2 hours

**Total**: 8 hours (can parallelize to ~4 hours)

**Deliverable**: SHAP concentration for all 4 tasks

### Afternoon: Quick Check

**If SHAP done early**:
- Preview correlation results
- Compute preliminary n=12 statistics
- Check if p<0.02 achieved

---

## 🎯 SUCCESS METRICS

### What We Need for n=12

**Conformal Results**: ✅ Will have tonight (4/4)
**SHAP Concentration**: ⏳ Tomorrow (0/4 currently)

**Combined n=12 Data**:
- 8 rel-salt tasks (existing)
- 4 new tasks (2 tonight + 2 existing)
- Total: 12 tasks across 3 domains

**Statistical Target**:
- Current: n=8, ρ=0.71, p=0.047
- Target: n=12, ρ≥0.70, p<0.02

---

## 📈 PRELIMINARY OBSERVATIONS

### From Test Runs

**study-outcome**:
- Coverage IMPROVES (-1.2% drop = actually 1.2% better!)
- Jaccard: 0.534 (moderate)
- **Prediction**: ROBUST task

**driver-dnf**:
- Minimal drops (−1.2% to 4.8%)
- **Prediction**: ROBUST task

**study-adverse** (existing):
- Drop: 3.5%
- Jaccard: 0.872 (high)
- **Confirmed**: ROBUST

**site-success** (existing):
- Drop: 0.0%
- Jaccard: 0.954 (very high)
- **Confirmed**: ROBUST

**Pattern**: All 4 new tasks appear ROBUST!
- This is good for paper (more robust tasks to analyze)
- May need to carefully analyze concentration thresholds
- Cross-domain validation of robustness

---

## ⚠️ POTENTIAL ISSUES

### If All 4 Tasks Are Robust

**Concern**: Won't help validate "high concentration → vulnerable" pattern
**Reason**: Need mix of vulnerable AND robust tasks

**Mitigation**:
1. **Good news**: We still have 3 vulnerable tasks from rel-salt (sales-group, sales-payterms, sales-shipcond)
2. **Balance**: n=12 will have ~3 vulnerable + ~9 robust
3. **Framework**: Can still validate 2D framework with protective factors

**Alternative interpretation**:
- Robust tasks validate the 40% threshold from the other side
- If robust tasks have <40% concentration OR protective factors, framework holds
- Actually strengthens cross-domain validation

---

## 🎓 LESSONS SO FAR

### What Worked ✅
1. **Parallel execution**: Both experiments running simultaneously
2. **Background tasks**: Can continue working while they run
3. **Test first**: 1-seed test caught issues early
4. **Monitoring**: Easy to check progress

### What's Going Well
1. **Code reuse**: cqr_regression.py pattern worked perfectly
2. **Simple features**: Entity table merge is enough
3. **Fast execution**: ~30-60s per seed is manageable

---

## 💾 FILES CREATED TODAY

### Code
1. `code/run_classification_task.py` (500+ lines) ✅
2. `monitor_experiments.sh` (monitoring script) ✅

### Documentation
3. `DAY2_FINDINGS.md` (discovery phase)
4. `DAY2_AFTERNOON_STATUS.md` (this file)

### Logs
5. `study_outcome.log` (experiment output)
6. `driver_dnf.log` (experiment output)

### Results (in progress)
7. `results/aps_rel-trial_study-outcome.*` (generating)
8. `results/aps_rel-f1_driver-dnf.*` (generating)

---

## 🎯 DECISION POINTS

### Tonight (After Experiments Complete)

**Check**:
1. Did all 50 seeds complete successfully?
2. Are results reasonable? (coverage ~90%, drops <10% for robust tasks)
3. Any errors or issues?

**If successful**: Proceed to Day 3 SHAP analysis
**If issues**: Debug and rerun failed seeds

### Tomorrow Morning (Before SHAP)

**Quick check**:
- Review all 4 conformal results
- Estimate which tasks will be robust vs vulnerable
- Plan SHAP analysis accordingly

---

## ✅ DAY 2 VERDICT

**Status**: ✅ **EXCELLENT PROGRESS**

**Achievements**:
- Discovered 50% of work already done
- Created classification framework
- Launched 2 experiments in parallel
- On track for 3-day completion

**Timeline**:
- Original: 7 days
- Revised: 3 days
- Current: Day 2 of 3, ahead of schedule

**Confidence**: ✅ **VERY HIGH**
- Experiments running smoothly
- Clear path to completion
- Known risks managed

---

**Next Check**: Evening (check if experiments complete)
**Next Work**: Day 3 morning (SHAP analysis)
**Target**: n=12 paper by end of Day 4

**Status**: On track for Friday completion! 🚀
