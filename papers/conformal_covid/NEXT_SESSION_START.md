# Next Session Quick Start Guide

**Last Updated:** 2025-12-26 01:30 KST

---

## 🚀 First Thing To Do When You Return

### Step 1: Check 50-Seed Ensemble Status (ETA: 03:00)

```bash
# Check if still running
ps -p 17377

# View last 30 lines of log
tail -30 /Users/i767700/Github/ai-in-finance/papers/conformal_covid/ensemble_50seeds.log

# Check for completion marker
grep "SUMMARY" /Users/i767700/Github/ai-in-finance/papers/conformal_covid/ensemble_50seeds.log
```

**If completed:** Proceed to Step 2
**If still running:** Wait for completion, check every 30 minutes
**If crashed:** Check `COMPLETE_ROADMAP.md` → Risk Management → 50-Seed Failure

---

### Step 2: Update Table 1 with 50-Seed Results (30 min)

**A. Verify results exist:**
```bash
cd /Users/i767700/Github/ai-in-finance/papers/conformal_covid

# Check for output files
ls -lh results/ensemble_50seeds_table.tex
ls -lh results/ensemble_50seeds_summary.json

# View summary
cat results/ensemble_50seeds_summary.json
```

**B. Replace Table 1 in paper:**
```bash
# Open main.tex and replace Table 1 (lines ~112-132) with content from:
cat results/ensemble_50seeds_table.tex
```

**C. Update methodology:**
- Change "5 seeds" → "50 seeds" in Section 3.2
- Add variance reduction note in Abstract

**D. Recompile:**
```bash
pdflatex main.tex && pdflatex main.tex
open main.pdf
```

**E. Verify:**
- All tasks should have std < mean
- No variance > 40% (was problematic before)

---

### Step 3: Begin Phase 2 - Feature Importance

**See detailed plan in:** `UAI_2026_COMPLETE_ROADMAP.md` → PHASE 2

**Quick version:**

**Day 1 (Design + Code):**
1. Morning (3h): Design experiment (catastrophic vs robust tasks)
2. Afternoon (5h): Implement `analyze_feature_importance.py`

**Day 2 (Run + Analyze):**
1. Evening (4h): Run SHAP on 2 tasks (can parallelize)
2. Next day (6h): Create Figure 3, analyze results
3. Afternoon (3h): Integrate into paper

**Files to create:**
```
papers/conformal_covid/code/
├── analyze_feature_importance.py  (main script)
├── shap_utils.py                  (helpers)
└── plot_shap_results.py           (visualization)
```

---

## 📋 Current Status Summary

### ✅ Completed:
- Bootstrap CI (statistically significant)
- References fixed
- Regression tasks (3 tasks, integrated into paper)
- Paper updated with regression results

### 🔄 In Progress:
- 50-seed ensemble (Task 4/8, ETA ~2 hours from 01:30 = 03:30)

### ❌ Not Started:
- Feature Importance (SHAP) - Phase 2
- Retraining experiments - Phase 3
- Final polish

---

## 🎯 Three Paths Forward

### Option 1: Full Completion (Recommended)
**Timeline:** 9 days
**Outcome:** Accept (75% probability)
**Path:**
1. 50-seed → Update paper (by 12/27)
2. Feature Importance (12/27-12/28)
3. Retraining (12/29-12/31)
4. Polish (12/31)
5. Submission-ready by 1/3

---

### Option 2: Core Only (Time-Constrained)
**Timeline:** 4 days
**Outcome:** Weak Accept (65% probability)
**Path:**
1. 50-seed → Update paper (by 12/27)
2. Feature Importance (12/27-12/28)
3. Polish (12/29)
4. Submission-ready by 12/30
5. **Skip retraining** (can add later if reviewers request)

---

### Option 3: Minimal (Emergency)
**Timeline:** 1 day
**Outcome:** Borderline (50% probability)
**Path:**
1. 50-seed → Update paper (by 12/27)
2. Submission as-is
3. **Risk:** May get "add more experiments" reviews

---

## 📊 Key Files Reference

### Results:
```
papers/conformal_covid/results/
├── bootstrap_correlation_results.json  ✅ Done
├── regression_table.tex                ✅ Done
├── regression_findings.md              ✅ Done
├── ensemble_50seeds_table.tex          🔄 Pending
├── ensemble_50seeds_summary.json       🔄 Pending
└── shap/                               ❌ Not started
    ├── shap_rel-salt_sales-shipcond.pkl
    └── shap_rel-salt_sales-office.pkl
```

### Paper:
```
papers/conformal_covid/
├── main.tex                 (primary paper file)
├── main.pdf                 (compiled output)
├── references.bib           (bibliography)
└── figures/
    ├── figure1_main_results.png          ✅ Exists
    ├── figure2_extended_experiments.png  ✅ Exists
    ├── figure3_feature_importance.pdf    ❌ To create
    └── figure4_retraining.pdf            ❌ To create
```

### Code:
```
papers/conformal_covid/code/
├── compute_confidence_intervals.py        ✅ Exists
├── bootstrap_correlation_analysis.py      ✅ Exists
├── run_50seed_ensemble.py                 ✅ Running
├── cqr_regression.py                      ✅ Exists
├── run_regression_experiments.py          ✅ Exists
├── analyze_feature_importance.py          ❌ To create
├── shap_utils.py                          ❌ To create
├── plot_shap_results.py                   ❌ To create
├── retraining_experiment.py               ❌ To create
└── plot_retraining_results.py             ❌ To create
```

---

## 🔍 Troubleshooting

### 50-Seed Ensemble Issues:

**Problem: "Process not found" (PID 17377)**
```bash
# Check if completed
ls -lh results/ensemble_50seeds_table.tex

# If file exists: Success!
# If not: Check for crash in log
tail -100 ensemble_50seeds.log | grep -i error

# Resume from checkpoint if crashed
PYTHONPATH=/Users/i767700/Github/ai-in-finance:$PYTHONPATH \
  python3 papers/conformal_covid/code/run_50seed_ensemble.py --resume
```

**Problem: "Results file missing"**
```bash
# Check for checkpoints
ls -lh results/checkpoints/

# Individual task results might be saved
ls -lh results/*checkpoint.pkl

# Can reconstruct from checkpoints if needed
```

**Problem: "Still running after 6+ hours"**
```bash
# Check progress - should be past task 4/8
tail -10 ensemble_50seeds.log

# Check if stuck (workers not consuming CPU)
ps -ef | grep 17377 | grep spawn

# If stuck, kill and restart
kill 17377
# Then rerun with --resume
```

---

## 💡 Quick Decisions

### "Should I wait for 50-seed or start Feature Importance?"

**Wait.** 50-seed is prerequisite for everything else.

Without it:
- Paper not evaluable (variance issues)
- Can't submit to UAI

**What to do while waiting:**
- Review SHAP implementation plan in `COMPLETE_ROADMAP.md`
- Sketch out Figure 3 layout
- Read SHAP paper (Lundberg & Lee, 2017)
- Prepare code skeleton

---

### "50-seed failed, what now?"

**Don't panic.** Three options:

1. **Resume from checkpoint** (best):
   ```bash
   python3 run_50seed_ensemble.py --resume
   ```

2. **Use partial results** (if >4 tasks completed):
   - Can still show variance improvement
   - Note in paper: "50 seeds for first 5 tasks, 5 seeds for remaining"

3. **Accept 5-seed limitation** (last resort):
   - Add caveat about variance
   - Lower target venue (AISTATS instead of UAI)

---

### "Not enough time for all three phases?"

**Prioritize Feature Importance over Retraining.**

Why:
- Feature Importance adds novelty (bigger impact on scores)
- Retraining adds significance (smaller marginal gain)
- Can always add retraining post-review if requested

**Minimum viable submission:**
- 50-seed ✅
- Feature Importance ✅
- Retraining ❌ (defer)
- **Result: Weak Accept (65%)**

---

## 📚 Reference Documents

**For detailed plans:**
- `UAI_2026_COMPLETE_ROADMAP.md` - Full 9-day plan with code templates

**For context:**
- `TODAYS_ACCOMPLISHMENTS.md` - What was done on 12/25
- `READY_TO_RUN.md` - 50-seed ensemble instructions
- `BOOTSTRAP_CI_SUMMARY.md` - Bootstrap analysis results

**For quick reference:**
- `QUICK_START.md` - 3-command summary (if exists)

---

## ⏱️ Time Estimates

**If starting now (12/26 01:30):**

| Task | Start | End | Duration |
|------|-------|-----|----------|
| 50-seed completion | 01:30 | 03:30 | 2h (auto) |
| Update Table 1 | 03:30 | 04:00 | 0.5h |
| Sleep | 04:00 | 09:00 | 5h |
| **Phase 2 Start** | 09:00 | - | - |
| SHAP design | 09:00 | 12:00 | 3h |
| SHAP code | 13:00 | 18:00 | 5h |
| SHAP run | 18:00 | 22:00 | 4h |
| **12/27 end** | - | 22:00 | - |
| SHAP analysis | 09:00 | 15:00 | 6h |
| Paper integration | 15:00 | 18:00 | 3h |
| **Phase 2 Done** | - | 18:00 | **33h total** |

**Phase 2 completion: 12/28 18:00**
**Phase 3 completion: 12/30 18:00** (if doing retraining)
**Final submission ready: 1/3**

---

## 🎬 Action Items for Next Session

**Immediate (next 2 hours):**
- [ ] Check 50-seed status
- [ ] Wait for completion (if still running)

**After 50-seed completion:**
- [ ] Verify results quality
- [ ] Update Table 1 in paper
- [ ] Recompile PDF

**Then (start Phase 2):**
- [ ] Review SHAP experiment design
- [ ] Create `analyze_feature_importance.py`
- [ ] Run SHAP on 2 tasks

**Later (Phase 3):**
- [ ] Implement retraining experiment
- [ ] Run overnight experiments
- [ ] Integrate results

---

**Ready to start? Check 50-seed status first!**

```bash
tail -30 /Users/i767700/Github/ai-in-finance/papers/conformal_covid/ensemble_50seeds.log
```
