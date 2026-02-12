# Top 5 Actionable Improvements — Detailed Elaboration

**Date**: 2026-02-10
**Context**: UAI 2026 simulated review panel returned mean 4.5/10 (Weak Reject).
These 5 improvements address the consensus and majority issues that all/most reviewers flagged.

---

## 1. Multi-Domain Validation (all 4 reviewers)

### Why it matters
This is the single biggest weakness. Every reviewer independently identified it:
- R2: "well-executed pilot study, not a UAI paper"
- R4: "8 views of the same dataset under the same shock"
- The entire Spearman rho=0.833 claim rests on 8 tasks from one database

### What already exists (UPDATED after deep investigation)

**Completed results (50-seed APS + SHAP):**

| Dataset | Task | Type | SHAP Conc. | Drop | Seeds | SHAP pkl | APS pkl |
|---------|------|------|-----------|------|-------|----------|---------|
| rel-salt | 8 tasks | Multiclass | 23.7–54.2% | 0.1–77.1% | 50 | `results/shap/shap_rel-salt_*.pkl` | `results/ensemble_50seeds.pkl` |
| rel-trial | study-outcome | Binary | **20.8%** | **-1.3%** | 50 | `results/shap/shap_rel-trial_study-outcome.pkl` | `results/conformal/aps_rel-trial_study-outcome.pkl` |
| rel-f1 | driver-dnf | Binary | **48.1%** | **2.9%** | 50 | `results/shap/shap_rel-f1_driver-dnf.pkl` | `results/conformal/aps_rel-f1_driver-dnf.pkl` |

**SHAP-only (no multi-seed APS — regression tasks, not classification):**

| Task | Type | SHAP Conc. | CQR Drop (5-seed) | Note |
|------|------|-----------|-------------------|------|
| trial/study-adverse | Regression | 17.0% | 3.5% | NOT APS-comparable |
| trial/site-success | Regression | 34.4% | 0.0% | NOT APS-comparable |

**Key correction**: Previous version listed 1-seed classification drops for study-adverse (63.1%) and site-success (52.0%). These are unreliable single-seed artifacts. Both are actually **regression** tasks (`TaskType.REGRESSION` in `relbench/tasks/trial.py`). The SHAP was computed via binary classifier proxy but the tasks themselves predict continuous outcomes. CQR (5-seed) shows much smaller drops.

### Available COVID-era datasets for new validation

All RelBench dataset temporal splits investigated:

| Dataset | Val timestamp | Test timestamp | COVID overlap | Classification tasks |
|---------|-------------|---------------|---------------|---------------------|
| **rel-salt** | 2020-02-01 | 2020-07-01 | YES (primary) | 8 multiclass |
| **rel-trial** | 2020-01-01 | 2021-01-01 | YES (strongest) | 1 binary: study-outcome |
| **rel-stack** | 2020-10-01 | 2021-01-01 | YES (mid-COVID) | 2 binary: user-engagement, user-badge |
| **rel-hm** | 2020-09-07 | 2020-09-14 | Marginal (1 week) | 1 binary: user-churn |
| rel-f1 | 2005-01-01 | 2010-01-01 | No | 2 binary: driver-dnf, driver-top3 |
| rel-amazon | 2015-10-01 | 2016-01-01 | No | 2 binary: user-churn, item-churn |
| rel-avito | 2015-05-08 | 2015-05-14 | No | 3 binary |
| rel-event | 2012-11-21 | 2012-11-29 | No | 2 binary |

### Existing scripts that already work generically

1. **`code/run_classification_task.py`** — Binary APS with `--dataset`/`--task`/`--num_seeds` args. Already produced 50-seed results for trial/study-outcome and f1/driver-dnf. **Can be used directly** for all new binary tasks.
2. **`code/compute_shap_classification.py`** — Binary SHAP with `--dataset`/`--task` args. Already produced SHAP pickles for trial and f1 tasks. **Can be used directly**.

No new script creation needed — just run existing scripts on new datasets.

### Execution Plan

**Phase 1: Assemble n=10 from existing data (0 compute, ~2 hours)**

Data for immediate n=10 correlation (all 50-seed, already computed):
```
SALT (8):       conc = [47.3, 54.2, 50.7, 48.8, 28.9, 23.9, 23.7, 42.6]
                drop = [71.2, 77.1, 71.6, 18.5, 11.3, 10.6,  8.5,  0.1]
study-outcome:  conc = 20.8, drop = -1.3
driver-dnf:     conc = 48.1, drop = 2.9
```

Create `code/compute_cross_domain_statistics.py`:
- Load SALT data from `results/statistical_rigor.json`
- Load cross-domain SHAP from `results/shap/shap_rel-{trial,f1}_*.pkl`
- Load cross-domain APS from `results/conformal/aps_rel-{trial,f1}_*.pkl`
- Compute n=10 Spearman rho, bootstrap CI, LOO, threshold test
- Save `results/cross_domain_statistics.json`

**Phase 2: Run APS + SHAP for new tasks (8-12 hours compute, run overnight)**

Test first (fast validation, ~2 min):
```bash
cd /Users/i767700/Github/ai-in-finance/papers/conformal_covid
python3 code/run_classification_task.py --dataset rel-f1 --task driver-top3 --num_seeds 1
```

Then 50-seed runs (background/overnight):
```bash
# COVID-era (HIGH priority)
python3 code/run_classification_task.py --dataset rel-stack --task user-engagement --num_seeds 50
python3 code/run_classification_task.py --dataset rel-stack --task user-badge --num_seeds 50

# Pre-COVID controls
python3 code/run_classification_task.py --dataset rel-f1 --task driver-top3 --num_seeds 50
python3 code/run_classification_task.py --dataset rel-amazon --task user-churn --num_seeds 50
python3 code/run_classification_task.py --dataset rel-amazon --task item-churn --num_seeds 50
```

SHAP computation (independent, ~5-15 min each):
```bash
python3 code/compute_shap_classification.py --dataset rel-stack --task user-engagement
python3 code/compute_shap_classification.py --dataset rel-stack --task user-badge
python3 code/compute_shap_classification.py --dataset rel-f1 --task driver-top3
python3 code/compute_shap_classification.py --dataset rel-amazon --task user-churn
python3 code/compute_shap_classification.py --dataset rel-amazon --task item-churn
```

Estimated compute per task:

| Task | Dataset size | Per-seed | 50 seeds |
|------|-------------|----------|----------|
| f1/driver-top3 | ~850 rows | ~30s | ~25 min |
| stack/user-engagement | ~260K rows | ~2-3 min | ~2-3 hrs |
| stack/user-badge | ~260K rows | ~2-3 min | ~2-3 hrs |
| amazon/user-churn | ~1.5M (→30K subsample) | ~2-3 min | ~2-3 hrs |
| amazon/item-churn | ~1.6M (→30K subsample) | ~2-3 min | ~2-3 hrs |
| **Total** | | | **~8-12 hrs** |

rel-hm/user-churn deprioritized: 1-week test window is too short for meaningful shift detection.

**Phase 3: Combined analysis (n=15, ~2 hours)**

After all runs complete:
- Extend `compute_cross_domain_statistics.py` for n=15
- Compute split analyses:
  - COVID-era only (n=11): 8 SALT + study-outcome + 2 stack tasks
  - Non-COVID controls (n=4): driver-dnf + driver-top3 + 2 amazon tasks
  - Full combined (n=15)
- Test 40% threshold transfer (precision/recall on non-SALT tasks)
- Binary APS ceiling effect analysis

**Phase 4: Paper updates (~2 hours)**

Expand Table 6 to include SHAP concentration column + new tasks.
Update cross-domain text (lines 346-371) with combined rho.
Update limitations.

### Key risk: Binary APS ceiling effect

All new tasks are **binary** (2 classes). APS prediction sets can only be {0}, {1}, or {0,1}.
Existing data supports this concern: study-outcome drop=-1.3%, driver-dnf drop=2.9% — both near zero despite very different concentrations (20.8% vs 48.1%).

If all binary tasks cluster near zero drop, the combined correlation will be weaker than SALT-only. This is actually an **interesting finding**: "SHAP concentration predicts failure for multiclass APS where prediction sets can narrow. Binary tasks have a structural ceiling effect."

### Expected impact on reviewers
- Phase 1 alone (n=10, zero compute) gives immediate improvement
- Phase 2 gives 4 genuinely independent domains (supply chain, clinical, tech, e-commerce/motorsports)
- Satisfies R4's "3 independent domains, same threshold, no re-tuning"
- Even if binary ceiling effect weakens overall rho, this is an honest finding that adds nuance

---

## 2. Ensemble Disagreement Baseline (R1, R2)

### Why it matters
The 50-seed ensemble is ALREADY COMPUTED but never used as a pre-deployment diagnostic. Both R1 and R2 independently identified this as "an obvious baseline that is available from the 50 seeds already computed." If ensemble disagreement predicts failures equally well, SHAP concentration is unnecessary overhead.

### What already exists

The `results/ensemble_50seeds.pkl` contains **per-seed, per-task** coverage results:
```
seed_results: [{seed, val_coverage, test_coverage, coverage_drop, val_set_size, test_set_size}, ...]
```

50 seeds x 8 tasks = 400 data points with both val and test coverage per seed.

### What to compute

**Metric 1: Validation coverage variance across seeds (pre-deployment)**
```python
# For each task, compute std/CV of val_coverage across 50 seeds
# High variance on validation = unstable model = potential vulnerability
disagreement_val = {task: np.std([s['val_coverage'] for s in seeds]) for task, seeds in ...}
```

**Metric 2: Validation set-size variance across seeds (pre-deployment)**
```python
# High set-size variance suggests the model is uncertain about calibration
size_disagreement = {task: np.std([s['val_set_size'] for s in seeds]) for task, seeds in ...}
```

**Metric 3: Seed-level prediction disagreement (pre-deployment)**
- For each validation instance, count how many of the 50 models include the true label
- Low agreement = high epistemic uncertainty = vulnerability signal
- This requires going back to the raw predictions (may need recomputation if not saved)

**Metric 4: Coverage stability ratio (pre-deployment)**
```python
# Ratio of val coverage IQR to mean — captures distributional spread
stability = {task: iqr / mean for task, (iqr, mean) in ...}
```

### Analysis to run

```python
from scipy.stats import spearmanr

# For each disagreement metric, compute Spearman rho against coverage_drop
for metric_name, metric_values in [('val_std', val_stds), ('size_var', size_vars), ...]:
    rho, p = spearmanr(metric_values, coverage_drops)
    print(f'{metric_name}: rho={rho:.3f}, p={p:.3f}')
    # Compare against SHAP concentration's rho=0.833, p=0.010
```

### Possible outcomes and what they mean

| Outcome | Interpretation | Paper impact |
|---------|---------------|-------------|
| Ensemble disagreement rho > 0.833 | SHAP concentration not needed, simpler baseline wins | Major rewrite — but honest |
| Ensemble disagreement rho ~ 0.5-0.8 (n.s.) | SHAP adds value beyond ensemble | Strengthens SHAP contribution |
| Ensemble disagreement rho ~ 0 | Disagreement doesn't predict failure | SHAP is uniquely informative |

**Also test**: R3's suggestion of native LightGBM `feature_importance(importance_type='gain')` concentration. This is zero-cost (no SHAP needed). If it matches SHAP concentration, the expensive TreeExplainer computation is over-engineered.

### Effort estimate
- Metrics 1-2, 4: ~30 minutes (data already in pickle)
- Metric 3: ~2-4 hours (may need to re-run predictions if raw predictions not saved)
- Native feature importance comparison: ~1 hour (retrain 1 model per task, extract importance)
- Total: **half a day**

### Expected impact on reviewers
- R1: "Why not use model disagreement as a pre-deployment diagnostic?" — directly answered
- R2: "Standard shift detection methods are entirely absent" — partially addressed
- Even if SHAP wins, showing the comparison dramatically strengthens the paper

---

## 3. Formalize or Demote Section 4 (all 4 reviewers)

### Why it matters
All 4 reviewers flag the "theoretical grounding" as problematic. The section currently occupies "an uncomfortable middle ground" (R1) between formal theorem and informal motivation. UAI expects theoretical rigor.

### Current state
- Section 4 (~18 lines) contains:
  - "Proposition (informal)" label
  - Stochastic dominance equation (Eq. 3) stated without proof
  - "Argument" paragraph with hand-waving
  - Portfolio diversification analogy
- Listed as contribution #2 in the introduction

### Option A: Formalize (HIGH effort, HIGH impact)

**What a formal version needs:**

1. **Explicit assumptions**:
   - A1: Model f is a tree ensemble (LightGBM) with features (x_1, ..., x_p)
   - A2: SHAP concentration C = phi_1 / sum(phi_j) > tau (threshold)
   - A3: Feature x_1 undergoes complete distribution shift: P_test(x_1) ⊥ P_train(x_1), i.e., Jaccard(x_1) ≈ 0
   - A4: APS conformity score s(x,y) = sum of sorted probabilities until true label included

2. **Formal proposition**:
   - Under A1-A4, show that E[s_test] > E[s_cal] (weaker than full stochastic dominance)
   - Or show P(s_test > q_alpha) > alpha where q_alpha is the calibration quantile

3. **Proof sketch for tree ensembles specifically**:
   - When x_1 is OOD, tree splits on x_1 route to arbitrary leaves
   - If x_1 dominates (high C), most prediction mass comes from x_1 splits
   - OOD routing → predicted class ≠ true class w.h.p. → true label has low predicted probability → s(x,y) is large
   - Key insight: for APS, "confidently wrong" means small predicted probability for true label, so cumulative mass to reach true label is high → score is high

4. **Empirical verification**:
   - Plot calibration vs. test conformity score CDFs for 1 catastrophic + 1 robust task
   - Run KS test to verify stochastic dominance empirically
   - This addresses R2's complaint about "zero visual evidence"

**Effort**: 2-3 days for a solid proposition. The math isn't hard for the tree-specific case, but making it clean enough for UAI reviewers requires care.

### Option B: Demote to motivation (LOW effort, MODERATE impact)

1. **Rename**: "Theoretical Grounding" → "Intuition and Motivation"
2. **Remove from contributions list**: Delete contribution #2 or reframe as "We provide intuition for why..."
3. **Add empirical verification**: Even without formal proof, adding conformity score CDF plots is cheap and addresses R2
4. **Add caveat**: "We leave formal analysis for future work; our contribution is empirical."

**Effort**: 1-2 hours of editorial work + ~2 hours for CDF plots.

### Recommendation
**Do Option B now + add CDF plots, pursue Option A for camera-ready if accepted.**
- Demoting is honest and fast
- CDF plots are high-value, low-effort empirical evidence
- A formal proof can be developed in parallel but shouldn't block submission

### CDF plot computation

```python
# From ensemble_50seeds.pkl, extract conformity scores per seed
# For each seed: compare calibration score distribution vs test score distribution
# Plot: CDF of s_cal vs CDF of s_test for sales-shipcond (catastrophic) and sales-office (robust)
# Run 2-sample KS test
```

Note: This requires raw conformity scores, which may not be saved in the pickle (only coverage and set sizes are). If scores aren't saved, need to re-run 1 seed per task with score saving enabled. ~1 hour.

---

## 4. Compute Effective Sample Size via ICC (R1, R2, R4)

### Why it matters
The paper uses 50-seed paired Wilcoxon tests but acknowledges pseudo-replication. Three reviewers want formal quantification. The claim "significance survives effective n as low as 5" is asserted without computation.

### What already exists

Per-seed test coverage for all 8 tasks (50 seeds each) in `ensemble_50seeds.pkl`:

| Task | Mean test cov | Std | Observation |
|------|-------------|-----|-------------|
| sales-shipcond | 0.218 | 0.270 | Very high variance |
| sales-group | 0.124 | 0.323 | Extreme variance |
| sales-payterms | 0.137 | 0.270 | Very high variance |
| item-plant | 0.814 | 0.084 | Moderate variance |
| item-shippoint | 0.727 | 0.361 | Extreme variance |
| sales-incoterms | 0.870 | 0.080 | Moderate variance |
| item-incoterms | 0.837 | 0.099 | Moderate variance |
| sales-office | 0.999 | 0.000 | Near-zero variance |

### What to compute

**Step 1: Intraclass Correlation Coefficient (ICC)**

The ICC measures how much variance is between-task vs. within-task:

```python
# ICC(1,1) for coverage across seeds within tasks
# High ICC → seeds within a task are very similar → pseudo-replication is severe
# Low ICC → seeds capture meaningful variation → less pseudo-replication

# Model: coverage_ij = mu_i + epsilon_ij
# where mu_i is task mean, epsilon_ij is seed-level noise
# ICC = var(mu) / (var(mu) + var(epsilon))
```

**Step 2: Effective sample size**

```python
# n_eff = n / (1 + (k-1) * ICC)
# where n = total observations, k = cluster size (50 seeds)
# If ICC = 0.95 (seeds very correlated): n_eff = 400 / (1 + 49*0.95) ≈ 8.3
# If ICC = 0.50: n_eff = 400 / (1 + 49*0.5) ≈ 15.7
```

**Step 3: Adjusted p-values**

For each task's paired Wilcoxon test (val vs test coverage):
```python
# Original: n=50 paired observations
# Adjusted: use effective n, or cluster-robust test
# For tasks with p < 1e-8, even n_eff=5 maintains significance
# For tasks with p ~ 0.005 (item-shippoint), effective n matters more
```

**Step 4: Report in paper**

Add a row to Table 1 or a footnote:
- "ICC = X.XX across 50 seeds, yielding effective n ≈ Y per task"
- "All tasks with p < 10^-8 remain significant at effective n = 5"
- "item-shippoint (p = 0.005 at n=50) achieves p = Z at effective n = Y"

### Expected results

Looking at the data: the extreme variance for catastrophic tasks (std 0.27-0.32) suggests seeds capture real model variability (some seeds learn useful features, others don't). This means ICC might be moderate (0.5-0.8) rather than extreme (>0.95). If so, pseudo-replication is real but not devastating for the strongest results.

### Effort estimate
- ICC computation: ~30 minutes
- Adjusted p-values: ~1 hour
- Paper integration: ~30 minutes
- **Total: ~2 hours**

### Expected impact on reviewers
- Directly addresses R1, R2, R4's concern
- Transforms "acknowledged but not quantified" weakness into a reported statistic
- If ICC is moderate, this actually STRENGTHENS the paper (shows seeds aren't trivially correlated)

---

## 5. Report SHAP Concentration for Cross-Domain Tasks (R2, R4)

### Why it matters
The "cross-domain validation" (Section 5.3, Table 6) currently validates Jaccard overlap, NOT SHAP concentration. R4: "This section validates Jaccard, not the paper's main contribution." Without SHAP concentration for cross-domain tasks, the paper's central diagnostic is never tested outside SALT.

### What already exists (CORRECTED after investigation)

**50-seed APS + SHAP already computed for 2 cross-domain classification tasks:**

| Task | Domain | Type | SHAP Conc. | APS Drop (50-seed) | Files |
|------|--------|------|-----------|-------------------|-------|
| trial/study-outcome | Clinical | Binary | **20.8%** | **-1.3 ± 0.9%** | `results/conformal/aps_rel-trial_study-outcome.pkl`, `results/shap/shap_rel-trial_study-outcome.pkl` |
| f1/driver-dnf | Motorsports | Binary | **48.1%** | **2.9 ± 3.4%** | `results/conformal/aps_rel-f1_driver-dnf.pkl`, `results/shap/shap_rel-f1_driver-dnf.pkl` |

**SHAP-only for 2 regression tasks (NOT usable for APS correlation):**

| Task | Type | SHAP Conc. | CQR Drop | Note |
|------|------|-----------|----------|------|
| trial/study-adverse | **Regression** | 17.0% | 3.5% | `TaskType.REGRESSION` in relbench — NOT classification |
| trial/site-success | **Regression** | 34.4% | 0.0% | `TaskType.REGRESSION` in relbench — NOT classification |

**Critical correction**: The paper's Table 6 shows study-adverse and site-success with single-seed "classification" drops of 63.1% and 52.0%. These came from treating regression tasks as classification via a binary proxy. The actual task types are regression (`relbench/tasks/trial.py` lines 80-84, 135-138). CQR regression (5-seed) shows much smaller drops (3.5%, 0.0%). The single-seed classification drops are artifacts of applying the wrong method.

### What this means RIGHT NOW (zero compute)

Combine the 8 SALT tasks + 2 existing cross-domain binary classification tasks for n=10:

```
                    Concentration   Coverage Drop
SALT (8 tasks):     23.7–54.2%      0.1–77.1%
study-outcome:      20.8%           -1.3%
driver-dnf:         48.1%           2.9%
```

Both binary tasks show near-zero drops despite very different concentrations (20.8% vs 48.1%). This is the **binary APS ceiling effect**: prediction sets {0}, {1}, or {0,1} structurally limit how much coverage can degrade.

### What needs to be done

**Step 1: Immediate — add SHAP concentration to existing Table 6 (0 compute, ~1 hour)**

Add Conc. column to Table 6 using existing SHAP pickles. This alone addresses R2/R4's complaint that cross-domain validation doesn't test the main diagnostic.

**Step 2: Create `code/compute_cross_domain_statistics.py` (~2 hours)**

Script to:
1. Load SALT data from `results/statistical_rigor.json`
2. Load SHAP from `results/shap/shap_rel-{trial,f1}_*.pkl` (extract `concentration_val`)
3. Load APS from `results/conformal/aps_rel-{trial,f1}_*.pkl` (extract coverage drop)
4. Compute n=10 Spearman rho + bootstrap CI + LOO + threshold transfer test
5. Save `results/cross_domain_statistics.json`

**Step 3: Run 5 new tasks for n=15 (see Improvement #1 execution plan above)**

Merged with Improvement #1. After running stack, amazon, and f1/driver-top3 tasks, extend this analysis to n=15.

### Possible outcomes and framing

| Scenario | What it means | Paper framing |
|----------|-------------|---------------|
| n=10 rho strong (>0.7) | SHAP concentration generalizes across conformal methods | "Validated across 3 domains" |
| n=10 rho weak because binary tasks cluster at zero drop | Binary APS has structural ceiling | "Concentration predicts multiclass APS failure; binary APS structurally protected" |
| n=15 rho with stack/amazon tasks adds signal | Genuine multi-domain validation | "Validated across 5 domains, 15 tasks" |
| n=15 still weak | The diagnostic is SALT-specific | Honest reframe as domain-specific finding |

**The honest answer is best regardless of outcome.**

### Effort estimate
- Step 1 (add column to Table 6): ~1 hour
- Step 2 (cross-domain statistics script + analysis): ~2 hours
- Step 3 (merged with Improvement #1 compute): see above
- **Total for #5 alone: ~3 hours, zero compute**

### Expected impact on reviewers
- R2: Directly addresses "cross-domain validation does not validate the diagnostic"
- R4: Addresses "this section validates Jaccard, not SHAP concentration"
- Binary ceiling effect finding adds nuance (interesting boundary condition)

---

## Priority Matrix (UPDATED)

| # | Improvement | Hands-on | Compute | Impact | Addresses |
|---|------------|----------|---------|--------|-----------|
| 4 | ICC / effective sample size | **2 hours** | 0 | MODERATE | R1, R2, R4 |
| 5 | Cross-domain SHAP concentration (existing data) | **3 hours** | 0 | HIGH | R2, R4 |
| 2 | Ensemble disagreement baseline | **Half day** | 0 | HIGH | R1, R2, R3 |
| 3 | Demote Section 4 + CDF plots | **Half day** | ~1 hour | MODERATE | All 4 |
| 1 | Multi-domain validation (new datasets) | **Half day setup** | **8-12 hrs overnight** | VERY HIGH | All 4 |

**Recommended execution order**: 4 → 5 → 2 → 3 → 1

1. ICC (2 hrs, quick win, zero compute)
2. Cross-domain SHAP from existing data (3 hrs, zero compute, n=10 immediate)
3. Ensemble disagreement baseline (half day, zero compute, addresses 3 reviewers)
4. Section 4 editorial + CDF plots (half day)
5. New dataset runs (set up in 30 min, run overnight, analyze next day)

**Key discovery**: `run_classification_task.py` and `compute_shap_classification.py` already work generically for any RelBench dataset. No new scripts needed for #1 — just run existing ones on new datasets.

**Total hands-on effort**: ~2.5 days
**Total compute time**: ~10-13 hours (parallelizable, run overnight)
**Minimum viable result**: #4 + #5 alone (5 hours, zero compute) gives n=10 correlation + ICC quantification
