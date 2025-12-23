# Run Test 1 TODAY - SHAP Baseline

**Time needed:** 30-60 minutes
**What it tests:** Does FK grouping make SHAP more stable?
**Success:** FK stability ρ > 0.85 AND better than individual features

---

## Step 1: Install SHAP (if needed)

```bash
pip install shap
```

---

## Step 2: Run the test

```bash
cd /Users/i767700/Github/ai-in-finance/papers/reluq/experiments
python test_1_shap_baseline.py
```

---

## Step 3: Check the output

You'll see:
```
TEST 1: SHAP Baseline - FK Grouping vs Individual Features
...
Individual Features Stability: ρ = 0.XXX
FK-Grouped Stability:          ρ = 0.XXX
Improvement:                   +X.X%

VERDICT
Result: ✅ PASS / ⚠️ MARGINAL / ❌ FAIL
Recommendation: ...
```

---

## Step 4: Interpret results

### ✅ PASS (FK ρ > 0.85 AND better than individual)
→ **Action:** Include SHAP baseline in paper
→ **Next:** Run Test 2 (Active Learning)

### ⚠️ MARGINAL (Better but < 0.85)
→ **Action:** Maybe include, downplay in paper
→ **Next:** Run Test 2, decide later

### ❌ FAIL (Not better OR < 0.85)
→ **Action:** Drop SHAP, stick with permutation
→ **Next:** Run Test 2 anyway

---

## What happens next?

Results saved to: `test_results/test_1_shap_baseline.json`

**Regardless of pass/fail, continue to Test 2**

Tests are independent - one failing doesn't doom the whole project.

---

## If it crashes...

### Error: "ModuleNotFoundError: No module named 'shap'"
```bash
pip install shap
```

### Error: "ModuleNotFoundError: No module named 'relbench'"
```bash
cd /Users/i767700/Github/ai-in-finance
pip install -e .
```

### Error: "Dataset not found"
First run will download dataset (5-10 min). Just wait.

### Other errors
Check:
1. Are you in the right conda environment? `conda activate gnn_env`
2. Is relbench installed? `pip show relbench`
3. Copy error message and we'll debug

---

## Expected runtime

- Data loading: 2-5 minutes (first time downloads dataset)
- Feature extraction: 1-2 minutes
- Training 3 models: 3-5 minutes
- SHAP computation: 10-20 minutes
- **Total: 30-60 minutes**

---

## What if I don't have time today?

Run simplified version:
```bash
# Edit test_1_shap_baseline.py
# Change: sample_size=3000 → sample_size=500
# Change: n_seeds=3 → n_seeds=2
# Runtime: ~10 minutes
```

Less rigorous but gives you a signal.

---

**START NOW. See you in 1 hour with results.** 🚀
