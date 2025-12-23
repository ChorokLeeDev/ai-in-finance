# Week 1-2 Action Plan: Rapid Experimental Validation

**Goal:** Test all 4 research directions, see what works, decide scope by Week 3
**Time Budget:** 12 days
**Output:** JSON results file + strategic recommendation

---

## Day 1-2: SHAP Baseline Test

### Question
Does FK grouping make SHAP attribution more stable than individual features?

### Experiment
```bash
cd papers/reluq/experiments
python quick_test_suite.py --test shap --dataset rel-salt
```

### What to Measure
1. **Stability:** Spearman ρ of FK attribution across 3 seeds
2. **Comparison:** FK-grouped SHAP vs individual feature SHAP
3. **Expected:** ρ(FK-SHAP) > 0.85

### Success Criteria
- ✅ **PASS:** ρ > 0.85 AND better than individual features
- ❌ **FAIL:** ρ ≤ 0.85 OR worse than individual

### If PASS
→ Include SHAP baseline comparison in main paper (adds rigor)

### If FAIL
→ Drop SHAP comparison, stick with permutation-based method

### Time
- Script setup: 4 hours
- Run experiments: 2 hours
- Analysis: 2 hours
- **Total: 8 hours (1 day)**

---

## Day 3-5: Active Learning Test

### Question
Does FK-guided data acquisition beat random sampling by >20%?

### Experiment
```bash
python quick_test_suite.py --test active_learning --dataset rel-salt
```

### What to Measure
1. **Learning curves:** Accuracy vs samples acquired
2. **Efficiency:** Samples needed to reach 90% of final accuracy
3. **Expected:** FK-guided needs 20% fewer samples than random

### Success Criteria
- ✅ **PASS:** >20% efficiency gain over random
- ⚠️ **MARGINAL:** 10-20% gain (include but don't emphasize)
- ❌ **FAIL:** <10% gain

### If PASS
→ Active learning becomes main novelty contribution (NeurIPS worthy)

### If MARGINAL
→ Include as "application" section, not main contribution

### If FAIL
→ Drop active learning entirely

### Time
- Implement acquisition strategies: 8 hours
- Run simulations (5 iterations): 4 hours
- Generate learning curves: 4 hours
- **Total: 16 hours (2 days)**

---

## Day 6-8: Epistemic/Aleatoric Decomposition Test

### Question
Can we separate epistemic (data scarcity) from aleatoric (inherent noise) uncertainty by FK?

### Experiment
```bash
python quick_test_suite.py --test decomposition --dataset rel-salt
```

### What to Measure
1. **Method 1:** Heteroscedastic model (predict mean + variance separately)
2. **Method 2:** Data augmentation (add synthetic FK data, measure reduction)
3. **Validation:** Inject known epistemic noise → verify decomposition detects it

### Success Criteria
- ✅ **PASS:** Can separate with >70% accuracy on synthetic test
- ⚠️ **MARGINAL:** 50-70% accuracy (interesting but weak)
- ❌ **FAIL:** <50% accuracy (can't reliably separate)

### If PASS
→ Include as theoretical extension (workshop paper material)

### If MARGINAL
→ Mention in future work, don't include experiments

### If FAIL
→ Drop entirely (too hard for timeline)

### Time
- Implement heteroscedastic model: 8 hours
- Synthetic validation: 6 hours
- Real data testing: 6 hours
- **Total: 20 hours (2.5 days)**

---

## Day 9-12: Causal Attribution Test

### Question
Does interventional (causal) attribution differ from observational (correlational)?

### Experiment
```bash
python quick_test_suite.py --test causal --dataset rel-f1
```

### What to Measure
1. **Observational:** Permutation-based attribution (current method)
2. **Interventional:** Set FK to mean, measure uncertainty change
3. **Compare rankings:** Do causal vs observational give different FK priorities?

### Success Criteria
- ✅ **PASS:** Top-1 FK differs AND makes domain sense
  - Example: Causal says "DRIVER", observational says "RESULTS"
  - Domain expert: "Yes, driver skill is root cause"
- ⚠️ **MARGINAL:** Rankings differ slightly (top-2 swap)
- ❌ **FAIL:** Same ranking (no added value)

### If PASS
→ Major theoretical contribution (UAI/CLeaR workshop material, maybe NeurIPS)

### If MARGINAL
→ Interesting but not strong enough for main paper

### If FAIL
→ Drop (FK relationships aren't causal in this domain)

### Time
- Implement intervention mechanism: 10 hours
- Run on rel-f1 (known structure): 4 hours
- Domain validation (check if makes sense): 6 hours
- **Total: 20 hours (2.5 days)**

---

## Day 13-14: Synthesis & Decision

### Generate Summary Report

Run full test suite:
```bash
python quick_test_suite.py --run-all
cat quick_test_results/test_results.json
```

### Expected Output

```json
{
  "shap": {
    "verdict": "PASS",
    "stability": 0.89,
    "recommendation": "Include SHAP baseline"
  },
  "active_learning": {
    "verdict": "PASS",
    "improvement_pct": 23.5,
    "recommendation": "Include in NeurIPS"
  },
  "decomposition": {
    "verdict": "MARGINAL",
    "accuracy": 0.68,
    "recommendation": "Workshop paper only"
  },
  "causal": {
    "verdict": "PASS",
    "rankings_differ": true,
    "recommendation": "Strong contribution"
  }
}
```

### Decision Tree

**Scenario 1: 4/4 PASS** (Best case)
```
NeurIPS Main Paper Scope:
✅ Core FK attribution
✅ SHAP baseline comparison
✅ Active learning (main novelty)
✅ Epistemic/aleatoric decomposition
✅ Causal attribution

Title: "FK-Guided Uncertainty Analysis: A Unified Framework"
Sections: 9 pages
Probability: 90%
```

**Scenario 2: 3/4 PASS** (Likely case)
```
Example: SHAP, Active Learning, Causal all pass; Decomposition marginal

NeurIPS Main Paper:
✅ Core FK attribution
✅ SHAP baseline
✅ Active learning
✅ Causal attribution

ICML Workshop:
✅ Epistemic/aleatoric (partial results)

Probability: 85% (NeurIPS) + 95% (workshop) = 1.8 papers expected
```

**Scenario 3: 2/4 PASS** (Conservative case)
```
Example: SHAP + Active Learning pass; others fail

NeurIPS Main Paper:
✅ Core FK attribution
✅ SHAP baseline
✅ Active learning

KDD Backup (same content):
Ready if NeurIPS rejects

Probability: 75% (NeurIPS) or 85% (KDD)
```

**Scenario 4: ≤1 PASS** (Worst case)
```
Only core FK attribution works

Submit to KDD directly (Week 6):
✅ Core FK attribution
✅ Error Propagation Hypothesis
✅ 6 domain validation

Probability: 85% (KDD loves this)
```

---

## Parallel Workstreams (If You Have Help)

### Solo Timeline
- Day 1-2: SHAP
- Day 3-5: Active Learning
- Day 6-8: Decomposition
- Day 9-12: Causal
- Day 13-14: Decision

### With 2 People
**Person A:**
- Day 1-5: SHAP + Active Learning
- Day 6-10: Analysis + writing

**Person B:**
- Day 1-5: Decomposition + Causal
- Day 6-10: Analysis + writing

**Day 11-14:** Joint decision + planning

### With 3 People
**Person A:** SHAP + Active Learning (5 days)
**Person B:** Decomposition (3 days)
**Person C:** Causal (4 days)

**Day 6-8:** Everyone runs analysis in parallel
**Day 9-10:** Joint decision meeting

---

## Key Questions to Answer by Week 3

1. **SHAP Test:**
   - Is FK grouping better than individual features? (Yes/No)
   - By how much? (Quantify stability improvement)

2. **Active Learning Test:**
   - Does FK-guided beat random? (Yes/No)
   - By what margin? (% efficiency gain)
   - Is it NeurIPS-worthy? (>20% gain)

3. **Decomposition Test:**
   - Can we separate epistemic/aleatoric? (Yes/No)
   - How reliably? (Accuracy on synthetic test)
   - Is theory sound? (Makes sense to domain experts)

4. **Causal Test:**
   - Do rankings differ? (Yes/No)
   - Which FK is true root cause? (Domain validation)
   - Is it actionable? (Would practitioner change strategy)

---

## Week 3 Deliverables

### 1. Results Summary (JSON)
```json
{
  "tests_passed": 3,
  "tests_total": 4,
  "recommendation": "Strategic Portfolio (Path 2)",
  "neurips_scope": ["core", "shap", "active_learning", "causal"],
  "workshop_scope": ["decomposition"],
  "probability": {
    "neurips": 0.85,
    "workshop": 0.95,
    "expected_papers": 1.8
  }
}
```

### 2. Strategic Decision
- Which path to take (1, 2, 3, or KDD)
- Paper scope finalized
- Timeline adjusted based on scope

### 3. Updated Battle Plan
- Weeks 3-20 plan updated
- Experiments prioritized
- Writing schedule locked in

---

## Risk Mitigation

### If Falling Behind Schedule

**Day 5 checkpoint:**
- If SHAP + Active Learning not done → Simplify causal test
- If >2 days behind → Drop lowest-priority test (decomposition)

**Day 10 checkpoint:**
- If <2 tests passing → Pivot to KDD immediately
- If 2 tests passing → Proceed with focused NeurIPS paper
- If ≥3 tests passing → Full speed ahead

### If Results Are Ambiguous

**Example:** Active learning shows 15% improvement (between PASS and FAIL)

**Decision:**
- Include in paper but don't emphasize as main contribution
- Position as "application" rather than "novelty"
- Rely on other passing tests for main claims

### If Implementation Fails

**Example:** Can't get SHAP to work with FK grouping

**Fallback:**
- Compare permutation-based FK attribution vs feature-level permutation
- Still valid comparison, just different baseline

---

## Immediate Next Steps (This Week)

### Today (Day 1)
1. Set up conda environment with SHAP library
   ```bash
   conda activate gnn_env
   pip install shap mapie
   ```

2. Test script on dummy data
   ```bash
   python experiments/quick_test_suite.py --test shap
   ```

3. If script works → Run on rel-salt
4. If script fails → Debug and fix

### Tomorrow (Day 2)
1. Analyze SHAP results
2. Generate comparison plot
3. Make PASS/FAIL decision on SHAP
4. Document findings

### Day 3
Start active learning test (regardless of SHAP outcome)

---

## Success Metrics

By end of Week 2, you should have:

✅ JSON file with 4 test results
✅ Clear PASS/FAIL verdict for each direction
✅ Strategic recommendation (which path to take)
✅ Updated scope for NeurIPS paper
✅ Confidence level in acceptance probability

**This de-risks the entire 20-week timeline** - no more guessing what will work!

---

## What If Everything Passes?

**Best case scenario:** All 4 tests PASS

**Then what?**

You have a choice:

**Option A: Unified Framework (Single NeurIPS Paper)**
- All 4 directions in main paper
- 9 pages + appendix
- Very ambitious scope
- Probability: 90% if you finish, 75% overall (risk of not finishing)

**Option B: Strategic Portfolio (Main + Workshops)**
- NeurIPS main: Core + Active Learning + Causal (strongest 2)
- ICML workshop: Epistemic/Aleatoric decomposition
- Both submissions in parallel
- Probability: 85% NeurIPS, 95% workshop = 1.8 expected papers

**My recommendation even if all pass: Option B**

Why? Because:
1. **De-risks NeurIPS** (don't cram everything into one paper)
2. **Tests ideas** (workshop feedback is cheap validation)
3. **Publication momentum** (2 papers in 2026 > 1 paper)
4. **Sets up 2027** (workshop → full paper pipeline)

But if all 4 pass and you're feeling ambitious → Option A is viable too!

---

*Start today: Install SHAP, run first test*
*Decision point: End of Week 2*
*Commitment point: Week 3 (scope locked in)*
