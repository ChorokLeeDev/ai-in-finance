# Neural Causal Discovery Research Loop

## Mission

Build a **Strong Accept** ICAIF paper on "Regime-aware Neural Causal Discovery for Financial Networks"

---

## Current Phase

Check `STATUS.md` for current phase. If it doesn't exist, start with Phase 1.

---

## Phase 1: Literature Review & Gap Analysis

**Goal:** Identify the exact novelty gap we fill.

**Tasks:**
1. Read and summarize key papers:
   - Tank et al. (2021) - Neural Granger Causality
   - Zheng et al. (2018) - NOTEARS
   - Kipf et al. (2018) - Neural Relational Inference
   - Runge et al. (2019) - PCMCI
   - Xu et al. (2021) - Deep Switching State Space Model

2. Create `docs/LITERATURE.md` with:
   - Paper summaries (1 paragraph each)
   - Limitations of each approach
   - Our differentiation

3. Write the **novelty claim** in one sentence.

**Completion:** When LITERATURE.md exists with 5+ papers and clear novelty claim.

---

## Phase 2: Architecture Design

**Goal:** Finalize model architecture with clear innovations.

**Tasks:**
1. Review `code/model.py` - current draft
2. Identify weaknesses in current design
3. Design improvements:
   - Graph structure learning (proper pair-wise)
   - DAG constraint implementation
   - Regime discovery integration
   - Interpretability mechanism

4. Create architecture diagram in `figures/architecture.png` or ASCII

5. Update `code/model.py` with complete implementation

**Completion:** When model.py runs on synthetic data and produces interpretable graphs.

---

## Phase 3: Baseline Implementation

**Goal:** Implement baselines for fair comparison.

**Tasks:**
1. Implement in `code/baselines.py`:
   - Linear Granger causality
   - NOTEARS (or use existing library)
   - Simple LSTM baseline
   - VAR model

2. Create `code/data_loader.py`:
   - Fama-French factor data loading
   - Synthetic data generation (with ground truth)

3. Run baselines on synthetic data, record metrics

**Completion:** When baselines run and produce metrics on synthetic data.

---

## Phase 4: Experiments

**Goal:** Demonstrate our method beats baselines.

**Tasks:**
1. Synthetic experiments:
   - Generate data with known causal structure
   - Compare causal discovery accuracy (F1, precision, recall)
   - Compare regime detection (ARI)

2. Real data experiments:
   - Fama-French 6 factors (1990-2024)
   - Measure prediction improvement
   - Analyze learned causal graphs

3. Crisis analysis:
   - Does causal graph change before 1998, 2008, 2020?
   - Early warning capability?

4. Save all results to `results/`

**Completion:** When we have results showing improvement over baselines.

---

## Phase 5: Paper Writing

**Goal:** Write complete ICAIF paper.

**Tasks:**
1. Create `main.tex` with:
   - Title, Abstract
   - Introduction (problem, contribution)
   - Related Work
   - Methodology
   - Experiments
   - Conclusion

2. Create figures:
   - Architecture diagram
   - Results tables
   - Causal graph visualizations

3. Target: 10 pages, ACM format

**Completion:** When main.tex compiles to PDF.

---

## Phase 6: Review & Iteration

**Goal:** Achieve Strong Accept quality.

**Tasks:**
1. Self-review against ICAIF criteria:
   - Novel ML contribution?
   - Finance application value?
   - Experimental rigor?
   - Presentation quality?

2. Identify weaknesses and fix them

3. Run internal review panel (3 reviewers)

4. Iterate until all reviewers give Accept or Strong Accept

**Completion:** When review panel gives unanimous Accept.

---

## Status Tracking

Update `STATUS.md` after each iteration:

```markdown
# Status

## Current Phase: [1-6]
## Iteration: [N]
## Last Action: [what you did]
## Next Action: [what to do next]
## Blockers: [any issues]
```

---

## Completion Signal

When the paper is ready for submission and review panel gives Strong Accept:

```
<promise>STRONG ACCEPT PAPER READY</promise>
```

---

## Important Notes

- Each iteration should make concrete progress
- Commit changes after each phase
- Don't skip phases
- Focus on ML novelty (this is for ICAIF)
- Be honest about limitations
