# Options Comparison: Novel Research Directions

**Created**: 2025-12-09
**Purpose**: Deep analysis of Option B (Multi-level hierarchy) vs Option C (COVID-19 shrinkage under shift)

---

## Executive Summary

| Option | Novelty Level | Effort | Venue Fit | Recommendation |
|--------|--------------|--------|-----------|----------------|
| Option B: Multi-level hierarchy | LOW | High | Weak | **Not recommended** |
| Option C: COVID-19 shrinkage shift | **MEDIUM-HIGH** | Medium | Good | **Recommended** |

---

## Option B: Multi-level Hierarchy / Graph-based Neighborhoods

### What We Searched For

1. Multi-level hierarchical shrinkage (3+ levels with different shrinkage per level)
2. Graph-based neighborhood definition for empirical Bayes
3. FK structure exploitation in ML
4. RelBench + uncertainty quantification

### What Already Exists

| Topic | Status | Key Reference |
|-------|--------|---------------|
| Multi-level hierarchical Bayes | **COVERED** | Gelman & Hill 2006, standard textbooks |
| Different shrinkage per level | **COVERED** | Global-local shrinkage priors (PMC3658361) |
| Graph-based EB shrinkage | **COVERED** | Stochastic blockmodels EB (PeerJ 2022) |
| GNN + uncertainty | **COVERED** | CF-GNN (NeurIPS 2023), Uncertainty in GNNs Survey |
| Conformal prediction for relational data | **COVERED** | CoRel (2025) |
| Statistical relational learning | **ESTABLISHED FIELD** | Wikipedia has overview |

### Potential Remaining Gaps

1. **RelBench + UQ**: RelBench doesn't explicitly address uncertainty
   - BUT: CF-GNN, PI3NN, and other GNN-UQ methods can be directly applied
   - This is ENGINEERING, not THEORY

2. **FK structure as neighborhood**: Not found specifically
   - BUT: FK structure IS just graph structure
   - Graph-based EB already exists

### Verdict on Option B

**NOT NOVEL ENOUGH FOR THEORY PAPER**

Reasons:
- Multi-level shrinkage is textbook material
- Graph-based EB exists
- GNN uncertainty is well-studied
- What remains is application/engineering work

**If pursued**: Would be application paper showing RelBench + UQ works
**Venue**: KDD, MLSys, not NeurIPS theory

---

## Option C: COVID-19 Shrinkage Under Distribution Shift

### What We Searched For

1. How shrinkage BEHAVIOR changes under distribution shift
2. When borrowing from historical data becomes harmful
3. Temporal shift + hierarchical model interaction
4. COVID-19 as natural experiment for shrinkage

### What Already Exists

| Topic | Status | Key Reference |
|-------|--------|---------------|
| COVID-19 distribution shift in ML | **COVERED** | BMC 2023, domain shift papers |
| Hierarchical Bayes for COVID | **COVERED** | Many spatio-temporal papers |
| Misspecification-robust shrinkage | **RECENT (2025)** | arXiv:2502.03693 (VAR focus) |
| Local EB for heterogeneity | **RECENT (2024)** | arXiv:2511.21282 |
| Individual shrinkage | **RECENT (2023)** | arXiv:2308.01596 |
| Exchangeability violations | **DISCUSSED** | Individual Shrinkage paper |

### What Does NOT Exist (THE GAP!)

**NOBODY HAS STUDIED:**

```
HOW SHRINKAGE BEHAVIOR CHANGES UNDER TEMPORAL DISTRIBUTION SHIFT

Existing work asks:
  "How does model PERFORMANCE change under shift?" ← EVERYONE STUDIES THIS

We would ask:
  "How does SHRINKAGE FACTOR change under shift?"
  "When does borrowing from pre-shift data become HARMFUL?"
  "Can we detect when historical data is 'poison' for shrinkage?"
```

### Why This Is A Real Gap

1. **Local EB (2024)** handles heterogeneity ACROSS experiments
   - Does NOT address TEMPORAL heterogeneity within same system

2. **Individual Shrinkage (2023)** targets individual accuracy
   - Does NOT address what happens when data REGIME changes

3. **Misspecification-robust shrinkage (2025)** focuses on VAR forecasting
   - General misspecification, NOT temporal shift specifically

4. **COVID papers** study epidemiological parameters
   - Do NOT study shrinkage behavior itself

### The Novel Research Question

```
MAIN QUESTION:

When data undergoes temporal distribution shift (e.g., COVID-19 boundary),
how should shrinkage be modified?

SUB-QUESTIONS:

1. Does optimal shrinkage factor change before vs after shift?
   - Hypothesis: Yes, borrowing from pre-shift data becomes less valuable

2. Can we detect when borrowing becomes harmful?
   - Hypothesis: When cross-validation error spikes for shrunk estimates

3. Should we use "temporal neighborhoods" for local EB?
   - Idea: Only borrow from temporally-adjacent data, not all historical

4. What's the tradeoff between data volume and data staleness?
   - More historical data = more shrinkage
   - But stale data under shift = harmful shrinkage
```

### Why SALT Dataset Is Perfect

```
SALT DATASET STRUCTURE:

Pre-COVID:  ... | Feb 2020 | ← Distribution shift boundary
Post-COVID: Jul 2020 | ...

WHAT WE CAN STUDY:

1. Train hierarchical model on pre-COVID data
2. Measure shrinkage factors for each FK/column/value
3. Apply to post-COVID data
4. Compare:
   - Performance with pre-COVID shrinkage
   - Performance with post-COVID only (no shrinkage)
   - Performance with "adaptive" shrinkage

THIS IS A NATURAL EXPERIMENT FOR SHRINKAGE!
```

### Potential Contribution

**Theoretical:**
- Formalize "temporal neighborhood" for shrinkage
- Derive when borrowing from historical data hurts
- Connect to Local EB framework

**Empirical:**
- SALT dataset natural experiment
- Before/after COVID shrinkage analysis
- Practical guidelines for practitioners

**Practical:**
- "Don't blindly apply pre-shift shrinkage to post-shift data"
- Detection criteria for when to stop borrowing

### Venue Fit

| Venue | Fit | Why |
|-------|-----|-----|
| NeurIPS | Medium | Novel angle on shrinkage + distribution shift |
| ICML | Good | Empirical + theoretical mix |
| AISTATS | Good | Bayesian methods focus |
| UAI | Good | Uncertainty focus |

---

## Direct Comparison

| Dimension | Option B (Multi-level) | Option C (COVID shift) |
|-----------|------------------------|------------------------|
| **Literature gap** | Small (application only) | **Real (temporal shrinkage)** |
| **Theoretical novelty** | Low | **Medium-High** |
| **Empirical novelty** | Low | **High (SALT natural experiment)** |
| **Effort required** | High (build system) | **Medium (analysis)** |
| **Risk of being scooped** | High (obvious) | **Low (niche intersection)** |
| **Practical value** | Medium | **High (shift detection)** |
| **Story clarity** | Weak | **Strong (COVID = clear shift)** |

---

## Recommendation

### Primary Recommendation: Option C

**"Shrinkage Under Temporal Distribution Shift: When Does Borrowing From History Hurt?"**

**Why:**
1. Real literature gap at intersection of shrinkage + distribution shift
2. SALT dataset provides natural experiment (COVID boundary)
3. Connects to recent work (Local EB, Individual Shrinkage) but extends temporally
4. Practical value: practitioners need to know when pre-shift models break
5. Clear narrative: COVID is universally understood as major shift

### Proposed Research Plan for Option C

```
PHASE 1: Establish baseline (1-2 weeks)
- Implement hierarchical Bayesian model on SALT
- Measure shrinkage factors pre-COVID
- Document FK/column/value level shrinkage

PHASE 2: Measure shift impact (1-2 weeks)
- Apply pre-COVID shrinkage to post-COVID data
- Compare to no-shrinkage baseline
- Identify where shrinkage HURTS

PHASE 3: Develop detection/adaptation (2-3 weeks)
- Can we detect when shrinkage is harmful?
- Propose "temporal neighborhood" approach
- Compare to Local EB ideas

PHASE 4: Theoretical framing (2-3 weeks)
- Connect to exchangeability violation literature
- Derive conditions for when borrowing hurts
- Position relative to Local EB, Individual Shrinkage

PHASE 5: Paper writing (2-3 weeks)
- Empirical findings + theoretical framing
- Clear practical recommendations
```

### Fallback: Option B

If Option C doesn't work out (e.g., COVID shift too weak in SALT):
- Pivot to RelBench + UQ application paper
- Lower ambition, but achievable
- Target KDD/ICML applications track

---

## Key References for Option C

### Must Read
1. [Local Empirical Bayes (2024)](https://arxiv.org/html/2511.21282) - Neighborhood-based shrinkage
2. [Individual Shrinkage (2023)](https://arxiv.org/html/2308.01596) - Individual vs aggregate accuracy
3. [Misspecification-Robust Shrinkage (2025)](https://arxiv.org/html/2502.03693) - Shrinkage under misspecification
4. [When Past Misleads (2025)](https://arxiv.org/html/2509.01060) - Temporal shift in training data

### Background
5. Efron & Morris (1971) - Tyranny of majority
6. COVID distribution shift papers (PMC10092913, etc.)
7. Exchangeable GPs for epidemics (arXiv:2512.05227)

---

## Next Steps

1. **Read Local EB and Individual Shrinkage papers in detail**
2. **Check SALT dataset COVID shift magnitude**
3. **Run preliminary experiment**: Pre vs post COVID shrinkage
4. **Draft theoretical framing** connecting to existing work

---

*This document recommends Option C as the primary research direction.*
