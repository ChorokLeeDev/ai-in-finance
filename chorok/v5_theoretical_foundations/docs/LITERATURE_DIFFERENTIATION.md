# Literature Differentiation: Honest Assessment

**Created**: 2025-12-09
**Last Updated**: 2025-12-09
**Purpose**: Honest assessment of what's novel vs what already exists

---

## 0. CRITICAL LITERATURE REVIEW (Updated 2025-12-09)

### What We Initially Claimed Was Novel

| Claim | Reality | Reference |
|-------|---------|-----------|
| Prospective variance reduction formula | **TEXTBOOK** | Every Bayesian stats book |
| Unknown unknowns detection | **DONE BY LAKKARAJU** | AAAI 2017 |
| Hierarchical ROI for data collection | **PARTIALLY EXPLORED** | Active learning, Bayesian exp. design |

### Papers That Already Exist On Our "Novel" Claims

#### 1. Variance Reduction Formula = Textbook

**The formula**: V(n) = 1/(1/τ² + n/σ²) = τ²σ²/(nτ² + σ²)

**Already in**:
- Murphy (2007) - "Conjugate Bayesian analysis of the Gaussian distribution"
- Wikipedia - Conjugate prior
- MIT 18.05 course notes
- Stanford Stats 200 lectures
- Gelman et al. "Bayesian Data Analysis" (the bible)

**Our "reduction formula"** Δn/(n + Δn + λ) is just algebraic rearrangement.

#### 2. Unknown Unknowns Detection = Lakkaraju et al.

**Paper**: "Identifying Unknown Unknowns in the Open World" (AAAI 2017)
- Lakkaraju, Kamar, Caruana, Horvitz

**What they did**:
- Same definition: "confident but wrong"
- Same problem: high-confidence errors
- Model-agnostic methodology
- Explore-exploit for discovery

**Our angle**: We apply to hierarchical Bayesian, they did discriminative models.
But the CONCEPT is theirs.

#### 3. Bayesian Experimental Design / Active Learning

**Existing work**:
- Chaloner & Verdinelli (1995) - "Bayesian Experimental Design: A Review"
- Wang & Gelfand (2002) - Simulation-based Bayesian sample size
- Sahu & Smith (2006) - Sample size determination
- Active learning literature (huge field)

**What they address**:
- "How much data do I need?"
- "Where should I collect data?"
- Uncertainty-guided acquisition

---

## 1. The Key Prior Work

### Gu et al. (2020) - "Borrowing Strength and Borrowing Index for Bayesian Hierarchical Models"

**Citation**: Computational Statistics & Data Analysis, 2020
**Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC7185234/

#### What They Did

| Aspect | Their Approach |
|--------|----------------|
| **Goal** | Measure how much borrowing happened after fitting model |
| **Method** | Distance between hierarchical vs independent posteriors |
| **Formula** | BI = d(F_ind, F_hier) / [d(F_ind, F_hier) + d(F_hier, F_pool)] |
| **Output** | Borrowing Index ∈ [0, 1] |
| **Question** | "How much information was shared across groups?" |

#### Their Key Results

1. **Borrowing Index**: Normalized measure of where hierarchical model falls between independence (0) and complete pooling (1)

2. **Sample Size Effect**: Smaller groups borrow proportionally more than larger groups

3. **Theorem**: Their distance approximation provides a lower bound on true distance

#### What They Explicitly Did NOT Do

From their paper:
> "Does not address **smart borrowing** — distinguishing correct from incorrect information sharing"

They acknowledge this as future work.

### Lakkaraju et al. (AAAI 2017) - "Identifying Unknown Unknowns in the Open World"

**Key contribution**: First algorithmic approach to discovering unknown unknowns

**Their definition** (same as ours):
- Known unknowns: low confidence + wrong
- Unknown unknowns: high confidence + wrong

**Their method**:
- Model-agnostic
- Two-phase: organize by similarity + confidence, then explore-exploit
- Oracle feedback to discover errors

**What they did NOT do**:
- Apply to hierarchical Bayesian models
- Connect to borrowing strength / shrinkage
- Provide closed-form characterization

### Bayesian Experimental Design Literature

**Chaloner & Verdinelli (1995)** - "Bayesian Experimental Design: A Review"
- Expected utility framework
- Information-theoretic criteria
- Optimal design under various objectives

**Wang & Gelfand (2002)** - "Sample size determination for hierarchical models"
- Simulation-based approach
- Performance under posterior model
- No closed-form for complex hierarchies

---

## 2. What Actually Remains Novel?

After honest literature review, here's what MIGHT still be novel:

### Potentially Novel Angle 1: FK Structure Exploitation

**Existing**: General hierarchical models
**Our angle**: Specific FK relational structure in databases

Questions:
- Does FK structure give additional structure beyond generic hierarchy?
- Can we exploit FK cardinality, fan-out patterns?

**Verdict**: Weak novelty. FK structure is just another hierarchy.

### Potentially Novel Angle 2: Combining Borrowing + Unknown Unknowns

**Existing separately**:
- Gu et al.: Borrowing strength measurement
- Lakkaraju et al.: Unknown unknowns detection

**Our angle**: When borrowing makes unknown unknowns WORSE

**Key insight**: Shrinkage increases confidence for sparse groups.
If sparse group is DIFFERENT, shrinkage makes it MORE confidently wrong.

**Verdict**: Possibly novel if we can formalize this connection.

### Potentially Novel Angle 3: Intervention-Aware Borrowing

**Existing**: Static borrowing measurement (Gu et al.)
**Our angle**: How borrowing affects intervention DECISIONS

Questions:
- When should you NOT borrow for intervention planning?
- How does borrowing distort ROI calculations?

**Verdict**: Needs more literature search.

---

## 3. Revised Gap Analysis

### What ACTUALLY Exists (More Than We Thought)

```
VARIANCE REDUCTION
├── Textbook posterior variance formulas (Gelman, Murphy, etc.)
├── Sample size determination (Chaloner, Wang, Sahu)
└── Active learning for data collection (huge field)

UNKNOWN UNKNOWNS
├── Definition and detection (Lakkaraju et al. 2017)
├── Explore-exploit discovery (Lakkaraju et al.)
└── OOD detection literature (separate but related)

BORROWING STRENGTH
├── Retrospective measurement (Gu et al. 2020)
├── Optimal shrinkage factors (Efron-Morris 1975)
└── Hierarchical model theory (Gelman & Hill)
```

### What MIGHT Still Be Missing

```
THE CONNECTION BETWEEN BORROWING AND UNKNOWN UNKNOWNS
├── When does shrinkage CREATE unknown unknowns? [UNEXPLORED]
├── Formal characterization of "dangerous borrowing" [UNEXPLORED]
└── Intervention planning under borrowing distortion [UNEXPLORED]
```

---

## 4. REVISED: What Is Actually Our Contribution?

### 4.1 Honest Assessment

| Our Initial Claim | Reality Check | Actual Status |
|-------------------|---------------|---------------|
| Prospective variance formula | Textbook algebra | **NOT NOVEL** |
| Unknown unknowns detection | Lakkaraju et al. 2017 | **NOT NOVEL** |
| When borrowing is dangerous | **Possibly unexplored** | **NEEDS VERIFICATION** |

### 4.2 The ONE Potentially Novel Angle

**The connection between SHRINKAGE and UNKNOWN UNKNOWNS**

```
EXISTING WORK (SEPARATE):
┌─────────────────────────────────────────────────────────────┐
│  Gu et al. 2020                 Lakkaraju et al. 2017      │
│  ─────────────                  ───────────────────         │
│  "Borrowing happens"            "Unknown unknowns exist"    │
│  "Smaller groups borrow more"   "High confidence + wrong"   │
│  No connection to harm          No connection to shrinkage  │
└─────────────────────────────────────────────────────────────┘

POTENTIAL GAP:
┌─────────────────────────────────────────────────────────────┐
│  SHRINKAGE CREATES UNKNOWN UNKNOWNS                        │
│  ──────────────────────────────────                        │
│  - Sparse group gets shrunk toward global mean             │
│  - Posterior variance DECREASES (looks confident)          │
│  - But if sparse group is DIFFERENT from others...         │
│  - Shrinkage makes it CONFIDENTLY WRONG                    │
│                                                             │
│  This is unknown unknown INDUCED BY hierarchical model!    │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Why This Might Actually Be Novel

**What Lakkaraju did NOT consider**:
- They study discriminative models (classifiers)
- Unknown unknowns arise from training data bias
- No connection to Bayesian hierarchical models
- No connection to shrinkage/borrowing

**What Gu et al. did NOT consider**:
- They measure borrowing, don't evaluate if it's HARMFUL
- They explicitly say "smart borrowing" is future work
- No connection to prediction errors

**The gap**: Nobody has formally characterized when shrinkage CREATES unknown unknowns.

### 4.4 What We Found When We Searched (CRITICAL!)

**We searched and found: THIS IS ALSO NOT NOVEL**

#### The "Tyranny of the Majority" Problem (Efron & Morris 1971!)

**Paper**: Efron & Morris (1971) - Already identified this 50+ years ago!

**Their observation**:
> "The James-Stein estimator shrinks individuals by the same amount regardless
> of their random effect, which can result in LARGE BIAS FOR OUTLIERS—
> individuals with random effects that are far away from the common mean."

**They called it**: "The relevance problem" / "Tyranny of the majority"

**Solutions already proposed**:
- Efron & Morris (1971): Identify and don't shrink outliers
- Efron (2010): Use covariates to identify relevant individuals
- "Limited translation estimators" - cap shrinkage
- "Discarding priors estimators" - use Cauchy-like heavy tails
- "The Clemente Problem" paper (2024) - treat exceptional cases differently

#### Local Empirical Bayes (2024 paper!)

**Paper**: "Local Empirical Bayes shrinkage framework" (arXiv 2511.21282)

**What they do**:
> "Replaces global pooling with small, experiment-specific neighborhoods"
> "Combines process features and outcome similarity to enable localized
> and selective information borrowing under nonstationarity and heterogeneity"

**Their theorem**:
> "Under treatment-effect heterogeneity, local EB achieves strictly lower MSE
> than classical EB. Improvement arises because variance reduction is preserved
> while systematic bias from inappropriate global pooling is avoided."

**This is EXACTLY what we were calling "smart borrowing"!**

#### Individual Shrinkage (2023 paper)

**Paper**: "Individual Shrinkage for Random Effects" (arXiv 2308.01596)

**What they do**:
> "Target individual-level accuracy rather than aggregate performance"
> "Shrinks each individual toward a common mean using individual-specific weights
> derived from that person's own data history alone"

**Directly addresses our "unknown unknowns" concern!**

### 4.5 REVISED: What Is Actually Left?

After thorough search, here's the honest assessment:

| Our Claim | Existing Work | Status |
|-----------|---------------|--------|
| Variance reduction formula | Textbook | **NOT NOVEL** |
| Unknown unknowns detection | Lakkaraju 2017 | **NOT NOVEL** |
| Shrinkage harms outliers | Efron & Morris 1971 | **NOT NOVEL** |
| Local/selective borrowing | Local EB 2024 | **NOT NOVEL** |
| Individual accuracy | Individual Shrinkage 2023 | **NOT NOVEL** |

**Potentially remaining**:
1. Application to FK/relational structure specifically
2. Connection to intervention/data collection decisions
3. Empirical validation in relational learning domain

**But these are APPLICATION papers, not THEORY papers.**

### 4.6 Honest Path Forward

Given the literature review, we have three options:

#### Option A: Pivot to Application Paper

**Venue**: KDD, ICML Applications Track, or domain venue
**Contribution**: Apply existing theory to relational learning
**Message**: "We show how Local EB / Individual Shrinkage ideas apply to FK-structured data"

**Pros**: Honest, achievable, useful
**Cons**: Not NeurIPS theory track

#### Option B: Find Genuinely Novel Theory Angle

**Potential angles NOT covered by existing work**:

1. **FK structure as graph** - Can we exploit graph structure of FK relationships?
   - Existing: General hierarchical models
   - Novel?: Graph-based neighborhood for local EB

2. **Temporal aspects** - FK relationships change over time
   - Existing: Local EB handles nonstationarity within experiments
   - Novel?: Cross-FK temporal borrowing

3. **Multi-level hierarchy** - FK → Column → Value
   - Existing: 2-level hierarchical models
   - Novel?: 3+ level borrowing with different shrinkage at each level?

**Needs more literature search on these specific angles.**

#### Option C: Combination Paper

**Contribution**: Novel APPLICATION + existing theory
**Format**:
- Theory section: Properly cite Efron-Morris, Local EB, etc.
- Novel part: How to apply to relational data + empirical validation
- "We bring these ideas to relational learning for the first time"

### 4.7 Key References We Must Cite

If we proceed, we MUST cite:

1. **Efron & Morris (1971, 1973, 1975)** - Tyranny of majority, shrinkage
2. **Lakkaraju et al. (2017)** - Unknown unknowns detection
3. **Gu et al. (2020)** - Borrowing strength index
4. **Individual Shrinkage (2023)** - arXiv 2308.01596
5. **Local Empirical Bayes (2024)** - arXiv 2511.21282
6. **The Clemente Problem (2024)** - Treating exceptional cases
7. **Chaloner & Verdinelli (1995)** - Bayesian experimental design
8. **Gelman et al.** - Bayesian Data Analysis

---

## 5. REVISED Positioning Statement

### What We CAN'T Claim (Based on Literature Review)

❌ "First to identify unknown unknowns" - Lakkaraju et al. 2017
❌ "First to recognize shrinkage harms outliers" - Efron & Morris 1971
❌ "First to propose local/selective borrowing" - Local EB 2024
❌ "Novel variance reduction formula" - Textbook

### What We MIGHT Be Able To Claim

**Option A (Application)**:
> "We are the first to apply Local Empirical Bayes and selective borrowing
> concepts to relational learning with FK-structured data, demonstrating
> that FK hierarchies naturally define neighborhoods for local shrinkage."

**Option B (If Novel Theory Found)**:
> "We extend Local EB to multi-level FK hierarchies, showing that optimal
> shrinkage at each level (FK → Column → Value) differs and provides
> tighter bounds than single-level approaches."

### Honest One-Sentence Summary

> **What we do**: Apply existing theory (Efron-Morris, Local EB, Individual Shrinkage)
> to the novel domain of FK-structured relational data.
>
> **What we don't do**: Invent new theory.

---

## 6. UPDATED Literature Landscape

### Complete Map of Existing Work

```
SHRINKAGE & OUTLIERS (1970s onward)
│
├── Efron & Morris (1971, 1973, 1975)
│   ├── James-Stein shrinkage
│   ├── "Tyranny of the majority" problem identified
│   └── Limited translation estimators
│
├── The Clemente Problem (2024)
│   └── Heavy-tailed priors for exceptional cases
│
└── Individual Shrinkage (2023)
    └── Individual-level accuracy vs aggregate MSE

UNKNOWN UNKNOWNS (2017)
│
└── Lakkaraju et al. (AAAI 2017)
    ├── Definition: confident but wrong
    ├── Explore-exploit discovery
    └── Model-agnostic framework

LOCAL/SELECTIVE BORROWING (2024)
│
└── Local Empirical Bayes (2024)
    ├── Neighborhood-based shrinkage
    ├── Exchangeability violation handling
    └── Strictly lower MSE than global EB

BORROWING MEASUREMENT (2020)
│
└── Gu et al. (2020)
    ├── Borrowing Index
    └── Retrospective measurement

BAYESIAN EXPERIMENTAL DESIGN (1990s onward)
│
├── Chaloner & Verdinelli (1995) - Review
├── Wang & Gelfand (2002) - Simulation-based
└── Active learning literature (huge)

TEXTBOOK RESULTS
│
├── Posterior variance formula
├── Conjugate normal-normal
└── Sample size determination basics
```

### Where Is The Gap?

```
POTENTIAL REMAINING GAPS (need verification):
│
├── FK-specific structure?
│   └── Graph structure of FK relationships
│   └── Multi-level hierarchy (FK→Column→Value)
│
├── Relational learning application?
│   └── First application of Local EB to relational data?
│
└── COVID-19 distribution shift?
    └── Empirical study of shrinkage under temporal shift?
```

---

## 7. Recommended Next Steps

### Immediate Actions

1. **Stop claiming theoretical novelty** on variance formula, unknown unknowns, or outlier handling

2. **Read the key papers thoroughly**:
   - Efron & Morris (1971) - Original tyranny of majority
   - Local EB (2024) - arXiv 2511.21282
   - Individual Shrinkage (2023) - arXiv 2308.01596
   - Lakkaraju et al. (2017) - Unknown unknowns

3. **Search for remaining gaps**:
   - Multi-level hierarchical shrinkage
   - Graph-based local neighborhoods
   - FK-specific structure exploitation

4. **Decide on paper type**:
   - Application paper (honest, achievable)
   - Theory paper (needs genuinely novel angle)

### Questions for User

1. Do you want to pursue NeurIPS theory track (risky) or application venue (safer)?
2. Should we search deeper for multi-level hierarchy novelty?
3. Should we pivot to COVID-19 shift study as empirical contribution?

---

## 8. Summary

### What We Learned From Literature Review

| Topic | Status | Key Paper |
|-------|--------|-----------|
| Variance reduction | Textbook | Gelman BDA |
| Unknown unknowns | DONE | Lakkaraju 2017 |
| Outlier shrinkage | DONE (1971!) | Efron-Morris |
| Local borrowing | DONE (2024) | Local EB |
| Individual accuracy | DONE (2023) | Individual Shrinkage |
| Borrowing measurement | DONE | Gu 2020 |

### The Honest Verdict

**Our initial claims were not novel.**

The good news: We now know the literature and can:
1. Position correctly
2. Cite properly
3. Find actual gaps if they exist

---

*This document updated 2025-12-09 after thorough literature search.*
*Previous claims of novelty have been revised based on findings.*

## References

### Must Read (Critical for positioning)
1. Efron & Morris (1971) - "Limiting the Risk of Bayes and Empirical Bayes Estimators"
2. Efron & Morris (1973) - "Stein's Estimation Rule and Its Competitors—An Empirical Bayes Approach"
3. Lakkaraju et al. (2017) - "Identifying Unknown Unknowns in the Open World" (AAAI)
4. Local EB (2024) - arXiv:2511.21282
5. Individual Shrinkage (2023) - arXiv:2308.01596
6. Gu et al. (2020) - "Borrowing Strength and Borrowing Index"
7. The Clemente Problem (2024) - arXiv:2506.10114

### Background
8. Gelman et al. - Bayesian Data Analysis
9. Gelman & Hill (2006) - Data Analysis Using Multilevel Models
10. Chaloner & Verdinelli (1995) - Bayesian Experimental Design Review
