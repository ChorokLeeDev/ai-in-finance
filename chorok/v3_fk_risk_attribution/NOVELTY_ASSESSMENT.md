# V3 FK Risk Attribution: Novelty Assessment

**Created**: 2025-12-09
**Purpose**: Honest assessment of v3's novelty after thorough literature search

---

## Executive Summary

| Aspect | Assessment |
|--------|------------|
| Is v3 novel? | **Partially yes** - the specific combination is new |
| Type of novelty | **Empirical/Applied**, not theoretical |
| Publication potential | **Yes**, but venue matters |
| Compared to v5 | **Much stronger** - v5 was completely covered |

---

## What Already Exists (Related Work)

### 1. InfoSHAP (NeurIPS 2023)
- **Paper**: [Explaining Predictive Uncertainty with Information Theoretic Shapley Values](https://arxiv.org/abs/2306.05724)
- **What it does**: Feature-level uncertainty attribution via Shapley values
- **How v3 differs**: v3 operates at FK group level, not feature level

### 2. Grouped Permutation Importance (Springer 2022)
- **Paper**: [Grouped feature importance and combined features effect plot](https://link.springer.com/article/10.1007/s10618-022-00840-5)
- **What it does**: Group features for importance calculation
- **How v3 differs**: v3 uses FK structure (semantic) vs statistical grouping

### 3. MPS-GNN (arXiv 2024)
- **Paper**: [A Self-Explainable Heterogeneous GNN for Relational Deep Learning](https://arxiv.org/abs/2412.00521)
- **What it does**: Meta-path level explainability for GNN on relational data
- **How v3 differs**: v3 focuses on uncertainty, not just prediction explanation

### 4. Concept-based Uncertainty (arXiv 2025)
- **Paper**: [Conceptualizing Uncertainty](https://arxiv.org/abs/2503.03443)
- **What it does**: Concept-level uncertainty explanation, actionable interventions
- **How v3 differs**: v3 uses FK structure (given), not learned concepts

### 5. GNNExplainer
- **What it does**: Subgraph-level explanation for GNN predictions
- **How v3 differs**: v3 focuses on uncertainty attribution, not prediction explanation

---

## V3's Claims: Novelty Check

| Claim | Found in Literature? | Novelty |
|-------|---------------------|---------|
| FK structure as grouping for UQ | Not found | **Novel** |
| 3-level drill-down (FK → Entity → Action) | Not found | **Novel** |
| Error Propagation hypothesis | Not found | **Novel** |
| Attribution-Error Validation (ρ=0.9) | Not found | **Novel method** |
| Works for transactional, not social | Not found | **Novel finding** |

---

## Type of Contribution

| Contribution Type | V3's Strength | Notes |
|-------------------|---------------|-------|
| **Theoretical** | Weak | No new bounds, no proofs |
| **Algorithmic** | Weak | Uses existing permutation importance |
| **Methodological** | Medium | FK as grouping is a design choice |
| **Empirical** | Strong | Error Propagation hypothesis, validation |
| **Practical** | Strong | Actionable recommendations |

---

## Venue Fit

| Venue | Fit | Rationale |
|-------|-----|-----------|
| NeurIPS Main (Theory) | Weak | Insufficient theoretical depth |
| NeurIPS Main (Applications) | Medium | Novel application angle |
| NeurIPS Datasets & Benchmarks | Good | Framework + validation |
| ICML | Good | Empirical ML contribution |
| KDD | Strong | Applied, business-relevant |
| VLDB/SIGMOD | Good | Relational database angle |
| AISTATS | Medium | Uncertainty focus |

---

## Comparison: V3 vs V5

| Aspect | V5 (Shrinkage under shift) | V3 (FK Attribution) |
|--------|---------------------------|---------------------|
| Literature coverage | **Fully covered** (Bayesian transfer learning) | **Partially novel** |
| Core claim | Rediscovery of negative transfer | New combination of existing ideas |
| Theoretical contribution | None (known since 2009) | Weak but present |
| Empirical contribution | Confirms known result | Novel findings |
| Publishable? | No | Yes (with caveats) |

---

## Weaknesses to Address

### 1. "Just Engineering"
**Criticism**: Using FK structure is just a design choice, not a scientific contribution.
**Counter**: The Error Propagation hypothesis explains WHEN this choice works.

### 2. "Incremental"
**Criticism**: Grouped importance exists; FK is just one type of grouping.
**Counter**: FK provides semantic meaning that statistical grouping lacks.

### 3. "Not Theoretical"
**Criticism**: No proofs, no bounds, no guarantees.
**Counter**: Need to add theoretical foundation (see next section).

---

## Key References

1. InfoSHAP - https://arxiv.org/abs/2306.05724
2. Grouped Permutation Importance - https://link.springer.com/article/10.1007/s10618-022-00840-5
3. MPS-GNN - https://arxiv.org/abs/2412.00521
4. Concept-based Uncertainty - https://arxiv.org/abs/2503.03443
5. RelBench - https://relbench.stanford.edu/paper.pdf

---

*This document assesses v3's novelty after thorough literature review.*
*Conclusion: v3 is more novel than v5, but needs theoretical strengthening.*
