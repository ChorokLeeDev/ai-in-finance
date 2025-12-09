# V3 Novelty Enhancement: Options Analysis

**Created**: 2025-12-09
**Goal**: Elevate v3 from "empirical finding" to "theoretical contribution"

---

## Current State

**v3's contribution today:**
- FK structure as grouping for uncertainty attribution (empirical choice)
- Error Propagation hypothesis (empirical observation)
- 3-level drill-down framework (practical contribution)
- Attribution-Error Validation showing ρ=0.9 (empirical validation)

**What's missing for top venue:**
- Theoretical justification for WHY FK grouping works
- Formal conditions under which FK attribution is valid
- Connection to established theory (information theory, causality)

---

## Option A: FK Structure as Causal DAG

### Core Thesis

> In transactional/process domains, FK structure encodes the data generation process (causal DAG). Therefore, FK-based uncertainty attribution equals interventional uncertainty attribution.

### Why This Might Be Novel

1. **Causal SHAP (Heskes 2020)** assumes you KNOW the causal DAG
2. **Our claim**: In relational databases, FK structure IS the causal DAG
3. **Novel insight**: "FK = Causal Structure" hypothesis is domain-dependent

### Theoretical Framework

```
Definition: FK-Causal Correspondence

A relational database has FK-Causal Correspondence if:
  For each FK relationship A → B:
    - A's attributes are causally upstream of B's attributes
    - Changing the entity referenced by FK changes downstream outcomes

Theorem (informal):
  Under FK-Causal Correspondence, FK-based permutation importance
  equals interventional importance (do-calculus).

Corollary:
  In domains WITH FK-Causal Correspondence (ERP, Clinical):
    Attribution-Error correlation is high (ρ ≈ 0.9)
  In domains WITHOUT (Social, Q&A):
    Attribution-Error correlation is low or negative
```

### Evidence We Already Have

| Domain | FK-Causal? | Attribution-Error ρ | Supports Theory? |
|--------|-----------|---------------------|------------------|
| SALT (ERP) | Yes (process flow) | 0.90 | ✓ |
| Trial (Clinical) | Yes (study design) | 0.94 | ✓ |
| Stack (Q&A) | No (associative) | -0.50 | ✓ |

### What We'd Need to Add

1. **Formal definition** of FK-Causal Correspondence
2. **Proof** connecting FK permutation to do-calculus intervention
3. **Empirical test** to verify FK structure matches causal structure
4. **More domains** to test the boundary conditions

### Novelty Assessment

| Aspect | Novelty |
|--------|---------|
| "FK = Causal DAG" claim | **High** - not found in literature |
| Connection to Causal SHAP | Medium - extends existing work |
| Domain categorization | **High** - new empirical finding |

### Risk

- May be "obvious" to causal inference experts
- Need to carefully position relative to Causal SHAP

### Literature Check Result (2025-12-09)

**Existing related work:**
- [Relational Causal Discovery (RCD)](https://github.com/edgeslab/RCD) - LEARNS causal structure from relational data
- [CARL (SIGMOD 2020)](https://netcause.github.io/) - Extends causal inference TO relational data
- [Causal Data Integration](https://dl.acm.org/doi/10.14778/3603581.3603602) - Builds DAGs from relational sources
- [RelFCI](https://arxiv.org/abs/2507.01700) - Causal discovery with latent confounders

**Key distinction:**
```
Existing work: "LEARN causal structure from relational data"
Our claim:     "FK structure IS the causal structure (in transactional domains)"
```

**This is novel!** No one explicitly claims FK = Causal DAG.
- Others discover structure; we claim it's already encoded
- This enables direct use without causal discovery
- The domain-dependence (transactional vs social) is a testable hypothesis

---

## Option B: Information-Theoretic Justification

### Core Thesis

> FK grouping maximizes the mutual information between the group and the entity it represents, making it the optimal grouping for entity-level decisions.

### Theoretical Framework

```
Definition: Entity Mutual Information

For FK group G referencing entity E:
  I(G; E) = mutual information between group features and entity identity

Claim:
  FK grouping achieves I(G; E) = H(E)  (maximal)
  Statistical grouping achieves I(G; E) < H(E)  (sub-optimal)

Intuition:
  - FK group contains ALL features from entity E
  - Statistical group may mix features from multiple entities
  - FK grouping = complete information about the entity
```

### Why This Matters

```
Feature-level attribution: "lead_time is important"
  → Which entity's lead_time? Unclear.
  → I(feature; entity) is low

FK-level attribution: "SUPPLIER group is important"
  → All features from that supplier
  → I(FK_group; entity) is maximal
  → Actionable: change the supplier
```

### What We'd Need to Add

1. **Formal proof** that FK grouping maximizes entity mutual information
2. **Comparison** showing statistical grouping has lower I(G; E)
3. **Connection** to optimal grouping literature

### Novelty Assessment

| Aspect | Novelty |
|--------|---------|
| Information-theoretic framing | Medium - new angle |
| Optimal grouping claim | **High** - new result |
| Connection to actionability | **High** - explains WHY |

### Risk

- May be "just" an application of information theory
- Proof may be straightforward (not deep)

---

## Option C: Hierarchical Uncertainty Decomposition

### Core Thesis

> Decompose total uncertainty into FK-specific components, enabling targeted data collection to reduce uncertainty where it matters most.

### Theoretical Framework

```
Current decomposition:
  Total Uncertainty = Epistemic + Aleatoric

Proposed decomposition:
  Total Uncertainty = Σ_k (Epistemic_k + Aleatoric_k)

  where k ∈ {CUSTOMER, SUPPLIER, PRODUCT, ...}

This gives:
  - Epistemic_SUPPLIER: uncertainty from limited supplier data
  - Aleatoric_SUPPLIER: inherent supplier variability

Actionable:
  - High Epistemic_SUPPLIER → collect more supplier data
  - High Aleatoric_SUPPLIER → supplier is inherently variable
```

### Connection to Existing Work

- **Depeweg (ICML 2018)**: Epistemic/Aleatoric decomposition
- **InfoSHAP (NeurIPS 2023)**: Feature-level uncertainty attribution
- **Our extension**: Epistemic/Aleatoric per FK group

### What We'd Need to Add

1. **Method** to decompose epistemic/aleatoric by FK group
2. **Validation** that the decomposition is meaningful
3. **Actionability study** showing targeted data collection works

### Novelty Assessment

| Aspect | Novelty |
|--------|---------|
| Per-FK uncertainty decomposition | **High** - not found |
| Actionable data collection | **High** - new application |
| Theoretical depth | Medium - extends Depeweg |

### Risk

- May be technically challenging to implement correctly
- Epistemic/aleatoric separation is known to be tricky

---

## Option D: Optimal Grouping Trade-off

### Core Thesis

> Formalize the stability-actionability trade-off and prove that FK grouping is Pareto-optimal under business-relevant constraints.

### Theoretical Framework

```
Define two objectives:

1. Stability(G) = consistency of attribution across data samples
   - Correlation-based grouping: high stability
   - FK grouping: medium stability
   - Random grouping: low stability

2. Actionability(G) = ability to map attribution to business decisions
   - FK grouping: high actionability (entity drill-down)
   - Correlation grouping: low actionability (no semantic meaning)
   - Random grouping: no actionability

Claim:
  FK grouping is Pareto-optimal in the (Stability, Actionability) space

  Proof sketch:
    - No grouping has BOTH higher stability AND higher actionability than FK
    - Correlation grouping: higher stability, lower actionability
    - Random grouping: lower both
```

### Visualization

```
Actionability
     ↑
     │   ★ FK grouping (Pareto-optimal)
     │
     │           ○ Correlation grouping
     │
     │                   × Random
     └─────────────────────→ Stability
```

### What We'd Need to Add

1. **Formal definitions** of Stability and Actionability metrics
2. **Proof** of Pareto-optimality
3. **Empirical validation** across multiple domains

### Novelty Assessment

| Aspect | Novelty |
|--------|---------|
| Trade-off formalization | **High** - new framing |
| Pareto-optimality claim | **High** - new result |
| Practical relevance | **High** - business-relevant |

### Risk

- "Actionability" is hard to formalize rigorously
- May seem like post-hoc justification

---

## Option E: Entity-Level Shapley Values

### Core Thesis

> Compute Shapley values where players are entities (specific customers, suppliers), not features or feature groups.

### Theoretical Framework

```
Standard SHAP:
  Players = features
  Characteristic function v(S) = E[f(X) | X_S]
  Shapley value = feature's contribution to prediction

InfoSHAP:
  Players = features
  Characteristic function v(S) = H(Y | X_S)
  Shapley value = feature's contribution to uncertainty

Entity SHAP (proposed):
  Players = entities (Customer_123, Supplier_456, ...)
  Characteristic function v(S) = H(Y | entities in S are "known")
  Shapley value = entity's contribution to uncertainty

This gives:
  "Customer 123 contributes 5% of total uncertainty"
  "Supplier 456 contributes 12% of total uncertainty"
```

### Why This Is Different

| Method | Players | Output |
|--------|---------|--------|
| Feature SHAP | Features | "lead_time contributes 10%" |
| InfoSHAP | Features | "lead_time contributes 10% of uncertainty" |
| FK Group Attribution | FK groups | "SUPPLIER group contributes 30%" |
| **Entity SHAP** | Entities | "Supplier_456 contributes 12%" |

### What We'd Need to Add

1. **Formalization** of entity-level Shapley game
2. **Efficient algorithm** (many entities = combinatorial explosion)
3. **Validation** that entity Shapley is meaningful

### Novelty Assessment

| Aspect | Novelty |
|--------|---------|
| Entity as Shapley player | **High** - new formulation |
| Computation challenge | Adds depth |
| Actionability | **Very high** - directly names entities |

### Risk

- Computational complexity (2^n entities)
- May need approximation algorithms
- Connection to FK grouping may be unclear

---

## Recommendation: Combined Approach

### Primary: Option A (FK as Causal DAG)

**Why**:
- Provides theoretical foundation
- Explains the Error Propagation hypothesis
- Connects to established causal inference literature
- Already have empirical evidence

### Secondary: Option C (Hierarchical Decomposition)

**Why**:
- Novel methodological contribution
- Highly actionable (targeted data collection)
- Extends Depeweg in new direction

### Supporting: Option D (Pareto-optimality)

**Why**:
- Formalizes the trade-off we already observed
- Justifies FK choice rigorously
- Easy to explain to practitioners

---

## Proposed Paper Structure

```
Title: "Causal Uncertainty Attribution in Relational Databases:
        When Foreign Key Structure Reveals Intervention Targets"

1. Introduction
   - Uncertainty attribution is important but not actionable
   - FK structure provides natural grouping aligned with business decisions
   - Our hypothesis: FK = Causal DAG in transactional domains

2. Related Work
   - InfoSHAP (feature-level uncertainty attribution)
   - Causal SHAP (requires known DAG)
   - Grouped importance (statistical grouping)
   - MPS-GNN (relational explainability, not uncertainty)

3. Theoretical Foundation
   - Definition: FK-Causal Correspondence
   - Theorem: FK attribution = Interventional attribution (under correspondence)
   - Corollary: Domain-dependent validity

4. Method
   - 3-level framework: FK Group → Entity → Action
   - Hierarchical uncertainty decomposition (epistemic/aleatoric per FK)
   - Trade-off analysis: stability vs actionability

5. Experiments
   - 4 domains: SALT, Trial, Amazon, Stack
   - Attribution-Error Validation (key result)
   - FK-Causal Correspondence test
   - Actionability study

6. Results
   - High ρ (0.9) for transactional domains
   - Low ρ (-0.5) for social domains
   - Confirms FK-Causal hypothesis

7. Discussion
   - When to use FK attribution (domain guidance)
   - Limitations and future work
```

---

## Next Steps

1. **Formalize FK-Causal Correspondence** (Definition + Theorem)
2. **Design test for FK-Causal match** (empirical verification)
3. **Implement hierarchical decomposition** (epistemic/aleatoric per FK)
4. **Run additional domains** to test boundaries
5. **Write theoretical sections** connecting to causal inference

---

*This document outlines options for enhancing v3's novelty.*
*Recommended: Option A (Causal) + Option C (Decomposition) + Option D (Trade-off)*
