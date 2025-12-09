# Recommended Research Direction: REVISED

**Created**: 2025-12-09
**Status**: ❌ THEORETICAL CLAIM INVALIDATED
**Target**: KDD 2026 or VLDB 2026 (Applied track)

---

## ⚠️ Critical Update: "FK = Causal" Claim Is Wrong

After deeper literature review, the theoretical claim we considered is **invalid**:

> ~~"In transactional/process domains, FK structure inherently encodes the causal DAG."~~

**See WHY_FK_NOT_CAUSAL.md for full analysis.**

Key reasons:
1. Database schema ≠ Causal model (explicitly stated in literature)
2. Relational Causal Discovery algorithms (RPC, RCD, RelFCI) exist BECAUSE FK ≠ Causal
3. FK encodes referential integrity, not causal direction
4. Propositionalization along FK can introduce spurious associations

---

## Revised Position: Empirical/Applied Contribution

### What We CAN Claim

| Claim | Type | Novelty |
|-------|------|---------|
| FK provides semantically meaningful grouping | Methodological | Medium |
| Attribution-Error validation method | Validation | **Novel** |
| Domain-dependent effectiveness | Empirical | Medium |
| 3-level drill-down framework | Practical | Medium |
| Actionability for practitioners | Applied | Medium |

### What We CANNOT Claim

| Invalid Claim | Why |
|---------------|-----|
| FK = Causal DAG | Contradicted by RCD literature |
| Theoretical guarantees | No formal results |
| Causal attribution | FK attribution ≠ causal attribution |

---

## Why This Is ~~Novel~~ Still Publishable

### What Exists (Related but Different)

| Work | What They Do | How We Differ |
|------|--------------|---------------|
| RCD (Relational Causal Discovery) | **Learn** causal structure from relational data | We **claim** FK IS the structure |
| CARL (SIGMOD 2020) | Extend causal inference to relational data | We use FK for attribution |
| Causal SHAP (Heskes 2020) | Attribution given a known DAG | We claim FK = DAG (no discovery needed) |
| InfoSHAP (NeurIPS 2023) | Feature-level uncertainty attribution | We do FK-group level |

### The Gap

```
Nobody says: "FK structure IS the causal DAG"

They say:    "Learn causal structure from relational data"  (RCD)
They say:    "Do causal inference on relational data"       (CARL)
They say:    "Given a DAG, compute causal attribution"      (Causal SHAP)

We say:      "FK structure = Causal DAG (in transactional domains)"
             "Therefore, FK attribution = Causal attribution"
             "No causal discovery needed"
```

---

## The Theoretical Contribution

### Definition: FK-Causal Correspondence

A relational database has **FK-Causal Correspondence** if:

1. Each FK relationship `A → B` encodes a causal dependency
2. A's attributes are causally upstream of B's prediction-relevant attributes
3. Intervening on FK (changing referenced entity) is equivalent to do(X)

### Theorem (Informal)

Under FK-Causal Correspondence:
```
FK-based permutation importance ≈ Interventional importance

Proof sketch:
- Permuting FK_A = removing information about entity A
- Under correspondence, entity A causally affects outcome
- Therefore, permutation effect = causal effect
```

### Testable Hypothesis

```
H1: Attribution-Error correlation depends on FK-Causal Correspondence

Domain with correspondence (transactional):
  FK structure reflects data generation process
  → Attribution predicts error impact
  → ρ ≈ 0.9

Domain without correspondence (social):
  FK structure is associative, not causal
  → Attribution does NOT predict error impact
  → ρ ≈ -0.5
```

---

## Evidence We Already Have

| Domain | Type | FK-Causal? | Attribution-Error ρ | Supports H1? |
|--------|------|------------|---------------------|--------------|
| SALT (ERP) | Transactional | Yes | **0.90** | ✓ |
| Trial (Clinical) | Process | Yes | **0.94** | ✓ |
| Amazon (E-commerce) | Mixed | Partial | N/A (only 2 FKs) | - |
| Stack (Q&A) | Social | No | **-0.50** | ✓ |

---

## Why This Matters

### 1. No Causal Discovery Needed

```
Traditional approach:
  1. Collect data
  2. Run causal discovery algorithm (RCD, PC, etc.)
  3. Get causal DAG
  4. Do causal attribution

Our approach:
  1. Have relational database with FK structure
  2. Check if domain has FK-Causal Correspondence
  3. If yes: FK attribution = Causal attribution (done!)
```

### 2. Actionability

```
Feature SHAP: "lead_time is important"
  → Which lead_time? From which entity?
  → Not actionable

Causal SHAP: "do(lead_time = x) has effect y"
  → Requires knowing the causal DAG
  → DAG discovery is hard

FK Attribution: "SUPPLIER group is important, specifically Supplier_456"
  → Entity-level, directly actionable
  → No DAG discovery needed (FK = DAG)
```

### 3. Domain Guidance

```
Our framework tells practitioners:

IF your domain is transactional/process-based:
  - ERP systems
  - Clinical trials
  - Manufacturing
  - Supply chain
  → FK attribution is valid and actionable

IF your domain is social/associative:
  - Social networks
  - Q&A platforms
  - Recommendation systems
  → FK attribution may NOT be valid, use with caution
```

---

## Paper Outline

### Title Options

1. "When Foreign Keys Reveal Causality: Interventional Uncertainty Attribution in Relational Databases"
2. "FK-Causal Correspondence: From Database Schema to Causal Attribution"
3. "Uncertainty Attribution Without Causal Discovery: Exploiting Relational Structure"

### Structure

```
1. Introduction
   - Uncertainty attribution is important but not actionable
   - Causal attribution is ideal but requires DAG discovery
   - Our insight: In some domains, FK structure IS the causal DAG

2. Related Work
   - Uncertainty attribution: InfoSHAP, ensemble methods
   - Causal attribution: Causal SHAP, do-calculus
   - Relational causal inference: CARL, RCD, RelFCI
   - Gap: Nobody claims FK = Causal DAG

3. FK-Causal Correspondence
   - Definition
   - When it holds (transactional) vs doesn't (social)
   - Connection to interventional attribution

4. Method
   - FK-based permutation importance
   - 3-level drill-down: FK Group → Entity → Action
   - Hierarchical uncertainty decomposition

5. Experiments
   - 4 domains: SALT, Trial, Amazon, Stack
   - Attribution-Error Validation
   - FK-Causal Correspondence test

6. Results
   - Confirms FK-Causal hypothesis
   - ρ ≈ 0.9 for transactional, ρ ≈ -0.5 for social
   - Actionability demonstrated

7. Discussion
   - When to use FK attribution
   - Limitations and failure modes
   - Connection to causal inference literature
```

---

## What We Need to Do

### Theoretical Work

| Task | Status | Priority |
|------|--------|----------|
| Formalize FK-Causal Correspondence | Not done | **High** |
| Prove connection to do-calculus | Not done | **High** |
| Characterize when correspondence holds | Partial (empirical) | Medium |

### Empirical Work

| Task | Status | Priority |
|------|--------|----------|
| Attribution-Error Validation | Done (4 domains) | - |
| Test FK-Causal match explicitly | Not done | **High** |
| Additional transactional domains | Not done | Medium |
| Additional social domains | Not done | Medium |

### Writing

| Task | Status | Priority |
|------|--------|----------|
| Related work positioning | Not done | High |
| Theoretical section | Not done | **High** |
| Experiments section | Partially done | Medium |

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| "FK = Causal" claim is obvious | Medium | Position as novel formalization |
| Causal inference reviewers reject | Medium | Emphasize practical value |
| Too applied for theory track | High | Submit to applications track |
| Similar work emerges | Low | Unique angle on relational data |

---

## Comparison: V3 Enhanced vs V5

| Aspect | V5 (Abandoned) | V3 Enhanced |
|--------|----------------|-------------|
| Literature coverage | Fully covered | **Novel angle found** |
| Core claim | Negative transfer (known) | FK = Causal DAG (novel) |
| Theoretical depth | None | Medium (with formalization) |
| Empirical evidence | Synthetic only | Real data, 4 domains |
| Actionability | None | **Strong** |
| Publication potential | No | **Yes** |

---

## Next Immediate Steps

1. **Formalize FK-Causal Correspondence** (Definition + sufficient conditions)
2. **Design explicit test** for FK-Causal match in SALT and Trial
3. **Write theoretical section** connecting to do-calculus
4. **Position relative to RCD, CARL, Causal SHAP**

---

*This document recommends the FK-Causal Correspondence direction.*
*The claim "FK structure IS the causal DAG" is novel and testable.*
