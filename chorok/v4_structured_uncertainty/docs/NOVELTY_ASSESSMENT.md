# Novelty Assessment & Research Directions

**Created**: 2025-12-09
**Purpose**: Honest assessment of what's novel and paths to theoretical contribution

---

## Current State: Honest Assessment

### What We Built (Engineering)

1. Hierarchical decomposition: FK → Column → Value
2. Bayesian credible intervals on attribution
3. Intervention effect quantification
4. Multi-domain validation
5. Practical risk manager guide

### What's NOT Novel

| Component | Why Not Novel |
|-----------|---------------|
| Permutation importance | Standard technique (Breiman, 2001) |
| Ensemble uncertainty | Well-established (Lakshminarayanan, 2017) |
| Hierarchical Bayesian models | Textbook material |
| Out-of-distribution detection | Huge existing field |
| Bootstrap/Bayesian CI | Standard statistical inference |

### What MIGHT Be Novel (But Weak)

| Claim | Counter-argument |
|-------|------------------|
| FK structure as prior | Just applying hierarchical Bayes to a new domain |
| Uncertainty over attribution | Bootstrap CI does similar thing |
| Unknown unknowns detection | OOD detection reframed |

### Honest Verdict

**This is an applications/engineering paper, not a theory paper.**

Good fit for: KDD, ICML (applications track), industry venues
Weak fit for: NeurIPS Bayesian ML (main track)

---

## Directions for Theoretical Novelty

### Direction 1: Causal Framework

**Idea**: Formalize intervention effects using causal inference (Pearl's do-calculus)

**What's needed**:
```
Current: "If we collect more data for region X, uncertainty decreases by Y%"
Novel:   "We prove intervention do(X=x) is identifiable from observational data
          under FK structure assumptions, and derive bounds on causal effect"
```

**Key questions**:
- Can we define a causal graph where FK structure implies conditional independencies?
- Is the intervention effect identifiable without randomized experiments?
- What assumptions are needed? (No hidden confounders between FKs?)

**Potential theorem**:
> Under FK structure G and assumptions A1-A3, the intervention effect
> E[U | do(collect_data(FK_i, n))] is identifiable and bounded by [L, U].

**Difficulty**: High
**Novelty**: High
**Relevant literature**: Causal inference, do-calculus, interventional queries

---

### Direction 2: Information-Theoretic Bounds

**Idea**: Derive theoretical bounds on uncertainty reduction from data collection

**What's needed**:
```
Current: "Empirically, 568 samples → 35% reduction"
Novel:   "We prove that collecting n samples in region R reduces epistemic
          uncertainty by at least f(n, |R|, σ²) with probability 1-δ"
```

**Key questions**:
- Can we bound expected uncertainty reduction as function of sample size?
- How does FK structure affect the bound? (Information sharing?)
- Connection to active learning theory?

**Potential theorem**:
> For a hierarchical model with K FKs and n_k samples per FK,
> the posterior variance on FK importance satisfies:
> Var(α_k | data) ≤ σ²_prior / (1 + n_k/n_0) + shrinkage_term(K)

**Difficulty**: Medium-High
**Novelty**: High
**Relevant literature**: PAC-Bayes, active learning, Bayesian experimental design

---

### Direction 3: Novel Inference for FK-Structured Posteriors

**Idea**: Develop specialized inference algorithm exploiting FK structure

**What's needed**:
```
Current: Standard VI with AutoNormal guide (Pyro)
Novel:   "We derive a structured variational family that exploits FK
          conditional independencies, achieving tighter ELBO and faster convergence"
```

**Key questions**:
- Does FK structure imply factorization that standard VI misses?
- Can we derive closed-form updates for some components?
- Message-passing on FK graph?

**Potential contribution**:
> FK-structured mean-field approximation with O(K) instead of O(K²) complexity,
> provably tighter ELBO than standard mean-field

**Difficulty**: Medium
**Novelty**: Medium
**Relevant literature**: Structured VI, graphical models, message passing

---

### Direction 4: Regret Bounds for Data Investment

**Idea**: Frame as sequential decision problem, derive regret bounds

**What's needed**:
```
Current: "Here's which region to collect data for"
Novel:   "We prove our policy achieves O(√T) regret for the data investment
          problem compared to oracle policy that knows true uncertainty sources"
```

**Key questions**:
- Define the "data investment bandit" problem formally
- Each arm = collect data for a specific FK/column/value
- Reward = uncertainty reduction
- Can we derive regret bounds?

**Potential theorem**:
> The hierarchical Thompson sampling policy for FK-structured data investment
> achieves regret R(T) ≤ O(K log(T) √T) where K is number of FKs

**Difficulty**: High
**Novelty**: High
**Relevant literature**: Bandits, Bayesian optimization, active learning

---

### Direction 5: Unknown Unknowns Theory

**Idea**: Formalize the "unknown unknowns" problem theoretically

**What's needed**:
```
Current: "Low uncertainty + low samples = dangerous (empirical observation)"
Novel:   "We define a coverage-adjusted uncertainty metric that provably
          detects unknown unknowns with probability 1-δ"
```

**Key questions**:
- Define "unknown unknown" formally (confident but wrong)
- Derive a test statistic that detects this condition
- Prove coverage guarantees

**Potential theorem**:
> Define adjusted uncertainty U_adj(x) = U(x) / coverage(x).
> Under assumptions A1-A2, if U_adj(x) > τ, then P(prediction wrong) > 1-δ.

**Difficulty**: Medium
**Novelty**: Medium-High
**Relevant literature**: Conformal prediction, selective prediction, calibration

---

## Recommended Path

### For Maximum Novelty: Direction 2 (Information-Theoretic Bounds)

**Why**:
1. Directly addresses "how much data do I need?" - practical AND theoretical
2. Builds on solid Bayesian theory foundation
3. Connects to active learning (established field, can cite)
4. Hierarchical structure gives us something to exploit

**Concrete goal**:
> Theorem: Uncertainty Reduction Bound
>
> For FK k with current samples n_k, collecting Δn additional samples
> reduces posterior variance on importance by:
>
> Var(α_k | n_k + Δn) ≤ Var(α_k | n_k) · (n_k / (n_k + Δn)) + ε(hierarchy)
>
> where ε(hierarchy) is the shrinkage benefit from hierarchical prior.

This gives practitioners a **formula** for ROI on data collection, not just empirical estimates.

---

## Next Steps to Explore Novelty

1. **Literature review**: What bounds exist for Bayesian posterior variance reduction?
2. **Formalize the model**: Write down the exact generative process
3. **Derive simple bound**: Start with single-FK case, then extend to hierarchy
4. **Verify empirically**: Check if bound holds on our experiments

---

*This document will be updated as we explore theoretical directions.*
