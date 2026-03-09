# NOTEARS Paper Summary

**Citation:** Zheng, X., Aragam, B., Ravikumar, P. K., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous Optimization for Structure Learning. *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*.

## 1. One-Paragraph Summary

NOTEARS reformulates directed acyclic graph (DAG) structure learning from a combinatorial search problem into a purely continuous optimization problem over real matrices. The key innovation is a smooth, exact characterization of acyclicity via the constraint h(W) = tr(exp(W ∘ W)) - d = 0, where W is the weighted adjacency matrix and d is the number of nodes. This constraint equals zero if and only if the graph is acyclic, enabling the use of standard numerical algorithms (L-BFGS with augmented Lagrangian) instead of specialized combinatorial solvers. The method assumes a linear structural equation model X = W^T X + z and optimizes a least-squares loss with optional L1 regularization for sparsity. Experiments on synthetic data (Erdos-Renyi and scale-free graphs) show NOTEARS outperforms greedy equivalence search (GES) and PC algorithm in terms of structural Hamming distance, particularly as graph density increases.

## 2. Key Innovation: Continuous Acyclicity Constraint

The central contribution is **Theorem 1**:

> A matrix W ∈ R^{d×d} represents a DAG if and only if h(W) = tr(exp(W ∘ W)) - d = 0

**Why this works:**
- Let S = W ∘ W (element-wise square, so S ≥ 0)
- The trace of S^k counts weighted k-cycles in the graph
- tr(exp(S)) = tr(I) + tr(S) + tr(S²)/2! + ... ≥ d
- Equality holds iff there are no cycles of any length

**Properties:**
- (a) Exact: h(W) = 0 ⟺ W is a DAG
- (b) Smooth: h is infinitely differentiable with gradient ∇h(W) = (exp(W∘W))^T ∘ 2W
- (c) Computable: Matrix exponential has O(d³) complexity
- (d) Quantifies "DAG-ness": h(W) > 0 measures how far W is from being acyclic

**Optimization:**
- Augmented Lagrangian method: L^ρ(W, α) = F(W) + ρ/2|h(W)|² + αh(W)
- Solved via L-BFGS with ~10 outer iterations
- Final thresholding removes small weights

## 3. Limitations

### Acknowledged by Authors:
1. **Nonconvex optimization**: The equality-constrained program is nonconvex; can only guarantee stationary points, not global optima (though empirically close to global)
2. **Computational complexity**: O(d³) per iteration due to matrix exponential; becomes expensive for large graphs
3. **Linear SEM assumption**: Core method assumes linear relationships X = W^T X + z (nonlinear extensions exist but less principled)
4. **Fixed thresholding**: Uses fixed ω > 0 for post-processing; data-adaptive thresholding would be preferable
5. **No identifiability guarantees**: Multiple SEMs can have same likelihood; parameter identifiability requires additional assumptions

### Known from Follow-up Literature:
6. **Equal noise variance**: Original formulation does not handle heteroscedastic noise well
7. **Scalability**: Struggles with d > 1000 nodes in practice
8. **Local minima**: While rare in experiments, pathological cases exist
9. **No latent confounders**: Assumes all relevant variables are observed
10. **Sensitivity to hyperparameters**: Performance depends on L1 regularization strength λ

## 4. How Our Work Differs

| Aspect | NOTEARS | Our Work |
|--------|---------|----------|
| **Goal** | Learn full DAG structure from scratch | Test specific causal hypotheses (factor→factor) |
| **Approach** | Continuous optimization over all edges | Regime-switching Granger causality |
| **Acyclicity** | Enforced via h(W) constraint | Not needed (testing pairwise relationships) |
| **Time dynamics** | Static structure | Explicit temporal dynamics via VAR/HMM |
| **Regime changes** | Single static structure | Allows time-varying causal relationships |
| **Interpretability** | Discovers structure | Tests hypothesized relationships |
| **Application** | General causal discovery | Financial factor predictability decay |

**Key distinctions for literature review:**

1. **Different problem scope**: NOTEARS discovers the entire causal graph; we test whether *specific* predictive relationships exist and how they evolve over time.

2. **Temporal structure**: NOTEARS learns a single static DAG; our HMM-based approach explicitly models regime-switching dynamics where causal relationships can appear and disappear.

3. **Hypothesis-driven vs. discovery**: NOTEARS is exploratory (discovers unknown structure); our approach is confirmatory (tests whether HML→SMB predictability exists in a given regime).

4. **No acyclicity needed**: Since we test pairwise Granger causality within a VAR framework, acyclicity constraints are irrelevant to our setting.

5. **Complementary approaches**: NOTEARS could be applied as a preprocessing step to suggest which factor pairs to test, but our regime-switching framework captures temporal dynamics that static methods miss.

## Suggested Citation for Paper

For the literature review, position as follows:

> "Recent advances in causal discovery include continuous optimization approaches that avoid combinatorial search [Zheng et al., 2018]. However, these methods learn static structures and do not capture the time-varying nature of financial relationships. Our regime-switching framework explicitly models how causal relationships decay over time, addressing a limitation of static causal discovery methods."
