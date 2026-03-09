# PCMCI Summary: Runge et al. (2019)

**Paper:** "Detecting and quantifying causal associations in large nonlinear time series datasets"
**Authors:** Jakob Runge, Sebastian Bathiany, Erik Bollt, et al.
**Journal:** Science Advances, Vol. 5, No. 11 (2019)

---

## 1. One-Paragraph Summary

PCMCI is a two-phase causal discovery framework designed for high-dimensional time series that addresses fundamental limitations of standard Granger causality. The method first applies the PC algorithm (a constraint-based graphical model approach from Spirtes et al.) to identify a superset of potential causal parents for each variable, dramatically reducing the dimensionality of the conditioning set. It then applies Momentary Conditional Independence (MCI) testing, which tests for conditional independence while controlling for common drivers, indirect links, and autocorrelation—three major confounds that plague naive Granger causality. The framework supports both linear (partial correlation) and nonlinear (kernel-based, mutual information) independence tests, making it applicable to complex Earth system datasets where the authors validated it. PCMCI achieves lower false positive rates and higher detection power than standard methods by avoiding the "curse of dimensionality" problem that occurs when conditioning on all other variables simultaneously.

---

## 2. Key Method: PC Algorithm + Momentary Conditional Independence

### Phase 1: PC Condition-Selection
- For each target variable Y, iteratively identifies a superset of plausible parent variables
- Uses constraint-based search: tests conditional independence at increasing conditioning set sizes
- Removes variables from the candidate parent set when conditional independence cannot be rejected
- Result: A sparse conditioning set S(Y) for each variable, avoiding the need to condition on all variables

### Phase 2: MCI Testing
- Tests: X_{t-τ} ⊥ Y_t | S(Y) ∪ S(X)
- Key innovation: Conditions on parents of BOTH source and target
- This controls for:
  - **Common drivers**: Variables Z that cause both X and Y (spurious correlations)
  - **Indirect effects**: Chains X → Z → Y (separates direct from indirect causation)
  - **Autocorrelation**: Y's own past values (the standard Granger conditioning)

### Test Statistics
- **Linear:** ParCorr (partial correlation)
- **Nonlinear:** CMIknn (conditional mutual information via k-nearest neighbors)
- Flexible: Can use any valid conditional independence test

---

## 3. Limitations

1. **Stationarity Assumption**: Assumes a single, time-invariant causal graph—the same structure governs all observations. Cannot detect regime-dependent causality.

2. **No Hidden Confounders**: Assumes causal sufficiency (no latent common causes). Violations produce false discoveries.

3. **Contemporaneous Links**: Standard PCMCI assumes no instantaneous effects (tau_min ≥ 1). PCMCIplus relaxes this but adds complexity.

4. **Faithfulness Required**: Assumes the data distribution is faithful to the causal graph (no exact cancellations of causal paths).

5. **Computational Scaling**: While more efficient than full conditioning, still scales with the number of variables and candidate parents.

6. **Effect Size Interpretation**: MCI test statistics reflect statistical association strength, not true causal effect magnitudes—they are test statistics, not structural coefficients.

---

## 4. How Our Work Differs

| Aspect | PCMCI (Runge et al. 2019) | Our Framework |
|--------|---------------------------|---------------|
| **Regime Handling** | Single global DAG (stationarity) | Per-regime Granger conditioning on HMM states |
| **Core Innovation** | Better conditioning sets | Latent-state conditional causality |
| **Dynamics** | Static causal structure | Causal structure that changes across regimes |
| **Application** | Earth system (climate) | Financial factor returns |
| **Tail Behavior** | Gaussian/CMI-based | Student-t HMM, quantile Granger, heavy tails |
| **Primary Goal** | Discover time-invariant graph | Predict WHEN causal links decay |

### Key Differentiation Statement (for paper)

PCMCI and related constraint-based methods (NOTEARS, Neural Granger) discover a single causal graph from observational data, assuming stationarity. Our framework addresses the complementary problem of **concept drift in causal structure**: a link X → Y may exist in one latent regime but vanish in another. By conditioning Granger tests on HMM-inferred states, we detect regime-dependent predictability that PCMCI would average away. The August 2007 quant meltdown—where factor relationships inverted overnight—exemplifies why regime-conditional causal discovery matters in finance.

---

## 5. Suggested Citation Context

For the literature review section:

> "Modern causal discovery methods like PCMCI~\cite{runge2019detecting} address the conditioning-set problem through constraint-based search, achieving lower false positives than naive Granger causality. However, PCMCI and related approaches (NOTEARS, neural Granger) assume a time-invariant causal structure—a single directed graph governs all observations. Our framework relaxes this stationarity assumption by conditioning on HMM-inferred latent states, enabling detection of regime-specific causal edges that global methods would average to null."

---

## 6. BibTeX Entry (already in references.bib)

```bibtex
@article{runge2019detecting,
  title={Detecting and quantifying causal associations in large nonlinear time series datasets},
  author={Runge, Jakob and Bathiany, Sebastian and Bollt, Erik and Camps-Valls, Gustau and Coumou, Dim and Deyle, Ethan and Glymour, Clark and Kretschmer, Marlene and Mahecha, Miguel D and Mu{\~n}oz-Mar{\'i}, Jordi and others},
  journal={Science Advances},
  volume={5},
  number={11},
  year={2019}
}
```

---

*Summary prepared for ICAIF 2026 paper literature review integration.*
