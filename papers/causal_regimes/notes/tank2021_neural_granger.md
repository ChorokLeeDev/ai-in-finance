# Tank et al. (2021) - Neural Granger Causality

**Citation:** Tank, A., Covert, I., Foti, N., Shojaie, A., & Fox, E. B. (2021). Neural Granger Causality. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(8), 4267-4279. arXiv:1802.05842

**Code:** https://github.com/iancovert/Neural-GC

---

## 1. One Paragraph Summary

Tank et al. propose neural network-based methods for nonlinear Granger causality detection, addressing a fundamental limitation of classical VAR-based approaches that assume linear dynamics. Their framework introduces component-wise MLPs (cMLP) and component-wise LSTMs (cLSTM) that model each output series with a separate network, enabling clear interpretation of which input series Granger-cause each output. By applying group lasso penalties on the input layer weights, they enforce sparsity that directly reveals causal structure: if all weights connecting input series j to output series i are zero, then j does not Granger-cause i. The cLSTM variant sidesteps lag selection entirely by leveraging recurrent architecture to capture long-range dependencies automatically. Evaluated on DREAM3 gene expression data (limited time points) and motion capture data, their methods outperform existing nonlinear approaches (OKVAR, kernel methods) and attention-based LSTMs, demonstrating that deep learning can be useful for structure discovery even with limited data.

---

## 2. Key Methods

### cMLP (Component-wise MLP)
- **Architecture:** Separate L-layer MLP for each output series i
- **Input:** Past K lags of ALL p series: x_{(t-1):(t-K)}
- **Output:** Single series prediction x_{ti}
- **Key insight:** First layer weights W^{1k}_j connect lags of series j to hidden units; if W^1_{:j} = 0 for all lags, series j does not Granger-cause series i

### cLSTM (Component-wise LSTM)
- **Architecture:** Separate LSTM for each output series
- **Advantage:** No need to specify maximum lag K; recurrent structure automatically captures long-range dependencies
- **Causality selection:** Group lasso on columns of input weight matrix W^1 (combines forget, input, output, cell gate weights)

### Sparsity-Inducing Penalties (Three Variants)
1. **GROUP:** Full group lasso on all lags: Ω(W^1_{:j}) = ||W^1_{:j}||_F
2. **MIXED (Sparse Group Lasso):** Sparsity across groups AND within groups: α||W^1_{:j}||_F + (1-α)Σ||W^{1k}_{:j}||_2
3. **HIER (Hierarchical):** Selects both causality AND lag order: Σ||(W^{1k}_{:j},...,W^{1K}_{:j})||_F

### Optimization
- Proximal gradient descent with group soft-thresholding
- Line search for step size
- BPTT for cLSTM gradients (truncated for long series)

---

## 3. Limitations

1. **Computational cost:** Requires training p separate networks (one per output series), scaling poorly with dimensionality

2. **Hyperparameter sensitivity:** Group lasso penalty λ requires careful tuning; cross-validation needed in practice

3. **Sufficient but not necessary:** Zero weights are sufficient for non-causality but not necessary—complex weight configurations could cancel out while still representing non-causal relationships

4. **Limited to stationary dynamics:** Assumes time-invariant causal structure; cannot detect regime changes or time-varying causality

5. **No temporal localization:** Identifies WHICH series cause which, but not WHEN the causality is active or how it changes over time

6. **Sample efficiency:** Despite claims of working with limited data, DREAM3 still has 46 replicates × 21 time points = 966 total observations per network

7. **Binary causality detection:** Produces sparse/non-sparse classification rather than quantifying causality strength over time

8. **No structural break detection:** Cannot identify when causal relationships change or decay

---

## 4. How Our Work Differs

| Aspect | Tank et al. (2021) | Our Work |
|--------|-------------------|----------|
| **Goal** | Detect static causal graph structure | Detect temporal evolution of causality |
| **Time assumption** | Time-invariant causality | Time-varying with regime changes |
| **Method** | Neural networks + group lasso | HMM-regularized VAR + structural breaks |
| **Output** | Binary causality matrix | Regime-specific Granger coefficients + breakpoints |
| **Key insight** | Sparsity reveals causality | Decay patterns reveal market evolution |
| **Temporal focus** | Which lags matter | When causality changes |
| **Break detection** | Not addressed | Central contribution (sup-F test) |
| **Interpretability** | Which → Which | Which → Which → When → How fast decay |

### Key Differentiators:

1. **Regime-awareness:** We explicitly model that causal relationships can change across market regimes (bull/bear/crisis), while Tank et al. assume a single static causal structure

2. **Structural break detection:** Our sup-F test identifies WHEN causality breaks down (1998 for HML→SMB), which neural methods cannot detect

3. **Decay dynamics:** We quantify HOW causality decays (half-life = 3.35 years), not just whether it exists

4. **Economic interpretability:** Our regime labels have direct financial interpretation; neural network hidden states do not

5. **Prospective validation:** We test whether detected causality predicts FUTURE relationships (pre-break OOS), while Tank et al. focus on recovering known structure

6. **Computational simplicity:** Our HMM+VAR is interpretable and fast; cMLP/cLSTM require training p separate networks with careful regularization tuning

### Complementary Aspects:

- Tank et al.'s methods could capture nonlinear causality we might miss with linear VAR
- Our regime framework could extend their approach to time-varying settings
- Their hierarchical penalty for lag selection is elegant; we fix lag via AIC

---

## 5. Potential Citation in Literature Review

> Classical Granger causality methods assume linear dynamics, prompting recent work on nonlinear extensions. Tank et al. (2021) propose neural network approaches (cMLP, cLSTM) that use group lasso penalties to identify causal structure in high-dimensional nonlinear systems. However, these methods assume time-invariant causality and cannot detect structural breaks or regime changes—precisely the phenomena we investigate. Our HMM-regularized framework addresses this gap by explicitly modeling time-varying causal relationships and identifying when predictive links decay.

---

## 6. Key Equations (for reference)

**cMLP objective with group lasso:**
$$\min_W \sum_{t=K}^T (x_{ti} - g_i(x_{(t-1):(t-K)}))^2 + \lambda \sum_{j=1}^p ||W^1_{:j}||_F$$

**cLSTM objective:**
$$\min_W \sum_{t=2}^T (x_{ti} - g_i(x_{<t}))^2 + \lambda \sum_{j=1}^p ||W^1_{:j}||_2$$

**Granger non-causality condition:** Series j does not Granger-cause series i if W^1_{:j} = 0

---

*Notes prepared for ICAIF 2026 paper literature review, March 2026*
