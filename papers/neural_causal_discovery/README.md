# Neural Causal Discovery for Financial Networks

**Target:** ICAIF 2026 (Fall) or NeurIPS 2026 Workshop

**Status:** Planning

---

## Core Idea

> Learn time-varying causal graphs in financial networks end-to-end, with regime-awareness and interpretability via attention mechanisms.

---

## Key Innovation

| Existing Work | Our Contribution |
|---------------|------------------|
| Linear Granger (static) | **Nonlinear + Dynamic** |
| Separate HMM + Granger | **End-to-end learning** |
| Pair-wise causality | **Graph-level inference** |
| Black-box neural | **Interpretable via attention** |

---

## Architecture (Draft)

```
Input:  Multivariate factor returns (T × N)
        ↓
[Temporal Encoder] - Transformer/LSTM
        ↓
[Graph Structure Learner] - Edge probability matrix
        ↓
[Causal Attention] - Time-varying edge weights
        ↓
Output: Dynamic causal graph + Regime labels
```

---

## Research Questions

1. Can neural models discover causal structure that linear methods miss?
2. Does the learned causal graph change BEFORE financial crises?
3. Are attention weights interpretable as causal strength?
4. Does regime-awareness improve causal discovery?

---

## Baselines

- Linear Granger causality
- NOTEARS (Zheng et al., 2018)
- Neural Granger (Tank et al., 2021)
- PCMCI (Runge et al., 2019)
- NRI (Kipf et al., 2018)

---

## Datasets

1. **Fama-French factors** (6 factors × 5 markets)
2. **Industry portfolios** (48 industries)
3. **Synthetic** (ground truth causal graph)

---

## Expected Results

| Metric | Baseline | Ours (Target) |
|--------|----------|---------------|
| Causal F1 | 0.45 | 0.65+ |
| Regime ARI | 0.50 | 0.75+ |
| Crisis lead time | 0 days | 14+ days |

---

## Timeline

| Week | Task |
|------|------|
| 1-2 | Literature review + Architecture design |
| 3-4 | Implementation (PyTorch) |
| 5-6 | Baseline experiments |
| 7-8 | Main experiments + Ablations |
| 9-10 | Paper writing |
| 11-12 | Review + Submission |

---

## Key References

- Tank et al. (2021) Neural Granger Causality
- Zheng et al. (2018) DAGs with NOTEARS
- Kipf et al. (2018) Neural Relational Inference
- Runge et al. (2019) PCMCI

---

## Files

```
neural_causal_discovery/
├── code/
│   ├── model.py          # Main architecture
│   ├── baselines.py      # Baseline implementations
│   ├── data_loader.py    # Data preprocessing
│   └── experiments.py    # Training + evaluation
├── data/
├── results/
├── figures/
└── docs/
    └── LITERATURE.md     # Literature review notes
```
