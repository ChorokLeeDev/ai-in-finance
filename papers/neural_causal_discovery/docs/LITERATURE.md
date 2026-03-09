# Literature Review: Neural Causal Discovery

## 1. Tank et al. (2021) - Neural Granger Causality

**Summary:** Tank et al. propose neural network-based methods for Granger causality detection, addressing the limitation that traditional approaches assume linear dynamics. They introduce component-wise MLPs (cMLP) and LSTMs (cLSTM) with group-lasso penalties on input weights to extract causal structure. The method achieves automatic lag selection and demonstrates superior performance on DREAM3 gene regulation data and human motion capture.

**Key Method:**
- Structured MLPs/RNNs with sparsity-inducing penalties
- Group-lasso on input layer weights → causal graph extraction
- Each output variable has separate network (component-wise)

**Limitations:**
- Assumes **stationary** causal structure (no regime switching)
- Requires careful regularization tuning
- Limited to pairwise/direct causality

**Our Differentiation:**
> We extend neural Granger to **regime-conditional** causal discovery, learning time-varying causal graphs that change across latent states.

---

## 2. Zheng et al. (2018) - NOTEARS

**Summary:** NOTEARS reformulates DAG structure learning from a combinatorial search problem to continuous optimization. The key innovation is a smooth, exact characterization of acyclicity: h(W) = tr(e^{W∘W}) - d = 0. This enables standard gradient-based optimization (L-BFGS) for structure learning, dramatically improving scalability.

**Key Method:**
- Continuous acyclicity constraint: h(W) = tr(e^{W∘W}) - d
- Linear SEM: X = W^T X + Z
- L-BFGS optimization with augmented Lagrangian

**Limitations:**
- Assumes **linear** relationships (extensions exist for nonlinear)
- **Static** graph (no temporal dynamics or regime switching)
- Sensitive to initialization and hyperparameters

**Our Differentiation:**
> We incorporate the acyclicity constraint into a **temporal neural architecture** that learns regime-dependent causal graphs, enabling discovery of time-varying causal structure.

---

## 3. Kipf et al. (2018) - Neural Relational Inference (NRI)

**Summary:** NRI is an unsupervised model that learns to infer interactions while simultaneously learning dynamics from observational data. Using a VAE framework, the latent code represents the interaction graph and reconstruction uses graph neural networks. Demonstrated on physics simulations, motion capture, and sports tracking.

**Key Method:**
- Encoder: GNN that infers edge types from trajectories
- Decoder: GNN that predicts dynamics given inferred graph
- VAE objective (ELBO) for end-to-end training
- Discrete edge types via Gumbel-softmax

**Limitations:**
- No explicit **regime/state switching** mechanism
- Assumes **static** interaction graph over time
- Limited to physical systems (no financial applications)

**Our Differentiation:**
> We extend the graph learning paradigm with **explicit regime discovery**, allowing the causal graph to vary across learned latent states—critical for financial data with regime changes.

---

## 4. Runge et al. (2019) - PCMCI

**Summary:** PCMCI is a constraint-based causal discovery method for time series that combines the PC algorithm with momentary conditional independence (MCI) testing. It efficiently handles high-dimensional, autocorrelated data common in climate science and other fields. Implemented in the Tigramite Python package.

**Key Method:**
- PC algorithm for skeleton learning (condition selection)
- MCI test: X_{t-τ} ⊥ Y_t | Parents(Y_t)∖{X_{t-τ}}, Past(X)
- Handles nonlinear dependencies via CMI estimators

**Limitations:**
- **Constraint-based** (discrete accept/reject), not continuous optimization
- Assumes **stationary** causal structure
- No regime switching or latent state modeling
- Computational cost scales with conditioning set size

**Our Differentiation:**
> We replace constraint-based testing with **end-to-end neural optimization**, enabling joint learning of causal structure and regime dynamics in a unified framework.

---

## 5. Xu et al. (2021) - Deep Switching State Space Model (DS³M)

**Summary:** DS³M combines RNNs with nonlinear switching state space models for time series forecasting with regime switching. It uses discrete latent variables (regimes) governed by a Markov chain and continuous latent variables for stochastic factors. Achieves accurate forecasting while identifying interpretable regimes.

**Key Method:**
- Discrete latent: regime variable z_t (Markov chain)
- Continuous latent: state variable s_t (state space)
- RNN-parameterized transition and emission
- Variational inference for training

**Limitations:**
- Focus is **forecasting**, not causal discovery
- No mechanism to extract **causal graph** between variables
- Regimes are learned but not linked to causal structure changes

**Our Differentiation:**
> We extend regime-switching deep learning to **causal discovery**, learning not just regimes but also how the causal graph changes across regimes—the core gap this paper fills.

---

## Summary: Research Gap

| Paper | Causal Discovery | Regime Switching | Neural/Deep | Time Series |
|-------|-----------------|------------------|-------------|-------------|
| Neural Granger | ✅ | ❌ | ✅ | ✅ |
| NOTEARS | ✅ | ❌ | ❌* | ❌ |
| NRI | ✅ | ❌ | ✅ | ✅ |
| PCMCI | ✅ | ❌ | ❌ | ✅ |
| DS³M | ❌ | ✅ | ✅ | ✅ |
| **Ours** | ✅ | ✅ | ✅ | ✅ |

*NOTEARS has neural extensions (NOTEARS-MLP) but not regime-aware.

---

## Our Novelty Claim (One Sentence)

> **"We propose the first neural architecture that jointly learns causal graph structure and latent regime dynamics, enabling discovery of time-varying causal relationships in financial networks."**

---

## Key References

1. Tank, A., Covert, I., Foti, N., Shojaie, A., & Fox, E. (2021). Neural Granger Causality. IEEE TPAMI.
2. Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. (2018). DAGs with NO TEARS. NeurIPS.
3. Kipf, T., Fetaya, E., Wang, K.C., Welling, M., & Zemel, R. (2018). Neural Relational Inference. ICML.
4. Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., & Sejdinovic, D. (2019). Detecting and quantifying causal associations. Science Advances.
5. Xu, X., Peng, H., & Chen, Y. (2021). Deep Switching State Space Model. arXiv:2106.02329.
