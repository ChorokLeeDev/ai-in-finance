```
+------------------+     +------------------+     +-------------------+
|  Input Series    |     |  Factor Encoder  |     |  Regime Encoder   |
|  X ∈ R^(T×n)     | --> |  (Per-factor     | --> |  (Transformer)    |
|                  |     |   LSTM)          |     |                   |
+------------------+     +------------------+     +-------------------+
                               |                         |
                               v                         v
                        +-------------+           +-------------+
                        | Factor      |           | Regime      |
                        | Embeddings  |           | Probs r_t   |
                        | e_i         |           |             |
                        +-------------+           +-------------+
                               |                         |
                               +------------+------------+
                                            |
                                            v
                            +-------------------------------+
                            |  Graph Structure Learner      |
                            |  A_ij = σ(MLP([e_i; e_j; r])) |
                            +-------------------------------+
                                            |
                                            v
                            +-------------------------------+
                            |  DAG Constraint               |
                            |  h(A) = tr(e^{A∘A}) - n = 0   |
                            +-------------------------------+
                                            |
                                            v
                            +-------------------------------+
                            |  Causal Predictor             |
                            |  x̂_{t+1} = f(A ⊙ x_{t-L:t})  |
                            +-------------------------------+
                                            |
                                            v
                            +-------------------------------+
                            |  Loss: L_pred + L_DAG +       |
                            |        L_sparse + L_regime    |
                            +-------------------------------+
```

## RANCD Architecture Summary

1. **Factor Encoder**: Independent LSTM per factor with temporal attention pooling
2. **Regime Encoder**: Transformer encoder → softmax regime probabilities
3. **Graph Structure Learner**: MLP predicting edge probability for each (i,j) pair
4. **DAG Constraint**: NOTEARS acyclicity loss
5. **Causal Predictor**: Graph-masked Granger-style prediction

## Key Innovation

The graph structure learner receives regime information as input, enabling
regime-conditional edge prediction:

```
A_ij = σ(MLP([e_i; e_j; r̄]))
```

where r̄ is the time-averaged regime distribution.

This allows the model to learn different causal structures for different
market regimes (e.g., crisis vs normal periods).
