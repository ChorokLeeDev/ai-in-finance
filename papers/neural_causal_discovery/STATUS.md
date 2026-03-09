# Status

## Current Phase: 3 → 4 (Transitioning)
## Iteration: 11
## Last Action: Completed Phase 3 - Baseline Implementation
## Next Action: Start Phase 4 - Full Experiments
## Blockers: None

### Phase 1 Completion ✅
- [x] Literature review (5 papers)
- [x] Novelty claim defined

### Phase 2 Completion ✅
- [x] Factor Encoder (per-factor LSTM with attention pooling)
- [x] Regime Encoder (Transformer + regime classifier)
- [x] Graph Structure Learner (regime-conditional pair-wise MLP)
- [x] DAG Constraint (NOTEARS-style tr(e^{W∘W}) - d)
- [x] Causal Predictor (graph-masked prediction)
- [x] Combined RANCD model with 4 loss terms
- [x] Model test passed ✅

### Phase 3 Completion ✅
- [x] Implement Linear Granger Causality (baselines.py)
- [x] Implement NOTEARS (baselines.py)
- [x] Implement VAR Model (baselines.py)
- [x] Implement SimpleLSTM baseline (baselines.py)
- [x] Create data_loader.py with:
  - [x] SyntheticCausalData (regime-switching)
  - [x] FamaFrenchLoader
  - [x] TimeSeriesDataset + DataLoader
- [x] Quick baseline tests passed:
  - Granger F1: 0.588
  - VAR F1: 0.667

### Phase 4 TODO
- [ ] Full synthetic experiments (5 trials)
- [ ] Regime detection experiments (ARI)
- [ ] Fama-French real data experiments
- [ ] Crisis analysis
- [ ] Save all results

### Architecture Summary
```
RANCD: Regime-Aware Neural Causal Discovery
├── FactorEncoder: Per-factor LSTM embeddings
├── RegimeEncoder: Transformer → regime probabilities
├── GraphStructureLearner: Regime-conditioned edge prediction
├── DAGConstraint: NOTEARS acyclicity
└── CausalPredictor: Graph-masked Granger prediction

Baselines:
├── LinearGrangerCausality: VAR + F-test
├── NOTEARS: Continuous DAG learning
├── VARModel: Vector autoregression
└── SimpleLSTM: Sequence prediction (no structure)
```
