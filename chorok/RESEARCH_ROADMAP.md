# Research Roadmap: Causal UQ for Relational Data

**Target Venue**: UAI / NeurIPS / ICML (Main Track)
**Working Title**: "Causal Uncertainty Attribution for Relational Data via Foreign Key Structure"

---

## Final Vision (Conference-Worthy)

**Core Contribution**: A framework that leverages FK relationships as causal structure to provide *interventional* (not just correlational) uncertainty attribution for tabular ML models.

**Why Novel**:
1. FK relationships encode causal DAG (parent → child)
2. Existing UQ methods (SHAP, ensembles) are correlational, not causal
3. Provides identification guarantees specific to relational schema
4. Works with standard tabular models (no need for Bayesian NNs)

---

## Checklist: Baseline to Conference

### Phase 1: Baseline Establishment (Current)
- [x] LOO FK Attribution implementation
- [x] SHAP baseline comparison
- [x] Permutation baseline
- [x] VFA baseline
- [x] Statistical significance analysis (ρ=-0.175, p=0.122)
- [x] COVID timeline case study
- [ ] Document: Current methods are CORRELATIONAL, not causal

### Phase 2: Theoretical Foundation
- [x] Literature review: Causal inference for UQ (see LITERATURE_REVIEW.md)
- [x] Formalize FK → Causal DAG mapping (see THEORY.md §2)
  - FK from table A to B implies A is caused by B's attributes
  - Define notation: G_FK = (V, E) where E derived from FK constraints
- [x] Define interventional UQ query (see THEORY.md §3)
  - do(FK_i = x): What is uncertainty if we intervene on FK relationship?
  - Contrast with LOO (observational: just remove features)
- [x] Identification theorem (see THEORY.md §4)
  - When is causal UQ identifiable from FK structure?
  - What assumptions needed? (no unmeasured confounders within FK path)
- [x] **FORMAL PROOF** (see THEORY.md §4.4) - Rigorous do-calculus proof added
- [x] Prove: LOO ≠ Interventional (theoretically, not just empirically) (see THEORY.md §3.3)

### Phase 3: Method Development
- [x] Algorithm: FK-Causal-UQ (see fk_causal_uq.py)
  - Input: Relational schema, trained model, test point
  - Output: Causal uncertainty attribution per FK
  - Use adjustment formula or do-calculus
- [ ] Estimator with guarantees
  - Finite sample bounds
  - Or: Conformal wrapper for coverage guarantee
- [x] Computational efficiency (see THEORY.md §5.3)
  - Avoid retraining per FK (unlike LOO) ✓
  - Use causal effect estimation techniques

### Phase 4: Experiments
- [x] Synthetic data with known ground truth causal effects
  - We KNOW the true causal UQ, compare methods
  - **Result**: Interventional ρ = 0.964 vs LOO ρ = 0.741 (see synthetic_causal_data.py)
- [x] Semi-synthetic: Inject known shifts into real FK structure ✅ DONE
  - Implementation: semi_synthetic_validation.py
  - **Result: Causal 60% (3/5), LOO 0% (0/5)**
  - Causal method correctly identifies shifted columns, LOO fails completely
- [ ] Real data: SALT (COVID), Stack, + 1-2 more datasets
- [x] Baselines:
  - LOO (our current work) - correlational ✓
  - SHAP - correlational ✓
  - VFA - epistemic but not causal ✓
  - Our method - causal ✓
- [x] Metrics:
  - Causal accuracy (on synthetic) ✓ (ρ = 0.964, p = 0.0005)
  - Calibration
  - Shift detection performance

### Phase 5: Paper Writing
- [ ] Abstract: "We propose... first framework... causal not correlational..."
- [ ] Related work: Causal inference + UQ (gap: relational data)
- [ ] Theory section with theorem
- [ ] Algorithm box
- [ ] Experiments showing causal ≠ correlational matters
- [ ] Limitations: assumptions, computational cost

---

## Key Differentiators from Current Work

| Current (Baseline) | Target (Novel) |
|-------------------|----------------|
| LightGBM + LOO | Any model + Causal estimator |
| Correlational | Interventional |
| No guarantees | Identification theorem |
| Empirical only | Theory + Experiments |
| "SHAP ≠ LOO" (weak) | "Correlational ≠ Causal" (strong) |
| p=0.122 (not sig) | Causal ground truth validation |

---

## What NOT to Do (Stay Focused)

- [ ] Don't add more correlational baselines
- [ ] Don't run more tasks with LOO (diminishing returns)
- [ ] Don't optimize LightGBM hyperparameters
- [ ] Don't chase p-value significance for LOO vs SHAP
- [ ] Don't write paper for current results alone

---

## Next Concrete Steps

1. **Literature review**: Causal inference for UQ (what exists?)
2. **Formalize**: FK → DAG mapping (write math)
3. **Define**: Interventional UQ query formally
4. **Prove**: Identification conditions
5. **Implement**: Causal estimator (not LOO)
6. **Validate**: Synthetic data with known ground truth

---

## Success Criteria

**Minimum for submission**:
- Theorem: Identification of causal UQ from FK structure
- Algorithm: Polynomial time, no retraining
- Experiment: Synthetic shows causal ≠ correlational
- Real data: Demonstrates practical value

**Ideal**:
- Coverage guarantees (conformal)
- Multiple real datasets
- Comparison to Bayesian methods
- Open-source implementation

---

**Current Status**: Phase 4 (Experiments) - 60% complete
**Next Phase**: Phase 5 (Paper) - 0% complete

**Key Result**: Synthetic validation shows interventional method (ρ = 0.964) significantly outperforms LOO (ρ = 0.741) in recovering true causal effects.

---

*Last Updated*: 2025-11-29
