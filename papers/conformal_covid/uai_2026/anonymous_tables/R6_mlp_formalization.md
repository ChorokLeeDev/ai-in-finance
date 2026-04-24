# Table R6: MLP Limitation Formalization

**Question**: When does SHAP concentration apply vs. fail?

## Failure Mode Definitions

### Definition 1: Concentrated-Dependence Failure
- SHAP concentration C > 40%
- Dominant feature undergoes distribution shift
- **Diagnostic applies** ✓

### Definition 2: Global-Sensitivity Failure
- SHAP concentration C ≤ 40%
- Coverage degrades via coordinated multi-feature sensitivity
- **Diagnostic does NOT apply** ✗

## Empirical Evidence

### LightGBM
- **ρ = 0.833, p = 0.010** (significant)
- Failure mode: Concentrated-dependence dominates

### MLP
- **ρ = 0.43, p = 0.29** (not significant)
- Failure mode: Global-sensitivity present

## Task-Level Comparison

|     Task      | MLP C | MLP Drop | LGB C | LGB Drop |  MLP Mode  |
|:-------------:|:-----:|:--------:|:-----:|:--------:|:----------:|
| s-group       | 27.5% |  78.4pp  | 47.3% |  71.2pp  | **Global** |
| s-payterms    | 28.0% |  60.6pp  | 54.2% |  77.1pp  | **Global** |
| i-shippoint   | 77.2% |  82.3pp  | 48.8% |  18.5pp  |    Conc    |
| s-incoterms   | 30.3% |  42.3pp  | 23.7% |   8.5pp  |   Mixed    |
| s-shipcond    | 53.1% |  67.1pp  | 50.7% |  71.6pp  |    Conc    |
| i-incoterms   | 31.0% |  24.1pp  | 28.9% |  11.3pp  |   Mixed    |
| i-plant       | 28.2% |  26.7pp  | 23.9% |  10.6pp  |   Mixed    |
| s-office      |  0.0% |   0.1pp  | 42.6% |   0.1pp  |    N/A     |

**Critical**: s-group/s-payterms show MLP C~28% yet 60-78pp drops → global-sensitivity

## Decision Rule

```
IF model ∈ {GBM, XGBoost, LightGBM, CatBoost}:
    IF C > 40%: VULNERABLE
    ELSE: Lower risk

ELIF model ∈ {MLP, NN}:
    IF C > 40%: VULNERABLE (likely)
    ELSE: Use gradient-based methods
```

## Quantitative Criteria

|    Criterion    | Concentrated | Global  |
|:---------------:|:------------:|:-------:|
| Top-1 SHAP      |   C > 40%    | C ≤ 40% |
| Gini coef       |    > 0.6     |  ≤ 0.6  |
| Top-3 cumul     |    > 70%     |  ≤ 70%  |

## Alternative Diagnostics for MLPs (C ≤ 40%)

1. **Gradient sensitivity**: Sⱼ = E[‖∂f/∂xⱼ‖₂]
2. **Feature ablation**: Δⱼ = Cov(D) - Cov(D|xⱼ=μⱼ)
3. **Lipschitz bound**: Cov_test ≥ Cov_val - L·W₂(D_val, D_test)
