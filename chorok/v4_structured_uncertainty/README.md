# v4: Hierarchical Bayesian Intervention Analysis

**Goal**: Structured uncertainty decomposition with actionable recommendations

## Core Idea

```
현재 (v3):  "PLANT 테이블이 37% 기여"
목표 (v4):  "PLANT 테이블 품질 개선 시 불확실성 -12% [95% CI: -15%, -9%]"
```

Not just "where is the problem" but "what to change by how much, with confidence intervals"

## Why Bayesian ML?

1. **Hierarchical Priors**: FK → Column → Value 구조를 자연스럽게 인코딩
2. **Credible Intervals**: Point estimate가 아닌 분포로 intervention effect 제공
3. **Posterior over Interventions**: "이걸 바꾸면 어떻게 될까"의 불확실성까지 정량화

## Directory Structure

```
v4_structured_uncertainty/
├── README.md                    # This file
├── docs/
│   ├── HIERARCHICAL_BAYESIAN_INTERVENTION.md  # Main design doc
│   └── bayesian_dl_onepager_kr.md             # Korean summary
├── methods/
│   ├── ensemble_lgbm.py         # LightGBM ensemble (baseline)
│   ├── mc_dropout.py            # MC Dropout ensemble
│   └── bnn_pyro.py              # Real BNN with Pyro (experimental)
├── experiments/
│   ├── feasibility_bnn.py       # BNN feasibility test
│   ├── feasibility_real_bnn.py  # Real BNN comparison
│   └── debug_real_bnn.py        # BNN debugging
└── results/
    └── ...
```

## Key Difference from v3

| Aspect | v3 (FK Attribution) | v4 (Hierarchical Bayesian) |
|--------|---------------------|---------------------------|
| Output | "PLANT contributes 37%" | "Improve PLANT → -12% uncertainty [CI: -15%, -9%]" |
| Level | Table only | Table → Column → Value Range |
| Uncertainty | None | Credible intervals on effects |
| Actionability | "Which table" | "What to change by how much" |

## Target Venue

NeurIPS 2026 Bayesian ML Track

## Status

- [x] Design document complete
- [ ] Core intervention framework
- [ ] Bayesian effect estimation
- [ ] Multi-domain validation
- [ ] Paper draft

---

*Created: 2025-12-09*
