# FK Uncertainty Attribution Research

**Research Question**: Which FK relationship causes model uncertainty?

---

## Current Phase: Baseline Comparison

We developed **Leave-One-Out FK Uncertainty Attribution** and compared it against 3 baselines.

### Key Finding

**Feature Importance ≠ Uncertainty Contribution (ρ = 0.064)**

| Method | What it Measures |
|--------|------------------|
| LOO (Ours) | Which FK causes uncertainty |
| SHAP | Which feature predicts target |
| Permutation | Which FK destabilizes predictions |
| VFA | Which FK causes ensemble disagreement |

---

## Directory Structure

```
chorok/
├── README.md                      # This file
├── SESSION_SUMMARY.md             # Current progress
├── BASELINE_COMPARISON.md         # Detailed findings
│
├── fk_uncertainty_attribution.py  # Our method (Leave-One-Out)
├── shap_attribution.py            # SHAP baseline
├── permutation_attribution.py     # Permutation baseline
├── vfa_attribution.py             # VFA baseline
├── compare_attribution_methods.py # Comparison framework
│
├── results/                       # JSON results
│   ├── fk_uncertainty_attribution.json
│   ├── shap_attribution.json
│   ├── permutation_attribution.json
│   └── vfa_attribution.json
│
├── figures/                       # Plots
│
└── archive/                       # Old research (Phase 1: PSI vs UQ)
    └── phase1_psi_uq/
```

---

## Quick Start

### Run FK Attribution (Our Method)
```bash
python chorok/fk_uncertainty_attribution.py --all_tasks --sample_size 10000
```

### Run All Baselines
```bash
python chorok/shap_attribution.py --all_tasks --sample_size 5000
python chorok/permutation_attribution.py --all_tasks --sample_size 5000
python chorok/vfa_attribution.py --all_tasks --sample_size 5000 --n_models 5
```

### Compare Methods
```bash
python chorok/compare_attribution_methods.py --all_tasks --sample_size 5000
```

---

## Results Summary

| Comparison | Spearman ρ | Top-3 Overlap |
|------------|------------|---------------|
| LOO vs SHAP | 0.064 | 41.7% |
| LOO vs Perm | -0.273 | 25.0% |
| LOO vs VFA | -0.085 | 54.2% |

---

## Next Steps

1. **H&M Dataset** - Validate on second dataset
2. **COVID Timeline** - Monthly attribution changes
3. **Paper Writing** - UAI submission

---

## Archive

Previous research (PSI vs UQ correlation) is in `archive/phase1_psi_uq/`.
That phase concluded: PSI does not correlate with epistemic uncertainty (r=0.425, p=0.294).

---

**Last Updated**: 2025-11-29
