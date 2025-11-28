# Causal UQ for Relational Data

**Research Question**: Can we use FK structure for *causal* (not correlational) uncertainty attribution?

---

## Current Phase: Baseline → Causal Framework

See **[RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md)** for the full plan.

### Status
- **Phase 1 (Baseline)**: 90% - LOO, SHAP, Permutation, VFA
- **Phase 2 (Theory)**: 0% - FK→DAG, Interventional UQ, Identification
- **Phase 3 (Method)**: 0% - Causal UQ algorithm
- **Phase 4 (Experiments)**: 0% - Synthetic + Real validation
- **Phase 5 (Paper)**: 0%

We developed **Leave-One-Out FK Uncertainty Attribution** and validated it against baselines.

### Key Finding

**Feature Importance ≠ Uncertainty Contribution**

- Pooled ρ = -0.175, 95% CI: [-0.396, 0.056], p = 0.122
- Cross-dataset average: ρ = -0.17

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
├── RESEARCH_ROADMAP.md            # ** MASTER PLAN: Baseline → Causal **
├── SESSION_SUMMARY.md             # Current progress
├── BASELINE_COMPARISON.md         # Phase 1 findings
├── CASE_STUDY.md                  # Business interpretation
├── PAPER_OUTLINE.md               # Paper outline (will evolve)
├── RESULTS_TABLE.md               # Phase 1 results
│
├── fk_uncertainty_attribution.py  # Our method (Leave-One-Out)
├── shap_attribution.py            # SHAP baseline
├── permutation_attribution.py     # Permutation baseline
├── vfa_attribution.py             # VFA baseline
├── compare_attribution_methods.py # Comparison framework
├── covid_timeline_analysis.py     # Monthly attribution timeline
├── statistical_significance.py    # Bootstrap CIs and p-values
│
├── results/                       # JSON results
│   ├── fk_uncertainty_attribution.json
│   ├── shap_attribution.json
│   ├── permutation_attribution.json
│   ├── vfa_attribution.json
│   └── statistical_significance.json
│
├── cache/                         # Cached datasets (pickle)
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

### SALT Dataset (8 tasks)

| Comparison | Spearman ρ | Top-3 Overlap |
|------------|------------|---------------|
| LOO vs SHAP | 0.064 | 41.7% |
| LOO vs Perm | -0.273 | 25.0% |
| LOO vs VFA | -0.085 | 54.2% |

### Stack Dataset (1 task)

| Comparison | Spearman ρ | Top FK Match |
|------------|------------|--------------|
| LOO vs SHAP | -0.400 | No |

### Cross-Dataset Summary

| Dataset | LOO vs SHAP ρ |
|---------|---------------|
| SALT | +0.064 |
| Stack | -0.400 |
| **Average** | **-0.168** |

---

## Next Steps

1. **COVID Timeline** - Monthly attribution changes (script created, running)
2. **Statistical Significance** - Bootstrap CIs complete (ρ=-0.175, p=0.122)
3. **Paper Writing** - UAI submission outline ready

### Run COVID Timeline
```bash
python chorok/covid_timeline_analysis.py --task sales-group --sample_size 2000
```

### Run Statistical Significance
```bash
python chorok/statistical_significance.py
```

---

## Archive

Previous research (PSI vs UQ correlation) is in `archive/phase1_psi_uq/`.
That phase concluded: PSI does not correlate with epistemic uncertainty (r=0.425, p=0.294).

---

**Last Updated**: 2025-11-29
