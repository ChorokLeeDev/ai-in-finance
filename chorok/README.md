# Causal UQ for Relational Data

**Research Question**: Can we use FK structure for *causal* (not correlational) uncertainty attribution?

---

## Current Phase: Baseline → Causal Framework

See **[RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md)** for the full plan.

### Status
- **Phase 1 (Baseline)**: 100% - LOO, SHAP, Permutation, VFA
- **Phase 2 (Theory)**: 100% - FK→DAG, Interventional UQ, Identification
- **Phase 3 (Method)**: 100% - FK-Causal-UQ algorithm implemented
- **Phase 4 (Experiments)**: 60% - Synthetic validation complete
- **Phase 5 (Paper)**: 0%

### Key Result (Synthetic Validation)
| Method | Correlation with Ground Truth |
|--------|------------------------------|
| **Interventional (Ours)** | **ρ = 0.964, p = 0.0005** |
| LOO (Baseline) | ρ = 0.741, p = 0.057 |

### Documents
- **[RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md)** - Master plan with checkboxes
- **[NOVELTY_ASSESSMENT.md](NOVELTY_ASSESSMENT.md)** - Honest evaluation of contribution
- **[LITERATURE_REVIEW.md](LITERATURE_REVIEW.md)** - Causal inference + UQ literature
- **[THEORY.md](THEORY.md)** - Formal theory: FK→DAG, identification theorem

### Code
- **fk_causal_uq.py** - Our method (interventional, no retraining)
- **synthetic_causal_data.py** - Synthetic validation with ground truth

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
│
├── # THEORY (Phase 2)
├── LITERATURE_REVIEW.md           # Causal inference + UQ literature
├── THEORY.md                      # FK→DAG mapping, identification theorem
│
├── # CAUSAL METHOD (Phase 3) - THE NOVEL CONTRIBUTION
├── fk_causal_uq.py                # ** OUR METHOD: Interventional, no retraining **
├── synthetic_causal_data.py       # Ground truth validation (ρ=0.964)
│
├── # BASELINES (Phase 1)
├── fk_uncertainty_attribution.py  # LOO baseline (correlational)
├── shap_attribution.py            # SHAP baseline
├── permutation_attribution.py     # Permutation baseline
├── vfa_attribution.py             # VFA baseline
│
├── # ANALYSIS
├── compare_attribution_methods.py # Comparison framework
├── covid_timeline_analysis.py     # Monthly attribution timeline
├── statistical_significance.py    # Bootstrap CIs and p-values
│
├── # DOCUMENTATION
├── PAPER_OUTLINE.md               # Paper outline
├── RESULTS_TABLE.md               # Phase 1 results
├── CASE_STUDY.md                  # Business interpretation
│
├── results/                       # JSON results
│   ├── synthetic_validation.json  # ** KEY: ρ=0.964 vs ρ=0.741 **
│   ├── fk_uncertainty_attribution.json
│   └── ...
│
├── cache/                         # Cached datasets (pickle)
└── figures/                       # Plots
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

## Next Steps (Priority Order)

1. **Run FK-Causal-UQ on SALT** - Test causal method on real COVID data
2. **Semi-synthetic validation** - Inject known shifts into real FK structure
3. **Paper writing** - Abstract with synthetic validation results

### Run Our Causal Method
```bash
python chorok/fk_causal_uq.py --task sales-group --sample_size 5000
```

### Run Synthetic Validation
```bash
python chorok/synthetic_causal_data.py
```

---

## For New Sessions

**TL;DR**: We're building a causal (not correlational) UQ attribution framework for relational data using FK structure as causal prior.

**Key files to read first**:
1. `RESEARCH_ROADMAP.md` - Master plan with checkboxes
2. `THEORY.md` - The theoretical contribution (identification theorem)
3. `fk_causal_uq.py` - The novel algorithm

**Current achievement**: Synthetic validation shows our interventional method (ρ=0.964) significantly outperforms LOO baseline (ρ=0.741) in recovering ground truth causal effects.

---

**Last Updated**: 2025-11-29
