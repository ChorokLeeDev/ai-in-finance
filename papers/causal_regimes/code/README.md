# Causal Structure Changes Across Market Regimes: Code

This directory contains the code for reproducing the experiments in the paper.

## Requirements

```bash
pip install numpy pandas scipy hmmlearn statsmodels matplotlib
```

## Data

The analysis uses Fama-French factor data, which is automatically downloaded from:
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

## Main Scripts

### Regime Detection (Section 3.2)
- `gate2_regime_detection.py` - Student-t HMM for regime detection
  - Implements multivariate Student-t emission model
  - Compares with Gaussian HMM baseline

### Causal Analysis (Section 4)
- `gate3_per_regime_dag.py` - Per-regime Granger causality analysis
  - Extracts regime-specific subsamples
  - Runs pairwise Granger causality tests
  - Applies Bonferroni correction

### Validation
- `gate1_validation.py` - Data quality checks
- `out_of_sample_2024.py` - 2024 out-of-sample validation

### Figures
- `fig2_regime_causal_dag.py` - Generate causal DAG figures

## Quick Start

```python
# Run full pipeline
python gate2_regime_detection.py  # Detect regimes
python gate3_per_regime_dag.py    # Analyze causality
```

## Key Results

| Regime | Direction | p-value | Lag |
|--------|-----------|---------|-----|
| Normal | - | - | - |
| Crowding | SMB → HML | 1.94e-4 | 3 |
| Crisis | HML → SMB | 1.89e-5 | 9 |

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{lee2025causal,
  title={Causal Structure Changes Across Market Regimes: Evidence from Factor Returns},
  author={Lee, Chorok},
  booktitle={ACM International Conference on AI in Finance},
  year={2025}
}
```
