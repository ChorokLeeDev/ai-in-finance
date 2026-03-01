# Causal Structure Changes Across Market Regimes: Reproducibility Package

## Overview

This package contains all code and instructions necessary to reproduce the results from "Causal Structure Changes Across Market Regimes: Evidence from Factor Returns" (2025).

**Key Result**: We detect three market regimes (Normal, Elevated, Crisis) using a Student-t HMM and show that causal relationships between Fama-French factors differ significantly across regimes. The HML→SMB causal link is strongest in the Elevated regime, suggesting crowding-driven behavior.

## Quick Start (5 minutes)

```bash
# 1. Install exact dependencies
pip install -r requirements_pinned.txt

# 2. Run verification script
python code/verify_reproduction.py

# 3. Check output
cat results/verify_reproduction.txt
```

Expected result: All tests pass with numerical outputs within specified tolerances.

## System Requirements

- **Python**: 3.10 or later
- **RAM**: 4GB minimum (8GB recommended for parallel runs)
- **Disk**: 200MB for data and results
- **Internet**: Required for downloading Fama-French data on first run

## Installation (Detailed)

### 1. Set Up Python Environment

```bash
# Create virtual environment (recommended)
python3.10 -m venv causal_regimes_env
source causal_regimes_env/bin/activate  # On Windows: causal_regimes_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 2. Install Pinned Dependencies

```bash
pip install -r requirements_pinned.txt
```

**What's included:**
- numpy==2.2.6 - Numerical arrays
- scipy==1.15.3 - Statistical functions
- pandas==2.3.3 - Data manipulation
- statsmodels==0.14.6 - Time-series and structural breaks
- scikit-learn==1.7.2 - Machine learning utilities
- matplotlib==3.10.8 - Plotting
- seaborn==0.13.2 - Statistical visualization
- torch==2.10.0 - Deep learning (for LSTM Granger analysis)
- pandas-datareader==0.10.0 - Downloads Kenneth French data
- requests==2.32.5 - HTTP library
- pyarrow==23.0.1 - Arrow format support

### 3. Verify Installation

```bash
python -c "import numpy, scipy, pandas, statsmodels, matplotlib, torch; print('All imports successful')"
```

## Execution Order & Script Descriptions

### Core Pipeline (Main Results)

The following scripts should be run in order:

#### 1. **gate1_validation.py** (~30 min)
**Purpose**: Out-of-sample validation of the core HML→SMB Granger causality

**Inputs**:
- Fama-French factors (auto-downloaded)

**Outputs**:
- Granger causality test results
- Permutation test results
- Out-of-sample prediction metrics

**Key Metrics**:
- HML→SMB lag (full sample): lag ≈ 9
- Out-of-sample p-value: p < 0.05 (passes Gate 1)

**Run**:
```bash
python code/gate1_validation.py
```

#### 2. **normal_regime_subsample.py** (~30 min)
**Purpose**: Verify HML→SMB causality in Normal regime across pre/post-GFC periods and compute exact Andrews HAC p-values

**Inputs**:
- Fama-French factors
- HMM regime assignments (seed 28)

**Outputs**:
- `results/normal_regime_subsample.json`
- Pre/post-2008 split results
- Andrews HAC bandwidth estimates

**Key Metrics**:
- Normal regime, full sample HML→SMB lag=1, F-p ≈ 0.0042
- Normal regime, full sample HAC(Andrews)-p ≈ 0.0341

**Run**:
```bash
python code/normal_regime_subsample.py
```

#### 3. **bai_perron_normal_regime.py** (~30 min)
**Purpose**: Test for structural breaks in Normal-regime HML→SMB coefficient

**Inputs**:
- Normal-regime observations
- HML and SMB daily returns

**Outputs**:
- `results/bai_perron_normal.json`
- sup-F statistic
- Estimated break dates
- Chow test at GFC (Jan 2008)
- Rolling coefficient estimates

**Key Metrics**:
- sup-F ≈ 8.91 (Andrews 5% CV = 8.85) → Reject H0 (no break)
- MLE break date: ~Jan 2008 (GFC onset)
- Chow F-test at Jan 2008: p < 0.001

**Run**:
```bash
python code/bai_perron_normal_regime.py
```

#### 4. **frozen_oos_50seeds.py** (~2-3 hours for 50 seeds)
**Purpose**: Out-of-sample robustness across 50 random seeds

**Inputs**:
- All 50 seeds from HMM multistart
- Train: 1990-2012, OOS: 2013-2024

**Outputs**:
- `results/frozen_oos_50seeds.json`
- Distribution of p-values across seeds

**Key Metrics**:
- Median Elevated OOS Granger p-value: ~0.33
- Robustness: Results consistent across seeds

**Run**:
```bash
python code/frozen_oos_50seeds.py  # Optional; time-intensive
```

### Supplementary Analysis Scripts

These scripts generate additional analyses and figures:

- **lstm_granger.py** (~10 min) - Neural Granger causality using LSTM networks
- **permutation_50k.py** (~45 min) - 50,000 permutation test for null distribution
- **crisis_trading_backtest.py** (~5 min) - Trading strategy performance during crises
- **canonical_table1.py** (~5 min) - Recreate paper Table 1 (regime statistics)
- **ff25_overlap_mechanism.py** (~10 min) - Mechanism analysis: FF25 portfolio overlap
- **improved_oos_validation.py** (~10 min) - Alternative OOS validation approaches

## Expected Numerical Outputs

All key numerical outputs are documented in `expected_outputs.json`. This file specifies:

1. **HMM Primary Fit** (seed 28, full sample)
   - Log-likelihood: -37375.77 ± 0.5
   - BIC: 75178.50 ± 1.0
   - Regime counts: Normal (3023), Elevated (4723), Crisis (1071)

2. **Granger Causality (HML→SMB, lag=1)**
   - Normal regime: F-p = 0.0042 ± 0.0001
   - Elevated regime: F-p = 8.749e-09 ± 1e-10
   - Crisis regime: F-p = 0.6954 ± 0.01 (not significant)

3. **Frozen OOS (2013-2024)**
   - Elevated regime: F-p = 0.3173 ± 0.01 (not significant)
   - Total OOS observations: 3020 ± 10
   - OOS regime distribution: Elevated 34.3%, Normal 34.9%, Crisis 30.8%

4. **Structural Breaks (Normal Regime)**
   - sup-F statistic: 8.91 (Andrews 5% CV = 8.85) → Significant
   - Estimated break date: Jan 2008
   - Chow F at Jan 2008: High F-statistic, p < 0.001

## Verification Script

Run the verification script to test reproducibility:

```bash
python code/verify_reproduction.py
```

This script:
1. Downloads fresh Fama-French data
2. Fits the primary HMM (seed 28)
3. Runs Granger causality tests
4. Validates frozen OOS setup
5. Compares outputs against `expected_outputs.json`
6. Reports PASS/FAIL for each check with tolerances

**Expected runtime**: 15-20 minutes

## Output Locations

```
mnt/causal_regimes/
├── results/
│   ├── canonical_table1.json          # Table 1: Regime statistics
│   ├── bai_perron_normal.json         # Structural break tests
│   ├── normal_regime_subsample.json   # Pre/post-GFC analysis
│   ├── frozen_oos_50seeds.json        # 50-seed robustness
│   ├── ff25_overlap_normal_seed28.json # Portfolio overlap mechanism
│   └── verify_reproduction.txt        # Verification script output
├── code/
│   ├── gate1_validation.py            # OOS validation
│   ├── normal_regime_subsample.py     # Normal regime analysis
│   ├── bai_perron_normal_regime.py    # Structural breaks
│   └── verify_reproduction.py         # Verification script
├── expected_outputs.json              # Expected numerical values
├── requirements_pinned.txt            # Pinned dependencies
└── REPRODUCE.md                       # This file
```

## Troubleshooting

### Issue: "Could not download data"

**Cause**: Network timeout or Kenneth French server unavailable

**Solution**:
```bash
# Manually download and place in mnt/causal_regimes/data/
# https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
# Then modify scripts to use local path instead of download
```

### Issue: "Insufficient memory"

**Solution**:
- Run on machine with ≥8GB RAM
- Reduce `n_iter` in HMM fitting (trade-off with convergence)
- Run OOS validation without 50-seed analysis

### Issue: "Different random seed effects"

**Reason**: Normal EM algorithm randomness

**Mitigation**:
- Always use seed=28 for primary results
- Seed=28 provides the "highest LL optimum" in the paper
- For robustness, run with multiple seeds (see frozen_oos_50seeds.py)

### Issue: "Structural break computation is slow"

**Reason**: Scanning all possible break dates requires n−1 Chow tests

**Solution**:
- Use parallel computation (see code comments for parallelization)
- This is expected; computation time is ~30 minutes

## Data Sources

### Fama-French Factors

**Daily returns** (in percent) for:
- **MKT** (Market excess return)
- **SMB** (Small minus Big: size factor)
- **HML** (High minus Low: value factor)
- **RMW** (Robust minus Weak: profitability)
- **CMA** (Conservative minus Aggressive: investment)
- **MOM** (Momentum)

**Source**: Kenneth French Data Library
**URL**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
**Period**: 1990-01-02 to 2024-12-31 (8,817 daily observations)
**License**: Public domain (Kenneth French's data)

## Citation

If you use this code, please cite:

```bibtex
@article{lee2025causal,
  title={Causal Structure Changes Across Market Regimes:
         Evidence from Factor Returns},
  author={Lee, Chorok},
  journal={ACM International Conference on AI in Finance},
  year={2025}
}
```

## Revision History

- **Original submission**: Q4 2024
- **First revision**: Feb 2025
  - Added normal regime pre/post-GFC analysis
  - Added exact Andrews HAC p-values
  - Added frozen OOS validation with filtered probabilities
  - Added Bai-Perron structural break tests

## Repository Status

- [x] Regime detection (gate2)
- [x] Causal analysis per regime (gate3)
- [x] Out-of-sample validation
- [x] Structural break analysis
- [x] Robustness across seeds
- [x] Reproducibility package

## Contact & Support

For reproducibility issues:
1. Run `verify_reproduction.py` and check output
2. Compare numerical results against `expected_outputs.json`
3. Check tolerances (floating-point precision varies slightly across systems)

---

**Last updated**: 2026-02-28
**Python version**: 3.10+
**Status**: Ready for reproduction
