#!/usr/bin/env python3
"""
common_driver_test.py — Test if common drivers (VIX, TED spread) explain HML→SMB causality

MOTIVATION:
  Reviewer criticism: "The HML→SMB Granger causality in Normal regime could simply reflect
  a common driver (e.g., funding liquidity) affecting both with different lags."

APPROACH:
  1. Download VIX daily data (1990-2024) from yfinance or FRED
  2. Download funding stress proxy (TEDRATE or DTB3) from FRED
  3. Merge with Fama-French factors and canonical regime assignments
  4. For each regime (Normal, Elevated, Crisis), run:
     (a) Baseline: SMB_t ~ SMB_{t-1} + HML_{t-1} (standard Granger)
     (b) VIX-controlled: SMB_t ~ SMB_{t-1} + HML_{t-1} + VIX_{t-1}
     (c) VIX+change: SMB_t ~ SMB_{t-1} + HML_{t-1} + VIX_{t-1} + ΔVIX_{t-1}
     (d) TED-controlled: SMB_t ~ SMB_{t-1} + HML_{t-1} + TED_{t-1}
  5. Report: F-tests, p-values, coefficient magnitudes for HML
  6. Summary: Does HML significance survive common-driver controls?

INTERPRETATION:
  - If p-value for HML stays significant after controls → true causal effect (not just common driver)
  - If p-value increases substantially after controls → likely common driver effect
"""

import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
import io, contextlib

# ── Configuration ──────────────────────────────────────────────────────────
BASE = Path("/sessions/festive-youthful-mccarthy/mnt/causal_regimes")
REGIMES_PATH = BASE / "results" / "canonical_regimes.json"
FF_DATA_PATH = BASE / "data" / "25_Portfolios_5x5_Daily.csv"
RESULTS_PATH = BASE / "results" / "common_driver_test.json"
FIGURE_PATH = BASE / "figures" / "common_driver_controls.pdf"

LAG = 9  # Match canonical_table1.py
FACTORS = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]

print("=" * 80)
print("COMMON DRIVER ROBUSTNESS TEST: HML→SMB Causality")
print("=" * 80)

# ── Step 1: Load canonical regime assignments ──────────────────────────────
print("\n[1] Loading canonical regime assignments...")
try:
    with open(REGIMES_PATH) as f:
        canon = json.load(f)

    meta = canon["metadata"]
    hmm = meta["hmm_params"]

    # Build regime series
    assign_df = pd.DataFrame(canon["assignments"])
    assign_df["date"] = pd.to_datetime(assign_df["date"])
    assign_df = assign_df.set_index("date").sort_index()
    regime_series = assign_df["regime_name"]

    id_to_name = {0: "Normal", 1: "Elevated", 2: "Crisis"}
    nu_map = {id_to_name[i]: hmm["nu"][i] for i in range(3)}
    diagA_map = {id_to_name[i]: hmm["diag_A"][i] for i in range(3)}

    print(f"   ✓ Loaded {len(regime_series)} regime assignments")
    print(f"   Range: {regime_series.index[0].date()} – {regime_series.index[-1].date()}")
except Exception as e:
    print(f"   ✗ Error loading regimes: {e}")
    sys.exit(1)

# ── Step 2: Load Fama-French factor data ───────────────────────────────────
print("\n[2] Loading Fama-French 5-factor + momentum data...")
try:
    ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily",
                         "famafrench", start="1990-01-01")[0]
    mom = web.DataReader("F-F_Momentum_Factor_daily",
                        "famafrench", start="1990-01-01")[0]

    df_factors = ff5.join(mom)
    df_factors.columns = ["MKT", "SMB", "HML", "RMW", "CMA", "RF", "MOM"]
    df_factors = df_factors.drop("RF", axis=1)

    # Convert PeriodIndex to DatetimeIndex if needed
    if isinstance(df_factors.index, pd.PeriodIndex):
        df_factors.index = df_factors.index.to_timestamp()

    df_factors = df_factors.loc["1990-01-02":"2024-12-31"]
    print(f"   ✓ Loaded {len(df_factors)} factor days")
except Exception as e:
    print(f"   ✗ Error loading Fama-French data: {e}")
    sys.exit(1)

# ── Step 3: Download VIX data ──────────────────────────────────────────────
print("\n[3] Downloading VIX volatility index...")
try:
    import yfinance as yf

    vix_data = yf.download("^VIX", start="1990-01-01", end="2024-12-31",
                           progress=False)['Close']
    vix_data = vix_data.dropna()

    print(f"   ✓ Downloaded {len(vix_data)} VIX observations")
    print(f"   VIX range: {vix_data.min():.1f} – {vix_data.max():.1f}")

except Exception as e:
    print(f"   ! Warning: yfinance VIX download failed ({e})")
    print("   Attempting FRED VIXCLS...")
    try:
        vix_data = web.DataReader("VIXCLS", "fred", start="1990-01-01")['VIXCLS']
        vix_data = vix_data.dropna()
        vix_data.index = pd.to_datetime(vix_data.index)
        print(f"   ✓ Downloaded {len(vix_data)} VIX observations from FRED")
    except Exception as e2:
        print(f"   ✗ FRED VIX also failed: {e2}")
        vix_data = None

# ── Step 4: Download funding stress proxy (TED spread or alternatives) ─────
print("\n[4] Downloading funding stress proxy (TED spread)...")
ted_data = None
try:
    # Try TEDRATE first (TED spread)
    ted_data = web.DataReader("TEDRATE", "fred", start="1990-01-01")['TEDRATE']
    ted_data = ted_data.dropna()
    ted_data.index = pd.to_datetime(ted_data.index)
    print(f"   ✓ Downloaded {len(ted_data)} TED spread observations")
    print(f"   TED range: {ted_data.min():.2f} – {ted_data.max():.2f} bps")
except Exception as e:
    print(f"   ! Warning: TEDRATE failed ({e})")
    print("   Trying DTB3 (3-month T-bill rate)...")
    try:
        ted_data = web.DataReader("DTB3", "fred", start="1990-01-01")['DTB3']
        ted_data = ted_data.dropna()
        ted_data.index = pd.to_datetime(ted_data.index)
        print(f"   ✓ Downloaded {len(ted_data)} DTB3 observations")
    except Exception as e2:
        print(f"   ! DTB3 also failed: {e2}")
        ted_data = None

# ── Step 5: Merge all data ─────────────────────────────────────────────────
print("\n[5] Merging factor, macro, and regime data...")

# Create master dataframe
df_master = df_factors.copy()
df_master['regime'] = regime_series

# Add VIX if available
if vix_data is not None:
    # Align dates
    common_dates_vix = df_master.index.intersection(vix_data.index)
    df_master = df_master.loc[common_dates_vix]
    df_master['VIX'] = vix_data.loc[df_master.index]
    has_vix = True
else:
    has_vix = False
    print("   ! VIX not available - skipping VIX specifications")

# Add TED if available
if ted_data is not None:
    common_dates_ted = df_master.index.intersection(ted_data.index)
    df_master = df_master.loc[common_dates_ted]
    df_master['TED'] = ted_data.loc[df_master.index]
    has_ted = True
else:
    has_ted = False
    print("   ! TED spread not available - skipping TED specifications")

# Remove rows with NaN in regimes
df_master = df_master.dropna(subset=['regime'])

print(f"   ✓ Master data: {len(df_master)} days after merging")
print(f"   Date range: {df_master.index[0].date()} – {df_master.index[-1].date()}")

# ── Step 6: Define Granger test function with controls ────────────────────
def granger_with_controls(y_col, x_col, control_cols, regime_name, lag, df_in):
    """
    Run Granger causality test: x → y, controlling for other variables.

    Design matrix:
      [y_t] = [SMB_t | HML_t-lag ... HML_t-1 | SMB_t-lag ... SMB_t-1 |
               CONTROL_t-lag ... CONTROL_t-1 | ...]

    We test if coefficients on HML are jointly zero (F-test).

    Parameters
    ----------
    y_col : str
        Target variable (e.g., 'SMB')
    x_col : str
        Causal variable to test (e.g., 'HML')
    control_cols : list of str
        Control variables (empty for baseline)
    regime_name : str
        Regime to filter to ('Normal', 'Elevated', 'Crisis')
    lag : int
        Number of lags
    df_in : DataFrame
        Input data with 'regime' column

    Returns
    -------
    dict with keys:
      - f_stat, p_val, n_clean
      - hml_coeff_mean: average HML coefficient across lags
      - r_squared: R² of the regression
    """

    # Get regime mask
    mask = (df_in['regime'] == regime_name).values

    # Find indices where full lag window is in-regime
    clean = np.ones(len(mask), dtype=bool)
    for k in range(lag + 1):
        clean[k:] &= mask[k:len(mask)]
    clean[:lag] = False

    n_clean = int(clean.sum())
    if n_clean < lag + 10:
        return {
            'f_stat': np.nan, 'p_val': np.nan, 'n_clean': n_clean,
            'hml_coeff_mean': np.nan, 'r_squared': np.nan
        }

    # Get indices
    idx = np.where(clean)[0]

    # Find contiguous runs
    runs = []
    if len(idx) > 0:
        start = idx[0]
        for i in range(1, len(idx)):
            if idx[i] != idx[i - 1] + 1:
                if idx[i - 1] - start + 1 > lag:
                    runs.append((start, idx[i - 1]))
                start = idx[i]
        if idx[-1] - start + 1 > lag:
            runs.append((start, idx[-1]))

    if not runs:
        return {
            'f_stat': np.nan, 'p_val': np.nan, 'n_clean': n_clean,
            'hml_coeff_mean': np.nan, 'r_squared': np.nan
        }

    # Build design matrix from runs
    pieces = []
    for s, e in runs:
        n_obs = e - s + 1

        # Dependent variable
        y_vals = df_in[y_col].values[s:e+1]

        # Lagged independent variables
        X_lagged = []

        # Lagged HML (the variable we're testing for Granger causality)
        for lag_i in range(1, lag + 1):
            X_lagged.append(df_in[x_col].values[s+lag-lag_i:e+1-lag_i])

        # Lagged dependent variable (own lags)
        for lag_i in range(1, lag + 1):
            X_lagged.append(df_in[y_col].values[s+lag-lag_i:e+1-lag_i])

        # Control variables and their lags
        for ctrl in control_cols:
            if ctrl in df_in.columns:
                for lag_i in range(1, lag + 1):
                    X_lagged.append(df_in[ctrl].values[s+lag-lag_i:e+1-lag_i])

        if X_lagged:
            X_mat = np.column_stack(X_lagged)
            # Use data from lag onwards to match X dimensions
            y_trimmed = y_vals[lag:]
            pieces.append((y_trimmed, X_mat))

    if not pieces:
        return {
            'f_stat': np.nan, 'p_val': np.nan, 'n_clean': n_clean,
            'hml_coeff_mean': np.nan, 'r_squared': np.nan
        }

    # Concatenate
    y_all = np.concatenate([p[0] for p in pieces])
    X_all = np.vstack([p[1] for p in pieces])

    if len(y_all) < lag + 10:
        return {
            'f_stat': np.nan, 'p_val': np.nan, 'n_clean': n_clean,
            'hml_coeff_mean': np.nan, 'r_squared': np.nan
        }

    # Fit regression
    try:
        from sklearn.linear_model import LinearRegression
        from scipy.stats import f

        reg = LinearRegression()
        reg.fit(X_all, y_all)
        coef = reg.coef_
        intercept = reg.intercept_

        # Predictions and residuals
        y_pred = reg.predict(X_all)
        residuals = y_all - y_pred

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_all - y_all.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # F-test for HML coefficients (first `lag` coefficients)
        # We'll use robust approach: test if HML lags are jointly zero

        # Residual standard error
        n_obs = len(y_all)
        n_vars = X_all.shape[1]
        rse = np.sqrt(ss_res / (n_obs - n_vars - 1))

        # Covariance of coefficients
        X_with_const = np.column_stack([np.ones(len(X_all)), X_all])
        try:
            cov_matrix = np.linalg.inv(X_with_const.T @ X_with_const) * (rse ** 2)

            # Extract HML coefficient covariance (first `lag` diagonals after intercept)
            hml_cov = cov_matrix[1:lag+1, 1:lag+1]
            hml_coef = coef[:lag]

            # F-statistic: (HML'@ Sigma^{-1} @ HML) / lag
            try:
                hml_cov_inv = np.linalg.inv(hml_cov)
                f_stat = (hml_coef @ hml_cov_inv @ hml_coef) / lag
                p_val = 1 - f.cdf(f_stat, lag, n_obs - n_vars - 1)
            except:
                # Fallback: use Wald-style test with mean coefficient
                hml_coeff_mean = np.mean(hml_coef)
                hml_se = np.std(hml_coef) / np.sqrt(lag)
                t_stat = hml_coeff_mean / (hml_se + 1e-10)
                f_stat = t_stat ** 2
                p_val = 1 - stats.f.cdf(f_stat, 1, n_obs - n_vars - 1)
        except:
            f_stat = np.nan
            p_val = np.nan

        hml_coeff_mean = np.mean(coef[:lag])

        return {
            'f_stat': f_stat, 'p_val': p_val, 'n_clean': n_clean,
            'hml_coeff_mean': hml_coeff_mean, 'r_squared': r_squared
        }

    except Exception as e:
        print(f"     Error in regression: {e}")
        return {
            'f_stat': np.nan, 'p_val': np.nan, 'n_clean': n_clean,
            'hml_coeff_mean': np.nan, 'r_squared': np.nan
        }


# ── Step 7: Run Granger tests for each regime ──────────────────────────────
print("\n[6] Running Granger causality tests with controls...\n")

results_by_regime = {}
regime_order = ["Normal", "Elevated", "Crisis"]

for regime_name in regime_order:
    print(f"  ┌─ {regime_name} Regime")

    results_by_spec = {}

    # Specification (a): Baseline (no controls)
    print(f"  │  (a) Baseline (no controls)")
    result_a = granger_with_controls('SMB', 'HML', [], regime_name, LAG, df_master)
    results_by_spec['baseline'] = result_a

    if not np.isnan(result_a['f_stat']):
        print(f"  │      F={result_a['f_stat']:.3f}  p={result_a['p_val']:.4f}  "
              f"HML_coef={result_a['hml_coeff_mean']:.4f}  n={result_a['n_clean']}")
    else:
        print(f"  │      N/A (insufficient data)")

    # Specification (b): VIX-controlled
    if has_vix:
        print(f"  │  (b) VIX-controlled")
        result_b = granger_with_controls('SMB', 'HML', ['VIX'], regime_name, LAG, df_master)
        results_by_spec['vix'] = result_b

        if not np.isnan(result_b['f_stat']):
            p_change = ((result_b['p_val'] - result_a['p_val']) / (result_a['p_val'] + 1e-10) * 100)
            print(f"  │      F={result_b['f_stat']:.3f}  p={result_b['p_val']:.4f} ({p_change:+.0f}%)  "
                  f"HML_coef={result_b['hml_coeff_mean']:.4f}  n={result_b['n_clean']}")
        else:
            print(f"  │      N/A")

    # Specification (c): VIX + change in VIX
    if has_vix:
        # Add VIX change
        df_master_vix = df_master.copy()
        df_master_vix['ΔVIX'] = df_master_vix['VIX'].diff()

        print(f"  │  (c) VIX + ΔVIX-controlled")
        result_c = granger_with_controls('SMB', 'HML', ['VIX', 'ΔVIX'],
                                         regime_name, LAG, df_master_vix)
        results_by_spec['vix_delta'] = result_c

        if not np.isnan(result_c['f_stat']):
            p_change = ((result_c['p_val'] - result_a['p_val']) / (result_a['p_val'] + 1e-10) * 100)
            print(f"  │      F={result_c['f_stat']:.3f}  p={result_c['p_val']:.4f} ({p_change:+.0f}%)  "
                  f"HML_coef={result_c['hml_coeff_mean']:.4f}  n={result_c['n_clean']}")
        else:
            print(f"  │      N/A")

    # Specification (d): TED-controlled
    if has_ted:
        print(f"  │  (d) TED-controlled")
        result_d = granger_with_controls('SMB', 'HML', ['TED'], regime_name, LAG, df_master)
        results_by_spec['ted'] = result_d

        if not np.isnan(result_d['f_stat']):
            p_change = ((result_d['p_val'] - result_a['p_val']) / (result_a['p_val'] + 1e-10) * 100)
            print(f"  │      F={result_d['f_stat']:.3f}  p={result_d['p_val']:.4f} ({p_change:+.0f}%)  "
                  f"HML_coef={result_d['hml_coeff_mean']:.4f}  n={result_d['n_clean']}")
        else:
            print(f"  │      N/A")

    print(f"  └─")
    results_by_regime[regime_name] = results_by_spec

# ── Step 8: Interpret results ──────────────────────────────────────────────
print("\n[7] Interpretation and Summary\n")

summary_rows = []

for regime_name in regime_order:
    specs = results_by_regime[regime_name]
    baseline = specs['baseline']

    if np.isnan(baseline['f_stat']):
        print(f"  {regime_name}: Insufficient data")
        continue

    sig_baseline = baseline['p_val'] < 0.05

    row = {
        'regime': regime_name,
        'baseline_f': round(baseline['f_stat'], 3),
        'baseline_p': round(baseline['p_val'], 4),
        'baseline_sig': sig_baseline,
        'baseline_coef': round(baseline['hml_coeff_mean'], 4),
    }

    # Check each control
    for spec_name, spec_key in [('VIX', 'vix'), ('VIX+ΔVIX', 'vix_delta'), ('TED', 'ted')]:
        if spec_key in specs:
            spec_result = specs[spec_key]
            if not np.isnan(spec_result['f_stat']):
                sig_controlled = spec_result['p_val'] < 0.05
                p_change = ((spec_result['p_val'] - baseline['p_val'])
                           / (baseline['p_val'] + 1e-10) * 100)
                survives = sig_baseline and sig_controlled

                row[f'{spec_name}_p'] = round(spec_result['p_val'], 4)
                row[f'{spec_name}_sig'] = sig_controlled
                row[f'{spec_name}_survives'] = survives
                row[f'{spec_name}_coef'] = round(spec_result['hml_coeff_mean'], 4)

                print(f"  {regime_name} + {spec_name}:")
                print(f"    p-value: {baseline['p_val']:.4f} → {spec_result['p_val']:.4f} "
                      f"({p_change:+.1f}%)")
                print(f"    HML coef: {baseline['hml_coeff_mean']:.4f} → {spec_result['hml_coeff_mean']:.4f}")
                print(f"    Survives control: {'YES ✓' if survives else 'NO ✗'}")
                print()

    summary_rows.append(row)

# ── Step 9: Save results ───────────────────────────────────────────────────
print("[8] Saving results...\n")

output = {
    "description": "Common driver robustness test: Does HML→SMB Granger causality survive controls?",
    "date": str(pd.Timestamp.now()),
    "lag": LAG,
    "date_range": f"{df_master.index[0].date()} to {df_master.index[-1].date()}",
    "n_observations": len(df_master),
    "has_vix": has_vix,
    "has_ted": has_ted,
    "results_by_regime": results_by_regime,
    "summary": summary_rows,
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"   ✓ Results saved to {RESULTS_PATH}")

# ── Step 10: Create visualization ──────────────────────────────────────────
print("\n[9] Creating visualization...\n")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    spec_colors = {
        'baseline': '#1f77b4',
        'vix': '#ff7f0e',
        'vix_delta': '#2ca02c',
        'ted': '#d62728',
    }

    spec_labels = {
        'baseline': 'Baseline',
        'vix': 'VIX-controlled',
        'vix_delta': 'VIX+ΔVIX-controlled',
        'ted': 'TED-controlled',
    }

    for ax_idx, regime_name in enumerate(regime_order):
        ax = axes[ax_idx]
        specs = results_by_regime[regime_name]

        # Collect p-values
        spec_names = []
        p_vals = []
        colors = []

        for spec_key in ['baseline', 'vix', 'vix_delta', 'ted']:
            if spec_key in specs:
                result = specs[spec_key]
                if not np.isnan(result['p_val']):
                    spec_names.append(spec_labels[spec_key])
                    p_vals.append(result['p_val'])
                    colors.append(spec_colors[spec_key])

        if p_vals:
            # Plot bars
            x_pos = np.arange(len(spec_names))
            bars = ax.bar(x_pos, p_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

            # Significance line at p=0.05
            ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='p=0.05')

            # Formatting
            ax.set_ylabel('p-value', fontsize=12, fontweight='bold')
            ax.set_title(f'{regime_name} Regime', fontsize=13, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(spec_names, rotation=45, ha='right', fontsize=10)
            ax.set_ylim(0, max(p_vals + [0.15]))
            ax.grid(axis='y', alpha=0.3, linestyle=':')

            # Add value labels on bars
            for bar, p_val in zip(bars, p_vals):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{p_val:.4f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{regime_name} Regime', fontsize=13, fontweight='bold')

    plt.suptitle('HML→SMB Granger Causality: Impact of Common-Driver Controls',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, dpi=300, bbox_inches='tight')
    print(f"   ✓ Figure saved to {FIGURE_PATH}")
    plt.close()

except Exception as e:
    print(f"   ! Error creating figure: {e}")

# ── Final Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

for regime_name in regime_order:
    specs = results_by_regime[regime_name]
    baseline = specs['baseline']

    if np.isnan(baseline['f_stat']):
        print(f"\n{regime_name}: Insufficient data")
        continue

    print(f"\n{regime_name} Regime:")
    print(f"  Baseline HML→SMB p-value: {baseline['p_val']:.4f}")

    baseline_sig = baseline['p_val'] < 0.05
    print(f"  Baseline significant: {'YES ✓' if baseline_sig else 'NO ✗'}")

    if baseline_sig:
        # Check if survives all controls
        survives_all = True
        for spec_key in ['vix', 'vix_delta', 'ted']:
            if spec_key in specs:
                spec_result = specs[spec_key]
                if not np.isnan(spec_result['p_val']):
                    if spec_result['p_val'] >= 0.05:
                        survives_all = False
                        break

        if survives_all:
            print(f"  → Effect SURVIVES all common-driver controls ✓")
            print(f"    Interpretation: True causal effect, not driven by VIX/TED")
        else:
            print(f"  → Effect LOST under some controls ✗")
            print(f"    Interpretation: Likely driven by common liquidity/volatility factors")
    else:
        print(f"  → Effect not significant in baseline")

print("\n" + "=" * 80)
