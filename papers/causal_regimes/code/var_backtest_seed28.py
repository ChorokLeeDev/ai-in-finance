"""
VaR Backtest — Primary Fit (Seed 28)
=====================================

Replicates the hybrid VaR backtest from hybrid_regime_detector.py using the
PRIMARY HMM fit (seed=28, 6 factors including MOM, relabeling by data norm).

Motivation:
  The existing VaR analysis (hybrid_regime_detector.py) uses seed=42 (sensitivity
  fit). The method critic identified this as a structural weakness. This script
  runs the identical VaR protocol under the primary fit to test whether one model
  (seed=28) simultaneously delivers:
    - Bonferroni-significant Granger (Normal, p=8.75e-9)
    - VaR performance at least comparable to seed=42

Design:
  - HMM: K=3 Student-t, seed=28, trained on 1990-2012 (same train/test split)
  - 6 factors: MKT, SMB, HML, RMW, CMA, MOM (canonical primary analysis)
  - Regime labeling: relabel_regimes_by_data_norm with TRAIN-data remap
  - Frozen OOS: filtered probabilities (predict_oos use_filtered=True)
  - VaR models: Unconditional, Regime-conditional, HML-Informed, Hybrid (HMM+Vol)
  - Thresholds: same calibrated params as existing analysis (locked on training)

Outputs:
  results/var_backtest_seed28.json
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not BASE_DIR or BASE_DIR == '/':
    BASE_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes'
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from multistart_hmm_pipeline import (
    download_ff_data,
    StudentTHMM,
    relabel_regimes_by_data_norm,
)

FACTOR_COLS = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
PRIMARY_SEED = 28
TRAIN_END = '2012-12-31'
TEST_START = '2013-01-01'

# Calibrated params (same as existing analysis, locked on training data)
# From hybrid_regime_detector.py / risk_monitoring_results.json
HML_THRESHOLD = -0.5
STRESS_MULTIPLIER = 2.3
WINDOW_CALM = 75
WINDOW_NORMAL = 50
WINDOW_CRISIS = 30
ALPHA = 0.05


# =============================================================================
# CHRISTOFFERSEN TEST
# =============================================================================

def christoffersen_test(violations):
    hits = np.array(violations, dtype=int)
    T = len(hits); n1 = hits.sum(); n0 = T - n1
    pi_hat = n1 / T if T > 0 else 0; alpha = 0.05
    if n1 == 0 or n0 == 0:
        LR_uc = np.nan; p_uc = np.nan
    else:
        LR_uc = -2 * (n1*np.log(alpha) + n0*np.log(1-alpha)
                      - n1*np.log(pi_hat) - n0*np.log(1-pi_hat))
        p_uc = float(1 - stats.chi2.cdf(LR_uc, 1))
    n00 = n01 = n10 = n11 = 0
    for t in range(1, T):
        if hits[t-1] == 0 and hits[t] == 0:   n00 += 1
        elif hits[t-1] == 0 and hits[t] == 1: n01 += 1
        elif hits[t-1] == 1 and hits[t] == 0: n10 += 1
        elif hits[t-1] == 1 and hits[t] == 1: n11 += 1
    if (n00+n01) == 0 or (n10+n11) == 0 or n01 == 0 or n10 == 0:
        LR_ind = np.nan; p_ind = np.nan
    else:
        pi01 = n01/(n00+n01); pi11 = n11/(n10+n11)
        pi_hat2 = (n01+n11)/(n00+n01+n10+n11)
        if any(v <= 0 or v >= 1 for v in [pi01, pi11, pi_hat2]):
            LR_ind = np.nan; p_ind = np.nan
        else:
            LR_ind = -2*((n00+n10)*np.log(1-pi_hat2) + (n01+n11)*np.log(pi_hat2)
                         - n00*np.log(1-pi01) - n01*np.log(pi01)
                         - n10*np.log(1-pi11) - n11*np.log(pi11))
            p_ind = float(1 - stats.chi2.cdf(LR_ind, 1))
    LR_cc = (LR_uc if not np.isnan(LR_uc) else 0) + (LR_ind if not np.isnan(LR_ind) else 0)
    p_cc = float(1 - stats.chi2.cdf(LR_cc, 2)) if not (np.isnan(LR_uc) and np.isnan(LR_ind)) else np.nan
    return {
        'LR_cc': float(LR_cc),
        'p_cc': float(p_cc) if not np.isnan(p_cc) else None,
        'LR_uc': float(LR_uc) if not np.isnan(LR_uc) else None,
        'p_uc': float(p_uc) if not np.isnan(p_uc) else None,
        'LR_ind': float(LR_ind) if not np.isnan(LR_ind) else None,
        'p_ind': float(p_ind) if not np.isnan(p_ind) else None,
    }


# =============================================================================
# VAR MODELS
# =============================================================================

def rolling_historical_var(returns, window=60, alpha=ALPHA):
    T = len(returns)
    var_est = np.full(T, np.nan)
    for t in range(window, T):
        var_est[t] = np.percentile(returns[t-window:t], alpha * 100)
    return var_est


def regime_conditional_var(returns, regimes, alpha=ALPHA,
                            window_calm=WINDOW_CALM, window_normal=WINDOW_NORMAL,
                            window_crisis=WINDOW_CRISIS):
    T = len(returns)
    var_est = np.full(T, np.nan)
    windows = {0: window_calm, 1: window_normal, 2: window_crisis}
    max_w = max(window_calm, window_normal, window_crisis)
    for t in range(max_w, T):
        w = windows.get(int(regimes[t-1]), window_normal)
        var_est[t] = np.percentile(returns[max(0, t-w):t], alpha * 100)
    return var_est


def hml_informed_var(returns, regimes, hml_cumul, alpha=ALPHA,
                     window_calm=WINDOW_CALM, window_normal=WINDOW_NORMAL,
                     window_crisis=WINDOW_CRISIS,
                     hml_threshold=HML_THRESHOLD, stress_multiplier=STRESS_MULTIPLIER):
    T = len(returns)
    var_est = np.full(T, np.nan)
    windows = {0: window_calm, 1: window_normal, 2: window_crisis}
    max_w = max(window_calm, window_normal, window_crisis)
    for t in range(max_w, T):
        r = int(regimes[t-1])
        w = windows.get(r, window_normal)
        base = np.percentile(returns[max(0, t-w):t], alpha * 100)
        if r == 2 and not np.isnan(hml_cumul[t-1]) and hml_cumul[t-1] < hml_threshold:
            var_est[t] = base * stress_multiplier
        else:
            var_est[t] = base
    return var_est


def hybrid_var(returns, hmm_regimes, realized_vol, hml_cumul, vol_threshold,
               alpha=ALPHA, window_calm=WINDOW_CALM, window_normal=WINDOW_NORMAL,
               window_crisis=WINDOW_CRISIS,
               hml_threshold=HML_THRESHOLD, stress_multiplier=STRESS_MULTIPLIER):
    T = len(returns)
    var_est = np.full(T, np.nan)
    windows = {0: window_calm, 1: window_normal, 2: window_crisis}
    vol_override = np.zeros(T, dtype=bool)
    eff_regimes = hmm_regimes.copy().astype(int)
    max_w = max(window_calm, window_normal, window_crisis)
    for t in range(max_w, T):
        r = int(hmm_regimes[t-1])
        if r != 2 and not np.isnan(realized_vol[t-1]) and realized_vol[t-1] > vol_threshold:
            r = 2
            vol_override[t] = True
            eff_regimes[t] = 2
        w = windows.get(r, window_normal)
        base = np.percentile(returns[max(0, t-w):t], alpha * 100)
        if r == 2 and not np.isnan(hml_cumul[t-1]) and hml_cumul[t-1] < hml_threshold:
            var_est[t] = base * stress_multiplier
        elif r == 2 and vol_override[t]:
            var_est[t] = base * 1.5
        else:
            var_est[t] = base
    return var_est, vol_override, eff_regimes


def evaluate_var(returns, var_est, name):
    valid = ~np.isnan(var_est)
    ret = returns[valid]; var = var_est[valid]; T = len(ret)
    viol = ret < var
    n_viol = viol.sum()
    viol_rate = n_viol / T
    avg_mag = float(np.mean(ret[viol] - var[viol])) if n_viol > 0 else 0.0
    cc = christoffersen_test(viol)
    return {
        'model': name,
        'n_days': int(T),
        'n_violations': int(n_viol),
        'violation_rate_pct': round(float(viol_rate * 100), 2),
        'deviation_pct': round(float((viol_rate - 0.05) * 100), 2),
        'avg_violation_magnitude': round(avg_mag, 4),
        'p_cc': cc['p_cc'],
        'p_uc': cc['p_uc'],
        'p_ind': cc['p_ind'],
        'LR_cc': cc['LR_cc'],
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(f"VaR BACKTEST — PRIMARY FIT (seed={PRIMARY_SEED}, 6 factors)")
    print("=" * 70)

    # ---- 1. Load data ----
    print("\nLoading FF data (6 factors including MOM)...")
    df = download_ff_data()
    print(f"  {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")

    dates = df.index
    smb = df['SMB'].values
    hml = df['HML'].values
    hml_cumul_9d = pd.Series(hml, index=dates).rolling(9).sum().values

    # ---- 2. Train/test split ----
    train_mask = dates <= pd.Timestamp(TRAIN_END)
    test_mask  = dates >= pd.Timestamp(TEST_START)
    print(f"\n  Train: {dates[train_mask][0].date()} → {dates[train_mask][-1].date()} ({train_mask.sum()} days)")
    print(f"  Test:  {dates[test_mask][0].date()} → {dates[test_mask][-1].date()} ({test_mask.sum()} days)")

    train_df = df.loc[train_mask]
    test_df  = df.loc[test_mask]

    # ---- 3. Realized volatility (calibrate threshold on training) ----
    factor_vals = df[FACTOR_COLS].values
    daily_norm = np.linalg.norm(factor_vals, axis=1)
    realized_vol_20d = pd.Series(daily_norm, index=dates).rolling(20).std().values

    train_vol = realized_vol_20d[train_mask]
    vol_threshold = float(np.nanpercentile(train_vol, 95))
    print(f"\n  Vol threshold (95th pctile of training): {vol_threshold:.4f}")

    # ---- 4. Fit HMM (seed=28) on training data ----
    print(f"\nFitting Student-t HMM (K=3, seed={PRIMARY_SEED}) on training data...")
    X_train = train_df[FACTOR_COLS].values
    hmm = StudentTHMM(n_regimes=3, n_iter=200, tol=1e-5, random_state=PRIMARY_SEED)
    hmm.fit(X_train)
    print(f"  Train LL: {hmm.log_likelihood_:.2f}")

    # Relabeling order from TRAINING data only
    train_raw = hmm.predict(X_train, use_filtered=False)
    _, remap = relabel_regimes_by_data_norm(train_df, train_raw, FACTOR_COLS)
    train_relabeled = np.array([remap[r] for r in train_raw])
    regime_names = {0: 'Normal', 1: 'Elevated', 2: 'Crisis'}
    print(f"  Remap (raw→named): {remap}")
    for k in range(3):
        n = int((train_relabeled == k).sum())
        print(f"  Train {regime_names[k]}: {n} days ({n/len(train_relabeled)*100:.1f}%)")

    # ---- 5. Frozen OOS: filtered probabilities on full sample ----
    print("\nApplying frozen HMM to full sample (filtered probabilities)...")
    X_all = df[FACTOR_COLS].values
    regimes_raw_all, _ = hmm.predict_oos(X_all, use_filtered=True)
    regimes_all = np.array([remap[r] for r in regimes_raw_all])

    test_regimes = regimes_all[test_mask]
    for k in range(3):
        n = int((test_regimes == k).sum())
        print(f"  OOS {regime_names[k]}: {n} days ({n/len(test_regimes)*100:.1f}%)")

    # ---- 6. VaR models on test data ----
    test_smb       = smb[test_mask]
    test_hml_cumul = hml_cumul_9d[test_mask]
    test_vol       = realized_vol_20d[test_mask]
    test_dates     = dates[test_mask]

    var_uncond = rolling_historical_var(test_smb, window=60)

    var_regime = regime_conditional_var(test_smb, test_regimes)

    var_hml = hml_informed_var(test_smb, test_regimes, test_hml_cumul)

    var_hybrid, vol_override, eff_regimes = hybrid_var(
        test_smb, test_regimes, test_vol, test_hml_cumul, vol_threshold)

    # ---- 7. Evaluate ----
    print(f"\n{'='*70}")
    print("VaR BACKTEST RESULTS (2013-2024)")
    print(f"{'='*70}")

    evals = [
        evaluate_var(test_smb, var_uncond, "Unconditional (rolling-60)"),
        evaluate_var(test_smb, var_regime, "Regime-conditional"),
        evaluate_var(test_smb, var_hml,   "HML-Informed"),
        evaluate_var(test_smb, var_hybrid, "Hybrid (HMM+Vol)"),
    ]

    header = f"  {'Model':<30} {'Viol%':>6} {'Dev':>7} {'CC p':>8} {'UC p':>8}"
    print(f"\n{header}")
    print("  " + "-" * 63)
    for ev in evals:
        pcc = f"{ev['p_cc']:.3f}" if ev['p_cc'] is not None else "  N/A"
        puc = f"{ev['p_uc']:.3f}" if ev['p_uc'] is not None else "  N/A"
        print(f"  {ev['model']:<30} {ev['violation_rate_pct']:>5.2f}% {ev['deviation_pct']:>+6.2f}% "
              f"{pcc:>8} {puc:>8}")

    # ---- 8. COVID detection analysis ----
    print(f"\n{'='*70}")
    print("COVID DETECTION (2020-02-20 to 2020-06-30)")
    print(f"{'='*70}")
    covid_start = pd.Timestamp('2020-02-20')
    covid_end   = pd.Timestamp('2020-06-30')
    covid_mask  = (test_dates >= covid_start) & (test_dates <= covid_end)
    n_covid = int(covid_mask.sum())
    hmm_crisis   = int((test_regimes[covid_mask] == 2).sum())
    vol_override_covid = int(vol_override[covid_mask].sum())
    hybrid_crisis = int((eff_regimes[covid_mask] == 2).sum())
    print(f"  COVID window: {n_covid} trading days")
    print(f"  HMM-only Crisis days:     {hmm_crisis} ({hmm_crisis/n_covid*100:.1f}%)")
    print(f"  Vol override days:         {vol_override_covid} ({vol_override_covid/n_covid*100:.1f}%)")
    print(f"  Hybrid Crisis/Alert days:  {hybrid_crisis} ({hybrid_crisis/n_covid*100:.1f}%)")

    # ---- 9. Save results ----
    # Try to load seed-42 results for comparison
    seed42_hybrid = None
    try:
        with open(os.path.join(RESULTS_DIR, 'hybrid_detector_results.json'), 'r') as f:
            s42 = json.load(f)
        seed42_hybrid = s42.get('results', {}).get('Hybrid (HMM+Vol)', None)
    except Exception:
        pass

    results = {
        'description': f'VaR Backtest — Primary Fit (seed={PRIMARY_SEED}, 6 factors)',
        'seed': PRIMARY_SEED,
        'factor_cols': FACTOR_COLS,
        'train_period': f'1990-01-02 to {TRAIN_END}',
        'test_period': f'{TEST_START} to 2024-12-31',
        'calibrated_params': {
            'hml_threshold': HML_THRESHOLD,
            'stress_multiplier': STRESS_MULTIPLIER,
            'window_calm': WINDOW_CALM,
            'window_normal': WINDOW_NORMAL,
            'window_crisis': WINDOW_CRISIS,
            'vol_threshold_95pctile': vol_threshold,
        },
        'train_ll': float(hmm.log_likelihood_),
        'remap': [int(x) for x in remap],
        'oos_regime_counts': {regime_names[k]: int((test_regimes == k).sum()) for k in range(3)},
        'var_results': {ev['model']: ev for ev in evals},
        'covid_detection': {
            'n_days': n_covid,
            'hmm_crisis_days': hmm_crisis,
            'vol_override_days': vol_override_covid,
            'hybrid_crisis_days': hybrid_crisis,
        },
        'seed42_hybrid_for_comparison': seed42_hybrid,
    }

    out_path = os.path.join(RESULTS_DIR, 'var_backtest_seed28.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")

    # ---- 10. Summary ----
    hybrid_ev = evals[-1]
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Seed: {PRIMARY_SEED}  Factors: 6 (MOM included)")
    print(f"  Hybrid VaR violation rate: {hybrid_ev['violation_rate_pct']:.2f}%  "
          f"(target 5.00%,  CC p={hybrid_ev['p_cc']:.3f})")
    if seed42_hybrid:
        s42_vr = seed42_hybrid.get('violation_rate_pct', '?')
        s42_cc = seed42_hybrid.get('christoffersen_p_cc', '?')
        print(f"  Seed-42 Hybrid (reference): {s42_vr}%  CC p={s42_cc}")
    print("  Done.")


if __name__ == '__main__':
    main()
