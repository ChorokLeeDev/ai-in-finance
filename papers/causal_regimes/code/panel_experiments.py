"""
Panel Review Experiments: 4 New Tests for ICAIF 2026
=====================================================
1. Pre-break training test (HMM on 1990-1997 only)
2. OOS decay model test (fit on 1995-2012, predict 2013-2024)
3. Regime parameter stability LRT (train vs test period)
4. Fair baseline with identical HMM (same model for both)

Uses the SAME StudentTHMM and helpers from multistart_hmm_pipeline.py
to ensure regime assignments match the paper's primary analysis.
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, optimize
from datetime import datetime

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = BASE_DIR / 'code'
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# Import from the paper's actual codebase
sys.path.insert(0, str(CODE_DIR))
from multistart_hmm_pipeline import (
    StudentTHMM,
    download_ff_data,
    relabel_regimes_by_data_norm,
    relabel_hmm_params,
    extract_regime_clean_indices,
    run_granger_at_lag,
    granger_ftest,
)

PRIMARY_SEED = 28
REGIME_NAMES = ['Normal', 'Elevated', 'Crisis']
FACTOR_COLS = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
FIXED_LAG = 1  # BIC-optimal from primary analysis


# =============================================================================
# Helper: fit HMM on a period and return relabeled regimes
# =============================================================================
def fit_and_label(df, start, end, seed=PRIMARY_SEED):
    """Fit Student-t HMM on date range, relabel by data norm. Returns (hmm, regimes, period_df, features_z, mu, std, order)."""
    period = df.loc[start:end]
    feats = period[FACTOR_COLS].values
    mu_f = feats.mean(axis=0)
    std_f = feats.std(axis=0)
    feats_z = (feats - mu_f) / std_f

    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=seed)
    hmm.fit(feats_z)
    raw = hmm.predict(feats_z)
    regimes, order = relabel_regimes_by_data_norm(period, raw, FACTOR_COLS)
    return hmm, regimes, period, feats_z, mu_f, std_f, order


def granger_on_regime(df, regimes, regime_id, source='HML', target='SMB', lag=FIXED_LAG):
    """Run Granger test on a specific regime using clean indices."""
    y_all = df[target].values
    x_all = df[source].values
    clean_idx = extract_regime_clean_indices(regimes, regime_id, lag)
    if len(clean_idx) < 2 * lag + 10:
        return None
    return run_granger_at_lag(y_all, x_all, clean_idx, lag)


# =============================================================================
# EXPERIMENT 1: Pre-Break Training Test
# =============================================================================
def experiment_1_prebreak(df):
    """Train HMM on 1990-1997, decode 1998-2024, test HML→SMB Granger."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Pre-Break Training Test")
    print("="*70)

    # Train on 1990-1997
    hmm, train_regimes, train_df, train_z, mu_f, std_f, order = \
        fit_and_label(df, '1990-01-01', '1997-12-31')

    train_counts = np.bincount(train_regimes, minlength=3)
    print(f"  Train: {train_df.index[0].date()} to {train_df.index[-1].date()}")
    print(f"  Train regime counts: {dict(zip(REGIME_NAMES, train_counts.tolist()))}")

    # Granger on train Normal
    train_result = granger_on_regime(train_df, train_regimes, 0)
    if train_result:
        print(f"  Train Normal: F={train_result['f_stat']:.4f}, p={train_result['f_p_value']:.2e}, n={train_result['n_obs']}")
    else:
        print(f"  Train Normal: insufficient observations")

    # Decode 1998-2024 using FROZEN model (predict_oos)
    test_df = df.loc['1998-01-01':'2024-12-31']
    test_feats = test_df[FACTOR_COLS].values
    test_z = (test_feats - mu_f) / std_f  # TRAIN normalization
    raw_test, _ = hmm.predict_oos(test_z)
    # Apply same relabeling order from training
    test_regimes = np.zeros_like(raw_test)
    for new_k, old_k in enumerate(order):
        test_regimes[raw_test == old_k] = new_k

    test_counts = np.bincount(test_regimes, minlength=3)
    print(f"  Test: {test_df.index[0].date()} to {test_df.index[-1].date()}")
    print(f"  Test regime counts: {dict(zip(REGIME_NAMES, test_counts.tolist()))}")

    # Granger on test sub-periods
    results = {}
    results['train_1990_1997'] = {
        'n_normal': int(train_counts[0]),
        **(train_result if train_result else {'f_stat': None, 'f_p_value': None, 'n_obs': 0})
    }

    # Full test period
    test_result = granger_on_regime(test_df, test_regimes, 0)
    results['test_1998_2024'] = {
        'n_normal': int(test_counts[0]),
        **(test_result if test_result else {'f_stat': None, 'f_p_value': None, 'n_obs': 0})
    }
    if test_result:
        print(f"  Test 1998-2024 Normal: F={test_result['f_stat']:.4f}, p={test_result['f_p_value']:.2e}, n={test_result['n_obs']}")

    # Sub-periods: 1998-2007
    sub1 = test_df.loc[:'2007-12-31']
    sub1_idx = len(sub1)
    sub1_regimes = test_regimes[:sub1_idx]
    sub1_result = granger_on_regime(sub1, sub1_regimes, 0)
    results['test_1998_2007'] = {
        'n_normal': int((sub1_regimes == 0).sum()),
        **(sub1_result if sub1_result else {'f_stat': None, 'f_p_value': None, 'n_obs': 0})
    }
    if sub1_result:
        print(f"  Test 1998-2007 Normal: F={sub1_result['f_stat']:.4f}, p={sub1_result['f_p_value']:.2e}, n={sub1_result['n_obs']}")

    # Sub-periods: 2008-2024
    sub2 = test_df.loc['2008-01-01':]
    sub2_regimes = test_regimes[sub1_idx:]
    sub2_result = granger_on_regime(sub2, sub2_regimes, 0)
    results['test_2008_2024'] = {
        'n_normal': int((sub2_regimes == 0).sum()),
        **(sub2_result if sub2_result else {'f_stat': None, 'f_p_value': None, 'n_obs': 0})
    }
    if sub2_result:
        print(f"  Test 2008-2024 Normal: F={sub2_result['f_stat']:.4f}, p={sub2_result['f_p_value']:.2e}, n={sub2_result['n_obs']}")

    output = {
        'experiment': 'Pre-Break Training Test',
        'methodology': 'Train Student-t HMM (K=3, seed=28) on 1990-1997 ONLY using '
                       'multistart_hmm_pipeline.StudentTHMM. Freeze model, decode '
                       '1998-2024 with predict_oos. Run Granger HML→SMB in Normal regime.',
        'train_period': '1990-01-01 to 1997-12-31',
        'test_period': '1998-01-01 to 2024-12-31',
        'train_regime_counts': {REGIME_NAMES[i]: int(c) for i, c in enumerate(train_counts)},
        'test_regime_counts': {REGIME_NAMES[i]: int(c) for i, c in enumerate(test_counts)},
        'granger_results': results,
        'timestamp': datetime.now().isoformat()
    }
    with open(RESULTS_DIR / 'pre_break_training_test.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print("  Saved: pre_break_training_test.json")
    return output


# =============================================================================
# EXPERIMENT 2: OOS Decay Model Test
# =============================================================================
def experiment_2_oos_decay(df):
    """Fit decay on windows ending ≤2012, predict 2013-2024 windows."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: OOS Decay Model Test")
    print("="*70)

    # Use canonical full-sample HMM for consistent regime assignments
    hmm, regimes, _, _, _, _, _ = fit_and_label(df, '1990-01-01', '2024-12-31')

    counts = np.bincount(regimes, minlength=3)
    print(f"  Canonical regime counts: {dict(zip(REGIME_NAMES, counts.tolist()))}")

    smb = df['SMB'].values
    hml = df['HML'].values
    dates = df.index

    # Rolling 5-year windows, 1-year step
    window_days = 252 * 5
    step_days = 252
    rolling = []

    for start in range(0, len(df) - window_days + 1, step_days):
        end = start + window_days
        w_regimes = regimes[start:end]
        w_dates = dates[start:end]
        clean_idx = extract_regime_clean_indices(w_regimes, 0, FIXED_LAG)  # Normal=0
        if len(clean_idx) < 20:
            continue
        y_w = smb[start:end]
        x_w = hml[start:end]
        result = run_granger_at_lag(y_w, x_w, clean_idx, FIXED_LAG)
        if result is None:
            continue
        end_year = w_dates[-1].year + w_dates[-1].month / 12.0
        rolling.append({
            'end_year': round(end_year, 2),
            'end_date': str(w_dates[-1].date()),
            'F': result['f_stat'],
            'p': result['f_p_value'],
            'n_normal': len(clean_idx),
            'hac_p': result['hac_p_value'],
        })
        print(f"  Window ending {w_dates[-1].date()}: Normal n={len(clean_idx)}, F={result['f_stat']:.2f}, p={result['f_p_value']:.2e}")

    # Split: training (ending ≤ 2013.0) and test (ending > 2013.0)
    train_w = [r for r in rolling if r['end_year'] <= 2013.0]
    test_w = [r for r in rolling if r['end_year'] > 2013.0]
    print(f"\n  Training windows (≤2012): {len(train_w)}")
    print(f"  Test windows (>2012): {len(test_w)}")

    if len(train_w) < 3:
        print("  ERROR: Too few training windows for decay fit")
        output = {
            'experiment': 'OOS Decay Model Test',
            'error': f'Only {len(train_w)} training windows (need ≥3)',
            'all_rolling_windows': rolling,
            'timestamp': datetime.now().isoformat()
        }
        with open(RESULTS_DIR / 'oos_decay_model_test.json', 'w') as f:
            json.dump(output, f, indent=2)
        return output

    t0 = train_w[0]['end_year']
    t_train = np.array([r['end_year'] - t0 for r in train_w])
    F_train = np.array([r['F'] for r in train_w])

    # Fit F(t) = A * exp(-lambda * t) on training windows
    valid = (F_train > 0) & np.isfinite(F_train)
    try:
        popt, pcov = optimize.curve_fit(
            lambda t, A, lam: A * np.exp(-lam * t),
            t_train[valid], F_train[valid],
            p0=[max(F_train[valid]), 0.2], maxfev=5000,
            bounds=([0, 0.001], [500, 5])
        )
        A_fit, lam_fit = popt
        half_life = np.log(2) / lam_fit
        pred_train = A_fit * np.exp(-lam_fit * t_train[valid])
        ss_res = np.sum((F_train[valid] - pred_train)**2)
        ss_tot = np.sum((F_train[valid] - F_train[valid].mean())**2)
        R2_train = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"  Decay fit: A={A_fit:.2f}, λ={lam_fit:.4f}, half-life={half_life:.2f}yr, R²={R2_train:.3f}")

        # OOS predictions
        predictions = []
        if test_w:
            t_test = np.array([r['end_year'] - t0 for r in test_w])
            F_actual = np.array([r['F'] for r in test_w])
            F_pred = A_fit * np.exp(-lam_fit * t_test)
            mse = float(np.mean((F_actual - F_pred)**2))
            mae = float(np.mean(np.abs(F_actual - F_pred)))
            corr = float(np.corrcoef(F_actual, F_pred)[0, 1]) if len(F_actual) > 2 else None
            for i, tw in enumerate(test_w):
                predictions.append({
                    'end_date': tw['end_date'],
                    'actual_F': round(tw['F'], 4),
                    'predicted_F': round(float(F_pred[i]), 4),
                })
            print(f"  OOS: MSE={mse:.4f}, MAE={mae:.4f}, corr={corr}")
            print(f"  Mean actual F: {F_actual.mean():.4f}, Mean predicted F: {F_pred.mean():.4f}")
        else:
            mse = mae = corr = None
    except Exception as e:
        print(f"  Curve fit error: {e}")
        A_fit = lam_fit = half_life = R2_train = mse = mae = corr = None
        predictions = []

    output = {
        'experiment': 'OOS Decay Model Test',
        'methodology': 'Fit F(t)=A·exp(-λt) on rolling 5-year Normal-regime Granger F-stats '
                       'for windows ending ≤2012. Predict F-stats for windows ending >2012.',
        'decay_parameters': {
            'A': round(float(A_fit), 4) if A_fit else None,
            'lambda': round(float(lam_fit), 4) if lam_fit else None,
            'half_life_years': round(float(half_life), 2) if half_life else None,
            'R2_train': round(float(R2_train), 3) if R2_train else None
        },
        'oos_metrics': {
            'MSE': round(float(mse), 4) if mse is not None else None,
            'MAE': round(float(mae), 4) if mae is not None else None,
            'correlation': round(float(corr), 4) if corr is not None else None,
            'n_train_windows': len(train_w),
            'n_test_windows': len(test_w)
        },
        'predictions': predictions,
        'all_rolling_windows': rolling,
        'timestamp': datetime.now().isoformat()
    }
    with open(RESULTS_DIR / 'oos_decay_model_test.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("  Saved: oos_decay_model_test.json")
    return output


# =============================================================================
# EXPERIMENT 3: Regime Parameter Stability LRT
# =============================================================================
def experiment_3_regime_stability(df):
    """Compare HMM parameters between 1990-2012 and 2013-2024 via LRT."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Regime Parameter Stability LRT")
    print("="*70)

    # Use GLOBAL normalization for fair comparison
    feats_all = df[FACTOR_COLS].values
    mu_g = feats_all.mean(axis=0)
    std_g = feats_all.std(axis=0)
    z_all = (feats_all - mu_g) / std_g

    # Full model
    hmm_full = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm_full.fit(z_all)

    # Split
    train_end = df.index.searchsorted(pd.Timestamp('2013-01-01'))
    z_train = z_all[:train_end]
    z_test = z_all[train_end:]
    df_train = df.iloc[:train_end]
    df_test = df.iloc[train_end:]

    # Separate models
    hmm_train = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm_train.fit(z_train)

    hmm_test = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm_test.fit(z_test)

    # Log-likelihoods
    ll_full_train = float(hmm_full.log_likelihood_) * len(z_train) / len(z_all)  # approximate
    # Better: recompute directly
    # Score = emission log-probs under frozen params
    def score_model(hmm, X):
        """Total log-likelihood of X under hmm (forward algorithm)."""
        log_B = hmm._compute_emission_probs(X)
        log_alpha = hmm._forward(log_B)
        return float(np.logaddexp.reduce(log_alpha[-1]))

    ll_pooled_train = score_model(hmm_full, z_train)
    ll_pooled_test = score_model(hmm_full, z_test)
    ll_sep_train = score_model(hmm_train, z_train)
    ll_sep_test = score_model(hmm_test, z_test)

    lrt_stat = -2.0 * ((ll_pooled_train + ll_pooled_test) - (ll_sep_train + ll_sep_test))

    K, d = 3, 5
    # Free params per model: K*d (means) + K*d*(d+1)/2 (cov) + K*(K-1) (transitions) + K (nu) + (K-1) (pi)
    n_free = K * d + K * d * (d + 1) // 2 + K * (K - 1) + K + (K - 1)
    # df = extra params in separate model
    df_lrt = n_free

    lrt_p = 1.0 - stats.chi2.cdf(max(lrt_stat, 0), df_lrt) if lrt_stat > 0 else 1.0

    print(f"  LL pooled (train): {ll_pooled_train:.2f}")
    print(f"  LL pooled (test):  {ll_pooled_test:.2f}")
    print(f"  LL separate train: {ll_sep_train:.2f}")
    print(f"  LL separate test:  {ll_sep_test:.2f}")
    print(f"  LRT stat: {lrt_stat:.2f}, df={df_lrt}, p={lrt_p:.2e}")

    # Parameter comparisons (relabel by data norm for fair comparison)
    reg_train, ord_train = relabel_regimes_by_data_norm(df_train, hmm_train.predict(z_train), FACTOR_COLS)
    reg_test, ord_test = relabel_regimes_by_data_norm(df_test, hmm_test.predict(z_test), FACTOR_COLS)

    # Reorder params
    mu_train = hmm_train.mu[ord_train]
    mu_test = hmm_test.mu[ord_test]
    nu_train = hmm_train.nu[ord_train]
    nu_test = hmm_test.nu[ord_test]
    A_train = hmm_train.A[ord_train][:, ord_train]
    A_test = hmm_test.A[ord_test][:, ord_test]

    mean_dist = {REGIME_NAMES[k]: round(float(np.linalg.norm(mu_train[k] - mu_test[k])), 4) for k in range(K)}
    nu_diff = {REGIME_NAMES[k]: round(float(abs(nu_train[k] - nu_test[k])), 2) for k in range(K)}
    A_frob = round(float(np.linalg.norm(A_train - A_test, 'fro')), 4)

    print(f"  Mean distances: {mean_dist}")
    print(f"  ν differences: {nu_diff}")
    print(f"  Transition Frobenius: {A_frob}")

    # Regime distributions
    tc = np.bincount(reg_train, minlength=3)
    ec = np.bincount(reg_test, minlength=3)

    output = {
        'experiment': 'Regime Parameter Stability LRT',
        'methodology': 'Fit Student-t HMM (pipeline implementation) separately on 1990-2012 '
                       'and 2013-2024 with global normalization. LRT against pooled model.',
        'log_likelihoods': {
            'pooled_on_train': round(ll_pooled_train, 2),
            'pooled_on_test': round(ll_pooled_test, 2),
            'separate_train': round(ll_sep_train, 2),
            'separate_test': round(ll_sep_test, 2),
        },
        'lrt': {
            'statistic': round(float(lrt_stat), 4),
            'degrees_of_freedom': df_lrt,
            'p_value': f"{lrt_p:.6e}"
        },
        'parameter_distances': {
            'mean_euclidean': mean_dist,
            'nu_differences': nu_diff,
            'transition_frobenius': A_frob,
        },
        'regime_distributions': {
            'train': {REGIME_NAMES[i]: int(c) for i, c in enumerate(tc)},
            'test': {REGIME_NAMES[i]: int(c) for i, c in enumerate(ec)},
        },
        'timestamp': datetime.now().isoformat()
    }
    with open(RESULTS_DIR / 'regime_stability_lrt.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("  Saved: regime_stability_lrt.json")
    return output


# =============================================================================
# EXPERIMENT 4: Fair Baseline with Identical HMM
# =============================================================================
def experiment_4_fair_baseline(df):
    """Same HMM for regime-conditional vs unconditional Granger."""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Fair Baseline with Identical HMM")
    print("="*70)

    # Fit canonical HMM (same as paper's primary)
    hmm, regimes, _, _, _, _, _ = fit_and_label(df, '1990-01-01', '2024-12-31')

    counts = np.bincount(regimes, minlength=3)
    print(f"  Regime counts: {dict(zip(REGIME_NAMES, counts.tolist()))}")

    smb = df['SMB'].values
    hml = df['HML'].values
    results = {}

    # (a) Regime-conditional: each regime
    for k in range(3):
        clean_idx = extract_regime_clean_indices(regimes, k, FIXED_LAG)
        if len(clean_idx) < 20:
            results[f'regime_{REGIME_NAMES[k]}'] = {'n_clean': len(clean_idx), 'note': 'too few obs'}
            continue
        res = run_granger_at_lag(smb, hml, clean_idx, FIXED_LAG)
        results[f'regime_{REGIME_NAMES[k]}'] = res if res else {'n_clean': len(clean_idx)}
        if res:
            print(f"  {REGIME_NAMES[k]}: F={res['f_stat']:.4f}, p={res['f_p_value']:.2e}, HAC-p={res['hac_p_value']:.2e}, n={res['n_obs']}")

    # (b) Unconditional: full sample
    all_idx = np.arange(FIXED_LAG, len(smb))
    y_curr = smb[all_idx]
    y_lag = np.column_stack([smb[all_idx - i - 1] for i in range(FIXED_LAG)])
    x_lag = np.column_stack([hml[all_idx - i - 1] for i in range(FIXED_LAG)])
    f_stat, f_p, delta_r2, r2_u = granger_ftest(y_curr, y_lag, x_lag)
    results['unconditional'] = {
        'n_obs': len(all_idx), 'f_stat': float(f_stat), 'f_p_value': float(f_p),
        'delta_r2': float(delta_r2), 'r2_unrestricted': float(r2_u)
    }
    print(f"  Unconditional: F={f_stat:.4f}, p={f_p:.2e}, n={len(all_idx)}")

    # F-ratio
    normal_res = results.get('regime_Normal', {})
    normal_F = normal_res.get('f_stat', 0)
    if f_stat > 0 and normal_F:
        f_ratio = normal_F / f_stat
        print(f"  F-ratio (Normal/Unconditional): {f_ratio:.2f}x")
    else:
        f_ratio = None

    output = {
        'experiment': 'Fair Baseline with Identical HMM',
        'methodology': 'Fit SAME StudentTHMM (seed=28, K=3, pipeline implementation) for '
                       'regime-conditional AND unconditional Granger. Eliminates implementation mismatch.',
        'regime_counts': {REGIME_NAMES[i]: int(c) for i, c in enumerate(counts)},
        'granger_results': results,
        'f_ratio_normal_vs_unconditional': round(float(f_ratio), 4) if f_ratio else None,
        'timestamp': datetime.now().isoformat()
    }
    with open(RESULTS_DIR / 'fair_baseline_identical_hmm.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print("  Saved: fair_baseline_identical_hmm.json")
    return output


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("Loading Fama-French factor data...")
    df = download_ff_data()

    r1 = experiment_1_prebreak(df)
    r2 = experiment_2_oos_decay(df)
    r3 = experiment_3_regime_stability(df)
    r4 = experiment_4_fair_baseline(df)

    print("\n" + "="*70)
    print("ALL 4 EXPERIMENTS COMPLETE")
    print("="*70)
    for name, r in [('Exp1', r1), ('Exp2', r2), ('Exp3', r3), ('Exp4', r4)]:
        if r:
            print(f"  {name}: {r.get('experiment', 'done')}")
