"""
Trivariate Granger Robustness Check — HML -> SMB | (SMB lags, MKT-RF lags)
===========================================================================
Uses the EXACT same frozen OOS regime assignments as frozen_oos_primary.py
(seed 28, n=836 Elevated clean) to test whether the HML->SMB finding survives
after conditioning on MKT-RF as a potential common driver.

Bivariate (primary):   SMB_t = a + b*SMB_{t-1} + c*HML_{t-1} + e
Trivariate (this):     SMB_t = a + b*SMB_{t-1} + c*MKT_{t-1} + d*HML_{t-1} + e
F-test H0: d = 0 (HML has no incremental content beyond SMB + MKT histories)

Also tests: MKT -> SMB | SMB (bivariate) to confirm MKT is not itself the
primary driver, and reports the full trivariate model for all factor pairs.
"""
import sys, json, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/code')
from multistart_hmm_pipeline import (
    download_ff_data,
    StudentTHMM,
    relabel_regimes_by_data_norm,
    extract_regime_clean_indices,
)

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results'
PRIMARY_SEED = 28
FIXED_LAG = 1
REGIME_NAMES = ['Normal', 'Elevated', 'Crisis']


def run_trivariate_granger(y, x1_lags, x2_lags, y_lags, clean_idx, lag=1):
    """
    Trivariate Granger: does x2 (HML) help predict y (SMB) beyond x1 (MKT) + y lags?
    y_lags: lagged values of y (SMB)
    x1_lags: lagged values of conditioning variable (MKT-RF)
    x2_lags: lagged values of test variable (HML)
    Returns dict with F-stat, F-p, HAC-p, delta_R2, n_obs.
    """
    usable = clean_idx[clean_idx >= lag]
    n = len(usable)
    if n < 3 * lag + 15:
        return None

    Y  = y[usable]
    YL = np.column_stack([y_lags[usable - i - 1] for i in range(lag)])
    X1L = np.column_stack([x1_lags[usable - i - 1] for i in range(lag)])
    X2L = np.column_stack([x2_lags[usable - i - 1] for i in range(lag)])

    # Restricted: SMB ~ const + SMB_lag + MKT_lag
    Xr = np.column_stack([np.ones(n), YL, X1L])
    # Unrestricted: SMB ~ const + SMB_lag + MKT_lag + HML_lag
    Xu = np.column_stack([np.ones(n), YL, X1L, X2L])

    br = np.linalg.lstsq(Xr, Y, rcond=None)[0]
    bu = np.linalg.lstsq(Xu, Y, rcond=None)[0]
    rr = float(np.sum((Y - Xr @ br) ** 2))
    ru = float(np.sum((Y - Xu @ bu) ** 2))

    k_r, k_u = Xr.shape[1], Xu.shape[1]
    df1 = k_u - k_r          # = lag (number of HML lags added)
    df2 = n - k_u             # residual df
    if df2 <= 0 or ru <= 0:
        return None

    F = ((rr - ru) / df1) / (ru / df2)
    from scipy import stats
    f_p = float(1 - stats.f.cdf(F, df1, df2))

    # HAC (Newey-West, bandwidth=lag) on unrestricted model
    res_u = Y - Xu @ bu
    Xu_df = pd.DataFrame(Xu)
    Y_s   = pd.Series(Y)
    try:
        ols_fit = sm.OLS(Y_s, Xu_df).fit()
        hac_fit = ols_fit.get_robustcov_results(cov_type='HAC', maxlags=lag)
        # Test last `lag` coefficients (HML lags) = 0
        hml_indices = list(range(k_u - lag, k_u))
        r_mat = np.zeros((lag, k_u))
        for i, idx in enumerate(hml_indices):
            r_mat[i, idx] = 1.0
        wald = hac_fit.wald_test(r_mat, use_f=True)
        hac_p = float(wald.pvalue)
    except Exception:
        hac_p = float('nan')

    # R2 improvement
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2_r = 1 - rr / ss_tot if ss_tot > 0 else 0
    r2_u = 1 - ru / ss_tot if ss_tot > 0 else 0
    delta_r2 = r2_u - r2_r

    return {
        'n_obs': n,
        'lag': lag,
        'f_stat': float(F),
        'f_p_value': f_p,
        'hac_p_value': hac_p,
        'delta_r2': delta_r2,
        'r2_restricted': r2_r,
        'r2_unrestricted': r2_u,
    }


def run_bivariate_granger(y, x_lags, y_lags, clean_idx, lag=1):
    """Bivariate: does x help predict y beyond y's own lags? (for comparison)"""
    usable = clean_idx[clean_idx >= lag]
    n = len(usable)
    if n < 2 * lag + 10:
        return None

    Y  = y[usable]
    YL = np.column_stack([y_lags[usable - i - 1] for i in range(lag)])
    XL = np.column_stack([x_lags[usable - i - 1] for i in range(lag)])

    Xr = np.column_stack([np.ones(n), YL])
    Xu = np.column_stack([np.ones(n), YL, XL])

    br = np.linalg.lstsq(Xr, Y, rcond=None)[0]
    bu = np.linalg.lstsq(Xu, Y, rcond=None)[0]
    rr = float(np.sum((Y - Xr @ br) ** 2))
    ru = float(np.sum((Y - Xu @ bu) ** 2))

    df1 = lag
    df2 = n - 2 * lag - 1
    if df2 <= 0 or ru <= 0:
        return None

    from scipy import stats
    F = ((rr - ru) / df1) / (ru / df2)
    f_p = float(1 - stats.f.cdf(F, df1, df2))

    res_u = Y - Xu @ bu
    try:
        ols_fit = sm.OLS(pd.Series(Y), pd.DataFrame(Xu)).fit()
        hac_fit = ols_fit.get_robustcov_results(cov_type='HAC', maxlags=lag)
        k_u = Xu.shape[1]
        r_mat = np.zeros((lag, k_u))
        for i in range(lag):
            r_mat[i, k_u - lag + i] = 1.0
        wald = hac_fit.wald_test(r_mat, use_f=True)
        hac_p = float(wald.pvalue)
    except Exception:
        hac_p = float('nan')

    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    delta_r2 = (rr - ru) / ss_tot if ss_tot > 0 else 0

    return {
        'n_obs': n, 'lag': lag,
        'f_stat': float(F), 'f_p_value': f_p,
        'hac_p_value': hac_p, 'delta_r2': delta_r2,
    }


def main():
    print("Loading Fama-French data...")
    df = download_ff_data()
    df = df / 100.0
    print(f"  Full: {df.index[0].date()} to {df.index[-1].date()}, n={len(df)}")

    train_df = df.loc[:'2012-12-31']
    test_df  = df.loc['2013-01-01':]
    factor_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    # Fit primary HMM (exact same as frozen_oos_primary.py, seed 28)
    print(f"\nFitting frozen HMM (seed={PRIMARY_SEED}, train 1990-2012)...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm.fit(train_df[factor_cols].values)
    print(f"  Train LL: {hmm.log_likelihood_:.2f}")

    # Relabeling from train only
    train_raw = hmm.predict(train_df[factor_cols].values, use_filtered=False)
    _, remap = relabel_regimes_by_data_norm(train_df, train_raw, factor_cols)

    # Apply to test with filtered labels (no future info)
    test_raw, _ = hmm.predict_oos(test_df[factor_cols].values, use_filtered=True)
    test_regimes = np.array([remap[r] for r in test_raw])
    test_counts = {REGIME_NAMES[k]: int((test_regimes == k).sum()) for k in range(3)}
    print(f"  Test regime counts: {test_counts}")

    smb = test_df['SMB'].values
    hml = test_df['HML'].values
    mkt = test_df['MKT'].values

    results = {}
    print("\n" + "="*70)
    print("TRIVARIATE GRANGER: HML -> SMB | (SMB_lag, MKT_lag)")
    print("="*70)
    print(f"{'Regime':<12} {'n_clean':>8} {'F-stat':>8} {'F-p':>10} {'HAC-p':>10} {'dR2':>8} {'sig'}")
    print("-"*70)

    for k, name in enumerate(REGIME_NAMES):
        clean = extract_regime_clean_indices(test_regimes, k, max_lag=FIXED_LAG)

        # Bivariate baseline (replicating primary result)
        biv = run_bivariate_granger(smb, hml, smb, clean, lag=FIXED_LAG)

        # Trivariate: HML -> SMB | SMB_lag + MKT_lag
        triv = run_trivariate_granger(smb, mkt, hml, smb, clean, lag=FIXED_LAG)

        # MKT -> SMB | SMB_lag (to understand MKT's own predictive role)
        mkt_biv = run_bivariate_granger(smb, mkt, smb, clean, lag=FIXED_LAG)

        results[name] = {
            'bivariate_hml_to_smb': biv,
            'trivariate_hml_to_smb_given_mkt': triv,
            'bivariate_mkt_to_smb': mkt_biv,
        }

        if triv:
            sig = ('***' if triv['f_p_value'] < 0.001 else
                   '**'  if triv['f_p_value'] < 0.01  else
                   '*'   if triv['f_p_value'] < 0.05  else
                   '†'   if triv['f_p_value'] < 0.10  else 'n.s.')
            print(f"{name:<12} {triv['n_obs']:>8} {triv['f_stat']:>8.3f} "
                  f"{triv['f_p_value']:>10.4f} {triv['hac_p_value']:>10.4f} "
                  f"{triv['delta_r2']:>8.4f} {sig}")
        else:
            print(f"{name:<12}  (insufficient observations)")

    print("\n" + "="*70)
    print("BIVARIATE COMPARISON (replicating primary result)")
    print("="*70)
    print(f"{'Regime':<12} {'n_clean':>8} {'F-p (biv)':>12} {'HAC-p (biv)':>13} {'F-p (triv)':>12} {'HAC-p (triv)':>13}")
    print("-"*70)
    for name in REGIME_NAMES:
        biv  = results[name].get('bivariate_hml_to_smb')
        triv = results[name].get('trivariate_hml_to_smb_given_mkt')
        if biv and triv:
            print(f"{name:<12} {biv['n_obs']:>8} "
                  f"{biv['f_p_value']:>12.4f} {biv['hac_p_value']:>13.4f} "
                  f"{triv['f_p_value']:>12.4f} {triv['hac_p_value']:>13.4f}")

    print("\n" + "="*70)
    print("MKT -> SMB | SMB_lag (bivariate, for reference)")
    print("="*70)
    for name in REGIME_NAMES:
        mkt_res = results[name].get('bivariate_mkt_to_smb')
        if mkt_res:
            sig = ('***' if mkt_res['f_p_value'] < 0.001 else
                   '*'   if mkt_res['f_p_value'] < 0.05  else 'n.s.')
            print(f"  {name}: F-p={mkt_res['f_p_value']:.4f} HAC-p={mkt_res['hac_p_value']:.4f} {sig}")

    # Save
    outpath = f"{RESULTS_DIR}/trivariate_granger.json"
    output = {
        'description': (
            'Trivariate Granger: HML->SMB | (SMB_lag, MKT_lag). '
            'Uses exact same frozen OOS regime assignments as primary (seed=28, filtered). '
            f'Lag={FIXED_LAG}. Tests omitted-variable robustness of the bivariate HML->SMB finding.'
        ),
        'primary_seed': PRIMARY_SEED,
        'fixed_lag': FIXED_LAG,
        'test_regime_counts': test_counts,
        'results': results,
    }
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {outpath}")

    # Key conclusion
    elev_triv = results['Elevated']['trivariate_hml_to_smb_given_mkt']
    elev_biv  = results['Elevated']['bivariate_hml_to_smb']
    if elev_triv and elev_biv:
        print("\n" + "="*70)
        print("KEY RESULT: Elevated regime HML->SMB")
        print(f"  Bivariate:   F-p={elev_biv['f_p_value']:.4f}  HAC-p={elev_biv['hac_p_value']:.4f}  dR2={elev_biv['delta_r2']:.4f}")
        print(f"  Trivariate:  F-p={elev_triv['f_p_value']:.4f}  HAC-p={elev_triv['hac_p_value']:.4f}  dR2={elev_triv['delta_r2']:.4f}")
        if elev_triv['f_p_value'] < 0.05:
            print("  VERDICT: HML->SMB SURVIVES conditioning on MKT-RF ✓")
        elif elev_triv['f_p_value'] < 0.10:
            print("  VERDICT: HML->SMB MARGINALLY SURVIVES (0.05 < p < 0.10) ~")
        else:
            print("  VERDICT: HML->SMB does NOT survive conditioning on MKT-RF ✗")
        print("="*70)


if __name__ == '__main__':
    main()
