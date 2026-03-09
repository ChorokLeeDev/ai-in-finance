"""
Extended VaR Analysis with Multiple Specifications
====================================================

Goal: Find VaR specifications where regime-conditional models achieve
statistically significant improvement (Christoffersen p < 0.05)

Analysis Design:
- Multiple confidence levels: 1%, 2.5%, 5%, 10%
- Multiple assets: SMB, HML, MOM
- Multiple models: Unconditional, GARCH(1,1), Regime-Conditional, Hybrid
- Backtest period: 2010-2024 (extended)
- Tests: Kupiec (unconditional), Christoffersen (conditional coverage)
- Expected Shortfall comparison

Output: results/var_extended.json
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from scipy import stats
from scipy.special import gammaln
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import arch package for GARCH
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    print("Installing arch package...")
    os.system('pip install --break-system-packages arch')
    from arch import arch_model
    HAS_ARCH = True

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
TRAIN_END = '2009-12-31'  # Extended backtest: 2010-2024
TEST_START = '2010-01-01'
PRIMARY_SEED = 28

# Multiple confidence levels
ALPHA_LEVELS = [0.01, 0.025, 0.05, 0.10]

# Assets to test
ASSET_COLS = ['SMB', 'HML', 'MOM']


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def kupiec_test(violations: np.ndarray, alpha: float) -> Dict:
    """
    Kupiec (1995) unconditional coverage test.
    H0: violation rate = alpha
    """
    violations = np.array(violations, dtype=int)
    T = len(violations)
    n = violations.sum()

    if n == 0 or n == T:
        return {'LR': np.nan, 'p_value': np.nan}

    pi_hat = n / T
    LR = 2 * (n * np.log(pi_hat / alpha) + (T - n) * np.log((1 - pi_hat) / (1 - alpha)))
    p_value = 1 - stats.chi2.cdf(LR, 1)

    return {'LR': float(LR), 'p_value': float(p_value)}


def christoffersen_test(violations: np.ndarray, alpha: float) -> Dict:
    """
    Christoffersen (1998) conditional coverage test.
    Tests: (1) unconditional coverage + (2) independence of violations
    """
    violations = np.array(violations, dtype=int)
    T = len(violations)
    n1 = violations.sum()
    n0 = T - n1

    # Unconditional coverage
    if n1 == 0 or n0 == 0:
        LR_uc = np.nan
        p_uc = np.nan
    else:
        pi_hat = n1 / T
        LR_uc = 2 * (n1 * np.log(pi_hat / alpha) + n0 * np.log((1 - pi_hat) / (1 - alpha)))
        p_uc = 1 - stats.chi2.cdf(LR_uc, 1)

    # Independence test (transition counts)
    n00 = n01 = n10 = n11 = 0
    for t in range(1, T):
        if violations[t-1] == 0 and violations[t] == 0:
            n00 += 1
        elif violations[t-1] == 0 and violations[t] == 1:
            n01 += 1
        elif violations[t-1] == 1 and violations[t] == 0:
            n10 += 1
        else:
            n11 += 1

    if (n00 + n01) == 0 or (n10 + n11) == 0 or n01 == 0 or n10 == 0:
        LR_ind = np.nan
        p_ind = np.nan
    else:
        pi01 = n01 / (n00 + n01)
        pi11 = n11 / (n10 + n11)
        pi_hat2 = (n01 + n11) / (n00 + n01 + n10 + n11)

        if any(v <= 0 or v >= 1 for v in [pi01, pi11, pi_hat2]):
            LR_ind = np.nan
            p_ind = np.nan
        else:
            LR_ind = -2 * (
                (n00 + n10) * np.log(1 - pi_hat2) + (n01 + n11) * np.log(pi_hat2)
                - n00 * np.log(1 - pi01) - n01 * np.log(pi01)
                - n10 * np.log(1 - pi11) - n11 * np.log(pi11)
            )
            p_ind = 1 - stats.chi2.cdf(LR_ind, 1)

    # Conditional coverage (joint test)
    if np.isnan(LR_uc) and np.isnan(LR_ind):
        LR_cc = np.nan
        p_cc = np.nan
    else:
        LR_cc = (LR_uc if not np.isnan(LR_uc) else 0) + (LR_ind if not np.isnan(LR_ind) else 0)
        p_cc = 1 - stats.chi2.cdf(LR_cc, 2)

    return {
        'LR_uc': float(LR_uc) if not np.isnan(LR_uc) else None,
        'p_uc': float(p_uc) if not np.isnan(p_uc) else None,
        'LR_ind': float(LR_ind) if not np.isnan(LR_ind) else None,
        'p_ind': float(p_ind) if not np.isnan(p_ind) else None,
        'LR_cc': float(LR_cc) if not np.isnan(LR_cc) else None,
        'p_cc': float(p_cc) if not np.isnan(p_cc) else None,
    }


# =============================================================================
# VAR MODELS
# =============================================================================

def unconditional_historical_var(returns: np.ndarray, alpha: float, window: int = 252) -> np.ndarray:
    """
    Simple rolling historical VaR (alpha-quantile of past window returns).
    """
    T = len(returns)
    var_est = np.full(T, np.nan)
    for t in range(window, T):
        var_est[t] = np.percentile(returns[t-window:t], alpha * 100)
    return var_est


def garch_var(returns: np.ndarray, alpha: float, min_window: int = 252) -> Tuple[np.ndarray, np.ndarray]:
    """
    GARCH(1,1) VaR with expanding window estimation.
    Returns VaR estimates and conditional volatilities.
    """
    T = len(returns)
    var_est = np.full(T, np.nan)
    cond_vol = np.full(T, np.nan)
    z_alpha = stats.norm.ppf(alpha)

    for t in range(min_window, T):
        if t % 500 == 0:
            print(f"    GARCH fitting day {t}/{T}...")
        try:
            model = arch_model(returns[:t] * 100, vol='Garch', p=1, q=1, mean='Constant')
            res = model.fit(disp='off', show_warning=False)
            sigma = res.conditional_volatility.iloc[-1] / 100  # Back to decimal
            cond_vol[t] = sigma
            var_est[t] = z_alpha * sigma
        except Exception:
            # Fallback to rolling std
            if t >= 60:
                sigma = np.std(returns[t-60:t])
                cond_vol[t] = sigma
                var_est[t] = z_alpha * sigma

    return var_est, cond_vol


def regime_conditional_var(
    returns: np.ndarray,
    regimes: np.ndarray,
    alpha: float,
    windows: Dict[int, int] = None
) -> np.ndarray:
    """
    Regime-conditional historical VaR.
    Uses regime-specific lookback windows.
    """
    if windows is None:
        windows = {0: 75, 1: 50, 2: 30}  # Normal, Elevated, Crisis

    T = len(returns)
    var_est = np.full(T, np.nan)
    max_w = max(windows.values())

    for t in range(max_w, T):
        regime = int(regimes[t-1])  # Use previous day's regime
        w = windows.get(regime, 50)
        var_est[t] = np.percentile(returns[max(0, t-w):t], alpha * 100)

    return var_est


def hybrid_regime_signal_var(
    returns: np.ndarray,
    regimes: np.ndarray,
    signal: np.ndarray,  # e.g., HML for SMB
    alpha: float,
    signal_threshold_pctile: float = 5,  # Extreme signal percentile
    stress_multiplier: float = 1.5,
    windows: Dict[int, int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Regime-conditional VaR with cross-factor signal enhancement.
    When in non-crisis regime but signal is extreme, tighten VaR.
    """
    if windows is None:
        windows = {0: 75, 1: 50, 2: 30}

    T = len(returns)
    var_est = np.full(T, np.nan)
    adjustments = np.zeros(T, dtype=bool)
    max_w = max(windows.values())

    # Compute signal threshold from first window
    signal_thresh = np.nanpercentile(signal[:max_w*2], signal_threshold_pctile)

    for t in range(max_w, T):
        regime = int(regimes[t-1])
        w = windows.get(regime, 50)
        base_var = np.percentile(returns[max(0, t-w):t], alpha * 100)

        # If in Normal/Elevated but signal is extremely negative, tighten VaR
        if regime in [0, 1] and not np.isnan(signal[t-1]) and signal[t-1] < signal_thresh:
            var_est[t] = base_var * stress_multiplier
            adjustments[t] = True
        else:
            var_est[t] = base_var

    return var_est, adjustments


# =============================================================================
# EXPECTED SHORTFALL
# =============================================================================

def compute_expected_shortfall(returns: np.ndarray, var_est: np.ndarray) -> Dict:
    """
    Compute Expected Shortfall (CVaR) - average loss when VaR is breached.
    """
    valid = ~np.isnan(var_est)
    ret = returns[valid]
    var = var_est[valid]

    violations = ret < var
    if violations.sum() == 0:
        return {'ES': None, 'ES_ratio': None}

    ES = np.mean(ret[violations])
    avg_var = np.mean(var[violations])
    ES_ratio = ES / avg_var if avg_var != 0 else None

    return {
        'ES': float(ES),
        'ES_ratio': float(ES_ratio) if ES_ratio else None,
        'n_violations': int(violations.sum())
    }


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_var_model(
    returns: np.ndarray,
    var_est: np.ndarray,
    alpha: float,
    model_name: str
) -> Dict:
    """
    Comprehensive VaR model evaluation.
    """
    valid = ~np.isnan(var_est)
    ret = returns[valid]
    var = var_est[valid]
    T = len(ret)

    if T == 0:
        return {'model': model_name, 'error': 'No valid estimates'}

    # Violations
    violations = ret < var
    n_viol = violations.sum()
    viol_rate = n_viol / T

    # Statistical tests
    kupiec = kupiec_test(violations, alpha)
    christoffersen = christoffersen_test(violations, alpha)

    # Expected Shortfall
    es_result = compute_expected_shortfall(ret, var)

    return {
        'model': model_name,
        'alpha': alpha,
        'n_days': int(T),
        'n_violations': int(n_viol),
        'violation_rate': float(viol_rate),
        'target_rate': float(alpha),
        'deviation_pp': float((viol_rate - alpha) * 100),
        'kupiec_LR': kupiec['LR'],
        'kupiec_p': kupiec['p_value'],
        'christoffersen_p_uc': christoffersen['p_uc'],
        'christoffersen_p_ind': christoffersen['p_ind'],
        'christoffersen_p_cc': christoffersen['p_cc'],
        'expected_shortfall': es_result['ES'],
        'es_ratio': es_result['ES_ratio'],
        'avg_var': float(np.mean(var)),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("EXTENDED VaR ANALYSIS: Multiple Specifications")
    print("=" * 80)

    # ---- Load data ----
    print("\nLoading Fama-French data...")
    df = download_ff_data()
    print(f"  {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")

    dates = df.index

    # ---- Train/test split (extended backtest: 2010-2024) ----
    train_mask = dates <= pd.Timestamp(TRAIN_END)
    test_mask = dates >= pd.Timestamp(TEST_START)

    print(f"\n  Train: {dates[train_mask][0].date()} to {dates[train_mask][-1].date()} ({train_mask.sum()} days)")
    print(f"  Test:  {dates[test_mask][0].date()} to {dates[test_mask][-1].date()} ({test_mask.sum()} days)")

    train_df = df.loc[train_mask]
    test_df = df.loc[test_mask]
    test_dates = dates[test_mask]

    # ---- Fit HMM on training data ----
    print(f"\nFitting Student-t HMM (K=3, seed={PRIMARY_SEED}) on training data...")
    X_train = train_df[FACTOR_COLS].values
    hmm = StudentTHMM(n_regimes=3, n_iter=200, tol=1e-5, random_state=PRIMARY_SEED)
    hmm.fit(X_train)
    print(f"  Train log-likelihood: {hmm.log_likelihood_:.2f}")

    # Relabel regimes by data norm (training data only)
    train_raw = hmm.predict(X_train, use_filtered=False)
    _, remap = relabel_regimes_by_data_norm(train_df, train_raw, FACTOR_COLS)
    print(f"  Regime remap: {remap}")

    # ---- Get OOS regimes (frozen HMM) ----
    print("\nComputing OOS regimes (frozen HMM, filtered probabilities)...")
    X_all = df[FACTOR_COLS].values
    regimes_raw_all, probs_all = hmm.predict_oos(X_all, use_filtered=True)
    regimes_all = np.array([remap[r] for r in regimes_raw_all])

    test_regimes = regimes_all[test_mask]
    regime_names = {0: 'Normal', 1: 'Elevated', 2: 'Crisis'}
    for k in range(3):
        pct = (test_regimes == k).mean() * 100
        print(f"  Test {regime_names[k]}: {pct:.1f}%")

    # ---- Results storage ----
    all_results = []
    significant_specs = []

    # ---- Run analysis for each asset and alpha level ----
    print("\n" + "=" * 80)
    print("RUNNING VAR BACKTEST FOR ALL SPECIFICATIONS")
    print("=" * 80)

    for asset in ASSET_COLS:
        print(f"\n{'='*60}")
        print(f"Asset: {asset}")
        print(f"{'='*60}")

        test_returns = df.loc[test_mask, asset].values

        # Signal for hybrid model (use related factor)
        if asset == 'SMB':
            signal = df.loc[test_mask, 'HML'].values
        elif asset == 'HML':
            signal = df.loc[test_mask, 'SMB'].values
        else:  # MOM
            signal = df.loc[test_mask, 'HML'].values

        for alpha in ALPHA_LEVELS:
            print(f"\n  Alpha = {alpha*100:.1f}% VaR")
            print("  " + "-" * 50)

            # Model 1: Unconditional Historical VaR
            var_uncond = unconditional_historical_var(test_returns, alpha)
            res_uncond = evaluate_var_model(test_returns, var_uncond, alpha,
                                             f"Unconditional_{asset}_{alpha}")
            all_results.append(res_uncond)

            # Model 2: GARCH(1,1) VaR
            print(f"    Fitting GARCH for {asset}...")
            var_garch, _ = garch_var(test_returns, alpha)
            res_garch = evaluate_var_model(test_returns, var_garch, alpha,
                                            f"GARCH_{asset}_{alpha}")
            all_results.append(res_garch)

            # Model 3: Regime-Conditional VaR
            var_regime = regime_conditional_var(test_returns, test_regimes, alpha)
            res_regime = evaluate_var_model(test_returns, var_regime, alpha,
                                             f"Regime_{asset}_{alpha}")
            all_results.append(res_regime)

            # Model 4: Hybrid (Regime + Signal)
            var_hybrid, adjustments = hybrid_regime_signal_var(
                test_returns, test_regimes, signal, alpha
            )
            res_hybrid = evaluate_var_model(test_returns, var_hybrid, alpha,
                                             f"Hybrid_{asset}_{alpha}")
            res_hybrid['n_signal_adjustments'] = int(adjustments.sum())
            all_results.append(res_hybrid)

            # Print summary
            for res in [res_uncond, res_garch, res_regime, res_hybrid]:
                model_short = res['model'].split('_')[0]
                p_cc = res.get('christoffersen_p_cc')
                p_cc_str = f"{p_cc:.4f}" if p_cc else "N/A"
                viol_str = f"{res['violation_rate']*100:.2f}%"
                target_str = f"{res['target_rate']*100:.1f}%"
                sig = "*" if p_cc and p_cc < 0.05 else ""
                print(f"    {model_short:<12} Viol={viol_str:>6} Target={target_str} CC_p={p_cc_str} {sig}")

                # Track significant results
                if p_cc and p_cc < 0.05:
                    significant_specs.append({
                        'asset': asset,
                        'alpha': alpha,
                        'model': model_short,
                        'p_cc': p_cc,
                        'violation_rate': res['violation_rate'],
                        'target_rate': res['target_rate'],
                        'deviation_pp': res['deviation_pp']
                    })

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("SUMMARY: Specifications with Christoffersen p < 0.05")
    print("=" * 80)

    if significant_specs:
        print(f"\nFound {len(significant_specs)} significant specifications:")
        for spec in sorted(significant_specs, key=lambda x: x['p_cc']):
            print(f"  {spec['asset']:<4} @ {spec['alpha']*100:>4.1f}% : {spec['model']:<12} "
                  f"p_cc={spec['p_cc']:.4f}  viol={spec['violation_rate']*100:.2f}% "
                  f"(dev={spec['deviation_pp']:+.2f}pp)")
    else:
        print("\nNo specifications achieved Christoffersen p < 0.05")

    # ---- Find best model ----
    print("\n" + "=" * 80)
    print("BEST PERFORMING MODELS (by deviation from target)")
    print("=" * 80)

    for alpha in ALPHA_LEVELS:
        alpha_results = [r for r in all_results if r.get('alpha') == alpha]
        if alpha_results:
            best = min(alpha_results, key=lambda x: abs(x.get('deviation_pp', 999)))
            print(f"\n  {alpha*100:.1f}% VaR: {best['model']}")
            print(f"    Violation: {best['violation_rate']*100:.2f}% (target {alpha*100:.1f}%)")
            print(f"    Deviation: {best['deviation_pp']:+.2f}pp")
            p_cc = best.get('christoffersen_p_cc')
            print(f"    Christoffersen p: {p_cc:.4f}" if p_cc else "    Christoffersen p: N/A")

    # ---- Improvement analysis ----
    print("\n" + "=" * 80)
    print("REGIME MODEL IMPROVEMENT OVER UNCONDITIONAL")
    print("=" * 80)

    for asset in ASSET_COLS:
        print(f"\n  {asset}:")
        for alpha in ALPHA_LEVELS:
            uncond = next((r for r in all_results
                          if f"Unconditional_{asset}_{alpha}" == r['model']), None)
            regime = next((r for r in all_results
                          if f"Regime_{asset}_{alpha}" == r['model']), None)

            if uncond and regime:
                uncond_dev = abs(uncond['deviation_pp'])
                regime_dev = abs(regime['deviation_pp'])
                improvement = uncond_dev - regime_dev
                print(f"    {alpha*100:>4.1f}%: Uncond dev={uncond_dev:.2f}pp, "
                      f"Regime dev={regime_dev:.2f}pp, Improvement={improvement:+.2f}pp")

    # ---- Save results ----
    output = {
        'description': 'Extended VaR Analysis with Multiple Specifications',
        'test_period': f'{TEST_START} to {test_dates[-1].date()}',
        'n_test_days': int(test_mask.sum()),
        'alpha_levels': ALPHA_LEVELS,
        'assets': ASSET_COLS,
        'models': ['Unconditional', 'GARCH', 'Regime', 'Hybrid'],
        'hmm_seed': PRIMARY_SEED,
        'significant_specs_p05': significant_specs,
        'all_results': all_results,
        'regime_distribution': {
            regime_names[k]: float((test_regimes == k).mean() * 100)
            for k in range(3)
        },
    }

    output_path = os.path.join(RESULTS_DIR, 'var_extended.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n\nResults saved to: {output_path}")

    # ---- Final report ----
    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)

    print(f"\n1. Specifications with Christoffersen p < 0.05: {len(significant_specs)}")
    if significant_specs:
        best_sig = min(significant_specs, key=lambda x: x['p_cc'])
        print(f"   Best: {best_sig['asset']} @ {best_sig['alpha']*100:.1f}% - {best_sig['model']} "
              f"(p={best_sig['p_cc']:.4f})")

    print(f"\n2. Best performing model across all specs:")
    best_overall = min(all_results, key=lambda x: abs(x.get('deviation_pp', 999)))
    print(f"   {best_overall['model']}: deviation {best_overall['deviation_pp']:+.2f}pp")

    print(f"\n3. Key findings:")
    # Count improvements
    improvements = 0
    for asset in ASSET_COLS:
        for alpha in ALPHA_LEVELS:
            uncond = next((r for r in all_results
                          if f"Unconditional_{asset}_{alpha}" == r['model']), None)
            regime = next((r for r in all_results
                          if f"Regime_{asset}_{alpha}" == r['model']), None)
            if uncond and regime:
                if abs(regime['deviation_pp']) < abs(uncond['deviation_pp']):
                    improvements += 1

    total_specs = len(ASSET_COLS) * len(ALPHA_LEVELS)
    print(f"   - Regime model improved over Unconditional in {improvements}/{total_specs} specs")

    return output


if __name__ == '__main__':
    results = main()
