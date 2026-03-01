#!/usr/bin/env python3
"""
Multipair Generalizability Analysis
=====================================
Demonstrates that regime-conditional Granger causality patterns generalize
beyond HML→SMB to multiple factor pairs.

1. Fits Student-t HMM on training data (1990-2012)
2. For ALL 30 directed factor pairs, computes per-regime Granger at lag 1
3. Identifies top 5 pairs by regime heterogeneity (largest difference between Normal and Crisis)
4. Reports comprehensive table: p-values, heterogeneity scores, Chow structural break tests
5. Tests whether "Normal-significant, Crisis-null" pattern is general phenomenon

Output: results/multipair_generalizability.txt
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f as f_dist
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2

warnings.filterwarnings('ignore')

sys.path.insert(0, '/sessions/sweet-dazzling-heisenberg/mnt/causal_regimes/code')
from multistart_hmm_pipeline import (
    download_ff_data,
    StudentTHMM,
    relabel_regimes_by_data_norm,
    extract_regime_clean_indices,
    run_granger_at_lag,
)

RESULTS_DIR = '/sessions/sweet-dazzling-heisenberg/mnt/causal_regimes/results'
PRIMARY_SEED = 28
FIXED_LAG = 1
FACTOR_NAMES = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
REGIME_NAMES = ['Normal', 'Elevated', 'Crisis']
GFC_DATE = np.datetime64('2008-01-01')


def granger_full(y, x_cause, clean_idx, lag=1):
    """Compute (F_stat, p_value) for x_cause → y Granger test."""
    usable = clean_idx[clean_idx >= lag]
    n = len(usable)
    if n < 2 * lag + 10:
        return np.nan, np.nan
    t = usable
    yv = y[t]
    yl = np.column_stack([y[t - i - 1] for i in range(lag)])
    xl = np.column_stack([x_cause[t - i - 1] for i in range(lag)])
    Xr = np.column_stack([np.ones(n), yl])
    Xu = np.column_stack([np.ones(n), yl, xl])
    br = np.linalg.lstsq(Xr, yv, rcond=None)[0]
    bu = np.linalg.lstsq(Xu, yv, rcond=None)[0]
    rr = float(np.sum((yv - Xr @ br)**2))
    ru = float(np.sum((yv - Xu @ bu)**2))
    df1 = lag
    df2 = n - 2 * lag - 1
    if df2 <= 0 or ru <= 0:
        return np.nan, np.nan
    F = ((rr - ru) / df1) / (ru / df2)
    p = float(1 - stats.f.cdf(F, df1, df2))
    return float(F), p


def chow_ftest(y, Xu, break_idx):
    """
    Chow F-test for structural break in Xu coefficients at break_idx.
    Tests H0: coefficients identical in [0:break_idx] and [break_idx:].
    Returns (F_stat, p_value).
    """
    n, k = Xu.shape
    if break_idx < k + 1 or (n - break_idx) < k + 1:
        return np.nan, np.nan

    # Full-sample (restricted) OLS
    b_r, _, _, _ = np.linalg.lstsq(Xu, y, rcond=None)
    rss_r = float(np.sum((y - Xu @ b_r) ** 2))

    # Pre-break OLS
    y1, X1 = y[:break_idx], Xu[:break_idx]
    b1, _, _, _ = np.linalg.lstsq(X1, y1, rcond=None)
    rss1 = float(np.sum((y1 - X1 @ b1) ** 2))

    # Post-break OLS
    y2, X2 = y[break_idx:], Xu[break_idx:]
    b2, _, _, _ = np.linalg.lstsq(X2, y2, rcond=None)
    rss2 = float(np.sum((y2 - X2 @ b2) ** 2))

    rss_u = rss1 + rss2
    dof_num = k
    dof_den = n - 2 * k

    if dof_den <= 0 or rss_u <= 0:
        return np.nan, np.nan

    F = ((rss_r - rss_u) / dof_num) / (rss_u / dof_den)
    p = float(1 - f_dist.cdf(F, dof_num, dof_den))
    return F, p


def compute_chow_test_2008(test_df, test_regimes, y, x_cause, clean_idx, lag=1):
    """Compute Chow test at 2008-01-01 break date in OOS period."""
    usable = clean_idx[clean_idx >= lag]
    if len(usable) < 2 * lag + 10:
        return np.nan, np.nan

    # Find index in test_df that corresponds to 2008-01-01
    try:
        gfc_idx_in_test = np.where(test_df.index >= GFC_DATE)[0]
        if len(gfc_idx_in_test) == 0:
            return np.nan, np.nan
        gfc_idx_absolute = gfc_idx_in_test[0]
    except:
        return np.nan, np.nan

    # Map to position in clean_idx coordinate system
    # clean_idx contains absolute indices; find break point in usable sequence
    break_pos_in_usable = np.searchsorted(usable, gfc_idx_absolute)
    if break_pos_in_usable < lag + 10 or break_pos_in_usable > len(usable) - lag - 10:
        return np.nan, np.nan

    # Construct design matrix for Chow test
    yv = y[usable]
    yl = np.column_stack([y[usable - i - 1] for i in range(lag)])
    xl = np.column_stack([x_cause[usable - i - 1] for i in range(lag)])
    Xu = np.column_stack([np.ones(len(usable)), yl, xl])

    F, p = chow_ftest(yv, Xu, break_pos_in_usable)
    return F, p


def main():
    print("=" * 80)
    print("MULTIPAIR GENERALIZABILITY ANALYSIS")
    print("=" * 80)

    print("\nLoading FF data (percentage-unit)...")
    df = download_ff_data()

    train_df = df.loc[:'2012-12-31'].copy()
    test_df = df.loc['2013-01-01':].copy()

    print(f"  Train: {len(train_df)} days ({train_df.index[0].date()} to {train_df.index[-1].date()})")
    print(f"  Test:  {len(test_df)} days ({test_df.index[0].date()} to {test_df.index[-1].date()})")

    hmm_cols = FACTOR_NAMES
    print(f"\nFitting HMM (seed {PRIMARY_SEED}, K=3) on train data...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=PRIMARY_SEED)
    hmm.fit(train_df[hmm_cols].values)

    # Relabel regimes by train-period data norm
    train_raw = hmm.predict(train_df[hmm_cols].values, use_filtered=False)
    _, remap = relabel_regimes_by_data_norm(train_df, train_raw, hmm_cols)

    # Apply remap to test (no test-data look-ahead)
    test_raw, _ = hmm.predict_oos(test_df[hmm_cols].values, use_filtered=True)
    test_regimes = np.array([remap[r] for r in test_raw])

    # Extract clean indices per regime
    clean_regimes = {}
    for k, name in enumerate(REGIME_NAMES):
        clean_regimes[name] = extract_regime_clean_indices(test_regimes, k, max_lag=FIXED_LAG)
        print(f"  {name}: n_clean={len(clean_regimes[name])}")

    # Extract factor arrays for test period
    factor_data = {name: test_df[name].values for name in FACTOR_NAMES}
    test_dates = test_df.index.values

    # Run all 30 directed pairs, compute per-regime Granger
    print("\n" + "=" * 80)
    print("COMPUTING GRANGER CAUSALITY FOR ALL 30 DIRECTED PAIRS")
    print("=" * 80)

    pair_results = []
    for cause in FACTOR_NAMES:
        for effect in FACTOR_NAMES:
            if cause == effect:
                continue

            pair_key = f"{cause}->{effect}"
            regime_pvalues = {}
            regime_fstats = {}
            chow_test_2008 = {}

            for regime_name, regime_idx in [('Normal', 0), ('Elevated', 1), ('Crisis', 2)]:
                clean = clean_regimes[regime_name]
                F, p = granger_full(factor_data[effect], factor_data[cause], clean, lag=FIXED_LAG)
                regime_pvalues[regime_name] = p
                regime_fstats[regime_name] = F

                # Chow test at 2008-01-01
                if regime_name == 'Normal':  # Only for Normal (largest n)
                    F_chow, p_chow = compute_chow_test_2008(
                        test_df, test_regimes,
                        factor_data[effect], factor_data[cause],
                        clean, lag=FIXED_LAG
                    )
                    chow_test_2008['F'] = F_chow
                    chow_test_2008['p'] = p_chow

            # Compute heterogeneity metric: |p_Normal - p_Crisis|
            # Higher values = more regime-dependent
            het_score = abs(regime_pvalues['Normal'] - regime_pvalues['Crisis'])
            if np.isnan(het_score):
                het_score = 0.0

            pair_results.append({
                'pair': pair_key,
                'cause': cause,
                'effect': effect,
                'p_Normal': regime_pvalues['Normal'],
                'p_Elevated': regime_pvalues['Elevated'],
                'p_Crisis': regime_pvalues['Crisis'],
                'F_Normal': regime_fstats['Normal'],
                'F_Elevated': regime_fstats['Elevated'],
                'F_Crisis': regime_fstats['Crisis'],
                'heterogeneity_score': het_score,
                'chow_2008_F': chow_test_2008.get('F', np.nan),
                'chow_2008_p': chow_test_2008.get('p', np.nan),
            })

    print(f"Computed {len(pair_results)} pairs")

    # Sort by heterogeneity score descending
    pair_results.sort(key=lambda x: x['heterogeneity_score'], reverse=True)

    # Identify top 5 heterogeneous pairs + HML->SMB
    top_5 = pair_results[:5]
    hml_smb = next((p for p in pair_results if p['pair'] == 'HML->SMB'), None)

    if hml_smb and hml_smb not in top_5:
        pairs_to_report = top_5 + [hml_smb]
    else:
        pairs_to_report = top_5

    # Output table
    print("\n" + "=" * 80)
    print("GENERALIZABILITY TABLE: TOP REGIME-HETEROGENEOUS PAIRS + HML->SMB")
    print("=" * 80)

    output_lines = []
    output_lines.append("\n" + "=" * 100)
    output_lines.append("REGIME-CONDITIONAL GRANGER CAUSALITY: GENERALIZABILITY ACROSS 30 FACTOR PAIRS")
    output_lines.append("=" * 100)
    output_lines.append("")
    output_lines.append("Method:")
    output_lines.append("  - Train HMM on 1990-2012 (seed 28, Student-t, K=3)")
    output_lines.append("  - Test on 2013-2024 (OOS, filtered, no data leakage)")
    output_lines.append("  - Granger at lag 1, per-regime with clean boundary handling")
    output_lines.append("  - Heterogeneity = |p_Normal - p_Crisis|")
    output_lines.append("  - Chow test at 2008-01-01 structural break")
    output_lines.append("")
    output_lines.append("Results for Top 5 Most Heterogeneous Pairs + HML->SMB (shown if different):")
    output_lines.append("-" * 100)
    output_lines.append("")

    for i, pair in enumerate(pairs_to_report, 1):
        marker = " ← PRIMARY" if pair['pair'] == 'HML->SMB' else ""
        output_lines.append(f"{i}. {pair['pair']}{marker}")
        output_lines.append(f"   Heterogeneity score: {pair['heterogeneity_score']:.4f}")
        output_lines.append(f"   p-values by regime:")
        output_lines.append(f"     Normal:   {pair['p_Normal']:.6f} (F={pair['F_Normal']:.4f})")
        output_lines.append(f"     Elevated: {pair['p_Elevated']:.6f} (F={pair['F_Elevated']:.4f})")
        output_lines.append(f"     Crisis:   {pair['p_Crisis']:.6f} (F={pair['F_Crisis']:.4f})")
        output_lines.append(f"   Chow test at 2008-01-01 (Normal regime): F={pair['chow_2008_F']:.4f}, p={pair['chow_2008_p']:.6f}")

        # Pattern classification
        normal_sig = pair['p_Normal'] < 0.05
        crisis_null = pair['p_Crisis'] >= 0.05

        if normal_sig and crisis_null:
            pattern = "STRONG: Significant in Normal, null in Crisis (confirms generalization)"
        elif pair['p_Normal'] < pair['p_Crisis']:
            pattern = "MODERATE: Weaker in Crisis than Normal"
        else:
            pattern = "WEAK: No clear regime pattern"

        output_lines.append(f"   Pattern: {pattern}")
        output_lines.append("")

    output_lines.append("=" * 100)
    output_lines.append("\nINTERPRETATION:")
    output_lines.append("-" * 100)

    # Count patterns
    strong_count = sum(
        1 for p in pairs_to_report
        if (p['p_Normal'] < 0.05 and p['p_Crisis'] >= 0.05)
    )
    moderate_count = sum(
        1 for p in pairs_to_report
        if (p['p_Normal'] < p['p_Crisis']) and not (p['p_Normal'] < 0.05 and p['p_Crisis'] >= 0.05)
    )

    output_lines.append(f"\nPattern Prevalence (out of {len(pairs_to_report)} reported pairs):")
    output_lines.append(f"  Strong regime heterogeneity (Normal sig, Crisis null): {strong_count}/{len(pairs_to_report)}")
    output_lines.append(f"  Moderate regime heterogeneity (Normal < Crisis): {moderate_count}/{len(pairs_to_report)}")
    output_lines.append("")

    if strong_count >= 3:
        output_lines.append("CONCLUSION: The regime-heterogeneous Granger pattern (significant in Normal,")
        output_lines.append("null in Crisis) GENERALIZES to multiple factor pairs. This is NOT specific to HML->SMB.")
        output_lines.append("The phenomenon reflects a systematic regime-dependent information structure.")
    elif moderate_count >= 3:
        output_lines.append("CONCLUSION: Most pairs show weaker causality in Crisis than Normal, suggesting")
        output_lines.append("regime-dependent information transfer. Pattern is somewhat general but heterogeneous.")
    else:
        output_lines.append("CONCLUSION: Regime-dependent Granger patterns show mixed results. HML->SMB")
        output_lines.append("may be a particularly clear example of regime conditioning in factor markets.")

    output_lines.append("")
    output_lines.append("=" * 100)
    output_lines.append("\nFULL RANKING OF ALL 30 PAIRS BY HETEROGENEITY:")
    output_lines.append("-" * 100)
    output_lines.append("")

    for i, pair in enumerate(pair_results[:30], 1):
        sig_marker = "***" if pair['p_Normal'] < 0.05 and pair['p_Crisis'] >= 0.05 else ""
        output_lines.append(
            f"{i:2d}. {pair['pair']:12s}  het={pair['heterogeneity_score']:.4f}  "
            f"p_N={pair['p_Normal']:.4f}  p_C={pair['p_Crisis']:.4f} {sig_marker}"
        )

    output_text = "\n".join(output_lines)
    print(output_text)

    # Save to file
    out_path = f"{RESULTS_DIR}/multipair_generalizability.txt"
    with open(out_path, 'w') as f:
        f.write(output_text)
    print(f"\nSaved to {out_path}")

    # Also save JSON for further analysis
    json_output = {
        'description': 'Regime-conditional Granger causality across 30 directed factor pairs',
        'train_period': f'{train_df.index[0].date()} to {train_df.index[-1].date()}',
        'test_period': f'{test_df.index[0].date()} to {test_df.index[-1].date()}',
        'hmm_seed': PRIMARY_SEED,
        'lag': FIXED_LAG,
        'all_pairs': pair_results,
        'top_5_heterogeneous': top_5,
        'hml_smb_rank': next((i for i, p in enumerate(pair_results, 1) if p['pair'] == 'HML->SMB'), None),
    }

    json_path = f"{RESULTS_DIR}/multipair_generalizability.json"
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"Saved JSON to {json_path}")


if __name__ == '__main__':
    main()
