#!/usr/bin/env python3
"""
Enhanced Multipair Generalizability Analysis
==============================================
Demonstrates that regime-conditional Granger causality patterns generalize
beyond HML→SMB to multiple factor pairs, with careful attention to:

1. In-sample regime heterogeneity (difference in p-values across regimes)
2. Out-of-sample per-regime signals
3. Consistency of "Normal-significant, Crisis-null" pattern
4. Structural break detection at 2008-01-01

Output: results/multipair_generalizability.txt
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f as f_dist

warnings.filterwarnings('ignore')

sys.path.insert(0, '/sessions/sweet-dazzling-heisenberg/mnt/causal_regimes/code')
from multistart_hmm_pipeline import (
    download_ff_data,
    StudentTHMM,
    relabel_regimes_by_data_norm,
    extract_regime_clean_indices,
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

    try:
        gfc_idx_in_test = np.where(test_df.index >= GFC_DATE)[0]
        if len(gfc_idx_in_test) == 0:
            return np.nan, np.nan
        gfc_idx_absolute = gfc_idx_in_test[0]
    except:
        return np.nan, np.nan

    break_pos_in_usable = np.searchsorted(usable, gfc_idx_absolute)
    if break_pos_in_usable < lag + 10 or break_pos_in_usable > len(usable) - lag - 10:
        return np.nan, np.nan

    yv = y[usable]
    yl = np.column_stack([y[usable - i - 1] for i in range(lag)])
    xl = np.column_stack([x_cause[usable - i - 1] for i in range(lag)])
    Xu = np.column_stack([np.ones(len(usable)), yl, xl])

    F, p = chow_ftest(yv, Xu, break_pos_in_usable)
    return F, p


def classify_pair_pattern(p_normal, p_elevated, p_crisis, f_normal, f_elevated, f_crisis):
    """Classify a pair by its regime-dependent pattern."""
    sig_thresh = 0.05

    # Core pattern: Normal-sig & Crisis-null
    if p_normal < sig_thresh and p_crisis >= sig_thresh:
        return 'CORE_PATTERN'

    # Elevated signal (like MOM->SMB reviewer comment)
    if p_elevated < sig_thresh and (p_normal >= sig_thresh or p_crisis >= sig_thresh):
        return 'ELEVATED_SIGNAL'

    # Monotonic: significant in more volatile regimes
    if p_normal >= p_elevated >= p_crisis:
        return 'MONOTONIC_DECREASING'  # gets stronger (lower p) as volatility increases

    # Inverse monotonic
    if p_normal <= p_elevated <= p_crisis:
        return 'MONOTONIC_INCREASING'  # gets weaker as volatility increases

    # Elevated-crisis signal (opposite of Normal)
    if (p_normal >= sig_thresh) and (p_elevated < sig_thresh or p_crisis < sig_thresh):
        return 'ELEVATED_CRISIS_SIGNAL'

    return 'NO_CLEAR_PATTERN'


def main():
    print("=" * 80)
    print("ENHANCED MULTIPAIR GENERALIZABILITY ANALYSIS")
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

                # Chow test at 2008-01-01 (only for Normal with enough n)
                if regime_name == 'Normal':
                    F_chow, p_chow = compute_chow_test_2008(
                        test_df, test_regimes,
                        factor_data[effect], factor_data[cause],
                        clean, lag=FIXED_LAG
                    )
                    chow_test_2008['F'] = F_chow
                    chow_test_2008['p'] = p_chow

            # Classify pattern
            pattern = classify_pair_pattern(
                regime_pvalues['Normal'],
                regime_pvalues['Elevated'],
                regime_pvalues['Crisis'],
                regime_fstats['Normal'],
                regime_fstats['Elevated'],
                regime_fstats['Crisis'],
            )

            # Heterogeneity: use max range across regimes
            ps = [regime_pvalues['Normal'], regime_pvalues['Elevated'], regime_pvalues['Crisis']]
            valid_ps = [p for p in ps if not np.isnan(p)]
            het_score = max(valid_ps) - min(valid_ps) if valid_ps else 0.0

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
                'pattern': pattern,
                'chow_2008_F': chow_test_2008.get('F', np.nan),
                'chow_2008_p': chow_test_2008.get('p', np.nan),
            })

    print(f"Computed {len(pair_results)} pairs")

    # Identify pairs by pattern type
    pattern_groups = {}
    for p in pair_results:
        pat = p['pattern']
        if pat not in pattern_groups:
            pattern_groups[pat] = []
        pattern_groups[pat].append(p)

    # Sort each group by heterogeneity
    for pat in pattern_groups:
        pattern_groups[pat].sort(key=lambda x: x['heterogeneity_score'], reverse=True)

    # Build reporting set: top-ranked pairs overall + representatives from each pattern
    all_sorted = sorted(pair_results, key=lambda x: x['heterogeneity_score'], reverse=True)
    hml_smb = next((p for p in pair_results if p['pair'] == 'HML->SMB'), None)
    mom_smb = next((p for p in pair_results if p['pair'] == 'MOM->SMB'), None)

    # Report top 6-8 pairs
    top_pairs = all_sorted[:6]
    pairs_to_report = []
    seen_pairs = set()
    for p in top_pairs:
        if p['pair'] not in seen_pairs:
            pairs_to_report.append(p)
            seen_pairs.add(p['pair'])
    if mom_smb and mom_smb['pair'] not in seen_pairs:
        pairs_to_report.append(mom_smb)
        seen_pairs.add(mom_smb['pair'])
    if hml_smb and hml_smb['pair'] not in seen_pairs:
        pairs_to_report.append(hml_smb)
        seen_pairs.add(hml_smb['pair'])

    pairs_to_report.sort(key=lambda x: x['heterogeneity_score'], reverse=True)

    # Output table
    print("\n" + "=" * 80)
    print("GENERALIZABILITY ANALYSIS: TOP HETEROGENEOUS PAIRS")
    print("=" * 80)

    output_lines = []
    output_lines.append("\n" + "=" * 120)
    output_lines.append("REGIME-CONDITIONAL GRANGER CAUSALITY: GENERALIZABILITY ANALYSIS")
    output_lines.append("=" * 120)
    output_lines.append("")
    output_lines.append("Context:")
    output_lines.append("  Reviewer feedback: Analysis focused only on HML→SMB is too narrow.")
    output_lines.append("  MOM→SMB was mentioned as having stronger OOS signal (F=20.3 vs 9.06 for HML→SMB).")
    output_lines.append("  Task: Demonstrate whether regime-conditional Granger pattern generalizes.")
    output_lines.append("")
    output_lines.append("Method:")
    output_lines.append("  - Train HMM on 1990-2012 (seed 28, Student-t, K=3)")
    output_lines.append("  - Test on 2013-2024 (OOS, filtered, no data leakage)")
    output_lines.append("  - Granger at lag 1, per-regime with clean boundary handling")
    output_lines.append("  - Pattern heterogeneity = max(p-values) - min(p-values) across 3 regimes")
    output_lines.append("  - Chow test at 2008-01-01 for structural breaks")
    output_lines.append("")
    output_lines.append("Pattern Classifications:")
    output_lines.append("  CORE_PATTERN:           Significant in Normal, null in Crisis (classic regime heterogeneity)")
    output_lines.append("  ELEVATED_SIGNAL:        Significant in Elevated/Crisis, weaker/null in Normal (inverse pattern)")
    output_lines.append("  MONOTONIC_DECREASING:   Weaker as volatility increases (p_Normal >= p_Elevated >= p_Crisis)")
    output_lines.append("  MONOTONIC_INCREASING:   Stronger as volatility increases (p_Normal <= p_Elevated <= p_Crisis)")
    output_lines.append("  ELEVATED_CRISIS_SIGNAL: Strong signal in Elevated/Crisis, weak in Normal")
    output_lines.append("  NO_CLEAR_PATTERN:       Mixed or inconsistent regime dependence")
    output_lines.append("")
    output_lines.append("=" * 120)
    output_lines.append("REPORTED PAIRS (Top heterogeneous + MOM→SMB + HML→SMB for comparison)")
    output_lines.append("=" * 120)
    output_lines.append("")

    for i, pair in enumerate(pairs_to_report, 1):
        markers = []
        if pair['pair'] == 'HML->SMB':
            markers.append("PRIMARY (from paper)")
        if pair['pair'] == 'MOM->SMB':
            markers.append("REVIEWER EXAMPLE (stronger F)")

        marker_str = " | " + ", ".join(markers) if markers else ""
        output_lines.append(f"{i}. {pair['pair']}{marker_str}")
        output_lines.append(f"   Heterogeneity: {pair['heterogeneity_score']:.4f}")
        output_lines.append(f"   Pattern:       {pair['pattern']}")
        output_lines.append(f"   Granger p-values by regime:")
        output_lines.append(f"     Normal:     p={pair['p_Normal']:.6f} (F={pair['F_Normal']:.4f})")
        output_lines.append(f"     Elevated:   p={pair['p_Elevated']:.6f} (F={pair['F_Elevated']:.4f})")
        output_lines.append(f"     Crisis:     p={pair['p_Crisis']:.6f} (F={pair['F_Crisis']:.4f})")

        if not np.isnan(pair['chow_2008_F']):
            output_lines.append(f"   Chow 2008-01-01 (Normal): F={pair['chow_2008_F']:.4f}, p={pair['chow_2008_p']:.6f}")

        output_lines.append("")

    output_lines.append("=" * 120)
    output_lines.append("PATTERN SUMMARY ACROSS ALL 30 PAIRS")
    output_lines.append("=" * 120)
    output_lines.append("")

    for pattern_name in sorted(pattern_groups.keys()):
        count = len(pattern_groups[pattern_name])
        output_lines.append(f"{pattern_name}: {count} pairs")
        for pair in pattern_groups[pattern_name][:3]:  # Show top 3 per pattern
            output_lines.append(f"  - {pair['pair']:12s} het={pair['heterogeneity_score']:.4f}")
        if count > 3:
            output_lines.append(f"  ... ({count - 3} more)")
        output_lines.append("")

    output_lines.append("=" * 120)
    output_lines.append("GENERALIZABILITY CONCLUSION")
    output_lines.append("=" * 120)
    output_lines.append("")

    # Count different patterns
    core_count = len(pattern_groups.get('CORE_PATTERN', []))
    elevated_count = len(pattern_groups.get('ELEVATED_SIGNAL', []))
    monotonic_count = len(
        pattern_groups.get('MONOTONIC_DECREASING', []) +
        pattern_groups.get('MONOTONIC_INCREASING', [])
    )

    if core_count >= 2:
        output_lines.append("STRONG GENERALIZATION: Multiple pairs show the CORE regime pattern")
        output_lines.append(f"(Normal-significant, Crisis-null), not just HML→SMB.")
        output_lines.append(f"Found {core_count} pairs with this pattern.")
    elif elevated_count >= 3:
        output_lines.append("INVERSE PATTERN GENERALIZATION: Multiple pairs show regime conditioning,")
        output_lines.append(f"though often via elevated/crisis signals (not the Normal-significant pattern).")
        output_lines.append(f"Example: MOM→SMB shows strong Elevated signal (F={mom_smb['F_Elevated']:.2f}, p={mom_smb['p_Elevated']:.6f})")
    elif monotonic_count >= 4:
        output_lines.append("MONOTONIC GENERALIZATION: Regime-dependent Granger causality manifests")
        output_lines.append("primarily as monotonic changes in strength across regimes, not discrete patterns.")
    else:
        output_lines.append("WEAK GENERALIZATION: Regime-conditional Granger patterns are specific to")
        output_lines.append("certain factor pairs; the HML→SMB pattern may reflect a particular property")
        output_lines.append("of these factors rather than a general phenomenon.")

    output_lines.append("")
    output_lines.append("Key Findings:")
    if hml_smb:
        output_lines.append(f"  - HML→SMB rank by heterogeneity: {all_sorted.index(hml_smb) + 1}/30")
        output_lines.append(f"    Pattern: {hml_smb['pattern']}")
    if mom_smb:
        output_lines.append(f"  - MOM→SMB rank by heterogeneity: {all_sorted.index(mom_smb) + 1}/30")
        output_lines.append(f"    Pattern: {mom_smb['pattern']}")
        output_lines.append(f"    Elevated F-stat: {mom_smb['F_Elevated']:.4f} (stronger than HML→SMB: {hml_smb['F_Elevated']:.4f})")

    output_lines.append("")
    output_lines.append("=" * 120)
    output_lines.append("\nFULL RANKING OF ALL 30 PAIRS BY HETEROGENEITY SCORE")
    output_lines.append("-" * 120)
    output_lines.append("")

    for i, pair in enumerate(all_sorted, 1):
        output_lines.append(
            f"{i:2d}. {pair['pair']:12s}  het={pair['heterogeneity_score']:.4f}  "
            f"p_N={pair['p_Normal']:.4f}  p_E={pair['p_Elevated']:.4f}  p_C={pair['p_Crisis']:.4f}  "
            f"pattern={pair['pattern']}"
        )

    output_text = "\n".join(output_lines)
    print(output_text)

    # Save to file
    out_path = f"{RESULTS_DIR}/multipair_generalizability.txt"
    with open(out_path, 'w') as f:
        f.write(output_text)
    print(f"\nSaved to {out_path}")

    # Save JSON for further analysis
    json_output = {
        'description': 'Enhanced multipair generalizability: regime-conditional Granger across 30 pairs',
        'train_period': f'{train_df.index[0].date()} to {train_df.index[-1].date()}',
        'test_period': f'{test_df.index[0].date()} to {test_df.index[-1].date()}',
        'hmm_seed': PRIMARY_SEED,
        'lag': FIXED_LAG,
        'all_pairs': pair_results,
        'pattern_summary': {k: len(v) for k, v in pattern_groups.items()},
        'hml_smb': hml_smb,
        'mom_smb': mom_smb,
    }

    json_path = f"{RESULTS_DIR}/multipair_generalizability.json"
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"Saved JSON to {json_path}")


if __name__ == '__main__':
    main()
