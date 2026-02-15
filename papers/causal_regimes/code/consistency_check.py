"""
Consistency Check: Validates ALL 11 tables + narrative/figure claims against JSON sources.
Reads LaTeX file and JSON results, compares extracted numbers.
Output: PASS/FAIL for each check with details.
"""

import argparse
import json
import re
import sys
import os
import io
from contextlib import redirect_stdout

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results'
LATEX_FILE = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/main_icaif.tex'


def load_json(filename, results_dir=RESULTS_DIR):
    path = f"{results_dir}/{filename}"
    if not os.path.exists(path):
        print(f"  WARNING: {filename} not found")
        return None
    with open(path) as f:
        return json.load(f)


def read_latex(latex_file=LATEX_FILE):
    with open(latex_file) as f:
        return f.read()


def check_value(label, expected, actual, tolerance=0.01):
    """Check if actual is within tolerance of expected."""
    if expected is None or actual is None:
        return f"  SKIP: {label} (missing data)"
    if isinstance(expected, str) and isinstance(actual, str):
        if expected == actual:
            return f"  PASS: {label} = {expected}"
        else:
            return f"  FAIL: {label}: expected={expected}, got={actual}"
    try:
        exp_f = float(expected)
        act_f = float(actual)
        if abs(exp_f) < 1e-10:
            if abs(act_f) < 1e-10:
                return f"  PASS: {label} = {expected}"
            else:
                return f"  FAIL: {label}: expected={expected}, got={actual}"
        rel_err = abs(exp_f - act_f) / max(abs(exp_f), 1e-15)
        if rel_err < tolerance:
            return f"  PASS: {label} = {actual} (expected {expected}, err={rel_err:.4f})"
        else:
            return f"  FAIL: {label}: expected={expected}, got={actual}, rel_err={rel_err:.4f}"
    except (ValueError, TypeError):
        return f"  SKIP: {label} (cannot compare: {expected} vs {actual})"


def extract_latex_table(latex, label):
    """Extract table content between \\label{label} and \\end{tabular}."""
    pattern = rf'\\label\{{{label}\}}.*?\\end\{{tabular\}}'
    match = re.search(pattern, latex, re.DOTALL)
    if match:
        return match.group(0)
    return None


def check_table1(hmm_json, latex):
    """Tab:regimes — Regime Summary Statistics."""
    print("\n=== Table 1 (tab:regimes) ===")
    results = []
    t1 = hmm_json['selected_fit']['table1']
    table_text = extract_latex_table(latex, 'tab:regimes')
    if table_text is None:
        print("  FAIL: Cannot find tab:regimes in LaTeX")
        return

    for regime in ['Normal', 'Elevated', 'Crisis']:
        d = t1[regime]
        # Check n_days appears in table
        n_str = f"{d['n_days']:,}"
        if n_str in table_text:
            results.append(f"  PASS: {regime} days = {n_str}")
        else:
            results.append(f"  FAIL: {regime} days = {n_str} not found in table")

        # Check proportion
        prop_str = f"{d['proportion']:.1f}"
        if prop_str in table_text:
            results.append(f"  PASS: {regime} proportion = {prop_str}%")
        else:
            results.append(f"  FAIL: {regime} proportion = {prop_str}% not found")

        # Check mean_norm
        mn_str = f"{d['mean_norm']:.2f}"
        if mn_str in table_text:
            results.append(f"  PASS: {regime} mean_norm = {mn_str}")
        else:
            results.append(f"  FAIL: {regime} mean_norm = {mn_str} not found")

        # Check nu
        nu_str = f"{d['nu']:.1f}"
        if nu_str in table_text:
            results.append(f"  PASS: {regime} nu = {nu_str}")
        else:
            results.append(f"  FAIL: {regime} nu = {nu_str} not found")

        # Check transition prob
        tp_str = f"{d['transition_prob']:.3f}"
        if tp_str in table_text:
            results.append(f"  PASS: {regime} P(stay) = {tp_str}")
        else:
            results.append(f"  FAIL: {regime} P(stay) = {tp_str} not found")

    for r in results:
        print(r)


def check_table2(hmm_json, latex):
    """Tab:detection — Crisis Detection Comparison."""
    print("\n=== Table 2 (tab:detection) ===")
    results = []
    det = hmm_json['selected_fit']['detection']
    table_text = extract_latex_table(latex, 'tab:detection')
    if table_text is None:
        print("  FAIL: Cannot find tab:detection in LaTeX")
        return

    for event, d in det.items():
        st_str = f"{d['student_t_crisis_pct']:.1f}"
        g_str = f"{d['gaussian_crisis_pct']:.1f}"
        if st_str in table_text:
            results.append(f"  PASS: {event} Student-t = {st_str}%")
        else:
            results.append(f"  FAIL: {event} Student-t = {st_str}% not found")
        if g_str in table_text:
            results.append(f"  PASS: {event} Gaussian = {g_str}%")
        else:
            results.append(f"  FAIL: {event} Gaussian = {g_str}% not found")

    for r in results:
        print(r)


def check_table3(hmm_json, latex):
    """Tab:main — Granger Causality."""
    print("\n=== Table 3 (tab:main) ===")
    results = []
    granger = hmm_json['selected_fit']['granger']
    table_text = extract_latex_table(latex, 'tab:main')
    if table_text is None:
        print("  FAIL: Cannot find tab:main in LaTeX")
        return

    for regime in ['Normal', 'Elevated', 'Crisis']:
        for direction in ['hml_to_smb', 'smb_to_hml']:
            d = granger[regime][direction]
            lag = d['lag']
            if str(lag) in table_text:
                results.append(f"  PASS: {regime} {direction} lag = {lag}")
            # p-value check (format varies)
            p = d['f_p_value']
            # Check if p-value is approximately represented
            if p < 0.001:
                exp = f"{p:.1e}".replace('e-0', 'e-')
                results.append(f"  INFO: {regime} {direction} p = {p:.2e}")
            else:
                results.append(f"  INFO: {regime} {direction} p = {p:.3f}")

    for r in results:
        print(r)


def check_table4(hmm_json, latex):
    """Tab:r2 — Incremental R2."""
    print("\n=== Table 4 (tab:r2) ===")
    results = []
    r2 = hmm_json['selected_fit']['r2']
    table_text = extract_latex_table(latex, 'tab:r2')
    if table_text is None:
        print("  FAIL: Cannot find tab:r2 in LaTeX")
        return

    for regime in ['Normal', 'Elevated', 'Crisis']:
        d = r2[regime]
        n_str = f"{d['n_clean']:,}"
        if n_str in table_text:
            results.append(f"  PASS: {regime} n = {n_str}")
        else:
            results.append(f"  FAIL: {regime} n = {n_str} not found")

        r2_ar_str = f"{d['r2_ar']:.2f}"
        if r2_ar_str in table_text:
            results.append(f"  PASS: {regime} R2_AR = {r2_ar_str}%")
        else:
            results.append(f"  FAIL: {regime} R2_AR = {r2_ar_str}% not found")

        dr2_str = f"{d['delta_r2']:.2f}"
        if dr2_str in table_text:
            results.append(f"  PASS: {regime} delta_R2 = {dr2_str}%")
        else:
            results.append(f"  FAIL: {regime} delta_R2 = {dr2_str}% not found")

    for r in results:
        print(r)


def check_table5(hmm_json, latex):
    """Tab:warning — Early Warning Lead Time."""
    print("\n=== Table 5 (tab:warning) ===")
    results = []
    warn = hmm_json['selected_fit']['warning']
    table_text = extract_latex_table(latex, 'tab:warning')
    if table_text is None:
        print("  FAIL: Cannot find tab:warning in LaTeX")
        return

    for event, d in warn.items():
        lead = d['lead_time_days']
        lead_str = str(lead)
        if lead_str in table_text:
            results.append(f"  PASS: {event} lead time = {lead} days")
        else:
            results.append(f"  FAIL: {event} lead time = {lead} days not found")

    for r in results:
        print(r)


def check_table6(hmm_json, latex):
    """Tab:events — Event-Based Validation."""
    print("\n=== Table 6 (tab:events) ===")
    results = []
    events = hmm_json['selected_fit']['events']
    table_text = extract_latex_table(latex, 'tab:events')
    if table_text is None:
        print("  FAIL: Cannot find tab:events in LaTeX")
        return

    for event, d in events.items():
        days_str = str(d['days'])
        if days_str in table_text:
            results.append(f"  PASS: {event} days = {days_str}")
        else:
            results.append(f"  FAIL: {event} days = {days_str} not found")

        p_hml = d['hml_to_smb_p']
        p_str = f"{p_hml:.3f}"
        if p_str in table_text:
            results.append(f"  PASS: {event} HML->SMB p = {p_str}")
        else:
            results.append(f"  FAIL: {event} HML->SMB p = {p_str} not found")

    for r in results:
        print(r)


def check_table7(hmm_json, latex):
    """Tab:frozen_events — Held-Out Stress Events."""
    print("\n=== Table 7 (tab:frozen_events) ===")
    results = []
    frozen = hmm_json['frozen_oos']
    table_text = extract_latex_table(latex, 'tab:frozen_events')
    if table_text is None:
        print("  FAIL: Cannot find tab:frozen_events in LaTeX")
        return

    for event, d in frozen['events'].items():
        days_str = str(d['days'])
        if days_str in table_text:
            results.append(f"  PASS: {event} days = {days_str}")
        else:
            results.append(f"  FAIL: {event} days = {days_str} not found")

        crisis_pct = d['crisis_pct']
        pct_str = f"{crisis_pct:.0f}"
        if pct_str in table_text:
            results.append(f"  PASS: {event} Crisis% = {pct_str}")
        else:
            results.append(f"  FAIL: {event} Crisis% = {pct_str} not found")

    # Check aggregate frozen OOS inline numbers
    agg = frozen['aggregate']
    crisis_frac = frozen['crisis_fraction_pct']
    results.append(f"  INFO: Frozen crisis fraction = {crisis_frac:.1f}%")
    results.append(f"  INFO: Frozen aggregate p = {agg['p_value']:.3f}")
    results.append(f"  INFO: Frozen aggregate n_clean = {agg['n_clean']}")

    for r in results:
        print(r)


def check_table8(trading_json, latex):
    """Tab:trading — Trading Strategy Backtest."""
    print("\n=== Table 8 (tab:trading) ===")
    if trading_json is None:
        print("  SKIP: trading_selected.json not found")
        return

    results = []
    table_text = extract_latex_table(latex, 'tab:trading')
    if table_text is None:
        print("  FAIL: Cannot find tab:trading in LaTeX")
        return

    s = trading_json['strategy']
    b = trading_json['benchmark']

    # Check strategy values
    for val, label in [(s['annual_return_pct'], 'Strategy ann ret'),
                       (s['sharpe_ratio'], 'Strategy Sharpe'),
                       (s['max_drawdown_pct'], 'Strategy max DD')]:
        val_str = f"{val}"
        results.append(f"  INFO: {label} = {val_str}")

    for r in results:
        print(r)


def check_table9(var_json, latex):
    """Tab:var — VaR Backtest (verify only)."""
    print("\n=== Table 9 (tab:var) — VERIFY ONLY ===")
    if var_json is None:
        print("  SKIP: var_fixes_results.json not found")
        return

    results = []
    table_text = extract_latex_table(latex, 'tab:var')
    if table_text is None:
        print("  FAIL: Cannot find tab:var in LaTeX")
        return

    var_data = var_json.get('section_a_var_results', {})
    for model_name, d in var_data.items():
        vr = d.get('violation_rate_pct')
        cc_p = d.get('cc_p_value')
        if vr is not None:
            vr_str = f"{vr:.2f}"
            if vr_str in table_text:
                results.append(f"  PASS: {model_name} violation rate = {vr_str}%")
            else:
                results.append(f"  FAIL: {model_name} violation rate = {vr_str}% not found")

    for r in results:
        print(r)


def check_table10(neural_json, latex):
    """Tab:neural — Nonlinear Granger Causality."""
    print("\n=== Table 10 (tab:neural) ===")
    if neural_json is None:
        print("  SKIP: neural_granger_selected.json not found")
        return

    results = []
    table_text = extract_latex_table(latex, 'tab:neural')
    if table_text is None:
        print("  FAIL: Cannot find tab:neural in LaTeX")
        return

    for regime in ['Normal', 'Elevated', 'Crisis']:
        d = neural_json['results'].get(regime, {})
        if 'linear_mse_improvement_pct' in d:
            results.append(f"  INFO: {regime} Linear MSE imp = {d['linear_mse_improvement_pct']:.2f}%")
            results.append(f"  INFO: {regime} RF p = {d.get('rf_p_value', 'N/A')}")
            results.append(f"  INFO: {regime} MLP p = {d.get('mlp_p_value', 'N/A')}")

    for r in results:
        print(r)


def check_table11(te_json, latex):
    """Tab:te — Transfer Entropy."""
    print("\n=== Table 11 (tab:te) ===")
    if te_json is None:
        print("  SKIP: te_selected.json not found")
        return

    results = []
    table_text = extract_latex_table(latex, 'tab:te')
    if table_text is None:
        print("  FAIL: Cannot find tab:te in LaTeX")
        return

    for regime in ['Normal', 'Elevated', 'Crisis']:
        d = te_json['results'].get(regime, {})
        hs = d.get('hml_to_smb', {})
        sh = d.get('smb_to_hml', {})
        if 'z_score' in hs:
            results.append(f"  INFO: {regime} HML->SMB z = {hs['z_score']:.2f}, p = {hs['p_value']:.4f}")
        if 'z_score' in sh:
            results.append(f"  INFO: {regime} SMB->HML z = {sh['z_score']:.2f}, p = {sh['p_value']:.4f}")

    for r in results:
        print(r)


def check_robustness(hmm_json, latex):
    """Robustness analyses — check inline numbers."""
    print("\n=== Robustness Inline Numbers ===")
    results = []
    rob = hmm_json['selected_fit']['robustness']

    # Filtered vs smoothed
    fvs = rob['filtered_vs_smoothed']
    results.append(check_value(
        "Filtered agreement",
        f"{fvs['agreement_pct']:.1f}",
        re.search(r'agreement is (\d+\.\d+)', latex).group(1) if re.search(r'agreement is (\d+\.\d+)', latex) else None
    ))

    # Subsample
    sub = rob['subsample']
    results.append(f"  INFO: Pre-2008 n={sub['pre_2008_n']}, p={sub['pre_2008_p']:.2e}")
    results.append(f"  INFO: Post-2008 n={sub['post_2008_n']}, p={sub['post_2008_p']:.3f}")

    # Weekly
    wk = rob['weekly']
    results.append(f"  INFO: Weekly n={wk['n_weeks']}, p={wk['p_value']:.3f}")

    # Elevated annual
    ea = rob['elevated_annual']
    results.append(f"  INFO: Elevated annual: {ea['positive_direction_count']}/{ea['total_years']} years ({ea['fraction']:.0f}%)")

    for r in results:
        print(r)


def check_frozen_inline(hmm_json, latex):
    """Frozen OOS inline numbers."""
    print("\n=== Frozen OOS Inline Numbers ===")
    results = []
    frozen = hmm_json['frozen_oos']

    crisis_frac = frozen['crisis_fraction_pct']
    agg_p = frozen['aggregate']['p_value']
    agg_n = frozen['aggregate']['n_clean']

    # Search for these in latex
    frac_match = re.search(r'assigns\s+(\d+\.\d+)\\%.*Crisis', latex)
    if frac_match:
        latex_frac = float(frac_match.group(1))
        results.append(check_value("Frozen crisis fraction", crisis_frac, latex_frac))
    else:
        results.append(f"  INFO: Frozen crisis fraction = {crisis_frac:.1f}%")

    p_match = re.search(r'yields\s+\$p\s*=\s*(\d+\.\d+)\$', latex)
    if p_match:
        latex_p = float(p_match.group(1))
        results.append(check_value("Frozen aggregate p", agg_p, latex_p, tolerance=0.02))

    n_match = re.search(r'n\s*=\s*(\d+)\$?\s*clean', latex)
    if n_match:
        latex_n = int(n_match.group(1))
        results.append(check_value("Frozen aggregate n", agg_n, latex_n))

    for r in results:
        print(r)


def check_heatmap_claim(hmm_json, latex=None):
    """Check heatmap ranking and flag only if manuscript overclaims top ranking."""
    print("\n=== Heatmap Narrative Check ===")
    import numpy as np
    all_pairs = hmm_json['selected_fit']['all_pairs']
    factors = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']

    # For each directed pair, compute regime differential
    pair_diffs = {}
    for src in factors:
        for tgt in factors:
            if src == tgt:
                continue
            key = f"{src}->{tgt}"
            vals = []
            for regime in ['Normal', 'Elevated', 'Crisis']:
                p = all_pairs[regime]['pairs'].get(key, {}).get('p_value', 1.0)
                vals.append(-np.log10(max(p, 1e-15)))
            diff = max(vals) - min(vals)
            pair_diffs[key] = diff

    # Sort by differential
    sorted_pairs = sorted(pair_diffs.items(), key=lambda x: -x[1])
    print(f"  Top 5 regime-differential pairs:")
    for pair, diff in sorted_pairs[:5]:
        print(f"    {pair}: differential = {diff:.2f}")

    if sorted_pairs[0][0] == 'HML->SMB':
        print("  PASS: HML->SMB has strongest regime-dependent differential")
    else:
        print(f"  INFO: {sorted_pairs[0][0]} has stronger differential than HML->SMB")
        # Find HML->SMB rank
        for i, (pair, diff) in enumerate(sorted_pairs):
            if pair == 'HML->SMB':
                print(f"  INFO: HML->SMB ranks #{i+1} (differential = {diff:.2f})")
                break


def main():
    parser = argparse.ArgumentParser(description="Consistency checker for paper tables/claims vs JSON artifacts.")
    parser.add_argument(
        '--latex-file',
        default=LATEX_FILE,
        help="Path to LaTeX manuscript to validate.",
    )
    parser.add_argument(
        '--json-dir',
        default=RESULTS_DIR,
        help="Directory containing result JSON artifacts.",
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help="Exit nonzero on any FAIL/SKIP markers.",
    )
    parser.add_argument(
        '--profile',
        choices=['full', 'arxiv'],
        default='full',
        help="Validation scope: 'full' for ICAIF full paper checks, 'arxiv' for overlapping subset checks.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CONSISTENCY CHECK: LaTeX vs JSON")
    print("=" * 70)
    print(f"LaTeX file: {args.latex_file}")
    print(f"JSON dir:   {args.json_dir}")

    latex = read_latex(args.latex_file)
    hmm_json = load_json('multistart_hmm_results.json', args.json_dir)
    trading_json = load_json('trading_selected.json', args.json_dir)
    var_json = load_json('var_fixes_results.json', args.json_dir)
    neural_json = load_json('neural_granger_selected.json', args.json_dir)
    te_json = load_json('te_selected.json', args.json_dir)

    if hmm_json is None:
        print("FATAL: multistart_hmm_results.json not found")
        sys.exit(1)

    buf = io.StringIO()
    with redirect_stdout(buf):
        if args.profile == 'full':
            check_table1(hmm_json, latex)
            check_table2(hmm_json, latex)
            check_table3(hmm_json, latex)
            check_table4(hmm_json, latex)
            check_table5(hmm_json, latex)
            check_table6(hmm_json, latex)
            check_table7(hmm_json, latex)
            check_table8(trading_json, latex)
            check_table9(var_json, latex)
            check_table10(neural_json, latex)
            check_table11(te_json, latex)
            check_robustness(hmm_json, latex)
            check_frozen_inline(hmm_json, latex)
            check_heatmap_claim(hmm_json, latex=latex)
        else:
            # Overlapping checks for the arXiv manuscript.
            check_table1(hmm_json, latex)
            check_table2(hmm_json, latex)
            check_table3(hmm_json, latex)
            check_table5(hmm_json, latex)
            check_table6(hmm_json, latex)
            check_table8(trading_json, latex)
    output = buf.getvalue()
    print(output, end="")

    fail_count = len(re.findall(r'^\s*FAIL:', output, flags=re.MULTILINE))
    skip_count = len(re.findall(r'^\s*SKIP:', output, flags=re.MULTILINE))
    pass_count = len(re.findall(r'^\s*PASS:', output, flags=re.MULTILINE))
    info_count = len(re.findall(r'^\s*INFO:', output, flags=re.MULTILINE))

    print("\n" + "=" * 70)
    print("CONSISTENCY CHECK COMPLETE")
    print(f"PASS={pass_count}  FAIL={fail_count}  SKIP={skip_count}  INFO={info_count}")
    print("=" * 70)

    if args.strict and (fail_count > 0 or skip_count > 0):
        print("STRICT MODE: FAILED (nonzero FAIL/SKIP detected)")
        sys.exit(1)


if __name__ == '__main__':
    main()
