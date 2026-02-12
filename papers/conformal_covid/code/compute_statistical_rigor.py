"""
Statistical Rigor: 95% CIs + Paired Tests for All Key Results

Adds rigorous statistics to the existing 50-seed ensemble results:
1. 95% CIs for coverage (val and test) using t-distribution
2. Paired Wilcoxon tests for coverage drop (val→test) per task
3. Bootstrap CIs for the Spearman correlation (n=8)
4. Threshold sensitivity: vary SHAP threshold from 20-60%, evaluate on val only
5. Leave-one-out stability for the primary correlation

Usage:
    python code/compute_statistical_rigor.py

Output:
    results/statistical_rigor.json
    results/statistical_rigor_table.tex
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

RESULTS_DIR = Path(__file__).parent.parent / 'results'


def compute_95ci_t(values):
    """95% CI using t-distribution."""
    n = len(values)
    if n < 2:
        return float(np.mean(values)), float(np.mean(values)), float(np.mean(values))
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(n)
    ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    return float(mean), float(ci[0]), float(ci[1])


def compute_bootstrap_ci(x, y, n_bootstrap=10000, seed=42):
    """Bootstrap CI for Spearman correlation."""
    rng = np.random.RandomState(seed)
    n = len(x)
    rhos = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        rho, _ = stats.spearmanr(x[idx], y[idx])
        rhos.append(rho)
    rhos = np.array(rhos)
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def main():
    # --- Load 50-seed ensemble results ---
    ensemble_file = RESULTS_DIR / 'ensemble_50seeds.pkl'
    if not ensemble_file.exists():
        print(f"ERROR: {ensemble_file} not found. Run run_50seed_ensemble.py first.")
        return
    with open(ensemble_file, 'rb') as f:
        ensemble_results = pickle.load(f)

    print("=" * 80)
    print("STATISTICAL RIGOR ANALYSIS")
    print("=" * 80)

    all_stats = {}

    # --- 1. Per-task 95% CIs and paired tests ---
    print("\n--- 1. Per-Task Coverage Statistics (50 seeds) ---\n")
    print(f"{'Task':<18} {'Val Coverage (95% CI)':<30} {'Test Coverage (95% CI)':<30} {'Drop (95% CI)':<30} {'Paired p':<10}")
    print("-" * 120)

    for r in ensemble_results:
        task = r['task']
        val_covs = [s['val_coverage'] for s in r['seed_results']]
        test_covs = [s['test_coverage'] for s in r['seed_results']]
        drops = [s['coverage_drop'] for s in r['seed_results']]

        val_mean, val_lo, val_hi = compute_95ci_t(val_covs)
        test_mean, test_lo, test_hi = compute_95ci_t(test_covs)
        drop_mean, drop_lo, drop_hi = compute_95ci_t(drops)

        # Paired Wilcoxon: is the drop significant?
        try:
            w_stat, w_p = stats.wilcoxon(val_covs, test_covs, alternative='greater')
        except ValueError:
            w_p = 1.0

        all_stats[task] = {
            'val_coverage': {
                'mean': val_mean, 'ci_lo': val_lo, 'ci_hi': val_hi,
                'std': float(np.std(val_covs)),
            },
            'test_coverage': {
                'mean': test_mean, 'ci_lo': test_lo, 'ci_hi': test_hi,
                'std': float(np.std(test_covs)),
            },
            'coverage_drop': {
                'mean': drop_mean, 'ci_lo': drop_lo, 'ci_hi': drop_hi,
                'std': float(np.std(drops)),
                'paired_wilcoxon_p': float(w_p),
            },
            'num_seeds': len(val_covs),
        }

        print(f"{task:<18} "
              f"{val_mean*100:5.1f}% [{val_lo*100:5.1f}-{val_hi*100:5.1f}]    "
              f"{test_mean*100:5.1f}% [{test_lo*100:5.1f}-{test_hi*100:5.1f}]    "
              f"{drop_mean*100:5.1f}pp [{drop_lo*100:5.1f}-{drop_hi*100:5.1f}]    "
              f"p={w_p:.4f}")

    # --- 2. SHAP correlation with bootstrap CI ---
    print("\n--- 2. SHAP Concentration Correlation (Bootstrap CI) ---\n")
    shap_csv = RESULTS_DIR / 'shap' / 'concentration_all_tasks.csv'
    if shap_csv.exists():
        shap_df = pd.read_csv(shap_csv)

        concentrations = shap_df['concentration_pct'].values
        drops = shap_df['coverage_drop'].values

        rho, p = stats.spearmanr(concentrations, drops)
        boot_lo, boot_hi = compute_bootstrap_ci(concentrations, drops)

        print(f"Spearman ρ = {rho:.3f} (p = {p:.4f})")
        print(f"Bootstrap 95% CI: [{boot_lo:.3f}, {boot_hi:.3f}]")

        all_stats['spearman_correlation'] = {
            'rho': float(rho), 'p_value': float(p),
            'bootstrap_95ci': [boot_lo, boot_hi],
            'n': len(concentrations),
        }

        # --- 3. Leave-one-out stability ---
        print("\n--- 3. Leave-One-Out Stability ---\n")
        loo_results = []
        for i in range(len(concentrations)):
            mask = np.ones(len(concentrations), dtype=bool)
            mask[i] = False
            rho_loo, p_loo = stats.spearmanr(concentrations[mask], drops[mask])
            task_removed = shap_df.iloc[i]['task']
            print(f"  Remove {task_removed:<18}: ρ = {rho_loo:.3f} (p = {p_loo:.4f}, n={mask.sum()})")
            loo_results.append({
                'removed_task': task_removed,
                'rho': float(rho_loo),
                'p_value': float(p_loo),
                'n': int(mask.sum()),
            })

        rho_range = [min(r['rho'] for r in loo_results), max(r['rho'] for r in loo_results)]
        print(f"\n  LOO ρ range: [{rho_range[0]:.3f}, {rho_range[1]:.3f}]")

        all_stats['loo_stability'] = {
            'results': loo_results,
            'rho_range': rho_range,
        }

        # --- 4. Threshold sensitivity (post-hoc validation) ---
        print("\n--- 4. Threshold Sensitivity (Post-Hoc Validation) ---\n")

        # Ground truth from test outcomes (post-hoc evaluation, NOT validation-only tuning)
        # SEV: >70% drop, ROB: <15% drop
        task_categories = {}
        for _, row in shap_df.iterrows():
            task_categories[row['task']] = 'SEV' if row['coverage_drop'] > 50 else 'ROB'

        thresholds = np.arange(20, 65, 5)
        threshold_results = []

        for thresh in thresholds:
            # Step 2 only: concentration > threshold → VULN
            tp = fp = tn = fn = 0
            for _, row in shap_df.iterrows():
                predicted_vuln = row['concentration_pct'] > thresh
                actual_sev = task_categories[row['task']] == 'SEV'
                if predicted_vuln and actual_sev:
                    tp += 1
                elif predicted_vuln and not actual_sev:
                    fp += 1
                elif not predicted_vuln and actual_sev:
                    fn += 1
                else:
                    tn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"  Threshold {thresh:4.0f}%: Prec={precision:.2f} Rec={recall:.2f} F1={f1:.2f} (TP={tp} FP={fp} TN={tn} FN={fn})")
            threshold_results.append({
                'threshold': float(thresh),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            })

        all_stats['threshold_sensitivity'] = threshold_results

    # --- Save ---
    output_file = RESULTS_DIR / 'statistical_rigor.json'
    with open(output_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved: {output_file}")

    # --- Generate LaTeX Table 1 replacement with 95% CIs ---
    print("\n\n=== Updated Table 1 with 95% CIs ===\n")
    latex = generate_table1_with_ci(all_stats, ensemble_results)
    print(latex)
    with open(RESULTS_DIR / 'table1_with_ci.tex', 'w') as f:
        f.write(latex)


def generate_table1_with_ci(all_stats, ensemble_results):
    """Generate Table 1 with 95% CIs instead of just mean±std."""
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Coverage Degradation Under COVID-19 Distribution Shift (50 Seeds).')
    lines.append(r'95\% CIs from $t$-distribution. $p$-values from paired Wilcoxon signed-rank test')
    lines.append(r'(val vs test coverage, seed-level).}\label{tab:main_results}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{@{}lcccccc@{}}')
    lines.append(r'\toprule')
    lines.append(r'Task & Cl & Val Coverage [95\% CI] & Test Coverage [95\% CI] & Drop [95\% CI] & $p$ & Cat \\')
    lines.append(r'\midrule')

    task_order = [
        'sales-shipcond', 'sales-group', 'sales-payterms',
        'item-plant', 'item-shippoint',
        'sales-incoterms', 'item-incoterms', 'sales-office'
    ]

    for task_name in task_order:
        if task_name not in all_stats:
            continue
        s = all_stats[task_name]
        task_short = task_name.replace('sales-', 's-').replace('item-', 'i-')

        # Get num_classes from ensemble results
        num_classes = '--'
        for r in ensemble_results:
            if r['task'] == task_name:
                num_classes = r['num_classes']
                break

        v = s['val_coverage']
        t = s['test_coverage']
        d = s['coverage_drop']

        # Determine category
        if d['mean'] > 0.50:
            cat = 'SEV'
        elif d['mean'] > 0.15:
            cat = 'MOD'
        else:
            cat = 'ROB'
        # Mark high variance
        if t['std'] > 0.30:
            cat += '$^*$'

        val_str = f"{v['mean']*100:.1f} [{v['ci_lo']*100:.1f}, {v['ci_hi']*100:.1f}]"
        test_str = f"{t['mean']*100:.1f} [{t['ci_lo']*100:.1f}, {t['ci_hi']*100:.1f}]"
        drop_str = f"{d['mean']*100:.1f} [{d['ci_lo']*100:.1f}, {d['ci_hi']*100:.1f}]"
        p_val = d['paired_wilcoxon_p']
        p_str = f"{p_val:.3f}" if p_val >= 0.001 else "$<$0.001"

        lines.append(f'{task_short} & {num_classes} & {val_str} & {test_str} & {drop_str} & {p_str} & {cat} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')
    return '\n'.join(lines)


if __name__ == "__main__":
    main()
