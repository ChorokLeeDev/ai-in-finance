"""
Compute ICC per task and partial Spearman correlation controlling for class cardinality.

Addresses reviewer concerns about:
1. Pseudo-replication: 50 seeds within each task are highly correlated
2. Confounding: Does SHAP concentration predict coverage drop after controlling for num_classes?

ICC computation uses one-way random effects ICC(1,1):
    ICC = (MSB - MSW) / (MSB + (k-1)*MSW)
where MSB = between-subjects mean square, MSW = within-subjects mean square,
k = number of measurements per subject.

Here "subjects" are the 50 seeds, and "measurements" are val and test coverage
(k=2). Alternatively, we compute ICC on test coverages alone (treating seeds
as repeated measures of the same task) using the variance decomposition approach.

For partial Spearman: rank all variables, then compute partial correlation on ranks.

Usage:
    python code/compute_icc_and_partial.py

Output:
    results/icc_and_partial.json
"""

import json
import math
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

RESULTS_DIR = Path(__file__).parent.parent / 'results'

# Class counts per task (from ensemble_50seeds_summary.json and task definitions)
CLASS_COUNTS = {
    'sales-shipcond': 45,
    'sales-group': 459,  # summary says 459
    'sales-payterms': 137,  # summary says 137
    'item-plant': 35,
    'item-shippoint': 69,  # summary says 69
    'sales-incoterms': 13,
    'item-incoterms': 13,
    'sales-office': 25,
}


def compute_icc_one_way(values):
    """
    Compute one-way random effects ICC(1,1) for a single group of repeated measurements.

    For a single task with n=50 seed coverages, this measures how much of the total
    variance is due to the "task identity" vs. random seed variation.

    For a single group, we use the Shrout & Fleiss ICC(1,1) formulation adapted
    for the case where we want to know how correlated any two randomly chosen
    seeds are within the same task.

    ICC = (MSB - MSW) / (MSB + (k-1)*MSW)

    But with a single task and 50 seeds measuring val/test pairs, we treat
    each seed as a "subject" and val/test as two "raters" (k=2).
    """
    # values is shape (n_seeds, 2) where columns are [val_coverage, test_coverage]
    n = values.shape[0]  # number of seeds (subjects)
    k = values.shape[1]  # number of measurements per seed (raters), here k=2

    # Grand mean
    grand_mean = np.mean(values)

    # Subject (seed) means
    subject_means = np.mean(values, axis=1)

    # Between-subjects sum of squares
    SSB = k * np.sum((subject_means - grand_mean) ** 2)
    dfB = n - 1

    # Within-subjects sum of squares
    SSW = np.sum((values - subject_means[:, np.newaxis]) ** 2)
    dfW = n * (k - 1)

    MSB = SSB / dfB
    MSW = SSW / dfW

    # ICC(1,1) = (MSB - MSW) / (MSB + (k-1)*MSW)
    icc = (MSB - MSW) / (MSB + (k - 1) * MSW)

    return float(icc), float(MSB), float(MSW)


def compute_icc_single_measure(coverages):
    """
    Compute ICC for a single vector of repeated measurements (50 test coverages).

    This is the "reliability" measure: how similar are the 50 seeds?
    We use the variance decomposition approach.

    For a single set of n measurements from the same task:
    ICC = 1 - (var_within / var_total)

    But since all measurements come from the same task, var_between_tasks = 0
    and ICC would be trivially 0 or undefined.

    Instead, what reviewers actually want is: the INTRACLASS CORRELATION
    measuring the correlation between any two randomly drawn seeds for the
    same task. This requires a two-level model:

    For each task, the 50 test coverages have some variance sigma^2_within.
    Across tasks, the mean coverages vary with sigma^2_between.

    ICC = sigma^2_between / (sigma^2_between + sigma^2_within)
    """
    # This needs data from ALL tasks together
    pass  # Handled in main


def compute_icc_multilevel(task_coverages):
    """
    Compute ICC from a nested/multilevel model: seeds nested within tasks.

    task_coverages: dict of {task_name: array of 50 coverage values}

    ICC = sigma^2_between / (sigma^2_between + sigma^2_within)

    Using one-way ANOVA decomposition:
    ICC = (MSB - MSW) / (MSB + (n0 - 1) * MSW)

    where MSB = between-group mean square, MSW = within-group mean square,
    n0 = common group size (50 here).
    """
    groups = list(task_coverages.values())
    k = len(groups)  # number of tasks
    n_per_group = [len(g) for g in groups]
    n0 = n_per_group[0]  # all same size (50)
    N = sum(n_per_group)

    # Grand mean
    all_values = np.concatenate(groups)
    grand_mean = np.mean(all_values)

    # Between-group sum of squares
    group_means = [np.mean(g) for g in groups]
    SSB = sum(n * (gm - grand_mean) ** 2 for n, gm in zip(n_per_group, group_means))
    dfB = k - 1
    MSB = SSB / dfB

    # Within-group sum of squares
    SSW = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)
    dfW = N - k
    MSW = SSW / dfW

    # ICC(1,1) for one-way random effects
    icc = (MSB - MSW) / (MSB + (n0 - 1) * MSW)

    # 95% CI using F-distribution (Shrout & Fleiss, 1979)
    F_obs = MSB / MSW
    F_lo = F_obs / stats.f.ppf(0.975, dfB, dfW)
    F_hi = F_obs / stats.f.ppf(0.025, dfB, dfW)

    icc_lo = (F_lo - 1) / (F_lo + n0 - 1)
    icc_hi = (F_hi - 1) / (F_hi + n0 - 1)

    return {
        'icc': float(icc),
        'icc_ci_lo': float(icc_lo),
        'icc_ci_hi': float(icc_hi),
        'MSB': float(MSB),
        'MSW': float(MSW),
        'F': float(F_obs),
        'dfB': int(dfB),
        'dfW': int(dfW),
        'n_groups': int(k),
        'n_per_group': int(n0),
    }


def compute_per_task_icc(task_val_test_pairs):
    """
    Per-task ICC(1,1): for each task, how correlated are the val and test
    coverages across 50 seeds?

    This uses the (seed x [val, test]) matrix for each task.
    """
    results = {}
    for task, pairs in task_val_test_pairs.items():
        # pairs is (50, 2) array: [val_coverage, test_coverage]
        icc, MSB, MSW = compute_icc_one_way(pairs)
        results[task] = {
            'icc_val_test': float(icc),
            'MSB': float(MSB),
            'MSW': float(MSW),
        }
    return results


def partial_spearman(x, y, z):
    """
    Partial Spearman correlation between x and y, controlling for z.

    Method: rank all three variables, then compute partial Pearson on ranks.

    partial_r(x,y|z) = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2) * (1 - r_yz^2))

    Then test using t = r * sqrt((n-2-1) / (1-r^2)) with df = n-2-1
    """
    n = len(x)

    # Rank the variables
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)

    # Pearson correlations on ranks
    r_xy, _ = stats.pearsonr(rx, ry)
    r_xz, _ = stats.pearsonr(rx, rz)
    r_yz, _ = stats.pearsonr(ry, rz)

    # Partial correlation
    numerator = r_xy - r_xz * r_yz
    denominator = math.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))

    if abs(denominator) < 1e-12:
        return float('nan'), float('nan')

    partial_r = numerator / denominator

    # Significance test: t-test with df = n - 2 - 1 (controlling for 1 variable)
    df = n - 2 - 1
    if df <= 0:
        return float(partial_r), float('nan')

    t_stat = partial_r * math.sqrt(df / (1 - partial_r ** 2))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    return float(partial_r), float(p_value)


def main():
    # --- Load 50-seed ensemble results ---
    ensemble_file = RESULTS_DIR / 'ensemble_50seeds.pkl'
    if not ensemble_file.exists():
        print(f"ERROR: {ensemble_file} not found.")
        return
    with open(ensemble_file, 'rb') as f:
        ensemble_results = pickle.load(f)

    print("=" * 80)
    print("ICC AND PARTIAL CORRELATION ANALYSIS")
    print("=" * 80)

    # --- Extract per-seed data ---
    task_val_coverages = {}
    task_test_coverages = {}
    task_drops = {}
    task_val_test_pairs = {}

    for r in ensemble_results:
        task = r['task']
        val_covs = np.array([s['val_coverage'] for s in r['seed_results']])
        test_covs = np.array([s['test_coverage'] for s in r['seed_results']])
        drops = np.array([s['coverage_drop'] for s in r['seed_results']])

        task_val_coverages[task] = val_covs
        task_test_coverages[task] = test_covs
        task_drops[task] = drops
        task_val_test_pairs[task] = np.column_stack([val_covs, test_covs])

    # =====================================================================
    # 1. ICC: Seeds nested within tasks (multilevel)
    # =====================================================================
    print("\n" + "=" * 80)
    print("1. MULTILEVEL ICC: Seeds nested within Tasks")
    print("=" * 80)
    print("\nThis ICC measures what fraction of total variance in test coverage")
    print("is between tasks vs. within tasks (across seeds).")
    print("High ICC => seeds are pseudo-replications (contribute little new info).\n")

    # ICC on test coverages
    icc_test = compute_icc_multilevel(task_test_coverages)
    print(f"ICC (test coverages): {icc_test['icc']:.4f}")
    print(f"  95% CI: [{icc_test['icc_ci_lo']:.4f}, {icc_test['icc_ci_hi']:.4f}]")
    print(f"  MSB (between tasks) = {icc_test['MSB']:.6f}")
    print(f"  MSW (within tasks)  = {icc_test['MSW']:.6f}")
    print(f"  F({icc_test['dfB']}, {icc_test['dfW']}) = {icc_test['F']:.2f}")

    # ICC on coverage drops
    icc_drop = compute_icc_multilevel(task_drops)
    print(f"\nICC (coverage drops): {icc_drop['icc']:.4f}")
    print(f"  95% CI: [{icc_drop['icc_ci_lo']:.4f}, {icc_drop['icc_ci_hi']:.4f}]")
    print(f"  MSB (between tasks) = {icc_drop['MSB']:.6f}")
    print(f"  MSW (within tasks)  = {icc_drop['MSW']:.6f}")
    print(f"  F({icc_drop['dfB']}, {icc_drop['dfW']}) = {icc_drop['F']:.2f}")

    # ICC on val coverages
    icc_val = compute_icc_multilevel(task_val_coverages)
    print(f"\nICC (val coverages): {icc_val['icc']:.4f}")
    print(f"  95% CI: [{icc_val['icc_ci_lo']:.4f}, {icc_val['icc_ci_hi']:.4f}]")

    # =====================================================================
    # 2. Per-task variance decomposition
    # =====================================================================
    print("\n" + "=" * 80)
    print("2. PER-TASK SEED VARIANCE")
    print("=" * 80)
    print(f"\n{'Task':<18} {'Test Mean':<12} {'Test Std':<12} {'Test CV':<12} {'Drop Mean':<12} {'Drop Std':<12}")
    print("-" * 78)

    per_task_stats = {}
    for task in sorted(task_test_coverages.keys()):
        test_covs = task_test_coverages[task]
        drops = task_drops[task]
        test_mean = np.mean(test_covs)
        test_std = np.std(test_covs, ddof=1)
        test_cv = test_std / test_mean if test_mean > 0.001 else float('inf')
        drop_mean = np.mean(drops)
        drop_std = np.std(drops, ddof=1)

        per_task_stats[task] = {
            'test_mean': float(test_mean),
            'test_std': float(test_std),
            'test_cv': float(test_cv),
            'drop_mean': float(drop_mean),
            'drop_std': float(drop_std),
            'n_seeds': len(test_covs),
        }

        print(f"{task:<18} {test_mean:.4f}      {test_std:.4f}      {test_cv:.4f}      {drop_mean:.4f}      {drop_std:.4f}")

    # =====================================================================
    # 3. Effective sample size
    # =====================================================================
    print("\n" + "=" * 80)
    print("3. EFFECTIVE SAMPLE SIZE")
    print("=" * 80)
    print("\nDesign effect = 1 + (k-1) * ICC, where k = 50 seeds")
    print("Effective n = total_n / design_effect\n")

    k = 50  # seeds per task

    # For coverage drops (the primary outcome)
    de_drop = 1 + (k - 1) * icc_drop['icc']
    eff_n_drop = (k * len(task_drops)) / de_drop
    print(f"Coverage drops:")
    print(f"  ICC = {icc_drop['icc']:.4f}")
    print(f"  Design effect = 1 + 49 * {icc_drop['icc']:.4f} = {de_drop:.2f}")
    print(f"  Total observations = {k} seeds x {len(task_drops)} tasks = {k * len(task_drops)}")
    print(f"  Effective n = {k * len(task_drops)} / {de_drop:.2f} = {eff_n_drop:.1f}")
    print(f"  (Effective n per task = {eff_n_drop / len(task_drops):.2f})")

    # For test coverages
    de_test = 1 + (k - 1) * icc_test['icc']
    eff_n_test = (k * len(task_test_coverages)) / de_test
    print(f"\nTest coverages:")
    print(f"  ICC = {icc_test['icc']:.4f}")
    print(f"  Design effect = {de_test:.2f}")
    print(f"  Effective n = {eff_n_test:.1f}")

    # Per-task effective n for the paired Wilcoxon test
    # The paired test uses 50 (val, test) pairs per task.
    # If the ICC within each task (val vs test correlation) is high,
    # the effective sample size for the paired test is reduced.
    print("\n\nPer-task effective n for paired Wilcoxon (val-test pairs):")
    print(f"{'Task':<18} {'ICC(val,test)':<16} {'Design Effect':<16} {'Eff n (of 50)':<16}")
    print("-" * 66)

    per_task_icc = compute_per_task_icc(task_val_test_pairs)
    for task in sorted(per_task_icc.keys()):
        icc_vt = per_task_icc[task]['icc_val_test']
        # For paired test, the design effect uses the correlation between
        # the two measurements. With k=2 (val and test):
        # Design effect = 1 + (k-1)*ICC = 1 + ICC
        de = 1 + icc_vt
        eff = 50 / de if de > 0 else 50
        per_task_icc[task]['design_effect'] = float(de)
        per_task_icc[task]['effective_n'] = float(eff)
        print(f"{task:<18} {icc_vt:>8.4f}        {de:>8.2f}          {eff:>8.1f}")

    # =====================================================================
    # 4. Partial Spearman correlation
    # =====================================================================
    print("\n" + "=" * 80)
    print("4. PARTIAL SPEARMAN CORRELATION")
    print("=" * 80)
    print("\nDoes SHAP concentration predict coverage drop AFTER controlling")
    print("for log(num_classes)?\n")

    # Load SHAP concentration data
    shap_csv = RESULTS_DIR / 'shap' / 'concentration_all_tasks.csv'
    shap_df = pd.read_csv(shap_csv)

    # Build aligned arrays
    tasks = shap_df['task'].values
    concentrations = shap_df['concentration_pct'].values
    drops = shap_df['coverage_drop'].values

    # Get num_classes for each task (use summary JSON values, override with CLASS_COUNTS)
    num_classes = np.array([CLASS_COUNTS.get(t, 0) for t in tasks], dtype=float)
    log_num_classes = np.log(num_classes)

    print("Data alignment check:")
    print(f"{'Task':<18} {'Concentration':<16} {'Drop':<12} {'Num Classes':<14} {'Log(Classes)':<14}")
    print("-" * 72)
    for i, t in enumerate(tasks):
        print(f"{t:<18} {concentrations[i]:>8.1f}%       {drops[i]:>6.1f}%     {num_classes[i]:>6.0f}        {log_num_classes[i]:>6.3f}")

    # Bivariate Spearman correlations
    rho_cd, p_cd = stats.spearmanr(concentrations, drops)
    rho_cn, p_cn = stats.spearmanr(concentrations, log_num_classes)
    rho_dn, p_dn = stats.spearmanr(drops, log_num_classes)

    print(f"\nBivariate Spearman correlations:")
    print(f"  Concentration vs Drop:         rho = {rho_cd:.4f}, p = {p_cd:.4f}")
    print(f"  Concentration vs Log(Classes): rho = {rho_cn:.4f}, p = {p_cn:.4f}")
    print(f"  Drop vs Log(Classes):          rho = {rho_dn:.4f}, p = {p_dn:.4f}")

    # Partial Spearman: concentration vs drop | log(num_classes)
    partial_rho, partial_p = partial_spearman(concentrations, drops, log_num_classes)
    print(f"\nPartial Spearman (concentration vs drop | log(num_classes)):")
    print(f"  Partial rho = {partial_rho:.4f}")
    print(f"  p-value     = {partial_p:.4f}")
    print(f"  df          = {len(tasks) - 3}")

    # Also: partial controlling for num_classes (not logged)
    partial_rho_raw, partial_p_raw = partial_spearman(concentrations, drops, num_classes)
    print(f"\nPartial Spearman (concentration vs drop | num_classes, not logged):")
    print(f"  Partial rho = {partial_rho_raw:.4f}")
    print(f"  p-value     = {partial_p_raw:.4f}")

    # Also compute: drop vs log(num_classes) | concentration
    partial_rho_nc, partial_p_nc = partial_spearman(drops, log_num_classes, concentrations)
    print(f"\nPartial Spearman (drop vs log(num_classes) | concentration):")
    print(f"  Partial rho = {partial_rho_nc:.4f}")
    print(f"  p-value     = {partial_p_nc:.4f}")

    # =====================================================================
    # 5. Summary interpretation
    # =====================================================================
    print("\n" + "=" * 80)
    print("5. INTERPRETATION SUMMARY")
    print("=" * 80)

    print(f"""
PSEUDO-REPLICATION ANALYSIS:
  - ICC for test coverages: {icc_test['icc']:.4f}
    => {icc_test['icc']*100:.1f}% of variance is between tasks, {(1-icc_test['icc'])*100:.1f}% within tasks (across seeds)
  - ICC for coverage drops: {icc_drop['icc']:.4f}
    => {icc_drop['icc']*100:.1f}% of variance is between tasks, {(1-icc_drop['icc'])*100:.1f}% within tasks
  - Design effect for drops: {de_drop:.1f}
  - Effective n for 400 observations: {eff_n_drop:.1f}
  - This confirms that the 50 seeds provide limited independent information.
  - The effective unit of analysis is the TASK (n=8), not the seed.

CONFOUNDING CHECK:
  - Bivariate rho(concentration, drop) = {rho_cd:.3f} (p={p_cd:.4f})
  - After controlling for log(num_classes):
    Partial rho(concentration, drop | log_classes) = {partial_rho:.3f} (p={partial_p:.4f})
  - The {'concentration effect persists' if partial_p < 0.10 else 'concentration effect is attenuated'} after controlling for class cardinality.
  - log(num_classes) alone explains: rho(drop, log_classes) = {rho_dn:.3f} (p={p_dn:.4f})
""")

    # =====================================================================
    # Save results
    # =====================================================================
    output = {
        'icc_multilevel': {
            'test_coverages': icc_test,
            'coverage_drops': icc_drop,
            'val_coverages': icc_val,
        },
        'effective_sample_size': {
            'coverage_drops': {
                'icc': icc_drop['icc'],
                'design_effect': float(de_drop),
                'total_n': k * len(task_drops),
                'effective_n': float(eff_n_drop),
                'effective_n_per_task': float(eff_n_drop / len(task_drops)),
            },
            'test_coverages': {
                'icc': icc_test['icc'],
                'design_effect': float(de_test),
                'total_n': k * len(task_test_coverages),
                'effective_n': float(eff_n_test),
            },
        },
        'per_task_icc_val_test': {
            task: {
                'icc_val_test': d['icc_val_test'],
                'design_effect': d['design_effect'],
                'effective_n': d['effective_n'],
            }
            for task, d in per_task_icc.items()
        },
        'per_task_variance': per_task_stats,
        'partial_correlation': {
            'concentration_vs_drop': {
                'rho': float(rho_cd),
                'p': float(p_cd),
                'method': 'Spearman',
            },
            'concentration_vs_log_num_classes': {
                'rho': float(rho_cn),
                'p': float(p_cn),
                'method': 'Spearman',
            },
            'drop_vs_log_num_classes': {
                'rho': float(rho_dn),
                'p': float(p_dn),
                'method': 'Spearman',
            },
            'concentration_vs_drop_controlling_log_num_classes': {
                'rho': float(partial_rho),
                'p': float(partial_p),
                'df': int(len(tasks) - 3),
                'method': 'Partial Spearman (rank then partial Pearson)',
            },
            'concentration_vs_drop_controlling_num_classes': {
                'rho': float(partial_rho_raw),
                'p': float(partial_p_raw),
                'method': 'Partial Spearman (rank then partial Pearson, raw classes)',
            },
            'drop_vs_log_num_classes_controlling_concentration': {
                'rho': float(partial_rho_nc),
                'p': float(partial_p_nc),
                'method': 'Partial Spearman',
            },
        },
        'data_used': {
            'tasks': list(tasks),
            'concentrations': [float(c) for c in concentrations],
            'drops': [float(d) for d in drops],
            'num_classes': [int(n) for n in num_classes],
            'log_num_classes': [float(l) for l in log_num_classes],
        },
        'class_counts_source': 'ensemble_50seeds_summary.json',
    }

    output_file = RESULTS_DIR / 'icc_and_partial.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
