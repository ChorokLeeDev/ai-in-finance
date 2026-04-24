#!/usr/bin/env python3
"""
P0: Top-k Ablation Experiment
Compute Spearman ρ for different concentration metrics: k={1,2,3,5,10}, HHI, Gini, Entropy
Address TFmu's concern: "What happens if the concentration metric is defined using top 2, 3, or 5 features?"
"""

import json
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from scipy.special import rel_entr

RESULTS_DIR = Path(__file__).parent.parent / "results"


def gini_coefficient(arr):
    """Compute Gini coefficient of feature importance array."""
    arr = np.abs(arr)
    if arr.sum() == 0:
        return 0
    sorted_arr = np.sort(arr)
    n = len(arr)
    cumsum = np.cumsum(sorted_arr)
    return (2 * np.sum((np.arange(1, n+1) * sorted_arr)) / (n * arr.sum()) - (n + 1) / n)


def entropy_concentration(arr):
    """Compute entropy-based concentration (1 - normalized entropy)."""
    arr = np.abs(arr)
    if arr.sum() == 0:
        return 0
    p = arr / arr.sum()
    p = p[p > 0]  # Remove zeros
    ent = -np.sum(p * np.log(p))
    max_ent = np.log(len(arr))
    return 1 - (ent / max_ent)  # Higher = more concentrated


def effective_features(arr):
    """Compute effective number of features (exp of entropy)."""
    arr = np.abs(arr)
    if arr.sum() == 0:
        return len(arr)
    p = arr / arr.sum()
    p = p[p > 0]
    ent = -np.sum(p * np.log(p))
    return np.exp(ent)


def hhi(arr):
    """Herfindahl-Hirschman Index (sum of squared shares)."""
    arr = np.abs(arr)
    if arr.sum() == 0:
        return 0
    p = arr / arr.sum()
    return np.sum(p ** 2)


def top_k_concentration(arr, k):
    """Top-k concentration (sum of top-k / total)."""
    arr = np.abs(arr)
    if arr.sum() == 0:
        return 0
    sorted_arr = np.sort(arr)[::-1]
    return sorted_arr[:k].sum() / arr.sum()


def load_shap_importance():
    """Load all SALT task SHAP values and compute mean importance per feature."""
    shap_dir = RESULTS_DIR / "shap"

    tasks = {
        'sales-shipcond': 'rel-salt',
        'sales-group': 'rel-salt',
        'sales-payterms': 'rel-salt',
        'item-plant': 'rel-salt',
        'item-shippoint': 'rel-salt',
        'sales-incoterms': 'rel-salt',
        'item-incoterms': 'rel-salt',
        'sales-office': 'rel-salt'
    }

    importance = {}
    for task, dataset in tasks.items():
        pkl_path = shap_dir / f"shap_{dataset}_{task}.pkl"
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            # Use validation SHAP values (pre-deployment diagnostic)
            shap_val = data['shap_values_val']
            # Mean absolute SHAP per feature
            mean_abs_shap = np.abs(shap_val).mean(axis=0)
            importance[task] = mean_abs_shap
        else:
            print(f"Warning: {pkl_path} not found")

    return importance


def load_coverage_drops():
    """Load coverage drops from statistical_rigor.json."""
    with open(RESULTS_DIR / "statistical_rigor.json") as f:
        data = json.load(f)

    drops = {}
    for task in ['sales-shipcond', 'sales-group', 'sales-payterms', 'item-plant',
                 'item-shippoint', 'sales-incoterms', 'item-incoterms', 'sales-office']:
        if task in data:
            drops[task] = data[task]['coverage_drop']['mean']

    return drops


def load_external_results():
    """Load external validation results for n=16 analysis."""
    # Check if cross_domain_statistics.json has the needed info
    cross_domain_path = RESULTS_DIR / "cross_domain_statistics.json"
    if cross_domain_path.exists():
        with open(cross_domain_path) as f:
            data = json.load(f)
        return data
    return None


def compute_metrics(importance_dict, drops_dict):
    """Compute all concentration metrics for each task."""
    metrics = {
        'top_1': lambda arr: top_k_concentration(arr, 1) * 100,
        'top_2': lambda arr: top_k_concentration(arr, 2) * 100,
        'top_3': lambda arr: top_k_concentration(arr, 3) * 100,
        'top_5': lambda arr: top_k_concentration(arr, min(5, len(arr))) * 100,
        'top_10': lambda arr: top_k_concentration(arr, min(10, len(arr))) * 100,
        'hhi': lambda arr: hhi(arr) * 100,
        'gini': lambda arr: gini_coefficient(arr) * 100,
        'entropy_conc': lambda arr: entropy_concentration(arr) * 100,
        'eff_features': effective_features,
    }

    results = {m: {} for m in metrics}

    for task in importance_dict:
        arr = importance_dict[task]
        for metric_name, metric_fn in metrics.items():
            results[metric_name][task] = metric_fn(arr)

    return results


def compute_correlations(metric_values, drops_dict):
    """Compute Spearman correlation between metric values and coverage drops."""
    tasks = list(metric_values.keys())

    x = [metric_values[t] for t in tasks]
    y = [drops_dict[t] * 100 for t in tasks]  # Convert to percentage

    rho, p = spearmanr(x, y)
    return rho, p


def main():
    print("=" * 60)
    print("P0: Top-k Ablation Experiment")
    print("=" * 60)

    # Load data
    importance = load_shap_importance()
    drops = load_coverage_drops()

    print(f"\nLoaded {len(importance)} SALT tasks")
    print(f"Tasks: {list(importance.keys())}")

    # Compute all metrics
    all_metrics = compute_metrics(importance, drops)

    # Compute correlations
    print("\n" + "=" * 60)
    print("Table R1: Top-k Ablation Results (n=8 SALT)")
    print("=" * 60)
    print(f"{'Metric':<20} {'ρ':>8} {'p-value':>12} {'Interpretation'}")
    print("-" * 60)

    results_list = []
    for metric_name in ['top_1', 'top_2', 'top_3', 'top_5', 'top_10', 'hhi', 'gini', 'entropy_conc', 'eff_features']:
        metric_values = all_metrics[metric_name]
        rho, p = compute_correlations(metric_values, drops)

        # For effective features, correlation should be negative (more features = less vulnerable)
        if metric_name == 'eff_features':
            interpretation = "✓ Negative as expected" if rho < 0 else "✗ Unexpected positive"
        else:
            interpretation = "✓ Positive as expected" if rho > 0 else "✗ Unexpected"

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{metric_name:<20} {rho:>8.4f} {p:>10.6f}{sig:>2} {interpretation}")

        results_list.append({
            'metric': metric_name,
            'rho': float(rho),
            'p_value': float(p),
            'significant': bool(p < 0.05),
            'n': len(metric_values)
        })

    # Print detailed values per task
    print("\n" + "=" * 60)
    print("Detailed Values per Task")
    print("=" * 60)

    header = f"{'Task':<18}"
    for m in ['top_1', 'top_2', 'top_3', 'hhi', 'gini', 'eff_feat']:
        header += f" {m:>8}"
    header += f" {'Drop%':>8}"
    print(header)
    print("-" * 100)

    for task in sorted(importance.keys()):
        row = f"{task:<18}"
        for m in ['top_1', 'top_2', 'top_3', 'hhi', 'gini', 'eff_features']:
            row += f" {all_metrics[m][task]:>8.1f}"
        row += f" {drops[task]*100:>8.1f}"
        print(row)

    # Save results
    output = {
        'n_salt': 8,
        'correlations': results_list,
        'values_per_task': {task: {m: float(all_metrics[m][task]) for m in all_metrics} for task in importance},
        'coverage_drops_pct': {task: float(drops[task] * 100) for task in drops},
        'best_metric': max(results_list, key=lambda x: abs(x['rho']) if x['metric'] != 'eff_features' else -x['rho'])['metric'],
        'notes': {
            'top_1': 'SHAP concentration (current paper metric)',
            'top_k': 'Sum of top-k feature importance / total',
            'hhi': 'Herfindahl-Hirschman Index (sum of squared shares)',
            'gini': 'Gini coefficient of importance distribution',
            'entropy_conc': '1 - normalized entropy (higher = more concentrated)',
            'eff_features': 'Effective number of features (exp of entropy)'
        }
    }

    output_path = RESULTS_DIR / "topk_ablation.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Analysis summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    # Find best k
    topk_results = [(r['metric'], r['rho'], r['p_value']) for r in results_list if r['metric'].startswith('top_')]
    best_topk = max(topk_results, key=lambda x: x[1])
    print(f"\nBest top-k metric: {best_topk[0]} (ρ={best_topk[1]:.4f}, p={best_topk[2]:.4f})")

    # Compare with alternatives
    hhi_result = next(r for r in results_list if r['metric'] == 'hhi')
    gini_result = next(r for r in results_list if r['metric'] == 'gini')

    print(f"\nComparison with alternative metrics:")
    print(f"  HHI: ρ={hhi_result['rho']:.4f} (p={hhi_result['p_value']:.4f})")
    print(f"  Gini: ρ={gini_result['rho']:.4f} (p={gini_result['p_value']:.4f})")

    # Conclusion for TFmu
    top1_rho = next(r['rho'] for r in results_list if r['metric'] == 'top_1')
    print(f"\n" + "=" * 60)
    print("CONCLUSION FOR TFMU")
    print("=" * 60)
    if top1_rho >= best_topk[1] - 0.01:
        print(f"✓ Top-1 concentration (ρ={top1_rho:.4f}) achieves the highest or near-highest")
        print("  correlation among all top-k metrics.")
        print("  This supports top-1 as a principled choice, not ad-hoc.")
    else:
        print(f"✗ Top-1 (ρ={top1_rho:.4f}) is NOT the best metric.")
        print(f"  Better: {best_topk[0]} (ρ={best_topk[1]:.4f})")
        print("  Consider revising the paper to use this metric.")


if __name__ == "__main__":
    main()
