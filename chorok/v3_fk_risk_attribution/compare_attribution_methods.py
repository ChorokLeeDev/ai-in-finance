"""
Compare FK Attribution Methods
===============================

Run all attribution methods and compare results:
1. Our method: Leave-one-out FK attribution
2. SHAP: TreeExplainer-based attribution
3. Permutation: Permutation importance for uncertainty
4. VFA: Variance Feature Attribution

Comparison metrics:
- Spearman correlation between methods
- Top-3 FK ranking consistency
- Correlation with distribution shift (MMD)

Author: ChorokLeeDev
Created: 2025-11-28
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# Import our attribution methods
from fk_uncertainty_attribution import run_attribution_comparison
from shap_attribution import run_shap_comparison
from permutation_attribution import run_permutation_comparison
from vfa_attribution import run_vfa_comparison


def compute_spearman_correlation(dict1: Dict[str, float], dict2: Dict[str, float]) -> float:
    """Compute Spearman correlation between two attribution dictionaries."""
    common_keys = set(dict1.keys()) & set(dict2.keys())
    if len(common_keys) < 3:
        return np.nan

    values1 = [dict1[k] for k in common_keys]
    values2 = [dict2[k] for k in common_keys]

    corr, _ = spearmanr(values1, values2)
    return corr


def get_top_k(attr_dict: Dict[str, float], k: int = 3) -> List[str]:
    """Get top-k features by attribution (absolute value)."""
    sorted_items = sorted(attr_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    return [item[0] for item in sorted_items[:k]]


def compute_ranking_overlap(ranking1: List[str], ranking2: List[str]) -> float:
    """Compute overlap ratio between two rankings."""
    overlap = len(set(ranking1) & set(ranking2))
    return overlap / len(ranking1) if ranking1 else 0


def run_comparison(task_name: str, sample_size: int = 5000):
    """Run all attribution methods on a task and compare."""

    print(f"\n{'#'*70}")
    print(f"# COMPARISON: {task_name}")
    print(f"{'#'*70}")

    results = {}

    # 1. Our method (Leave-one-out)
    print("\n[1/4] Running Leave-One-Out FK Attribution...")
    try:
        loo_results = run_attribution_comparison(task_name, sample_size)
        results['leave_one_out'] = {
            'train': loo_results['train_contributions'],
            'val': loo_results['val_contributions'],
            'deltas': {k: v['delta'] for k, v in loo_results['deltas'].items()}
        }
    except Exception as e:
        print(f"  Error: {e}")
        results['leave_one_out'] = None

    # 2. SHAP
    print("\n[2/4] Running SHAP Attribution...")
    try:
        shap_results = run_shap_comparison(task_name, sample_size)
        results['shap'] = {
            'train': shap_results['train_attribution'],
            'val': shap_results['val_attribution'],
            'deltas': {k: v['delta'] for k, v in shap_results['deltas'].items()}
        }
    except Exception as e:
        print(f"  Error: {e}")
        results['shap'] = None

    # 3. Permutation
    print("\n[3/4] Running Permutation Attribution...")
    try:
        perm_results = run_permutation_comparison(task_name, sample_size)
        results['permutation'] = {
            'train': perm_results['train_attribution'],
            'val': perm_results['val_attribution'],
            'deltas': {k: v['delta_raw'] for k, v in perm_results['deltas'].items()}
        }
    except Exception as e:
        print(f"  Error: {e}")
        results['permutation'] = None

    # 4. VFA
    print("\n[4/4] Running VFA Attribution...")
    try:
        vfa_results = run_vfa_comparison(task_name, sample_size, n_models=5)
        results['vfa'] = {
            'train': vfa_results['train_attribution'],
            'val': vfa_results['val_attribution'],
            'deltas': {k: v['delta_raw'] for k, v in vfa_results['deltas'].items()}
        }
    except Exception as e:
        print(f"  Error: {e}")
        results['vfa'] = None

    # Compare methods
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")

    methods = ['leave_one_out', 'shap', 'permutation', 'vfa']
    available_methods = [m for m in methods if results.get(m) is not None]

    if len(available_methods) < 2:
        print("Not enough methods succeeded for comparison")
        return results

    # Spearman correlations (on delta values)
    print("\n--- Spearman Correlation (Delta Values) ---")
    correlation_matrix = {}

    for i, m1 in enumerate(available_methods):
        for m2 in available_methods[i+1:]:
            corr = compute_spearman_correlation(
                results[m1]['deltas'],
                results[m2]['deltas']
            )
            key = f"{m1} vs {m2}"
            correlation_matrix[key] = corr
            print(f"  {key}: {corr:.3f}")

    # Top-3 ranking overlap
    print("\n--- Top-3 FK Ranking (by |delta|) ---")
    rankings = {}

    for m in available_methods:
        top3 = get_top_k(results[m]['deltas'], k=3)
        rankings[m] = top3
        print(f"  {m}: {top3}")

    print("\n--- Ranking Overlap ---")
    overlap_matrix = {}
    for i, m1 in enumerate(available_methods):
        for m2 in available_methods[i+1:]:
            overlap = compute_ranking_overlap(rankings[m1], rankings[m2])
            key = f"{m1} vs {m2}"
            overlap_matrix[key] = overlap
            print(f"  {key}: {overlap:.1%}")

    # Combine results
    comparison = {
        'task': task_name,
        'results': results,
        'correlations': correlation_matrix,
        'rankings': rankings,
        'ranking_overlap': overlap_matrix
    }

    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sales-group")
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--all_tasks", action="store_true")
    args = parser.parse_args()

    output_dir = Path("chorok/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_tasks:
        tasks = [
            "sales-office", "sales-group", "sales-payterms",
            "sales-shipcond", "sales-incoterms",
            "item-plant", "item-shippoint", "item-incoterms"
        ]
    else:
        tasks = [args.task]

    all_comparisons = {}

    for task in tasks:
        try:
            comparison = run_comparison(task, args.sample_size)
            all_comparisons[task] = comparison
        except Exception as e:
            print(f"Error on {task}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_file = output_dir / "attribution_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(all_comparisons, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Final Summary
    print("\n" + "#"*70)
    print("# FINAL SUMMARY")
    print("#"*70)

    # Aggregate correlations
    all_corrs = {
        'leave_one_out vs shap': [],
        'leave_one_out vs permutation': [],
        'leave_one_out vs vfa': [],
        'shap vs permutation': [],
        'shap vs vfa': [],
        'permutation vs vfa': []
    }

    for task, comp in all_comparisons.items():
        if 'correlations' in comp:
            for key, corr in comp['correlations'].items():
                if key in all_corrs and not np.isnan(corr):
                    all_corrs[key].append(corr)

    print("\n--- Average Spearman Correlations Across Tasks ---")
    for key, corrs in all_corrs.items():
        if corrs:
            mean_corr = np.mean(corrs)
            print(f"  {key}: {mean_corr:.3f} (n={len(corrs)})")

    # Key finding summary
    print("\n--- Key Finding ---")
    loo_vs_shap = all_corrs.get('leave_one_out vs shap', [])
    if loo_vs_shap:
        mean_corr = np.mean(loo_vs_shap)
        if mean_corr < 0.5:
            print(f"  Leave-one-out ≠ SHAP (r={mean_corr:.2f})")
            print("  → Our uncertainty attribution captures DIFFERENT information than feature importance!")
        else:
            print(f"  Leave-one-out ≈ SHAP (r={mean_corr:.2f})")
            print("  → Methods are similar")


if __name__ == "__main__":
    main()
