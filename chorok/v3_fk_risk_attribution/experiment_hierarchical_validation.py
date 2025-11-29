"""
Hierarchical Attribution Validation
====================================

Validate Level 2 (Feature) and Level 3 (Entity) attribution:
1. Stability: Are rankings consistent across runs?
2. Statistical significance: Filter small samples, compute confidence intervals
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from collections import defaultdict
from scipy.stats import spearmanr, ttest_ind
import json
from pathlib import Path

from data_loader import load_salt_data
from ensemble import train_ensemble, compute_uncertainty
from hierarchical_attribution import HierarchicalAttribution


def validate_level2_stability(
    X: pd.DataFrame,
    y: pd.Series,
    col_to_fk: Dict[str, str],
    n_runs: int = 5,
    n_models: int = 5
) -> Dict:
    """
    Validate Level 2 (Feature within FK) stability.
    """
    print("\n" + "=" * 60)
    print("LEVEL 2 STABILITY VALIDATION")
    print("=" * 60)

    # Get FKs with multiple features
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)

    multi_feature_fks = {fk: cols for fk, cols in fk_to_cols.items() if len(cols) > 1}
    print(f"FKs with multiple features: {list(multi_feature_fks.keys())}")

    if not multi_feature_fks:
        print("No FKs with multiple features to validate")
        return {'status': 'skipped', 'reason': 'no multi-feature FKs'}

    results = {}

    for fk, cols in multi_feature_fks.items():
        print(f"\n--- Validating {fk} ({len(cols)} features) ---")

        rankings = []

        for run in range(n_runs):
            # Train with different seed
            models, _, _ = train_ensemble(X, y, n_models=n_models, random_state=42 + run)

            # Create attribution
            ha = HierarchicalAttribution(models, X, col_to_fk, n_perturbations=5)

            # Get Level 2 attribution
            l2_result = ha.attribute_level2(fk)
            ranking = sorted(l2_result.contributions.keys(),
                           key=lambda k: -l2_result.contributions[k])
            rankings.append(ranking)

            top_feature = ranking[0]
            top_pct = l2_result.contributions[top_feature]
            print(f"  Run {run+1}: top = {top_feature} ({top_pct:.1f}%)")

        # Compute stability
        if len(rankings) >= 2:
            correlations = []
            for i in range(len(rankings)):
                for j in range(i + 1, len(rankings)):
                    r1_ranks = [rankings[i].index(c) for c in cols]
                    r2_ranks = [rankings[j].index(c) for c in cols]
                    corr, _ = spearmanr(r1_ranks, r2_ranks)
                    if not np.isnan(corr):
                        correlations.append(corr)

            stability = np.mean(correlations) if correlations else 0
        else:
            stability = 1.0

        # Check if top feature is consistent
        top_features = [r[0] for r in rankings]
        top_consistency = max(top_features.count(f) for f in set(top_features)) / len(top_features)

        results[fk] = {
            'stability': stability,
            'top_consistency': top_consistency,
            'rankings': rankings
        }

        print(f"  Stability (Spearman): {stability:.3f}")
        print(f"  Top feature consistency: {top_consistency:.1%}")

    # Summary
    print("\n" + "=" * 60)
    print("LEVEL 2 SUMMARY")
    print("=" * 60)
    print(f"{'FK':<20} {'Stability':<12} {'Top Consistency':<15}")
    print("-" * 50)
    for fk, res in results.items():
        print(f"{fk:<20} {res['stability']:>10.3f} {res['top_consistency']:>13.1%}")

    avg_stability = np.mean([r['stability'] for r in results.values()])
    print(f"\nAverage stability: {avg_stability:.3f}")

    if avg_stability > 0.8:
        verdict = "PASS - Level 2 is stable"
    elif avg_stability > 0.5:
        verdict = "MARGINAL - Level 2 is partially stable"
    else:
        verdict = "FAIL - Level 2 is unstable"

    print(f"Verdict: {verdict}")

    return {'results': results, 'avg_stability': avg_stability, 'verdict': verdict}


def validate_level3_significance(
    X: pd.DataFrame,
    y: pd.Series,
    col_to_fk: Dict[str, str],
    n_models: int = 5,
    min_samples: int = 10
) -> Dict:
    """
    Validate Level 3 (Entity) with statistical significance.
    """
    print("\n" + "=" * 60)
    print("LEVEL 3 SIGNIFICANCE VALIDATION")
    print("=" * 60)
    print(f"Minimum samples per entity: {min_samples}")

    # Train models
    models, _, _ = train_ensemble(X, y, n_models=n_models)
    ha = HierarchicalAttribution(models, X, col_to_fk)

    # Compute per-sample uncertainty
    sample_uncertainty = compute_uncertainty(models, X.values, method='entropy')
    baseline_unc = sample_uncertainty.mean()
    print(f"Baseline uncertainty: {baseline_unc:.4f}")

    results = {}

    for col in X.columns:
        unique_vals = X[col].unique()

        # Filter by sample size
        valid_vals = []
        for val in unique_vals:
            mask = X[col] == val
            if mask.sum() >= min_samples:
                valid_vals.append(val)

        if len(valid_vals) < 2:
            continue

        print(f"\n--- {col} ({len(valid_vals)} entities with n>={min_samples}) ---")

        entity_stats = []
        for val in valid_vals:
            mask = X[col] == val
            unc = sample_uncertainty[mask]
            entity_stats.append({
                'value': val,
                'n': mask.sum(),
                'mean_unc': unc.mean(),
                'std_unc': unc.std(),
                'uncertainties': unc
            })

        # Sort by uncertainty
        entity_stats.sort(key=lambda x: -x['mean_unc'])

        # Statistical test: compare highest vs lowest
        if len(entity_stats) >= 2:
            high = entity_stats[0]
            low = entity_stats[-1]

            t_stat, p_value = ttest_ind(high['uncertainties'], low['uncertainties'])

            print(f"  Highest: {col}={high['value']} (unc={high['mean_unc']:.3f}, n={high['n']})")
            print(f"  Lowest:  {col}={low['value']} (unc={low['mean_unc']:.3f}, n={low['n']})")
            print(f"  t-test: t={t_stat:.2f}, p={p_value:.4f}")

            if p_value < 0.05:
                significance = "SIGNIFICANT"
            else:
                significance = "NOT SIGNIFICANT"
            print(f"  Result: {significance}")

            results[col] = {
                'n_valid_entities': len(valid_vals),
                'highest': {'value': high['value'], 'mean': high['mean_unc'], 'n': high['n']},
                'lowest': {'value': low['value'], 'mean': low['mean_unc'], 'n': low['n']},
                't_stat': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

    # Summary
    print("\n" + "=" * 60)
    print("LEVEL 3 SUMMARY")
    print("=" * 60)

    n_significant = sum(1 for r in results.values() if r['significant'])
    n_total = len(results)

    print(f"Features with significant entity differences: {n_significant}/{n_total}")

    for col, res in results.items():
        sig = "***" if res['significant'] else ""
        print(f"  {col}: p={res['p_value']:.4f} {sig}")

    if n_significant > n_total / 2:
        verdict = "PASS - Entity-level differences are significant"
    elif n_significant > 0:
        verdict = "MARGINAL - Some entity differences are significant"
    else:
        verdict = "FAIL - No significant entity differences"

    print(f"\nVerdict: {verdict}")

    return {'results': results, 'n_significant': n_significant, 'n_total': n_total, 'verdict': verdict}


def run_full_validation(
    task_name: str = 'sales-group',
    sample_size: int = 500,
    n_models: int = 5,
    n_runs: int = 5
):
    """Run full validation of hierarchical attribution."""

    print("=" * 60)
    print("HIERARCHICAL ATTRIBUTION VALIDATION")
    print("=" * 60)

    # Load data
    X, y, feature_cols, col_to_fk = load_salt_data(task_name, sample_size)
    print(f"Data: {X.shape}, Features: {len(feature_cols)}")

    # Validate Level 2
    l2_results = validate_level2_stability(X, y, col_to_fk, n_runs=n_runs, n_models=n_models)

    # Validate Level 3
    l3_results = validate_level3_significance(X, y, col_to_fk, n_models=n_models, min_samples=10)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Level 2 (Feature): {l2_results.get('verdict', 'N/A')}")
    print(f"Level 3 (Entity):  {l3_results.get('verdict', 'N/A')}")

    # Save results
    output = {
        'task': task_name,
        'sample_size': sample_size,
        'level2': {
            'avg_stability': l2_results.get('avg_stability'),
            'verdict': l2_results.get('verdict')
        },
        'level3': {
            'n_significant': l3_results.get('n_significant'),
            'n_total': l3_results.get('n_total'),
            'verdict': l3_results.get('verdict')
        }
    }

    output_path = Path(__file__).parent / 'results' / 'hierarchical_validation.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='sales-group')
    parser.add_argument('--sample_size', type=int, default=500)
    parser.add_argument('--n_models', type=int, default=5)
    parser.add_argument('--n_runs', type=int, default=5)
    args = parser.parse_args()

    run_full_validation(
        task_name=args.task,
        sample_size=args.sample_size,
        n_models=args.n_models,
        n_runs=args.n_runs
    )
