"""
Experiment: Stability Test (rel-f1 Regression)
===============================================

Test stability of FK-level attribution across sample sizes.

Key test: Do rankings remain stable when scaling from n=1000 to n=5000?
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from scipy.stats import spearmanr

from data_loader_f1 import load_f1_data, get_fk_groups
from ensemble_f1 import train_regression_ensemble, compute_regression_uncertainty


def run_noise_injection_quick(
    X: pd.DataFrame,
    col_to_fk: Dict[str, str],
    n_models: int = 5,
    n_perturbations: int = 5
) -> Dict[str, float]:
    """Quick noise injection experiment."""
    from sklearn.model_selection import train_test_split

    # Get target (position) - for this quick test, use synthetic
    y = pd.Series(np.random.randn(len(X)))

    # Reload proper data
    X_full, y_full, _, _ = load_f1_data('driver-position', len(X), use_cache=True)

    # Train models
    models, X_test, y_test = train_regression_ensemble(X_full, y_full, n_models=n_models)

    # Baseline uncertainty
    baseline = compute_regression_uncertainty(models, X_full.values).mean()

    # FK groups
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)

    feature_cols = list(X_full.columns)

    results = {}
    for fk, cols in fk_to_cols.items():
        col_indices = [feature_cols.index(c) for c in cols if c in feature_cols]
        if not col_indices:
            continue

        deltas = []
        for _ in range(n_perturbations):
            X_noisy = X_full.values.copy()
            for idx in col_indices:
                X_noisy[:, idx] = np.random.permutation(X_noisy[:, idx])

            noisy_unc = compute_regression_uncertainty(models, X_noisy).mean()
            deltas.append(noisy_unc - baseline)

        results[fk] = np.mean(deltas)

    # Normalize
    total_positive = sum(max(0, v) for v in results.values())
    if total_positive > 0:
        contributions = {fk: max(0, v) / total_positive * 100 for fk, v in results.items()}
    else:
        contributions = {fk: 0 for fk in results}

    return contributions


def run_stability_test(sample_sizes: List[int], n_models: int = 5):
    """Test stability across different sample sizes."""
    print("="*70)
    print("FK-Level Risk Attribution: Stability Test (Regression)")
    print("="*70)
    print(f"Sample sizes: {sample_sizes}")
    print(f"Uncertainty measure: Ensemble variance")

    all_results = {}

    for size in sample_sizes:
        print(f"\n--- Sample size: {size} ---")

        # Load data
        X, y, feature_cols, col_to_fk = load_f1_data('driver-position', size, use_cache=True)
        print(f"Loaded {len(X)} samples, {len(feature_cols)} features")

        # Train and run noise injection
        models, X_test, y_test = train_regression_ensemble(X, y, n_models=n_models)

        # Baseline uncertainty
        baseline = compute_regression_uncertainty(models, X.values).mean()
        print(f"Baseline variance: {baseline:.4f}")

        # FK groups
        fk_to_cols = defaultdict(list)
        for col, fk in col_to_fk.items():
            fk_to_cols[fk].append(col)

        # Run noise injection
        results = {}
        for fk, cols in fk_to_cols.items():
            col_indices = [feature_cols.index(c) for c in cols if c in feature_cols]
            if not col_indices:
                continue

            deltas = []
            for _ in range(5):  # 5 perturbations
                X_noisy = X.values.copy()
                for idx in col_indices:
                    X_noisy[:, idx] = np.random.permutation(X_noisy[:, idx])

                noisy_unc = compute_regression_uncertainty(models, X_noisy).mean()
                deltas.append(noisy_unc - baseline)

            results[fk] = np.mean(deltas)

        # Normalize
        total_positive = sum(max(0, v) for v in results.values())
        if total_positive > 0:
            contributions = {fk: max(0, v) / total_positive * 100 for fk, v in results.items()}
        else:
            contributions = {fk: 0 for fk in results}

        all_results[size] = contributions

        print(f"Top FK: {max(contributions, key=contributions.get)} ({max(contributions.values()):.1f}%)")

    # Compare rankings across sample sizes
    print("\n" + "="*70)
    print("STABILITY ANALYSIS")
    print("="*70)

    # Print comparison table
    fks = list(all_results[sample_sizes[0]].keys())
    print(f"\n{'FK':<15}", end="")
    for size in sample_sizes:
        print(f"n={size:<8}", end="")
    print()
    print("-" * (15 + 12 * len(sample_sizes)))

    for fk in sorted(fks):
        print(f"{fk:<15}", end="")
        for size in sample_sizes:
            pct = all_results[size].get(fk, 0)
            print(f"{pct:>8.1f}%   ", end="")
        print()

    # Compute ranking stability
    print("\n--- Ranking Stability ---")
    rankings = {}
    for size in sample_sizes:
        sorted_fks = sorted(fks, key=lambda f: -all_results[size].get(f, 0))
        rankings[size] = {fk: rank for rank, fk in enumerate(sorted_fks)}

    # Pairwise Spearman correlations
    print("\nSpearman rank correlation between sample sizes:")
    for i, s1 in enumerate(sample_sizes):
        for j, s2 in enumerate(sample_sizes):
            if i < j:
                r1 = [rankings[s1][fk] for fk in fks]
                r2 = [rankings[s2][fk] for fk in fks]
                corr, pval = spearmanr(r1, r2)
                print(f"  n={s1} vs n={s2}: rho = {corr:.3f} (p={pval:.3f})")

    # Overall stability metric: average pairwise correlation
    correlations = []
    for i, s1 in enumerate(sample_sizes):
        for j, s2 in enumerate(sample_sizes):
            if i < j:
                r1 = [rankings[s1][fk] for fk in fks]
                r2 = [rankings[s2][fk] for fk in fks]
                corr, _ = spearmanr(r1, r2)
                correlations.append(corr)

    avg_stability = np.mean(correlations)
    print(f"\nOverall stability (avg Spearman rho): {avg_stability:.3f}")

    # Verdict
    if avg_stability > 0.8:
        verdict = "PASS - Rankings are stable"
    elif avg_stability > 0.5:
        verdict = "PARTIAL - Some stability"
    else:
        verdict = "FAIL - Rankings are unstable"

    print(f"Verdict: {verdict}")

    # Save results
    output = {
        'sample_sizes': sample_sizes,
        'results_by_size': all_results,
        'average_stability': avg_stability,
        'verdict': verdict
    }

    output_path = Path(__file__).parent / 'results' / 'stability_f1.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000, 2000, 3000, 5000])
    parser.add_argument('--n_models', type=int, default=5)
    args = parser.parse_args()

    run_stability_test(args.sizes, args.n_models)
