"""
Unified Experiments for rel-stack (FK-Level Risk Attribution)
==============================================================

Run all validation experiments on rel-stack dataset.
Domain: Q&A Platform (Stack Overflow)
Task: post-votes (regression - predict popularity)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from data_loader_stack import load_stack_data
from ensemble_f1 import train_regression_ensemble, compute_regression_uncertainty


def run_decomposition(
    models: List,
    X: pd.DataFrame,
    col_to_fk: Dict[str, str],
    n_perturbations: int = 10
) -> Dict[str, float]:
    """Noise injection attribution."""
    X_np = X.values
    baseline = compute_regression_uncertainty(models, X_np).mean()

    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)

    feature_cols = list(X.columns)

    deltas = {}
    for fk, cols in fk_to_cols.items():
        col_indices = [feature_cols.index(c) for c in cols if c in feature_cols]
        if not col_indices:
            continue

        fk_deltas = []
        for _ in range(n_perturbations):
            X_noisy = X_np.copy()
            for idx in col_indices:
                X_noisy[:, idx] = np.random.permutation(X_noisy[:, idx])

            noisy_unc = compute_regression_uncertainty(models, X_noisy).mean()
            fk_deltas.append(noisy_unc - baseline)

        deltas[fk] = np.mean(fk_deltas)

    # Normalize
    total_positive = sum(max(0, v) for v in deltas.values())
    if total_positive > 0:
        contributions = {fk: max(0, v) / total_positive * 100 for fk, v in deltas.items()}
    else:
        contributions = {fk: 0 for fk in deltas}

    return contributions


def run_stability_test(sample_sizes: List[int], n_models: int = 5):
    """Test stability across sample sizes."""
    all_results = {}

    for size in sample_sizes:
        print(f"\n--- Sample size: {size} ---")
        X, y, feature_cols, col_to_fk = load_stack_data('post-votes', size, use_cache=True)
        models, X_test, y_test = train_regression_ensemble(X, y, n_models=n_models)

        contributions = run_decomposition(models, X, col_to_fk, n_perturbations=5)
        all_results[size] = contributions

        print(f"Top FK: {max(contributions, key=contributions.get)} ({max(contributions.values()):.1f}%)")

    # Compute stability
    fks = list(all_results[sample_sizes[0]].keys())
    correlations = []
    for i, s1 in enumerate(sample_sizes):
        for j, s2 in enumerate(sample_sizes):
            if i < j:
                r1 = [all_results[s1].get(fk, 0) for fk in fks]
                r2 = [all_results[s2].get(fk, 0) for fk in fks]
                if len(set(r1)) > 1 and len(set(r2)) > 1:
                    corr, _ = spearmanr(r1, r2)
                    if not np.isnan(corr):
                        correlations.append(corr)
                        print(f"  n={s1} vs n={s2}: ρ = {corr:.3f}")

    avg_stability = np.mean(correlations) if correlations else 0.0
    print(f"\nOverall stability: {avg_stability:.3f}")

    return all_results, avg_stability


def run_all_experiments():
    """Run all experiments on rel-stack."""
    print("=" * 70)
    print("FK-Level Risk Attribution: rel-stack Validation")
    print("=" * 70)
    print("Domain: Q&A Platform (Stack Overflow)")
    print("Task: post-votes (predict post popularity)")

    results = {}

    # 1. Decomposition
    print("\n[1] DECOMPOSITION EXPERIMENT")
    print("-" * 50)
    X, y, feature_cols, col_to_fk = load_stack_data('post-votes', 3000, use_cache=True)
    print(f"Loaded {len(X)} samples, {len(feature_cols)} features")

    models, X_test, y_test = train_regression_ensemble(X, y, n_models=5)
    contributions = run_decomposition(models, X, col_to_fk)

    print("\nFK Contributions:")
    for fk, pct in sorted(contributions.items(), key=lambda x: -x[1]):
        print(f"  {fk}: {pct:.1f}%")

    results['decomposition'] = contributions

    # 2. Stability
    print("\n[2] STABILITY EXPERIMENT")
    print("-" * 50)
    stability_results, avg_stability = run_stability_test([1000, 2000, 3000, 5000])
    results['stability'] = avg_stability

    # 3. Summary
    print("\n" + "=" * 70)
    print("SUMMARY: rel-stack")
    print("=" * 70)

    top_fk = max(contributions, key=contributions.get)
    print(f"\nTop FK: {top_fk} ({contributions[top_fk]:.1f}%)")
    print(f"Stability: {avg_stability:.3f}")

    verdict = "PASS" if avg_stability > 0.7 else "PARTIAL" if avg_stability > 0.5 else "FAIL"
    print(f"Verdict: {verdict}")

    results['top_fk'] = top_fk
    results['verdict'] = verdict

    # Save
    output_path = Path(__file__).parent / 'results' / 'stack_validation.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_all_experiments()
