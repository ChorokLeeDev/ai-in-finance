"""
Experiment: FK vs Correlation-based Grouping (rel-f1 Regression)
=================================================================

Compare FK-based grouping with data-driven correlation clustering.

Key Question:
- Does FK grouping provide similar or better results than correlation clustering?

Methods:
1. FK Grouping: Use schema-defined foreign key relationships
2. Correlation Clustering: Group features by correlation matrix hierarchical clustering

Comparison Metrics:
- Attribution stability across runs
- Interpretability (FK has semantic meaning, correlation doesn't)
- Consistency with feature-level attribution
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from data_loader_f1 import load_f1_data
from ensemble_f1 import train_regression_ensemble, compute_regression_uncertainty


def create_correlation_groups(X: pd.DataFrame, n_groups: int = 5) -> Dict[str, str]:
    """
    Create feature groups based on correlation clustering.

    Uses hierarchical clustering on correlation matrix.

    Returns:
        col_to_group: column -> group_id mapping
    """
    # Compute correlation matrix
    corr_matrix = X.corr().values

    # Handle NaN correlations (constant columns)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Convert correlation to distance (1 - |corr|)
    dist_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(dist_matrix, 0)  # Distance to self is 0

    # Ensure symmetry and non-negativity
    dist_matrix = np.maximum(dist_matrix, 0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    # Hierarchical clustering
    condensed = squareform(dist_matrix)
    linkage_matrix = linkage(condensed, method='average')

    # Cut tree to get n_groups clusters
    cluster_labels = fcluster(linkage_matrix, n_groups, criterion='maxclust')

    # Create mapping
    col_to_group = {}
    for col, label in zip(X.columns, cluster_labels):
        col_to_group[col] = f"CORR_GROUP_{label}"

    return col_to_group


def compute_group_attribution(
    models: List,
    X: pd.DataFrame,
    col_to_group: Dict[str, str],
    n_perturbations: int = 10
) -> Dict[str, float]:
    """Compute group-level attribution via noise injection."""
    X_np = X.values
    baseline = compute_regression_uncertainty(models, X_np).mean()

    group_to_cols = defaultdict(list)
    for col, grp in col_to_group.items():
        group_to_cols[grp].append(col)

    feature_cols = list(X.columns)

    deltas = {}
    for grp, cols in group_to_cols.items():
        col_indices = [feature_cols.index(c) for c in cols if c in feature_cols]
        if not col_indices:
            continue

        grp_deltas = []
        for _ in range(n_perturbations):
            X_noisy = X_np.copy()
            for idx in col_indices:
                X_noisy[:, idx] = np.random.permutation(X_noisy[:, idx])

            noisy_unc = compute_regression_uncertainty(models, X_noisy).mean()
            grp_deltas.append(noisy_unc - baseline)

        deltas[grp] = np.mean(grp_deltas)

    # Normalize
    total_positive = sum(max(0, v) for v in deltas.values())
    if total_positive > 0:
        contributions = {grp: max(0, v) / total_positive * 100 for grp, v in deltas.items()}
    else:
        contributions = {grp: 0 for grp in deltas}

    return contributions


def run_comparison_experiment(
    task_name: str = 'driver-position',
    sample_size: int = 3000,
    n_models: int = 5,
    n_runs: int = 5
) -> Dict:
    """Run FK vs Correlation comparison experiment."""

    print("=" * 70)
    print("FK vs Correlation-based Grouping Comparison")
    print("=" * 70)
    print(f"Dataset: rel-f1, Task: {task_name}, Sample size: {sample_size}")
    print(f"Runs: {n_runs} (for stability measurement)")

    # Load data
    X, y, feature_cols, col_to_fk = load_f1_data(task_name, sample_size, use_cache=True)
    n_fks = len(set(col_to_fk.values()))
    print(f"Loaded {len(X)} samples, {len(feature_cols)} features, {n_fks} FK groups")

    # Create correlation groups (same number as FKs for fair comparison)
    col_to_corr = create_correlation_groups(X, n_groups=n_fks)
    n_corr = len(set(col_to_corr.values()))
    print(f"Created {n_corr} correlation-based groups")

    # Print group mappings
    print("\n--- FK Groups ---")
    fk_groups = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_groups[fk].append(col)
    for fk, cols in sorted(fk_groups.items()):
        print(f"  {fk}: {len(cols)} features")

    print("\n--- Correlation Groups ---")
    corr_groups = defaultdict(list)
    for col, grp in col_to_corr.items():
        corr_groups[grp].append(col)
    for grp, cols in sorted(corr_groups.items()):
        print(f"  {grp}: {cols[:3]}{'...' if len(cols) > 3 else ''}")

    # Run multiple trials for stability
    fk_results = []
    corr_results = []

    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")

        # Train ensemble with different seed
        np.random.seed(run * 100)
        models, X_test, y_test = train_regression_ensemble(
            X, y, n_models=n_models, random_state=42 + run * 10
        )

        # FK attribution
        fk_attr = compute_group_attribution(models, X, col_to_fk, n_perturbations=5)
        fk_results.append(fk_attr)

        # Correlation attribution
        corr_attr = compute_group_attribution(models, X, col_to_corr, n_perturbations=5)
        corr_results.append(corr_attr)

        print(f"  FK top: {max(fk_attr, key=fk_attr.get)} ({max(fk_attr.values()):.1f}%)")
        print(f"  Corr top: {max(corr_attr, key=corr_attr.get)} ({max(corr_attr.values()):.1f}%)")

    # Compute stability for each method
    print("\n" + "=" * 70)
    print("STABILITY ANALYSIS")
    print("=" * 70)

    def compute_stability(results: List[Dict]) -> float:
        """Compute average pairwise Spearman correlation."""
        groups = list(results[0].keys())
        correlations = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                v1 = [results[i].get(g, 0) for g in groups]
                v2 = [results[j].get(g, 0) for g in groups]
                if len(set(v1)) > 1 and len(set(v2)) > 1:
                    corr, _ = spearmanr(v1, v2)
                    if not np.isnan(corr):
                        correlations.append(corr)
        return np.mean(correlations) if correlations else 0.0

    fk_stability = compute_stability(fk_results)
    corr_stability = compute_stability(corr_results)

    print(f"\nFK Grouping Stability: {fk_stability:.3f}")
    print(f"Correlation Grouping Stability: {corr_stability:.3f}")

    # Average attribution per method
    print("\n--- Average Attribution (%) ---")
    print(f"\n{'FK Grouping':<20}")
    avg_fk = {}
    for fk in fk_results[0].keys():
        avg = np.mean([r[fk] for r in fk_results])
        std = np.std([r[fk] for r in fk_results])
        avg_fk[fk] = (avg, std)
        print(f"  {fk}: {avg:.1f}% ± {std:.1f}%")

    print(f"\n{'Correlation Grouping':<20}")
    avg_corr = {}
    for grp in corr_results[0].keys():
        avg = np.mean([r[grp] for r in corr_results])
        std = np.std([r[grp] for r in corr_results])
        avg_corr[grp] = (avg, std)
        print(f"  {grp}: {avg:.1f}% ± {std:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    better = "FK" if fk_stability > corr_stability else "Correlation"
    diff = abs(fk_stability - corr_stability)

    print(f"\nStability comparison:")
    print(f"  FK: {fk_stability:.3f}")
    print(f"  Correlation: {corr_stability:.3f}")
    print(f"  Winner: {better} (by {diff:.3f})")

    print(f"\nInterpretability:")
    print(f"  FK: Groups have semantic meaning (DRIVER, RACE, etc.)")
    print(f"  Correlation: Groups are data-driven (CORR_GROUP_1, etc.)")
    print(f"  Winner: FK (always)")

    # Verdict
    if fk_stability >= corr_stability - 0.1:  # FK wins if within 0.1 of correlation
        verdict = "FK GROUPING RECOMMENDED"
        reason = "Similar or better stability with superior interpretability"
    else:
        verdict = "CORRELATION GROUPING MORE STABLE"
        reason = f"But FK still recommended for interpretability (stability gap: {diff:.3f})"

    print(f"\nVerdict: {verdict}")
    print(f"Reason: {reason}")

    # Save results
    output = {
        'dataset': 'rel-f1',
        'task': task_name,
        'sample_size': sample_size,
        'n_runs': n_runs,
        'fk_stability': fk_stability,
        'corr_stability': corr_stability,
        'fk_avg_attribution': {k: v[0] for k, v in avg_fk.items()},
        'corr_avg_attribution': {k: v[0] for k, v in avg_corr.items()},
        'verdict': verdict,
        'reason': reason
    }

    output_path = Path(__file__).parent / 'results' / 'fk_vs_datadriven_f1.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='driver-position')
    parser.add_argument('--sample_size', type=int, default=3000)
    parser.add_argument('--n_models', type=int, default=5)
    parser.add_argument('--n_runs', type=int, default=5)
    args = parser.parse_args()

    run_comparison_experiment(
        task_name=args.task,
        sample_size=args.sample_size,
        n_models=args.n_models,
        n_runs=args.n_runs
    )
