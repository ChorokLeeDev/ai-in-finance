"""
Experiment 3: FK Grouping vs Data-Driven Grouping
==================================================

Compare FK-based grouping (domain knowledge) with data-driven alternatives:
- Correlation clustering: group features by their correlation
- Random grouping: baseline

Measure: Stability of attribution across multiple runs (different seeds)
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

from cache import save_cache, load_cache
from data_loader import load_salt_data, get_fk_group
from ensemble import train_ensemble, compute_uncertainty


def get_fk_grouping(col_to_fk: Dict[str, str]) -> Dict[str, List[str]]:
    """Get FK-based grouping."""
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)
    return dict(fk_to_cols)


def get_correlation_grouping(X: pd.DataFrame, n_groups: int) -> Dict[str, List[str]]:
    """
    Group features by correlation using hierarchical clustering.
    """
    # Compute correlation matrix
    corr_matrix = X.corr().abs().values
    np.fill_diagonal(corr_matrix, 0)

    # Convert to distance matrix
    dist_matrix = 1 - corr_matrix

    # Handle NaN values
    dist_matrix = np.nan_to_num(dist_matrix, nan=1.0)

    # Ensure symmetry
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    np.fill_diagonal(dist_matrix, 0)

    # Hierarchical clustering
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method='average')
    labels = fcluster(Z, n_groups, criterion='maxclust')

    # Create groups
    groups = defaultdict(list)
    for col, label in zip(X.columns, labels):
        groups[f"CORR_{label}"].append(col)

    return dict(groups)


def get_random_grouping(columns: List[str], n_groups: int, seed: int) -> Dict[str, List[str]]:
    """Random grouping of features."""
    np.random.seed(seed)
    shuffled = np.random.permutation(columns)

    groups = defaultdict(list)
    for i, col in enumerate(shuffled):
        group_id = i % n_groups
        groups[f"RAND_{group_id}"].append(col)

    return dict(groups)


def compute_attribution(
    models: List,
    X: pd.DataFrame,
    grouping: Dict[str, List[str]],
    n_perturbations: int = 10
) -> Dict[str, float]:
    """Compute attribution for a given grouping."""
    X_np = X.values
    feature_cols = list(X.columns)
    baseline = compute_uncertainty(models, X_np, method='entropy').mean()

    results = {}
    for group, cols in grouping.items():
        col_indices = [feature_cols.index(c) for c in cols if c in feature_cols]
        if not col_indices:
            continue

        deltas = []
        for _ in range(n_perturbations):
            X_noisy = X_np.copy()
            for idx in col_indices:
                X_noisy[:, idx] = np.random.permutation(X_noisy[:, idx])
            noisy = compute_uncertainty(models, X_noisy, method='entropy').mean()
            deltas.append(noisy - baseline)

        results[group] = np.mean(deltas)

    # Normalize
    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def compute_stability(rankings: List[List[str]]) -> float:
    """
    Compute stability as average pairwise Spearman correlation.
    """
    if len(rankings) < 2:
        return 1.0

    correlations = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            # Convert rankings to numeric
            r1 = rankings[i]
            r2 = rankings[j]

            # Find common items
            common = set(r1) & set(r2)
            if len(common) < 3:
                continue

            # Get ranks
            rank1 = [r1.index(item) for item in common]
            rank2 = [r2.index(item) for item in common]

            corr, _ = spearmanr(rank1, rank2)
            if not np.isnan(corr):
                correlations.append(corr)

    return np.mean(correlations) if correlations else 0.0


def run_comparison(
    task_name: str = 'sales-group',
    sample_size: int = 500,
    n_models: int = 5,
    n_runs: int = 5
) -> Dict:
    """Run FK vs Data-driven comparison."""

    print("=" * 60)
    print("Experiment 3: FK vs Data-Driven Grouping")
    print("=" * 60)
    print(f"Task: {task_name}, Sample: {sample_size}, Runs: {n_runs}")

    # Load data
    X, y, feature_cols, col_to_fk = load_salt_data(task_name, sample_size)
    n_fk_groups = len(set(col_to_fk.values()))
    print(f"Features: {len(feature_cols)}, FK groups: {n_fk_groups}")

    # Create groupings
    fk_grouping = get_fk_grouping(col_to_fk)
    corr_grouping = get_correlation_grouping(X, n_fk_groups)

    print(f"\nFK Grouping: {list(fk_grouping.keys())}")
    print(f"Corr Grouping: {list(corr_grouping.keys())}")

    # Run multiple times with different seeds
    fk_rankings = []
    corr_rankings = []
    rand_rankings = []

    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} (seed={42 + run}) ---")

        # Train ensemble with different seed
        np.random.seed(42 + run)
        models, _, _ = train_ensemble(X, y, n_models=n_models, random_state=42 + run)

        # FK attribution
        fk_attr = compute_attribution(models, X, fk_grouping, n_perturbations=5)
        fk_ranking = sorted(fk_attr.keys(), key=lambda k: -fk_attr[k])
        fk_rankings.append(fk_ranking)
        print(f"  FK top: {fk_ranking[:3]}")

        # Correlation attribution
        corr_attr = compute_attribution(models, X, corr_grouping, n_perturbations=5)
        corr_ranking = sorted(corr_attr.keys(), key=lambda k: -corr_attr[k])
        corr_rankings.append(corr_ranking)
        print(f"  Corr top: {corr_ranking[:3]}")

        # Random attribution (different random grouping each run)
        rand_grouping = get_random_grouping(feature_cols, n_fk_groups, seed=42 + run)
        rand_attr = compute_attribution(models, X, rand_grouping, n_perturbations=5)
        rand_ranking = sorted(rand_attr.keys(), key=lambda k: -rand_attr[k])
        rand_rankings.append(rand_ranking)

    # Compute stability
    fk_stability = compute_stability(fk_rankings)
    corr_stability = compute_stability(corr_rankings)
    rand_stability = compute_stability(rand_rankings)

    print("\n" + "=" * 60)
    print("STABILITY COMPARISON")
    print("=" * 60)
    print(f"{'Method':<20} {'Stability (Spearman)':<20}")
    print("-" * 40)
    print(f"{'FK Grouping':<20} {fk_stability:>18.3f}")
    print(f"{'Correlation':<20} {corr_stability:>18.3f}")
    print(f"{'Random':<20} {rand_stability:>18.3f}")

    # Determine winner
    if fk_stability > corr_stability and fk_stability > rand_stability:
        verdict = "FK WINS - Domain knowledge is more stable!"
    elif corr_stability > fk_stability:
        verdict = "CORRELATION WINS - Data-driven is more stable"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\nVerdict: {verdict}")

    # Save results
    output = {
        'task': task_name,
        'sample_size': sample_size,
        'n_runs': n_runs,
        'fk_stability': fk_stability,
        'corr_stability': corr_stability,
        'rand_stability': rand_stability,
        'fk_rankings': fk_rankings,
        'corr_rankings': corr_rankings,
        'verdict': verdict
    }

    output_path = Path(__file__).parent / 'results' / 'fk_vs_datadriven.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='sales-group')
    parser.add_argument('--sample_size', type=int, default=500)
    parser.add_argument('--n_models', type=int, default=5)
    parser.add_argument('--n_runs', type=int, default=5)
    args = parser.parse_args()

    run_comparison(
        task_name=args.task,
        sample_size=args.sample_size,
        n_models=args.n_models,
        n_runs=args.n_runs
    )
