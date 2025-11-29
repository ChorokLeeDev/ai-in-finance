"""
Experiment 2: Calibration Validation
=====================================

Test if FK attribution is calibrated:
- If we attribute X% to FK_i, fixing FK_i should reduce uncertainty by ~X%

This validates that our attribution is actionable and predictive.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr

from cache import save_cache, load_cache
from data_loader import load_salt_data, get_fk_group, print_fk_summary
from ensemble import train_ensemble, compute_uncertainty


def get_fk_columns(col_to_fk: Dict[str, str]) -> Dict[str, List[str]]:
    """Get FK groups with their columns."""
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)
    return dict(fk_to_cols)


def compute_noise_attribution(
    models: List,
    X: pd.DataFrame,
    col_to_fk: Dict[str, str],
    n_perturbations: int = 10
) -> Dict[str, float]:
    """
    Compute FK attribution using noise injection.
    Returns contribution percentages.
    """
    X_np = X.values
    baseline_uncertainty = compute_uncertainty(models, X_np, method='entropy').mean()

    fk_to_cols = get_fk_columns(col_to_fk)
    feature_cols = list(X.columns)

    results = {}
    for fk, cols in fk_to_cols.items():
        col_indices = [feature_cols.index(c) for c in cols if c in feature_cols]
        if not col_indices:
            continue

        deltas = []
        for _ in range(n_perturbations):
            X_noisy = X_np.copy()
            for idx in col_indices:
                X_noisy[:, idx] = np.random.permutation(X_noisy[:, idx])

            noisy_uncertainty = compute_uncertainty(models, X_noisy, method='entropy').mean()
            deltas.append(noisy_uncertainty - baseline_uncertainty)

        results[fk] = np.mean(deltas)

    # Normalize to percentages
    total_positive = sum(max(0, v) for v in results.values())
    if total_positive > 0:
        contributions = {fk: max(0, v) / total_positive * 100 for fk, v in results.items()}
    else:
        contributions = {fk: 0 for fk in results}

    return contributions, baseline_uncertainty


def compute_fix_reduction(
    models: List,
    X: pd.DataFrame,
    col_to_fk: Dict[str, str],
    baseline_uncertainty: float
) -> Dict[str, float]:
    """
    Compute actual uncertainty reduction when each FK is "fixed".
    Fix = replace with mode (most frequent value) for each column.

    Returns reduction percentages relative to baseline.
    """
    X_np = X.values
    fk_to_cols = get_fk_columns(col_to_fk)
    feature_cols = list(X.columns)

    # Compute mode for each column
    col_modes = []
    for col in feature_cols:
        mode_val = X[col].mode().iloc[0] if len(X[col].mode()) > 0 else X[col].median()
        col_modes.append(mode_val)
    col_modes = np.array(col_modes)

    results = {}
    for fk, cols in fk_to_cols.items():
        col_indices = [feature_cols.index(c) for c in cols if c in feature_cols]
        if not col_indices:
            continue

        # Fix FK columns with mode values
        X_fixed = X_np.copy()
        for idx in col_indices:
            X_fixed[:, idx] = col_modes[idx]

        fixed_uncertainty = compute_uncertainty(models, X_fixed, method='entropy').mean()

        # Reduction = baseline - fixed (positive means reduced)
        reduction = baseline_uncertainty - fixed_uncertainty
        reduction_pct = (reduction / baseline_uncertainty) * 100 if baseline_uncertainty > 0 else 0

        results[fk] = reduction_pct

    return results


def run_calibration_experiment(
    task_name: str = 'sales-group',
    sample_size: int = 500,
    n_models: int = 5
) -> Dict:
    """Run calibration validation experiment."""

    print("=" * 60)
    print("Experiment 2: Calibration Validation")
    print("=" * 60)
    print(f"Task: {task_name}, Sample size: {sample_size}")

    # Load data
    X, y, feature_cols, col_to_fk = load_salt_data(task_name, sample_size)
    print(f"Features: {len(feature_cols)}, FK groups: {len(set(col_to_fk.values()))}")

    # Train ensemble
    print("\n[1] Training ensemble...")
    models, X_test, y_test = train_ensemble(X, y, n_models=n_models)
    print(f"Trained {len(models)} models")

    # Compute attribution
    print("\n[2] Computing Noise Injection attribution...")
    attribution, baseline = compute_noise_attribution(models, X, col_to_fk)
    print(f"Baseline uncertainty: {baseline:.4f}")
    print("\nAttribution (%):")
    for fk, pct in sorted(attribution.items(), key=lambda x: -x[1]):
        print(f"  {fk}: {pct:.1f}%")

    # Compute actual reduction
    print("\n[3] Computing actual reduction when FK is fixed...")
    reduction = compute_fix_reduction(models, X, col_to_fk, baseline)
    print("\nActual reduction (%):")
    for fk, pct in sorted(reduction.items(), key=lambda x: -x[1]):
        print(f"  {fk}: {pct:+.1f}%")

    # Compare attribution vs reduction
    print("\n" + "=" * 60)
    print("CALIBRATION COMPARISON")
    print("=" * 60)
    print(f"{'FK':<15} {'Attribution':<12} {'Reduction':<12} {'Diff':<10}")
    print("-" * 50)

    fks = sorted(set(attribution.keys()) & set(reduction.keys()))
    attr_vals = []
    red_vals = []

    for fk in fks:
        attr = attribution.get(fk, 0)
        red = reduction.get(fk, 0)
        diff = attr - red
        attr_vals.append(attr)
        red_vals.append(red)
        print(f"{fk:<15} {attr:>10.1f}% {red:>10.1f}% {diff:>+9.1f}%")

    # Compute correlation
    if len(attr_vals) >= 3:
        pearson_r, pearson_p = pearsonr(attr_vals, red_vals)
        spearman_r, spearman_p = spearmanr(attr_vals, red_vals)
    else:
        pearson_r, pearson_p = 0, 1
        spearman_r, spearman_p = 0, 1

    print("\n" + "=" * 60)
    print("CALIBRATION METRICS")
    print("=" * 60)
    print(f"Pearson correlation:  r = {pearson_r:.3f} (p = {pearson_p:.3f})")
    print(f"Spearman correlation: r = {spearman_r:.3f} (p = {spearman_p:.3f})")

    # Determine success
    if pearson_r > 0.7:
        verdict = "PASS - Well calibrated!"
    elif pearson_r > 0.5:
        verdict = "MARGINAL - Partially calibrated"
    else:
        verdict = "FAIL - Not calibrated"

    print(f"\nVerdict: {verdict}")

    # Save results
    output = {
        'task': task_name,
        'sample_size': sample_size,
        'baseline_uncertainty': baseline,
        'attribution': attribution,
        'reduction': reduction,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'verdict': verdict
    }

    output_path = Path(__file__).parent / 'results' / 'calibration_validation.json'
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
    args = parser.parse_args()

    run_calibration_experiment(
        task_name=args.task,
        sample_size=args.sample_size,
        n_models=args.n_models
    )
