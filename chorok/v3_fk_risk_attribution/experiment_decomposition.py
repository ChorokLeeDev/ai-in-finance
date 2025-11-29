"""
Experiment: Decomposition Validation
=====================================

Validate FK-level risk attribution using three methods:
A. Noise Injection
B. Leave-One-Out
C. Counterfactual

Goal: Show that FK-level decomposition correctly identifies uncertainty sources.
"""

import argparse
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from cache import save_cache, load_cache, cache_result, load_result
from data_loader import load_salt_data, get_fk_group, print_fk_summary
from ensemble import train_ensemble, compute_uncertainty


def get_fk_columns(col_to_fk: Dict[str, str]) -> Dict[str, List[str]]:
    """Get FK groups with their columns."""
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)
    return dict(fk_to_cols)


# =============================================================================
# Method A: Noise Injection
# =============================================================================

def experiment_noise_injection(
    models: List,
    X: pd.DataFrame,
    col_to_fk: Dict[str, str],
    n_perturbations: int = 10
) -> Dict[str, float]:
    """
    Inject noise into each FK's columns and measure uncertainty increase.

    If FK contributes to uncertainty, adding noise to it should INCREASE uncertainty.
    """
    print("\n[1A] Noise Injection Experiment")
    print("-" * 40)

    X_np = X.values
    baseline_uncertainty = compute_uncertainty(models, X_np, method='entropy').mean()
    print(f"Baseline uncertainty: {baseline_uncertainty:.4f}")

    fk_to_cols = get_fk_columns(col_to_fk)
    feature_cols = list(X.columns)

    results = {}
    for fk, cols in fk_to_cols.items():
        col_indices = [feature_cols.index(c) for c in cols if c in feature_cols]
        if not col_indices:
            continue

        # Inject noise multiple times and average
        deltas = []
        for _ in range(n_perturbations):
            X_noisy = X_np.copy()
            for idx in col_indices:
                # Permute column values (distribution-preserving noise)
                X_noisy[:, idx] = np.random.permutation(X_noisy[:, idx])

            noisy_uncertainty = compute_uncertainty(models, X_noisy, method='entropy').mean()
            deltas.append(noisy_uncertainty - baseline_uncertainty)

        avg_delta = np.mean(deltas)
        results[fk] = avg_delta
        print(f"  {fk}: delta = {avg_delta:+.4f}")

    # Normalize to percentages
    total_positive = sum(max(0, v) for v in results.values())
    if total_positive > 0:
        contributions = {fk: max(0, v) / total_positive * 100 for fk, v in results.items()}
    else:
        contributions = {fk: 0 for fk in results}

    print(f"\nContributions (%):")
    for fk, pct in sorted(contributions.items(), key=lambda x: -x[1]):
        print(f"  {fk}: {pct:.1f}%")

    return {'raw_deltas': results, 'contributions': contributions}


# =============================================================================
# Method B: Leave-One-Out
# =============================================================================

def experiment_loo(
    X: pd.DataFrame,
    y: pd.Series,
    col_to_fk: Dict[str, str],
    n_models: int = 5,
    task_name: str = 'sales-group',
    sample_size: int = 500
) -> Dict[str, float]:
    """
    Remove each FK's columns and measure uncertainty change.

    If FK contributes to uncertainty, removing it should CHANGE uncertainty.
    """
    print("\n[1B] Leave-One-Out Experiment")
    print("-" * 40)

    X_np = X.values
    feature_cols = list(X.columns)

    # Train full model
    print("Training full model...")
    full_models, X_test, y_test = train_ensemble(X, y, n_models=n_models)
    baseline_uncertainty = compute_uncertainty(full_models, X_test, method='entropy').mean()
    print(f"Baseline uncertainty: {baseline_uncertainty:.4f}")

    fk_to_cols = get_fk_columns(col_to_fk)

    results = {}
    for fk, cols in fk_to_cols.items():
        remaining_cols = [c for c in feature_cols if c not in cols]
        if not remaining_cols:
            continue

        # Check cache for LOO model
        cache_key = f"loo_{task_name}_{sample_size}_{fk}"
        cached = load_cache(cache_key)

        X_loo = X[remaining_cols]

        if cached is not None:
            loo_models = cached['models']
            print(f"  [CACHE] Loaded LOO model without {fk}")
        else:
            print(f"  Training without {fk}...")
            loo_models, X_test_loo, _ = train_ensemble(X_loo, y, n_models=n_models)
            save_cache(cache_key, {'models': loo_models})

        # Get test samples for same indices
        X_test_loo_filtered = X_loo.iloc[-len(X_test):].values
        loo_uncertainty = compute_uncertainty(loo_models, X_test_loo_filtered, method='entropy').mean()

        delta = loo_uncertainty - baseline_uncertainty
        results[fk] = delta
        print(f"    {fk}: uncertainty without = {loo_uncertainty:.4f}, delta = {delta:+.4f}")

    # Contribution = how much uncertainty increases when FK is removed
    total_positive = sum(max(0, v) for v in results.values())
    if total_positive > 0:
        contributions = {fk: max(0, v) / total_positive * 100 for fk, v in results.items()}
    else:
        contributions = {fk: 0 for fk in results}

    print(f"\nContributions (%):")
    for fk, pct in sorted(contributions.items(), key=lambda x: -x[1]):
        print(f"  {fk}: {pct:.1f}%")

    return {'raw_deltas': results, 'contributions': contributions}


# =============================================================================
# Method C: Counterfactual
# =============================================================================

def experiment_counterfactual(
    models: List,
    X: pd.DataFrame,
    col_to_fk: Dict[str, str]
) -> Dict[str, float]:
    """
    Replace each FK's columns with mean values and measure uncertainty decrease.

    If FK contributes to uncertainty, replacing with "safe" values should DECREASE uncertainty.
    """
    print("\n[1C] Counterfactual Experiment")
    print("-" * 40)

    X_np = X.values
    feature_cols = list(X.columns)

    baseline_uncertainty = compute_uncertainty(models, X_np, method='entropy').mean()
    print(f"Baseline uncertainty: {baseline_uncertainty:.4f}")

    fk_to_cols = get_fk_columns(col_to_fk)

    # Compute mean values for each column
    col_means = X.mean().values

    results = {}
    for fk, cols in fk_to_cols.items():
        col_indices = [feature_cols.index(c) for c in cols if c in feature_cols]
        if not col_indices:
            continue

        # Replace FK columns with mean values
        X_cf = X_np.copy()
        for idx in col_indices:
            X_cf[:, idx] = col_means[idx]

        cf_uncertainty = compute_uncertainty(models, X_cf, method='entropy').mean()
        delta = baseline_uncertainty - cf_uncertainty  # Positive = reduced uncertainty
        results[fk] = delta
        print(f"  {fk}: delta = {delta:+.4f} (uncertainty {'decreased' if delta > 0 else 'increased'})")

    # Contribution = how much uncertainty decreases when FK is "fixed"
    total_positive = sum(max(0, v) for v in results.values())
    if total_positive > 0:
        contributions = {fk: max(0, v) / total_positive * 100 for fk, v in results.items()}
    else:
        contributions = {fk: 0 for fk in results}

    print(f"\nContributions (%):")
    for fk, pct in sorted(contributions.items(), key=lambda x: -x[1]):
        print(f"  {fk}: {pct:.1f}%")

    return {'raw_deltas': results, 'contributions': contributions}


# =============================================================================
# Main
# =============================================================================

def run_all_experiments(
    task_name: str = 'sales-group',
    sample_size: int = 500,
    n_models: int = 5
) -> Dict:
    """Run all three decomposition validation experiments."""

    print("="*60)
    print("FK-Level Risk Attribution: Decomposition Validation")
    print("="*60)
    print(f"Task: {task_name}, Sample size: {sample_size}")

    # Load data
    X, y, feature_cols, col_to_fk = load_salt_data(task_name, sample_size)
    print_fk_summary(col_to_fk)

    # Train ensemble
    print("\n[0] Training ensemble...")
    models, X_test, y_test = train_ensemble(X, y, n_models=n_models)
    print(f"Trained {len(models)} models, test set: {X_test.shape}")

    # Run experiments
    results = {}

    # 1A: Noise Injection
    results['noise_injection'] = experiment_noise_injection(models, X, col_to_fk)

    # 1B: LOO
    results['loo'] = experiment_loo(X, y, col_to_fk, n_models=n_models, task_name=task_name, sample_size=sample_size)

    # 1C: Counterfactual
    results['counterfactual'] = experiment_counterfactual(models, X, col_to_fk)

    # Compare results
    print("\n" + "="*60)
    print("COMPARISON: FK Contributions (%)")
    print("="*60)
    print(f"{'FK':<15} {'Noise':<10} {'LOO':<10} {'Counterfact':<12}")
    print("-"*50)

    all_fks = set()
    for method_result in results.values():
        all_fks.update(method_result['contributions'].keys())

    for fk in sorted(all_fks):
        noise = results['noise_injection']['contributions'].get(fk, 0)
        loo = results['loo']['contributions'].get(fk, 0)
        cf = results['counterfactual']['contributions'].get(fk, 0)
        print(f"{fk:<15} {noise:>8.1f}% {loo:>8.1f}% {cf:>10.1f}%")

    # Save results
    output = {
        'task': task_name,
        'sample_size': sample_size,
        'n_models': n_models,
        'results': {
            'noise_injection': results['noise_injection']['contributions'],
            'loo': results['loo']['contributions'],
            'counterfactual': results['counterfactual']['contributions']
        }
    }

    output_path = Path(__file__).parent / 'results' / 'decomposition_validation.json'
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

    run_all_experiments(
        task_name=args.task,
        sample_size=args.sample_size,
        n_models=args.n_models
    )
