"""
Experiment: Calibration Validation (rel-f1 Regression)
=======================================================

Validate that FK attribution predicts actual uncertainty sensitivity.

Key Idea:
- If FK contributes X% to uncertainty via noise injection, then modifying
  that FK should cause proportionally large uncertainty changes.

Method:
1. Compute FK attribution via noise injection (add noise → uncertainty increases)
2. "Fix" each FK by replacing with column mean
3. Measure absolute uncertainty change |delta|
   - Note: For regression, fixing to mean may INCREASE uncertainty
     because it creates out-of-distribution samples
4. Compare: Does high noise-injection attribution correlate with high |delta|?

Interpretation:
- Noise injection: "How much does uncertainty increase when we corrupt this FK?"
- Mean fixing: "How sensitive is uncertainty to this FK's values?"
- Both measure FK importance; ranks should correlate

Success Criterion:
- Spearman correlation between predicted and actual > 0.6
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy.stats import spearmanr

from data_loader_f1 import load_f1_data, get_fk_groups
from ensemble_f1 import train_regression_ensemble, compute_regression_uncertainty


def compute_noise_injection_attribution(
    models: List,
    X: pd.DataFrame,
    col_to_fk: Dict[str, str],
    n_perturbations: int = 10
) -> Tuple[Dict[str, float], float]:
    """
    Compute FK attribution via noise injection.

    Returns:
        contributions: FK -> contribution percentage
        baseline_uncertainty: Original uncertainty level
    """
    X_np = X.values
    baseline_uncertainty = compute_regression_uncertainty(models, X_np).mean()

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
            fk_deltas.append(noisy_unc - baseline_uncertainty)

        deltas[fk] = np.mean(fk_deltas)

    # Normalize to percentages
    total_positive = sum(max(0, v) for v in deltas.values())
    if total_positive > 0:
        contributions = {fk: max(0, v) / total_positive * 100 for fk, v in deltas.items()}
    else:
        contributions = {fk: 0 for fk in deltas}

    return contributions, baseline_uncertainty


def compute_actual_reduction(
    models: List,
    X: pd.DataFrame,
    col_to_fk: Dict[str, str],
    baseline_uncertainty: float
) -> Dict[str, float]:
    """
    Compute actual uncertainty reduction by "fixing" each FK.

    "Fixing" = replace column values with their mean (for numeric) or mode (for categorical).
    This simulates what happens when we have perfect information about that FK.

    Returns:
        reductions: FK -> reduction in uncertainty (positive = reduced)
    """
    X_np = X.values
    feature_cols = list(X.columns)

    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)

    reductions = {}
    for fk, cols in fk_to_cols.items():
        col_indices = [feature_cols.index(c) for c in cols if c in feature_cols]
        if not col_indices:
            continue

        # "Fix" FK by replacing with column means
        X_fixed = X_np.copy()
        for idx in col_indices:
            col_mean = np.mean(X_fixed[:, idx])
            X_fixed[:, idx] = col_mean

        fixed_uncertainty = compute_regression_uncertainty(models, X_fixed).mean()

        # Reduction = how much uncertainty decreased
        reduction = baseline_uncertainty - fixed_uncertainty
        reductions[fk] = reduction

    return reductions


def run_calibration_experiment(
    task_name: str = 'driver-position',
    sample_size: int = 3000,
    n_models: int = 5,
    n_perturbations: int = 10
) -> Dict:
    """Run calibration validation experiment."""

    print("=" * 70)
    print("FK-Level Risk Attribution: Calibration Validation")
    print("=" * 70)
    print(f"Dataset: rel-f1, Task: {task_name}, Sample size: {sample_size}")
    print(f"Method: Compare predicted attribution vs actual uncertainty reduction")

    # Load data
    X, y, feature_cols, col_to_fk = load_f1_data(task_name, sample_size, use_cache=True)
    print(f"Loaded {len(X)} samples, {len(feature_cols)} features")

    # Train ensemble
    print(f"\n[1] Training {n_models}-model ensemble...")
    models, X_test, y_test = train_regression_ensemble(X, y, n_models=n_models)

    # Compute attribution (predicted contribution)
    print(f"\n[2] Computing FK attribution via noise injection...")
    predicted, baseline = compute_noise_injection_attribution(
        models, X, col_to_fk, n_perturbations=n_perturbations
    )
    print(f"Baseline uncertainty: {baseline:.4f}")

    print("\nPredicted contributions (noise injection):")
    for fk, pct in sorted(predicted.items(), key=lambda x: -x[1]):
        print(f"  {fk}: {pct:.1f}%")

    # Compute actual reduction
    print(f"\n[3] Computing actual uncertainty reduction (fix each FK)...")
    actual_raw = compute_actual_reduction(models, X, col_to_fk, baseline)

    # Normalize actual changes to percentages
    # NOTE: For regression, "fixing" may INCREASE uncertainty (creates OOD samples)
    # We use |delta| as a measure of FK importance (how much uncertainty changes)
    total_change = sum(abs(v) for v in actual_raw.values())
    if total_change > 0:
        actual = {fk: abs(v) / total_change * 100 for fk, v in actual_raw.items()}
    else:
        actual = {fk: 0 for fk in actual_raw}

    print("\nActual uncertainty change (|delta|):")
    for fk, pct in sorted(actual.items(), key=lambda x: -x[1]):
        raw = actual_raw[fk]
        direction = "↑" if raw < 0 else "↓"  # Negative = increased uncertainty
        print(f"  {fk}: {pct:.1f}% (raw: {raw:+.4f} {direction})")

    # Compare predicted vs actual
    print("\n" + "=" * 70)
    print("CALIBRATION ANALYSIS")
    print("=" * 70)

    fks = list(predicted.keys())
    pred_values = [predicted[fk] for fk in fks]
    actual_values = [actual[fk] for fk in fks]

    # Spearman correlation (rank-based)
    corr, pval = spearmanr(pred_values, actual_values)
    print(f"\nSpearman correlation: {corr:.3f} (p={pval:.4f})")

    # Comparison table
    print(f"\n{'FK':<15} {'Predicted':<12} {'Actual':<12} {'Diff':<12}")
    print("-" * 50)
    for fk in sorted(fks):
        pred = predicted[fk]
        act = actual[fk]
        diff = pred - act
        print(f"{fk:<15} {pred:>10.1f}% {act:>10.1f}% {diff:>+10.1f}%")

    # Check rank agreement
    pred_rank = sorted(fks, key=lambda f: -predicted[f])
    actual_rank = sorted(fks, key=lambda f: -actual[f])

    print(f"\nRank comparison:")
    print(f"  Predicted top: {pred_rank[:3]}")
    print(f"  Actual top:    {actual_rank[:3]}")

    top1_match = pred_rank[0] == actual_rank[0]
    top3_match = set(pred_rank[:3]) == set(actual_rank[:3])

    print(f"\n  Top-1 match: {'YES' if top1_match else 'NO'}")
    print(f"  Top-3 match: {'YES' if top3_match else 'NO'}")

    # Verdict
    if corr > 0.8:
        verdict = "EXCELLENT - High correlation"
    elif corr > 0.6:
        verdict = "GOOD - Moderate correlation"
    elif corr > 0.4:
        verdict = "FAIR - Weak correlation"
    else:
        verdict = "POOR - Low correlation"

    print(f"\nVerdict: {verdict}")

    # Save results
    output = {
        'dataset': 'rel-f1',
        'task': task_name,
        'sample_size': sample_size,
        'n_models': n_models,
        'baseline_uncertainty': baseline,
        'predicted_contributions': predicted,
        'actual_reductions': actual,
        'actual_raw': actual_raw,
        'spearman_correlation': corr,
        'spearman_pvalue': pval,
        'top1_match': top1_match,
        'top3_match': top3_match,
        'verdict': verdict
    }

    output_path = Path(__file__).parent / 'results' / 'calibration_f1.json'
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
    parser.add_argument('--n_perturbations', type=int, default=10)
    args = parser.parse_args()

    run_calibration_experiment(
        task_name=args.task,
        sample_size=args.sample_size,
        n_models=args.n_models,
        n_perturbations=args.n_perturbations
    )
