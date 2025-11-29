"""
Intervention Simulation Experiment
==================================

Validates that FK attribution predicts actual uncertainty reduction.

Key Question: If we "improve" an FK group's data quality, does uncertainty
decrease proportionally to the predicted attribution?

Method:
1. Train ensemble, compute FK attribution (predicted importance)
2. Define "reference set": samples with low uncertainty (bottom 20%)
3. For each FK group:
   - Replace features with reference values (simulated improvement)
   - Measure actual uncertainty change
4. Compare predicted vs. actual rankings (Spearman correlation)
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution')
from data_loader_f1 import load_f1_data


def train_ensemble(X, y, n_models=5, base_seed=42):
    """Train LightGBM ensemble with subsampling for diversity."""
    models = []
    for i in range(n_models):
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=base_seed + i,
            verbose=-1,
            force_col_wise=True
        )
        model.fit(X, y)
        models.append(model)
    return models


def get_uncertainty(models, X):
    """Compute ensemble variance (epistemic uncertainty) per sample."""
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0)


def get_mean_uncertainty(models, X):
    """Compute mean uncertainty across samples."""
    return get_uncertainty(models, X).mean()


def fk_attribution(models, X, col_to_fk, n_perm=5):
    """
    Compute FK-level attribution via permutation.
    Returns: dict of {fk_name: attribution_percentage}
    """
    base_unc = get_mean_uncertainty(models, X)

    # Group columns by FK
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        if col in X.columns:
            fk_to_cols[fk].append(col)

    # Compute attribution for each FK
    fk_deltas = {}
    for fk, cols in fk_to_cols.items():
        deltas = []
        for _ in range(n_perm):
            X_perm = X.copy()
            for col in cols:
                X_perm[col] = np.random.permutation(X_perm[col].values)
            perm_unc = get_mean_uncertainty(models, X_perm)
            deltas.append(perm_unc - base_unc)
        fk_deltas[fk] = np.mean(deltas)

    # Normalize to percentages
    total = sum(max(0, d) for d in fk_deltas.values())
    if total > 0:
        return {fk: max(0, d) / total * 100 for fk, d in fk_deltas.items()}
    return {fk: 0.0 for fk in fk_deltas}


def intervention_simulation(models, X, col_to_fk, reference_percentile=20):
    """
    Simulate intervention by replacing FK features with reference values.

    Reference set: samples with uncertainty in the bottom `reference_percentile`%
    Intervention: replace FK features with mean of reference set

    Returns: dict of {fk_name: actual_uncertainty_reduction_percentage}
    """
    # Compute per-sample uncertainty
    sample_unc = get_uncertainty(models, X)
    base_mean_unc = sample_unc.mean()

    # Define reference set (low uncertainty samples)
    threshold = np.percentile(sample_unc, reference_percentile)
    reference_mask = sample_unc <= threshold
    X_reference = X[reference_mask]

    print(f"  Reference set: {reference_mask.sum()} samples (bottom {reference_percentile}% uncertainty)")
    print(f"  Base uncertainty: {base_mean_unc:.6f}")

    # Group columns by FK
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        if col in X.columns:
            fk_to_cols[fk].append(col)

    # Simulate intervention for each FK
    fk_reductions = {}
    for fk, cols in fk_to_cols.items():
        # Create intervened dataset
        X_intervened = X.copy()
        for col in cols:
            # Replace with mean of reference values
            ref_value = X_reference[col].mean()
            X_intervened[col] = ref_value

        # Measure new uncertainty
        new_unc = get_mean_uncertainty(models, X_intervened)
        reduction = base_mean_unc - new_unc
        fk_reductions[fk] = reduction

    # Convert to percentage of base uncertainty
    total_reduction = sum(max(0, r) for r in fk_reductions.values())
    if total_reduction > 0:
        return {fk: max(0, r) / base_mean_unc * 100 for fk, r in fk_reductions.items()}
    return {fk: 0.0 for fk in fk_reductions}


def run_experiment(n_seeds=3):
    """Run full intervention simulation experiment."""
    print("=" * 70)
    print("Intervention Simulation Experiment")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    X, y, feature_cols, col_to_fk = load_f1_data(sample_size=3000, use_cache=True)
    print(f"Data shape: {X.shape}")
    print(f"FK groups: {set(col_to_fk.values())}")

    all_predicted = []
    all_actual = []

    for seed in range(n_seeds):
        print(f"\n[2/4] Seed {seed+1}/{n_seeds}: Training ensemble...")
        models = train_ensemble(X, y, n_models=5, base_seed=100*seed)

        print(f"[3/4] Computing FK attribution (predicted)...")
        predicted = fk_attribution(models, X, col_to_fk, n_perm=5)

        print(f"[4/4] Running intervention simulation (actual)...")
        actual = intervention_simulation(models, X, col_to_fk, reference_percentile=20)

        all_predicted.append(predicted)
        all_actual.append(actual)

        print(f"\n  Results for seed {seed+1}:")
        print(f"  {'FK Group':<15} {'Predicted (%)':<15} {'Actual (%)':<15}")
        print("  " + "-" * 45)
        for fk in predicted.keys():
            print(f"  {fk:<15} {predicted[fk]:<15.1f} {actual[fk]:<15.1f}")

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)

    fk_names = list(all_predicted[0].keys())

    avg_predicted = {fk: np.mean([p[fk] for p in all_predicted]) for fk in fk_names}
    avg_actual = {fk: np.mean([a[fk] for a in all_actual]) for fk in fk_names}

    print(f"\n{'FK Group':<15} {'Predicted (%)':<15} {'Actual (%)':<15} {'Rank Match':<12}")
    print("-" * 55)

    # Sort by predicted for display
    sorted_fks = sorted(fk_names, key=lambda x: avg_predicted[x], reverse=True)
    pred_ranks = {fk: i+1 for i, fk in enumerate(sorted_fks)}
    sorted_by_actual = sorted(fk_names, key=lambda x: avg_actual[x], reverse=True)
    actual_ranks = {fk: i+1 for i, fk in enumerate(sorted_by_actual)}

    for fk in sorted_fks:
        match = "✓" if pred_ranks[fk] == actual_ranks[fk] else ""
        print(f"{fk:<15} {avg_predicted[fk]:<15.1f} {avg_actual[fk]:<15.1f} {match:<12}")

    # Compute calibration (Spearman correlation)
    pred_values = [avg_predicted[fk] for fk in fk_names]
    actual_values = [avg_actual[fk] for fk in fk_names]

    if len(set(pred_values)) > 1 and len(set(actual_values)) > 1:
        rho, pval = stats.spearmanr(pred_values, actual_values)
    else:
        rho, pval = np.nan, np.nan

    print(f"\nCalibration (Spearman ρ): {rho:.3f} (p={pval:.4f})")

    # Also compute per-seed correlations
    seed_correlations = []
    for i in range(n_seeds):
        pred_vals = [all_predicted[i][fk] for fk in fk_names]
        actual_vals = [all_actual[i][fk] for fk in fk_names]
        if len(set(pred_vals)) > 1 and len(set(actual_vals)) > 1:
            r, _ = stats.spearmanr(pred_vals, actual_vals)
            seed_correlations.append(r)

    if seed_correlations:
        print(f"Per-seed correlations: {seed_correlations}")
        print(f"Mean ± std: {np.mean(seed_correlations):.3f} ± {np.std(seed_correlations):.3f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    if rho > 0.7:
        print("✓ HIGH CALIBRATION: FK attribution accurately predicts intervention impact")
        print("  → Practitioners can trust attribution for prioritizing data improvements")
    elif rho > 0.4:
        print("~ MODERATE CALIBRATION: FK attribution is somewhat predictive")
        print("  → Useful for rough prioritization but not precise predictions")
    else:
        print("✗ LOW CALIBRATION: FK attribution does not predict intervention impact well")
        print("  → Need to investigate why predicted ≠ actual")

    print("=" * 70)

    return {
        'predicted': avg_predicted,
        'actual': avg_actual,
        'calibration_rho': rho,
        'calibration_pval': pval,
        'per_seed_rho': seed_correlations
    }


if __name__ == "__main__":
    results = run_experiment(n_seeds=3)
