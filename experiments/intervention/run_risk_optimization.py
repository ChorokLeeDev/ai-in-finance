"""
Risk Optimization Experiment
============================

Question: For high-uncertainty samples, what feature values would MINIMIZE uncertainty?

This is counterfactual optimization:
1. Identify high-uncertainty samples
2. For each FK group, find "optimal" values that minimize uncertainty
3. Measure: How much can uncertainty be reduced? Which FK has most room?

Key insight: This tells us WHERE intervention has the most POTENTIAL,
not just which FK "causes" uncertainty.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from scipy.optimize import minimize_scalar
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution')
from data_loader_f1 import load_f1_data


def train_ensemble(X, y, n_models=5, base_seed=42):
    """Train LightGBM ensemble."""
    models = []
    for i in range(n_models):
        model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=base_seed + i, verbose=-1, force_col_wise=True
        )
        model.fit(X, y)
        models.append(model)
    return models


def get_sample_uncertainty(models, X):
    """Per-sample uncertainty."""
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0)


def get_mean_uncertainty(models, X):
    """Mean uncertainty."""
    return get_sample_uncertainty(models, X).mean()


def find_optimal_fk_values(models, X_high_unc, X_all, col_to_fk, fk_name):
    """
    For high-uncertainty samples, find FK values that minimize uncertainty.

    Strategy: Try values from low-uncertainty samples and see which reduces uncertainty most.
    """
    # Get columns for this FK
    fk_cols = [c for c, fk in col_to_fk.items() if fk == fk_name and c in X_all.columns]
    if not fk_cols:
        return None, 0

    # Get low-uncertainty samples as candidates
    all_unc = get_sample_uncertainty(models, X_all)
    low_unc_mask = all_unc <= np.percentile(all_unc, 20)
    X_low_unc = X_all[low_unc_mask]

    # Current uncertainty of high-unc samples
    base_unc = get_mean_uncertainty(models, X_high_unc)

    # Try each low-uncertainty sample's FK values
    best_reduction = 0
    best_values = None

    # Sample candidates to speed up
    n_candidates = min(50, len(X_low_unc))
    candidate_indices = np.random.choice(len(X_low_unc), n_candidates, replace=False)

    for idx in candidate_indices:
        candidate = X_low_unc.iloc[idx]

        # Replace FK columns with candidate values
        X_modified = X_high_unc.copy()
        for col in fk_cols:
            X_modified[col] = candidate[col]

        new_unc = get_mean_uncertainty(models, X_modified)
        reduction = base_unc - new_unc

        if reduction > best_reduction:
            best_reduction = reduction
            best_values = {col: candidate[col] for col in fk_cols}

    return best_values, best_reduction


def run_optimization_experiment():
    """Run risk optimization experiment."""
    print("=" * 70)
    print("Risk Optimization Experiment")
    print("=" * 70)
    print("\nQuestion: What feature values would MINIMIZE uncertainty?")
    print("Method: For high-uncertainty samples, search for optimal FK values\n")

    # Load data
    print("[1/4] Loading data...")
    X, y, feature_cols, col_to_fk = load_f1_data(sample_size=3000, use_cache=True)
    print(f"Data shape: {X.shape}")

    # Train ensemble
    print("[2/4] Training ensemble...")
    models = train_ensemble(X, y, n_models=5, base_seed=42)

    # Identify high-uncertainty samples (top 20%)
    print("[3/4] Identifying high-uncertainty samples...")
    sample_unc = get_sample_uncertainty(models, X)
    high_unc_threshold = np.percentile(sample_unc, 80)
    high_unc_mask = sample_unc >= high_unc_threshold
    X_high_unc = X[high_unc_mask].copy()

    print(f"  High-uncertainty samples: {high_unc_mask.sum()}")
    print(f"  Their mean uncertainty: {sample_unc[high_unc_mask].mean():.6f}")
    print(f"  Overall mean uncertainty: {sample_unc.mean():.6f}")

    # For each FK, find optimal values
    print("\n[4/4] Finding optimal FK values...")
    fk_names = list(set(col_to_fk.values()))

    results = {}
    base_high_unc = get_mean_uncertainty(models, X_high_unc)

    for fk in fk_names:
        print(f"\n  Optimizing {fk}...")
        best_values, reduction = find_optimal_fk_values(
            models, X_high_unc, X, col_to_fk, fk
        )

        if best_values:
            reduction_pct = (reduction / base_high_unc) * 100
            results[fk] = {
                'optimal_values': best_values,
                'uncertainty_reduction': reduction,
                'reduction_pct': reduction_pct
            }
            print(f"    Max reduction: {reduction_pct:.1f}%")
            print(f"    Optimal values: {best_values}")
        else:
            results[fk] = {'reduction_pct': 0}

    # Summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"\nHigh-uncertainty samples: {high_unc_mask.sum()}")
    print(f"Base uncertainty: {base_high_unc:.6f}")

    print(f"\n{'FK Group':<15} {'Max Reduction (%)':<20} {'Optimization Potential'}")
    print("-" * 55)

    sorted_fks = sorted(results.keys(), key=lambda x: results[x]['reduction_pct'], reverse=True)
    for fk in sorted_fks:
        r = results[fk]['reduction_pct']
        bar = "█" * int(r / 5) if r > 0 else "-"
        print(f"{fk:<15} {r:<20.1f} {bar}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if sorted_fks:
        best_fk = sorted_fks[0]
        best_reduction = results[best_fk]['reduction_pct']

        if best_reduction > 10:
            print(f"\n✓ ACTIONABLE: Changing {best_fk} values can reduce uncertainty by {best_reduction:.1f}%")
            print(f"  Optimal {best_fk} values found in low-uncertainty samples")
            print(f"\n  RECOMMENDATION: Investigate why some samples have these 'good' values")
            print(f"  → Is it data quality? Missing values? Specific entities?")
        else:
            print(f"\n⚠ LIMITED POTENTIAL: Maximum reduction is only {best_reduction:.1f}%")
            print("  → Uncertainty may be inherent (aleatoric) rather than fixable")

    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_optimization_experiment()
