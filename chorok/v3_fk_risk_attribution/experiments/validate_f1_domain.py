"""
F1 Domain Validation: FK Attribution Test
==========================================

Test FK Attribution on F1 Racing domain (4 FK groups).
Expected: EP domain (hierarchical FK structure) → should work

Author: ChorokLeeDev
Created: 2025-12-08
"""

import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loader_f1 import load_f1_data

CACHE_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache')
RESULTS_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/results')


def train_ensemble(X, y, n_models=5, base_seed=42):
    """Train ensemble for uncertainty estimation."""
    models = []
    for i in range(n_models):
        model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=base_seed + i, verbose=-1
        )
        model.fit(X, y)
        models.append(model)
    return models


def get_uncertainty(models, X):
    """Compute ensemble uncertainty."""
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0)


def get_prediction(models, X):
    """Get ensemble mean prediction."""
    preds = np.array([m.predict(X) for m in models])
    return preds.mean(axis=0)


def get_fk_grouping(col_to_fk):
    """Convert column->FK mapping to FK->columns mapping."""
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)
    return dict(fk_to_cols)


def compute_fk_attribution(models, X, fk_grouping, n_permute=10):
    """Compute FK attribution using permutation."""
    base_unc = get_uncertainty(models, X).mean()
    results = {}

    for fk_group, cols in fk_grouping.items():
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            continue

        deltas = []
        for _ in range(n_permute):
            X_perm = X.copy()
            for col in valid_cols:
                X_perm[col] = np.random.permutation(X_perm[col].values)
            perm_unc = get_uncertainty(models, X_perm).mean()
            deltas.append(perm_unc - base_unc)

        results[fk_group] = np.mean(deltas)

    # Normalize to percentages
    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def compute_error_impact(models, X, y, fk_grouping, n_permute=10):
    """Compute error impact using permutation."""
    y_arr = np.array(y).copy()  # Ensure writable copy
    base_pred = get_prediction(models, X)
    base_mae = mean_absolute_error(y_arr, base_pred)

    results = {}

    for fk_group, cols in fk_grouping.items():
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            continue

        deltas = []
        for _ in range(n_permute):
            X_perm = X.copy()
            for col in valid_cols:
                X_perm[col] = np.random.permutation(X_perm[col].values)
            perm_pred = get_prediction(models, X_perm)
            perm_mae = mean_absolute_error(y_arr, perm_pred)
            deltas.append(perm_mae - base_mae)

        results[fk_group] = np.mean(deltas)

    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def run_validation(n_runs=5):
    """Run FK Attribution validation on F1 domain."""
    print("=" * 70)
    print("F1 DOMAIN VALIDATION")
    print("Testing FK Attribution on Formula 1 Racing")
    print("=" * 70)

    # Load data
    print("\n[1] Loading F1 data...")
    X, y, feature_cols, col_to_fk = load_f1_data(sample_size=3000)
    fk_grouping = get_fk_grouping(col_to_fk)
    fk_list = list(fk_grouping.keys())

    print(f"\n  Features: {len(feature_cols)}")
    print(f"  FK groups: {len(fk_list)} - {fk_list}")

    # Run experiments
    print(f"\n[2] Running {n_runs} experiments...")

    all_correlations = []
    all_attributions = []
    all_errors = []

    for run in range(n_runs):
        seed = 42 + run * 10
        np.random.seed(seed)
        print(f"\n  Run {run+1}/{n_runs} (seed={seed})...")

        # Train ensemble
        models = train_ensemble(X, y, n_models=5, base_seed=seed)

        # Compute attribution and error impact
        attribution = compute_fk_attribution(models, X, fk_grouping, n_permute=10)
        error_impact = compute_error_impact(models, X, y, fk_grouping, n_permute=10)

        # Compute correlation
        attr_vals = [attribution.get(fk, 0) for fk in fk_list]
        err_vals = [error_impact.get(fk, 0) for fk in fk_list]

        if len(fk_list) >= 3:
            corr, p_val = spearmanr(attr_vals, err_vals)
        else:
            corr, p_val = float('nan'), float('nan')

        all_correlations.append(corr)
        all_attributions.append(attribution)
        all_errors.append(error_impact)

        print(f"    Attribution: {[f'{k}:{v:.1f}%' for k,v in attribution.items()]}")
        print(f"    Error Impact: {[f'{k}:{v:.1f}%' for k,v in error_impact.items()]}")
        print(f"    Correlation: ρ = {corr:.3f}")

    # Summary
    print(f"\n{'='*70}")
    print("F1 VALIDATION SUMMARY")
    print("="*70)

    mean_corr = np.nanmean(all_correlations)
    std_corr = np.nanstd(all_correlations)

    print(f"\n  Mean Correlation: ρ = {mean_corr:.3f} ± {std_corr:.3f}")

    # Average attribution per FK
    print(f"\n  Average Attribution per FK group:")
    for fk in fk_list:
        mean_attr = np.mean([a.get(fk, 0) for a in all_attributions])
        mean_err = np.mean([e.get(fk, 0) for e in all_errors])
        print(f"    {fk:15s}: Attr={mean_attr:5.1f}%, Error={mean_err:5.1f}%")

    # Verdict
    if mean_corr >= 0.7:
        verdict = f"✓ STRONG: F1 is EP domain (ρ={mean_corr:.2f})"
    elif mean_corr >= 0.4:
        verdict = f"~ MODERATE: F1 shows EP behavior (ρ={mean_corr:.2f})"
    elif mean_corr >= 0:
        verdict = f"? WEAK: F1 shows weak EP behavior (ρ={mean_corr:.2f})"
    else:
        verdict = f"✗ NOT EP: F1 is not an Error Propagation domain (ρ={mean_corr:.2f})"

    print(f"\n  Verdict: {verdict}")

    # Save results
    import json

    results = {
        'domain': 'F1 Racing',
        'n_fk_groups': len(fk_list),
        'fk_groups': fk_list,
        'n_runs': n_runs,
        'mean_correlation': float(mean_corr),
        'std_correlation': float(std_corr),
        'correlations': [float(c) if not np.isnan(c) else None for c in all_correlations],
        'verdict': verdict
    }

    output_path = RESULTS_DIR / 'f1_validation.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {output_path}")

    return results


if __name__ == "__main__":
    results = run_validation(n_runs=5)
