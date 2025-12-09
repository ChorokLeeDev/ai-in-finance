"""
Feasibility Test: Does FK Attribution Work with BNN?
=====================================================

Key question: Is FK attribution a property of epistemic uncertainty itself,
or just an artifact of a specific UQ method (e.g., LightGBM ensemble)?

If FK attribution gives SAME rankings across:
- LightGBM Ensemble
- MC Dropout MLP
- Bayesian Neural Network

Then it's a fundamental property → Strong NeurIPS contribution

Test Protocol:
1. Load SALT data (has clear FK structure)
2. Train each UQ method
3. Compute FK attribution for each
4. Compare rankings (Spearman correlation)

Author: ChorokLeeDev
Created: 2025-12-09
"""

import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'methods'))

from methods.bnn import train_bnn_ensemble
from methods.ensemble_lgbm import train_lgbm_ensemble

CACHE_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache')
RESULTS_DIR = Path(__file__).parent.parent / 'results'


def load_salt_data():
    """Load SALT data from v3 cache."""
    cache_file = CACHE_DIR / 'data_salt_PLANT_2000.pkl'
    if not cache_file.exists():
        print(f"Cache not found: {cache_file}")
        return None, None, None, None

    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    if len(data) == 4:
        X, y, feature_cols, col_to_fk = data
        return X, y, feature_cols, col_to_fk

    return None, None, None, None


def get_fk_grouping(col_to_fk):
    """Convert column-to-FK mapping to FK-to-columns mapping."""
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)
    return dict(fk_to_cols)


def compute_fk_attribution(get_uncertainty_fn, X, fk_grouping, n_permute=5):
    """
    Compute FK-level uncertainty attribution.

    Args:
        get_uncertainty_fn: Function that takes X and returns uncertainty (scalar)
        X: Feature matrix
        fk_grouping: Dict mapping FK name to list of columns
        n_permute: Number of permutation runs

    Returns:
        Dict mapping FK name to attribution percentage
    """
    base_unc = get_uncertainty_fn(X)
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
            perm_unc = get_uncertainty_fn(X_perm)
            deltas.append(perm_unc - base_unc)

        results[fk_group] = np.mean(deltas)

    # Normalize to percentages
    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def compute_error_impact(predict_fn, X, y, fk_grouping, n_permute=5):
    """Compute FK-level error impact (MAE increase under permutation)."""
    base_pred = predict_fn(X)
    base_mae = np.abs(base_pred - y).mean()

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
            perm_pred = predict_fn(X_perm)
            perm_mae = np.abs(perm_pred - y).mean()
            deltas.append(perm_mae - base_mae)

        results[fk_group] = np.mean(deltas)

    # Normalize to percentages
    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def run_feasibility_test():
    """Main feasibility test."""
    print("="*70)
    print("FEASIBILITY TEST: FK Attribution Across UQ Methods")
    print("="*70)

    # Load data
    print("\n[1/5] Loading SALT data...")
    X, y, feature_cols, col_to_fk = load_salt_data()
    if X is None:
        print("ERROR: Could not load SALT data")
        return None

    # Convert to numpy if needed, keep as DataFrame for column access
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_cols)

    y = np.array(y).flatten()

    print(f"  Data shape: {X.shape}")
    print(f"  Target shape: {y.shape}")

    fk_grouping = get_fk_grouping(col_to_fk)
    print(f"  FK groups: {list(fk_grouping.keys())}")

    # Subsample for faster testing
    n_samples = min(2000, len(X))
    idx = np.random.RandomState(42).permutation(len(X))[:n_samples]
    X_sub = X.iloc[idx].reset_index(drop=True)
    y_sub = y[idx]

    print(f"  Using {n_samples} samples for testing")

    results = {}

    # Method 1: LightGBM Ensemble
    print("\n[2/5] Training LightGBM Ensemble...")
    lgbm_ensemble = train_lgbm_ensemble(X_sub.values, y_sub, n_models=5, seed=42)

    def lgbm_uncertainty(X_input):
        if isinstance(X_input, pd.DataFrame):
            X_input = X_input.values
        return lgbm_ensemble.get_uncertainty(X_input).mean()

    def lgbm_predict(X_input):
        if isinstance(X_input, pd.DataFrame):
            X_input = X_input.values
        return lgbm_ensemble.get_mean_prediction(X_input)

    lgbm_attr = compute_fk_attribution(lgbm_uncertainty, X_sub, fk_grouping)
    lgbm_error = compute_error_impact(lgbm_predict, X_sub, y_sub, fk_grouping)

    print(f"  LightGBM Attribution: {lgbm_attr}")
    rho_lgbm, _ = spearmanr(list(lgbm_attr.values()), list(lgbm_error.values()))
    print(f"  LightGBM Attribution-Error Corr: {rho_lgbm:.3f}")

    results['lgbm'] = {
        'attribution': lgbm_attr,
        'error_impact': lgbm_error,
        'rho': rho_lgbm
    }

    # Method 2: BNN Ensemble
    print("\n[3/5] Training Bayesian Neural Network...")
    bnn_ensemble = train_bnn_ensemble(
        X_sub.values, y_sub,
        n_networks=5, mc_samples=10, epochs=100, seed=42
    )

    def bnn_uncertainty(X_input):
        if isinstance(X_input, pd.DataFrame):
            X_input = X_input.values
        return bnn_ensemble.get_uncertainty(X_input).mean()

    def bnn_predict(X_input):
        if isinstance(X_input, pd.DataFrame):
            X_input = X_input.values
        return bnn_ensemble.get_mean_prediction(X_input)

    bnn_attr = compute_fk_attribution(bnn_uncertainty, X_sub, fk_grouping)
    bnn_error = compute_error_impact(bnn_predict, X_sub, y_sub, fk_grouping)

    print(f"  BNN Attribution: {bnn_attr}")
    rho_bnn, _ = spearmanr(list(bnn_attr.values()), list(bnn_error.values()))
    print(f"  BNN Attribution-Error Corr: {rho_bnn:.3f}")

    results['bnn'] = {
        'attribution': bnn_attr,
        'error_impact': bnn_error,
        'rho': rho_bnn
    }

    # Method 3: Compare rankings between methods
    print("\n[4/5] Comparing Attribution Rankings...")

    # Get FK names in consistent order
    fk_names = list(lgbm_attr.keys())

    lgbm_ranks = [lgbm_attr[fk] for fk in fk_names]
    bnn_ranks = [bnn_attr[fk] for fk in fk_names]

    rho_cross, p_cross = spearmanr(lgbm_ranks, bnn_ranks)

    print(f"\n  FK Attribution Comparison:")
    print(f"  {'FK':<15} {'LightGBM':<12} {'BNN':<12}")
    print(f"  {'-'*39}")
    for fk in fk_names:
        print(f"  {fk:<15} {lgbm_attr[fk]:>10.1f}% {bnn_attr[fk]:>10.1f}%")

    print(f"\n  Cross-method Spearman correlation: {rho_cross:.3f} (p={p_cross:.4f})")

    results['cross_method'] = {
        'rho': rho_cross,
        'p_value': p_cross
    }

    # Summary
    print("\n" + "="*70)
    print("FEASIBILITY TEST RESULTS")
    print("="*70)

    print(f"\n  Method               | Attr-Error ρ | Status")
    print(f"  {'-'*50}")
    print(f"  LightGBM Ensemble    | {rho_lgbm:>11.3f} | {'PASS' if rho_lgbm > 0.7 else 'FAIL'}")
    print(f"  Bayesian NN          | {rho_bnn:>11.3f} | {'PASS' if rho_bnn > 0.7 else 'FAIL'}")
    print(f"  {'-'*50}")
    print(f"  Cross-Method ρ       | {rho_cross:>11.3f} | {'PASS' if rho_cross > 0.7 else 'CHECK'}")

    # Verdict
    print(f"\n  VERDICT:")
    if rho_lgbm > 0.7 and rho_bnn > 0.7 and rho_cross > 0.7:
        verdict = "STRONG PASS - FK Attribution is UQ-method agnostic!"
        success = True
    elif rho_lgbm > 0.7 and rho_bnn > 0.5:
        verdict = "PARTIAL PASS - FK Attribution works but rankings differ somewhat"
        success = True
    else:
        verdict = "NEEDS INVESTIGATION - Results inconsistent across methods"
        success = False

    print(f"  {verdict}")

    results['verdict'] = verdict
    results['success'] = success

    # Save results
    import json

    def convert(obj):
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        return obj

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / 'feasibility_bnn.json'
    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"\n[5/5] Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = run_feasibility_test()
