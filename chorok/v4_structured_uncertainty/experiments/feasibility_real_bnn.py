"""
Feasibility Test: FK Attribution with REAL Bayesian Neural Network
===================================================================

This is the proper test - using actual BNN with weight distributions,
not MC Dropout approximations.

Key question: Does FK Attribution work with true Bayesian inference?

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

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'methods'))

from methods.bnn_pyro import train_real_bnn
from methods.ensemble_lgbm import train_lgbm_ensemble
from methods.mc_dropout import train_mc_dropout_ensemble

CACHE_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache')
RESULTS_DIR = Path(__file__).parent.parent / 'results'


def load_salt_data():
    """Load SALT data from v3 cache."""
    cache_file = CACHE_DIR / 'data_salt_PLANT_2000.pkl'
    if not cache_file.exists():
        return None, None, None, None

    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    if len(data) == 4:
        return data
    return None, None, None, None


def get_fk_grouping(col_to_fk):
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)
    return dict(fk_to_cols)


def compute_fk_attribution(get_uncertainty_fn, X, fk_grouping, n_permute=5):
    """Compute FK-level uncertainty attribution."""
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

    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def compute_error_impact(predict_fn, X, y, fk_grouping, n_permute=5):
    """Compute FK-level error impact."""
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

    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def run_test():
    """Main test comparing all three UQ methods."""
    print("="*70)
    print("FK ATTRIBUTION: Real BNN vs MC Dropout vs LightGBM Ensemble")
    print("="*70)

    # Load data
    print("\n[1/6] Loading SALT data...")
    X, y, feature_cols, col_to_fk = load_salt_data()
    if X is None:
        print("ERROR: Could not load data")
        return None

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_cols)
    y = np.array(y).flatten()

    fk_grouping = get_fk_grouping(col_to_fk)
    print(f"  Data shape: {X.shape}")
    print(f"  FK groups: {list(fk_grouping.keys())}")

    # Subsample
    n_samples = min(1500, len(X))
    idx = np.random.RandomState(42).permutation(len(X))[:n_samples]
    X_sub = X.iloc[idx].reset_index(drop=True)
    y_sub = y[idx]
    print(f"  Using {n_samples} samples")

    results = {}

    # Method 1: LightGBM Ensemble
    print("\n[2/6] Training LightGBM Ensemble...")
    lgbm = train_lgbm_ensemble(X_sub.values, y_sub, n_models=5, seed=42)

    def lgbm_unc(X_in):
        return lgbm.get_uncertainty(X_in.values if hasattr(X_in, 'values') else X_in).mean()

    def lgbm_pred(X_in):
        return lgbm.get_mean_prediction(X_in.values if hasattr(X_in, 'values') else X_in)

    lgbm_attr = compute_fk_attribution(lgbm_unc, X_sub, fk_grouping)
    lgbm_error = compute_error_impact(lgbm_pred, X_sub, y_sub, fk_grouping)
    rho_lgbm, _ = spearmanr(list(lgbm_attr.values()), list(lgbm_error.values()))

    print(f"  Attribution-Error ρ: {rho_lgbm:.3f}")
    results['lgbm'] = {'attr': lgbm_attr, 'error': lgbm_error, 'rho': rho_lgbm}

    # Method 2: MC Dropout Ensemble (NOT real BNN)
    print("\n[3/6] Training MC Dropout Ensemble (not real BNN)...")
    mc_dropout = train_mc_dropout_ensemble(X_sub.values, y_sub, n_networks=5, mc_samples=10, epochs=100, seed=42)

    def mc_unc(X_in):
        return mc_dropout.get_uncertainty(X_in.values if hasattr(X_in, 'values') else X_in).mean()

    def mc_pred(X_in):
        return mc_dropout.get_mean_prediction(X_in.values if hasattr(X_in, 'values') else X_in)

    mc_attr = compute_fk_attribution(mc_unc, X_sub, fk_grouping)
    mc_error = compute_error_impact(mc_pred, X_sub, y_sub, fk_grouping)
    rho_mc, _ = spearmanr(list(mc_attr.values()), list(mc_error.values()))

    print(f"  Attribution-Error ρ: {rho_mc:.3f}")
    results['mc_dropout'] = {'attr': mc_attr, 'error': mc_error, 'rho': rho_mc}

    # Method 3: REAL BNN with Pyro
    print("\n[4/6] Training REAL Bayesian Neural Network (Pyro VI)...")
    real_bnn = train_real_bnn(X_sub.values, y_sub, hidden_dim=64, epochs=800, lr=0.01, seed=42)

    def bnn_unc(X_in):
        return real_bnn.get_uncertainty(X_in.values if hasattr(X_in, 'values') else X_in, n_samples=50).mean()

    def bnn_pred(X_in):
        return real_bnn.get_mean_prediction(X_in.values if hasattr(X_in, 'values') else X_in, n_samples=50)

    print("\n[5/6] Computing FK Attribution for Real BNN...")
    bnn_attr = compute_fk_attribution(bnn_unc, X_sub, fk_grouping, n_permute=3)
    bnn_error = compute_error_impact(bnn_pred, X_sub, y_sub, fk_grouping, n_permute=3)
    rho_bnn, _ = spearmanr(list(bnn_attr.values()), list(bnn_error.values()))

    print(f"  Attribution-Error ρ: {rho_bnn:.3f}")
    results['real_bnn'] = {'attr': bnn_attr, 'error': bnn_error, 'rho': rho_bnn}

    # Comparison
    print("\n[6/6] Cross-Method Comparison")
    print("="*70)

    fk_names = list(lgbm_attr.keys())

    print(f"\n{'FK':<15} {'LightGBM':<12} {'MC Dropout':<12} {'Real BNN':<12}")
    print("-"*51)
    for fk in fk_names:
        print(f"{fk:<15} {lgbm_attr[fk]:>10.1f}% {mc_attr[fk]:>10.1f}% {bnn_attr[fk]:>10.1f}%")

    # Cross-method correlations
    lgbm_ranks = [lgbm_attr[fk] for fk in fk_names]
    mc_ranks = [mc_attr[fk] for fk in fk_names]
    bnn_ranks = [bnn_attr[fk] for fk in fk_names]

    rho_lgbm_mc, _ = spearmanr(lgbm_ranks, mc_ranks)
    rho_lgbm_bnn, _ = spearmanr(lgbm_ranks, bnn_ranks)
    rho_mc_bnn, _ = spearmanr(mc_ranks, bnn_ranks)

    print(f"\nCross-Method Correlations:")
    print(f"  LightGBM ↔ MC Dropout: {rho_lgbm_mc:.3f}")
    print(f"  LightGBM ↔ Real BNN:   {rho_lgbm_bnn:.3f}")
    print(f"  MC Dropout ↔ Real BNN: {rho_mc_bnn:.3f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n  Method            | Attr-Error ρ | Status")
    print(f"  {'-'*50}")
    print(f"  LightGBM Ensemble | {rho_lgbm:>12.3f} | {'PASS' if rho_lgbm > 0.7 else 'FAIL'}")
    print(f"  MC Dropout        | {rho_mc:>12.3f} | {'PASS' if rho_mc > 0.7 else 'FAIL'}")
    print(f"  Real BNN (Pyro)   | {rho_bnn:>12.3f} | {'PASS' if rho_bnn > 0.7 else 'FAIL'}")

    # Verdict
    all_pass = rho_lgbm > 0.7 and rho_mc > 0.7 and rho_bnn > 0.7
    cross_agree = rho_lgbm_bnn > 0.5

    if all_pass and cross_agree:
        verdict = "STRONG PASS: FK Attribution works across all UQ methods including real BNN!"
    elif all_pass:
        verdict = "PASS: All methods show Attr-Error correlation, but rankings differ"
    elif rho_bnn > 0.7:
        verdict = "PARTIAL: Real BNN works, some other methods may have issues"
    else:
        verdict = "NEEDS INVESTIGATION: Real BNN results inconsistent"

    print(f"\n  VERDICT: {verdict}")

    results['cross_method'] = {
        'lgbm_mc': rho_lgbm_mc,
        'lgbm_bnn': rho_lgbm_bnn,
        'mc_bnn': rho_mc_bnn
    }
    results['verdict'] = verdict

    # Save
    import json

    def convert(obj):
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        return obj

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / 'feasibility_real_bnn.json'
    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"\n  Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = run_test()
