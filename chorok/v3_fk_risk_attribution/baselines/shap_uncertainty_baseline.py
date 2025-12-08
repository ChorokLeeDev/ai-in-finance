"""
SHAP Uncertainty Attribution Baseline
=====================================

Compare RelUQ (permutation-based) with SHAP-based uncertainty attribution.

Key question: Does FK grouping benefit SHAP too, or is it specific to permutation?

Methods:
1. RelUQ (Permutation): Permute FK columns → measure uncertainty increase
2. SHAP Variance: Variance of SHAP values across ensemble → aggregate by FK

Expected outcome:
- Both should show similar FK rankings (FK grouping is the contribution)
- SHAP may be slower but more theoretically grounded
- RelUQ advantage: directly measures uncertainty, not prediction importance

Author: ChorokLeeDev
Created: 2025-12-08
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from collections import defaultdict
from scipy.stats import spearmanr
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: shap not installed. Run: pip install shap")
    SHAP_AVAILABLE = False

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader_salt import load_salt_data
from data_loader_trial import load_trial_data
from data_loader_stack import load_stack_data

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/results'


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
    """Compute ensemble uncertainty (variance of predictions)."""
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


# =============================================================================
# Method 1: RelUQ (Permutation-based Uncertainty Attribution)
# =============================================================================

def compute_reluq_attribution(models, X, fk_grouping, n_permute=10):
    """
    RelUQ method: Permute FK columns → measure uncertainty increase.
    """
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


# =============================================================================
# Method 2: SHAP Variance Attribution
# =============================================================================

def compute_shap_variance_attribution(models, X, fk_grouping, sample_size=500):
    """
    SHAP-based uncertainty attribution:
    - Compute SHAP values for each model in ensemble
    - Variance of SHAP values across models = uncertainty attribution
    - Aggregate by FK group
    """
    if not SHAP_AVAILABLE:
        return None

    # Sample for computational efficiency
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X

    # Compute SHAP values for each model
    all_shap_values = []
    for model in models:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        all_shap_values.append(shap_values)

    # Stack: (n_models, n_samples, n_features)
    shap_stack = np.array(all_shap_values)

    # Variance across models for each feature
    # Shape: (n_samples, n_features)
    shap_variance = np.var(shap_stack, axis=0)

    # Mean variance per feature
    # Shape: (n_features,)
    feature_variance = shap_variance.mean(axis=0)

    # Aggregate by FK group
    feature_cols = list(X.columns)
    results = {}

    for fk_group, cols in fk_grouping.items():
        valid_cols = [c for c in cols if c in feature_cols]
        if not valid_cols:
            continue

        # Sum variance for features in this FK group
        fk_variance = 0
        for col in valid_cols:
            col_idx = feature_cols.index(col)
            fk_variance += feature_variance[col_idx]

        results[fk_group] = fk_variance

    # Normalize to percentages
    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


# =============================================================================
# Method 3: SHAP Mean Absolute (Feature Importance, not Uncertainty)
# =============================================================================

def compute_shap_importance_attribution(models, X, fk_grouping, sample_size=500):
    """
    Traditional SHAP importance (mean |SHAP|), NOT uncertainty.
    Included for comparison: how different is importance vs uncertainty?
    """
    if not SHAP_AVAILABLE:
        return None

    # Sample for computational efficiency
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X

    # Use ensemble mean model's SHAP (representative)
    # Or average SHAP across all models
    all_shap_abs = []
    for model in models:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        all_shap_abs.append(np.abs(shap_values))

    # Mean absolute SHAP: (n_samples, n_features)
    mean_abs_shap = np.mean(all_shap_abs, axis=0)

    # Mean across samples
    feature_importance = mean_abs_shap.mean(axis=0)

    # Aggregate by FK group
    feature_cols = list(X.columns)
    results = {}

    for fk_group, cols in fk_grouping.items():
        valid_cols = [c for c in cols if c in feature_cols]
        if not valid_cols:
            continue

        fk_importance = 0
        for col in valid_cols:
            col_idx = feature_cols.index(col)
            fk_importance += feature_importance[col_idx]

        results[fk_group] = fk_importance

    # Normalize to percentages
    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


# =============================================================================
# Ground Truth: Error Impact
# =============================================================================

def compute_error_impact(models, X, y, fk_grouping, n_permute=10):
    """
    Ground truth: Error increase when FK is permuted.
    """
    base_pred = get_prediction(models, X)
    base_mae = mean_absolute_error(y, base_pred)

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
            perm_mae = mean_absolute_error(y, perm_pred)
            deltas.append(perm_mae - base_mae)

        results[fk_group] = np.mean(deltas)

    # Normalize to percentages
    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


# =============================================================================
# Main Comparison
# =============================================================================

def run_shap_comparison(X, y, col_to_fk, domain_name, n_runs=3):
    """
    Compare all methods against error impact ground truth.
    """
    print(f"\n{'='*70}")
    print(f"SHAP BASELINE COMPARISON: {domain_name}")
    print(f"{'='*70}")

    fk_grouping = get_fk_grouping(col_to_fk)
    fk_list = list(fk_grouping.keys())
    n_fk_groups = len(fk_list)

    print(f"Features: {len(X.columns)}, FK groups: {n_fk_groups}")
    print(f"FK groups: {fk_list}")

    all_results = {
        'reluq': [],
        'shap_var': [],
        'shap_imp': [],
        'error': []
    }

    for run in range(n_runs):
        print(f"\n  Run {run+1}/{n_runs}...")

        # Train ensemble
        models = train_ensemble(X, y, n_models=5, base_seed=42 + run * 10)

        # Method 1: RelUQ (Permutation)
        reluq = compute_reluq_attribution(models, X, fk_grouping, n_permute=5)
        all_results['reluq'].append(reluq)

        # Method 2: SHAP Variance
        if SHAP_AVAILABLE:
            shap_var = compute_shap_variance_attribution(models, X, fk_grouping)
            all_results['shap_var'].append(shap_var)

            # Method 3: SHAP Importance
            shap_imp = compute_shap_importance_attribution(models, X, fk_grouping)
            all_results['shap_imp'].append(shap_imp)

        # Ground Truth: Error Impact
        error = compute_error_impact(models, X, y, fk_grouping, n_permute=5)
        all_results['error'].append(error)

    # Average across runs
    avg_results = {}
    for method, runs in all_results.items():
        if runs and runs[0] is not None:
            avg = {}
            for fk in fk_list:
                vals = [r.get(fk, 0) for r in runs if r is not None]
                avg[fk] = np.mean(vals) if vals else 0
            avg_results[method] = avg

    # Compute correlations with error impact
    error_vals = [avg_results['error'].get(fk, 0) for fk in fk_list]
    correlations = {}

    for method in ['reluq', 'shap_var', 'shap_imp']:
        if method in avg_results:
            method_vals = [avg_results[method].get(fk, 0) for fk in fk_list]
            if len(fk_list) >= 3:
                corr, p = spearmanr(method_vals, error_vals)
                correlations[method] = {'spearman': corr, 'p_value': p}
            else:
                correlations[method] = {'spearman': None, 'p_value': None}

    # Print results
    print(f"\n  --- Attribution Results ---")
    print(f"  {'FK Group':<15} {'RelUQ':<10} {'SHAP-Var':<10} {'SHAP-Imp':<10} {'Error':<10}")
    print(f"  {'-'*55}")

    for fk in fk_list:
        reluq_val = avg_results.get('reluq', {}).get(fk, 0)
        shap_var_val = avg_results.get('shap_var', {}).get(fk, 0)
        shap_imp_val = avg_results.get('shap_imp', {}).get(fk, 0)
        error_val = avg_results.get('error', {}).get(fk, 0)
        print(f"  {fk:<15} {reluq_val:>8.1f}% {shap_var_val:>8.1f}% {shap_imp_val:>8.1f}% {error_val:>8.1f}%")

    print(f"\n  --- Correlation with Error Impact ---")
    print(f"  {'Method':<15} {'Spearman':<12} {'p-value':<12}")
    print(f"  {'-'*35}")

    for method, corr_data in correlations.items():
        corr = corr_data['spearman']
        p = corr_data['p_value']
        corr_str = f"{corr:.3f}" if corr is not None else "N/A"
        p_str = f"{p:.3f}" if p is not None else "N/A"
        print(f"  {method:<15} {corr_str:<12} {p_str:<12}")

    # Determine winner
    if correlations:
        valid_methods = {k: v['spearman'] for k, v in correlations.items() if v['spearman'] is not None}
        if valid_methods:
            best_method = max(valid_methods.keys(), key=lambda k: valid_methods[k])
            print(f"\n  Best method: {best_method} (ρ={valid_methods[best_method]:.3f})")

    return {
        'domain': domain_name,
        'n_fk_groups': n_fk_groups,
        'attributions': avg_results,
        'correlations': correlations
    }


def run_all_domains():
    """Run SHAP comparison across all domains."""
    print("="*70)
    print("SHAP BASELINE COMPARISON")
    print("RelUQ vs SHAP Variance vs SHAP Importance")
    print("="*70)

    all_results = {}

    # SALT (ERP)
    print("\n[1/3] Loading SALT data...")
    X_salt, y_salt, _, col_to_fk_salt = load_salt_data(sample_size=2000)
    salt_results = run_shap_comparison(X_salt, y_salt, col_to_fk_salt, "SALT (ERP)")
    all_results['salt'] = salt_results

    # Trial (Clinical)
    print("\n[2/3] Loading Trial data...")
    X_trial, y_trial, _, col_to_fk_trial = load_trial_data(sample_size=2000)
    trial_results = run_shap_comparison(X_trial, y_trial, col_to_fk_trial, "Trial (Clinical)")
    all_results['trial'] = trial_results

    # Stack (Q&A)
    print("\n[3/3] Loading Stack data...")
    X_stack, y_stack, _, col_to_fk_stack = load_stack_data(sample_size=2000)
    stack_results = run_shap_comparison(X_stack, y_stack, col_to_fk_stack, "Stack (Q&A)")
    all_results['stack'] = stack_results

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Method Comparison")
    print("="*70)

    print(f"\n{'Domain':<25} {'RelUQ':<12} {'SHAP-Var':<12} {'SHAP-Imp':<12}")
    print("-"*60)

    for domain, results in all_results.items():
        corrs = results['correlations']
        reluq = corrs.get('reluq', {}).get('spearman', None)
        shap_var = corrs.get('shap_var', {}).get('spearman', None)
        shap_imp = corrs.get('shap_imp', {}).get('spearman', None)

        reluq_str = f"{reluq:.3f}" if reluq is not None else "N/A"
        shap_var_str = f"{shap_var:.3f}" if shap_var is not None else "N/A"
        shap_imp_str = f"{shap_imp:.3f}" if shap_imp is not None else "N/A"

        print(f"{results['domain']:<25} {reluq_str:<12} {shap_var_str:<12} {shap_imp_str:<12}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = f"{RESULTS_DIR}/shap_baseline_comparison.json"

    def convert_to_serializable(obj):
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if obj is None:
            return None
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\n[SAVED] {output_path}")

    return all_results


if __name__ == "__main__":
    results = run_all_domains()
