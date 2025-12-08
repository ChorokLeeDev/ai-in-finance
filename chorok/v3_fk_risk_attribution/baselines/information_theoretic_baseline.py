"""
Information-Theoretic Uncertainty Attribution Baseline
======================================================

Inspired by InfoSHAP (Watson et al., NeurIPS 2023), but adapted for ensemble methods.

InfoSHAP: Explains predictive uncertainty using information-theoretic Shapley values
- Original: MC Dropout neural networks
- Our adaptation: Ensemble variance decomposition using mutual information concepts

Key Idea from InfoSHAP:
- Attribution should measure how much information each feature provides about uncertainty
- Uses KL divergence / mutual information to quantify

Our Adaptation:
- Feature's contribution to uncertainty = mutual information between feature and prediction variance
- I(X_fk; Var(Y)) where X_fk are features from FK group

Methods implemented:
1. MI-based attribution: Mutual information between FK features and ensemble variance
2. Entropy-based attribution: Change in predictive entropy when FK is permuted
3. KL-based attribution: KL divergence between full and FK-ablated prediction distributions

Author: ChorokLeeDev
Created: 2025-12-08
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import entropy, spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


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


def get_ensemble_predictions(models, X):
    """Get all ensemble member predictions."""
    return np.array([m.predict(X) for m in models])


def get_uncertainty(models, X):
    """Compute ensemble uncertainty (variance of predictions)."""
    preds = get_ensemble_predictions(models, X)
    return preds.var(axis=0)


def get_prediction(models, X):
    """Get ensemble mean prediction."""
    preds = get_ensemble_predictions(models, X)
    return preds.mean(axis=0)


def get_fk_grouping(col_to_fk):
    """Convert column->FK mapping to FK->columns mapping."""
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)
    return dict(fk_to_cols)


def compute_mi_attribution(models, X, fk_grouping, n_bins=10):
    """
    Information-Theoretic Attribution Method 1: Mutual Information

    Computes I(X_fk; σ²) - mutual information between FK features and uncertainty.
    Higher MI = FK features are more informative about uncertainty.

    Inspired by InfoSHAP's use of information theory for uncertainty explanation.
    """
    # Get uncertainty for each sample
    uncertainty = get_uncertainty(models, X)

    # Discretize uncertainty for MI computation
    uncertainty_discrete = KBinsDiscretizer(
        n_bins=n_bins, encode='ordinal', strategy='quantile'
    ).fit_transform(uncertainty.reshape(-1, 1)).flatten()

    results = {}

    for fk_group, cols in fk_grouping.items():
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            continue

        # Compute MI between each feature and uncertainty
        mi_scores = []
        for col in valid_cols:
            feature_values = X[col].values.reshape(-1, 1)
            try:
                mi = mutual_info_regression(feature_values, uncertainty_discrete, random_state=42)[0]
                mi_scores.append(mi)
            except:
                mi_scores.append(0)

        # Aggregate: sum of MI scores for FK group
        results[fk_group] = np.sum(mi_scores)

    # Normalize to percentages
    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def compute_entropy_attribution(models, X, fk_grouping, n_permute=5, n_bins=20):
    """
    Information-Theoretic Attribution Method 2: Predictive Entropy Change

    Measures change in entropy of prediction distribution when FK is permuted.
    H(P_permuted) - H(P_original)

    Higher entropy increase = FK is important for reducing predictive uncertainty.
    """
    # Get ensemble predictions
    preds = get_ensemble_predictions(models, X)  # (n_models, n_samples)

    # Compute base entropy of prediction distribution
    def compute_pred_entropy(predictions):
        """Entropy of the prediction distribution."""
        all_preds = predictions.flatten()
        hist, _ = np.histogram(all_preds, bins=n_bins, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        return entropy(hist)

    base_entropy = compute_pred_entropy(preds)

    results = {}

    for fk_group, cols in fk_grouping.items():
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            continue

        entropy_changes = []
        for _ in range(n_permute):
            X_perm = X.copy()
            for col in valid_cols:
                X_perm[col] = np.random.permutation(X_perm[col].values)

            preds_perm = get_ensemble_predictions(models, X_perm)
            perm_entropy = compute_pred_entropy(preds_perm)
            entropy_changes.append(perm_entropy - base_entropy)

        results[fk_group] = np.mean(entropy_changes)

    # Normalize to percentages
    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def compute_kl_attribution(models, X, fk_grouping, n_permute=5, n_bins=20):
    """
    Information-Theoretic Attribution Method 3: KL Divergence

    Measures KL(P_original || P_permuted) - how much the prediction distribution
    changes when FK is permuted.

    Higher KL divergence = FK has greater impact on prediction distribution.
    """
    # Get ensemble predictions
    preds = get_ensemble_predictions(models, X)  # (n_models, n_samples)

    # Compute base distribution
    all_preds_base = preds.flatten()
    hist_base, bin_edges = np.histogram(all_preds_base, bins=n_bins, density=True)
    hist_base = hist_base + 1e-10  # Avoid division by zero

    results = {}

    for fk_group, cols in fk_grouping.items():
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            continue

        kl_divergences = []
        for _ in range(n_permute):
            X_perm = X.copy()
            for col in valid_cols:
                X_perm[col] = np.random.permutation(X_perm[col].values)

            preds_perm = get_ensemble_predictions(models, X_perm)
            all_preds_perm = preds_perm.flatten()

            hist_perm, _ = np.histogram(all_preds_perm, bins=bin_edges, density=True)
            hist_perm = hist_perm + 1e-10

            # KL divergence: D_KL(P_base || P_perm)
            kl_div = entropy(hist_base, hist_perm)
            kl_divergences.append(kl_div)

        results[fk_group] = np.mean(kl_divergences)

    # Normalize to percentages
    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def compute_reluq_attribution(models, X, fk_grouping, n_permute=5):
    """
    RelUQ: Our method - permutation-based uncertainty attribution.
    Included for direct comparison.
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


def compute_error_impact(models, X, y, fk_grouping, n_permute=5):
    """Ground truth: Error impact when FK is permuted."""
    from sklearn.metrics import mean_absolute_error

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


def run_info_theoretic_comparison(X, y, col_to_fk, domain_name, n_runs=3):
    """
    Compare information-theoretic attribution methods against RelUQ and error impact.
    """
    print(f"\n{'='*70}")
    print(f"INFORMATION-THEORETIC BASELINE COMPARISON: {domain_name}")
    print(f"{'='*70}")

    fk_grouping = get_fk_grouping(col_to_fk)
    fk_list = list(fk_grouping.keys())
    n_fk_groups = len(fk_list)

    print(f"Features: {len(X.columns)}, FK groups: {n_fk_groups}")
    print(f"FK groups: {fk_list}")

    # Collect results across runs
    all_results = {
        'mi': [],      # Mutual Information
        'entropy': [], # Predictive Entropy
        'kl': [],      # KL Divergence
        'reluq': [],   # Our method
        'error': []    # Ground truth
    }

    for run in range(n_runs):
        print(f"\n  Run {run+1}/{n_runs}...")

        # Train ensemble
        models = train_ensemble(X, y, n_models=5, base_seed=42 + run * 10)

        # Information-theoretic methods
        print(f"    Computing MI attribution...")
        mi_attr = compute_mi_attribution(models, X, fk_grouping)
        all_results['mi'].append(mi_attr)

        print(f"    Computing Entropy attribution...")
        entropy_attr = compute_entropy_attribution(models, X, fk_grouping)
        all_results['entropy'].append(entropy_attr)

        print(f"    Computing KL attribution...")
        kl_attr = compute_kl_attribution(models, X, fk_grouping)
        all_results['kl'].append(kl_attr)

        # RelUQ (our method)
        print(f"    Computing RelUQ attribution...")
        reluq_attr = compute_reluq_attribution(models, X, fk_grouping)
        all_results['reluq'].append(reluq_attr)

        # Error impact (ground truth)
        print(f"    Computing Error impact...")
        error_attr = compute_error_impact(models, X, y, fk_grouping)
        all_results['error'].append(error_attr)

    # Average across runs
    avg_results = {}
    for method, results_list in all_results.items():
        avg_results[method] = {}
        for fk in fk_list:
            values = [r.get(fk, 0) for r in results_list]
            avg_results[method][fk] = np.mean(values)

    # Compute correlations with error impact
    error_values = [avg_results['error'].get(fk, 0) for fk in fk_list]

    correlations = {}
    for method in ['mi', 'entropy', 'kl', 'reluq']:
        method_values = [avg_results[method].get(fk, 0) for fk in fk_list]
        if len(fk_list) >= 3:
            corr, p_val = spearmanr(method_values, error_values)
        else:
            corr, p_val = float('nan'), float('nan')
        correlations[method] = {'spearman': corr, 'p_value': p_val}

    # Display results
    print(f"\n  --- Attribution Results (%) ---")
    print(f"  {'FK Group':<15} {'MI':<10} {'Entropy':<10} {'KL':<10} {'RelUQ':<10} {'Error':<10}")
    print(f"  {'-'*65}")

    for fk in fk_list:
        mi_val = avg_results['mi'].get(fk, 0)
        ent_val = avg_results['entropy'].get(fk, 0)
        kl_val = avg_results['kl'].get(fk, 0)
        reluq_val = avg_results['reluq'].get(fk, 0)
        err_val = avg_results['error'].get(fk, 0)
        print(f"  {fk:<15} {mi_val:>8.1f}% {ent_val:>8.1f}% {kl_val:>8.1f}% {reluq_val:>8.1f}% {err_val:>8.1f}%")

    print(f"\n  --- Correlation with Error Impact (Spearman ρ) ---")
    method_names = {
        'mi': 'Mutual Information',
        'entropy': 'Predictive Entropy',
        'kl': 'KL Divergence',
        'reluq': 'RelUQ (Ours)'
    }

    for method, stats in correlations.items():
        corr = stats['spearman']
        p_val = stats['p_value']
        corr_str = f"{corr:.3f}" if not np.isnan(corr) else "N/A"
        p_str = f"{p_val:.3f}" if not np.isnan(p_val) else "N/A"
        print(f"  {method_names[method]:<25} ρ={corr_str:<8} (p={p_str})")

    return {
        'domain': domain_name,
        'n_fk_groups': n_fk_groups,
        'attributions': avg_results,
        'correlations': correlations
    }


def load_from_cache(cache_file):
    """Load data from cached pickle file."""
    import pickle
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    # Handle different cache formats
    if len(data) == 4:
        X, y, feature_cols, col_to_fk = data
        # Check if col_to_fk is actually a dict
        if isinstance(col_to_fk, dict):
            return X, y, feature_cols, col_to_fk
        else:
            # Invalid format
            return None, None, None, None
    return None, None, None, None


def run_all_domains():
    """Run comparison across all domains."""
    import os
    from pathlib import Path

    CACHE_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache')

    print("="*70)
    print("INFORMATION-THEORETIC BASELINE COMPARISON")
    print("Inspired by InfoSHAP (Watson et al., NeurIPS 2023)")
    print("="*70)

    all_results = {}

    # SALT (from cache)
    print("\n[1/4] Loading SALT data from cache...")
    salt_cache = CACHE_DIR / 'data_salt_PLANT_2000.pkl'
    if salt_cache.exists():
        result = load_from_cache(salt_cache)
        if result[0] is not None:
            X_salt, y_salt, _, col_to_fk_salt = result
            salt_results = run_info_theoretic_comparison(X_salt, y_salt, col_to_fk_salt, "SALT (ERP)")
            all_results['salt'] = salt_results
        else:
            print("  SALT cache format invalid, skipping...")
    else:
        print("  SALT cache not found, skipping...")

    # Stack (from cache)
    print("\n[2/4] Loading Stack data from cache...")
    stack_cache = CACHE_DIR / 'data_stack_post-votes_2000.pkl'
    if stack_cache.exists():
        result = load_from_cache(stack_cache)
        if result[0] is not None:
            X_stack, y_stack, _, col_to_fk_stack = result
            stack_results = run_info_theoretic_comparison(X_stack, y_stack, col_to_fk_stack, "Stack (Q&A)")
            all_results['stack'] = stack_results
        else:
            print("  Stack cache format invalid, skipping...")
    else:
        print("  Stack cache not found, skipping...")

    # Avito (from cache)
    print("\n[3/4] Loading Avito data from cache...")
    avito_cache = CACHE_DIR / 'data_avito_ad-ctr_3000.pkl'
    if avito_cache.exists():
        result = load_from_cache(avito_cache)
        if result[0] is not None:
            X_avito, y_avito, _, col_to_fk_avito = result
            # Subsample to 2000 for consistency
            if len(X_avito) > 2000:
                idx = X_avito.sample(2000, random_state=42).index
                X_avito = X_avito.loc[idx]
                y_avito = y_avito.loc[idx]
            avito_results = run_info_theoretic_comparison(X_avito, y_avito, col_to_fk_avito, "Avito (Classifieds)")
            all_results['avito'] = avito_results
        else:
            print("  Avito cache format invalid, skipping...")
    else:
        print("  Avito cache not found, skipping...")

    # Amazon (from cache)
    print("\n[4/4] Loading Amazon data from cache...")
    amazon_cache = CACHE_DIR / 'data_amazon_user-ltv_3000.pkl'
    if amazon_cache.exists():
        result = load_from_cache(amazon_cache)
        if result[0] is not None:
            X_amazon, y_amazon, _, col_to_fk_amazon = result
            # Subsample to 2000 for consistency
            if len(X_amazon) > 2000:
                idx = X_amazon.sample(2000, random_state=42).index
                X_amazon = X_amazon.loc[idx]
                y_amazon = y_amazon.loc[idx]
            amazon_results = run_info_theoretic_comparison(X_amazon, y_amazon, col_to_fk_amazon, "Amazon (E-commerce)")
            all_results['amazon'] = amazon_results
        else:
            print("  Amazon cache format invalid, skipping...")
    else:
        print("  Amazon cache not found, skipping...")

    # Summary Table
    print("\n" + "="*70)
    print("SUMMARY: Correlation with Error Impact (Spearman ρ)")
    print("="*70)
    print(f"\n{'Domain':<20} {'MI':<12} {'Entropy':<12} {'KL':<12} {'RelUQ':<12}")
    print("-"*70)

    for domain_key, results in all_results.items():
        domain = results['domain']
        mi = results['correlations']['mi']['spearman']
        ent = results['correlations']['entropy']['spearman']
        kl = results['correlations']['kl']['spearman']
        reluq = results['correlations']['reluq']['spearman']

        mi_str = f"{mi:.3f}" if not np.isnan(mi) else "N/A"
        ent_str = f"{ent:.3f}" if not np.isnan(ent) else "N/A"
        kl_str = f"{kl:.3f}" if not np.isnan(kl) else "N/A"
        reluq_str = f"{reluq:.3f}" if not np.isnan(reluq) else "N/A"

        print(f"{domain:<20} {mi_str:<12} {ent_str:<12} {kl_str:<12} {reluq_str:<12}")

    # Save results
    import json
    import os

    RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/results'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if obj is None or (isinstance(obj, float) and np.isnan(obj)):
            return None
        return obj

    output_path = f"{RESULTS_DIR}/info_theoretic_baseline_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\n[SAVED] {output_path}")

    return all_results


if __name__ == "__main__":
    results = run_all_domains()
