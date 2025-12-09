"""
Scale-Up Validation: Large Sample Size Experiments
===================================================

Goal: Validate that FK Attribution works at scale (10K+ samples).

Experiments:
1. Attribution stability across sample sizes (1K → 10K)
2. Correlation with error impact at scale
3. Confidence intervals for all metrics

Author: ChorokLeeDev
Created: 2025-12-08
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr, bootstrap
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

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
    """Compute FK attribution with more permutations for stability."""
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

    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def compute_error_impact(models, X, y, fk_grouping, n_permute=10):
    """Compute error impact with more permutations."""
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

    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def compute_correlation_with_ci(attr_values, error_values, n_bootstrap=1000, ci=0.95):
    """
    Compute Spearman correlation with bootstrap confidence interval.
    """
    if len(attr_values) < 3:
        return None, None, None, None

    # Point estimate
    corr, p_value = spearmanr(attr_values, error_values)

    # Bootstrap CI
    rng = np.random.default_rng(42)

    def statistic(indices):
        return spearmanr(np.array(attr_values)[indices], np.array(error_values)[indices])[0]

    n = len(attr_values)
    bootstrap_corrs = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        try:
            bootstrap_corrs.append(statistic(indices))
        except:
            pass

    if bootstrap_corrs:
        alpha = 1 - ci
        ci_low = np.percentile(bootstrap_corrs, alpha/2 * 100)
        ci_high = np.percentile(bootstrap_corrs, (1 - alpha/2) * 100)
    else:
        ci_low, ci_high = None, None

    return corr, p_value, ci_low, ci_high


def run_scale_experiment(X_full, y_full, col_to_fk, domain_name,
                         sample_sizes=[1000, 2000, 3000, 5000, 10000],
                         n_runs=5):
    """
    Run experiments at multiple sample sizes with statistical rigor.
    """
    print(f"\n{'='*70}")
    print(f"SCALE-UP VALIDATION: {domain_name}")
    print(f"{'='*70}")

    fk_grouping = get_fk_grouping(col_to_fk)
    fk_list = list(fk_grouping.keys())

    print(f"Total samples available: {len(X_full)}")
    print(f"FK groups: {fk_list}")

    all_results = []

    for sample_size in sample_sizes:
        if sample_size > len(X_full):
            print(f"\n  Skipping n={sample_size} (not enough data)")
            continue

        print(f"\n  --- Sample Size: {sample_size} ---")

        correlations = []
        attributions_list = []

        for run in range(n_runs):
            seed = 42 + run * 10
            np.random.seed(seed)

            # Sample data
            idx = np.random.choice(len(X_full), size=sample_size, replace=False)
            X = X_full.iloc[idx].copy()
            y = y_full.iloc[idx].copy()

            # Train and evaluate
            models = train_ensemble(X, y, n_models=5, base_seed=seed)
            attr = compute_fk_attribution(models, X, fk_grouping, n_permute=10)
            error = compute_error_impact(models, X, y, fk_grouping, n_permute=10)

            attributions_list.append(attr)

            # Correlation
            attr_vals = [attr.get(fk, 0) for fk in fk_list]
            err_vals = [error.get(fk, 0) for fk in fk_list]

            if len(fk_list) >= 3:
                corr, _ = spearmanr(attr_vals, err_vals)
                correlations.append(corr)

        # Statistics
        if correlations:
            mean_corr = np.mean(correlations)
            std_corr = np.std(correlations)
            ci_low = np.percentile(correlations, 2.5)
            ci_high = np.percentile(correlations, 97.5)

            print(f"    Correlation: ρ = {mean_corr:.3f} ± {std_corr:.3f}")
            print(f"    95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

            # Attribution stability (coefficient of variation across runs)
            stability_scores = []
            for fk in fk_list:
                fk_values = [a.get(fk, 0) for a in attributions_list]
                if np.mean(fk_values) > 0:
                    cv = np.std(fk_values) / np.mean(fk_values)
                    stability_scores.append(1 - min(cv, 1))  # Higher = more stable

            avg_stability = np.mean(stability_scores) if stability_scores else 0
            print(f"    Attribution Stability: {avg_stability:.3f}")

            all_results.append({
                'sample_size': sample_size,
                'mean_correlation': mean_corr,
                'std_correlation': std_corr,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'stability': avg_stability,
                'n_runs': n_runs
            })

    # Summary
    print(f"\n  --- Scale-Up Summary ---")
    print(f"  {'Size':<10} {'ρ Mean':<10} {'ρ Std':<10} {'95% CI':<20} {'Stability':<10}")
    print(f"  {'-'*60}")

    for r in all_results:
        ci_str = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        print(f"  {r['sample_size']:<10} {r['mean_correlation']:<10.3f} {r['std_correlation']:<10.3f} {ci_str:<20} {r['stability']:<10.3f}")

    return {
        'domain': domain_name,
        'n_fk_groups': len(fk_list),
        'scale_results': all_results
    }


def load_from_cache(cache_file):
    """Load data from cached pickle file."""
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    if len(data) == 4:
        X, y, feature_cols, col_to_fk = data
        if isinstance(col_to_fk, dict):
            return X, y, feature_cols, col_to_fk
    return None, None, None, None


def run_all_domains():
    """Run scale-up validation across domains."""
    print("="*70)
    print("SCALE-UP VALIDATION")
    print("Testing FK Attribution at 10K+ samples with statistical rigor")
    print("="*70)

    all_results = {}

    # SALT - check for larger cache
    print("\n[1/2] Loading SALT data...")
    salt_caches = [
        ('data_salt_PLANT_10000.pkl', 10000),
        ('data_salt_PLANT_5000.pkl', 5000),
        ('data_salt_PLANT_3000.pkl', 3000),
        ('data_salt_PLANT_2000.pkl', 2000),
    ]

    for cache_name, expected_size in salt_caches:
        cache_path = CACHE_DIR / cache_name
        if cache_path.exists():
            result = load_from_cache(cache_path)
            if result[0] is not None:
                X, y, _, col_to_fk = result
                print(f"  Loaded {cache_name}: {len(X)} samples")

                # Determine sample sizes based on available data
                max_size = len(X)
                sample_sizes = [s for s in [1000, 2000, 3000, 5000, 10000] if s <= max_size]

                salt_results = run_scale_experiment(
                    X, y, col_to_fk, "SALT (ERP)",
                    sample_sizes=sample_sizes,
                    n_runs=5
                )
                all_results['salt'] = salt_results
                break

    # Avito
    print("\n[2/2] Loading Avito data...")
    avito_cache = CACHE_DIR / 'data_avito_ad-ctr_3000.pkl'
    if avito_cache.exists():
        result = load_from_cache(avito_cache)
        if result[0] is not None:
            X, y, _, col_to_fk = result
            print(f"  Loaded: {len(X)} samples")

            sample_sizes = [s for s in [1000, 2000, 3000] if s <= len(X)]

            avito_results = run_scale_experiment(
                X, y, col_to_fk, "Avito (Classifieds)",
                sample_sizes=sample_sizes,
                n_runs=5
            )
            all_results['avito'] = avito_results

    # Final summary
    print("\n" + "="*70)
    print("SCALE-UP VALIDATION SUMMARY")
    print("="*70)

    for domain_key, results in all_results.items():
        domain = results['domain']
        print(f"\n{domain}:")
        if results['scale_results']:
            largest = results['scale_results'][-1]
            print(f"  Largest scale: n={largest['sample_size']}")
            print(f"  Correlation: ρ = {largest['mean_correlation']:.3f} [{largest['ci_low']:.3f}, {largest['ci_high']:.3f}]")
            print(f"  Stability: {largest['stability']:.3f}")

    # Save results
    import json

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

    output_path = RESULTS_DIR / 'scale_up_validation.json'
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\n[SAVED] {output_path}")

    return all_results


if __name__ == "__main__":
    results = run_all_domains()
