"""
Ablation Studies for RelUQ
==========================

Tests sensitivity to hyperparameters:
1. K: Ensemble size (number of models)
2. P: Permutation runs
3. n: Sample size
4. subsample: Subsampling rate for diversity
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
import lightgbm as lgb
import json
import time
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution')
from data_loader_f1 import load_f1_data


# =============================================================================
# Core Functions
# =============================================================================

def train_ensemble(X, y, n_models=5, base_seed=42, subsample=0.8, colsample=0.8):
    """Train LightGBM ensemble."""
    models = []
    for i in range(n_models):
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            subsample=subsample,
            colsample_bytree=colsample,
            random_state=base_seed + i,
            verbose=-1,
            force_col_wise=True
        )
        model.fit(X, y)
        models.append(model)
    return models


def get_uncertainty(models, X):
    """Compute ensemble variance."""
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0).mean()


def fk_attribution(models, X, col_to_fk, n_perm=5):
    """FK-level attribution via permutation."""
    base_unc = get_uncertainty(models, X)

    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        if col in X.columns:
            fk_to_cols[fk].append(col)

    fk_deltas = {}
    for fk, cols in fk_to_cols.items():
        deltas = []
        for _ in range(n_perm):
            X_perm = X.copy()
            for col in cols:
                X_perm[col] = np.random.permutation(X_perm[col].values)
            perm_unc = get_uncertainty(models, X_perm)
            deltas.append(perm_unc - base_unc)
        fk_deltas[fk] = np.mean(deltas)

    total = sum(max(0, d) for d in fk_deltas.values())
    if total > 0:
        return {fk: max(0, d) / total * 100 for fk, d in fk_deltas.items()}
    return {fk: 0.0 for fk in fk_deltas}


def compute_stability(results_list):
    """Compute pairwise Spearman correlation."""
    if len(results_list) < 2:
        return np.nan

    keys = list(results_list[0].keys())
    correlations = []

    for i in range(len(results_list)):
        for j in range(i + 1, len(results_list)):
            vals_i = [results_list[i].get(k, 0) for k in keys]
            vals_j = [results_list[j].get(k, 0) for k in keys]
            if len(set(vals_i)) > 1 and len(set(vals_j)) > 1:
                rho, _ = stats.spearmanr(vals_i, vals_j)
                if not np.isnan(rho):
                    correlations.append(rho)

    return np.mean(correlations) if correlations else np.nan


# =============================================================================
# Ablation 1: Ensemble Size (K)
# =============================================================================

def ablation_ensemble_size(X, y, col_to_fk):
    """Test sensitivity to ensemble size K."""
    print("\n" + "=" * 60)
    print("Ablation 1: Ensemble Size (K)")
    print("=" * 60)

    K_values = [3, 5, 7, 10, 15]
    results = {}

    for K in K_values:
        print(f"\n  K={K}...")
        stability_results = []

        for seed in range(3):
            models = train_ensemble(X, y, n_models=K, base_seed=100*seed)
            attr = fk_attribution(models, X, col_to_fk, n_perm=3)
            stability_results.append(attr)

        stability = compute_stability(stability_results)
        avg_attr = {fk: np.mean([r[fk] for r in stability_results])
                    for fk in stability_results[0].keys()}
        top_fk = max(avg_attr.items(), key=lambda x: x[1])

        results[K] = {
            'stability': stability,
            'top_fk': top_fk[0],
            'top_pct': top_fk[1]
        }
        print(f"    Stability: {stability:.3f}, Top: {top_fk[0]} ({top_fk[1]:.1f}%)")

    return results


# =============================================================================
# Ablation 2: Permutation Runs (P)
# =============================================================================

def ablation_permutation_runs(X, y, col_to_fk):
    """Test sensitivity to permutation runs P."""
    print("\n" + "=" * 60)
    print("Ablation 2: Permutation Runs (P)")
    print("=" * 60)

    P_values = [1, 3, 5, 10, 20]
    results = {}

    # Train ensemble once
    models = train_ensemble(X, y, n_models=5, base_seed=42)

    for P in P_values:
        print(f"\n  P={P}...")
        start = time.time()

        stability_results = []
        for seed in range(3):
            np.random.seed(200 + seed)
            attr = fk_attribution(models, X, col_to_fk, n_perm=P)
            stability_results.append(attr)

        elapsed = time.time() - start
        stability = compute_stability(stability_results)
        avg_attr = {fk: np.mean([r[fk] for r in stability_results])
                    for fk in stability_results[0].keys()}
        top_fk = max(avg_attr.items(), key=lambda x: x[1])

        results[P] = {
            'stability': stability,
            'top_fk': top_fk[0],
            'top_pct': top_fk[1],
            'time_sec': elapsed
        }
        print(f"    Stability: {stability:.3f}, Time: {elapsed:.1f}s")

    return results


# =============================================================================
# Ablation 3: Sample Size (n)
# =============================================================================

def ablation_sample_size(col_to_fk_fn):
    """Test sensitivity to sample size n."""
    print("\n" + "=" * 60)
    print("Ablation 3: Sample Size (n)")
    print("=" * 60)

    n_values = [500, 1000, 2000, 3000, 5000]
    results = {}

    for n in n_values:
        print(f"\n  n={n}...")

        # Load data with specific sample size
        X, y, _, col_to_fk = load_f1_data(sample_size=n, use_cache=True)

        stability_results = []
        for seed in range(3):
            models = train_ensemble(X, y, n_models=5, base_seed=100*seed)
            attr = fk_attribution(models, X, col_to_fk, n_perm=3)
            stability_results.append(attr)

        stability = compute_stability(stability_results)
        avg_attr = {fk: np.mean([r[fk] for r in stability_results])
                    for fk in stability_results[0].keys()}
        top_fk = max(avg_attr.items(), key=lambda x: x[1])

        results[n] = {
            'stability': stability,
            'top_fk': top_fk[0],
            'top_pct': top_fk[1],
            'attribution': avg_attr
        }
        print(f"    Stability: {stability:.3f}, Top: {top_fk[0]} ({top_fk[1]:.1f}%)")

    return results


# =============================================================================
# Ablation 4: Subsampling Rate
# =============================================================================

def ablation_subsampling(X, y, col_to_fk):
    """Test sensitivity to subsampling rate."""
    print("\n" + "=" * 60)
    print("Ablation 4: Subsampling Rate")
    print("=" * 60)

    rates = [0.5, 0.7, 0.8, 0.9, 1.0]
    results = {}

    for rate in rates:
        print(f"\n  subsample={rate}...")

        stability_results = []
        base_uncertainties = []

        for seed in range(3):
            models = train_ensemble(X, y, n_models=5, base_seed=100*seed,
                                    subsample=rate, colsample=rate)
            base_unc = get_uncertainty(models, X)
            base_uncertainties.append(base_unc)

            attr = fk_attribution(models, X, col_to_fk, n_perm=3)
            stability_results.append(attr)

        stability = compute_stability(stability_results)
        avg_base_unc = np.mean(base_uncertainties)
        avg_attr = {fk: np.mean([r[fk] for r in stability_results])
                    for fk in stability_results[0].keys()}
        top_fk = max(avg_attr.items(), key=lambda x: x[1])

        results[rate] = {
            'stability': stability,
            'base_uncertainty': avg_base_unc,
            'top_fk': top_fk[0],
            'top_pct': top_fk[1]
        }
        print(f"    Stability: {stability:.3f}, Base UQ: {avg_base_unc:.4f}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("RelUQ Ablation Studies")
    print("=" * 70)

    # Load base data
    print("\nLoading data (n=3000)...")
    X, y, feature_cols, col_to_fk = load_f1_data(sample_size=3000, use_cache=True)
    print(f"Data shape: {X.shape}")

    all_results = {}

    # Run ablations
    all_results['K'] = ablation_ensemble_size(X, y, col_to_fk)
    all_results['P'] = ablation_permutation_runs(X, y, col_to_fk)
    all_results['n'] = ablation_sample_size(col_to_fk)
    all_results['subsample'] = ablation_subsampling(X, y, col_to_fk)

    # Summary
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)

    print("\n1. Ensemble Size (K):")
    print(f"   {'K':<6} {'Stability':<12} {'Top FK':<15}")
    print("   " + "-" * 35)
    for K, data in all_results['K'].items():
        print(f"   {K:<6} {data['stability']:<12.3f} {data['top_fk']} ({data['top_pct']:.1f}%)")

    print("\n2. Permutation Runs (P):")
    print(f"   {'P':<6} {'Stability':<12} {'Time (s)':<10}")
    print("   " + "-" * 30)
    for P, data in all_results['P'].items():
        print(f"   {P:<6} {data['stability']:<12.3f} {data['time_sec']:<10.1f}")

    print("\n3. Sample Size (n):")
    print(f"   {'n':<8} {'Stability':<12} {'Top FK':<15}")
    print("   " + "-" * 38)
    for n, data in all_results['n'].items():
        print(f"   {n:<8} {data['stability']:<12.3f} {data['top_fk']} ({data['top_pct']:.1f}%)")

    print("\n4. Subsampling Rate:")
    print(f"   {'Rate':<8} {'Stability':<12} {'Base UQ':<12}")
    print("   " + "-" * 35)
    for rate, data in all_results['subsample'].items():
        print(f"   {rate:<8} {data['stability']:<12.3f} {data['base_uncertainty']:<12.4f}")

    # Save results
    # Convert keys to strings for JSON
    json_results = {
        'K': {str(k): v for k, v in all_results['K'].items()},
        'P': {str(k): v for k, v in all_results['P'].items()},
        'n': {str(k): {kk: vv for kk, vv in v.items() if kk != 'attribution'}
              for k, v in all_results['n'].items()},
        'subsample': {str(k): v for k, v in all_results['subsample'].items()}
    }

    with open('/Users/i767700/Github/ai-in-finance/experiments/ablation/results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print("\nResults saved to experiments/ablation/results.json")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = main()
