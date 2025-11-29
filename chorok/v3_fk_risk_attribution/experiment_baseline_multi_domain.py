"""
Multi-Domain Baseline Comparison: FK vs Correlation vs Random
=============================================================

Run FK vs data-driven grouping comparison across all 3 domains:
- SALT (ERP)
- Amazon (E-commerce)
- Stack (Q&A)

This validates that the trade-off (Correlation more stable, FK more interpretable)
holds across different domains.

=== EXPECTED RESULTS ===

| Domain | FK Stability | Corr Stability | Random |
|--------|--------------|----------------|--------|
| SALT   | ~0.34        | ~0.49          | ~0.10  |
| Amazon | TBD          | TBD            | TBD    |
| Stack  | TBD          | TBD            | TBD    |

=== KEY INSIGHT ===

Correlation grouping consistently more stable, but FK enables:
1. Interpretable group names (CUSTOMER, PRODUCT vs CORR_1, CORR_2)
2. Entity-level drill-down (which customer? which product?)
3. Actionable recommendations
"""

import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from data_loader_salt import load_salt_data
from data_loader_amazon import load_amazon_data
from data_loader_stack import load_stack_data

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/results'


def train_ensemble(X, y, n_models=5, base_seed=42):
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
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0)


def get_fk_grouping(col_to_fk):
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)
    return dict(fk_to_cols)


def get_correlation_grouping(X, n_groups):
    corr_matrix = X.corr().abs().values
    np.fill_diagonal(corr_matrix, 0)
    dist_matrix = 1 - corr_matrix
    dist_matrix = np.nan_to_num(dist_matrix, nan=1.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    np.fill_diagonal(dist_matrix, 0)

    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method='average')
    labels = fcluster(Z, n_groups, criterion='maxclust')

    groups = defaultdict(list)
    for col, label in zip(X.columns, labels):
        groups[f"CORR_{label}"].append(col)
    return dict(groups)


def get_random_grouping(columns, n_groups, seed):
    np.random.seed(seed)
    shuffled = np.random.permutation(columns)
    groups = defaultdict(list)
    for i, col in enumerate(shuffled):
        groups[f"RAND_{i % n_groups}"].append(col)
    return dict(groups)


def compute_attribution(models, X, grouping, n_permute=5):
    base_unc = get_uncertainty(models, X).mean()
    results = {}

    for group, cols in grouping.items():
        col_indices = [list(X.columns).index(c) for c in cols if c in X.columns]
        if not col_indices:
            continue

        deltas = []
        for _ in range(n_permute):
            X_perm = X.copy()
            for idx in col_indices:
                col_name = X.columns[idx]
                X_perm[col_name] = np.random.permutation(X_perm[col_name].values)
            perm_unc = get_uncertainty(models, X_perm).mean()
            deltas.append(perm_unc - base_unc)

        results[group] = np.mean(deltas)

    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def compute_stability(rankings):
    if len(rankings) < 2:
        return 1.0

    correlations = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            common = set(rankings[i]) & set(rankings[j])
            if len(common) < 3:
                continue
            rank1 = [rankings[i].index(item) for item in common]
            rank2 = [rankings[j].index(item) for item in common]
            corr, _ = spearmanr(rank1, rank2)
            if not np.isnan(corr):
                correlations.append(corr)

    return np.mean(correlations) if correlations else 0.0


def run_baseline_comparison(X, y, col_to_fk, domain_name, n_runs=5):
    """Run FK vs Correlation vs Random comparison for one domain."""
    print(f"\n{'='*60}")
    print(f"BASELINE COMPARISON: {domain_name}")
    print(f"{'='*60}")

    n_fk_groups = len(set(col_to_fk.values()))
    print(f"Features: {len(X.columns)}, FK groups: {n_fk_groups}")

    fk_grouping = get_fk_grouping(col_to_fk)
    corr_grouping = get_correlation_grouping(X, n_fk_groups)

    print(f"FK groups: {list(fk_grouping.keys())}")
    print(f"Corr groups: {list(corr_grouping.keys())}")

    fk_rankings = []
    corr_rankings = []
    rand_rankings = []

    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...", end=" ", flush=True)

        models = train_ensemble(X, y, n_models=5, base_seed=42 + run * 10)

        # FK attribution
        fk_attr = compute_attribution(models, X, fk_grouping)
        fk_ranking = sorted(fk_attr.keys(), key=lambda k: -fk_attr[k])
        fk_rankings.append(fk_ranking)

        # Correlation attribution
        corr_attr = compute_attribution(models, X, corr_grouping)
        corr_ranking = sorted(corr_attr.keys(), key=lambda k: -corr_attr[k])
        corr_rankings.append(corr_ranking)

        # Random attribution
        rand_grouping = get_random_grouping(list(X.columns), n_fk_groups, seed=42 + run)
        rand_attr = compute_attribution(models, X, rand_grouping)
        rand_ranking = sorted(rand_attr.keys(), key=lambda k: -rand_attr[k])
        rand_rankings.append(rand_ranking)

        print("done")

    fk_stability = compute_stability(fk_rankings)
    corr_stability = compute_stability(corr_rankings)
    rand_stability = compute_stability(rand_rankings)

    print(f"\n  Results:")
    print(f"    FK Stability:   {fk_stability:.3f}")
    print(f"    Corr Stability: {corr_stability:.3f}")
    print(f"    Rand Stability: {rand_stability:.3f}")

    return {
        'domain': domain_name,
        'fk_stability': fk_stability,
        'corr_stability': corr_stability,
        'rand_stability': rand_stability,
        'n_features': len(X.columns),
        'n_fk_groups': n_fk_groups
    }


def run_all_baselines():
    print("="*70)
    print("MULTI-DOMAIN BASELINE COMPARISON")
    print("FK vs Correlation vs Random Grouping")
    print("="*70)

    all_results = {}

    # SALT
    print("\n[1/3] Loading SALT data...")
    X_salt, y_salt, _, col_to_fk_salt = load_salt_data(sample_size=3000)
    salt_results = run_baseline_comparison(X_salt, y_salt, col_to_fk_salt, "SALT (ERP)")
    all_results['salt'] = salt_results

    # Amazon
    print("\n[2/3] Loading Amazon data...")
    X_amazon, y_amazon, _, col_to_fk_amazon = load_amazon_data(sample_size=3000)
    amazon_results = run_baseline_comparison(X_amazon, y_amazon, col_to_fk_amazon, "Amazon (E-commerce)")
    all_results['amazon'] = amazon_results

    # Stack
    print("\n[3/3] Loading Stack data...")
    X_stack, y_stack, _, col_to_fk_stack = load_stack_data(sample_size=3000)
    stack_results = run_baseline_comparison(X_stack, y_stack, col_to_fk_stack, "Stack (Q&A)")
    all_results['stack'] = stack_results

    # Summary
    print("\n" + "="*70)
    print("MULTI-DOMAIN BASELINE SUMMARY")
    print("="*70)
    print(f"\n{'Domain':<20} {'FK':<12} {'Correlation':<12} {'Random':<12}")
    print("-"*56)
    for domain, results in all_results.items():
        print(f"{results['domain']:<20} {results['fk_stability']:<12.3f} "
              f"{results['corr_stability']:<12.3f} {results['rand_stability']:<12.3f}")

    # Trade-off analysis
    print("\n" + "="*70)
    print("TRADE-OFF ANALYSIS")
    print("="*70)

    for domain, results in all_results.items():
        diff = results['corr_stability'] - results['fk_stability']
        pct = (diff / results['fk_stability'] * 100) if results['fk_stability'] > 0 else 0
        print(f"\n{results['domain']}:")
        print(f"  Correlation is {pct:.1f}% more stable than FK")
        print(f"  BUT: FK provides interpretable business entities")

    # Save results
    output_path = f"{RESULTS_DIR}/baseline_multi_domain.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] {output_path}")

    return all_results


if __name__ == "__main__":
    results = run_all_baselines()
