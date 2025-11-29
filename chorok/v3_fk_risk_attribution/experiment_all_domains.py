"""
Multi-Domain FK Attribution Experiments for NeurIPS 2026
=========================================================

Runs complete FK attribution + entity optimization + verification
on all 3 domains: SALT (ERP), Amazon (E-commerce), Stack (Q&A)
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import warnings
import json
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Domain-specific loaders
from data_loader_salt import load_salt_data
from data_loader_amazon import load_amazon_data
from data_loader_stack import load_stack_data

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_ensemble(X, y, n_models=10, base_seed=42):
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
    if len(X) == 0:
        return np.array([])
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0)


def fk_attribution(models, X, col_to_fk, n_permute=10):
    """Compute FK-level uncertainty attribution."""
    base_unc = get_uncertainty(models, X).mean()
    fk_groups = list(set(col_to_fk.values()))
    attribution = {}

    for fk in fk_groups:
        fk_cols = [c for c, f in col_to_fk.items() if f == fk]
        unc_changes = []

        for _ in range(n_permute):
            X_perm = X.copy()
            for col in fk_cols:
                X_perm[col] = np.random.permutation(X_perm[col].values)
            perm_unc = get_uncertainty(models, X_perm).mean()
            unc_changes.append(perm_unc - base_unc)

        attribution[fk] = {
            'mean_increase': float(np.mean(unc_changes)),
            'std': float(np.std(unc_changes)),
            'features': fk_cols
        }

    return attribution, base_unc


def entity_analysis(models, X, y, entity_col, min_samples=3):
    """Analyze entity-level uncertainty."""
    if entity_col not in X.columns:
        return None

    uncertainties = get_uncertainty(models, X)
    data = pd.DataFrame({
        'entity': X[entity_col].values,
        'uncertainty': uncertainties,
        'target': y.values
    })

    entity_stats = data.groupby('entity').agg({
        'uncertainty': ['mean', 'std', 'count'],
        'target': 'mean'
    }).reset_index()
    entity_stats.columns = ['entity', 'mean_unc', 'std_unc', 'count', 'avg_target']
    entity_stats = entity_stats[entity_stats['count'] >= min_samples]

    if len(entity_stats) == 0:
        return None

    high_unc = entity_stats.nlargest(5, 'mean_unc')
    low_unc = entity_stats.nsmallest(5, 'mean_unc')

    return {
        'high_uncertainty': high_unc.to_dict('records'),
        'low_uncertainty': low_unc.to_dict('records'),
        'total_entities': len(entity_stats),
        'uncertainty_range': (float(entity_stats['mean_unc'].min()),
                              float(entity_stats['mean_unc'].max()))
    }


def verification_tests(X, y, col_to_fk, entity_col, n_seeds=3):
    """Run 4 verification tests."""
    results = {}

    # Test 1: Held-out consistency
    correlations = []
    for seed in range(n_seeds):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42 + seed
        )
        models = train_ensemble(X_train, y_train, n_models=5, base_seed=seed*10)

        train_unc = get_uncertainty(models, X_train)
        test_unc = get_uncertainty(models, X_test)

        train_data = pd.DataFrame({'entity': X_train[entity_col].values, 'unc': train_unc})
        test_data = pd.DataFrame({'entity': X_test[entity_col].values, 'unc': test_unc})

        train_stats = train_data.groupby('entity')['unc'].mean()
        test_stats = test_data.groupby('entity')['unc'].mean()

        common = train_stats.index.intersection(test_stats.index)
        if len(common) > 5:
            corr = train_stats[common].corr(test_stats[common])
            correlations.append(corr)

    results['held_out_correlation'] = float(np.mean(correlations)) if correlations else 0.0

    # Test 2: Bootstrap stability
    models = train_ensemble(X, y, n_models=10)
    entity_rankings = defaultdict(list)

    for i in range(5):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X.iloc[indices].reset_index(drop=True)
        y_boot = y.iloc[indices].reset_index(drop=True)

        boot_models = train_ensemble(X_boot, y_boot, n_models=3, base_seed=i*20)
        boot_unc = get_uncertainty(boot_models, X_boot)

        data = pd.DataFrame({'entity': X_boot[entity_col].values, 'unc': boot_unc})
        stats = data.groupby('entity')['unc'].mean().sort_values()

        for rank, (entity, _) in enumerate(stats.items()):
            entity_rankings[entity].append(rank)

    rank_stds = [np.std(ranks) for ranks in entity_rankings.values() if len(ranks) >= 3]
    avg_rank_std = np.mean(rank_stds) if rank_stds else 0
    max_possible = len(entity_rankings) / 2
    results['bootstrap_stability'] = float(1 - avg_rank_std / max_possible) if max_possible > 0 else 0.0

    # Test 3: Uncertainty-error calibration
    uncertainties = get_uncertainty(models, X)
    predictions = np.mean([m.predict(X) for m in models], axis=0)
    errors = np.abs(predictions - y.values)

    data = pd.DataFrame({
        'entity': X[entity_col].values,
        'uncertainty': uncertainties,
        'error': errors
    })
    entity_stats = data.groupby('entity').agg({'uncertainty': 'mean', 'error': 'mean'})
    results['uncertainty_error_correlation'] = float(entity_stats['uncertainty'].corr(entity_stats['error']))

    # Test 4: Counterfactual simulation
    entity_data = pd.DataFrame({'entity': X[entity_col].values, 'unc': uncertainties})
    entity_mean_unc = entity_data.groupby('entity')['unc'].mean()

    if len(entity_mean_unc) >= 2:
        worst = entity_mean_unc.idxmax()
        best = entity_mean_unc.idxmin()

        worst_mask = X[entity_col] == worst
        best_mask = X[entity_col] == best

        if worst_mask.sum() > 0 and best_mask.sum() > 0:
            X_sim = X.copy()
            best_samples = X[best_mask]

            for idx in X[worst_mask].index:
                replacement_idx = np.random.choice(best_samples.index)
                X_sim.loc[idx] = X.loc[replacement_idx]

            baseline_unc = uncertainties.mean()
            sim_unc = get_uncertainty(models, X_sim).mean()
            results['simulation_reduction'] = float((baseline_unc - sim_unc) / baseline_unc * 100)
        else:
            results['simulation_reduction'] = 0.0
    else:
        results['simulation_reduction'] = 0.0

    return results


def run_domain_experiment(domain_name, load_func, sample_size=10000, **load_kwargs):
    """Run complete experiment for one domain."""
    print(f"\n{'='*80}")
    print(f"DOMAIN: {domain_name}")
    print(f"{'='*80}")

    # Load data
    print(f"\n[1/5] Loading data...")
    try:
        X, y, feature_cols, col_to_fk = load_func(sample_size=sample_size, **load_kwargs)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None

    print(f"  Shape: X={X.shape}, y={len(y)}")
    print(f"  FK groups: {set(col_to_fk.values())}")

    # Train ensemble
    print(f"\n[2/5] Training ensemble...")
    models = train_ensemble(X, y, n_models=10)

    # FK attribution
    print(f"\n[3/5] FK attribution...")
    attribution, base_unc = fk_attribution(models, X, col_to_fk)

    print(f"\n  Base uncertainty: {base_unc:.6f}")
    total_increase = sum(a['mean_increase'] for a in attribution.values() if a['mean_increase'] > 0)

    print(f"\n  FK Attribution:")
    sorted_attr = sorted(attribution.items(), key=lambda x: x[1]['mean_increase'], reverse=True)
    for fk, attr in sorted_attr:
        pct = attr['mean_increase'] / total_increase * 100 if total_increase > 0 else 0
        print(f"    {fk}: {pct:.1f}%")

    # Entity analysis
    print(f"\n[4/5] Entity analysis...")
    entity_results = {}

    # Find entity columns
    entity_cols = [c for c in X.columns if c in col_to_fk and
                   X[c].nunique() > 1 and X[c].nunique() < len(X) / 2]

    for entity_col in entity_cols[:3]:  # Top 3 entity columns
        analysis = entity_analysis(models, X, y, entity_col)
        if analysis:
            entity_results[entity_col] = analysis
            print(f"\n  {entity_col}:")
            print(f"    Total entities: {analysis['total_entities']}")
            print(f"    Uncertainty range: {analysis['uncertainty_range'][0]:.6f} - {analysis['uncertainty_range'][1]:.6f}")

    # Verification
    print(f"\n[5/5] Verification tests...")
    if entity_cols:
        verification = verification_tests(X, y, col_to_fk, entity_cols[0])
    else:
        verification = {}

    print(f"\n  Results:")
    for test, value in verification.items():
        status = "PASS" if (
            (test == 'held_out_correlation' and value > 0.5) or
            (test == 'bootstrap_stability' and value > 0.5) or
            (test == 'uncertainty_error_correlation' and value > 0.3) or
            (test == 'simulation_reduction' and value > 0)
        ) else "FAIL"
        print(f"    {test}: {value:.3f} [{status}]")

    # Compile results
    results = {
        'domain': domain_name,
        'data_shape': {'n_samples': len(y), 'n_features': len(feature_cols)},
        'fk_groups': list(set(col_to_fk.values())),
        'base_uncertainty': float(base_unc),
        'attribution': {fk: {'contribution': attr['mean_increase'] / total_increase * 100 if total_increase > 0 else 0,
                             'features': attr['features']}
                        for fk, attr in attribution.items()},
        'entity_analysis': entity_results,
        'verification': verification,
        'timestamp': datetime.now().isoformat()
    }

    return results


def run_all_experiments():
    """Run experiments on all domains."""
    print("="*80)
    print("MULTI-DOMAIN FK ATTRIBUTION EXPERIMENTS")
    print("NeurIPS 2026 Paper")
    print("="*80)

    all_results = {}

    # Domain 1: SALT (ERP) - smaller sample
    salt_results = run_domain_experiment(
        domain_name="SALT (SAP ERP)",
        load_func=load_salt_data,
        sample_size=5000,
        target='PLANT'
    )
    if salt_results:
        all_results['salt'] = salt_results

    # Domain 2: Amazon (E-commerce) - smaller sample
    amazon_results = run_domain_experiment(
        domain_name="Amazon (E-commerce)",
        load_func=load_amazon_data,
        sample_size=3000,
        task_name='user-ltv'
    )
    if amazon_results:
        all_results['amazon'] = amazon_results

    # Domain 3: Stack (Q&A) - smaller sample
    stack_results = run_domain_experiment(
        domain_name="Stack Overflow (Q&A)",
        load_func=load_stack_data,
        sample_size=3000,
        task_name='post-votes'
    )
    if stack_results:
        all_results['stack'] = stack_results

    # Save results
    results_file = f"{RESULTS_DIR}/multi_domain_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[SAVED] Results to {results_file}")

    # Summary
    print("\n" + "="*80)
    print("MULTI-DOMAIN SUMMARY")
    print("="*80)

    print("\n### FK Attribution Comparison ###")
    for domain, results in all_results.items():
        print(f"\n{domain.upper()}:")
        for fk, data in sorted(results['attribution'].items(),
                                key=lambda x: x[1]['contribution'], reverse=True):
            print(f"  {fk}: {data['contribution']:.1f}%")

    print("\n### Verification Results ###")
    print(f"{'Domain':<20} {'Held-Out':<12} {'Bootstrap':<12} {'Calibration':<12} {'Simulation':<12}")
    print("-" * 68)
    for domain, results in all_results.items():
        v = results['verification']
        print(f"{domain:<20} {v.get('held_out_correlation', 0):.3f}        "
              f"{v.get('bootstrap_stability', 0):.3f}        "
              f"{v.get('uncertainty_error_correlation', 0):.3f}        "
              f"{v.get('simulation_reduction', 0):.1f}%")

    print("\n" + "="*80)

    return all_results


if __name__ == "__main__":
    results = run_all_experiments()
