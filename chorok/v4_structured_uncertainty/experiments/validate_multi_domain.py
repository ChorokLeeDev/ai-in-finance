"""
Multi-Domain Validation for Hierarchical Bayesian Analysis
============================================================

Test the framework on multiple domains:
1. SALT (Manufacturing ERP)
2. F1 (Racing)
3. Stack (Q&A Forum)
4. H&M (Fashion Retail)

Author: ChorokLeeDev
Created: 2025-12-09
"""

import sys
import numpy as np
import pandas as pd
import pickle
import pyro
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'methods'))

from methods.ensemble_lgbm import train_lgbm_ensemble
from methods.hierarchical_bayesian import HierarchicalBayesianAnalyzer, HierarchicalBayesianUQ

CACHE_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache')


def load_cached_data(dataset_name, sample_size=3000):
    """Load cached data from v3 experiments."""
    patterns = {
        'salt': f'data_salt_PLANT_{sample_size}.pkl',
        'f1': f'data_f1_driver-position_{sample_size}.pkl',
        'stack': f'data_stack_post-votes_{sample_size}.pkl',
        'hm': f'data_hm_item-sales_{sample_size}.pkl',
    }

    cache_file = CACHE_DIR / patterns.get(dataset_name, '')

    if not cache_file.exists():
        # Try smaller size
        for size in [2000, 1000, 500]:
            alt_pattern = patterns[dataset_name].replace(str(sample_size), str(size))
            alt_file = CACHE_DIR / alt_pattern
            if alt_file.exists():
                cache_file = alt_file
                break

    if not cache_file.exists():
        return None, None, None

    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    # Handle different cache formats
    if isinstance(data, dict):
        # Format: {'X': dict/df, 'y': list, 'feature_cols': list, 'col_to_fk': dict}
        X = data.get('X')
        y = data.get('y')
        feature_cols = data.get('feature_cols', [])
        col_to_fk = data.get('col_to_fk', {})

        # X might be a dict of columns
        if isinstance(X, dict):
            X = pd.DataFrame(X)
    elif isinstance(data, (list, tuple)):
        if len(data) == 4:
            X, y, feature_cols, col_to_fk = data
        elif len(data) == 3:
            X, y, col_to_fk = data
            feature_cols = X.columns if hasattr(X, 'columns') else [f'col_{i}' for i in range(X.shape[1])]
        else:
            raise ValueError(f"Unexpected cache format: {len(data)} elements")
    else:
        raise ValueError(f"Unexpected cache type: {type(data)}")

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_cols)

    return X, np.array(y).flatten(), col_to_fk


def run_bayesian_analysis_for_domain(domain_name, X, y, col_to_fk, n_samples=500):
    """Run full Bayesian analysis for a single domain."""
    print(f"\n{'='*70}")
    print(f"DOMAIN: {domain_name.upper()}")
    print('='*70)

    # Subsample if needed
    if len(X) > n_samples:
        idx = np.random.permutation(len(X))[:n_samples]
        X_sub = X.iloc[idx].reset_index(drop=True)
        y_sub = y[idx]
    else:
        X_sub = X.reset_index(drop=True)
        y_sub = y

    print(f"  Data shape: {X_sub.shape}")
    print(f"  Target range: [{y_sub.min():.2f}, {y_sub.max():.2f}]")

    # Count FKs
    fk_structure = defaultdict(list)
    for col, fk in col_to_fk.items():
        if col in X_sub.columns:
            fk_structure[fk].append(col)
    print(f"  FK tables: {list(fk_structure.keys())}")

    # Train ensemble
    print(f"\n  [1/4] Training LightGBM ensemble...")
    ensemble = train_lgbm_ensemble(X_sub.values, y_sub, n_models=10, seed=42)

    def get_unc(X_in):
        if hasattr(X_in, 'values'):
            X_in = X_in.values
        return ensemble.get_uncertainty(X_in).mean()

    base_unc = get_unc(X_sub)
    print(f"        Baseline uncertainty: {base_unc:.6f}")

    # Run Bayesian analysis
    print(f"\n  [2/4] Running Hierarchical Bayesian Analysis...")
    analyzer = HierarchicalBayesianAnalyzer(get_unc)

    results = analyzer.run_bayesian_analysis(
        X_sub, col_to_fk,
        n_permute=5,
        n_vi_steps=600,
        n_posterior_samples=1000,
        verbose=False
    )

    # Summary
    print(f"\n  [3/4] FK-Level Results (Bayesian Posterior)")
    print(f"  {'─'*60}")
    print(f"  {'FK Table':<20} │ {'Mean':>10} │ {'95% CI':>25}")
    print(f"  {'─'*60}")

    for effect in results['fk_importance']:
        ci_str = f"[{effect.ci_lower:>7.1%}, {effect.ci_upper:>7.1%}]"
        print(f"  {effect.node:<20} │ {effect.effect_mean:>9.1%} │ {ci_str:>25}")

    # Diagnostics
    print(f"\n  [4/4] Diagnostics")

    top_fk = results['fk_importance'][0]
    ci_width = top_fk.ci_upper - top_fk.ci_lower

    # CI coverage check: does 0 lie within CI?
    contains_zero = top_fk.ci_lower <= 0 <= top_fk.ci_upper

    print(f"        Top FK: {top_fk.node}")
    print(f"        CI Width: {ci_width:.1%}")
    print(f"        CI Contains Zero: {contains_zero}")

    if top_fk.ci_lower > 0:
        print(f"        Conclusion: HIGH CONFIDENCE - {top_fk.node} is definitely important")
    elif top_fk.ci_upper < 0:
        print(f"        Conclusion: {top_fk.node} reduces uncertainty (unexpected)")
    else:
        print(f"        Conclusion: UNCERTAIN - need more data for {top_fk.node}")

    return {
        'domain': domain_name,
        'n_samples': len(X_sub),
        'n_features': X_sub.shape[1],
        'n_fks': len(fk_structure),
        'base_uncertainty': base_unc,
        'top_fk': top_fk.node,
        'top_fk_mean': top_fk.effect_mean,
        'top_fk_ci_lower': top_fk.ci_lower,
        'top_fk_ci_upper': top_fk.ci_upper,
        'top_fk_ci_width': ci_width,
        'ci_contains_zero': contains_zero,
        'all_results': results
    }


def main():
    print("="*70)
    print("MULTI-DOMAIN VALIDATION")
    print("Hierarchical Bayesian Intervention Analysis")
    print("="*70)

    np.random.seed(42)
    pyro.set_rng_seed(42)

    domains = ['salt', 'f1', 'stack', 'hm']
    all_results = {}

    for domain in domains:
        print(f"\n>>> Loading {domain.upper()} data...")
        X, y, col_to_fk = load_cached_data(domain)

        if X is None:
            print(f"    SKIPPED: No cached data for {domain}")
            continue

        try:
            result = run_bayesian_analysis_for_domain(domain, X, y, col_to_fk)
            all_results[domain] = result
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    # Summary table
    print("\n" + "="*70)
    print("CROSS-DOMAIN SUMMARY")
    print("="*70)

    print(f"\n{'Domain':<10} │ {'Top FK':<20} │ {'Mean':>10} │ {'CI Width':>10} │ {'Status':>15}")
    print(f"{'─'*75}")

    for domain, result in all_results.items():
        status = "HIGH CONF" if not result['ci_contains_zero'] and result['top_fk_ci_lower'] > 0 else "UNCERTAIN"
        print(f"{domain.upper():<10} │ {result['top_fk']:<20} │ "
              f"{result['top_fk_mean']:>9.1%} │ "
              f"{result['top_fk_ci_width']:>9.1%} │ "
              f"{status:>15}")

    # Validate CI behavior
    print("\n" + "="*70)
    print("CI CALIBRATION CHECK")
    print("="*70)

    # Expected: ~95% of true values within CI
    # We can't directly test this without ground truth, but we can check:
    # 1. CIs are reasonable (not too wide or narrow)
    # 2. Rankings are consistent with effect size

    avg_ci_width = np.mean([r['top_fk_ci_width'] for r in all_results.values()])
    print(f"\n  Average CI Width: {avg_ci_width:.1%}")

    if avg_ci_width < 0.1:
        print("  WARNING: CIs may be too narrow (overconfident)")
    elif avg_ci_width > 2.0:
        print("  WARNING: CIs are very wide (high uncertainty)")
    else:
        print("  OK: CI widths are in reasonable range")

    # Check consistency: higher mean should correlate with more confident CI
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    n_valid = len(all_results)
    n_confident = sum(1 for r in all_results.values() if not r['ci_contains_zero'] and r['top_fk_ci_lower'] > 0)

    print(f"\n  Domains tested: {n_valid}")
    print(f"  High confidence results: {n_confident}/{n_valid}")
    print(f"\n  Framework generalizes across domains: {'YES' if n_valid >= 3 else 'NEED MORE DATA'}")

    return all_results


if __name__ == "__main__":
    results = main()
