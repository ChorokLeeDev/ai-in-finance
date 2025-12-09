"""
Test Hierarchical Bayesian Model on SALT Data
==============================================

Compare:
1. Bootstrap CI (current implementation)
2. Bayesian Credible Intervals (new Pyro implementation)

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
from methods.hierarchical_bayesian import HierarchicalBayesianAnalyzer

CACHE_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache')


def load_salt_data():
    cache_file = CACHE_DIR / 'data_salt_PLANT_2000.pkl'
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    X, y, feature_cols, col_to_fk = data
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_cols)
    return X, np.array(y).flatten(), col_to_fk


def main():
    print("="*70)
    print("HIERARCHICAL BAYESIAN ANALYSIS - SALT Data")
    print("True Credible Intervals via Variational Inference")
    print("="*70)

    np.random.seed(42)
    pyro.set_rng_seed(42)

    # Load data
    print("\n[1/4] Loading SALT data...")
    X, y, col_to_fk = load_salt_data()

    n_samples = min(1000, len(X))
    idx = np.random.permutation(len(X))[:n_samples]
    X_sub = X.iloc[idx].reset_index(drop=True)
    y_sub = y[idx]

    print(f"  Data: {X_sub.shape}")

    # Train ensemble
    print("\n[2/4] Training LightGBM ensemble...")
    ensemble = train_lgbm_ensemble(X_sub.values, y_sub, n_models=10, seed=42)

    def get_unc(X_in):
        if hasattr(X_in, 'values'):
            X_in = X_in.values
        return ensemble.get_uncertainty(X_in).mean()

    base_unc = get_unc(X_sub)
    print(f"  Baseline uncertainty: {base_unc:.6f}")

    # Run Bayesian analysis
    print("\n[3/4] Running Hierarchical Bayesian Analysis...")
    analyzer = HierarchicalBayesianAnalyzer(get_unc)

    results = analyzer.run_bayesian_analysis(
        X_sub, col_to_fk,
        n_permute=5,           # Permutations for importance
        n_vi_steps=800,        # VI optimization steps
        n_posterior_samples=1000,  # Posterior samples
        verbose=True
    )

    # Print report
    print("\n[4/4] Results")
    analyzer.print_report(results, "SALT - BAYESIAN CREDIBLE INTERVALS")

    # Compare with raw permutation scores
    print("\n" + "="*70)
    print("COMPARISON: Raw Permutation vs Bayesian Posterior")
    print("="*70)

    print(f"\n{'Column':<25} │ {'Raw Score':>12} │ {'Bayes Mean':>12} │ {'Bayes 95% CI':>20}")
    print(f"{'─'*75}")

    raw_scores = results['importance_scores']
    col_importance = {e.node: e for e in results['column_importance']}

    for col in sorted(raw_scores.keys(), key=lambda x: -raw_scores[x]):
        raw = raw_scores[col]
        if col in col_importance:
            bayes = col_importance[col]
            print(f"{col:<25} │ {raw:>11.1%} │ {bayes.effect_mean:>11.1%} │ "
                  f"[{bayes.ci_lower:>7.1%}, {bayes.ci_upper:>7.1%}]")

    # Save results
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    top_fk = results['fk_importance'][0]
    print(f"\n  Top FK: {top_fk.node}")
    print(f"  Bayesian estimate: {top_fk.effect_mean:.1%}")
    print(f"  95% Credible Interval: [{top_fk.ci_lower:.1%}, {top_fk.ci_upper:.1%}]")

    ci_width = top_fk.ci_upper - top_fk.ci_lower
    print(f"\n  CI Width: {ci_width:.1%}")

    if ci_width < 0.5:  # Less than 50% width
        print("  → NARROW CI: High confidence in importance ranking")
    else:
        print("  → WIDE CI: More data or better model needed")

    return results


if __name__ == "__main__":
    results = main()
