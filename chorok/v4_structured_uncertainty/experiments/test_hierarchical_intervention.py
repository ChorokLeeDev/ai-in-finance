"""
Test Hierarchical Bayesian Intervention Analysis on SALT Data
==============================================================

This is the key experiment:
- Load real SALT data with FK structure
- Train ensemble model
- Run hierarchical intervention analysis
- Validate that CIs are meaningful

Author: ChorokLeeDev
Created: 2025-12-09
"""

import sys
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'methods'))

from methods.ensemble_lgbm import train_lgbm_ensemble
from methods.hierarchical_intervention import HierarchicalInterventionAnalyzer

CACHE_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache')
RESULTS_DIR = Path(__file__).parent.parent / 'results'


def load_salt_data():
    """Load SALT data from v3 cache."""
    cache_file = CACHE_DIR / 'data_salt_PLANT_2000.pkl'
    if not cache_file.exists():
        print(f"ERROR: Cache not found: {cache_file}")
        return None, None, None, None

    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    if len(data) == 4:
        X, y, feature_cols, col_to_fk = data
        return X, y, feature_cols, col_to_fk

    return None, None, None, None


def run_salt_experiment():
    """Run hierarchical intervention analysis on SALT data."""
    print("="*70)
    print("HIERARCHICAL BAYESIAN INTERVENTION ANALYSIS")
    print("Domain: SALT (Manufacturing ERP)")
    print("="*70)

    # Load data
    print("\n[1/4] Loading SALT data...")
    X, y, feature_cols, col_to_fk = load_salt_data()

    if X is None:
        print("ERROR: Could not load SALT data")
        return None

    # Convert to DataFrame
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_cols)
    y = np.array(y).flatten()

    print(f"  Data shape: {X.shape}")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")

    # FK structure
    from collections import defaultdict
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)
    print(f"  FK groups: {dict((k, len(v)) for k, v in fk_to_cols.items())}")

    # Subsample for faster testing
    n_samples = min(1500, len(X))
    idx = np.random.RandomState(42).permutation(len(X))[:n_samples]
    X_sub = X.iloc[idx].reset_index(drop=True)
    y_sub = y[idx]
    print(f"  Using {n_samples} samples")

    # Train ensemble
    print("\n[2/4] Training LightGBM Ensemble...")
    ensemble = train_lgbm_ensemble(X_sub.values, y_sub, n_models=10, seed=42)

    # Create uncertainty function
    def get_unc(X_in):
        if hasattr(X_in, 'values'):
            X_in = X_in.values
        return ensemble.get_uncertainty(X_in).mean()

    # Verify uncertainty function works
    base_unc = get_unc(X_sub)
    print(f"  Baseline uncertainty: {base_unc:.6f}")

    # Run hierarchical analysis
    print("\n[3/4] Running Hierarchical Intervention Analysis...")
    analyzer = HierarchicalInterventionAnalyzer(get_unc)

    results = analyzer.run_full_analysis(
        X_sub,
        col_to_fk,
        intervention_type='permute',  # Permute to measure importance
        n_bootstrap=100
    )

    # Print report
    analyzer.print_report(results, "SALT MANUFACTURING ERP - INTERVENTION ANALYSIS")

    # Additional: Compare intervention types
    print("\n[4/4] Comparing Intervention Types...")
    print("="*70)

    intervention_types = ['impute_mean', 'reduce_variance', 'remove_outliers']

    if results['level1_fk']:
        top_fk = results['level1_fk'][0].node
        fk_cols = [c for c in fk_to_cols[top_fk] if c in X_sub.columns]

        print(f"\nIntervention comparison for {top_fk}:")
        print(f"{'Intervention Type':<20} │ {'Effect':>10} │ {'95% CI':>20}")
        print(f"{'─'*55}")

        for int_type in intervention_types:
            effect = analyzer.estimate_effect_bootstrap(
                X_sub, fk_cols, int_type, n_bootstrap=50
            )
            print(f"{int_type:<20} │ {effect.effect_mean:>9.1%} │ "
                  f"[{effect.ci_lower:>7.1%}, {effect.ci_upper:>7.1%}]")

    # Save results
    print("\n" + "="*70)
    print("Saving results...")

    def convert(obj):
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if hasattr(obj, '__dict__'): return {k: convert(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(v) for v in obj]
        return str(obj) if not isinstance(obj, (int, float, str, type(None))) else obj

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / 'hierarchical_intervention_salt.json'

    output = {
        'domain': 'SALT',
        'timestamp': datetime.now().isoformat(),
        'n_samples': n_samples,
        'baseline_uncertainty': float(base_unc),
        'level1_fk': convert(results['level1_fk']),
        'level2_column': convert(results['level2_column']),
        'level3_value_range': convert(results['level3_value_range']),
        'summary': convert(results['summary'])
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  Saved to: {output_path}")

    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if results['level1_fk']:
        top_effect = results['level1_fk'][0]
        if top_effect.ci_upper < 0:
            print(f"\n  ✓ HIGH CONFIDENCE: Intervening on {top_effect.node} will reduce uncertainty")
            print(f"    Expected: {top_effect.effect_mean:.1%}, Worst case: {top_effect.ci_upper:.1%}")
        elif top_effect.effect_mean < 0:
            print(f"\n  ? MODERATE CONFIDENCE: {top_effect.node} intervention likely helps")
            print(f"    Expected: {top_effect.effect_mean:.1%}, but CI includes positive values")
        else:
            print(f"\n  ✗ LOW CONFIDENCE: No clear intervention target found")

    return results


if __name__ == "__main__":
    results = run_salt_experiment()
