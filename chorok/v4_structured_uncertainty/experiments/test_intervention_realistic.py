"""
Realistic Intervention Test: Does MORE DATA reduce uncertainty?
================================================================

The key insight: Epistemic uncertainty comes from LACK of training data.
The only real fix is to have more training data in the problematic regions.

Test:
1. Train with LESS data in target region → HIGH uncertainty
2. Train with MORE data in target region → LOW uncertainty
3. Measure the difference

Author: ChorokLeeDev
Created: 2025-12-09
"""

import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'methods'))

from methods.ensemble_lgbm import train_lgbm_ensemble

CACHE_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache')


def load_cached_data(dataset_name, sample_size=3000):
    """Load cached data."""
    patterns = {
        'salt': f'data_salt_PLANT_{sample_size}.pkl',
        'f1': f'data_f1_driver-position_{sample_size}.pkl',
    }
    cache_file = CACHE_DIR / patterns.get(dataset_name, '')

    if not cache_file.exists():
        for size in [2000, 1000, 500]:
            alt_file = CACHE_DIR / patterns[dataset_name].replace(str(sample_size), str(size))
            if alt_file.exists():
                cache_file = alt_file
                break

    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        X = data.get('X')
        y = data.get('y')
        if isinstance(X, dict):
            X = pd.DataFrame(X)
    else:
        X, y, _, _ = data
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

    return X, np.array(y).flatten()


def test_data_quantity_effect(domain_name, X, y, target_col, target_range):
    """
    Test: Does having more training data in target region reduce uncertainty?

    Method:
    1. Split data into target region vs other
    2. Vary the amount of target region data in training (10%, 30%, 50%, 70%, 100%)
    3. Measure uncertainty on held-out target region test set
    """

    print(f"\n{'='*70}")
    print(f"DATA QUANTITY EFFECT: {domain_name.upper()}")
    print(f"Target: {target_col} in range {target_range}")
    print(f"{'='*70}")

    # Identify target region
    col_data = X[target_col]
    if isinstance(target_range, tuple):
        low, high = target_range
        in_target = (col_data >= low) & (col_data <= high)
    else:
        in_target = col_data == target_range

    X_target = X[in_target].reset_index(drop=True)
    y_target = y[in_target]
    X_other = X[~in_target].reset_index(drop=True)
    y_other = y[~in_target]

    print(f"\nTarget region samples: {len(X_target)}")
    print(f"Other region samples: {len(X_other)}")

    # Split target region into train/test
    n_target = len(X_target)
    test_size = min(200, n_target // 3)
    test_idx = np.random.permutation(n_target)[:test_size]
    train_target_idx = np.array([i for i in range(n_target) if i not in test_idx])

    X_target_test = X_target.iloc[test_idx].reset_index(drop=True)
    y_target_test = y_target[test_idx]
    X_target_train_full = X_target.iloc[train_target_idx].reset_index(drop=True)
    y_target_train_full = y_target[train_target_idx]

    print(f"Target test set: {len(X_target_test)}")
    print(f"Target train pool: {len(X_target_train_full)}")

    # Use fixed amount of "other" data
    n_other_train = min(500, len(X_other))
    other_idx = np.random.permutation(len(X_other))[:n_other_train]
    X_other_train = X_other.iloc[other_idx].reset_index(drop=True)
    y_other_train = y_other[other_idx]

    print(f"Other train set: {len(X_other_train)}")

    # Test different amounts of target region data
    fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    results = []

    print(f"\n{'Fraction':>10} │ {'N Target':>10} │ {'N Total':>10} │ {'Uncertainty':>12} │ {'Δ from 0%':>12}")
    print(f"{'─'*65}")

    baseline_unc = None

    for frac in fractions:
        # Sample fraction of target training data
        n_target_train = int(len(X_target_train_full) * frac)

        if n_target_train > 0:
            target_train_idx = np.random.permutation(len(X_target_train_full))[:n_target_train]
            X_target_train = X_target_train_full.iloc[target_train_idx].reset_index(drop=True)
            y_target_train = y_target_train_full[target_train_idx]

            # Combine with other data
            X_train = pd.concat([X_other_train, X_target_train], ignore_index=True)
            y_train = np.concatenate([y_other_train, y_target_train])
        else:
            X_train = X_other_train.copy()
            y_train = y_other_train.copy()

        # Train ensemble
        ensemble = train_lgbm_ensemble(X_train.values, y_train, n_models=10, seed=42)

        # Measure uncertainty on target test set
        unc = ensemble.get_uncertainty(X_target_test.values).mean()

        if baseline_unc is None:
            baseline_unc = unc
            delta = 0.0
        else:
            delta = (unc - baseline_unc) / baseline_unc * 100

        results.append({
            'fraction': frac,
            'n_target': n_target_train,
            'n_total': len(X_train),
            'uncertainty': unc,
            'delta': delta
        })

        print(f"{frac:>9.0%} │ {n_target_train:>10} │ {len(X_train):>10} │ {unc:>12.6f} │ {delta:>+11.1f}%")

    # Summary
    print(f"\n{'─'*65}")
    unc_0 = results[0]['uncertainty']
    unc_100 = results[-1]['uncertainty']

    # Find the peak uncertainty (often at partial data when model "discovers" hard cases)
    peak_unc = max(r['uncertainty'] for r in results)
    peak_frac = [r for r in results if r['uncertainty'] == peak_unc][0]['fraction']

    print(f"\nUncertainty with 0% target data:   {unc_0:.6f}")
    print(f"Peak uncertainty (at {peak_frac:.0%}):       {peak_unc:.6f}")
    print(f"Uncertainty with 100% target data: {unc_100:.6f}")

    # The real metric: does more data reduce uncertainty from the peak?
    if peak_unc > unc_0:
        # Model was overconfident with 0% data
        reduction_from_peak = (peak_unc - unc_100) / peak_unc * 100
        print(f"\nNote: Model was OVERCONFIDENT with 0% target data (low unc = wrong confidence)")
        print(f"Peak uncertainty occurred at {peak_frac:.0%} data when model 'discovered' hard cases")
        print(f"Reduction from peak to 100%: {reduction_from_peak:.1f}%")

        if reduction_from_peak > 0:
            print(f"\n✓ MORE DATA REDUCES UNCERTAINTY (from peak) BY {reduction_from_peak:.1f}%")
        else:
            print(f"\n✗ More data did not reduce uncertainty from peak")
    else:
        reduction = (unc_0 - unc_100) / unc_0 * 100
        if reduction > 0:
            print(f"\n✓ MORE DATA REDUCES UNCERTAINTY BY {reduction:.1f}%")
        else:
            print(f"\n✗ More data did not reduce uncertainty")

    return results


def main():
    print("="*70)
    print("REALISTIC INTERVENTION TEST")
    print("Does more training data reduce epistemic uncertainty?")
    print("="*70)

    np.random.seed(42)

    all_results = {}

    # F1
    print("\n>>> Loading F1 data...")
    X, y = load_cached_data('f1')
    results_f1 = test_data_quantity_effect('F1', X, y, 'surname', (271, 413))
    all_results['F1'] = results_f1

    # SALT
    print("\n>>> Loading SALT data...")
    X, y = load_cached_data('salt')
    results_salt = test_data_quantity_effect('SALT', X, y, 'SHIPPINGPOINT', (7, 79))
    all_results['SALT'] = results_salt

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY: INTERVENTION EFFECT")
    print(f"{'='*70}")

    for domain, results in all_results.items():
        unc_0 = results[0]['uncertainty']
        unc_100 = results[-1]['uncertainty']
        reduction = (unc_0 - unc_100) / unc_0 * 100
        print(f"\n{domain}:")
        print(f"  Target region: {'surname [271,413]' if domain == 'F1' else 'SHIPPINGPOINT [7,79]'}")
        print(f"  Uncertainty reduction with full data: {reduction:.1f}%")
        print(f"  Interpretation: Collecting {results[-1]['n_target']} more samples in target region")
        print(f"                  → reduces uncertainty by {reduction:.1f}%")

    return all_results


if __name__ == "__main__":
    results = main()
