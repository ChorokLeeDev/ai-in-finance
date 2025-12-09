"""
Test Intervention Effects: Does fixing data actually reduce uncertainty?
========================================================================

Based on hierarchical decomposition results:
- F1: DRIVER → surname → [271,413] has highest uncertainty
- SALT: ITEM → SHIPPINGPOINT → [7,79] has highest uncertainty

Test: If we "fix" these problematic regions, does uncertainty actually decrease?

Interventions to test:
1. Add more samples (simulate data collection)
2. Reduce variance (simulate data quality improvement)
3. Impute with better values (simulate data correction)

Author: ChorokLeeDev
Created: 2025-12-09
"""

import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict
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
        col_to_fk = data.get('col_to_fk', {})
        if isinstance(X, dict):
            X = pd.DataFrame(X)
    else:
        X, y, _, col_to_fk = data
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

    return X, np.array(y).flatten(), col_to_fk


def intervention_add_samples(X_train, y_train, X_test, target_col, target_range, n_new=100):
    """
    Simulate collecting more data for a specific value range.

    Strategy: Duplicate samples from the target range (with small noise)
    to simulate having more training data for that subpopulation.
    """
    col_data = X_train[target_col]

    if isinstance(target_range, tuple):
        low, high = target_range
        mask = (col_data >= low) & (col_data <= high)
    else:
        mask = col_data == target_range

    if mask.sum() == 0:
        return X_train, y_train

    # Sample from existing data in this range (with replacement)
    idx_in_range = np.where(mask)[0]
    new_idx = np.random.choice(idx_in_range, size=min(n_new, len(idx_in_range)*2), replace=True)

    X_new = X_train.iloc[new_idx].copy()
    y_new = y_train[new_idx].copy()

    # Add small noise to avoid exact duplicates
    for col in X_new.select_dtypes(include=[np.number]).columns:
        noise = np.random.normal(0, X_new[col].std() * 0.01, len(X_new))
        X_new[col] = X_new[col] + noise

    X_augmented = pd.concat([X_train, X_new], ignore_index=True)
    y_augmented = np.concatenate([y_train, y_new])

    return X_augmented, y_augmented


def intervention_reduce_variance(X, target_col, target_range, shrink_factor=0.5):
    """
    Simulate improving data quality by reducing variance in target range.

    Strategy: Shrink values toward the mean within the range.
    """
    X_new = X.copy()
    col_data = X_new[target_col]

    if isinstance(target_range, tuple):
        low, high = target_range
        mask = (col_data >= low) & (col_data <= high)
    else:
        mask = col_data == target_range

    if mask.sum() == 0:
        return X_new

    mean_val = col_data[mask].mean()
    X_new.loc[mask, target_col] = mean_val + shrink_factor * (col_data[mask] - mean_val)

    return X_new


def intervention_impute_outliers(X, target_col, target_range):
    """
    Simulate fixing outliers by replacing extreme values with median.
    """
    X_new = X.copy()
    col_data = X_new[target_col]

    if isinstance(target_range, tuple):
        low, high = target_range
        mask = (col_data >= low) & (col_data <= high)
    else:
        mask = col_data == target_range

    if mask.sum() == 0:
        return X_new

    # Replace with median of the range
    median_val = col_data[mask].median()
    X_new.loc[mask, target_col] = median_val

    return X_new


def test_interventions_on_domain(domain_name, X, y, col_to_fk, target_col, target_range):
    """Test various interventions and measure uncertainty reduction."""

    print(f"\n{'='*70}")
    print(f"INTERVENTION TESTING: {domain_name.upper()}")
    print(f"Target: {target_col} in range {target_range}")
    print(f"{'='*70}")

    # Split data
    n = len(X)
    train_idx = np.random.permutation(n)[:int(n*0.7)]
    test_idx = np.array([i for i in range(n) if i not in train_idx])

    X_train, y_train = X.iloc[train_idx].reset_index(drop=True), y[train_idx]
    X_test, y_test = X.iloc[test_idx].reset_index(drop=True), y[test_idx]

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Count samples in target range
    col_data = X_test[target_col]
    if isinstance(target_range, tuple):
        low, high = target_range
        mask = (col_data >= low) & (col_data <= high)
    else:
        mask = col_data == target_range
    print(f"Test samples in target range: {mask.sum()}")

    results = {}

    # ===== BASELINE =====
    print(f"\n[1/4] Training BASELINE model...")
    ensemble_base = train_lgbm_ensemble(X_train.values, y_train, n_models=10, seed=42)

    unc_base_all = ensemble_base.get_uncertainty(X_test.values).mean()
    unc_base_target = ensemble_base.get_uncertainty(X_test[mask].values).mean() if mask.sum() > 0 else 0

    print(f"      Baseline uncertainty (all test): {unc_base_all:.6f}")
    print(f"      Baseline uncertainty (target range): {unc_base_target:.6f}")

    results['baseline'] = {
        'unc_all': unc_base_all,
        'unc_target': unc_base_target
    }

    # ===== INTERVENTION 1: Add Samples =====
    print(f"\n[2/4] Testing INTERVENTION: Add samples to target range...")
    X_aug, y_aug = intervention_add_samples(X_train, y_train, X_test, target_col, target_range, n_new=100)

    ensemble_aug = train_lgbm_ensemble(X_aug.values, y_aug, n_models=10, seed=42)

    unc_aug_all = ensemble_aug.get_uncertainty(X_test.values).mean()
    unc_aug_target = ensemble_aug.get_uncertainty(X_test[mask].values).mean() if mask.sum() > 0 else 0

    reduction_all = (unc_base_all - unc_aug_all) / unc_base_all * 100
    reduction_target = (unc_base_target - unc_aug_target) / unc_base_target * 100 if unc_base_target > 0 else 0

    print(f"      After adding {len(X_aug) - len(X_train)} samples:")
    print(f"      Uncertainty (all): {unc_aug_all:.6f} ({reduction_all:+.1f}%)")
    print(f"      Uncertainty (target): {unc_aug_target:.6f} ({reduction_target:+.1f}%)")

    results['add_samples'] = {
        'unc_all': unc_aug_all,
        'unc_target': unc_aug_target,
        'reduction_all': reduction_all,
        'reduction_target': reduction_target,
        'samples_added': len(X_aug) - len(X_train)
    }

    # ===== INTERVENTION 2: Reduce Variance (at test time) =====
    print(f"\n[3/4] Testing INTERVENTION: Reduce variance in target range...")
    X_test_reduced = intervention_reduce_variance(X_test, target_col, target_range, shrink_factor=0.3)

    # Use baseline model but test on "cleaned" data
    unc_reduced_all = ensemble_base.get_uncertainty(X_test_reduced.values).mean()

    col_data_reduced = X_test_reduced[target_col]
    if isinstance(target_range, tuple):
        mask_reduced = (col_data_reduced >= low) & (col_data_reduced <= high)
    else:
        mask_reduced = col_data_reduced == target_range
    unc_reduced_target = ensemble_base.get_uncertainty(X_test_reduced[mask_reduced].values).mean() if mask_reduced.sum() > 0 else 0

    reduction_all = (unc_base_all - unc_reduced_all) / unc_base_all * 100
    reduction_target = (unc_base_target - unc_reduced_target) / unc_base_target * 100 if unc_base_target > 0 else 0

    print(f"      After reducing variance (shrink=0.3):")
    print(f"      Uncertainty (all): {unc_reduced_all:.6f} ({reduction_all:+.1f}%)")
    print(f"      Uncertainty (target): {unc_reduced_target:.6f} ({reduction_target:+.1f}%)")

    results['reduce_variance'] = {
        'unc_all': unc_reduced_all,
        'unc_target': unc_reduced_target,
        'reduction_all': reduction_all,
        'reduction_target': reduction_target
    }

    # ===== INTERVENTION 3: Impute to median =====
    print(f"\n[4/4] Testing INTERVENTION: Impute outliers to median...")
    X_test_imputed = intervention_impute_outliers(X_test, target_col, target_range)

    unc_imputed_all = ensemble_base.get_uncertainty(X_test_imputed.values).mean()

    # After imputation, the "target range" is now all the same value
    # So we measure uncertainty on those same indices
    unc_imputed_target = ensemble_base.get_uncertainty(X_test_imputed[mask].values).mean() if mask.sum() > 0 else 0

    reduction_all = (unc_base_all - unc_imputed_all) / unc_base_all * 100
    reduction_target = (unc_base_target - unc_imputed_target) / unc_base_target * 100 if unc_base_target > 0 else 0

    print(f"      After imputing to median:")
    print(f"      Uncertainty (all): {unc_imputed_all:.6f} ({reduction_all:+.1f}%)")
    print(f"      Uncertainty (target): {unc_imputed_target:.6f} ({reduction_target:+.1f}%)")

    results['impute_median'] = {
        'unc_all': unc_imputed_all,
        'unc_target': unc_imputed_target,
        'reduction_all': reduction_all,
        'reduction_target': reduction_target
    }

    return results


def print_summary(all_results):
    """Print summary table."""

    print(f"\n{'='*70}")
    print("INTERVENTION EFFECTS SUMMARY")
    print(f"{'='*70}")

    print(f"\n{'Domain':<10} │ {'Intervention':<20} │ {'Δ Unc (All)':>12} │ {'Δ Unc (Target)':>14}")
    print(f"{'─'*70}")

    for domain, results in all_results.items():
        for intervention, data in results.items():
            if intervention == 'baseline':
                continue
            print(f"{domain:<10} │ {intervention:<20} │ {data['reduction_all']:>+11.1f}% │ {data['reduction_target']:>+13.1f}%")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    print("""
    Positive % = uncertainty REDUCED (good)
    Negative % = uncertainty INCREASED (bad)

    Note: Negative results indicate the interventions are NOT reducing uncertainty.
    This could mean:
    1. The intervention simulation is unrealistic
    2. Variance reduction / imputation shifts data OOD (out of distribution)
    3. Need different intervention strategies

    - add_samples: Retrain model with more data in target range
    - reduce_variance: Shrink values toward mean (simulate better data quality)
    - impute_median: Replace all values with median (simulate standardization)
    """)


def main():
    print("="*70)
    print("TESTING INTERVENTION EFFECTS")
    print("Does fixing identified problems actually reduce uncertainty?")
    print("="*70)

    np.random.seed(42)

    all_results = {}

    # ===== F1: DRIVER → surname → [271, 413] =====
    print("\n>>> Loading F1 data...")
    X, y, col_to_fk = load_cached_data('f1')

    # From decomposition: surname in range [271, 413] has highest uncertainty
    results_f1 = test_interventions_on_domain(
        'F1', X, y, col_to_fk,
        target_col='surname',
        target_range=(271, 413)
    )
    all_results['F1'] = results_f1

    # ===== SALT: ITEM → SHIPPINGPOINT → [7, 79] =====
    print("\n>>> Loading SALT data...")
    X, y, col_to_fk = load_cached_data('salt')

    # From decomposition: SHIPPINGPOINT in range [7, 79] has highest uncertainty
    results_salt = test_interventions_on_domain(
        'SALT', X, y, col_to_fk,
        target_col='SHIPPINGPOINT',
        target_range=(7, 79)
    )
    all_results['SALT'] = results_salt

    # Summary
    print_summary(all_results)

    return all_results


if __name__ == "__main__":
    results = main()
