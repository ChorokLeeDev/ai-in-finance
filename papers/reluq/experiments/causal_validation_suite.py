"""
Causal Validation Suite for RelUQ Paper
========================================

This script implements three critical experiments to strengthen the NeurIPS submission:

1. CORRUPTION EXPERIMENT (P0)
   - Corrupt top-uncertainty FK, retrain, measure error increase
   - Validates: "uncertainty predicts corruption sensitivity"
   - Success: ρ > 0.7 between uncertainty and error increase

2. LEARNING CURVE EXPERIMENT (P2)
   - Subsample data at various rates, measure FK uncertainty
   - Validates: "uncertainty is epistemic (reducible with more data)"
   - Success: high-uncertainty FKs show steeper decrease

3. EP DOMAIN DETECTION (P1)
   - Compute correlation between uncertainty and error attribution
   - Validates: "EP domains have ρ ≈ 1, non-EP domains have ρ < 0.3"
   - Provides: diagnostic tool for practitioners

Usage:
    python causal_validation_suite.py --dataset rel-f1 --task driver-position
    python causal_validation_suite.py --all  # Run on all domains
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import lightgbm as lgb
from relbench.datasets import get_dataset
from relbench.tasks import get_task

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =============================================================================
# CORE UTILITIES (from fk_active_learning.py)
# =============================================================================

def extract_features_with_fk(dataset, task, sample_size=3000):
    """Extract features and track FK groups."""
    db = dataset.get_db()
    train_table = task.get_table("train")

    entity_table_name = task.entity_table
    entity_table = db.table_dict[entity_table_name]
    entity_df = entity_table.df.copy()
    train_df = train_table.df.copy()

    if len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=42)

    fk_to_entity = list(train_table.fkey_col_to_pkey_table.keys())[0]
    entity_pkey = entity_table.pkey_col

    merged_df = train_df.merge(entity_df, how='left', left_on=fk_to_entity,
                               right_on=entity_pkey, suffixes=('', '_entity'))

    target_col = task.target_col
    y = merged_df[target_col].values

    col_to_fk = {}
    feature_cols = []

    # TRAIN features
    for col in train_df.columns:
        if col == target_col:
            continue
        if col.endswith('Id') or col.endswith('_id'):
            continue
        if train_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            if col in merged_df.columns:
                feature_cols.append(col)
                col_to_fk[col] = 'TRAIN'

    # ENTITY features
    for col in entity_df.columns:
        if col == entity_pkey:
            continue
        if col.endswith('Id') or col.endswith('_id'):
            continue
        if entity_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            col_name = col if col in merged_df.columns else f"{col}_entity"
            if col_name in merged_df.columns and col_name not in feature_cols:
                feature_cols.append(col_name)
                col_to_fk[col_name] = entity_table_name.upper()

    # FK table features (aggregated)
    for table_name, table in db.table_dict.items():
        if table_name == entity_table_name:
            continue
        if hasattr(table, 'fkey_col_to_pkey_table'):
            for fk_col, ref_table in table.fkey_col_to_pkey_table.items():
                if ref_table == entity_table_name:
                    table_df = table.df
                    numeric_cols = [c for c in table_df.select_dtypes(include=[np.number]).columns
                                   if not c.endswith('Id') and c != fk_col]

                    if numeric_cols:
                        agg_df = table_df.groupby(fk_col)[numeric_cols].mean().reset_index()
                        agg_df.columns = [fk_col] + [f'{table_name}_{c}_mean' for c in numeric_cols]

                        merged_df = merged_df.merge(agg_df, how='left', left_on=fk_to_entity,
                                                    right_on=fk_col, suffixes=('', f'_{table_name}'))

                        for col in agg_df.columns[1:]:
                            if col in merged_df.columns and col not in feature_cols:
                                feature_cols.append(col)
                                col_to_fk[col] = table_name.upper()

    X = merged_df[feature_cols].fillna(0).values

    # FK column mapping
    fk_to_cols = defaultdict(list)
    for i, col in enumerate(feature_cols):
        fk_name = col_to_fk[col]
        fk_to_cols[fk_name].append(i)

    return X, y, col_to_fk, feature_cols, dict(fk_to_cols)


def train_ensemble(X, y, n_models=5, seed=42, is_classification=False):
    """Train LightGBM ensemble."""
    models = []
    for i in range(n_models):
        idx = np.random.RandomState(seed+i).choice(len(X), int(0.8 * len(X)), replace=True)

        if is_classification:
            model = lgb.LGBMClassifier(
                n_estimators=50, max_depth=6, learning_rate=0.1,
                random_state=seed+i, verbose=-1
            )
        else:
            model = lgb.LGBMRegressor(
                n_estimators=50, max_depth=6, learning_rate=0.1,
                random_state=seed+i, verbose=-1
            )
        model.fit(X[idx], y[idx])
        models.append(model)

    return models


def ensemble_variance(models, X, is_classification=False):
    """Compute ensemble variance (epistemic uncertainty)."""
    if is_classification:
        # Use probability variance
        preds = []
        for m in models:
            if hasattr(m, 'predict_proba'):
                proba = m.predict_proba(X)
                if proba.shape[1] == 2:
                    preds.append(proba[:, 1])
                else:
                    preds.append(proba.max(axis=1))
            else:
                preds.append(m.predict(X))
        preds = np.array(preds)
    else:
        preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0)


def compute_mae(models, X, y, is_classification=False):
    """Compute ensemble MAE (or error rate for classification)."""
    preds = np.array([m.predict(X) for m in models])
    mean_pred = preds.mean(axis=0)

    if is_classification:
        return (mean_pred.round() != y).mean()
    else:
        return np.abs(mean_pred - y).mean()


# =============================================================================
# EXPERIMENT 1: CORRUPTION VALIDATION
# =============================================================================

def run_corruption_experiment(X, y, fk_to_cols, feature_cols, n_models=5, seed=42,
                               corruption_levels=[0.1, 0.25, 0.5, 0.75, 1.0],
                               is_classification=False):
    """
    CORRUPTION EXPERIMENT: Validate that uncertainty predicts corruption sensitivity.

    Method:
    1. Train baseline ensemble on clean data
    2. Compute FK-level uncertainty contributions
    3. For each FK, corrupt its features at various noise levels
    4. Retrain on corrupted data, measure error increase
    5. Correlate uncertainty contribution with corruption sensitivity

    Success criterion: Spearman ρ > 0.7
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: CORRUPTION VALIDATION")
    print("="*70)
    print("\nGoal: Show that FK uncertainty predicts corruption sensitivity")
    print("Method: Corrupt each FK's features, retrain, measure error increase")
    print("Success: Spearman ρ > 0.7 between uncertainty and error increase")

    # Step 1: Train baseline
    print("\n[Step 1] Training baseline ensemble...")
    models = train_ensemble(X, y, n_models=n_models, seed=seed, is_classification=is_classification)
    base_error = compute_mae(models, X, y, is_classification)
    print(f"  Baseline error: {base_error:.4f}")

    # Step 2: Compute FK uncertainties
    print("\n[Step 2] Computing FK-level uncertainties...")
    base_unc = ensemble_variance(models, X, is_classification).mean()

    fk_uncertainties = {}
    for fk_name, col_indices in fk_to_cols.items():
        if not col_indices:
            continue

        # Permute and measure variance change
        contributions = []
        for _ in range(5):
            X_perm = X.copy()
            for col_idx in col_indices:
                X_perm[:, col_idx] = np.random.permutation(X_perm[:, col_idx])
            perm_unc = ensemble_variance(models, X_perm, is_classification).mean()
            # Positive = FK adds uncertainty (removing it reduces variance)
            # Negative = FK reduces uncertainty (removing it increases variance)
            contrib = (base_unc - perm_unc) / base_unc * 100
            contributions.append(contrib)

        fk_uncertainties[fk_name] = np.mean(contributions)
        print(f"  {fk_name}: {fk_uncertainties[fk_name]:+.1f}% uncertainty contribution")

    # Step 3: Corrupt each FK and measure sensitivity
    print("\n[Step 3] Measuring corruption sensitivity...")

    fk_corruption_sensitivity = {}
    detailed_results = {}

    for fk_name, col_indices in fk_to_cols.items():
        if not col_indices:
            continue

        print(f"\n  Corrupting {fk_name} ({len(col_indices)} features)...")

        error_increases = []

        for noise_level in corruption_levels:
            # Corrupt FK features
            X_corrupted = X.copy()
            for col_idx in col_indices:
                col_std = X[:, col_idx].std()
                if col_std > 0:
                    noise = np.random.normal(0, col_std * noise_level, len(X))
                    X_corrupted[:, col_idx] = X[:, col_idx] + noise

            # Retrain on corrupted data
            models_corrupted = train_ensemble(X_corrupted, y, n_models=n_models,
                                              seed=seed+100, is_classification=is_classification)

            # Measure error on ORIGINAL clean data (transfer error)
            corrupted_error = compute_mae(models_corrupted, X, y, is_classification)
            error_increase = (corrupted_error - base_error) / base_error * 100
            error_increases.append(error_increase)

            print(f"    noise={noise_level:.0%}: error increase = {error_increase:+.1f}%")

        # Use max error increase as sensitivity metric
        fk_corruption_sensitivity[fk_name] = max(error_increases)
        detailed_results[fk_name] = {
            'uncertainty': fk_uncertainties[fk_name],
            'corruption_sensitivity': max(error_increases),
            'error_by_noise_level': dict(zip(corruption_levels, error_increases))
        }

    # Step 4: Compute correlation
    print("\n[Step 4] Computing correlation...")

    fk_names = list(fk_uncertainties.keys())
    uncertainties = [fk_uncertainties[fk] for fk in fk_names]
    sensitivities = [fk_corruption_sensitivity[fk] for fk in fk_names]

    if len(fk_names) >= 3:
        rho, p_value = stats.spearmanr(uncertainties, sensitivities)
    else:
        rho, p_value = 0.0, 1.0

    print(f"\n  Spearman ρ = {rho:.3f} (p = {p_value:.4f})")

    # Verdict
    print("\n" + "="*70)
    if rho > 0.7:
        verdict = "PASS"
        print(f"VERDICT: ✅ {verdict} - Strong correlation (ρ = {rho:.3f} > 0.7)")
        print("  FK uncertainty DOES predict corruption sensitivity!")
    elif rho > 0.5:
        verdict = "MARGINAL"
        print(f"VERDICT: ⚠️  {verdict} - Moderate correlation (ρ = {rho:.3f})")
        print("  FK uncertainty shows some predictive power")
    else:
        verdict = "FAIL"
        print(f"VERDICT: ❌ {verdict} - Weak correlation (ρ = {rho:.3f} < 0.5)")
        print("  FK uncertainty does NOT reliably predict corruption sensitivity")
    print("="*70)

    return {
        'experiment': 'corruption',
        'verdict': verdict,
        'spearman_rho': float(rho),
        'p_value': float(p_value),
        'base_error': float(base_error),
        'fk_results': detailed_results,
        'corruption_levels': corruption_levels
    }


# =============================================================================
# EXPERIMENT 2: LEARNING CURVE VALIDATION
# =============================================================================

def run_learning_curve_experiment(X, y, fk_to_cols, feature_cols, n_models=5, seed=42,
                                   data_fractions=[0.2, 0.4, 0.6, 0.8, 1.0],
                                   is_classification=False):
    """
    LEARNING CURVE EXPERIMENT: Validate that uncertainty is epistemic (reducible).

    Method:
    1. Train ensemble on subsets of data (20%, 40%, 60%, 80%, 100%)
    2. Compute FK-level uncertainty at each data level
    3. Show that high-uncertainty FKs have steeper decrease

    Success criterion: Negative correlation between initial uncertainty and decrease rate
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: LEARNING CURVE (EPISTEMIC VALIDATION)")
    print("="*70)
    print("\nGoal: Show that uncertainty is epistemic (reducible with more data)")
    print("Method: Train on data subsets, measure uncertainty decrease")
    print("Success: High-uncertainty FKs show steeper decrease")

    results_by_fk = {fk: [] for fk in fk_to_cols.keys()}

    for frac in data_fractions:
        n = int(len(X) * frac)
        print(f"\n[Data fraction: {frac:.0%} ({n} samples)]")

        # Subsample
        idx = np.random.RandomState(seed).choice(len(X), n, replace=False)
        X_sub, y_sub = X[idx], y[idx]

        # Train ensemble
        models = train_ensemble(X_sub, y_sub, n_models=n_models, seed=seed,
                               is_classification=is_classification)

        # Compute FK uncertainties
        base_unc = ensemble_variance(models, X_sub, is_classification).mean()

        for fk_name, col_indices in fk_to_cols.items():
            if not col_indices:
                continue

            contributions = []
            for _ in range(3):
                X_perm = X_sub.copy()
                for col_idx in col_indices:
                    X_perm[:, col_idx] = np.random.permutation(X_perm[:, col_idx])
                perm_unc = ensemble_variance(models, X_perm, is_classification).mean()
                contrib = (base_unc - perm_unc) / base_unc * 100
                contributions.append(contrib)

            fk_unc = np.mean(contributions)
            results_by_fk[fk_name].append(fk_unc)
            print(f"  {fk_name}: {fk_unc:+.1f}%")

    # Compute decrease rates
    print("\n[Computing decrease rates]")

    decrease_rates = {}
    initial_uncertainties = {}

    for fk_name, uncertainties in results_by_fk.items():
        if len(uncertainties) >= 2:
            initial_unc = uncertainties[0]
            final_unc = uncertainties[-1]
            decrease_rate = initial_unc - final_unc  # Positive = decreased

            decrease_rates[fk_name] = decrease_rate
            initial_uncertainties[fk_name] = initial_unc

            print(f"  {fk_name}: {initial_unc:+.1f}% → {final_unc:+.1f}% (decrease: {decrease_rate:+.1f}%)")

    # Correlate initial uncertainty with decrease rate
    if len(decrease_rates) >= 3:
        fk_names = list(decrease_rates.keys())
        initials = [initial_uncertainties[fk] for fk in fk_names]
        decreases = [decrease_rates[fk] for fk in fk_names]

        rho, p_value = stats.spearmanr(initials, decreases)
    else:
        rho, p_value = 0.0, 1.0

    # Verdict
    print("\n" + "="*70)

    # Check if high-uncertainty FKs have positive decrease (epistemic)
    high_unc_fks = [fk for fk, unc in initial_uncertainties.items() if unc > 0]
    if high_unc_fks:
        high_unc_decreases = [decrease_rates[fk] for fk in high_unc_fks]
        avg_decrease = np.mean(high_unc_decreases)
        epistemic_validated = avg_decrease > 0
    else:
        avg_decrease = 0
        epistemic_validated = False

    if epistemic_validated:
        verdict = "PASS"
        print(f"VERDICT: ✅ {verdict} - High-uncertainty FKs show avg decrease of {avg_decrease:+.1f}%")
        print("  Uncertainty IS epistemic (reducible with more data)")
    else:
        verdict = "FAIL"
        print(f"VERDICT: ❌ {verdict} - High-uncertainty FKs don't decrease with more data")
        print("  Uncertainty may be aleatoric (irreducible)")
    print("="*70)

    return {
        'experiment': 'learning_curve',
        'verdict': verdict,
        'data_fractions': data_fractions,
        'fk_uncertainties_by_fraction': {fk: list(map(float, uncs))
                                          for fk, uncs in results_by_fk.items()},
        'decrease_rates': {k: float(v) for k, v in decrease_rates.items()},
        'initial_uncertainties': {k: float(v) for k, v in initial_uncertainties.items()},
        'avg_high_unc_decrease': float(avg_decrease) if high_unc_fks else 0.0,
        'correlation_rho': float(rho),
        'correlation_p': float(p_value)
    }


# =============================================================================
# EXPERIMENT 3: EP DOMAIN DETECTION
# =============================================================================

def run_ep_detection_experiment(X, y, fk_to_cols, feature_cols, n_models=5, seed=42,
                                 is_classification=False):
    """
    EP DOMAIN DETECTION: Validate Error Propagation property.

    Method:
    1. Compute FK-level uncertainty attribution
    2. Compute FK-level error attribution (permutation importance)
    3. Correlate the two

    Interpretation:
    - ρ > 0.7: EP domain - FK attribution is valid
    - ρ < 0.3: Non-EP domain - use alternative methods

    This becomes a diagnostic tool for practitioners.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: EP DOMAIN DETECTION")
    print("="*70)
    print("\nGoal: Determine if this domain satisfies Error Propagation (EP)")
    print("Method: Correlate FK uncertainty attribution with FK error attribution")
    print("Interpretation: ρ > 0.7 = EP domain, ρ < 0.3 = non-EP domain")

    # Train baseline
    print("\n[Step 1] Training baseline ensemble...")
    models = train_ensemble(X, y, n_models=n_models, seed=seed, is_classification=is_classification)
    base_error = compute_mae(models, X, y, is_classification)
    base_unc = ensemble_variance(models, X, is_classification).mean()

    print(f"  Baseline error: {base_error:.4f}")
    print(f"  Baseline uncertainty: {base_unc:.6f}")

    # Compute FK attributions
    print("\n[Step 2] Computing FK attributions...")

    fk_uncertainty_attr = {}
    fk_error_attr = {}

    for fk_name, col_indices in fk_to_cols.items():
        if not col_indices:
            continue

        unc_contribs = []
        err_contribs = []

        for _ in range(5):
            X_perm = X.copy()
            for col_idx in col_indices:
                X_perm[:, col_idx] = np.random.permutation(X_perm[:, col_idx])

            # Uncertainty attribution
            perm_unc = ensemble_variance(models, X_perm, is_classification).mean()
            unc_contrib = (base_unc - perm_unc) / base_unc * 100
            unc_contribs.append(unc_contrib)

            # Error attribution (permutation importance)
            perm_error = compute_mae(models, X_perm, y, is_classification)
            err_contrib = (perm_error - base_error) / base_error * 100
            err_contribs.append(err_contrib)

        fk_uncertainty_attr[fk_name] = np.mean(unc_contribs)
        fk_error_attr[fk_name] = np.mean(err_contribs)

        print(f"  {fk_name}:")
        print(f"    Uncertainty: {fk_uncertainty_attr[fk_name]:+.1f}%")
        print(f"    Error:       {fk_error_attr[fk_name]:+.1f}%")

    # Compute correlation
    print("\n[Step 3] Computing EP correlation...")

    fk_names = list(fk_uncertainty_attr.keys())
    uncertainties = [fk_uncertainty_attr[fk] for fk in fk_names]
    errors = [fk_error_attr[fk] for fk in fk_names]

    if len(fk_names) >= 3:
        rho, p_value = stats.spearmanr(uncertainties, errors)
    else:
        rho, p_value = 0.0, 1.0

    print(f"\n  Spearman ρ = {rho:.3f} (p = {p_value:.4f})")

    # Classification
    print("\n" + "="*70)
    if rho > 0.7:
        domain_type = "EP"
        verdict = "EP DOMAIN"
        print(f"VERDICT: ✅ {verdict} (ρ = {rho:.3f} > 0.7)")
        print("  FK-level attribution IS valid for this domain")
        print("  Uncertainty correctly identifies data investment targets")
    elif rho > 0.3:
        domain_type = "MIXED"
        verdict = "MIXED DOMAIN"
        print(f"VERDICT: ⚠️  {verdict} (0.3 < ρ = {rho:.3f} < 0.7)")
        print("  FK-level attribution has moderate validity")
        print("  Interpret results with caution")
    else:
        domain_type = "NON-EP"
        verdict = "NON-EP DOMAIN"
        print(f"VERDICT: ❌ {verdict} (ρ = {rho:.3f} < 0.3)")
        print("  FK-level attribution is NOT valid for this domain")
        print("  Use alternative methods (e.g., sample-level uncertainty)")
    print("="*70)

    return {
        'experiment': 'ep_detection',
        'verdict': verdict,
        'domain_type': domain_type,
        'spearman_rho': float(rho),
        'p_value': float(p_value),
        'fk_uncertainty_attribution': {k: float(v) for k, v in fk_uncertainty_attr.items()},
        'fk_error_attribution': {k: float(v) for k, v in fk_error_attr.items()},
        'threshold_ep': 0.7,
        'threshold_non_ep': 0.3
    }


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_all_experiments(dataset_name, task_name, sample_size=2000, n_models=5, seed=42):
    """Run all three experiments for a dataset/task pair."""
    print("\n" + "="*70)
    print(f"CAUSAL VALIDATION SUITE: {dataset_name} / {task_name}")
    print("="*70)

    # Load data
    print("\nLoading dataset...")
    try:
        dataset = get_dataset(dataset_name, download=True)
        task = get_task(dataset_name, task_name, download=True)
    except Exception as e:
        print(f"ERROR: Could not load {dataset_name}/{task_name}: {e}")
        return None

    # Extract features
    print("Extracting features...")
    try:
        X, y, col_to_fk, feature_cols, fk_to_cols = extract_features_with_fk(
            dataset, task, sample_size=sample_size
        )
    except Exception as e:
        print(f"ERROR: Feature extraction failed: {e}")
        return None

    print(f"\nDataset shape: {X.shape}")
    print(f"FK groups: {list(fk_to_cols.keys())}")

    # Detect if classification
    unique_targets = len(np.unique(y))
    is_classification = unique_targets < 20 and task_name not in ['driver-position']
    print(f"Task type: {'classification' if is_classification else 'regression'}")

    # Run experiments
    results = {
        'dataset': dataset_name,
        'task': task_name,
        'sample_size': len(X),
        'n_features': X.shape[1],
        'n_fk_groups': len(fk_to_cols),
        'fk_groups': list(fk_to_cols.keys()),
        'is_classification': is_classification,
        'experiments': {}
    }

    # Experiment 1: Corruption
    try:
        corruption_result = run_corruption_experiment(
            X, y, fk_to_cols, feature_cols, n_models=n_models, seed=seed,
            is_classification=is_classification
        )
        results['experiments']['corruption'] = corruption_result
    except Exception as e:
        print(f"ERROR in corruption experiment: {e}")
        import traceback
        traceback.print_exc()
        results['experiments']['corruption'] = {'error': str(e)}

    # Experiment 2: Learning Curve
    try:
        learning_curve_result = run_learning_curve_experiment(
            X, y, fk_to_cols, feature_cols, n_models=n_models, seed=seed,
            is_classification=is_classification
        )
        results['experiments']['learning_curve'] = learning_curve_result
    except Exception as e:
        print(f"ERROR in learning curve experiment: {e}")
        import traceback
        traceback.print_exc()
        results['experiments']['learning_curve'] = {'error': str(e)}

    # Experiment 3: EP Detection
    try:
        ep_result = run_ep_detection_experiment(
            X, y, fk_to_cols, feature_cols, n_models=n_models, seed=seed,
            is_classification=is_classification
        )
        results['experiments']['ep_detection'] = ep_result
    except Exception as e:
        print(f"ERROR in EP detection experiment: {e}")
        import traceback
        traceback.print_exc()
        results['experiments']['ep_detection'] = {'error': str(e)}

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for exp_name, exp_result in results['experiments'].items():
        if 'error' in exp_result:
            print(f"  {exp_name}: ERROR - {exp_result['error']}")
        else:
            verdict = exp_result.get('verdict', 'UNKNOWN')
            print(f"  {exp_name}: {verdict}")

    return results


def create_summary_figure(all_results, output_path):
    """Create summary figure across all domains."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Colors for domains
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    # Plot 1: Corruption validation (ρ values)
    ax1 = axes[0]
    domains = []
    rhos = []
    for i, result in enumerate(all_results):
        if 'corruption' in result['experiments']:
            corr = result['experiments']['corruption']
            if 'spearman_rho' in corr:
                domains.append(f"{result['dataset']}\n{result['task']}")
                rhos.append(corr['spearman_rho'])

    if domains:
        bars = ax1.bar(range(len(domains)), rhos, color='steelblue')
        ax1.axhline(y=0.7, color='green', linestyle='--', label='EP threshold (0.7)')
        ax1.axhline(y=0.3, color='red', linestyle='--', label='Non-EP threshold (0.3)')
        ax1.set_xticks(range(len(domains)))
        ax1.set_xticklabels(domains, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('Spearman ρ')
        ax1.set_title('Corruption Experiment\n(Uncertainty vs Sensitivity)')
        ax1.legend(fontsize=8)
        ax1.set_ylim(-1, 1)

    # Plot 2: EP Detection (ρ values)
    ax2 = axes[1]
    domains = []
    rhos = []
    for i, result in enumerate(all_results):
        if 'ep_detection' in result['experiments']:
            ep = result['experiments']['ep_detection']
            if 'spearman_rho' in ep:
                domains.append(f"{result['dataset']}\n{result['task']}")
                rhos.append(ep['spearman_rho'])

    if domains:
        colors_ep = ['green' if r > 0.7 else 'orange' if r > 0.3 else 'red' for r in rhos]
        bars = ax2.bar(range(len(domains)), rhos, color=colors_ep)
        ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.5)
        ax2.set_xticks(range(len(domains)))
        ax2.set_xticklabels(domains, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Spearman ρ')
        ax2.set_title('EP Detection\n(Uncertainty vs Error Attribution)')
        ax2.set_ylim(-1, 1)

    # Plot 3: Learning Curve (avg decrease for high-unc FKs)
    ax3 = axes[2]
    domains = []
    decreases = []
    for i, result in enumerate(all_results):
        if 'learning_curve' in result['experiments']:
            lc = result['experiments']['learning_curve']
            if 'avg_high_unc_decrease' in lc:
                domains.append(f"{result['dataset']}\n{result['task']}")
                decreases.append(lc['avg_high_unc_decrease'])

    if domains:
        colors_lc = ['green' if d > 0 else 'red' for d in decreases]
        bars = ax3.bar(range(len(domains)), decreases, color=colors_lc)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_xticks(range(len(domains)))
        ax3.set_xticklabels(domains, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Avg Uncertainty Decrease (%)')
        ax3.set_title('Learning Curve\n(Epistemic Validation)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Causal Validation Suite for RelUQ")
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--task", type=str, default="driver-position")
    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument("--n_models", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="Run on all domains")
    args = parser.parse_args()

    output_dir = Path(__file__).parent / 'validation_results'
    output_dir.mkdir(exist_ok=True)

    if args.all:
        # Run on all domains
        experiments = [
            ('rel-f1', 'driver-position'),
            ('rel-f1', 'driver-dnf'),
            ('rel-f1', 'driver-top3'),
            ('rel-trial', 'study-outcome'),
            ('rel-salt', 'item-plant'),
            ('rel-avito', 'ad-ctr'),
            ('rel-event', 'user-attendance'),
        ]

        all_results = []
        for dataset_name, task_name in experiments:
            print(f"\n{'#'*70}")
            print(f"# {dataset_name} / {task_name}")
            print(f"{'#'*70}")

            result = run_all_experiments(
                dataset_name, task_name,
                sample_size=args.sample_size,
                n_models=args.n_models,
                seed=args.seed
            )

            if result:
                all_results.append(result)

                # Save individual result
                output_file = output_dir / f'{dataset_name}_{task_name}_validation.json'
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)

        # Save combined results
        combined_file = output_dir / 'all_validation_results.json'
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        # Create summary figure
        fig_path = str(output_dir / 'validation_summary.png')
        create_summary_figure(all_results, fig_path)

        # Print final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY ACROSS ALL DOMAINS")
        print("="*70)

        corruption_pass = sum(1 for r in all_results
                             if r['experiments'].get('corruption', {}).get('verdict') == 'PASS')
        learning_pass = sum(1 for r in all_results
                           if r['experiments'].get('learning_curve', {}).get('verdict') == 'PASS')
        ep_pass = sum(1 for r in all_results
                     if r['experiments'].get('ep_detection', {}).get('domain_type') == 'EP')

        total = len(all_results)

        print(f"\nCorruption Experiment: {corruption_pass}/{total} PASS")
        print(f"Learning Curve: {learning_pass}/{total} PASS")
        print(f"EP Domains: {ep_pass}/{total} detected")

        print(f"\nResults saved to: {output_dir}")

    else:
        # Run on single dataset/task
        result = run_all_experiments(
            args.dataset, args.task,
            sample_size=args.sample_size,
            n_models=args.n_models,
            seed=args.seed
        )

        if result:
            output_file = output_dir / f'{args.dataset}_{args.task}_validation.json'
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
