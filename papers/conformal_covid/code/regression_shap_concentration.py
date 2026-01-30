#!/usr/bin/env python3
"""
Compute SHAP Concentration for Regression Tasks

CRITICAL ISSUE: Paper only tests concentration threshold on CATEGORICAL features.
Need to validate whether 40% threshold applies to CONTINUOUS features.

This script:
1. Trains LightGBM regressors on continuous-feature tasks
2. Computes SHAP concentration for each
3. Tests if concentration predicts coverage degradation for regression
4. Determines if 40% threshold generalizes to continuous features

Tasks to test:
- rel-f1/driver-position (continuous: lap times, speeds, etc.)
- rel-trial/study-adverse (continuous: medical measurements)
- rel-trial/site-success (continuous: trial metrics)
"""

import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Tuple

# TODO: Add actual implementation once feature engineering is ready


def compute_regression_shap_concentration(
    model,
    X_val: pd.DataFrame,
    feature_types: Dict[str, str],
    num_samples: int = 10000
) -> Dict:
    """
    Compute SHAP concentration for regression task.

    Args:
        model: Trained LightGBM regressor
        X_val: Validation features
        feature_types: Dict mapping feature name → type ('categorical', 'continuous', 'mixed')
        num_samples: Samples for SHAP computation

    Returns:
        Dictionary with:
        - concentration: % of importance in top feature
        - top_feature: Name of top feature
        - top_feature_type: Type of top feature (categorical/continuous)
        - categorical_concentration: Concentration among categorical features only
        - continuous_concentration: Concentration among continuous features only
        - all_importances: Dict of all feature importances
    """
    import shap

    # Subsample
    if len(X_val) > num_samples:
        X_subsample = X_val.sample(num_samples, random_state=42)
    else:
        X_subsample = X_val

    # Compute SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_subsample)

    # Mean absolute SHAP per feature
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    total_importance = np.sum(feature_importance)

    # Overall concentration
    top_idx = np.argmax(feature_importance)
    top_feature = X_val.columns[top_idx]
    concentration = (feature_importance[top_idx] / total_importance) * 100

    # By feature type
    categorical_importance = 0
    continuous_importance = 0
    max_categorical_importance = 0
    max_continuous_importance = 0

    for i, col in enumerate(X_val.columns):
        feat_type = feature_types.get(col, 'unknown')
        imp = feature_importance[i]

        if feat_type == 'categorical':
            categorical_importance += imp
            max_categorical_importance = max(max_categorical_importance, imp)
        elif feat_type == 'continuous':
            continuous_importance += imp
            max_continuous_importance = max(max_continuous_importance, imp)

    # Concentration within each type
    categorical_concentration = (
        (max_categorical_importance / categorical_importance * 100)
        if categorical_importance > 0 else 0
    )
    continuous_concentration = (
        (max_continuous_importance / continuous_importance * 100)
        if continuous_importance > 0 else 0
    )

    return {
        'concentration': concentration,
        'top_feature': top_feature,
        'top_feature_type': feature_types.get(top_feature, 'unknown'),
        'categorical_concentration': categorical_concentration,
        'continuous_concentration': continuous_concentration,
        'categorical_importance_pct': (categorical_importance / total_importance) * 100,
        'continuous_importance_pct': (continuous_importance / total_importance) * 100,
        'all_importances': {
            col: feature_importance[i]
            for i, col in enumerate(X_val.columns)
        }
    }


def analyze_feature_type_dependency(results: list) -> Dict:
    """
    Analyze if concentration threshold depends on feature type.

    Key question: Does 40% threshold work for continuous features?

    Args:
        results: List of dicts with concentration, drop, feature_types

    Returns:
        Analysis dictionary
    """
    categorical_tasks = []
    continuous_tasks = []
    mixed_tasks = []

    for r in results:
        if r['continuous_importance_pct'] < 20:
            categorical_tasks.append(r)
        elif r['categorical_importance_pct'] < 20:
            continuous_tasks.append(r)
        else:
            mixed_tasks.append(r)

    def compute_stats(task_list, name):
        if not task_list:
            return None

        concentrations = [t['concentration'] for t in task_list]
        drops = [t['drop'] for t in task_list]

        from scipy.stats import spearmanr
        if len(task_list) >= 3:
            rho, pval = spearmanr(concentrations, drops)
        else:
            rho, pval = None, None

        return {
            'type': name,
            'n_tasks': len(task_list),
            'mean_concentration': np.mean(concentrations),
            'mean_drop': np.mean(drops),
            'correlation_rho': rho,
            'correlation_p': pval,
            'tasks': task_list
        }

    analysis = {
        'categorical': compute_stats(categorical_tasks, 'Categorical-dominant'),
        'continuous': compute_stats(continuous_tasks, 'Continuous-dominant'),
        'mixed': compute_stats(mixed_tasks, 'Mixed'),
    }

    # Key finding: Does threshold work for continuous?
    print("\n" + "="*80)
    print("CRITICAL ANALYSIS: Does 40% threshold generalize to continuous features?")
    print("="*80)

    for feat_type, stats in analysis.items():
        if stats is None:
            continue

        print(f"\n{stats['type']} Tasks (n={stats['n_tasks']}):")
        print(f"  Mean concentration: {stats['mean_concentration']:.1f}%")
        print(f"  Mean drop: {stats['mean_drop']:.1f}%")

        if stats['correlation_rho'] is not None:
            print(f"  Correlation: ρ={stats['correlation_rho']:.3f}, p={stats['correlation_p']:.4f}")

            # Test 40% threshold
            vulnerable_count = sum(1 for t in stats['tasks'] if t['concentration'] > 40 and t['drop'] > 50)
            robust_count = sum(1 for t in stats['tasks'] if t['concentration'] <= 40 and t['drop'] < 15)
            total = len(stats['tasks'])

            accuracy = (vulnerable_count + robust_count) / total if total > 0 else 0
            print(f"  40% threshold accuracy: {accuracy:.1%} ({vulnerable_count+robust_count}/{total})")

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Test SHAP concentration on regression tasks with continuous features"
    )
    parser.add_argument('--datasets', nargs='+',
                       default=['rel-f1', 'rel-trial'],
                       help='Datasets to test')
    parser.add_argument('--num_seeds', type=int, default=5)

    args = parser.parse_args()

    print("="*80)
    print("ADDRESSING CRITICAL ISSUE: Continuous Feature Validation")
    print("="*80)
    print()
    print("Current state: 40% threshold derived from n=8 CATEGORICAL feature tasks")
    print("Unknown: Does threshold apply to CONTINUOUS features?")
    print()
    print("This script will:")
    print("1. Compute SHAP concentration for regression tasks (continuous features)")
    print("2. Test if concentration-drop correlation holds")
    print("3. Determine if 40% threshold needs adjustment for continuous features")
    print()

    # TODO: Implement actual experiments
    print("STATUS: PLACEHOLDER - Awaiting feature engineering implementation")
    print()
    print("Required next steps:")
    print("1. Extract features for rel-f1 regression tasks (continuous: lap times, speeds)")
    print("2. Extract features for rel-trial regression tasks (continuous: medical metrics)")
    print("3. Train LightGBM regressors")
    print("4. Compute SHAP concentration split by feature type")
    print("5. Test correlation and threshold accuracy")


if __name__ == "__main__":
    print("""
    ========================================================================
    CRITICAL LIMITATION: Categorical Features Only
    ========================================================================

    Paper results based on:
    - 8 tasks from rel-salt (supply chain)
    - ALL categorical features (transaction IDs, product codes, org units)
    - 40% concentration threshold derived from these tasks

    Validation gap:
    - NO continuous features tested (prices, measurements, embeddings)
    - NO high-dimensional features (images, text)
    - Threshold may not generalize

    This threatens practical applicability:
    - Many real-world tasks use continuous features
    - Concentration dynamics may differ
    - Need validation before recommending to practitioners

    Target validation:
    - rel-f1: Continuous motorsports data (lap times, speeds, positions)
    - rel-trial: Continuous medical data (measurements, counts)
    - Minimum n=3 continuous-dominant tasks

    ========================================================================
    """)
    main()
