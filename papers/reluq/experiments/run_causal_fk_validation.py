"""
Causal FK Hypothesis Validation Experiment

This script validates the hypothesis:
"Causal FKs increase epistemic uncertainty, while correlational FKs stabilize predictions"

Methodology:
1. For each dataset/task, train an ensemble of LightGBM models
2. Compute FK-level uncertainty contribution via permutation
3. Compute SHAP values for validation
4. Compare causal vs correlational FK uncertainty patterns
5. Statistical analysis (effect size, p-value)

Usage:
    python run_causal_fk_validation.py --dataset rel-f1 --task driver-position
    python run_causal_fk_validation.py --all  # Run all datasets
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from relbench.datasets import get_dataset
from relbench.tasks import get_task

from causal_fk_config import (
    get_fk_classification,
    get_task_fks,
    F1_FK_CLASSIFICATION,
    SALT_FK_CLASSIFICATION,
    TRIAL_FK_CLASSIFICATION
)


def extract_features_with_fk_groups(dataset, task, sample_size=10000):
    """
    Extract features and identify FK groups for uncertainty attribution.

    Returns:
        X: feature matrix
        y: target vector
        fk_groups: dict mapping FK name to column indices
        feature_names: list of feature names
    """
    db = dataset.get_db()
    train_table = task.get_table("train")

    # Get the entity table (primary table for the task)
    entity_table_name = task.entity_table if hasattr(task, 'entity_table') else None

    # For simplicity, we'll create a flattened feature matrix
    # In practice, this would use the RelBench feature engineering pipeline

    train_df = train_table.df

    if len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=42)

    # Identify target column
    target_col = task.target_col if hasattr(task, 'target_col') else train_df.columns[-1]

    # Separate features and target
    feature_cols = [c for c in train_df.columns if c != target_col and c != task.entity_col]

    # Identify FK columns and group them
    fk_groups = {}

    # Get FK definitions from the database schema
    for table_name, table in db.table_dict.items():
        for fk_col, pkey_table in table.fkey_col_to_pkey_table.items():
            if fk_col in feature_cols:
                group_name = f"{table_name}_{fk_col}"
                if group_name not in fk_groups:
                    fk_groups[group_name] = []
                fk_groups[group_name].append(feature_cols.index(fk_col))

    # If no FKs found in feature columns, create groups based on column prefixes
    if not fk_groups:
        for i, col in enumerate(feature_cols):
            # Group by table name prefix if available
            parts = col.split('_')
            if len(parts) > 1:
                group_name = parts[0]
            else:
                group_name = "other"

            if group_name not in fk_groups:
                fk_groups[group_name] = []
            fk_groups[group_name].append(i)

    # Prepare X and y
    X = train_df[feature_cols].values
    y = train_df[target_col].values

    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)

    return X, y, fk_groups, feature_cols


def train_ensemble(X, y, n_models=5, seeds=None):
    """Train an ensemble of LightGBM models."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not installed. Using sklearn RandomForest as fallback.")
        from sklearn.ensemble import RandomForestRegressor

        if seeds is None:
            seeds = list(range(42, 42 + n_models))

        models = []
        for seed in seeds:
            model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
            model.fit(X, y)
            models.append(model)

        return models

    if seeds is None:
        seeds = list(range(42, 42 + n_models))

    models = []
    for seed in seeds:
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': seed
        }

        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(params, train_data, num_boost_round=100)
        models.append(model)

    return models


def compute_ensemble_predictions(models, X):
    """Get predictions from all models in the ensemble."""
    predictions = []
    for model in models:
        if hasattr(model, 'predict'):
            pred = model.predict(X)
        else:
            pred = model.predict(X, num_iteration=model.best_iteration)
        predictions.append(pred)
    return np.array(predictions)


def compute_ensemble_uncertainty(models, X):
    """Compute epistemic uncertainty as ensemble variance."""
    predictions = compute_ensemble_predictions(models, X)
    return np.var(predictions, axis=0)


def compute_fk_uncertainty_contribution(models, X, fk_groups, n_permutations=5):
    """
    Compute FK-level uncertainty contribution via permutation.

    For each FK group:
    1. Permute the columns belonging to that FK
    2. Measure the change in ensemble variance
    3. Positive = FK increases uncertainty, Negative = FK stabilizes

    Returns:
        dict: {fk_name: uncertainty_contribution}
    """
    # Baseline uncertainty
    base_uncertainty = compute_ensemble_uncertainty(models, X)
    base_mean = np.mean(base_uncertainty)

    fk_contributions = {}

    for fk_name, col_indices in fk_groups.items():
        if not col_indices:
            continue

        contribution_samples = []

        for _ in range(n_permutations):
            # Create a copy and permute FK columns
            X_perm = X.copy()
            for col_idx in col_indices:
                X_perm[:, col_idx] = np.random.permutation(X_perm[:, col_idx])

            # Compute uncertainty with permuted FK
            perm_uncertainty = compute_ensemble_uncertainty(models, X_perm)
            perm_mean = np.mean(perm_uncertainty)

            # Contribution = (permuted - base) / base
            # Positive means removing FK info increases uncertainty
            # So the FK was REDUCING uncertainty (stabilizing)
            # We want to report it from the FK's perspective:
            # Positive = FK increases uncertainty
            # Negative = FK stabilizes (reduces uncertainty)
            contribution = (base_mean - perm_mean) / base_mean * 100
            contribution_samples.append(contribution)

        fk_contributions[fk_name] = {
            'mean': np.mean(contribution_samples),
            'std': np.std(contribution_samples),
            'samples': contribution_samples
        }

    return fk_contributions


def compute_shap_importance(models, X, feature_names):
    """Compute SHAP-based feature importance."""
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Skipping SHAP analysis.")
        return None

    # Use the first model for SHAP (they should be similar)
    model = models[0]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:1000])  # Subsample for speed

        # Aggregate by feature
        importance = np.abs(shap_values).mean(axis=0)

        return dict(zip(feature_names, importance))
    except Exception as e:
        print(f"SHAP computation failed: {e}")
        return None


def classify_fk_contributions(fk_contributions, dataset):
    """Classify FK contributions as causal vs correlational."""
    causal_contributions = []
    correlational_contributions = []

    for fk_name, contrib in fk_contributions.items():
        # Parse FK name to get table and column
        parts = fk_name.split('_')
        table = parts[0] if parts else fk_name

        # Get classification
        classification = get_fk_classification(dataset, table, fk_name)

        if classification['type'] == 'causal':
            causal_contributions.append(contrib['mean'])
        elif classification['type'] == 'correlational':
            correlational_contributions.append(contrib['mean'])

    return causal_contributions, correlational_contributions


def compute_statistics(causal_contribs, corr_contribs):
    """Compute statistical comparison between causal and correlational FKs."""
    results = {}

    if len(causal_contribs) > 0 and len(corr_contribs) > 0:
        # Means
        results['causal_mean'] = np.mean(causal_contribs)
        results['correlational_mean'] = np.mean(corr_contribs)
        results['difference'] = results['causal_mean'] - results['correlational_mean']

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(causal_contribs) + np.var(corr_contribs)) / 2
        )
        if pooled_std > 0:
            results['cohens_d'] = results['difference'] / pooled_std
        else:
            results['cohens_d'] = 0

        # Statistical test (Mann-Whitney U)
        if len(causal_contribs) >= 2 and len(corr_contribs) >= 2:
            stat, p_value = stats.mannwhitneyu(
                causal_contribs, corr_contribs,
                alternative='greater'  # Causal > Correlational
            )
            results['p_value'] = p_value
        else:
            results['p_value'] = 1.0

        results['n_causal'] = len(causal_contribs)
        results['n_correlational'] = len(corr_contribs)

    return results


def run_experiment(dataset_name, task_name, n_models=5, sample_size=10000):
    """Run the full experiment for a single dataset/task."""
    print(f"\n{'='*60}")
    print(f"Running: {dataset_name} / {task_name}")
    print(f"{'='*60}")

    # Load data
    print("Loading dataset...")
    dataset = get_dataset(dataset_name, download=True)
    task = get_task(dataset_name, task_name, download=True)

    # Extract features
    print("Extracting features...")
    X, y, fk_groups, feature_names = extract_features_with_fk_groups(
        dataset, task, sample_size=sample_size
    )
    print(f"  Data shape: {X.shape}")
    print(f"  FK groups: {list(fk_groups.keys())}")

    # Train ensemble
    print(f"Training ensemble ({n_models} models)...")
    models = train_ensemble(X, y, n_models=n_models)

    # Compute FK uncertainty contributions
    print("Computing FK uncertainty contributions...")
    fk_contributions = compute_fk_uncertainty_contribution(models, X, fk_groups)

    for fk_name, contrib in fk_contributions.items():
        print(f"  {fk_name}: {contrib['mean']:.2f}% ± {contrib['std']:.2f}%")

    # Compute SHAP (if available)
    print("Computing SHAP importance...")
    shap_importance = compute_shap_importance(models, X, feature_names)

    # Classify and compare
    print("Classifying FK contributions...")
    causal_contribs, corr_contribs = classify_fk_contributions(
        fk_contributions, dataset_name
    )

    stats_results = compute_statistics(causal_contribs, corr_contribs)

    # Print results
    print("\n" + "-"*40)
    print("RESULTS:")
    print("-"*40)
    if stats_results:
        print(f"  Causal FKs mean:        {stats_results.get('causal_mean', 'N/A'):.2f}%")
        print(f"  Correlational FKs mean: {stats_results.get('correlational_mean', 'N/A'):.2f}%")
        print(f"  Difference:             {stats_results.get('difference', 'N/A'):.2f}%")
        print(f"  Cohen's d:              {stats_results.get('cohens_d', 'N/A'):.2f}")
        print(f"  p-value:                {stats_results.get('p_value', 'N/A'):.4f}")

    # Compile results
    results = {
        'dataset': dataset_name,
        'task': task_name,
        'timestamp': datetime.now().isoformat(),
        'data_shape': list(X.shape),
        'n_models': n_models,
        'fk_groups': list(fk_groups.keys()),
        'fk_contributions': {k: v['mean'] for k, v in fk_contributions.items()},
        'fk_contributions_std': {k: v['std'] for k, v in fk_contributions.items()},
        'causal_contributions': causal_contribs,
        'correlational_contributions': corr_contribs,
        'statistics': stats_results,
        'hypothesis_supported': (
            stats_results.get('difference', 0) > 0 and
            stats_results.get('p_value', 1) < 0.1
        ) if stats_results else False
    }

    return results


def run_all_experiments(output_dir='test_results'):
    """Run experiments on all configured datasets and tasks."""
    all_results = []

    experiments = [
        # F1 tasks
        ('rel-f1', 'driver-position'),
        ('rel-f1', 'driver-dnf'),
        ('rel-f1', 'driver-top3'),
        # SALT tasks
        ('rel-salt', 'item-plant'),
        ('rel-salt', 'sales-office'),
        # Trial tasks
        ('rel-trial', 'study-outcome'),
        ('rel-trial', 'study-adverse'),
    ]

    for dataset_name, task_name in experiments:
        try:
            results = run_experiment(dataset_name, task_name)
            all_results.append(results)
        except Exception as e:
            print(f"Error running {dataset_name}/{task_name}: {e}")
            continue

    # Aggregate analysis
    print("\n" + "="*60)
    print("AGGREGATE ANALYSIS")
    print("="*60)

    supported = sum(1 for r in all_results if r.get('hypothesis_supported', False))
    total = len(all_results)

    print(f"Hypothesis supported: {supported}/{total} tasks")

    # Compute overall statistics
    all_causal = []
    all_corr = []
    for r in all_results:
        all_causal.extend(r.get('causal_contributions', []))
        all_corr.extend(r.get('correlational_contributions', []))

    if all_causal and all_corr:
        overall_stats = compute_statistics(all_causal, all_corr)
        print(f"\nOverall:")
        print(f"  Causal mean:        {overall_stats.get('causal_mean', 'N/A'):.2f}%")
        print(f"  Correlational mean: {overall_stats.get('correlational_mean', 'N/A'):.2f}%")
        print(f"  Cohen's d:          {overall_stats.get('cohens_d', 'N/A'):.2f}")
        print(f"  p-value:            {overall_stats.get('p_value', 'N/A'):.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'causal_fk_validation_results.json')

    with open(output_path, 'w') as f:
        json.dump({
            'experiments': all_results,
            'aggregate': {
                'supported_count': supported,
                'total_count': total,
                'all_causal_contributions': all_causal,
                'all_correlational_contributions': all_corr,
                'overall_statistics': overall_stats if all_causal and all_corr else {}
            }
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Causal FK Hypothesis Validation')
    parser.add_argument('--dataset', type=str, help='Dataset name (e.g., rel-f1)')
    parser.add_argument('--task', type=str, help='Task name (e.g., driver-position)')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--n_models', type=int, default=5, help='Number of ensemble models')
    parser.add_argument('--sample_size', type=int, default=10000, help='Sample size for training')

    args = parser.parse_args()

    if args.all:
        run_all_experiments()
    elif args.dataset and args.task:
        results = run_experiment(
            args.dataset, args.task,
            n_models=args.n_models,
            sample_size=args.sample_size
        )
        print(f"\nResults: {json.dumps(results, indent=2, default=str)}")
    else:
        print("Please specify --dataset and --task, or use --all")
        parser.print_help()


if __name__ == '__main__':
    main()
