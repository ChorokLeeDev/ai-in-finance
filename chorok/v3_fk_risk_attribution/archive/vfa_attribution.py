"""
[ARCHIVED - 2025-11-29]
========================
원래 목적: Variance Feature Attribution (VFA) 기반 FK attribution
폐기 이유: 실험 결과 variance가 0으로 나옴 (classification task에서)
           entropy 기반 uncertainty로 전환 (ensemble.py)
           VFA 논문 방법론은 regression에 적합, classification에서는 entropy 사용
========================

Variance Feature Attribution (VFA)
===================================

Baseline comparison: Attribute prediction variance (epistemic uncertainty) to features.

Based on: arxiv.org/abs/2312.07252v1 (2023)

Approach:
1. Train ensemble of models (5 LightGBM with different seeds)
2. For each sample, compute variance across ensemble predictions
3. Attribute variance to features via perturbation
4. Compare train vs val attribution

Author: ChorokLeeDev
Created: 2025-11-28
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import entropy, spearmanr

from relbench.datasets import get_dataset
from relbench.tasks import get_task

warnings.filterwarnings('ignore')

VAL_TIMESTAMP = pd.Timestamp("2020-02-01")
TEST_TIMESTAMP = pd.Timestamp("2020-07-01")


def load_task_data(task_name: str):
    """Load task and split by temporal periods."""
    task = get_task("rel-salt", task_name, download=False)
    dataset = get_dataset("rel-salt", download=False)
    db = dataset.get_db(upto_test_timestamp=False)

    entity_table = db.table_dict[task.entity_table]
    df = entity_table.df.copy()

    # Split by temporal periods
    train_df = df[df['CREATIONTIMESTAMP'] < VAL_TIMESTAMP].copy()
    val_df = df[(df['CREATIONTIMESTAMP'] >= VAL_TIMESTAMP) &
                (df['CREATIONTIMESTAMP'] < TEST_TIMESTAMP)].copy()

    return train_df, val_df, task.target_col, entity_table


def prepare_features(df, entity_table, target_col, sample_size=10000):
    """Prepare features and identify FK groups."""
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    # Exclude non-feature columns
    exclude = {'CREATIONTIMESTAMP', target_col, entity_table.pkey_col}

    # Get feature columns
    feature_cols = [c for c in df.columns if c not in exclude]

    # For SALT, each column represents a different data source/FK relationship
    feature_groups = {col: [col] for col in feature_cols}

    # Prepare numeric features
    X_df = df[feature_cols].copy()
    for col in X_df.columns:
        if X_df[col].dtype == 'object':
            X_df[col] = X_df[col].astype('category').cat.codes
        X_df[col] = X_df[col].fillna(-999)

    y = df[target_col].copy()
    if y.dtype == 'object':
        y = y.astype('category').cat.codes

    return X_df, y, feature_groups, feature_cols


def train_ensemble(X, y, n_models=5):
    """Train an ensemble of LightGBM models with different random seeds."""
    models = []
    for seed in range(n_models):
        model = lgb.LGBMClassifier(
            n_estimators=100,
            random_state=42 + seed,
            verbose=-1,
            subsample=0.8,  # Add randomness
            colsample_bytree=0.8
        )
        model.fit(X, y)
        models.append(model)
    return models


def get_ensemble_variance(models: List, X) -> np.ndarray:
    """
    Compute variance of predictions across ensemble.

    For classification: variance of predicted probabilities.
    Returns mean variance across classes.
    """
    all_proba = []
    for model in models:
        proba = model.predict_proba(X)
        all_proba.append(proba)

    # Stack: (n_models, n_samples, n_classes)
    proba_stack = np.array(all_proba)

    # Variance across models for each (sample, class)
    variance_per_class = np.var(proba_stack, axis=0)  # (n_samples, n_classes)

    # Mean variance across classes per sample
    mean_variance = np.mean(variance_per_class, axis=1)  # (n_samples,)

    return mean_variance


def compute_vfa_attribution(X, y, feature_cols, n_models=5, n_repeats=3):
    """
    Compute Variance Feature Attribution.

    For each feature:
    1. Compute baseline ensemble variance
    2. Permute feature values
    3. Measure variance change
    4. Feature that reduces variance most when permuted = contributes most to variance

    Note: If permuting a feature INCREASES variance, that feature was stabilizing.
    If permuting a feature DECREASES variance, that feature was causing variance.
    """
    # Train ensemble
    print(f"  Training ensemble of {n_models} models...")
    models = train_ensemble(X, y, n_models)

    # Baseline variance
    baseline_variance = get_ensemble_variance(models, X)
    baseline_mean = baseline_variance.mean()

    print(f"  Baseline ensemble variance: {baseline_mean:.6f}")

    # VFA for each feature
    attribution = {}

    for col in feature_cols:
        deltas = []

        for i in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col].values)

            permuted_variance = get_ensemble_variance(models, X_permuted)
            permuted_mean = permuted_variance.mean()

            # Delta: positive means permuting increases variance
            # (feature was reducing variance / stabilizing)
            delta = permuted_mean - baseline_mean
            deltas.append(delta)

        mean_delta = np.mean(deltas)
        attribution[col] = mean_delta

        print(f"  Permute {col}: variance_delta={mean_delta:+.6f}")

    # Normalize by absolute values
    total = sum(abs(v) for v in attribution.values())
    attribution_norm = {k: abs(v) / total if total > 0 else 0
                        for k, v in attribution.items()}

    return attribution, attribution_norm, baseline_mean


def run_vfa_comparison(task_name: str, sample_size: int = 10000, n_models: int = 5):
    """Run VFA comparison for train vs val."""

    print(f"\n{'='*60}")
    print(f"VFA (Variance Feature Attribution): {task_name}")
    print(f"{'='*60}")

    # Load data
    train_df, val_df, target_col, entity_table = load_task_data(task_name)

    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples: {len(val_df):,}")
    print(f"Target: {target_col}")

    # Prepare features
    X_train, y_train, feature_groups, feature_cols = prepare_features(
        train_df, entity_table, target_col, sample_size
    )
    X_val, y_val, _, _ = prepare_features(
        val_df, entity_table, target_col, sample_size
    )

    print(f"\nFeature groups: {list(feature_groups.keys())}")

    # VFA on TRAIN
    print(f"\n--- TRAIN (pre-COVID) ---")
    train_attr, train_attr_norm, train_var = compute_vfa_attribution(
        X_train, y_train, feature_cols, n_models
    )

    # VFA on VAL
    print(f"\n--- VAL (COVID) ---")
    val_attr, val_attr_norm, val_var = compute_vfa_attribution(
        X_val, y_val, feature_cols, n_models
    )

    # Compare: delta
    print(f"\n--- COMPARISON: Delta (Val - Train) ---")
    deltas = {}
    for col in feature_cols:
        delta = val_attr.get(col, 0) - train_attr.get(col, 0)
        delta_norm = val_attr_norm.get(col, 0) - train_attr_norm.get(col, 0)
        deltas[col] = {
            'train_raw': train_attr.get(col, 0),
            'val_raw': val_attr.get(col, 0),
            'delta_raw': delta,
            'train_norm': train_attr_norm.get(col, 0),
            'val_norm': val_attr_norm.get(col, 0),
            'delta_norm': delta_norm
        }
        print(f"  {col}: {train_attr.get(col, 0):+.6f} → {val_attr.get(col, 0):+.6f} (delta={delta:+.6f})")

    # Find biggest change
    if deltas:
        max_delta_col = max(deltas.keys(), key=lambda k: abs(deltas[k]['delta_raw']))
        print(f"\n>>> Biggest VFA change: {max_delta_col} (delta={deltas[max_delta_col]['delta_raw']:+.6f})")

    results = {
        'task': task_name,
        'method': 'VFA',
        'n_models': n_models,
        'train_variance': train_var,
        'val_variance': val_var,
        'variance_change_pct': (val_var - train_var) / train_var * 100 if train_var > 0 else 0,
        'train_attribution': train_attr,
        'train_attribution_norm': train_attr_norm,
        'val_attribution': val_attr,
        'val_attribution_norm': val_attr_norm,
        'deltas': deltas
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sales-group")
    parser.add_argument("--sample_size", type=int, default=10000)
    parser.add_argument("--n_models", type=int, default=5, help="Number of models in ensemble")
    parser.add_argument("--all_tasks", action="store_true", help="Run all 8 SALT tasks")
    args = parser.parse_args()

    output_dir = Path("chorok/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_tasks:
        tasks = [
            "item-plant", "item-shippoint", "item-incoterms",
            "sales-office", "sales-group", "sales-payterms",
            "sales-shipcond", "sales-incoterms"
        ]
    else:
        tasks = [args.task]

    all_results = {}
    for task in tasks:
        try:
            results = run_vfa_comparison(task, args.sample_size, args.n_models)
            all_results[task] = results
        except Exception as e:
            print(f"Error on {task}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_file = output_dir / "vfa_attribution.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: VFA Attribution Changes (Train → Val)")
    print("="*60)

    for task, results in all_results.items():
        print(f"\n{task}:")
        print(f"  Ensemble variance: {results['train_variance']:.6f} → {results['val_variance']:.6f} ({results['variance_change_pct']:+.1f}%)")
        if results['deltas']:
            max_col = max(results['deltas'].keys(), key=lambda k: abs(results['deltas'][k]['delta_raw']))
            print(f"  Biggest delta: {max_col} ({results['deltas'][max_col]['delta_raw']:+.6f})")


if __name__ == "__main__":
    main()
