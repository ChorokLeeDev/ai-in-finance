"""
Permutation-based FK Uncertainty Attribution
=============================================

Baseline comparison: Use permutation importance to attribute uncertainty.

Approach:
1. Train LightGBM classifier
2. For each FK: permute its values, measure entropy change
3. Higher entropy increase = FK contributes more to model certainty
4. Compare train vs val attribution

This differs from our leave-one-out method:
- Leave-one-out: Remove FK entirely, retrain model
- Permutation: Keep model fixed, break FK-target relationship

Author: ChorokLeeDev
Created: 2025-11-28
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
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


def get_prediction_entropy(proba):
    """Calculate entropy of prediction probabilities."""
    proba = np.clip(proba, 1e-10, 1 - 1e-10)
    return entropy(proba, axis=1)


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


def compute_permutation_attribution(X, y, feature_cols, n_repeats=5):
    """
    Compute permutation-based uncertainty attribution.

    For each feature:
    1. Compute baseline entropy (model certainty)
    2. Permute feature values (break relationship)
    3. Measure entropy change
    4. Feature that increases entropy most = most important for certainty

    Note: Positive delta means removing this feature increases uncertainty
    (i.e., the feature contributes to model certainty).
    """
    # Train model
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X, y)

    # Baseline entropy
    baseline_proba = model.predict_proba(X)
    baseline_entropy = get_prediction_entropy(baseline_proba).mean()

    print(f"  Baseline entropy: {baseline_entropy:.4f}")

    # Permutation importance for each feature
    attribution = {}

    for col in feature_cols:
        deltas = []

        for i in range(n_repeats):
            X_permuted = X.copy()
            # Permute this column
            X_permuted[col] = np.random.permutation(X_permuted[col].values)

            # Compute new entropy
            permuted_proba = model.predict_proba(X_permuted)
            permuted_entropy = get_prediction_entropy(permuted_proba).mean()

            # Delta: how much entropy increased
            delta = permuted_entropy - baseline_entropy
            deltas.append(delta)

        mean_delta = np.mean(deltas)
        attribution[col] = mean_delta

        print(f"  Permute {col}: entropy_delta={mean_delta:+.4f}")

    # Normalize (use absolute values for importance ranking)
    total = sum(abs(v) for v in attribution.values())
    attribution_norm = {k: abs(v) / total if total > 0 else 0
                        for k, v in attribution.items()}

    return attribution, attribution_norm, baseline_entropy


def run_permutation_comparison(task_name: str, sample_size: int = 10000):
    """Run permutation attribution comparison for train vs val."""

    print(f"\n{'='*60}")
    print(f"Permutation Uncertainty Attribution: {task_name}")
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

    # Permutation attribution on TRAIN
    print(f"\n--- TRAIN (pre-COVID) ---")
    train_attr, train_attr_norm, train_entropy = compute_permutation_attribution(
        X_train, y_train, feature_cols
    )

    # Permutation attribution on VAL
    print(f"\n--- VAL (COVID) ---")
    val_attr, val_attr_norm, val_entropy = compute_permutation_attribution(
        X_val, y_val, feature_cols
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
        print(f"  {col}: {train_attr.get(col, 0):+.4f} → {val_attr.get(col, 0):+.4f} (delta={delta:+.4f})")

    # Find biggest change
    if deltas:
        max_delta_col = max(deltas.keys(), key=lambda k: abs(deltas[k]['delta_raw']))
        print(f"\n>>> Biggest permutation change: {max_delta_col} (delta={deltas[max_delta_col]['delta_raw']:+.4f})")

    results = {
        'task': task_name,
        'method': 'Permutation',
        'train_entropy': train_entropy,
        'val_entropy': val_entropy,
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
            results = run_permutation_comparison(task, args.sample_size)
            all_results[task] = results
        except Exception as e:
            print(f"Error on {task}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_file = output_dir / "permutation_attribution.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Permutation Attribution Changes (Train → Val)")
    print("="*60)

    for task, results in all_results.items():
        print(f"\n{task}:")
        print(f"  Entropy: {results['train_entropy']:.4f} → {results['val_entropy']:.4f}")
        if results['deltas']:
            max_col = max(results['deltas'].keys(), key=lambda k: abs(results['deltas'][k]['delta_raw']))
            print(f"  Biggest delta: {max_col} ({results['deltas'][max_col]['delta_raw']:+.4f})")


if __name__ == "__main__":
    main()
