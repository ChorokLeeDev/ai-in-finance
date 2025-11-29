"""
[ARCHIVED - 2025-11-29]
========================
원래 목적: SHAP 기반 FK별 uncertainty attribution (baseline 비교)
폐기 이유: 실험 방향 변경 - "여러 방법 비교"에서 "Risk Attribution 프레임워크 검증"으로 전환
           SHAP은 feature importance를 측정하며, uncertainty attribution과 다름
           새로운 프레임워크: hierarchical_attribution.py (3-level drill-down)
========================

SHAP-based FK Uncertainty Attribution
======================================

Baseline comparison: Use SHAP to attribute model uncertainty to FK groups.

Approach:
1. Train LightGBM classifier
2. Compute SHAP values for prediction probabilities
3. Aggregate SHAP importance by FK group
4. Compare train vs val SHAP attribution

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

try:
    import shap
except ImportError:
    print("Please install shap: pip install shap")
    raise

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


def compute_shap_attribution(X, y, feature_cols):
    """
    Compute SHAP-based uncertainty attribution.

    For classification, we use SHAP values on class probabilities.
    Uncertainty attribution = variance of SHAP across classes (higher = more uncertain contribution).
    """
    # Train model
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X, y)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Get SHAP values for all classes
    shap_values = explainer.shap_values(X)

    # Convert to numpy array for consistent handling
    # shap_values can be: list of arrays, 3D array, or 2D array
    if isinstance(shap_values, list):
        shap_array = np.array(shap_values)  # shape: (n_classes, n_samples, n_features)
    else:
        shap_array = np.array(shap_values)

    # Handle different array shapes
    if shap_array.ndim == 3:
        # Multi-class: (n_classes, n_samples, n_features)
        # Mean absolute SHAP per feature (averaged over samples and classes)
        mean_abs_shap = np.mean(np.abs(shap_array), axis=(0, 1))  # shape: (n_features,)
        # Variance of SHAP across classes (uncertainty indicator)
        shap_variance = np.var(shap_array, axis=0).mean(axis=0)  # shape: (n_features,)
    elif shap_array.ndim == 2:
        # Binary classification or single output: (n_samples, n_features)
        mean_abs_shap = np.mean(np.abs(shap_array), axis=0)
        shap_variance = np.var(shap_array, axis=0)
    else:
        # Unexpected shape, flatten appropriately
        mean_abs_shap = np.mean(np.abs(shap_array.reshape(-1, len(feature_cols))), axis=0)
        shap_variance = np.var(shap_array.reshape(-1, len(feature_cols)), axis=0)

    # Ensure 1D arrays
    mean_abs_shap = np.atleast_1d(mean_abs_shap).flatten()
    shap_variance = np.atleast_1d(shap_variance).flatten()

    # Create attribution dict
    attribution = {}
    attribution_variance = {}

    for i, col in enumerate(feature_cols):
        attribution[col] = float(mean_abs_shap[i]) if i < len(mean_abs_shap) else 0.0
        attribution_variance[col] = float(shap_variance[i]) if i < len(shap_variance) else 0.0

    # Normalize
    total_shap = sum(attribution.values())
    total_var = sum(attribution_variance.values())

    attribution_norm = {k: v / total_shap if total_shap > 0 else 0
                        for k, v in attribution.items()}
    variance_norm = {k: v / total_var if total_var > 0 else 0
                     for k, v in attribution_variance.items()}

    return attribution, attribution_norm, attribution_variance, variance_norm


def run_shap_comparison(task_name: str, sample_size: int = 10000):
    """Run SHAP attribution comparison for train vs val."""

    print(f"\n{'='*60}")
    print(f"SHAP Uncertainty Attribution: {task_name}")
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

    # SHAP attribution on TRAIN
    print(f"\n--- TRAIN (pre-COVID) ---")
    train_attr, train_attr_norm, train_var, train_var_norm = compute_shap_attribution(
        X_train, y_train, feature_cols
    )

    print("SHAP importance (normalized):")
    for k, v in sorted(train_attr_norm.items(), key=lambda x: -x[1])[:5]:
        print(f"  {k}: {v:.4f}")

    # SHAP attribution on VAL
    print(f"\n--- VAL (COVID) ---")
    val_attr, val_attr_norm, val_var, val_var_norm = compute_shap_attribution(
        X_val, y_val, feature_cols
    )

    print("SHAP importance (normalized):")
    for k, v in sorted(val_attr_norm.items(), key=lambda x: -x[1])[:5]:
        print(f"  {k}: {v:.4f}")

    # Compare: delta
    print(f"\n--- COMPARISON: Delta (Val - Train) ---")
    deltas = {}
    for col in feature_cols:
        delta = val_attr_norm.get(col, 0) - train_attr_norm.get(col, 0)
        deltas[col] = {
            'train_shap': train_attr_norm.get(col, 0),
            'val_shap': val_attr_norm.get(col, 0),
            'delta': delta,
            'train_var': train_var_norm.get(col, 0),
            'val_var': val_var_norm.get(col, 0),
            'var_delta': val_var_norm.get(col, 0) - train_var_norm.get(col, 0)
        }
        print(f"  {col}: {train_attr_norm.get(col, 0):.4f} → {val_attr_norm.get(col, 0):.4f} (delta={delta:+.4f})")

    # Find biggest change
    if deltas:
        max_delta_col = max(deltas.keys(), key=lambda k: abs(deltas[k]['delta']))
        print(f"\n>>> Biggest SHAP change: {max_delta_col} (delta={deltas[max_delta_col]['delta']:+.4f})")

    results = {
        'task': task_name,
        'method': 'SHAP',
        'train_attribution': train_attr_norm,
        'val_attribution': val_attr_norm,
        'train_variance': train_var_norm,
        'val_variance': val_var_norm,
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
            results = run_shap_comparison(task, args.sample_size)
            all_results[task] = results
        except Exception as e:
            print(f"Error on {task}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_file = output_dir / "shap_attribution.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: SHAP Attribution Changes (Train → Val)")
    print("="*60)

    for task, results in all_results.items():
        print(f"\n{task}:")
        if results['deltas']:
            max_col = max(results['deltas'].keys(), key=lambda k: abs(results['deltas'][k]['delta']))
            print(f"  Biggest delta: {max_col} ({results['deltas'][max_col]['delta']:+.4f})")


if __name__ == "__main__":
    main()
