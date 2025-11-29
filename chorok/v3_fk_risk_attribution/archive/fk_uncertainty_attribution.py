"""
[ARCHIVED - 2025-11-29]
========================
원래 목적: Train vs Val FK별 uncertainty 비교 (COVID shift detection)
폐기 이유: 실험 방향 변경 - "Shift Detection"이 아닌 "Risk Attribution"으로 전환
           새로운 실험은 experiment_decomposition.py, experiment_calibration.py로 대체됨
           COVID-19 shift 감지가 아닌 "어느 FK가 불확실성에 기여하는가"로 연구 초점 변경
========================

FK Uncertainty Attribution: Train vs Val Comparison
=====================================================

Option A 구현: Distribution shift 전후 FK별 uncertainty 기여 비교

방법:
1. Train set (pre-COVID)에서 FK별 uncertainty 기여 측정
2. Val set (COVID)에서 FK별 uncertainty 기여 측정
3. 비교: 어느 FK에서 delta가 큰가?

핵심 질문: "예측이 불확실한 이유가 어떤 데이터 소스(FK relationship)에서 오는가?"
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
    # Group each column as its own "FK group" for leave-one-out analysis
    feature_groups = {}
    for col in feature_cols:
        feature_groups[col] = [col]

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


def measure_fk_uncertainty_contribution(X, y, feature_groups, feature_cols):
    """
    Measure each FK group's contribution to uncertainty via leave-one-out.

    Returns dict: {fk_name: uncertainty_contribution}
    """
    # Train full model
    model_full = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model_full.fit(X, y)
    proba_full = model_full.predict_proba(X)
    entropy_full = get_prediction_entropy(proba_full).mean()

    print(f"  Full model entropy: {entropy_full:.4f}")

    # Leave-one-out for each FK group
    contributions = {}

    for group_name, group_features in feature_groups.items():
        # Features to keep (remove this group)
        keep_features = [f for f in feature_cols if f not in group_features]

        if len(keep_features) == 0:
            continue

        X_reduced = X[keep_features]

        # Train model without this group
        model_reduced = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        model_reduced.fit(X_reduced, y)
        proba_reduced = model_reduced.predict_proba(X_reduced)
        entropy_reduced = get_prediction_entropy(proba_reduced).mean()

        # Contribution = how much entropy increases when this group is removed
        contribution = entropy_reduced - entropy_full
        contributions[group_name] = contribution

        print(f"  Without {group_name}: entropy={entropy_reduced:.4f}, delta={contribution:+.4f}")

    # Normalize contributions
    total = sum(abs(v) for v in contributions.values())
    if total > 0:
        contributions_norm = {k: abs(v) / total for k, v in contributions.items()}
    else:
        contributions_norm = contributions

    return contributions, contributions_norm, entropy_full


def run_attribution_comparison(task_name: str, sample_size: int = 10000):
    """Run FK uncertainty attribution comparison for train vs val."""

    print(f"\n{'='*60}")
    print(f"FK Uncertainty Attribution: {task_name}")
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

    # Measure uncertainty contribution on TRAIN
    print(f"\n--- TRAIN (pre-COVID) ---")
    train_contrib, train_contrib_norm, train_entropy = measure_fk_uncertainty_contribution(
        X_train, y_train, feature_groups, feature_cols
    )

    # Measure uncertainty contribution on VAL
    print(f"\n--- VAL (COVID) ---")
    val_contrib, val_contrib_norm, val_entropy = measure_fk_uncertainty_contribution(
        X_val, y_val, feature_groups, feature_cols
    )

    # Compare: delta
    print(f"\n--- COMPARISON: Delta (Val - Train) ---")
    deltas = {}
    for group in train_contrib.keys():
        if group in val_contrib:
            delta = val_contrib[group] - train_contrib[group]
            delta_pct = (val_contrib_norm.get(group, 0) - train_contrib_norm.get(group, 0)) * 100
            deltas[group] = {
                'train': train_contrib[group],
                'val': val_contrib[group],
                'delta': delta,
                'train_norm': train_contrib_norm.get(group, 0),
                'val_norm': val_contrib_norm.get(group, 0),
                'delta_pct': delta_pct
            }
            print(f"  {group}: Train={train_contrib[group]:+.4f} → Val={val_contrib[group]:+.4f} (delta={delta:+.4f})")

    # Find biggest change
    if deltas:
        max_delta_group = max(deltas.keys(), key=lambda k: abs(deltas[k]['delta']))
        print(f"\n>>> Biggest change: {max_delta_group} (delta={deltas[max_delta_group]['delta']:+.4f})")

    results = {
        'task': task_name,
        'train_entropy': train_entropy,
        'val_entropy': val_entropy,
        'entropy_change_pct': (val_entropy - train_entropy) / train_entropy * 100 if train_entropy > 0 else 0,
        'train_contributions': train_contrib,
        'val_contributions': val_contrib,
        'train_contributions_norm': train_contrib_norm,
        'val_contributions_norm': val_contrib_norm,
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
            results = run_attribution_comparison(task, args.sample_size)
            all_results[task] = results
        except Exception as e:
            print(f"Error on {task}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output_file = output_dir / "fk_uncertainty_attribution.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: FK Uncertainty Attribution Changes (Train → Val)")
    print("="*60)

    for task, results in all_results.items():
        print(f"\n{task}:")
        print(f"  Entropy: {results['train_entropy']:.4f} → {results['val_entropy']:.4f} ({results['entropy_change_pct']:+.1f}%)")

        if results['deltas']:
            max_group = max(results['deltas'].keys(), key=lambda k: abs(results['deltas'][k]['delta']))
            print(f"  Biggest FK delta: {max_group} ({results['deltas'][max_group]['delta']:+.4f})")


if __name__ == "__main__":
    main()
