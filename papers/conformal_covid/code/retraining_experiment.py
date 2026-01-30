"""
Retraining Experiment for Conformal Prediction

Tests whether coverage can be restored through periodic retraining after
distribution shift (COVID-19).

Research Question:
    Can retraining restore coverage? How often should we retrain?

Experimental Setup:
    - Test task: sales-shipcond (most severe degradation, 93% drop)
    - Test 4 retraining frequencies:
        1. No retrain (baseline): Train once on pre-COVID data
        2. Monthly: Retrain every month (12/year)
        3. Quarterly: Retrain every 3 months (4/year)
        4. Semi-annual: Retrain every 6 months (2/year)
    - Track coverage over 11 months (Feb-Dec 2020)

Usage:
    python retraining_experiment.py --freq none
    python retraining_experiment.py --freq 1M
    python retraining_experiment.py --freq 3M
    python retraining_experiment.py --freq 6M

Output:
    results/retraining/retrain_{freq}_{task}.pkl
    results/retraining/retrain_{freq}_{task}.json

Author: UAI 2026 Conformal COVID Paper
Date: 2025-12-26
"""

import argparse
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

ALPHA = 0.1  # 90% target coverage


class ConformalClassifier:
    """Adaptive Prediction Sets (APS) for classification."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.quantile = None

    def _compute_scores(self, probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute nonconformity scores."""
        n = len(y_true)
        scores = np.zeros(n)
        for i in range(n):
            sorted_idx = np.argsort(probs[i])[::-1]
            cumsum = 0
            for j, idx in enumerate(sorted_idx):
                cumsum += probs[i][idx]
                if idx == y_true[i]:
                    scores[i] = cumsum
                    break
        return scores

    def calibrate(self, probs: np.ndarray, y_true: np.ndarray):
        """Calibrate on validation data."""
        scores = self._compute_scores(probs, y_true)
        n = len(scores)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.quantile = np.quantile(scores, q_level)
        return self

    def predict_sets(self, probs: np.ndarray) -> List[set]:
        """Generate prediction sets."""
        sets = []
        for i in range(len(probs)):
            sorted_idx = np.argsort(probs[i])[::-1]
            pred_set = set()
            cumsum = 0
            for idx in sorted_idx:
                pred_set.add(idx)
                cumsum += probs[i][idx]
                if cumsum >= self.quantile:
                    break
            sets.append(pred_set)
        return sets

    def evaluate_coverage(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        """Compute empirical coverage."""
        pred_sets = self.predict_sets(probs)
        coverage = sum(1 for i, s in enumerate(pred_sets) if y_true[i] in s) / len(pred_sets)
        return coverage


def split_by_month(df: pd.DataFrame, timestamp_col: str = 'CREATIONTIMESTAMP') -> List[Tuple[str, pd.DataFrame]]:
    """
    Split dataframe into monthly chunks.

    Returns:
        List of (year_month, data) tuples sorted chronologically
    """
    df = df.copy()
    df['year_month'] = pd.to_datetime(df[timestamp_col]).dt.to_period('M')

    months = []
    for period in sorted(df['year_month'].unique()):
        month_data = df[df['year_month'] == period].copy()
        month_data = month_data.drop(columns=['year_month'])
        months.append((str(period), month_data))

    return months


def preprocess_data(
    train_df: pd.DataFrame,
    target_col: str,
    label_encoders: Dict = None,
    target_le: LabelEncoder = None
) -> Tuple[np.ndarray, np.ndarray, Dict, LabelEncoder]:
    """
    Preprocess features for LightGBM.

    Returns:
        X: Feature matrix
        y: Encoded labels
        label_encoders: Dict of label encoders (updated if provided)
        target_le: Target label encoder (updated if provided)
    """
    # Exclude columns
    exclude_cols = [target_col, 'CREATIONTIMESTAMP']
    id_cols = [c for c in train_df.columns if c.endswith('_id') or c.endswith('Id') or c == 'ID']
    exclude_cols.extend(id_cols)
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    # Initialize encoders if not provided
    if label_encoders is None:
        label_encoders = {}

    # Note: target_le should be provided pre-fitted on all data
    # If not provided, fit on current data (but this may cause issues with new classes)
    if target_le is None:
        target_le = LabelEncoder()
        target_le.fit(train_df[target_col].values)

    # Encode categorical features
    X = train_df[feature_cols].copy()
    for col in feature_cols:
        if train_df[col].dtype == 'object' or train_df[col].dtype.name == 'category':
            if col not in label_encoders:
                le = LabelEncoder()
                le.fit(train_df[col].astype(str).fillna('__MISSING__'))
                label_encoders[col] = le
            else:
                le = label_encoders[col]

            X[col] = X[col].astype(str).fillna('__MISSING__')
            X[col] = X[col].apply(lambda x: x if x in le.classes_ else '__MISSING__')
            if '__MISSING__' not in le.classes_:
                le.classes_ = np.append(le.classes_, '__MISSING__')
            X[col] = le.transform(X[col])

        # Convert to numeric
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-999)

    # Encode target (handle unseen classes by adding them to encoder)
    target_values = train_df[target_col].values
    for val in target_values:
        if val not in target_le.classes_:
            # Add new class to encoder
            target_le.classes_ = np.append(target_le.classes_, val)

    y = target_le.transform(target_values)

    return X.values.astype(np.float32), y, label_encoders, target_le


def train_model_with_conformal(
    train_data: pd.DataFrame,
    target_col: str,
    num_classes: int,
    label_encoders: Dict = None,
    target_le: LabelEncoder = None,
    seed: int = 42
) -> Tuple[lgb.Booster, ConformalClassifier, Dict, LabelEncoder]:
    """
    Train LightGBM model and calibrate conformal predictor.

    Args:
        train_data: Training dataframe
        target_col: Name of target column
        num_classes: Number of classes
        label_encoders: Existing label encoders (or None)
        target_le: Existing target encoder (or None)
        seed: Random seed

    Returns:
        model: Trained LightGBM model
        conformal: Calibrated conformal predictor
        label_encoders: Updated label encoders
        target_le: Updated target encoder
    """
    # Split train data for calibration
    n = len(train_data)
    n_calib = n // 2

    np.random.seed(seed)
    idx = np.random.permutation(n)
    fit_data = train_data.iloc[idx[n_calib:]].copy()
    calib_data = train_data.iloc[idx[:n_calib]].copy()

    # Preprocess
    X_fit, y_fit, label_encoders, target_le = preprocess_data(
        fit_data, target_col, label_encoders, target_le
    )
    X_calib, y_calib, _, _ = preprocess_data(
        calib_data, target_col, label_encoders, target_le
    )

    # Train LightGBM
    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': seed,
        'n_jobs': -1,
    }

    train_ds = lgb.Dataset(X_fit, label=y_fit)
    model = lgb.train(params, train_ds, num_boost_round=100)  # Faster training

    # Calibrate conformal predictor
    calib_probs = model.predict(X_calib)
    conformal = ConformalClassifier(alpha=ALPHA)
    conformal.calibrate(calib_probs, y_calib)

    return model, conformal, label_encoders, target_le


def compute_mean_jaccard(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_col: str
) -> float:
    """Compute mean Jaccard similarity across all features."""
    exclude_cols = [target_col, 'CREATIONTIMESTAMP', 'timestamp']
    id_cols = [c for c in train_data.columns if c.endswith('_id') or c.endswith('Id') or c == 'ID']
    exclude_cols.extend(id_cols)
    feature_cols = [c for c in train_data.columns if c not in exclude_cols]

    jaccards = []
    for col in feature_cols:
        train_vals = train_data[col].dropna().unique()
        test_vals = test_data[col].dropna().unique()

        if len(train_vals) > 0 and len(test_vals) > 0:
            train_set = set(train_vals)
            test_set = set(test_vals)
            intersection = len(train_set & test_set)
            union = len(train_set | test_set)
            if union > 0:
                jaccards.append(intersection / union)

    return np.mean(jaccards) if jaccards else 0.0


def run_retraining_experiment(
    dataset_name: str,
    task_name: str,
    retrain_freq: str = '1M',
    seed: int = 42
) -> List[Dict]:
    """
    Run retraining experiment with specified frequency.

    Args:
        dataset_name: Dataset name (e.g., 'rel-salt')
        task_name: Task name (e.g., 'sales-shipcond')
        retrain_freq: '1M', '3M', '6M', or 'none'
        seed: Random seed

    Returns:
        results: List of dicts with monthly results
    """
    from relbench.tasks import get_task

    print(f"Loading task {dataset_name}/{task_name}...")
    task = get_task(dataset_name, task_name, download=False)

    # Get tables
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    # Get entity data
    dataset = task.dataset
    entity_table = dataset.get_db().table_dict[task.entity_table]
    entity_df = entity_table.df.copy()

    # Merge tables with entity data
    def merge_with_entity(table):
        df = table.df.copy()
        entity_df_copy = entity_df.copy()
        left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
        entity_df_copy = entity_df_copy.astype({entity_table.pkey_col: df[left_entity].dtype})

        for col in set(entity_df_copy.columns).intersection(set(df.columns)):
            if col != entity_table.pkey_col:
                entity_df_copy = entity_df_copy.drop(columns=[col])

        merged = df.merge(
            entity_df_copy, how="left",
            left_on=left_entity, right_on=entity_table.pkey_col,
        )
        return merged

    train_df = merge_with_entity(train_table)
    val_df = merge_with_entity(val_table)
    test_df = merge_with_entity(test_table)

    print(f"  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")

    # Split validation and test into months
    val_months = split_by_month(val_df)
    test_months = split_by_month(test_df)
    all_months = val_months + test_months

    print(f"  Months to evaluate: {len(all_months)}")
    for i, (month_str, _) in enumerate(all_months):
        print(f"    {i+1}. {month_str}")

    # Determine retrain schedule
    if retrain_freq == 'none':
        retrain_schedule = []
    elif retrain_freq == '1M':
        retrain_schedule = list(range(len(all_months)))
    elif retrain_freq == '3M':
        retrain_schedule = list(range(0, len(all_months), 3))
    elif retrain_freq == '6M':
        retrain_schedule = list(range(0, len(all_months), 6))
    else:
        raise ValueError(f"Invalid frequency: {retrain_freq}")

    print(f"\nRetrain schedule ({retrain_freq}): {retrain_schedule}")

    # Initialize
    target_col = task.target_col
    num_classes = len(train_df[target_col].unique())

    # FIT TARGET ENCODER ON ALL DATA (train + val + test) to handle new classes
    target_le = LabelEncoder()
    all_targets = pd.concat([
        train_df[target_col],
        val_df[target_col],
        test_df[target_col]
    ])
    target_le.fit(all_targets)
    num_classes = len(target_le.classes_)  # Update with actual number
    print(f"  Total classes (all splits): {num_classes}")

    current_model = None
    current_conformal = None
    label_encoders = None
    current_train_data = train_df.copy()

    results = []

    print(f"\n{'='*80}")
    print(f"Starting retraining experiment: {retrain_freq}")
    print(f"{'='*80}\n")

    for i, (month_str, month_data) in enumerate(all_months):
        print(f"Month {i+1}/{len(all_months)}: {month_str} ({len(month_data)} samples)")

        # Retrain if scheduled
        if i in retrain_schedule or current_model is None:
            print(f"  → Retraining model...")

            current_model, current_conformal, label_encoders, target_le = \
                train_model_with_conformal(
                    current_train_data, target_col, num_classes,
                    label_encoders, target_le, seed
                )
            retrained = True
        else:
            retrained = False

        # Evaluate on current month
        X_test, y_test, _, _ = preprocess_data(
            month_data, target_col, label_encoders, target_le
        )
        test_probs = current_model.predict(X_test)
        coverage = current_conformal.evaluate_coverage(test_probs, y_test)

        # Compute feature freshness (Jaccard)
        jaccard = compute_mean_jaccard(current_train_data, month_data, target_col)

        print(f"  Coverage: {coverage*100:.1f}%, Jaccard: {jaccard:.3f}, Retrained: {retrained}")

        results.append({
            'month': i,
            'month_str': month_str,
            'coverage': coverage,
            'jaccard': jaccard,
            'retrained': retrained,
            'n_samples': len(month_data),
        })

        # Add current month to training data (for rolling window)
        current_train_data = pd.concat([current_train_data, month_data], ignore_index=True)

        # Keep only last 12 months (rolling window)
        if len(current_train_data) > len(train_df) * 12:
            current_train_data = current_train_data.iloc[-len(train_df)*12:].copy()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Retraining experiment for conformal prediction"
    )
    parser.add_argument('--dataset', type=str, default='rel-salt',
                       help='Dataset name')
    parser.add_argument('--task', type=str, default='sales-shipcond',
                       help='Task name')
    parser.add_argument('--freq', type=str, required=True,
                       choices=['none', '1M', '3M', '6M'],
                       help='Retraining frequency')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str,
                       default='papers/conformal_covid/results/retraining',
                       help='Output directory')
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"RETRAINING EXPERIMENT")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Frequency: {args.freq}")
    print(f"Seed: {args.seed}")
    print(f"{'='*80}\n")

    # Run experiment
    results = run_retraining_experiment(
        args.dataset, args.task, args.freq, args.seed
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save pickle
    output_file_pkl = output_dir / f'retrain_{args.freq}_{args.task}.pkl'
    with open(output_file_pkl, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Saved: {output_file_pkl}")

    # Save JSON
    output_file_json = output_dir / f'retrain_{args.freq}_{args.task}.json'
    with open(output_file_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved: {output_file_json}")

    # Summary statistics
    coverages = [r['coverage'] for r in results]
    jaccards = [r['jaccard'] for r in results]
    n_retrains = sum(r['retrained'] for r in results)

    print(f"\n{'='*80}")
    print(f"SUMMARY ({args.freq})")
    print(f"{'='*80}")
    print(f"Mean coverage:  {np.mean(coverages)*100:.1f}%")
    print(f"Min coverage:   {np.min(coverages)*100:.1f}%")
    print(f"Max coverage:   {np.max(coverages)*100:.1f}%")
    print(f"Std coverage:   {np.std(coverages)*100:.1f}%")
    print(f"Mean Jaccard:   {np.mean(jaccards):.3f}")
    print(f"Retrains:       {n_retrains}")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    main()
