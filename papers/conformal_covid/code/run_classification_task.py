#!/usr/bin/env python3
"""
Run Classification Task with Adaptive Prediction Sets (APS)

Implements conformal prediction for binary classification tasks using APS.
Follows same pattern as cqr_regression.py for consistency.

Tasks:
- rel-trial/study-outcome (binary classification)
- rel-f1/driver-dnf (binary classification)

Usage:
    python run_classification_task.py --dataset rel-trial --task study-outcome --num_seeds 50
    python run_classification_task.py --dataset rel-f1 --task driver-dnf --num_seeds 50
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Add relbench to path
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance')

# Configuration
ALPHA = 0.1  # 90% target coverage
SAMPLE_SIZE = 30000


class APS:
    """Adaptive Prediction Sets for classification."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.model = None
        self.quantile = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_cal: np.ndarray, y_cal: np.ndarray, seed: int = 42):
        """
        Fit APS model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_cal: Calibration features
            y_cal: Calibration labels
            seed: Random seed
        """
        # Train binary classifier
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
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

        train_data = lgb.Dataset(X_train, label=y_train)
        self.model = lgb.train(params, train_data, num_boost_round=500)

        # Calibrate: compute quantile for prediction sets
        probs_cal = self.model.predict(X_cal)

        # For each calibration example, compute conformity score
        # Score = 1 - P(true_class)
        scores = 1 - np.where(y_cal == 1, probs_cal, 1 - probs_cal)

        # Compute quantile
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(scores, min(q_level, 1.0))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict conformal prediction sets.

        Returns:
            Array of prediction sets (list of classes in set for each example)
        """
        probs = self.model.predict(X)

        # Prediction set includes class if 1 - P(class) <= quantile
        # Equivalently, P(class) >= 1 - quantile
        threshold = 1 - self.quantile

        prediction_sets = []
        for p in probs:
            pred_set = []
            # Class 1
            if p >= threshold:
                pred_set.append(1)
            # Class 0
            if (1 - p) >= threshold:
                pred_set.append(0)

            # Handle empty sets (shouldn't happen with proper calibration)
            if len(pred_set) == 0:
                pred_set = [0, 1]  # Include both classes

            prediction_sets.append(pred_set)

        return np.array(prediction_sets, dtype=object)

    def coverage(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute empirical coverage."""
        pred_sets = self.predict(X)
        covered = np.array([y[i] in pred_sets[i] for i in range(len(y))])
        return np.mean(covered)

    def avg_set_size(self, X: np.ndarray) -> float:
        """Compute average prediction set size."""
        pred_sets = self.predict(X)
        sizes = np.array([len(s) for s in pred_sets])
        return np.mean(sizes)


def compute_jaccard(train_values: np.ndarray, test_values: np.ndarray) -> float:
    """
    Compute Jaccard similarity for categorical features.

    Args:
        train_values: Values from training set
        test_values: Values from test set

    Returns:
        Jaccard similarity (0 to 1)
    """
    train_set = set(train_values)
    test_set = set(test_values)

    if len(train_set) == 0 and len(test_set) == 0:
        return 1.0

    intersection = len(train_set & test_set)
    union = len(train_set | test_set)

    return intersection / union if union > 0 else 0.0


def run_single_seed(task, task_name: str, seed: int,
                   sample_size: int = SAMPLE_SIZE) -> Dict:
    """
    Run APS for a single seed.

    Args:
        task: RelBench task object
        task_name: Task name
        seed: Random seed
        sample_size: Training sample size

    Returns:
        Results dictionary
    """
    # Load data
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    dataset = task.dataset
    entity_table = dataset.get_db().table_dict[task.entity_table]
    entity_df = entity_table.df.copy()

    # Merge entity features (same as CQR)
    dfs = {}
    for split, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        entity_df_copy = entity_df.copy()
        left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
        entity_df_copy = entity_df_copy.astype({entity_table.pkey_col: table.df[left_entity].dtype})

        # Remove duplicate columns
        for col in set(entity_df_copy.columns).intersection(set(table.df.columns)):
            if col != entity_table.pkey_col:
                entity_df_copy = entity_df_copy.drop(columns=[col])

        dfs[split] = table.df.merge(
            entity_df_copy,
            how="left",
            left_on=left_entity,
            right_on=entity_table.pkey_col,
        )

    # Subsample training data
    if sample_size and sample_size < len(dfs["train"]):
        np.random.seed(seed)
        idx = np.random.permutation(len(dfs["train"]))[:sample_size]
        dfs["train"] = dfs["train"].iloc[idx].copy()

    target_col = task.target_col

    # Feature engineering
    all_data = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)
    exclude_cols = [target_col, 'CREATIONTIMESTAMP', 'timestamp']

    # Exclude ID columns
    id_cols = [c for c in all_data.columns if c.endswith('_id') or c.endswith('Id') or c == 'ID']
    exclude_cols.extend(id_cols)

    feature_cols = [c for c in all_data.columns if c not in exclude_cols]

    # Encode categorical features
    label_encoders = {}
    for col in feature_cols:
        if all_data[col].dtype == 'object' or all_data[col].dtype.name == 'category':
            le = LabelEncoder()
            all_data[col] = all_data[col].astype(str).fillna('__MISSING__')
            le.fit(all_data[col])
            label_encoders[col] = le

    # Prepare datasets
    X_data, y_data = {}, {}
    for split, df in dfs.items():
        X = df[feature_cols].copy()

        # Apply label encoding
        for col, le in label_encoders.items():
            X[col] = X[col].astype(str).fillna('__MISSING__')
            X[col] = X[col].apply(lambda x: x if x in le.classes_ else '__MISSING__')
            if '__MISSING__' not in le.classes_:
                le.classes_ = np.append(le.classes_, '__MISSING__')
            X[col] = le.transform(X[col])

        # Convert to numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-999)

        X_data[split] = X.values.astype(np.float32)
        y_data[split] = df[target_col].values.astype(np.int32)

    # Split validation for calibration
    n_val = len(X_data['val'])
    n_calib = n_val // 2

    X_train = X_data['train']
    y_train = y_data['train']
    X_calib = X_data['val'][:n_calib]
    y_calib = y_data['val'][:n_calib]
    X_val = X_data['val'][n_calib:]
    y_val = y_data['val'][n_calib:]
    X_test = X_data['test']
    y_test = y_data['test']

    # Train APS
    aps = APS(alpha=ALPHA)
    aps.fit(X_train, y_train, X_calib, y_calib, seed=seed)

    # Compute coverage
    val_coverage = aps.coverage(X_val, y_val)
    test_coverage = aps.coverage(X_test, y_test)

    # Compute set sizes
    val_size = aps.avg_set_size(X_val)
    test_size = aps.avg_set_size(X_test)

    # Compute feature Jaccard
    feature_jaccard = {}
    for i, col in enumerate(feature_cols):
        train_vals = X_data['train'][:, i]
        test_vals = X_data['test'][:, i]
        jaccard = compute_jaccard(train_vals, test_vals)
        feature_jaccard[col] = jaccard

    mean_jaccard = np.mean(list(feature_jaccard.values()))

    return {
        'task': task_name,
        'seed': seed,
        'val_coverage': val_coverage,
        'test_coverage': test_coverage,
        'coverage_drop': val_coverage - test_coverage,
        'val_set_size': val_size,
        'test_set_size': test_size,
        'mean_jaccard': mean_jaccard,
        'feature_jaccard': feature_jaccard,
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'target_mean': np.mean(y_train),
        'target_std': np.std(y_train),
        'model': aps.model,  # Save for SHAP later
    }


def run_task(dataset_name: str, task_name: str, num_seeds: int = 5) -> Dict:
    """
    Run APS experiment for a classification task.

    Args:
        dataset_name: Dataset name
        task_name: Task name
        num_seeds: Number of seeds

    Returns:
        Aggregated results
    """
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}, Task: {task_name}")
    print(f"Running {num_seeds} seeds with APS...")
    print(f"{'='*80}")

    dataset = get_dataset(dataset_name, download=True)
    task = get_task(dataset_name, task_name, download=True)

    if 'CLASSIFICATION' not in task.task_type.name:
        raise ValueError(f"Task {task_name} is not classification (type: {task.task_type})")

    seed_results = []
    for seed in range(42, 42 + num_seeds):
        print(f"  Seed {seed}...", end=" ", flush=True)
        try:
            result = run_single_seed(task, task_name, seed)
            seed_results.append(result)
            print(f"coverage_drop={result['coverage_drop']*100:.1f}%")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not seed_results:
        raise RuntimeError("All seeds failed!")

    # Aggregate results
    val_covs = [r['val_coverage'] for r in seed_results]
    test_covs = [r['test_coverage'] for r in seed_results]
    drops = [r['coverage_drop'] for r in seed_results]
    val_sizes = [r['val_set_size'] for r in seed_results]
    test_sizes = [r['test_set_size'] for r in seed_results]
    jaccards = [r['mean_jaccard'] for r in seed_results]

    result = {
        'dataset': dataset_name,
        'task': task_name,
        'num_seeds': len(seed_results),
        'task_type': 'binary_classification',

        # Coverage metrics
        'val_coverage_mean': np.mean(val_covs),
        'val_coverage_std': np.std(val_covs),
        'test_coverage_mean': np.mean(test_covs),
        'test_coverage_std': np.std(test_covs),
        'drop_mean': np.mean(drops),
        'drop_std': np.std(drops),

        # Set size metrics
        'val_size_mean': np.mean(val_sizes),
        'test_size_mean': np.mean(test_sizes),

        # Feature stability
        'jaccard_mean': np.mean(jaccards),
        'jaccard_std': np.std(jaccards),

        # Target statistics
        'target_mean': seed_results[0]['target_mean'],
        'target_std': seed_results[0]['target_std'],

        # Raw data
        'seed_results': seed_results,
    }

    # Print summary
    print(f"\nResults for {task_name}:")
    print(f"  Val coverage:   {result['val_coverage_mean']*100:.1f} ± {result['val_coverage_std']*100:.1f}%")
    print(f"  Test coverage:  {result['test_coverage_mean']*100:.1f} ± {result['test_coverage_std']*100:.1f}%")
    print(f"  Drop:           {result['drop_mean']*100:.1f} ± {result['drop_std']*100:.1f}%")
    print(f"  Mean Jaccard:   {result['jaccard_mean']:.3f}")
    print(f"  Val set size:   {result['val_size_mean']:.2f}")
    print(f"  Test set size:  {result['test_size_mean']:.2f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run APS on binary classification tasks"
    )
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (rel-trial or rel-f1)')
    parser.add_argument('--task', type=str, required=True,
                       help='Task name (study-outcome or driver-dnf)')
    parser.add_argument('--num_seeds', type=int, default=50,
                       help='Number of seeds (default: 50)')
    parser.add_argument('--output_dir', type=str,
                       default='papers/conformal_covid/results',
                       help='Output directory')

    args = parser.parse_args()

    # Run experiment
    result = run_task(args.dataset, args.task, args.num_seeds)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pickle (full results)
    pkl_file = output_dir / f"aps_{args.dataset}_{args.task}.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"\n✓ Results saved to: {pkl_file}")

    # JSON (summary)
    json_data = {
        'dataset': result['dataset'],
        'task': result['task'],
        'num_seeds': result['num_seeds'],
        'val_coverage': f"{result['val_coverage_mean']*100:.1f} ± {result['val_coverage_std']*100:.1f}%",
        'test_coverage': f"{result['test_coverage_mean']*100:.1f} ± {result['test_coverage_std']*100:.1f}%",
        'drop': f"{result['drop_mean']*100:.1f} ± {result['drop_std']*100:.1f}%",
        'jaccard': f"{result['jaccard_mean']:.3f} ± {result['jaccard_std']:.3f}",
    }

    json_file = output_dir / f"aps_{args.dataset}_{args.task}.json"
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ Summary saved to: {json_file}")


if __name__ == "__main__":
    import os
    os.chdir('/Users/i767700/Github/ai-in-finance')
    main()
