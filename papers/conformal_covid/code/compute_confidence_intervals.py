"""
Compute Confidence Intervals for Main Results Table

Runs conformal prediction with 5 seeds per task to compute:
- Mean coverage ± std
- Coverage drop with confidence intervals

Usage:
    python compute_confidence_intervals.py

Output:
    results/confidence_intervals.json - CI data for paper
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

NUM_SEEDS = 5
ALPHA = 0.1  # 90% target coverage
SAMPLE_SIZE = 30000


class ConformalClassifier:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.quantile = None

    def _compute_scores(self, probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
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
        scores = self._compute_scores(probs, y_true)
        n = len(scores)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.quantile = np.quantile(scores, q_level)
        return self

    def predict_sets(self, probs: np.ndarray) -> List[set]:
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


def run_single_seed(task, seed: int, sample_size: int = SAMPLE_SIZE) -> Dict:
    """Run conformal prediction for a single seed."""
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    dataset = task.dataset
    entity_table = dataset.get_db().table_dict[task.entity_table]
    entity_df = entity_table.df.copy()

    dfs = {}
    for split, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        entity_df_copy = entity_df.copy()
        left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
        entity_df_copy = entity_df_copy.astype({entity_table.pkey_col: table.df[left_entity].dtype})

        for col in set(entity_df_copy.columns).intersection(set(table.df.columns)):
            if col != entity_table.pkey_col:
                entity_df_copy = entity_df_copy.drop(columns=[col])

        dfs[split] = table.df.merge(
            entity_df_copy, how="left",
            left_on=left_entity, right_on=entity_table.pkey_col,
        )

    # Subsample
    if sample_size and sample_size < len(dfs["train"]):
        np.random.seed(seed)
        idx = np.random.permutation(len(dfs["train"]))[:sample_size]
        dfs["train"] = dfs["train"].iloc[idx].copy()

    target_col = task.target_col

    # Prepare features
    all_data = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)
    exclude_cols = [target_col, 'CREATIONTIMESTAMP', 'timestamp']
    id_cols = [c for c in all_data.columns if c.endswith('_id') or c.endswith('Id') or c == 'ID']
    exclude_cols.extend(id_cols)
    feature_cols = [c for c in all_data.columns if c not in exclude_cols]

    label_encoders = {}
    for col in feature_cols:
        if all_data[col].dtype == 'object' or all_data[col].dtype.name == 'category':
            le = LabelEncoder()
            all_data[col] = all_data[col].astype(str).fillna('__MISSING__')
            le.fit(all_data[col])
            label_encoders[col] = le

    X_data, y_data = {}, {}
    for split, df in dfs.items():
        X = df[feature_cols].copy()
        for col, le in label_encoders.items():
            X[col] = X[col].astype(str).fillna('__MISSING__')
            X[col] = X[col].apply(lambda x: x if x in le.classes_ else '__MISSING__')
            if '__MISSING__' not in le.classes_:
                le.classes_ = np.append(le.classes_, '__MISSING__')
            X[col] = le.transform(X[col])
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-999)
        X_data[split] = X.values.astype(np.float32)
        y_data[split] = df[target_col].values

    target_le = LabelEncoder()
    all_y = np.concatenate([y_data['train'], y_data['val'], y_data['test']])
    target_le.fit(all_y)
    for split in y_data:
        y_data[split] = target_le.transform(y_data[split])

    num_classes = len(target_le.classes_)

    # Train model
    params = {
        'objective': 'multiclass', 'num_class': num_classes,
        'metric': 'multi_logloss', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 5, 'verbose': -1, 'seed': seed, 'n_jobs': -1,
    }

    train_data = lgb.Dataset(X_data['train'], label=y_data['train'])
    val_data = lgb.Dataset(X_data['val'], label=y_data['val'], reference=train_data)

    model = lgb.train(params, train_data, num_boost_round=500,
                      valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(50, verbose=False)])

    val_probs = model.predict(X_data['val'])
    test_probs = model.predict(X_data['test'])

    # Calibration split
    n_val = len(val_probs)
    n_calib = n_val // 2

    calib_probs, calib_y = val_probs[:n_calib], y_data['val'][:n_calib]
    eval_probs, eval_y = val_probs[n_calib:], y_data['val'][n_calib:]

    # Conformal prediction
    conf = ConformalClassifier(alpha=ALPHA)
    conf.calibrate(calib_probs, calib_y)

    val_sets = conf.predict_sets(eval_probs)
    test_sets = conf.predict_sets(test_probs)

    val_cov = sum(1 for i, s in enumerate(val_sets) if eval_y[i] in s) / len(val_sets)
    test_cov = sum(1 for i, s in enumerate(test_sets) if y_data['test'][i] in s) / len(test_sets)

    val_size = np.mean([len(s) for s in val_sets])
    test_size = np.mean([len(s) for s in test_sets])

    return {
        'seed': seed,
        'val_coverage': val_cov,
        'test_coverage': test_cov,
        'coverage_drop': val_cov - test_cov,
        'val_set_size': val_size,
        'test_set_size': test_size,
        'num_classes': num_classes,
    }


def compute_ci_for_task(task, task_name: str, num_seeds: int = NUM_SEEDS) -> Dict:
    """Compute confidence intervals for a task across multiple seeds."""
    print(f"\n>>> {task_name} ({num_seeds} seeds)")

    seed_results = []
    for seed in range(42, 42 + num_seeds):
        print(f"  Seed {seed}...", end=" ", flush=True)
        result = run_single_seed(task, seed)
        seed_results.append(result)
        print(f"drop={result['coverage_drop']*100:.1f}%")

    # Aggregate
    val_covs = [r['val_coverage'] for r in seed_results]
    test_covs = [r['test_coverage'] for r in seed_results]
    drops = [r['coverage_drop'] for r in seed_results]
    val_sizes = [r['val_set_size'] for r in seed_results]
    test_sizes = [r['test_set_size'] for r in seed_results]

    return {
        'task': task_name,
        'num_classes': seed_results[0]['num_classes'],
        'val_coverage_mean': np.mean(val_covs),
        'val_coverage_std': np.std(val_covs),
        'test_coverage_mean': np.mean(test_covs),
        'test_coverage_std': np.std(test_covs),
        'drop_mean': np.mean(drops),
        'drop_std': np.std(drops),
        'val_size_mean': np.mean(val_sizes),
        'test_size_mean': np.mean(test_sizes),
        'seed_results': seed_results,
    }


def main():
    from relbench.tasks import get_task

    tasks = [
        'sales-shipcond', 'sales-group', 'sales-payterms', 'item-plant',
        'item-shippoint', 'sales-incoterms', 'item-incoterms', 'sales-office'
    ]

    results = []

    print("=" * 70)
    print(f"COMPUTING CONFIDENCE INTERVALS ({NUM_SEEDS} seeds per task)")
    print("=" * 70)

    for task_name in tasks:
        try:
            task = get_task('rel-salt', task_name, download=False)
            result = compute_ci_for_task(task, task_name, NUM_SEEDS)
            results.append(result)
        except Exception as e:
            print(f"  Error: {e}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "confidence_intervals.pkl", 'wb') as f:
        pickle.dump(results, f)

    # Also save as JSON for easy reading
    json_results = []
    for r in results:
        json_results.append({
            'task': r['task'],
            'num_classes': r['num_classes'],
            'val_coverage': f"{r['val_coverage_mean']*100:.1f} ± {r['val_coverage_std']*100:.1f}",
            'test_coverage': f"{r['test_coverage_mean']*100:.1f} ± {r['test_coverage_std']*100:.1f}",
            'drop': f"{r['drop_mean']*100:.1f} ± {r['drop_std']*100:.1f}",
        })

    with open(output_dir / "confidence_intervals.json", 'w') as f:
        json.dump(json_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS WITH CONFIDENCE INTERVALS")
    print("=" * 70)
    print(f"{'Task':<16} {'Val Coverage':>16} {'Test Coverage':>16} {'Drop':>16}")
    print("-" * 70)

    for r in results:
        print(f"{r['task']:<16} "
              f"{r['val_coverage_mean']*100:>5.1f} ± {r['val_coverage_std']*100:>4.1f}% "
              f"{r['test_coverage_mean']*100:>5.1f} ± {r['test_coverage_std']*100:>4.1f}% "
              f"{r['drop_mean']*100:>5.1f} ± {r['drop_std']*100:>4.1f}%")

    print(f"\nResults saved to {output_dir}")

    return results


if __name__ == "__main__":
    main()
