"""
Stable Confidence Interval Computation

Key fix: Use fixed data splits, only vary model random seed.
This isolates model variance from data sampling variance.
"""

import json
import warnings
from pathlib import Path
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

NUM_SEEDS = 5
ALPHA = 0.1
SAMPLE_SIZE = 30000


class ConformalClassifier:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.quantile = None

    def _compute_scores(self, probs, y_true):
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

    def calibrate(self, probs, y_true):
        scores = self._compute_scores(probs, y_true)
        n = len(scores)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.quantile = np.quantile(scores, q_level)
        return self

    def predict_sets(self, probs):
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


def prepare_fixed_data(task, sample_size=SAMPLE_SIZE, random_seed=42):
    """Prepare data with FIXED random seed for reproducibility."""
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

    # FIXED random seed for subsampling
    if sample_size and sample_size < len(dfs["train"]):
        np.random.seed(random_seed)  # Always use same seed!
        idx = np.random.permutation(len(dfs["train"]))[:sample_size]
        dfs["train"] = dfs["train"].iloc[idx].copy()

    target_col = task.target_col

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
    return X_data, y_data, num_classes


def run_with_model_seed(X_data, y_data, num_classes, model_seed):
    """Run conformal prediction with specific model seed."""
    params = {
        'objective': 'multiclass', 'num_class': num_classes,
        'metric': 'multi_logloss', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 5, 'verbose': -1, 'seed': model_seed, 'n_jobs': -1,
    }

    train_data = lgb.Dataset(X_data['train'], label=y_data['train'])
    val_data = lgb.Dataset(X_data['val'], label=y_data['val'], reference=train_data)

    model = lgb.train(params, train_data, num_boost_round=500,
                      valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(50, verbose=False)])

    val_probs = model.predict(X_data['val'])
    test_probs = model.predict(X_data['test'])

    # Fixed calibration split
    n_val = len(val_probs)
    n_calib = n_val // 2

    calib_probs, calib_y = val_probs[:n_calib], y_data['val'][:n_calib]
    eval_probs, eval_y = val_probs[n_calib:], y_data['val'][n_calib:]

    conf = ConformalClassifier(alpha=ALPHA)
    conf.calibrate(calib_probs, calib_y)

    val_sets = conf.predict_sets(eval_probs)
    test_sets = conf.predict_sets(test_probs)

    val_cov = sum(1 for i, s in enumerate(val_sets) if eval_y[i] in s) / len(val_sets)
    test_cov = sum(1 for i, s in enumerate(test_sets) if y_data['test'][i] in s) / len(test_sets)

    val_size = np.mean([len(s) for s in val_sets])
    test_size = np.mean([len(s) for s in test_sets])

    return {
        'val_coverage': val_cov,
        'test_coverage': test_cov,
        'coverage_drop': val_cov - test_cov,
        'val_set_size': val_size,
        'test_set_size': test_size,
    }


def analyze_task(task, task_name, num_seeds=NUM_SEEDS):
    """Analyze task with multiple model seeds."""
    print(f"\n>>> {task_name}")

    # Prepare data ONCE with fixed seed
    print("  Preparing data...")
    X_data, y_data, num_classes = prepare_fixed_data(task, SAMPLE_SIZE, random_seed=42)

    # Run with different model seeds
    results = []
    for seed in range(42, 42 + num_seeds):
        print(f"  Model seed {seed}...", end=" ", flush=True)
        r = run_with_model_seed(X_data, y_data, num_classes, seed)
        results.append(r)
        print(f"drop={r['coverage_drop']*100:.1f}%")

    # Aggregate
    val_covs = [r['val_coverage'] for r in results]
    test_covs = [r['test_coverage'] for r in results]
    drops = [r['coverage_drop'] for r in results]

    return {
        'task': task_name,
        'num_classes': num_classes,
        'val_coverage_mean': np.mean(val_covs),
        'val_coverage_std': np.std(val_covs),
        'test_coverage_mean': np.mean(test_covs),
        'test_coverage_std': np.std(test_covs),
        'drop_mean': np.mean(drops),
        'drop_std': np.std(drops),
        'val_size': np.mean([r['val_set_size'] for r in results]),
        'test_size': np.mean([r['test_set_size'] for r in results]),
    }


def main():
    from relbench.tasks import get_task

    tasks = [
        'sales-shipcond', 'sales-group', 'sales-payterms', 'item-plant',
        'item-shippoint', 'sales-incoterms', 'item-incoterms', 'sales-office'
    ]

    all_results = []

    print("=" * 70)
    print(f"STABLE CI COMPUTATION ({NUM_SEEDS} model seeds, fixed data)")
    print("=" * 70)

    for task_name in tasks:
        try:
            task = get_task('rel-salt', task_name, download=False)
            result = analyze_task(task, task_name, NUM_SEEDS)
            all_results.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Save
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "confidence_intervals_stable.pkl", 'wb') as f:
        pickle.dump(all_results, f)

    # Print table
    print("\n" + "=" * 80)
    print("RESULTS WITH CONFIDENCE INTERVALS (mean ± std)")
    print("=" * 80)
    print(f"{'Task':<16} {'Val Coverage':>16} {'Test Coverage':>16} {'Drop':>16}")
    print("-" * 80)

    for r in all_results:
        val_str = f"{r['val_coverage_mean']*100:.1f} ± {r['val_coverage_std']*100:.1f}%"
        test_str = f"{r['test_coverage_mean']*100:.1f} ± {r['test_coverage_std']*100:.1f}%"
        drop_str = f"{r['drop_mean']*100:.1f} ± {r['drop_std']*100:.1f}%"
        print(f"{r['task']:<16} {val_str:>16} {test_str:>16} {drop_str:>16}")

    # LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)
    for r in all_results:
        print(f"{r['task'].replace('-', '')} & {r['num_classes']} & "
              f"${r['val_coverage_mean']*100:.1f} \\pm {r['val_coverage_std']*100:.1f}$ & "
              f"${r['test_coverage_mean']*100:.1f} \\pm {r['test_coverage_std']*100:.1f}$ & "
              f"${r['drop_mean']*100:.1f} \\pm {r['drop_std']*100:.1f}$ \\\\")

    return all_results


if __name__ == "__main__":
    main()
