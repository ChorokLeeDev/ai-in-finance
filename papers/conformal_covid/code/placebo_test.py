"""
Placebo Test: Pre-COVID Temporal Split

This test addresses the reviewer question:
"Would a non-COVID temporal shift (e.g., 2018→2019) show similar degradation patterns?"

Methodology:
1. Use data from 2018 (train) → 2019 (test) - no COVID involved
2. Run same conformal prediction analysis
3. Compare coverage degradation patterns

Expected result if COVID is special:
- Pre-COVID temporal shift should show LESS degradation
- Or different pattern (gradual vs catastrophic)

Expected result if temporal shift is the real cause:
- Similar degradation patterns regardless of COVID

This is critical for establishing causality: COVID vs general temporal drift.
"""

import sys
from pathlib import Path

# Add local relbench fork to path (contains rel-salt dataset)
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

import warnings
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List

warnings.filterwarnings('ignore')


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


def run_placebo_test(task, task_name: str, pre_covid_cutoff: str = "2019-01-01",
                     placebo_cutoff: str = "2019-07-01") -> Dict:
    """
    Run placebo test with pre-COVID temporal split.

    Uses ONLY training data (pre-Feb 2020) to create:
    - Placebo train: 2018
    - Placebo val: 2019 H1
    - Placebo test: 2019 H2

    This tests whether temporal drift alone (without COVID) causes similar degradation.
    """
    train_table = task.get_table("train")

    # Get entity table for features
    dataset = task.dataset
    entity_table = dataset.get_db().table_dict[task.entity_table]
    entity_df = entity_table.df.copy()

    # Merge to get full data
    left_entity = list(train_table.fkey_col_to_pkey_table.keys())[0]
    entity_df_copy = entity_df.copy()
    entity_df_copy = entity_df_copy.astype({entity_table.pkey_col: train_table.df[left_entity].dtype})

    # Remove duplicate columns before merge
    for col in set(entity_df_copy.columns).intersection(set(train_table.df.columns)):
        if col != entity_table.pkey_col:
            entity_df_copy = entity_df_copy.drop(columns=[col])

    full_df = train_table.df.merge(
        entity_df_copy, how="left",
        left_on=left_entity, right_on=entity_table.pkey_col,
    )

    # Find timestamp column - check train_table first (it should have CREATIONTIMESTAMP)
    ts_col = None
    if 'CREATIONTIMESTAMP' in train_table.df.columns:
        # Use timestamp from original train table
        full_df['_timestamp'] = train_table.df['CREATIONTIMESTAMP'].values
        ts_col = '_timestamp'
    elif 'timestamp' in full_df.columns:
        ts_col = 'timestamp'
    elif 'CREATIONTIMESTAMP' in full_df.columns:
        ts_col = 'CREATIONTIMESTAMP'
    else:
        raise ValueError("No timestamp column found for placebo test")

    full_df[ts_col] = pd.to_datetime(full_df[ts_col])

    # Placebo split: 2018 (train) | 2019-H1 (val) | 2019-H2 (test)
    pre_covid_cutoff = pd.to_datetime(pre_covid_cutoff)
    placebo_cutoff = pd.to_datetime(placebo_cutoff)

    placebo_train = full_df[full_df[ts_col] < pre_covid_cutoff].copy()
    placebo_val = full_df[(full_df[ts_col] >= pre_covid_cutoff) &
                          (full_df[ts_col] < placebo_cutoff)].copy()
    placebo_test = full_df[full_df[ts_col] >= placebo_cutoff].copy()

    print(f"  Placebo split sizes: train={len(placebo_train)}, val={len(placebo_val)}, test={len(placebo_test)}")

    if len(placebo_train) < 1000 or len(placebo_val) < 100 or len(placebo_test) < 100:
        print("  Insufficient data for placebo test")
        return {'task': task_name, 'error': 'insufficient_data'}

    # Prepare features (same as main analysis)
    target_col = task.target_col
    exclude_cols = [target_col, ts_col, 'timestamp', 'CREATIONTIMESTAMP']
    id_cols = [c for c in full_df.columns if c.endswith('_id') or c.endswith('Id') or c == 'ID']
    exclude_cols.extend(id_cols)

    feature_cols = [c for c in full_df.columns if c not in exclude_cols and not c.startswith('_')]

    # Encode features
    label_encoders = {}
    all_data = pd.concat([placebo_train, placebo_val, placebo_test], ignore_index=True)

    for col in feature_cols:
        if all_data[col].dtype == 'object' or all_data[col].dtype.name == 'category':
            le = LabelEncoder()
            all_data[col] = all_data[col].astype(str).fillna('__MISSING__')
            le.fit(all_data[col])
            label_encoders[col] = le

    def prepare_split(df):
        X = df[feature_cols].copy()
        for col, le in label_encoders.items():
            X[col] = X[col].astype(str).fillna('__MISSING__')
            X[col] = X[col].apply(lambda x: x if x in le.classes_ else '__MISSING__')
            if '__MISSING__' not in le.classes_:
                le.classes_ = np.append(le.classes_, '__MISSING__')
            X[col] = le.transform(X[col])
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-999)
        return X.values.astype(np.float32)

    X_train = prepare_split(placebo_train)
    X_val = prepare_split(placebo_val)
    X_test = prepare_split(placebo_test)

    # Encode targets
    target_le = LabelEncoder()
    all_y = np.concatenate([
        placebo_train[target_col].values,
        placebo_val[target_col].values,
        placebo_test[target_col].values
    ])
    target_le.fit(all_y)

    y_train = target_le.transform(placebo_train[target_col].values)
    y_val = target_le.transform(placebo_val[target_col].values)
    y_test = target_le.transform(placebo_test[target_col].values)

    num_classes = len(target_le.classes_)

    # Train model
    params = {
        'objective': 'multiclass', 'num_class': num_classes,
        'metric': 'multi_logloss', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'learning_rate': 0.05,
        'verbose': -1, 'seed': 42, 'n_jobs': -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(params, train_data, num_boost_round=500,
                      valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(50, verbose=False)])

    val_probs = model.predict(X_val)
    test_probs = model.predict(X_test)

    # Conformal prediction
    n_calib = len(val_probs) // 2
    calib_probs, calib_y = val_probs[:n_calib], y_val[:n_calib]
    eval_probs, eval_y = val_probs[n_calib:], y_val[n_calib:]

    conf = ConformalClassifier(alpha=0.1)
    conf.calibrate(calib_probs, calib_y)

    val_sets = conf.predict_sets(eval_probs)
    test_sets = conf.predict_sets(test_probs)

    val_coverage = sum(1 for i, s in enumerate(val_sets) if eval_y[i] in s) / len(val_sets)
    test_coverage = sum(1 for i, s in enumerate(test_sets) if y_test[i] in s) / len(test_sets)

    return {
        'task': task_name,
        'placebo_val_coverage': val_coverage,
        'placebo_test_coverage': test_coverage,
        'placebo_drop': val_coverage - test_coverage,
        'num_classes': num_classes,
        'train_size': len(placebo_train),
        'val_size': len(placebo_val),
        'test_size': len(placebo_test),
    }


def main():
    """Run placebo tests for all tasks."""
    from relbench.tasks import get_task

    tasks = [
        'sales-shipcond', 'sales-group', 'sales-payterms', 'item-plant',
        'item-shippoint', 'sales-incoterms', 'item-incoterms', 'sales-office'
    ]

    # Known COVID results for comparison
    covid_results = {
        'sales-shipcond': 93.1,
        'sales-group': 86.7,
        'sales-payterms': 33.8,
        'item-plant': 29.1,
        'item-shippoint': 18.9,
        'sales-incoterms': 3.6,
        'item-incoterms': 0.5,
        'sales-office': 0.1,
    }

    results = []

    print("=" * 70)
    print("PLACEBO TEST: Pre-COVID Temporal Split (2018 → 2019)")
    print("=" * 70)

    for task_name in tasks:
        print(f"\n>>> {task_name}")
        try:
            task = get_task('rel-salt', task_name, download=False)
            result = run_placebo_test(task, task_name)
            results.append(result)

            if 'error' not in result:
                covid_drop = covid_results.get(task_name, 0)
                print(f"  Placebo drop: {result['placebo_drop']*100:.1f}%")
                print(f"  COVID drop:   {covid_drop:.1f}%")
                print(f"  Ratio:        {result['placebo_drop']*100/covid_drop:.2f}x" if covid_drop > 0 else "")
        except Exception as e:
            print(f"  Error: {e}")
            results.append({'task': task_name, 'error': str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("PLACEBO vs COVID COMPARISON")
    print("=" * 70)
    print(f"{'Task':<18} {'Placebo Drop':>14} {'COVID Drop':>12} {'Ratio':>8}")
    print("-" * 55)

    for r in results:
        if 'error' not in r:
            covid_drop = covid_results.get(r['task'], 0)
            ratio = r['placebo_drop'] * 100 / covid_drop if covid_drop > 0 else 0
            print(f"{r['task']:<18} {r['placebo_drop']*100:>13.1f}% {covid_drop:>11.1f}% {ratio:>7.2f}x")

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "placebo_test_results.pkl", 'wb') as f:
        pickle.dump(results, f)

    print(f"\nResults saved to {results_dir / 'placebo_test_results.pkl'}")

    return results


if __name__ == "__main__":
    main()
