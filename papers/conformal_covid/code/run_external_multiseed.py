#!/usr/bin/env python3
"""
Multi-seed external validation across 4 external datasets:
  1. Forest Covertype (geographic shift, 7 classes)
  2. KDDCup99 (temporal shift, 11 classes)
  3. Gas Sensor Array Drift (sensor drift, 6 classes)
  4. Stack Overflow / rel-stack badges-class (temporal shift, 3 classes)

For each dataset: 10 seeds, per seed train LightGBM, compute SHAP concentration,
run APS conformal, record coverage drop. Report mean/std across seeds.

Usage:
    PYTHONPATH=/Users/i767700/Github/ai-in-finance:$PYTHONPATH \
    /usr/bin/python3 -u papers/conformal_covid/code/run_external_multiseed.py
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_FILE = RESULTS_DIR / "external_multiseed_validation.json"
NUM_SEEDS = 10
SEEDS = list(range(42, 42 + NUM_SEEDS))
ALPHA = 0.1
THRESHOLD = 40.0


# ============================================================================
# Shared utilities
# ============================================================================

def train_lgb(X_train, y_train, X_val, y_val, num_classes, seed=42):
    """Train LightGBM classifier."""
    import lightgbm as lgb

    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': seed,
        'verbose': -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    return model


def compute_shap_concentration(model, X_val, feature_names, seed=42):
    """Compute SHAP concentration (top-1 / total)."""
    import shap

    explainer = shap.TreeExplainer(model)
    n_sample = min(5000, len(X_val))
    idx = np.random.RandomState(seed).choice(len(X_val), n_sample, replace=False)
    X_sample = X_val[idx] if isinstance(X_val, np.ndarray) else X_val.values[idx]
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    mean_abs = np.array(mean_abs).flatten()
    total = float(mean_abs.sum())
    top1 = float(mean_abs.max())
    concentration = (top1 / total * 100) if total > 0 else 0

    top_idx = int(np.argmax(mean_abs))
    top_feature = feature_names[top_idx] if top_idx < len(feature_names) else f"feature_{top_idx}"

    return concentration, top_feature


def run_aps_conformal(model, X_val, y_val, X_test, y_test, num_classes, seed=42, alpha=0.1):
    """Run APS conformal prediction with random cal/eval split."""
    val_probs = model.predict(X_val)
    test_probs = model.predict(X_test)

    # Random cal/eval split of validation set (seed-dependent)
    n_val = len(y_val)
    perm = np.random.RandomState(seed).permutation(n_val)
    n_cal = n_val // 2
    cal_idx = perm[:n_cal]
    eval_idx = perm[n_cal:]

    cal_probs = val_probs[cal_idx]
    cal_y = y_val[cal_idx]
    eval_probs = val_probs[eval_idx]
    eval_y = y_val[eval_idx]

    def aps_score(probs, true_label):
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = 0.0
        for idx in sorted_idx:
            cumsum += probs[idx]
            if idx == true_label:
                return cumsum
        return 1.0

    cal_scores = np.array([aps_score(cal_probs[i], cal_y[i]) for i in range(len(cal_y))])
    n = len(cal_scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    qhat = np.quantile(cal_scores, q_level)

    eval_scores = np.array([aps_score(eval_probs[i], eval_y[i]) for i in range(len(eval_y))])
    val_coverage = float(np.mean(eval_scores <= qhat) * 100)

    test_scores = np.array([aps_score(test_probs[i], y_test[i]) for i in range(len(y_test))])
    test_coverage = float(np.mean(test_scores <= qhat) * 100)

    def get_set_sizes(probs_arr, qhat_val):
        sizes = []
        for i in range(len(probs_arr)):
            sorted_idx = np.argsort(probs_arr[i])[::-1]
            cumsum = 0.0
            size = 0
            for idx in sorted_idx:
                cumsum += probs_arr[i][idx]
                size += 1
                if cumsum >= qhat_val:
                    break
            sizes.append(size)
        return np.array(sizes)

    val_sizes = get_set_sizes(eval_probs, qhat)
    test_sizes = get_set_sizes(test_probs, qhat)

    return {
        'val_coverage': round(val_coverage, 2),
        'test_coverage': round(test_coverage, 2),
        'coverage_drop': round(val_coverage - test_coverage, 2),
        'val_mean_set_size': round(float(val_sizes.mean()), 2),
        'test_mean_set_size': round(float(test_sizes.mean()), 2),
        'qhat': round(float(qhat), 4),
    }


def save_incremental(all_results):
    """Save results incrementally to prevent data loss."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)


# ============================================================================
# Dataset loaders
# ============================================================================

def load_covertype():
    """Load Covertype: split by wilderness area for geographic shift."""
    from sklearn.datasets import fetch_covtype
    print("  Loading Covertype...")
    data = fetch_covtype()
    X = data.data
    y = data.target - 1

    feature_names = [f'feat_{i}' for i in range(X.shape[1])]
    if hasattr(data, 'feature_names'):
        feature_names = list(data.feature_names)
    num_classes = len(np.unique(y))

    # Split by wilderness area (cols 10-13 one-hot)
    area1 = X[:, 10] == 1
    area2 = X[:, 11] == 1
    area3 = X[:, 12] == 1
    area4 = X[:, 13] == 1

    X_train, y_train = X[area1 | area2], y[area1 | area2]
    X_val, y_val = X[area3], y[area3]
    X_test, y_test = X[area4], y[area4]

    print(f"  Covertype: {num_classes} classes, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, feature_names


def load_kddcup99():
    """Load KDDCup99: temporal split."""
    from sklearn.datasets import fetch_kddcup99
    from sklearn.preprocessing import LabelEncoder
    print("  Loading KDDCup99...")
    data = fetch_kddcup99(subset=None, percent10=True)
    X_raw = data.data
    y_raw = data.target

    feature_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]

    df = pd.DataFrame(X_raw, columns=feature_names[:X_raw.shape[1]])
    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Filter to classes with >= 100 samples
    label_counts = pd.Series(y_raw).value_counts()
    valid_labels = label_counts[label_counts >= 100].index
    mask = np.isin(y_raw, valid_labels)
    df = df[mask].reset_index(drop=True)
    y_filtered = y_raw[mask]

    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y_filtered)
    num_classes = len(le_y.classes_)

    X = df.values
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train, y_train = X[:train_end], y_encoded[:train_end]
    X_val, y_val = X[train_end:val_end], y_encoded[train_end:val_end]
    X_test, y_test = X[val_end:], y_encoded[val_end:]

    print(f"  KDDCup99: {num_classes} classes, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, list(df.columns)


def load_gas_sensor():
    """Load Gas Sensor Array Drift: batch 1-6 train, 7-8 val, 9-10 test."""
    from sklearn.preprocessing import LabelEncoder
    data_dir = RESULTS_DIR / 'gas_sensor_drift' / 'Dataset'
    print("  Loading Gas Sensor...")

    all_rows, all_labels, all_batches = [], [], []
    for batch_id in range(1, 11):
        fpath = data_dir / f'batch{batch_id}.dat'
        with open(fpath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                label = int(parts[0])
                features = np.zeros(128)
                for p in parts[1:]:
                    idx_str, val_str = p.split(':')
                    features[int(idx_str) - 1] = float(val_str)
                all_rows.append(features)
                all_labels.append(label)
                all_batches.append(batch_id)

    X = np.array(all_rows)
    labels = np.array(all_labels)
    batches = np.array(all_batches)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    num_classes = len(le.classes_)

    feature_names = [f'sensor_{i+1}' for i in range(128)]

    train_mask = np.isin(batches, [1,2,3,4,5,6])
    val_mask = np.isin(batches, [7,8])
    test_mask = np.isin(batches, [9,10])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Gas Sensor: {num_classes} classes, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, feature_names


def load_stackoverflow():
    """Load Stack Overflow badges-class from RelBench. Uses fast pd.Categorical encoding."""
    import pooch
    from sklearn.preprocessing import LabelEncoder
    from relbench.datasets.stack import StackDataset
    from relbench.base.task_autocomplete import AutoCompleteTask
    from relbench.base.task_base import TaskType

    print("  Loading Stack Overflow (rel-stack badges-class)...")
    t0 = time.time()
    cache_dir = os.path.join(str(pooch.os_cache('relbench')), 'rel-stack')
    dataset = StackDataset(cache_dir=cache_dir)
    task = AutoCompleteTask(
        dataset=dataset,
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        entity_table='badges',
        target_col='Class',
        remove_columns=[('badges', 'TagBased'), ('badges', 'Name')],
    )

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    db = dataset.get_db()
    entity_table = db.table_dict[task.entity_table]
    entity_df = entity_table.df.copy()

    # Join entity features
    dfs = {}
    for split, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        entity_copy = entity_df.copy()
        left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
        entity_copy = entity_copy.astype(
            {entity_table.pkey_col: table.df[left_entity].dtype}
        )
        for col in set(entity_copy.columns).intersection(set(table.df.columns)):
            if col != entity_table.pkey_col:
                entity_copy = entity_copy.drop(columns=[col])
        dfs[split] = table.df.merge(
            entity_copy, how="left",
            left_on=left_entity, right_on=entity_table.pkey_col,
        )

    # Join user features
    users_df = db.table_dict["users"].df.copy()
    for split in dfs:
        if 'UserId' in dfs[split].columns:
            users_merge = users_df.rename(columns={'Id': 'UserId'}).copy()
            for col in set(users_merge.columns).intersection(set(dfs[split].columns)):
                if col != 'UserId':
                    users_merge = users_merge.rename(columns={col: f'{col}_user'})
            dfs[split] = dfs[split].merge(users_merge, on='UserId', how='left')

    # Add temporal features
    for split in dfs:
        df = dfs[split]
        if 'Date' in df.columns:
            dt = pd.to_datetime(df['Date'])
            df['badge_month'] = dt.dt.month
            df['badge_dayofweek'] = dt.dt.dayofweek
            df['badge_hour'] = dt.dt.hour
            df['badge_year'] = dt.dt.year
        if 'UserId' in df.columns:
            df['user_id_numeric'] = pd.to_numeric(df['UserId'], errors='coerce').fillna(-1)
        dfs[split] = df

    target_col = task.target_col

    # Feature columns: exclude target, time, ID columns
    all_cols = set()
    for split in dfs:
        all_cols.update(dfs[split].columns)
    all_cols = sorted(all_cols)

    exclude_cols = {target_col}
    for c in all_cols:
        cl = c.lower()
        if 'date' in cl or 'time' in cl:
            exclude_cols.add(c)
        if cl.endswith('_id') or cl.endswith('id') or cl == 'id':
            exclude_cols.add(c)
    feature_cols = [c for c in all_cols if c not in exclude_cols and c in dfs['train'].columns]

    # FAST categorical encoding using pd.Categorical
    for split in dfs:
        dfs[split] = dfs[split][feature_cols + [target_col]].copy()

    all_data = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)

    for col in feature_cols:
        if all_data[col].dtype == 'object' or all_data[col].dtype.name == 'category':
            all_data[col] = all_data[col].astype(str).fillna('__MISSING__')
            cat = pd.Categorical(all_data[col])
            all_data[col] = cat.codes
        else:
            all_data[col] = pd.to_numeric(all_data[col], errors='coerce').fillna(-999)

    n_train = len(dfs['train'])
    n_val = len(dfs['val'])

    X_all = all_data[feature_cols].values.astype(np.float32)
    y_all_raw = all_data[target_col].values

    X_train = X_all[:n_train]
    X_val = X_all[n_train:n_train+n_val]
    X_test = X_all[n_train+n_val:]

    target_le = LabelEncoder()
    y_all_enc = target_le.fit_transform(y_all_raw)
    y_train = y_all_enc[:n_train]
    y_val = y_all_enc[n_train:n_train+n_val]
    y_test = y_all_enc[n_train+n_val:]

    num_classes = len(target_le.classes_)

    # Subsample training to 50k (consistent with original)
    SAMPLE_SIZE = 50000
    if SAMPLE_SIZE < len(X_train):
        np.random.seed(42)  # Fixed seed for consistent subsample
        idx = np.random.permutation(len(X_train))[:SAMPLE_SIZE]
        X_train = X_train[idx]
        y_train = y_train[idx]

    print(f"  Stack Overflow: {num_classes} classes, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"  Total load time: {time.time()-t0:.1f}s")
    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, feature_cols


# ============================================================================
# Per-dataset multi-seed runner
# ============================================================================

def run_multiseed(dataset_name, X_train, y_train, X_val, y_val, X_test, y_test,
                  num_classes, feature_names, seeds):
    """Run multi-seed evaluation for one dataset."""
    print(f"\n{'='*60}")
    print(f"Multi-seed evaluation: {dataset_name}")
    print(f"  {num_classes} classes, {len(seeds)} seeds")
    print(f"{'='*60}")

    seed_results = []

    for i, seed in enumerate(seeds):
        t0 = time.time()
        print(f"\n  Seed {seed} ({i+1}/{len(seeds)})...")

        # Train
        model = train_lgb(X_train, y_train, X_val, y_val, num_classes, seed=seed)

        # SHAP concentration (use seed for SHAP sampling too)
        concentration, top_feature = compute_shap_concentration(
            model, X_val, feature_names, seed=seed)

        # APS conformal (seed for cal/eval split)
        conformal = run_aps_conformal(
            model, X_val, y_val, X_test, y_test, num_classes, seed=seed, alpha=ALPHA)

        # Accuracy
        val_pred = np.argmax(model.predict(X_val), axis=1)
        test_pred = np.argmax(model.predict(X_test), axis=1)
        val_acc = float(np.mean(val_pred == y_val) * 100)
        test_acc = float(np.mean(test_pred == y_test) * 100)

        elapsed = time.time() - t0

        result = {
            'seed': seed,
            'concentration': round(concentration, 2),
            'top_feature': top_feature,
            'val_coverage': conformal['val_coverage'],
            'test_coverage': conformal['test_coverage'],
            'coverage_drop': conformal['coverage_drop'],
            'val_mean_set_size': conformal['val_mean_set_size'],
            'test_mean_set_size': conformal['test_mean_set_size'],
            'val_accuracy': round(val_acc, 1),
            'test_accuracy': round(test_acc, 1),
            'qhat': conformal['qhat'],
            'time_s': round(elapsed, 1),
        }
        seed_results.append(result)
        print(f"    C={concentration:.1f}%, drop={conformal['coverage_drop']:.1f}pp, "
              f"val_cov={conformal['val_coverage']:.1f}%, test_cov={conformal['test_coverage']:.1f}%, "
              f"time={elapsed:.1f}s")

    # Aggregate
    concentrations = [r['concentration'] for r in seed_results]
    drops = [r['coverage_drop'] for r in seed_results]
    val_covs = [r['val_coverage'] for r in seed_results]
    test_covs = [r['test_coverage'] for r in seed_results]

    mean_concentration = float(np.mean(concentrations))
    std_concentration = float(np.std(concentrations))
    mean_drop = float(np.mean(drops))
    std_drop = float(np.std(drops))

    # Check threshold
    predicted = "vulnerable" if mean_concentration > THRESHOLD else "robust"
    actual = "severe" if mean_drop > 15 else "robust"
    correct = (predicted == "vulnerable" and actual == "severe") or \
              (predicted == "robust" and actual == "robust")

    per_seed_correct = sum(
        1 for r in seed_results
        if ((r['concentration'] > THRESHOLD and r['coverage_drop'] > 15) or
            (r['concentration'] <= THRESHOLD and r['coverage_drop'] <= 15))
    )

    summary = {
        'dataset': dataset_name,
        'num_classes': num_classes,
        'n_train': len(y_train),
        'n_val': len(y_val),
        'n_test': len(y_test),
        'num_seeds': len(seeds),
        'concentration_mean': round(mean_concentration, 2),
        'concentration_std': round(std_concentration, 2),
        'coverage_drop_mean': round(mean_drop, 2),
        'coverage_drop_std': round(std_drop, 2),
        'val_coverage_mean': round(float(np.mean(val_covs)), 2),
        'val_coverage_std': round(float(np.std(val_covs)), 2),
        'test_coverage_mean': round(float(np.mean(test_covs)), 2),
        'test_coverage_std': round(float(np.std(test_covs)), 2),
        'threshold': THRESHOLD,
        'predicted_from_mean': predicted,
        'actual_category': actual,
        'threshold_correct_on_mean': correct,
        'per_seed_threshold_correct': per_seed_correct,
        'per_seed_threshold_correct_pct': round(per_seed_correct / len(seeds) * 100, 1),
        'seed_results': seed_results,
    }

    print(f"\n  SUMMARY for {dataset_name}:")
    print(f"    Concentration: {mean_concentration:.1f} +/- {std_concentration:.1f}%")
    print(f"    Coverage drop: {mean_drop:.1f} +/- {std_drop:.1f}pp")
    print(f"    Val coverage:  {np.mean(val_covs):.1f} +/- {np.std(val_covs):.1f}%")
    print(f"    Test coverage: {np.mean(test_covs):.1f} +/- {np.std(test_covs):.1f}%")
    print(f"    Threshold prediction (mean): {predicted} -> {actual} | Correct: {correct}")
    print(f"    Per-seed threshold correct: {per_seed_correct}/{len(seeds)} ({per_seed_correct/len(seeds)*100:.0f}%)")

    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Multi-Seed External Validation (10 seeds x 4 datasets)")
    print("=" * 70)
    t_total = time.time()

    all_results = {
        'metadata': {
            'num_seeds': NUM_SEEDS,
            'seeds': SEEDS,
            'alpha': ALPHA,
            'threshold': THRESHOLD,
            'description': 'Multi-seed external validation across 4 datasets',
        },
        'datasets': {},
    }

    # ---- 1. Covertype ----
    print("\n[1/4] Loading Covertype...")
    try:
        X_tr, y_tr, X_v, y_v, X_te, y_te, nc, fn = load_covertype()
        result = run_multiseed("Forest Covertype", X_tr, y_tr, X_v, y_v, X_te, y_te, nc, fn, SEEDS)
        all_results['datasets']['covertype'] = result
        save_incremental(all_results)
        print("  [Saved incremental results]")
    except Exception as e:
        print(f"  ERROR on Covertype: {e}")
        import traceback; traceback.print_exc()

    # ---- 2. KDDCup99 ----
    print("\n[2/4] Loading KDDCup99...")
    try:
        X_tr, y_tr, X_v, y_v, X_te, y_te, nc, fn = load_kddcup99()
        result = run_multiseed("KDDCup99", X_tr, y_tr, X_v, y_v, X_te, y_te, nc, fn, SEEDS)
        all_results['datasets']['kddcup99'] = result
        save_incremental(all_results)
        print("  [Saved incremental results]")
    except Exception as e:
        print(f"  ERROR on KDDCup99: {e}")
        import traceback; traceback.print_exc()

    # ---- 3. Gas Sensor ----
    print("\n[3/4] Loading Gas Sensor...")
    try:
        X_tr, y_tr, X_v, y_v, X_te, y_te, nc, fn = load_gas_sensor()
        result = run_multiseed("Gas Sensor Array Drift", X_tr, y_tr, X_v, y_v, X_te, y_te, nc, fn, SEEDS)
        all_results['datasets']['gas_sensor'] = result
        save_incremental(all_results)
        print("  [Saved incremental results]")
    except Exception as e:
        print(f"  ERROR on Gas Sensor: {e}")
        import traceback; traceback.print_exc()

    # ---- 4. Stack Overflow ----
    print("\n[4/4] Loading Stack Overflow...")
    try:
        X_tr, y_tr, X_v, y_v, X_te, y_te, nc, fn = load_stackoverflow()
        result = run_multiseed("Stack Overflow", X_tr, y_tr, X_v, y_v, X_te, y_te, nc, fn, SEEDS)
        all_results['datasets']['stackoverflow'] = result
        save_incremental(all_results)
        print("  [Saved incremental results]")
    except Exception as e:
        print(f"  ERROR on Stack Overflow: {e}")
        import traceback; traceback.print_exc()

    # ---- Final summary ----
    total_time = time.time() - t_total
    all_results['metadata']['total_time_s'] = round(total_time, 1)

    # Compute cross-dataset summary
    summary_rows = []
    for key, ds in all_results['datasets'].items():
        summary_rows.append({
            'dataset': ds['dataset'],
            'num_classes': ds['num_classes'],
            'C_mean': ds['concentration_mean'],
            'C_std': ds['concentration_std'],
            'drop_mean': ds['coverage_drop_mean'],
            'drop_std': ds['coverage_drop_std'],
            'predicted': ds['predicted_from_mean'],
            'actual': ds['actual_category'],
            'correct': ds['threshold_correct_on_mean'],
            'per_seed_correct_pct': ds['per_seed_threshold_correct_pct'],
        })
    all_results['summary'] = summary_rows

    n_correct = sum(1 for r in summary_rows if r['correct'])
    all_results['metadata']['overall_correct'] = f"{n_correct}/{len(summary_rows)}"

    save_incremental(all_results)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<25} {'Cl':>3} {'C (mean+/-std)':>16} {'Drop (mean+/-std)':>18} {'Pred':>10} {'Actual':>8} {'OK':>4} {'Seed%':>6}")
    print("-" * 95)
    for r in summary_rows:
        print(f"{r['dataset']:<25} {r['num_classes']:>3} "
              f"{r['C_mean']:>6.1f} +/- {r['C_std']:>4.1f}   "
              f"{r['drop_mean']:>7.1f} +/- {r['drop_std']:>5.1f}   "
              f"{r['predicted']:>10} {r['actual']:>8} "
              f"{'Y' if r['correct'] else 'N':>4} "
              f"{r['per_seed_correct_pct']:>5.0f}%")
    print("-" * 95)
    print(f"Overall threshold correct: {n_correct}/{len(summary_rows)}")
    print(f"Total time: {total_time:.0f}s")
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
