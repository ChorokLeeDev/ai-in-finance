#!/usr/bin/env python3
"""
Phase 2 external validation: 3 new multiclass datasets with natural shift.
  1. Avila Bible (12 classes, humanities, temporal/geographic shift)
  2. Shuttle / Statlog (7 classes, aerospace, temporal shift)
  3. PAMAP2 (12 activities, wearable/health, cross-person shift)

Follows exact same pattern as run_external_multiseed.py.

Usage:
    PYTHONPATH=/Users/i767700/Github/ai-in-finance:$PYTHONPATH \
    /usr/bin/python3 -u papers/conformal_covid/code/run_external_phase2.py
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import urllib.request

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_FILE = RESULTS_DIR / "external_phase2_validation.json"
NUM_SEEDS = 10
SEEDS = list(range(42, 42 + NUM_SEEDS))
ALPHA = 0.1
THRESHOLD = 40.0


# ============================================================================
# Shared utilities (copied from run_external_multiseed.py for standalone use)
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

        model = train_lgb(X_train, y_train, X_val, y_val, num_classes, seed=seed)
        concentration, top_feature = compute_shap_concentration(
            model, X_val, feature_names, seed=seed)
        conformal = run_aps_conformal(
            model, X_val, y_val, X_test, y_test, num_classes, seed=seed, alpha=ALPHA)

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

    concentrations = [r['concentration'] for r in seed_results]
    drops = [r['coverage_drop'] for r in seed_results]
    val_covs = [r['val_coverage'] for r in seed_results]
    test_covs = [r['test_coverage'] for r in seed_results]

    mean_concentration = float(np.mean(concentrations))
    std_concentration = float(np.std(concentrations))
    mean_drop = float(np.mean(drops))
    std_drop = float(np.std(drops))

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
# NEW Dataset loaders
# ============================================================================

def load_avila():
    """Load Avila Bible dataset: 12 copyists, pre-split train/test.

    The Avila dataset contains features extracted from 800 images of pages
    from the 12th-century Avila Bible, written by different copyists.
    The train/test split has natural distribution shift from different
    page collections.
    """
    from sklearn.preprocessing import LabelEncoder

    data_dir = RESULTS_DIR / 'avila'
    data_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / 'avila-tr.txt'
    test_file = data_dir / 'avila-ts.txt'

    # Download if needed
    if not train_file.exists():
        print("  Downloading Avila dataset from UCI...")
        url = "https://archive.ics.uci.edu/static/public/459/avila.zip"
        response = urllib.request.urlopen(url)
        zipdata = BytesIO(response.read())
        with ZipFile(zipdata) as zf:
            for name in zf.namelist():
                if name.endswith('avila-tr.txt'):
                    with open(train_file, 'wb') as f:
                        f.write(zf.read(name))
                elif name.endswith('avila-ts.txt'):
                    with open(test_file, 'wb') as f:
                        f.write(zf.read(name))
        print("  Downloaded and extracted.")

    feature_names = [
        'intercolumnar_distance', 'upper_margin', 'lower_margin',
        'exploitation', 'row_number', 'modular_ratio',
        'interlinear_spacing', 'weight', 'peak_number',
        'modular_ratio_interlinear'
    ]

    # Load train
    df_train = pd.read_csv(train_file, header=None,
                           names=feature_names + ['copyist'])
    # Load test
    df_test = pd.read_csv(test_file, header=None,
                          names=feature_names + ['copyist'])

    # Encode labels
    le = LabelEncoder()
    le.fit(pd.concat([df_train['copyist'], df_test['copyist']]))

    X_all_train = df_train[feature_names].values.astype(np.float64)
    y_all_train = le.transform(df_train['copyist'])

    X_test = df_test[feature_names].values.astype(np.float64)
    y_test = le.transform(df_test['copyist'])

    num_classes = len(le.classes_)

    # Split train into train/val (80/20)
    n = len(X_all_train)
    perm = np.random.RandomState(42).permutation(n)
    n_train = int(n * 0.8)

    X_train = X_all_train[perm[:n_train]]
    y_train = y_all_train[perm[:n_train]]
    X_val = X_all_train[perm[n_train:]]
    y_val = y_all_train[perm[n_train:]]

    print(f"  Avila: {num_classes} classes ({list(le.classes_)})")
    print(f"    train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, feature_names


def load_shuttle():
    """Load Shuttle/Statlog: 7 classes, temporal shift via official split.

    Radiator control data from Space Shuttle with temporal ordering.
    Official train/test split has known distribution shift.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder

    print("  Loading Shuttle from OpenML...")
    data = fetch_openml(name='shuttle', version=1, as_frame=False, parser='auto')
    X = data.data.astype(np.float64)
    y_raw = data.target

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    feature_names = [f'attr_{i+1}' for i in range(X.shape[1])]
    if hasattr(data, 'feature_names') and data.feature_names is not None:
        feature_names = list(data.feature_names)

    # Shuttle has temporal ordering; use positional split
    # First 60% train, next 20% val, last 20% test
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"  Shuttle: {num_classes} classes")
    print(f"    train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Report class distributions
    train_dist = np.bincount(y_train, minlength=num_classes) / len(y_train) * 100
    test_dist = np.bincount(y_test, minlength=num_classes) / len(y_test) * 100
    print(f"    Train class dist: {[f'{d:.1f}%' for d in train_dist]}")
    print(f"    Test class dist:  {[f'{d:.1f}%' for d in test_dist]}")

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, feature_names


def load_pamap2():
    """Load PAMAP2 Physical Activity Monitoring: 12 activities, subject-based shift.

    IMU sensor data from 9 subjects performing 12 activities.
    Train on subjects 1-6, val on subject 7, test on subjects 8-9.
    This is a canonical domain generalization benchmark.
    """
    data_dir = RESULTS_DIR / 'pamap2'
    data_dir.mkdir(parents=True, exist_ok=True)

    protocol_dir = data_dir / 'Protocol'

    # Download if needed
    if not protocol_dir.exists():
        print("  Downloading PAMAP2 dataset from UCI...")
        url = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
        try:
            response = urllib.request.urlopen(url, timeout=120)
            zipdata = BytesIO(response.read())
            with ZipFile(zipdata) as zf:
                # Extract only Protocol files
                for name in zf.namelist():
                    if 'Protocol/subject' in name and name.endswith('.dat'):
                        # Extract to data_dir
                        target = data_dir / name.split('PAMAP2_Dataset/')[-1] if 'PAMAP2_Dataset/' in name else data_dir / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with open(target, 'wb') as f:
                            f.write(zf.read(name))
            print("  Downloaded and extracted.")
        except Exception as e:
            print(f"  Download failed: {e}")
            print("  Trying alternative nested zip structure...")
            # UCI sometimes nests the zip
            response = urllib.request.urlopen(url, timeout=120)
            outer_data = BytesIO(response.read())
            with ZipFile(outer_data) as outer_zf:
                for outer_name in outer_zf.namelist():
                    if outer_name.endswith('.zip'):
                        inner_data = BytesIO(outer_zf.read(outer_name))
                        with ZipFile(inner_data) as inner_zf:
                            for name in inner_zf.namelist():
                                if 'Protocol/subject' in name and name.endswith('.dat'):
                                    basename = os.path.basename(name)
                                    protocol_dir.mkdir(parents=True, exist_ok=True)
                                    with open(protocol_dir / basename, 'wb') as f:
                                        f.write(inner_zf.read(name))
            print("  Downloaded and extracted (nested zip).")

    # Column names for PAMAP2 (54 columns total)
    # timestamp, activityID, heart_rate, then 3 IMUs x 17 features
    imu_features = ['temperature', 'acc_x_16g', 'acc_y_16g', 'acc_z_16g',
                    'acc_x_6g', 'acc_y_6g', 'acc_z_6g',
                    'gyro_x', 'gyro_y', 'gyro_z',
                    'mag_x', 'mag_y', 'mag_z',
                    'orientation_1', 'orientation_2', 'orientation_3', 'orientation_4']
    imu_locations = ['hand', 'chest', 'ankle']

    col_names = ['timestamp', 'activityID', 'heart_rate']
    for loc in imu_locations:
        for feat in imu_features:
            col_names.append(f'{loc}_{feat}')

    # Load all subjects
    all_data = []
    all_subjects = []

    # Find protocol directory
    possible_dirs = [
        protocol_dir,
        data_dir / 'PAMAP2_Dataset' / 'Protocol',
        data_dir,
    ]

    actual_dir = None
    for d in possible_dirs:
        if d.exists() and list(d.glob('subject*.dat')):
            actual_dir = d
            break

    if actual_dir is None:
        raise FileNotFoundError(f"Cannot find PAMAP2 subject files in {data_dir}")

    for subject_id in range(1, 10):  # subjects 1-9
        fpath = actual_dir / f'subject10{subject_id}.dat'
        if not fpath.exists():
            # Try alternative naming
            fpath = actual_dir / f'subject{subject_id}.dat'
        if not fpath.exists():
            print(f"    Warning: subject {subject_id} file not found, skipping")
            continue

        df = pd.read_csv(fpath, sep=' ', header=None, names=col_names[:min(len(col_names), 54)])
        df['subject'] = subject_id
        all_data.append(df)
        all_subjects.append(subject_id)
        print(f"    Loaded subject {subject_id}: {len(df)} samples")

    df_all = pd.concat(all_data, ignore_index=True)

    # Filter to the 12 basic activities (IDs 1-7, 12-13, 16-17, 24)
    # Common activities: 1=lying, 2=sitting, 3=standing, 4=walking, 5=running,
    # 6=cycling, 7=Nordic walking, 12=ascending stairs, 13=descending stairs,
    # 16=vacuum cleaning, 17=ironing, 24=rope jumping
    valid_activities = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
    df_all = df_all[df_all['activityID'].isin(valid_activities)].copy()

    # Drop columns with >50% NaN
    feature_cols = [c for c in df_all.columns if c not in ['timestamp', 'activityID', 'subject']]
    nan_frac = df_all[feature_cols].isna().mean()
    keep_cols = nan_frac[nan_frac < 0.5].index.tolist()

    # Fill remaining NaN with column median
    df_all[keep_cols] = df_all[keep_cols].fillna(df_all[keep_cols].median())

    # Drop any remaining NaN rows
    df_all = df_all.dropna(subset=keep_cols)

    # Encode activities
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_all = le.fit_transform(df_all['activityID'].values)
    X_all = df_all[keep_cols].values.astype(np.float64)
    subjects = df_all['subject'].values
    num_classes = len(le.classes_)

    # Split by subject: train=1-6, val=7, test=8-9
    train_mask = np.isin(subjects, [1, 2, 3, 4, 5, 6])
    val_mask = subjects == 7
    test_mask = np.isin(subjects, [8, 9])

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    print(f"  PAMAP2: {num_classes} activities, {len(keep_cols)} features")
    print(f"    train={len(X_train)} (subj 1-6), val={len(X_val)} (subj 7), test={len(X_test)} (subj 8-9)")

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, keep_cols


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Phase 2 External Validation (10 seeds x 3 new datasets)")
    print("=" * 70)
    t_total = time.time()

    all_results = {
        'metadata': {
            'num_seeds': NUM_SEEDS,
            'seeds': SEEDS,
            'alpha': ALPHA,
            'threshold': THRESHOLD,
            'description': 'Phase 2 external validation: Avila, Shuttle, PAMAP2',
        },
        'datasets': {},
    }

    # ---- 1. Avila Bible ----
    print("\n[1/3] Loading Avila Bible...")
    try:
        X_tr, y_tr, X_v, y_v, X_te, y_te, nc, fn = load_avila()
        result = run_multiseed("Avila Bible", X_tr, y_tr, X_v, y_v, X_te, y_te, nc, fn, SEEDS)
        all_results['datasets']['avila'] = result
        save_incremental(all_results)
        print("  [Saved incremental results]")
    except Exception as e:
        print(f"  ERROR on Avila: {e}")
        import traceback; traceback.print_exc()

    # ---- 2. Shuttle ----
    print("\n[2/3] Loading Shuttle...")
    try:
        X_tr, y_tr, X_v, y_v, X_te, y_te, nc, fn = load_shuttle()
        result = run_multiseed("Shuttle (Statlog)", X_tr, y_tr, X_v, y_v, X_te, y_te, nc, fn, SEEDS)
        all_results['datasets']['shuttle'] = result
        save_incremental(all_results)
        print("  [Saved incremental results]")
    except Exception as e:
        print(f"  ERROR on Shuttle: {e}")
        import traceback; traceback.print_exc()

    # ---- 3. PAMAP2 ----
    print("\n[3/3] Loading PAMAP2...")
    try:
        X_tr, y_tr, X_v, y_v, X_te, y_te, nc, fn = load_pamap2()
        result = run_multiseed("PAMAP2 Activity", X_tr, y_tr, X_v, y_v, X_te, y_te, nc, fn, SEEDS)
        all_results['datasets']['pamap2'] = result
        save_incremental(all_results)
        print("  [Saved incremental results]")
    except Exception as e:
        print(f"  ERROR on PAMAP2: {e}")
        import traceback; traceback.print_exc()

    # ---- Final summary ----
    total_time = time.time() - t_total
    all_results['metadata']['total_time_s'] = round(total_time, 1)

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
    print("PHASE 2 FINAL SUMMARY")
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
