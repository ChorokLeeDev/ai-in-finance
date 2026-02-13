#!/usr/bin/env python3
"""
CWRU Bearing Fault external validation: 10 fault classes, cross-load shift.

Pre-registered acceptance criteria (defined before seeing results):
  Gate 1: C >= 40% (high class concentration)
  Gate 2: Coverage drop >= 25pp
  Reporting: All results reported regardless of outcome.

Natural shift: Train on loads 0-1 HP, test on loads 2-3 HP.
Domain: Industrial predictive maintenance (NEW domain for paper).

Data source: Case Western Reserve University Bearing Data Center
https://engineering.case.edu/bearingdatacenter

Usage:
    PYTHONPATH=/Users/i767700/Github/ai-in-finance:$PYTHONPATH \
    /usr/bin/python3 -u papers/conformal_covid/code/run_cwru_bearing_validation.py
"""

import json
import os
import time
import warnings
from pathlib import Path
from io import BytesIO
import urllib.request

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis, skew

warnings.filterwarnings('ignore')

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_FILE = RESULTS_DIR / "cwru_bearing_validation.json"
DATA_DIR = RESULTS_DIR / "cwru_bearing"
NUM_SEEDS = 10
SEEDS = list(range(42, 42 + NUM_SEEDS))
ALPHA = 0.1
THRESHOLD = 40.0
WINDOW_SIZE = 2048  # samples per segment
STEP_SIZE = 1024    # 50% overlap


# ============================================================================
# CWRU Data Download & Feature Extraction
# ============================================================================

# File mapping: (file_number, fault_type, fault_diameter, load_hp)
# 12K Drive End data
CWRU_FILES = {
    # Normal baseline
    '97':  ('Normal', 0, 0),
    '98':  ('Normal', 0, 1),
    '99':  ('Normal', 0, 2),
    '100': ('Normal', 0, 3),
    # Inner Race 0.007"
    '105': ('IR007', 0.007, 0),
    '106': ('IR007', 0.007, 1),
    '107': ('IR007', 0.007, 2),
    '108': ('IR007', 0.007, 3),
    # Inner Race 0.014"
    '169': ('IR014', 0.014, 0),
    '170': ('IR014', 0.014, 1),
    '171': ('IR014', 0.014, 2),
    '172': ('IR014', 0.014, 3),
    # Inner Race 0.021"
    '209': ('IR021', 0.021, 0),
    '210': ('IR021', 0.021, 1),
    '211': ('IR021', 0.021, 2),
    '212': ('IR021', 0.021, 3),
    # Ball 0.007"
    '118': ('B007', 0.007, 0),
    '119': ('B007', 0.007, 1),
    '120': ('B007', 0.007, 2),
    '121': ('B007', 0.007, 3),
    # Ball 0.014"
    '185': ('B014', 0.014, 0),
    '186': ('B014', 0.014, 1),
    '187': ('B014', 0.014, 2),
    '188': ('B014', 0.014, 3),
    # Ball 0.021"
    '222': ('B021', 0.021, 0),
    '223': ('B021', 0.021, 1),
    '224': ('B021', 0.021, 2),
    '225': ('B021', 0.021, 3),
    # Outer Race 0.007" (@6:00)
    '130': ('OR007', 0.007, 0),
    '131': ('OR007', 0.007, 1),
    '132': ('OR007', 0.007, 2),
    '133': ('OR007', 0.007, 3),
    # Outer Race 0.014"
    '197': ('OR014', 0.014, 0),
    '198': ('OR014', 0.014, 1),
    '199': ('OR014', 0.014, 2),
    '200': ('OR014', 0.014, 3),
    # Outer Race 0.021"
    '234': ('OR021', 0.021, 0),
    '235': ('OR021', 0.021, 1),
    '236': ('OR021', 0.021, 2),
    '237': ('OR021', 0.021, 3),
}

# Class labels (10 classes)
CLASS_NAMES = ['Normal', 'IR007', 'IR014', 'IR021', 'B007', 'B014', 'B021',
               'OR007', 'OR014', 'OR021']
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


def download_cwru_file(file_num):
    """Download a single CWRU .mat file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_path = DATA_DIR / f'{file_num}.mat'

    if local_path.exists():
        return local_path

    # CWRU data center direct download URL
    url = f"https://engineering.case.edu/bearingdatacenter/download-data-file/{file_num}"
    print(f"    Downloading file {file_num}.mat...")
    try:
        urllib.request.urlretrieve(url, local_path)
    except Exception as e:
        print(f"    Failed to download {file_num}: {e}")
        # Try alternative URL format
        url2 = f"https://engineering.case.edu/sites/default/files/{file_num}.mat"
        try:
            urllib.request.urlretrieve(url2, local_path)
        except Exception as e2:
            print(f"    Alternative URL also failed: {e2}")
            return None

    return local_path


def extract_de_signal(mat_path):
    """Extract Drive End accelerometer signal from .mat file."""
    data = loadmat(str(mat_path))

    # Find the DE (Drive End) time series key
    # Keys follow pattern: X{NNN}_DE_time or similar
    de_key = None
    for key in data.keys():
        if 'DE_time' in key:
            de_key = key
            break

    if de_key is None:
        # Fallback: look for any array that's a long time series
        for key in data.keys():
            if not key.startswith('_'):
                arr = data[key]
                if hasattr(arr, 'shape') and len(arr.shape) >= 1 and arr.shape[0] > 10000:
                    de_key = key
                    break

    if de_key is None:
        raise ValueError(f"Cannot find DE signal in {mat_path}. Keys: {list(data.keys())}")

    signal = data[de_key].flatten().astype(np.float64)
    return signal


def extract_features(signal, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Extract statistical features from vibration signal windows.

    Features per window:
    1. Mean
    2. Std
    3. RMS (root mean square)
    4. Peak (max absolute value)
    5. Crest factor (peak / RMS)
    6. Kurtosis
    7. Skewness
    8. Peak-to-peak
    9. Shape factor (RMS / mean of absolute values)
    10. Impulse factor (peak / mean of absolute values)
    11. Clearance factor (peak / (mean of sqrt(abs))^2)
    12. Energy (sum of squared values / N)
    """
    n_windows = (len(signal) - window_size) // step_size + 1
    features = []

    for i in range(n_windows):
        start = i * step_size
        window = signal[start:start + window_size]

        mean_val = np.mean(window)
        std_val = np.std(window)
        rms = np.sqrt(np.mean(window ** 2))
        peak = np.max(np.abs(window))
        abs_mean = np.mean(np.abs(window))
        sqrt_mean = np.mean(np.sqrt(np.abs(window)))

        feat = [
            mean_val,                                    # mean
            std_val,                                     # std
            rms,                                         # RMS
            peak,                                        # peak
            peak / rms if rms > 0 else 0,               # crest factor
            kurtosis(window),                            # kurtosis
            skew(window),                                # skewness
            np.max(window) - np.min(window),             # peak-to-peak
            rms / abs_mean if abs_mean > 0 else 0,       # shape factor
            peak / abs_mean if abs_mean > 0 else 0,      # impulse factor
            peak / (sqrt_mean ** 2) if sqrt_mean > 0 else 0,  # clearance factor
            np.mean(window ** 2),                        # energy
        ]
        features.append(feat)

    return np.array(features)


FEATURE_NAMES = ['mean', 'std', 'rms', 'peak', 'crest_factor', 'kurtosis',
                 'skewness', 'peak_to_peak', 'shape_factor', 'impulse_factor',
                 'clearance_factor', 'energy']


def load_cwru_bearing():
    """Load CWRU Bearing dataset with cross-load shift.

    Split: Train on loads 0-1 HP, val from loads 0-1 HP (held out),
           test on loads 2-3 HP (shift).
    """
    print("  Downloading and processing CWRU Bearing Fault data...")
    print(f"  Window size: {WINDOW_SIZE}, Step size: {STEP_SIZE}")

    all_features = []
    all_labels = []
    all_loads = []

    for file_num, (fault_type, fault_diam, load_hp) in sorted(CWRU_FILES.items(), key=lambda x: x[0]):
        mat_path = download_cwru_file(file_num)
        if mat_path is None:
            print(f"    Skipping file {file_num} (download failed)")
            continue

        try:
            signal = extract_de_signal(mat_path)
            features = extract_features(signal)
            n_windows = len(features)

            label_idx = CLASS_TO_IDX[fault_type]
            all_features.append(features)
            all_labels.append(np.full(n_windows, label_idx))
            all_loads.append(np.full(n_windows, load_hp))

            print(f"    File {file_num}: {fault_type} load={load_hp}HP -> {n_windows} windows")
        except Exception as e:
            print(f"    Error processing file {file_num}: {e}")
            continue

    if not all_features:
        raise RuntimeError("No data files were successfully processed!")

    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    loads = np.concatenate(all_loads)

    num_classes = len(CLASS_NAMES)

    # Cross-load split: train/val from loads 0-1, test from loads 2-3
    train_val_mask = (loads == 0) | (loads == 1)
    test_mask = (loads == 2) | (loads == 3)

    X_train_val = X[train_val_mask]
    y_train_val = y[train_val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    # Split train_val into train (80%) and val (20%)
    n = len(X_train_val)
    perm = np.random.RandomState(42).permutation(n)
    n_train = int(n * 0.8)

    X_train = X_train_val[perm[:n_train]]
    y_train = y_train_val[perm[:n_train]]
    X_val = X_train_val[perm[n_train:]]
    y_val = y_train_val[perm[n_train:]]

    print(f"\n  CWRU Bearing: {num_classes} classes ({CLASS_NAMES})")
    print(f"    train={len(X_train)} (loads 0-1), val={len(X_val)} (loads 0-1), "
          f"test={len(X_test)} (loads 2-3)")
    print(f"    Features: {FEATURE_NAMES}")

    # Report class distributions
    train_dist = np.bincount(y_train, minlength=num_classes) / len(y_train) * 100
    test_dist = np.bincount(y_test, minlength=num_classes) / len(y_test) * 100
    print(f"    Train class dist: {[f'{d:.1f}%' for d in train_dist]}")
    print(f"    Test class dist:  {[f'{d:.1f}%' for d in test_dist]}")

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, FEATURE_NAMES


# ============================================================================
# Shared utilities (same as run_external_phase2.py)
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

    # Pre-registered gate check
    print(f"\n  PRE-REGISTERED GATE CHECK:")
    print(f"    Gate 1 (C >= 40%):          {'PASS' if mean_concentration >= 40 else 'FAIL'} (C={mean_concentration:.1f}%)")
    print(f"    Gate 2 (drop >= 25pp):       {'PASS' if mean_drop >= 25 else 'FAIL'} (drop={mean_drop:.1f}pp)")
    if mean_concentration >= 40 and mean_drop >= 25:
        print(f"    ==> CATASTROPHIC candidate confirmed")
    else:
        print(f"    ==> Classified as ROBUST (adds to correct-prediction evidence)")

    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("CWRU Bearing Fault External Validation (10 seeds, cross-load shift)")
    print("Pre-registered gates: C >= 40%, coverage drop >= 25pp")
    print("=" * 70)
    t_total = time.time()

    all_results = {
        'metadata': {
            'num_seeds': NUM_SEEDS,
            'seeds': SEEDS,
            'alpha': ALPHA,
            'threshold': THRESHOLD,
            'preregistered_gates': {
                'gate1_concentration': '>= 40%',
                'gate2_coverage_drop': '>= 25pp',
            },
            'description': 'CWRU Bearing Fault: 10 classes, cross-load shift (0-1 HP -> 2-3 HP)',
            'shift_type': 'cross-load (natural operating condition shift)',
            'domain': 'industrial/predictive maintenance',
            'feature_extraction': {
                'window_size': WINDOW_SIZE,
                'step_size': STEP_SIZE,
                'features': FEATURE_NAMES,
            },
        },
    }

    print("\nLoading CWRU Bearing Fault data...")
    try:
        X_tr, y_tr, X_v, y_v, X_te, y_te, nc, fn = load_cwru_bearing()
        result = run_multiseed("CWRU Bearing Fault", X_tr, y_tr, X_v, y_v, X_te, y_te, nc, fn, SEEDS)
        all_results['result'] = result
        save_incremental(all_results)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        all_results['error'] = str(e)
        save_incremental(all_results)
        return

    total_time = time.time() - t_total
    all_results['metadata']['total_time_s'] = round(total_time, 1)
    save_incremental(all_results)

    print(f"\nTotal time: {total_time:.0f}s")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
