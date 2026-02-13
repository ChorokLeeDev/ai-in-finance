#!/usr/bin/env python3
"""
Pendigits external validation: 10 digit classes, cross-writer shift.

Pre-registered acceptance criteria (defined before seeing results):
  Gate 1: C >= 40% (high class concentration)
  Gate 2: Coverage drop >= 25pp
  Reporting: All results reported regardless of outcome.

Natural shift: 30 train writers -> 14 test writers (UCI built-in split).

Usage:
    PYTHONPATH=/Users/i767700/Github/ai-in-finance:$PYTHONPATH \
    /usr/bin/python3 -u papers/conformal_covid/code/run_pendigits_validation.py
"""

import json
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
OUTPUT_FILE = RESULTS_DIR / "pendigits_validation.json"
NUM_SEEDS = 10
SEEDS = list(range(42, 42 + NUM_SEEDS))
ALPHA = 0.1
THRESHOLD = 40.0


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
# Dataset loader
# ============================================================================

def load_pendigits():
    """Load Pendigits: 10 digit classes, cross-writer natural shift.

    UCI #81: Pen-Based Recognition of Handwritten Digits.
    30 writers for training, 14 different writers for testing.
    16 features: x,y coordinates of 8 resampled pen trajectory points.
    """
    data_dir = RESULTS_DIR / 'pendigits'
    data_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / 'pendigits.tra'
    test_file = data_dir / 'pendigits.tes'

    # Download if needed
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits"
    if not train_file.exists():
        print("  Downloading Pendigits training data from UCI...")
        urllib.request.urlretrieve(f"{base_url}/pendigits.tra", train_file)
        print("  Downloaded training data.")
    if not test_file.exists():
        print("  Downloading Pendigits test data from UCI...")
        urllib.request.urlretrieve(f"{base_url}/pendigits.tes", test_file)
        print("  Downloaded test data.")

    # Feature names: 8 pen trajectory points, each with x,y coordinate
    feature_names = []
    for i in range(8):
        feature_names.append(f'x{i+1}')
        feature_names.append(f'y{i+1}')

    # Load training data (30 writers)
    df_train_full = pd.read_csv(train_file, header=None,
                                names=feature_names + ['digit'])

    # Load test data (14 different writers)
    df_test = pd.read_csv(test_file, header=None,
                          names=feature_names + ['digit'])

    X_all_train = df_train_full[feature_names].values.astype(np.float64)
    y_all_train = df_train_full['digit'].values.astype(np.int64)

    X_test = df_test[feature_names].values.astype(np.float64)
    y_test = df_test['digit'].values.astype(np.int64)

    num_classes = len(np.unique(np.concatenate([y_all_train, y_test])))

    # Split training into train/val (80/20)
    n = len(X_all_train)
    perm = np.random.RandomState(42).permutation(n)
    n_train = int(n * 0.8)

    X_train = X_all_train[perm[:n_train]]
    y_train = y_all_train[perm[:n_train]]
    X_val = X_all_train[perm[n_train:]]
    y_val = y_all_train[perm[n_train:]]

    print(f"  Pendigits: {num_classes} classes (digits 0-9)")
    print(f"    train={len(X_train)} (from 30 writers), val={len(X_val)}, "
          f"test={len(X_test)} (14 different writers)")

    # Report class distributions
    train_dist = np.bincount(y_train, minlength=num_classes) / len(y_train) * 100
    test_dist = np.bincount(y_test, minlength=num_classes) / len(y_test) * 100
    print(f"    Train class dist: {[f'{d:.1f}%' for d in train_dist]}")
    print(f"    Test class dist:  {[f'{d:.1f}%' for d in test_dist]}")

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, feature_names


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Pendigits External Validation (10 seeds, cross-writer shift)")
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
            'description': 'Pendigits: 10 digits, cross-writer shift (30 train / 14 test writers)',
            'shift_type': 'cross-writer (natural, built-in UCI split)',
            'domain': 'handwriting/HCI',
        },
    }

    print("\nLoading Pendigits...")
    try:
        X_tr, y_tr, X_v, y_v, X_te, y_te, nc, fn = load_pendigits()
        result = run_multiseed("Pendigits", X_tr, y_tr, X_v, y_v, X_te, y_te, nc, fn, SEEDS)
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
