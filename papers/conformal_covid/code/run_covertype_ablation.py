#!/usr/bin/env python3
"""
Controlled feature ablation on Covertype: causal test of SHAP concentration.

Pre-registered predictions:
  - Ablated (no Elevation): C < 40% -> ROBUST (coverage drop < 15pp)
  - Concentrated (top-3 only): C > 40% -> CATASTROPHIC (coverage drop > 15pp)

This is a causal intervention: by manipulating which features the model sees,
we change the SHAP concentration C, and test whether the C > 40% threshold
correctly predicts conformal prediction failure under geographic shift.
"""
import json, os, sys, time, warnings
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_covtype

warnings.filterwarnings('ignore')

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_FILE = RESULTS_DIR / "covertype_ablation.json"
NUM_SEEDS = 10
SEEDS = list(range(42, 42 + NUM_SEEDS))
ALPHA = 0.1
THRESHOLD = 40.0

# From the full-model validation (covertype_validation.json):
# Top-3 features by SHAP importance:
#   Elevation (idx 0): 49.92%
#   Horizontal_Distance_To_Roadways (idx 5): 9.46%
#   Horizontal_Distance_To_Fire_Points (idx 9): 7.41%
TOP3_INDICES = [0, 5, 9]

ALL_FEATURE_NAMES = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points',
] + [f'Wilderness_Area_{i}' for i in range(1, 5)] + [f'Soil_Type_{i}' for i in range(1, 41)]


def load_covertype():
    """Load Covertype with geographic shift (wilderness area split)."""
    print("Loading Covertype dataset...")
    data = fetch_covtype()
    X, y = data.data, data.target - 1  # 0-indexed

    num_classes = len(np.unique(y))

    # Wilderness area columns are indices 10-13 (one-hot)
    area1 = X[:, 10] == 1
    area2 = X[:, 11] == 1
    area3 = X[:, 12] == 1
    area4 = X[:, 13] == 1

    # Train: wilderness 1-2; Val: wilderness 3; Test: wilderness 4
    # (matches the existing covertype_validation.py split)
    train_mask = area1 | area2
    val_mask = area3
    test_mask = area4

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Train (areas 1-2): {len(X_train)} samples")
    print(f"  Val (area 3):      {len(X_val)} samples")
    print(f"  Test (area 4):     {len(X_test)} samples")
    print(f"  Classes: {num_classes}")

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes


def train_lgb(X_train, y_train, X_val, y_val, num_classes, seed=42):
    """Train LightGBM multiclass classifier."""
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
    """Compute SHAP concentration (top-1 / total) and feature ranking."""
    import shap

    explainer = shap.TreeExplainer(model)
    n_sample = min(5000, len(X_val))
    idx = np.random.RandomState(seed).choice(len(X_val), n_sample, replace=False)
    X_sample = X_val[idx]

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

    # Top-5 ranking
    order = np.argsort(mean_abs)[::-1]
    ranking = {}
    for j in range(min(5, len(order))):
        i = int(order[j])
        fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        ranking[fname] = round(float(mean_abs[i]) / total * 100, 2)

    return concentration, top_feature, ranking


def run_aps_conformal(model, X_val, y_val, X_test, y_test, num_classes, seed=42, alpha=0.1):
    """Run APS conformal prediction with random cal/eval split."""
    val_probs = model.predict(X_val)
    test_probs = model.predict(X_test)

    # Random cal/eval split from val
    n_val = len(y_val)
    perm = np.random.RandomState(seed).permutation(n_val)
    n_cal = n_val // 2
    cal_idx, eval_idx = perm[:n_cal], perm[n_cal:]
    cal_probs, cal_y = val_probs[cal_idx], y_val[cal_idx]
    eval_probs, eval_y = val_probs[eval_idx], y_val[eval_idx]

    def aps_score(probs, true_label):
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = 0.0
        for idx in sorted_idx:
            cumsum += probs[idx]
            if idx == true_label:
                return cumsum
        return 1.0

    # Calibration scores
    cal_scores = np.array([aps_score(cal_probs[i], cal_y[i]) for i in range(len(cal_y))])
    n = len(cal_scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    qhat = float(np.quantile(cal_scores, q_level))

    # Eval (val holdout) coverage
    eval_scores = np.array([aps_score(eval_probs[i], eval_y[i]) for i in range(len(eval_y))])
    val_coverage = float(np.mean(eval_scores <= qhat) * 100)

    # Test coverage
    test_scores = np.array([aps_score(test_probs[i], y_test[i]) for i in range(len(y_test))])
    test_coverage = float(np.mean(test_scores <= qhat) * 100)

    return {
        'val_coverage': round(val_coverage, 2),
        'test_coverage': round(test_coverage, 2),
        'coverage_drop': round(val_coverage - test_coverage, 2),
        'qhat': round(qhat, 4),
    }


def run_condition(condition_name, X_train, y_train, X_val, y_val, X_test, y_test,
                  feature_names, num_classes, seeds, alpha=0.1):
    """Run one experimental condition across multiple seeds."""
    print(f"\n{'='*60}")
    print(f"CONDITION: {condition_name}")
    print(f"  Features: {len(feature_names)} ({', '.join(feature_names[:5])}{'...' if len(feature_names)>5 else ''})")
    print(f"  Seeds: {seeds}")
    print(f"{'='*60}")

    seed_results = []
    concentrations = []
    coverage_drops = []
    val_coverages = []
    test_coverages = []

    for i, seed in enumerate(seeds):
        t0 = time.time()
        print(f"\n  Seed {seed} ({i+1}/{len(seeds)})...")

        # Train
        model = train_lgb(X_train, y_train, X_val, y_val, num_classes, seed=seed)

        # Accuracy
        val_pred = np.argmax(model.predict(X_val), axis=1)
        test_pred = np.argmax(model.predict(X_test), axis=1)
        val_acc = float(np.mean(val_pred == y_val) * 100)
        test_acc = float(np.mean(test_pred == y_test) * 100)

        # SHAP
        conc, top_feat, ranking = compute_shap_concentration(model, X_val, feature_names, seed=seed)
        concentrations.append(conc)

        # Conformal
        cp = run_aps_conformal(model, X_val, y_val, X_test, y_test, num_classes, seed=seed, alpha=alpha)
        coverage_drops.append(cp['coverage_drop'])
        val_coverages.append(cp['val_coverage'])
        test_coverages.append(cp['test_coverage'])

        elapsed = time.time() - t0
        print(f"    Val acc={val_acc:.1f}%, Test acc={test_acc:.1f}%")
        print(f"    SHAP C={conc:.1f}% (top: {top_feat})")
        print(f"    Val cov={cp['val_coverage']:.1f}%, Test cov={cp['test_coverage']:.1f}%, Drop={cp['coverage_drop']:.1f}pp")
        print(f"    Time: {elapsed:.1f}s")

        seed_results.append({
            'seed': seed,
            'val_accuracy': round(val_acc, 1),
            'test_accuracy': round(test_acc, 1),
            'shap_concentration': round(conc, 2),
            'top_feature': top_feat,
            'shap_ranking': ranking,
            'val_coverage': cp['val_coverage'],
            'test_coverage': cp['test_coverage'],
            'coverage_drop': cp['coverage_drop'],
            'qhat': cp['qhat'],
        })

    # Summary statistics
    summary = {
        'condition': condition_name,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'n_seeds': len(seeds),
        'shap_concentration_mean': round(float(np.mean(concentrations)), 2),
        'shap_concentration_std': round(float(np.std(concentrations)), 2),
        'shap_concentration_min': round(float(np.min(concentrations)), 2),
        'shap_concentration_max': round(float(np.max(concentrations)), 2),
        'coverage_drop_mean': round(float(np.mean(coverage_drops)), 2),
        'coverage_drop_std': round(float(np.std(coverage_drops)), 2),
        'coverage_drop_min': round(float(np.min(coverage_drops)), 2),
        'coverage_drop_max': round(float(np.max(coverage_drops)), 2),
        'val_coverage_mean': round(float(np.mean(val_coverages)), 2),
        'test_coverage_mean': round(float(np.mean(test_coverages)), 2),
        'seed_results': seed_results,
    }

    return summary


def main():
    t_start = time.time()
    print("=" * 70)
    print("CONTROLLED FEATURE ABLATION: CAUSAL TEST OF SHAP CONCENTRATION")
    print("=" * 70)
    print(f"\nPre-registered predictions (threshold C = {THRESHOLD}%):")
    print(f"  Ablated (no Elevation):  C < {THRESHOLD}% -> ROBUST (drop < 15pp)")
    print(f"  Concentrated (top-3):    C > {THRESHOLD}% -> CATASTROPHIC (drop > 15pp)")

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes = load_covertype()

    # ---------------------------------------------------------------
    # CONDITION 1: Ablated (remove Elevation = column 0)
    # ---------------------------------------------------------------
    ablated_cols = [i for i in range(X_train.shape[1]) if i != 0]
    ablated_feature_names = [ALL_FEATURE_NAMES[i] for i in ablated_cols]

    X_train_abl = X_train[:, ablated_cols]
    X_val_abl = X_val[:, ablated_cols]
    X_test_abl = X_test[:, ablated_cols]

    ablated_results = run_condition(
        "Ablated (no Elevation)",
        X_train_abl, y_train, X_val_abl, y_val, X_test_abl, y_test,
        ablated_feature_names, num_classes, SEEDS, alpha=ALPHA,
    )

    # ---------------------------------------------------------------
    # CONDITION 2: Concentrated (top-3 features only)
    # ---------------------------------------------------------------
    concentrated_feature_names = [ALL_FEATURE_NAMES[i] for i in TOP3_INDICES]

    X_train_conc = X_train[:, TOP3_INDICES]
    X_val_conc = X_val[:, TOP3_INDICES]
    X_test_conc = X_test[:, TOP3_INDICES]

    concentrated_results = run_condition(
        "Concentrated (top-3 only)",
        X_train_conc, y_train, X_val_conc, y_val, X_test_conc, y_test,
        concentrated_feature_names, num_classes, SEEDS, alpha=ALPHA,
    )

    # ---------------------------------------------------------------
    # EVALUATION OF PRE-REGISTERED PREDICTIONS
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PRE-REGISTERED PREDICTION EVALUATION")
    print("=" * 70)

    # Gate 1: Ablated condition
    abl_C = ablated_results['shap_concentration_mean']
    abl_drop = ablated_results['coverage_drop_mean']
    abl_predicted = "CATASTROPHIC" if abl_C > THRESHOLD else "ROBUST"
    abl_actual = "CATASTROPHIC" if abl_drop > 15 else "ROBUST"
    abl_gate_C = abl_C < THRESHOLD  # We predicted C < 40%
    abl_gate_outcome = abl_drop < 15  # We predicted drop < 15pp
    abl_correct = (abl_predicted == abl_actual)

    print(f"\n  CONDITION 1: Ablated (no Elevation)")
    print(f"    SHAP C = {abl_C:.1f}% +/- {ablated_results['shap_concentration_std']:.1f}%")
    print(f"    Pre-registered gate: C < {THRESHOLD}%  ->  {'PASS' if abl_gate_C else 'FAIL'}")
    print(f"    Coverage drop = {abl_drop:.1f}pp +/- {ablated_results['coverage_drop_std']:.1f}pp")
    print(f"    Pre-registered gate: drop < 15pp  ->  {'PASS' if abl_gate_outcome else 'FAIL'}")
    print(f"    Predicted: {abl_predicted}, Actual: {abl_actual}")
    print(f"    THRESHOLD CORRECT: {abl_correct}")

    # Gate 2: Concentrated condition
    conc_C = concentrated_results['shap_concentration_mean']
    conc_drop = concentrated_results['coverage_drop_mean']
    conc_predicted = "CATASTROPHIC" if conc_C > THRESHOLD else "ROBUST"
    conc_actual = "CATASTROPHIC" if conc_drop > 15 else "ROBUST"
    conc_gate_C = conc_C > THRESHOLD  # We predicted C > 40%
    conc_gate_outcome = conc_drop > 15  # We predicted drop > 15pp
    conc_correct = (conc_predicted == conc_actual)

    print(f"\n  CONDITION 2: Concentrated (top-3 only)")
    print(f"    SHAP C = {conc_C:.1f}% +/- {concentrated_results['shap_concentration_std']:.1f}%")
    print(f"    Pre-registered gate: C > {THRESHOLD}%  ->  {'PASS' if conc_gate_C else 'FAIL'}")
    print(f"    Coverage drop = {conc_drop:.1f}pp +/- {concentrated_results['coverage_drop_std']:.1f}pp")
    print(f"    Pre-registered gate: drop > 15pp  ->  {'PASS' if conc_gate_outcome else 'FAIL'}")
    print(f"    Predicted: {conc_predicted}, Actual: {conc_actual}")
    print(f"    THRESHOLD CORRECT: {conc_correct}")

    # Overall causal test
    both_correct = abl_correct and conc_correct
    print(f"\n  OVERALL CAUSAL TEST: {'PASS' if both_correct else 'FAIL'}")
    print(f"    Both conditions match pre-registered predictions: {both_correct}")

    # Effect size: difference in coverage drop between conditions
    delta_C = conc_C - abl_C
    delta_drop = conc_drop - abl_drop
    print(f"\n  EFFECT SIZE:")
    print(f"    Delta C (concentrated - ablated):  {delta_C:+.1f}pp")
    print(f"    Delta drop (concentrated - ablated): {delta_drop:+.1f}pp")

    # Full model reference
    print(f"\n  REFERENCE (full model from covertype_validation.json):")
    print(f"    C = 49.9%, coverage drop = 83.3pp -> CATASTROPHIC (correct)")

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    output = {
        'experiment': 'Covertype Feature Ablation (Causal Test)',
        'pre_registered_threshold': THRESHOLD,
        'pre_registered_predictions': {
            'ablated': 'C < 40% -> ROBUST (drop < 15pp)',
            'concentrated': 'C > 40% -> CATASTROPHIC (drop > 15pp)',
        },
        'full_model_reference': {
            'shap_concentration': 49.92,
            'coverage_drop': 83.28,
            'outcome': 'CATASTROPHIC',
        },
        'ablated': ablated_results,
        'concentrated': concentrated_results,
        'evaluation': {
            'ablated': {
                'shap_concentration_mean': abl_C,
                'coverage_drop_mean': abl_drop,
                'predicted': abl_predicted,
                'actual': abl_actual,
                'gate_C_pass': abl_gate_C,
                'gate_outcome_pass': abl_gate_outcome,
                'threshold_correct': abl_correct,
            },
            'concentrated': {
                'shap_concentration_mean': conc_C,
                'coverage_drop_mean': conc_drop,
                'predicted': conc_predicted,
                'actual': conc_actual,
                'gate_C_pass': conc_gate_C,
                'gate_outcome_pass': conc_gate_outcome,
                'threshold_correct': conc_correct,
            },
            'both_correct': both_correct,
            'delta_C': round(delta_C, 2),
            'delta_coverage_drop': round(delta_drop, 2),
        },
        'num_seeds': NUM_SEEDS,
        'alpha': ALPHA,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

    elapsed_total = time.time() - t_start
    print(f"Total time: {elapsed_total:.1f}s")

    return output


if __name__ == '__main__':
    main()
