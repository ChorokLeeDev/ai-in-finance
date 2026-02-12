#!/usr/bin/env python3
"""
External validation: Forest Covertype Dataset (UCI/sklearn)
- 7 cover types, 54 features, 581K samples
- Cartographic features from 4 wilderness areas
- Split by wilderness area for distribution shift
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def load_covertype():
    """Load Covertype from sklearn."""
    from sklearn.datasets import fetch_covtype

    print("Loading Covertype dataset...")
    data = fetch_covtype()
    X = data.data
    y = data.target - 1  # 0-index the labels (originally 1-7)

    feature_names = data.feature_names if hasattr(data, 'feature_names') else [f'feat_{i}' for i in range(X.shape[1])]
    num_classes = len(np.unique(y))

    print(f"Shape: {X.shape}")
    print(f"Classes: {num_classes}")
    print(f"Class distribution:")
    for cls, cnt in sorted(pd.Series(y).value_counts().items()):
        print(f"  Class {cls}: {cnt}")

    return X, y, num_classes, list(feature_names)


def split_by_wilderness_area(X, y):
    """Split by wilderness area columns (features 10-13 are one-hot wilderness area).
    Train on areas 1-2, val on area 3, test on area 4 for geographic shift."""
    # Columns 10-13 are Wilderness_Area1-4 (binary one-hot)
    area1 = X[:, 10] == 1
    area2 = X[:, 11] == 1
    area3 = X[:, 12] == 1
    area4 = X[:, 13] == 1

    train_mask = area1 | area2
    val_mask = area3
    test_mask = area4

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Split by wilderness area:")
    print(f"  Train (areas 1-2): {len(X_train)} samples")
    print(f"  Val (area 3): {len(X_val)} samples")
    print(f"  Test (area 4): {len(X_test)} samples")

    # Check class distributions
    for name, ys in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        dist = pd.Series(ys).value_counts().sort_index()
        print(f"  {name} classes: {dict(dist)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(X_train, y_train, X_val, y_val, num_classes, seed=42):
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


def compute_shap_concentration(model, X_val, feature_names):
    """Compute SHAP concentration (top-1 / total)."""
    import shap

    explainer = shap.TreeExplainer(model)
    n_sample = min(10000, len(X_val))
    idx = np.random.RandomState(42).choice(len(X_val), n_sample, replace=False)
    shap_values = explainer.shap_values(X_val[idx])

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

    order = np.argsort(mean_abs)[::-1]
    ranking = {}
    for j in range(min(5, len(order))):
        i = int(order[j])
        fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        ranking[fname] = round(mean_abs[i] / total * 100, 2)

    return concentration, top_feature, ranking


def run_aps_conformal(model, X_val, y_val, X_test, y_test, num_classes, alpha=0.1):
    """Run APS conformal prediction."""
    val_probs = model.predict(X_val)
    test_probs = model.predict(X_test)

    # Split val into cal/eval
    n_cal = len(X_val) // 2
    cal_probs, eval_probs = val_probs[:n_cal], val_probs[n_cal:]
    cal_y, eval_y = y_val[:n_cal], y_val[n_cal:]

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

    val_set_sizes = []
    for i in range(len(eval_probs)):
        sorted_idx = np.argsort(eval_probs[i])[::-1]
        cumsum = 0.0
        size = 0
        for idx in sorted_idx:
            cumsum += eval_probs[i][idx]
            size += 1
            if cumsum >= qhat:
                break
        val_set_sizes.append(size)

    test_scores = np.array([aps_score(test_probs[i], y_test[i]) for i in range(len(y_test))])
    test_coverage = float(np.mean(test_scores <= qhat) * 100)

    test_set_sizes = []
    for i in range(len(test_probs)):
        sorted_idx = np.argsort(test_probs[i])[::-1]
        cumsum = 0.0
        size = 0
        for idx in sorted_idx:
            cumsum += test_probs[i][idx]
            size += 1
            if cumsum >= qhat:
                break
        test_set_sizes.append(size)

    return {
        'val_coverage': round(val_coverage, 2),
        'test_coverage': round(test_coverage, 2),
        'coverage_drop': round(val_coverage - test_coverage, 2),
        'val_mean_set_size': round(float(np.mean(val_set_sizes)), 2),
        'test_mean_set_size': round(float(np.mean(test_set_sizes)), 2),
        'n_cal': n_cal,
        'n_eval': len(eval_y),
        'n_test': len(y_test),
        'qhat': round(float(qhat), 4),
    }


def main():
    print("=" * 60)
    print("External Validation: Forest Covertype")
    print("=" * 60)

    X, y, num_classes, feature_names = load_covertype()

    # Geographic split by wilderness area
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_wilderness_area(X, y)

    # Train model
    print("\nTraining LightGBM...")
    model = train_model(X_train, y_train, X_val, y_val, num_classes)

    # Accuracy
    val_pred = np.argmax(model.predict(X_val), axis=1)
    test_pred = np.argmax(model.predict(X_test), axis=1)
    val_acc = float(np.mean(val_pred == y_val) * 100)
    test_acc = float(np.mean(test_pred == y_test) * 100)
    print(f"Accuracy: val={val_acc:.1f}%, test={test_acc:.1f}%")

    # SHAP concentration
    print("\nComputing SHAP concentration...")
    concentration, top_feature, ranking = compute_shap_concentration(model, X_val, feature_names)
    print(f"SHAP Concentration: {concentration:.2f}%")
    print(f"Top feature: {top_feature}")
    print(f"Feature ranking: {ranking}")

    # APS conformal
    print("\nRunning APS conformal prediction...")
    conformal = run_aps_conformal(model, X_val, y_val, X_test, y_test, num_classes)
    print(f"Val coverage: {conformal['val_coverage']}%")
    print(f"Test coverage: {conformal['test_coverage']}%")
    print(f"Coverage drop: {conformal['coverage_drop']}pp")
    print(f"Mean set size: val={conformal['val_mean_set_size']}, test={conformal['test_mean_set_size']}")

    # Assessment
    threshold = 40.0
    predicted = "vulnerable" if concentration > threshold else "robust"
    actual_drop = conformal['coverage_drop']
    actual = "severe" if actual_drop > 15 else "robust"
    correct = (predicted == "vulnerable" and actual == "severe") or (predicted == "robust" and actual == "robust")

    results = {
        'dataset': 'Forest Covertype',
        'source': 'UCI / sklearn',
        'domain': 'Ecological remote sensing',
        'num_classes': num_classes,
        'n_features': len(feature_names),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'val_accuracy': round(val_acc, 1),
        'test_accuracy': round(test_acc, 1),
        'shap': {
            'concentration': round(concentration, 2),
            'top_feature': top_feature,
            'feature_ranking': ranking,
        },
        'conformal': conformal,
        'assessment': {
            'threshold': threshold,
            'predicted': predicted,
            'actual_category': actual,
            'actual_drop': actual_drop,
            'threshold_correct': str(correct),
        },
    }

    out_path = os.path.join(RESULTS_DIR, 'covertype_validation.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 60)
    print(f"ASSESSMENT: Concentration={concentration:.1f}% → predict '{predicted}'")
    print(f"            Actual drop={actual_drop}pp → '{actual}'")
    print(f"            Threshold correct: {correct}")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()
