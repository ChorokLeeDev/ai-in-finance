#!/usr/bin/env python3
"""
External validation: Gas Sensor Array Drift Dataset (UCI #224)
- 6 gas classes, 128 features, 13,910 samples
- 10 temporal batches over 36 months (known sensor drift)
- Train on early batches, test on late batches
- Compute SHAP concentration + APS conformal prediction
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

def load_gas_sensor_data():
    """Load Gas Sensor Array Drift Dataset from local .dat files (libsvm format)."""
    data_dir = os.path.join(RESULTS_DIR, 'gas_sensor_drift', 'Dataset')

    all_rows = []
    all_labels = []
    all_batches = []

    for batch_id in range(1, 11):
        fpath = os.path.join(data_dir, f'batch{batch_id}.dat')
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

    X = pd.DataFrame(all_rows, columns=[f'sensor_{i+1}' for i in range(128)])
    y = pd.DataFrame({'class': all_labels, 'batch': all_batches})

    print(f"Loaded Gas Sensor Array Drift: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {sorted(y['class'].unique())}, Batches: {sorted(y['batch'].unique())}")

    return X, y


def prepare_data(X, y):
    """Split by batch: early batches for train, middle for val, late for test."""
    labels = y['class'].values
    batches = y['batch'].values

    # Encode labels to 0-indexed
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")
    print(f"Class distribution: {pd.Series(labels).value_counts().sort_index().to_dict()}")

    # Split: batches 1-6 train, 7-8 val, 9-10 test (temporal drift)
    train_batches = [1, 2, 3, 4, 5, 6]
    val_batches = [7, 8]
    test_batches = [9, 10]

    print(f"Train batches: {train_batches}")
    print(f"Val batches: {val_batches}")
    print(f"Test batches: {test_batches}")

    train_mask = np.isin(batches, train_batches)
    val_mask = np.isin(batches, val_batches)
    test_mask = np.isin(batches, test_batches)

    X_train, y_train = X.values[train_mask], labels_encoded[train_mask]
    X_val, y_val = X.values[val_mask], labels_encoded[val_mask]
    X_test, y_test = X.values[test_mask], labels_encoded[test_mask]

    print(f"\nSplit sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, X.columns.tolist()


def train_model(X_train, y_train, X_val, y_val, num_classes, seed=42):
    """Train LightGBM classifier."""
    import lightgbm as lgb

    objective = 'multiclass' if num_classes > 2 else 'binary'

    params = {
        'objective': objective,
        'num_class': num_classes if num_classes > 2 else 1,
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

    # For multiclass, shap_values may be list of arrays or 3D array
    if isinstance(shap_values, list):
        # Average absolute SHAP across classes
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # 3D: (n_samples, n_features, n_classes)
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    # Flatten to 1D if needed
    mean_abs = np.array(mean_abs).flatten()
    print(f"  mean_abs shape: {mean_abs.shape}")

    total = float(mean_abs.sum())
    top1 = float(mean_abs.max())
    concentration = (top1 / total * 100) if total > 0 else 0

    top_idx = int(np.argmax(mean_abs))
    top_feature = feature_names[top_idx] if feature_names else f"feature_{top_idx}"

    # Feature ranking (top 5)
    order = np.argsort(mean_abs)[::-1]
    ranking = {}
    for j in range(min(5, len(order))):
        i = int(order[j])
        fname = feature_names[i] if feature_names else f"feature_{i}"
        ranking[fname] = round(mean_abs[i] / total * 100, 2)

    return concentration, top_feature, ranking


def run_aps_conformal(model, X_val, y_val, X_test, y_test, num_classes, alpha=0.1):
    """Run APS conformal prediction."""
    # Get probability predictions
    val_probs = model.predict(X_val)
    test_probs = model.predict(X_test)

    if num_classes == 2:
        # Binary: model.predict returns 1D probabilities for class 1
        val_probs = np.column_stack([1 - val_probs, val_probs])
        test_probs = np.column_stack([1 - test_probs, test_probs])

    # Split val into cal/eval
    n_cal = len(X_val) // 2
    cal_probs, eval_probs = val_probs[:n_cal], val_probs[n_cal:]
    cal_y, eval_y = y_val[:n_cal], y_val[n_cal:]

    def aps_score(probs, true_label):
        """APS conformity score: cumulative prob mass until true label included."""
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = 0.0
        for idx in sorted_idx:
            cumsum += probs[idx]
            if idx == true_label:
                return cumsum
        return 1.0

    # Calibration scores
    cal_scores = np.array([aps_score(cal_probs[i], cal_y[i]) for i in range(len(cal_y))])

    # Quantile
    n = len(cal_scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    qhat = np.quantile(cal_scores, q_level)

    # Evaluate on val-eval
    eval_scores = np.array([aps_score(eval_probs[i], eval_y[i]) for i in range(len(eval_y))])
    val_coverage = float(np.mean(eval_scores <= qhat) * 100)

    # Prediction set sizes on val-eval
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

    # Evaluate on test
    test_scores = np.array([aps_score(test_probs[i], y_test[i]) for i in range(len(y_test))])
    test_coverage = float(np.mean(test_scores <= qhat) * 100)

    # Prediction set sizes on test
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
    print("External Validation: Gas Sensor Array Drift Dataset")
    print("=" * 60)

    # Load data
    X, y = load_gas_sensor_data()

    # Prepare temporal split
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes, feature_names = prepare_data(X, y)

    # Train model
    print("\nTraining LightGBM...")
    model = train_model(X_train, y_train, X_val, y_val, num_classes)

    # Accuracy
    import lightgbm as lgb
    val_pred = np.argmax(model.predict(X_val), axis=1) if num_classes > 2 else (model.predict(X_val) > 0.5).astype(int)
    test_pred = np.argmax(model.predict(X_test), axis=1) if num_classes > 2 else (model.predict(X_test) > 0.5).astype(int)
    val_acc = float(np.mean(val_pred == y_val) * 100)
    test_acc = float(np.mean(test_pred == y_test) * 100)
    print(f"Accuracy: val={val_acc:.1f}%, test={test_acc:.1f}%")

    # SHAP concentration
    print("\nComputing SHAP concentration...")
    concentration, top_feature, ranking = compute_shap_concentration(model, X_val, feature_names)
    print(f"SHAP Concentration: {concentration:.2f}%")
    print(f"Top feature: {top_feature}")
    print(f"Feature ranking: {ranking}")

    # APS conformal prediction
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
        'dataset': 'Gas Sensor Array Drift',
        'source': 'UCI #224',
        'domain': 'Chemical sensing',
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

    # Save results
    out_path = os.path.join(RESULTS_DIR, 'gas_sensor_validation.json')
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
