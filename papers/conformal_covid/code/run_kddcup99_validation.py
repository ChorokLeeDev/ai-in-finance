#!/usr/bin/env python3
"""
External validation: KDDCup99 Intrusion Detection Dataset
- 23 attack classes, 42 features, ~4.9M samples
- Network intrusion detection with temporal structure
- Known distribution shift between attack patterns
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def load_kddcup99():
    """Load KDDCup99 from sklearn (full 10% subset, all attack types)."""
    from sklearn.datasets import fetch_kddcup99
    from sklearn.preprocessing import LabelEncoder

    print("Loading KDDCup99 (full 10% subset, all attack types)...")
    data = fetch_kddcup99(subset=None, percent10=True)
    X_raw = data.data
    y_raw = data.target

    print(f"Raw shape: {X_raw.shape}")
    print(f"Unique labels: {len(np.unique(y_raw))}")

    # Convert to DataFrame for feature processing
    # KDDCup99 features: mix of numeric and categorical
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

    # Encode categorical features
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == object:
            cat_cols.append(col)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    print(f"Categorical features encoded: {cat_cols}")

    # Filter to classes with >= 100 samples for meaningful conformal prediction
    label_counts = pd.Series(y_raw).value_counts()
    valid_labels = label_counts[label_counts >= 100].index
    mask = np.isin(y_raw, valid_labels)
    df = df[mask].reset_index(drop=True)
    y_filtered = y_raw[mask]

    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y_filtered)
    num_classes = len(le_y.classes_)

    print(f"After filtering (>= 100 samples): {len(df)} samples, {num_classes} classes")
    print(f"Classes: {list(le_y.classes_)}")
    print(f"Class distribution:")
    for cls, cnt in sorted(pd.Series(y_filtered).value_counts().items(), key=lambda x: -x[1]):
        print(f"  {cls}: {cnt}")

    return df, y_encoded, num_classes, list(df.columns)


def split_temporal(X, y, train_frac=0.6, val_frac=0.2):
    """Split data preserving temporal order (data is roughly time-ordered)."""
    n = len(X)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
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

    # For multiclass, shap_values may be list or 3D array
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
    top_feature = feature_names[top_idx] if feature_names else f"feature_{top_idx}"

    order = np.argsort(mean_abs)[::-1]
    ranking = {}
    for j in range(min(5, len(order))):
        i = int(order[j])
        fname = feature_names[i] if feature_names else f"feature_{i}"
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

    # Calibration scores
    cal_scores = np.array([aps_score(cal_probs[i], cal_y[i]) for i in range(len(cal_y))])

    n = len(cal_scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    qhat = np.quantile(cal_scores, q_level)

    # Val-eval coverage
    eval_scores = np.array([aps_score(eval_probs[i], eval_y[i]) for i in range(len(eval_y))])
    val_coverage = float(np.mean(eval_scores <= qhat) * 100)

    # Val set sizes
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

    # Test coverage
    test_scores = np.array([aps_score(test_probs[i], y_test[i]) for i in range(len(y_test))])
    test_coverage = float(np.mean(test_scores <= qhat) * 100)

    # Test set sizes
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
    print("External Validation: KDDCup99 Intrusion Detection")
    print("=" * 60)

    # Load data
    df, y, num_classes, feature_names = load_kddcup99()

    X = df.values

    # Temporal split
    print("\nSplitting data (temporal order)...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_temporal(X, y)

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
        'dataset': 'KDDCup99',
        'source': 'sklearn / OpenML',
        'domain': 'Network intrusion detection',
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

    out_path = os.path.join(RESULTS_DIR, 'kddcup99_validation.json')
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
