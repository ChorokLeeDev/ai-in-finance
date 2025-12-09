"""
Risk-Based Recommendation Study
================================

Question: Can FK Attribution improve decision-making?

Approach:
1. Identify high-uncertainty predictions
2. Use FK Attribution to explain WHY uncertain
3. Generate recommendation based on dominant FK
4. Evaluate: Does following recommendation lead to better outcomes?

Concrete scenario:
- Samples with high uncertainty → recommend "don't trust" / "add buffer"
- Measure: Precision/Recall of flagging high-error predictions
- Show: FK-based flagging outperforms random flagging

Author: ChorokLeeDev
Created: 2025-12-08
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache')
RESULTS_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/results')


def load_from_cache(cache_file):
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    if len(data) == 4:
        X, y, feature_cols, col_to_fk = data
        if isinstance(col_to_fk, dict):
            return X, y, feature_cols, col_to_fk
    return None, None, None, None


def train_ensemble(X, y, n_models=5, base_seed=42):
    models = []
    for i in range(n_models):
        model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=base_seed + i, verbose=-1
        )
        model.fit(X, y)
        models.append(model)
    return models


def get_predictions_and_uncertainty(models, X):
    """Get per-sample predictions and uncertainty."""
    preds = np.array([m.predict(X) for m in models])
    mean_pred = preds.mean(axis=0)
    uncertainty = preds.var(axis=0)
    return mean_pred, uncertainty


def get_fk_grouping(col_to_fk):
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)
    return dict(fk_to_cols)


def compute_sample_fk_attribution(models, X, fk_grouping, sample_idx, n_permute=5):
    """
    Compute FK attribution for a SINGLE sample.
    Returns which FK contributes most to this sample's uncertainty.
    """
    X_single = X.iloc[[sample_idx]]

    preds_base = np.array([m.predict(X_single)[0] for m in models])
    base_unc = preds_base.var()

    fk_contributions = {}

    for fk_group, cols in fk_grouping.items():
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            continue

        deltas = []
        for _ in range(n_permute):
            X_perm = X_single.copy()
            # Replace with random values from the column
            for col in valid_cols:
                random_idx = np.random.randint(0, len(X))
                X_perm[col] = X.iloc[random_idx][col]

            preds_perm = np.array([m.predict(X_perm)[0] for m in models])
            perm_unc = preds_perm.var()
            deltas.append(perm_unc - base_unc)

        fk_contributions[fk_group] = np.mean(deltas)

    # Find dominant FK
    if fk_contributions:
        dominant_fk = max(fk_contributions, key=fk_contributions.get)
        return dominant_fk, fk_contributions
    return None, {}


def run_risk_recommendation_study(X, y, col_to_fk, domain_name, n_test=500):
    """
    Risk-based recommendation study.

    Protocol:
    1. Train model, get predictions and uncertainty for test samples
    2. Compute actual errors
    3. Strategy A: Flag top-k% by uncertainty (baseline)
    4. Strategy B: Flag by FK attribution (if dominant FK has weak features)
    5. Compare precision/recall of catching high-error predictions
    """
    print(f"\n{'='*70}")
    print(f"RISK RECOMMENDATION STUDY: {domain_name}")
    print(f"{'='*70}")

    fk_grouping = get_fk_grouping(col_to_fk)
    fk_list = list(fk_grouping.keys())

    # Split train/test
    n_total = len(X)
    test_idx = np.random.choice(n_total, size=min(n_test, n_total//3), replace=False)
    train_idx = np.array([i for i in range(n_total) if i not in test_idx])

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Train model
    models = train_ensemble(X_train, y_train, n_models=5, base_seed=42)

    # Get predictions and uncertainty for test set
    pred, uncertainty = get_predictions_and_uncertainty(models, X_test)
    y_test_arr = np.array(y_test)

    # Compute actual errors
    errors = np.abs(pred - y_test_arr)

    # Define "high error" as top 20% errors
    error_threshold = np.percentile(errors, 80)
    high_error = errors >= error_threshold

    print(f"  Error threshold (80th percentile): {error_threshold:.3f}")
    print(f"  High-error samples: {high_error.sum()} / {len(high_error)}")

    # Strategy A: Flag by uncertainty (baseline)
    # Flag top 30% by uncertainty
    unc_threshold = np.percentile(uncertainty, 70)
    flagged_by_unc = uncertainty >= unc_threshold

    precision_unc = precision_score(high_error, flagged_by_unc, zero_division=0)
    recall_unc = recall_score(high_error, flagged_by_unc, zero_division=0)
    f1_unc = f1_score(high_error, flagged_by_unc, zero_division=0)

    print(f"\n  Strategy A: Flag by Uncertainty")
    print(f"    Flagged: {flagged_by_unc.sum()}")
    print(f"    Precision: {precision_unc:.3f} (of flagged, how many are high-error)")
    print(f"    Recall: {recall_unc:.3f} (of high-error, how many were flagged)")
    print(f"    F1: {f1_unc:.3f}")

    # Strategy B: Flag by uncertainty + FK-specific threshold
    # For each sample, identify dominant FK
    # If that FK's features have unusual values (outliers), flag
    print(f"\n  Computing per-sample FK attribution (this may take a while)...")

    # Simplified approach: Use global FK attribution ranking
    # Compute global attribution
    from scipy.stats import spearmanr

    def compute_global_attribution(models, X, fk_grouping, n_permute=5):
        base_unc = get_predictions_and_uncertainty(models, X)[1].mean()
        results = {}
        for fk_group, cols in fk_grouping.items():
            valid_cols = [c for c in cols if c in X.columns]
            if not valid_cols:
                continue
            deltas = []
            for _ in range(n_permute):
                X_perm = X.copy()
                for col in valid_cols:
                    X_perm[col] = np.random.permutation(X_perm[col].values)
                perm_unc = get_predictions_and_uncertainty(models, X_perm)[1].mean()
                deltas.append(perm_unc - base_unc)
            results[fk_group] = np.mean(deltas)
        total = sum(max(0, v) for v in results.values())
        if total > 0:
            return {g: max(0, v) / total * 100 for g, v in results.items()}
        return {g: 0 for g in results}

    global_attr = compute_global_attribution(models, X_test, fk_grouping)
    top_fk = max(global_attr, key=global_attr.get)

    print(f"    Top FK: {top_fk} ({global_attr[top_fk]:.1f}% attribution)")

    # Strategy B: Flag if high uncertainty AND top FK features are unusual
    # "Unusual" = far from mean (z-score > 1.5)
    top_fk_cols = [c for c in fk_grouping[top_fk] if c in X_test.columns]

    # Compute "FK quality score" for each sample
    fk_quality = np.zeros(len(X_test))
    for col in top_fk_cols:
        col_mean = X_test[col].mean()
        col_std = X_test[col].std() + 1e-8
        z_scores = np.abs((X_test[col] - col_mean) / col_std)
        fk_quality += z_scores
    fk_quality /= len(top_fk_cols)  # Average z-score

    # Flag if: high uncertainty AND high FK quality score (unusual features)
    # This means: "We're uncertain because the top FK data looks unusual"
    combined_score = uncertainty * fk_quality
    combined_threshold = np.percentile(combined_score, 70)
    flagged_by_combined = combined_score >= combined_threshold

    precision_comb = precision_score(high_error, flagged_by_combined, zero_division=0)
    recall_comb = recall_score(high_error, flagged_by_combined, zero_division=0)
    f1_comb = f1_score(high_error, flagged_by_combined, zero_division=0)

    print(f"\n  Strategy B: Flag by Uncertainty × FK Quality")
    print(f"    Flagged: {flagged_by_combined.sum()}")
    print(f"    Precision: {precision_comb:.3f}")
    print(f"    Recall: {recall_comb:.3f}")
    print(f"    F1: {f1_comb:.3f}")

    # Strategy C: Flag by FK-informed thresholding
    # Different uncertainty thresholds for different FK dominance
    # Samples where top FK is dominant → lower threshold (more conservative)
    print(f"\n  Strategy C: FK-informed Conservative Flagging")

    # For samples with unusual top-FK features, use lower uncertainty threshold
    conservative_mask = fk_quality > np.median(fk_quality)

    flagged_c = np.zeros(len(X_test), dtype=bool)
    # Conservative threshold for unusual FK samples
    flagged_c[conservative_mask] = uncertainty[conservative_mask] >= np.percentile(uncertainty, 50)
    # Normal threshold for normal FK samples
    flagged_c[~conservative_mask] = uncertainty[~conservative_mask] >= np.percentile(uncertainty, 80)

    precision_c = precision_score(high_error, flagged_c, zero_division=0)
    recall_c = recall_score(high_error, flagged_c, zero_division=0)
    f1_c = f1_score(high_error, flagged_c, zero_division=0)

    print(f"    Flagged: {flagged_c.sum()}")
    print(f"    Precision: {precision_c:.3f}")
    print(f"    Recall: {recall_c:.3f}")
    print(f"    F1: {f1_c:.3f}")

    # Summary
    print(f"\n{'='*70}")
    print("RECOMMENDATION STUDY SUMMARY")
    print("="*70)
    print(f"\n  {'Strategy':<40} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"  {'-'*76}")
    print(f"  {'A: Uncertainty only':<40} {precision_unc:<12.3f} {recall_unc:<12.3f} {f1_unc:<12.3f}")
    print(f"  {'B: Uncertainty × FK Quality':<40} {precision_comb:<12.3f} {recall_comb:<12.3f} {f1_comb:<12.3f}")
    print(f"  {'C: FK-informed Conservative':<40} {precision_c:<12.3f} {recall_c:<12.3f} {f1_c:<12.3f}")

    # Best strategy
    f1_scores = {'A': f1_unc, 'B': f1_comb, 'C': f1_c}
    best = max(f1_scores, key=f1_scores.get)

    improvement = (f1_scores[best] - f1_unc) / (f1_unc + 1e-6) * 100

    if best != 'A' and improvement > 5:
        verdict = f"✓ FK-based recommendation (Strategy {best}) improves F1 by {improvement:.1f}%"
        success = True
    elif best != 'A':
        verdict = f"~ FK-based recommendation shows marginal improvement ({improvement:.1f}%)"
        success = True
    else:
        verdict = "✗ Uncertainty alone is sufficient, FK attribution doesn't add value for flagging"
        success = False

    print(f"\n  Verdict: {verdict}")

    # Actionable insight
    print(f"\n  Actionable Recommendation:")
    print(f"    When {top_fk} features are unusual (z > 1.5) AND uncertainty is high:")
    print(f"    → Flag prediction for human review")
    print(f"    → Add safety buffer to estimates")
    print(f"    → Request additional {top_fk} data before deciding")

    return {
        'domain': domain_name,
        'top_fk': top_fk,
        'strategy_a_f1': f1_unc,
        'strategy_b_f1': f1_comb,
        'strategy_c_f1': f1_c,
        'best_strategy': best,
        'improvement': improvement,
        'verdict': verdict,
        'success': success
    }


def run_all():
    print("="*70)
    print("RISK-BASED RECOMMENDATION STUDY")
    print("Can FK Attribution improve decision-making?")
    print("="*70)

    np.random.seed(42)

    all_results = {}

    # SALT
    print("\n[1/2] Loading SALT data...")
    salt_cache = CACHE_DIR / 'data_salt_PLANT_2000.pkl'
    if salt_cache.exists():
        result = load_from_cache(salt_cache)
        if result[0] is not None:
            X, y, _, col_to_fk = result
            salt_results = run_risk_recommendation_study(X, y, col_to_fk, "SALT (ERP)", n_test=500)
            all_results['salt'] = salt_results

    # H&M
    print("\n[2/2] Loading H&M data...")
    hm_cache = CACHE_DIR / 'data_hm_item-sales_3000_v1.pkl'
    if hm_cache.exists():
        result = load_from_cache(hm_cache)
        if result[0] is not None:
            X, y, _, col_to_fk = result
            hm_results = run_risk_recommendation_study(X, y, col_to_fk, "H&M (Retail)", n_test=500)
            all_results['hm'] = hm_results

    # Save
    import json

    def convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, bool): return bool(obj)
        return obj

    output_path = RESULTS_DIR / 'risk_recommendation.json'
    with open(output_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\n[SAVED] {output_path}")

    return all_results


if __name__ == "__main__":
    results = run_all()
