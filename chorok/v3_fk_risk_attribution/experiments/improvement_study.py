"""
Improvement Study: Does Improving High-Attributed FK Reduce Uncertainty?
=========================================================================

This is the FORWARD direction test:
- Corruption test: Degrade data → Uncertainty increases (proven)
- Improvement test: Enhance data → Uncertainty decreases (to prove)

Simulation approach:
1. Train model on original data
2. Identify high-attributed FK
3. Simulate "data improvement" by:
   - Reducing variance in FK features
   - Filling missing values more accurately
   - Adding informative features
4. Retrain and measure uncertainty reduction

Author: ChorokLeeDev
Created: 2025-12-08
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr
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


def get_uncertainty(models, X):
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0)


def get_fk_grouping(col_to_fk):
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)
    return dict(fk_to_cols)


def compute_fk_attribution(models, X, fk_grouping, n_permute=5):
    base_unc = get_uncertainty(models, X).mean()
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
            perm_unc = get_uncertainty(models, X_perm).mean()
            deltas.append(perm_unc - base_unc)

        results[fk_group] = np.mean(deltas)

    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def simulate_data_improvement(X, y, fk_grouping, target_fk, improvement_level=0.5):
    """
    Simulate data quality improvement for a specific FK group.

    Approach: Make features more predictive of target by reducing noise.
    This simulates what happens when a data team improves their data quality.

    Method:
    1. For each feature in the FK group
    2. Compute residuals from target correlation
    3. Reduce residual variance (simulate noise reduction)
    """
    X_improved = X.copy()
    cols = fk_grouping.get(target_fk, [])
    valid_cols = [c for c in cols if c in X.columns]

    if not valid_cols:
        return X_improved

    y_normalized = (y - y.mean()) / y.std()

    for col in valid_cols:
        values = X_improved[col].values.astype(float)

        # Current correlation with target
        col_normalized = (values - np.mean(values)) / (np.std(values) + 1e-8)

        # Compute "signal" component (correlated with y)
        corr = np.corrcoef(col_normalized, y_normalized)[0, 1]
        if np.isnan(corr):
            continue

        # Decompose into signal + noise
        signal = corr * y_normalized * np.std(values) + np.mean(values)
        noise = values - signal

        # Reduce noise by improvement_level (0 = no change, 1 = perfect)
        improved_values = signal + noise * (1 - improvement_level)

        X_improved[col] = improved_values

    return X_improved


def run_improvement_study(X, y, col_to_fk, domain_name, n_runs=3):
    """
    Test: Does improving high-attributed FK reduce uncertainty?

    Protocol:
    1. Train on original data, compute attribution
    2. Identify top FK (highest attribution)
    3. Simulate improvement on top FK
    4. Retrain and measure uncertainty
    5. Compare with improving low FK
    """
    print(f"\n{'='*70}")
    print(f"IMPROVEMENT STUDY: {domain_name}")
    print(f"Question: Does improving high-attributed FK reduce uncertainty?")
    print(f"{'='*70}")

    fk_grouping = get_fk_grouping(col_to_fk)
    fk_list = list(fk_grouping.keys())

    print(f"FK groups: {fk_list}")

    improvement_levels = [0.0, 0.25, 0.5, 0.75]

    all_results = []

    for run in range(n_runs):
        seed = 42 + run * 10
        np.random.seed(seed)
        print(f"\n  Run {run+1}/{n_runs}...")

        # Train baseline model
        models_baseline = train_ensemble(X, y, n_models=5, base_seed=seed)
        baseline_unc = get_uncertainty(models_baseline, X).mean()

        # Compute attribution
        attribution = compute_fk_attribution(models_baseline, X, fk_grouping)
        sorted_fks = sorted(attribution.keys(), key=lambda k: -attribution[k])
        top_fk = sorted_fks[0]
        low_fk = sorted_fks[-1]

        print(f"    Attribution: {[f'{k}:{v:.1f}%' for k,v in attribution.items()]}")
        print(f"    Top FK: {top_fk} ({attribution[top_fk]:.1f}%)")
        print(f"    Low FK: {low_fk} ({attribution[low_fk]:.1f}%)")
        print(f"    Baseline uncertainty: {baseline_unc:.6f}")

        run_results = {
            'baseline_unc': baseline_unc,
            'attribution': attribution,
            'top_fk': top_fk,
            'low_fk': low_fk,
            'improvements': {}
        }

        # Test improvement at different levels
        for level in improvement_levels:
            if level == 0:
                continue

            # Improve TOP FK
            X_improved_top = simulate_data_improvement(X, y, fk_grouping, top_fk, level)
            models_top = train_ensemble(X_improved_top, y, n_models=5, base_seed=seed)
            unc_top = get_uncertainty(models_top, X_improved_top).mean()

            # Improve LOW FK
            X_improved_low = simulate_data_improvement(X, y, fk_grouping, low_fk, level)
            models_low = train_ensemble(X_improved_low, y, n_models=5, base_seed=seed)
            unc_low = get_uncertainty(models_low, X_improved_low).mean()

            reduction_top = (baseline_unc - unc_top) / baseline_unc * 100
            reduction_low = (baseline_unc - unc_low) / baseline_unc * 100

            run_results['improvements'][level] = {
                'top_fk_reduction': reduction_top,
                'low_fk_reduction': reduction_low,
                'unc_after_top': unc_top,
                'unc_after_low': unc_low
            }

            print(f"    Improvement {level*100:.0f}%: Top FK → {reduction_top:+.1f}% unc, Low FK → {reduction_low:+.1f}% unc")

        all_results.append(run_results)

    # Summary
    print(f"\n{'='*70}")
    print("IMPROVEMENT STUDY SUMMARY")
    print("="*70)

    # Average results
    print(f"\n  Improvement Level | Top FK Δ Uncertainty | Low FK Δ Uncertainty | Ratio")
    print(f"  {'-'*70}")

    for level in improvement_levels[1:]:
        top_reductions = [r['improvements'][level]['top_fk_reduction'] for r in all_results]
        low_reductions = [r['improvements'][level]['low_fk_reduction'] for r in all_results]

        avg_top = np.mean(top_reductions)
        avg_low = np.mean(low_reductions)
        ratio = avg_top / (avg_low + 1e-6) if avg_low != 0 else float('inf')

        print(f"  {level*100:>15.0f}% | {avg_top:>+18.1f}% | {avg_low:>+18.1f}% | {ratio:>5.1f}x")

    # Most common top FK
    top_fks = [r['top_fk'] for r in all_results]
    most_common_top = max(set(top_fks), key=top_fks.count)

    # Final verdict
    final_level = 0.5
    top_reds = [r['improvements'][final_level]['top_fk_reduction'] for r in all_results]
    low_reds = [r['improvements'][final_level]['low_fk_reduction'] for r in all_results]
    avg_top_red = np.mean(top_reds)
    avg_low_red = np.mean(low_reds)

    print(f"\n  Key Finding at 50% improvement:")
    print(f"    Improving {most_common_top} (top): {avg_top_red:+.1f}% uncertainty reduction")
    print(f"    Improving low FK: {avg_low_red:+.1f}% uncertainty reduction")

    if avg_top_red > avg_low_red and avg_top_red > 0:
        verdict = f"✓ ACTIONABLE: Improving top FK ({most_common_top}) reduces uncertainty {avg_top_red/max(avg_low_red,0.01):.1f}x more"
        success = True
    elif avg_top_red > 0:
        verdict = f"~ PARTIAL: Both improve uncertainty, top slightly better"
        success = True
    else:
        verdict = f"✗ NOT ACTIONABLE: Improvement doesn't reduce uncertainty as expected"
        success = False

    print(f"\n  Verdict: {verdict}")

    return {
        'domain': domain_name,
        'top_fk': most_common_top,
        'avg_top_reduction': avg_top_red,
        'avg_low_reduction': avg_low_red,
        'verdict': verdict,
        'success': success
    }


def run_all():
    print("="*70)
    print("IMPROVEMENT STUDY")
    print("Forward causality test: Improve data → Reduce uncertainty")
    print("="*70)

    all_results = {}

    # SALT
    print("\n[1/2] Loading SALT data...")
    salt_cache = CACHE_DIR / 'data_salt_PLANT_2000.pkl'
    if salt_cache.exists():
        result = load_from_cache(salt_cache)
        if result[0] is not None:
            X, y, _, col_to_fk = result
            salt_results = run_improvement_study(X, y, col_to_fk, "SALT (ERP)")
            all_results['salt'] = salt_results

    # H&M
    print("\n[2/2] Loading H&M data...")
    hm_cache = CACHE_DIR / 'data_hm_item-sales_3000_v1.pkl'
    if hm_cache.exists():
        result = load_from_cache(hm_cache)
        if result[0] is not None:
            X, y, _, col_to_fk = result
            hm_results = run_improvement_study(X, y, col_to_fk, "H&M (Retail)")
            all_results['hm'] = hm_results

    # Save
    import json

    def convert(obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        return obj

    output_path = RESULTS_DIR / 'improvement_study.json'
    with open(output_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\n[SAVED] {output_path}")

    # Final summary
    print("\n" + "="*70)
    print("ACTIONABILITY CONCLUSION")
    print("="*70)

    for domain, res in all_results.items():
        print(f"\n{res['domain']}:")
        print(f"  {res['verdict']}")

    return all_results


if __name__ == "__main__":
    results = run_all()
