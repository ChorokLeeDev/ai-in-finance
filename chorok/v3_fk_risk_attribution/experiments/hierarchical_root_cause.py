"""
Hierarchical Root Cause Analysis
=================================

Goal: Drill down from FK → Column → Condition

Level 1: Which FK table? (done)
Level 2: Which column within that FK?
Level 3: Which subset of data (conditional)?

Example output:
  "Uncertainty comes from ITEM table (37%)
   → specifically ITEM.weight column (68% of ITEM's contribution)
   → specifically items where category='electronics' (82% of weight's contribution)"

This makes root cause analysis truly actionable.

Author: ChorokLeeDev
Created: 2025-12-08
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache')
RESULTS_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/results')


def load_from_cache(cache_file):
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    if len(data) == 4:
        return data
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
    return preds.var(axis=0).mean()


def get_sample_uncertainty(models, X):
    """Get per-sample uncertainty."""
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0)


def get_fk_grouping(col_to_fk):
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)
    return dict(fk_to_cols)


def level1_fk_attribution(models, X, fk_grouping, n_permute=10):
    """Level 1: Which FK table contributes most to uncertainty?"""
    base_unc = get_uncertainty(models, X)
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
            perm_unc = get_uncertainty(models, X_perm)
            deltas.append(perm_unc - base_unc)

        results[fk_group] = np.mean(deltas)

    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def level2_column_attribution(models, X, fk_grouping, target_fk, n_permute=10):
    """Level 2: Which column within the FK contributes most?"""
    cols = [c for c in fk_grouping.get(target_fk, []) if c in X.columns]
    if not cols:
        return {}

    base_unc = get_uncertainty(models, X)
    results = {}

    for col in cols:
        deltas = []
        for _ in range(n_permute):
            X_perm = X.copy()
            X_perm[col] = np.random.permutation(X_perm[col].values)
            perm_unc = get_uncertainty(models, X_perm)
            deltas.append(perm_unc - base_unc)

        results[col] = np.mean(deltas)

    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {c: max(0, v) / total * 100 for c, v in results.items()}
    return {c: 0 for c in results}


def level3_conditional_attribution(models, X, target_col, n_bins=4, n_permute=5):
    """
    Level 3: Which subset of data contributes most?

    Approach: Split data by target_col values, measure uncertainty contribution per bin.
    """
    if target_col not in X.columns:
        return {}

    values = X[target_col].values
    sample_unc = get_sample_uncertainty(models, X)

    # Bin the values
    try:
        bins = pd.qcut(values, q=n_bins, labels=False, duplicates='drop')
    except:
        bins = pd.cut(values, bins=n_bins, labels=False)

    results = {}
    unique_bins = np.unique(bins[~np.isnan(bins)])

    for b in unique_bins:
        mask = bins == b
        if mask.sum() < 10:
            continue

        # Uncertainty for this subset
        subset_unc = sample_unc[mask].mean()
        subset_size = mask.sum()

        # Value range for this bin
        bin_values = values[mask]
        val_min, val_max = bin_values.min(), bin_values.max()

        results[f"bin_{int(b)}"] = {
            'uncertainty': subset_unc,
            'size': subset_size,
            'value_range': (val_min, val_max),
            'contribution': subset_unc * subset_size  # Weighted contribution
        }

    # Normalize contributions
    total_contrib = sum(r['contribution'] for r in results.values())
    if total_contrib > 0:
        for k in results:
            results[k]['contribution_pct'] = results[k]['contribution'] / total_contrib * 100

    return results


def run_hierarchical_analysis(X, y, col_to_fk, domain_name):
    """
    Full hierarchical root cause analysis.

    Output:
      Level 1: ITEM (37.2%)
      Level 2: ITEM.weight (45.3% of ITEM)
      Level 3: weight in [0, 0.5] range (67.8% of weight's contribution)
    """
    print(f"\n{'='*70}")
    print(f"HIERARCHICAL ROOT CAUSE ANALYSIS: {domain_name}")
    print(f"{'='*70}")

    fk_grouping = get_fk_grouping(col_to_fk)

    # Train model
    print(f"\n  Training model...")
    models = train_ensemble(X, y, n_models=5, base_seed=42)

    # Level 1: FK Attribution
    print(f"\n  === LEVEL 1: FK Table Attribution ===")
    fk_attr = level1_fk_attribution(models, X, fk_grouping)

    ranked_fks = sorted(fk_attr.items(), key=lambda x: -x[1])
    for fk, attr in ranked_fks:
        print(f"    {fk}: {attr:.1f}%")

    top_fk = ranked_fks[0][0]
    top_fk_attr = ranked_fks[0][1]
    print(f"\n  → Root Cause (L1): {top_fk} ({top_fk_attr:.1f}%)")

    # Level 2: Column Attribution within top FK
    print(f"\n  === LEVEL 2: Column Attribution within {top_fk} ===")
    col_attr = level2_column_attribution(models, X, fk_grouping, top_fk)

    ranked_cols = sorted(col_attr.items(), key=lambda x: -x[1])
    for col, attr in ranked_cols[:5]:  # Top 5
        print(f"    {col}: {attr:.1f}%")

    if ranked_cols:
        top_col = ranked_cols[0][0]
        top_col_attr = ranked_cols[0][1]
        print(f"\n  → Root Cause (L2): {top_fk}.{top_col} ({top_col_attr:.1f}% of {top_fk})")

        # Level 3: Conditional Attribution
        print(f"\n  === LEVEL 3: Conditional Attribution for {top_col} ===")
        cond_attr = level3_conditional_attribution(models, X, top_col)

        if cond_attr:
            ranked_conds = sorted(cond_attr.items(),
                                   key=lambda x: -x[1].get('contribution_pct', 0))

            for cond, info in ranked_conds:
                val_range = info.get('value_range', (None, None))
                contrib = info.get('contribution_pct', 0)
                unc = info.get('uncertainty', 0)
                size = info.get('size', 0)
                print(f"    {top_col} in [{val_range[0]:.2f}, {val_range[1]:.2f}]: "
                      f"{contrib:.1f}% contrib, unc={unc:.4f}, n={size}")

            top_cond = ranked_conds[0]
            top_cond_range = top_cond[1]['value_range']
            top_cond_contrib = top_cond[1].get('contribution_pct', 0)

            print(f"\n  → Root Cause (L3): {top_col} in [{top_cond_range[0]:.2f}, {top_cond_range[1]:.2f}] "
                  f"({top_cond_contrib:.1f}%)")

    # Summary
    print(f"\n{'='*70}")
    print("HIERARCHICAL ROOT CAUSE SUMMARY")
    print("="*70)

    print(f"\n  🔍 Root Cause Drill-Down:")
    print(f"\n  Level 1 (FK Table):")
    print(f"    └─ {top_fk} contributes {top_fk_attr:.1f}% of uncertainty")

    if ranked_cols:
        print(f"\n  Level 2 (Column):")
        print(f"    └─ {top_col} contributes {top_col_attr:.1f}% within {top_fk}")

        if cond_attr and ranked_conds:
            print(f"\n  Level 3 (Condition):")
            print(f"    └─ Values in [{top_cond_range[0]:.2f}, {top_cond_range[1]:.2f}] "
                  f"contribute {top_cond_contrib:.1f}%")

    # Actionable recommendation
    print(f"\n  📋 Actionable Recommendation:")
    print(f"    1. Contact {top_fk} data team")
    if ranked_cols:
        print(f"    2. Audit '{top_col}' column specifically")
        if cond_attr and ranked_conds:
            print(f"    3. Focus on records where {top_col} ∈ [{top_cond_range[0]:.2f}, {top_cond_range[1]:.2f}]")

    return {
        'domain': domain_name,
        'level1_fk': top_fk,
        'level1_attribution': top_fk_attr,
        'level2_column': top_col if ranked_cols else None,
        'level2_attribution': top_col_attr if ranked_cols else None,
        'level3_condition': top_cond_range if (cond_attr and ranked_conds) else None,
        'level3_attribution': top_cond_contrib if (cond_attr and ranked_conds) else None
    }


def run_all():
    print("="*70)
    print("HIERARCHICAL ROOT CAUSE ANALYSIS")
    print("FK → Column → Condition drill-down")
    print("="*70)

    np.random.seed(42)
    all_results = {}

    # SALT
    print("\n[1/3] SALT...")
    salt_cache = CACHE_DIR / 'data_salt_PLANT_2000.pkl'
    if salt_cache.exists():
        X, y, _, col_to_fk = load_from_cache(salt_cache)
        if X is not None:
            result = run_hierarchical_analysis(X, y, col_to_fk, "SALT (ERP)")
            all_results['salt'] = result

    # F1
    print("\n[2/3] F1...")
    f1_cache = CACHE_DIR / 'data_f1_driver-position_3000_v3.pkl'
    if f1_cache.exists():
        X, y, _, col_to_fk = load_from_cache(f1_cache)
        if X is not None:
            result = run_hierarchical_analysis(X, y, col_to_fk, "F1 (Racing)")
            all_results['f1'] = result

    # H&M
    print("\n[3/3] H&M...")
    hm_cache = CACHE_DIR / 'data_hm_item-sales_3000_v1.pkl'
    if hm_cache.exists():
        X, y, _, col_to_fk = load_from_cache(hm_cache)
        if X is not None:
            result = run_hierarchical_analysis(X, y, col_to_fk, "H&M (Retail)")
            all_results['hm'] = result

    # Save
    import json

    def convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)): return int(obj)
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, np.ndarray): return [convert(x) for x in obj.tolist()]
        if isinstance(obj, tuple): return [convert(x) for x in obj]
        if isinstance(obj, dict): return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        if obj is None: return None
        return str(obj) if not isinstance(obj, (int, float, str)) else obj

    output_path = RESULTS_DIR / 'hierarchical_root_cause.json'
    with open(output_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\n[SAVED] {output_path}")

    return all_results


if __name__ == "__main__":
    results = run_all()
