"""
TEST 1: SHAP Baseline - Can Run TODAY
======================================

Question: Does FK grouping make SHAP attribution more stable than individual features?

Success Criteria: FK stability ρ > 0.85 AND better than individual features

Time: ~30-60 minutes on rel-f1 (small dataset)

Usage:
    python test_1_shap_baseline.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add relbench to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    import shap
    print("✅ SHAP installed")
except ImportError:
    print("❌ SHAP not installed. Run: pip install shap")
    sys.exit(1)

import lightgbm as lgb
from relbench.datasets import get_dataset
from relbench.tasks import get_task


def extract_features_with_fk(dataset, task, sample_size=3000):
    """Extract features and track FK groups (simplified version)."""
    db = dataset.get_db()
    train_table = task.get_table("train")

    entity_table_name = task.entity_table
    entity_table = db.table_dict[entity_table_name]
    entity_df = entity_table.df.copy()
    train_df = train_table.df.copy()

    # Sample if needed
    if len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=42)

    # Find FK to entity
    fk_to_entity = list(train_table.fkey_col_to_pkey_table.keys())[0]
    entity_pkey = entity_table.pkey_col

    # Merge
    merged_df = train_df.merge(entity_df, how='left', left_on=fk_to_entity,
                                right_on=entity_pkey, suffixes=('', '_entity'))

    target_col = task.target_col
    y = merged_df[target_col].values

    col_to_fk = {}
    feature_cols = []

    # Get numeric columns from train
    for col in train_df.columns:
        if col == target_col or col.endswith('Id') or col.endswith('_id'):
            continue
        if train_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            if col in merged_df.columns:
                feature_cols.append(col)
                col_to_fk[col] = 'TRAIN'

    # Get numeric columns from entity
    for col in entity_df.columns:
        if col == entity_pkey or col.endswith('Id') or col.endswith('_id'):
            continue
        if entity_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            col_name = col if col in merged_df.columns else f"{col}_entity"
            if col_name in merged_df.columns and col_name not in feature_cols:
                feature_cols.append(col_name)
                col_to_fk[col_name] = entity_table_name.upper()

    # Join other FK tables (simplified - just aggregate)
    for table_name, table in db.table_dict.items():
        if table_name == entity_table_name:
            continue
        if hasattr(table, 'fkey_col_to_pkey_table'):
            for fk_col, ref_table in table.fkey_col_to_pkey_table.items():
                if ref_table == entity_table_name:
                    table_df = table.df
                    numeric_cols = [c for c in table_df.select_dtypes(include=[np.number]).columns
                                   if not c.endswith('Id') and c != fk_col]

                    if numeric_cols:
                        # Aggregate by FK
                        agg_df = table_df.groupby(fk_col)[numeric_cols].mean().reset_index()
                        agg_df.columns = [fk_col] + [f'{table_name}_{c}_mean' for c in numeric_cols]

                        merged_df = merged_df.merge(agg_df, how='left', left_on=fk_to_entity,
                                                    right_on=fk_col, suffixes=('', f'_{table_name}'))

                        for col in agg_df.columns[1:]:
                            if col in merged_df.columns and col not in feature_cols:
                                feature_cols.append(col)
                                col_to_fk[col] = table_name.upper()

    # Extract features
    X = merged_df[feature_cols].fillna(0).values

    # Build FK groups
    fk_to_cols = defaultdict(list)
    for i, col in enumerate(feature_cols):
        fk_name = col_to_fk.get(col, 'UNKNOWN')
        fk_to_cols[fk_name].append(i)

    print(f"✅ Extracted {len(feature_cols)} features from {len(fk_to_cols)} FK groups")
    print(f"   Sample size: {len(X)}")
    for fk_name, cols in fk_to_cols.items():
        print(f"   - {fk_name}: {len(cols)} features")

    return X, y, dict(fk_to_cols), feature_cols


def compute_shap_attribution(model, X, fk_to_cols, method='individual'):
    """Compute SHAP values and aggregate by FK if needed."""
    print(f"\nComputing SHAP values ({method})...")

    # Subsample for speed
    if len(X) > 1000:
        idx = np.random.choice(len(X), 1000, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if method == 'individual':
        # Feature-level attribution
        attribution = np.abs(shap_values).mean(axis=0)
        # Normalize
        attribution = attribution / attribution.sum()
        return attribution

    elif method == 'fk_grouped':
        # FK-level attribution
        fk_attribution = {}
        abs_shap = np.abs(shap_values)

        for fk_name, col_indices in fk_to_cols.items():
            # Sum importance across FK group
            fk_importance = abs_shap[:, col_indices].sum()
            fk_attribution[fk_name] = fk_importance

        # Normalize
        total = sum(fk_attribution.values())
        fk_attribution = {k: v/total for k, v in fk_attribution.items()}

        return fk_attribution


def test_stability(X, y, fk_to_cols, n_seeds=3):
    """Test stability across random seeds."""
    print("\n" + "="*60)
    print("Testing Stability Across Random Seeds")
    print("="*60)

    individual_attrs = []
    fk_attrs = []

    for seed in range(42, 42 + n_seeds):
        print(f"\nSeed {seed}:")

        # Train model
        model = lgb.LGBMRegressor(n_estimators=100, random_state=seed, verbose=-1)
        model.fit(X, y)

        # Individual feature attribution
        ind_attr = compute_shap_attribution(model, X, fk_to_cols, method='individual')
        individual_attrs.append(ind_attr)

        # FK-grouped attribution
        fk_attr = compute_shap_attribution(model, X, fk_to_cols, method='fk_grouped')
        fk_attrs.append(fk_attr)

        print(f"  Top 3 FKs: {sorted(fk_attr.items(), key=lambda x: x[1], reverse=True)[:3]}")

    # Compute stability (Spearman correlation between seeds)
    print("\n" + "="*60)
    print("Stability Results")
    print("="*60)

    # Individual feature stability
    ind_correlations = []
    for i in range(n_seeds):
        for j in range(i+1, n_seeds):
            rho, _ = spearmanr(individual_attrs[i], individual_attrs[j])
            ind_correlations.append(rho)
    ind_stability = np.mean(ind_correlations)

    # FK stability
    fk_correlations = []
    fk_names = sorted(fk_to_cols.keys())
    for i in range(n_seeds):
        for j in range(i+1, n_seeds):
            vec_i = [fk_attrs[i][fk] for fk in fk_names]
            vec_j = [fk_attrs[j][fk] for fk in fk_names]
            rho, _ = spearmanr(vec_i, vec_j)
            fk_correlations.append(rho)
    fk_stability = np.mean(fk_correlations)

    print(f"\nIndividual Features Stability: ρ = {ind_stability:.3f}")
    print(f"FK-Grouped Stability:          ρ = {fk_stability:.3f}")

    improvement = (fk_stability - ind_stability) / ind_stability * 100
    print(f"Improvement:                   {improvement:+.1f}%")

    return {
        'individual_stability': float(ind_stability),
        'fk_stability': float(fk_stability),
        'improvement_pct': float(improvement),
        'n_features': len(individual_attrs[0]),
        'n_fk_groups': len(fk_to_cols),
    }


def main():
    print("\n" + "="*60)
    print("TEST 1: SHAP Baseline - FK Grouping vs Individual Features")
    print("="*60)

    # Use rel-f1 (small, fast)
    dataset_name = 'rel-f1'
    task_name = 'driver-position'

    print(f"\nDataset: {dataset_name}")
    print(f"Task: {task_name}")

    # Load data
    print("\nLoading data...")
    dataset = get_dataset(dataset_name, download=True)
    task = get_task(dataset_name, task_name, download=True)

    # Extract features
    X, y, fk_to_cols, feature_names = extract_features_with_fk(dataset, task, sample_size=3000)

    # Run stability test
    results = test_stability(X, y, fk_to_cols, n_seeds=3)

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    fk_stable = results['fk_stability'] > 0.85
    better_than_individual = results['fk_stability'] > results['individual_stability']

    if fk_stable and better_than_individual:
        verdict = "✅ PASS"
        recommendation = "Include SHAP baseline in paper - FK grouping improves stability"
    elif better_than_individual:
        verdict = "⚠️  MARGINAL"
        recommendation = f"FK grouping is better but stability only {results['fk_stability']:.3f} (target: >0.85)"
    else:
        verdict = "❌ FAIL"
        recommendation = "FK grouping doesn't improve SHAP stability - stick with permutation method"

    print(f"\nResult: {verdict}")
    print(f"FK Stability: {results['fk_stability']:.3f} (target: >0.85)")
    print(f"Better than individual: {better_than_individual}")
    print(f"\nRecommendation: {recommendation}")

    # Save results
    output_dir = Path(__file__).parent / 'test_results'
    output_dir.mkdir(exist_ok=True)

    output = {
        'test': 'shap_baseline',
        'dataset': dataset_name,
        'task': task_name,
        'verdict': verdict,
        'recommendation': recommendation,
        'results': results,
    }

    output_file = output_dir / 'test_1_shap_baseline.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return output


if __name__ == '__main__':
    result = main()
