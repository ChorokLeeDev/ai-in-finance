"""
TEST 4: Causal Attribution
===========================

Question: Does causal (interventional) attribution differ from observational (correlational)?

Success Criteria: Top-1 FK differs between causal and observational AND makes domain sense

Time: ~5-10 minutes

Usage:
    python test_4_causal.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import lightgbm as lgb
from relbench.datasets import get_dataset
from relbench.tasks import get_task


def extract_features_with_fk(dataset, task, sample_size=1500):
    """Extract features with FK tracking."""
    db = dataset.get_db()
    train_table = task.get_table("train")

    entity_table_name = task.entity_table
    entity_table = db.table_dict[entity_table_name]
    entity_df = entity_table.df.copy()
    train_df = train_table.df.copy()

    if len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=42)

    fk_to_entity = list(train_table.fkey_col_to_pkey_table.keys())[0]
    entity_pkey = entity_table.pkey_col

    merged_df = train_df.merge(entity_df, how='left', left_on=fk_to_entity,
                                right_on=entity_pkey, suffixes=('', '_entity'))

    target_col = task.target_col
    y = merged_df[target_col].values

    col_to_fk = {}
    feature_cols = []
    fk_to_col_indices = defaultdict(list)

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

    # Join other FK tables
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
                        agg_df = table_df.groupby(fk_col)[numeric_cols].mean().reset_index()
                        agg_df.columns = [fk_col] + [f'{table_name}_{c}_mean' for c in numeric_cols]

                        merged_df = merged_df.merge(agg_df, how='left', left_on=fk_to_entity,
                                                    right_on=fk_col, suffixes=('', f'_{table_name}'))

                        for col in agg_df.columns[1:]:
                            if col in merged_df.columns and col not in feature_cols:
                                feature_cols.append(col)
                                col_to_fk[col] = table_name.upper()

    # Extract features and build FK mapping
    X_df = merged_df[feature_cols].fillna(0)
    X = X_df.values

    for i, col in enumerate(feature_cols):
        fk_name = col_to_fk.get(col, 'UNKNOWN')
        fk_to_col_indices[fk_name].append(i)

    return X, y, dict(fk_to_col_indices), feature_cols


def train_ensemble(X, y, n_models=5):
    """Train ensemble."""
    models = []
    for i in range(n_models):
        idx = np.random.choice(len(X), int(0.8 * len(X)), replace=True)
        model = lgb.LGBMRegressor(n_estimators=50, random_state=42+i, verbose=-1)
        model.fit(X[idx], y[idx])
        models.append(model)
    return models


def ensemble_variance(models, X):
    """Compute ensemble variance."""
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0).mean()


def observational_attribution(models, X, fk_to_cols):
    """Observational attribution via permutation."""
    base_unc = ensemble_variance(models, X)

    attribution = {}
    for fk_name, col_indices in fk_to_cols.items():
        X_perm = X.copy()
        X_perm[:, col_indices] = np.random.permutation(X_perm[:, col_indices])
        perm_unc = ensemble_variance(models, X_perm)
        attribution[fk_name] = perm_unc - base_unc

    # Normalize
    total = sum(max(0, v) for v in attribution.values())
    if total > 0:
        attribution = {k: max(0, v)/total for k, v in attribution.items()}

    return attribution


def causal_attribution(models, X, fk_to_cols):
    """Causal (interventional) attribution via mean replacement."""
    base_unc = ensemble_variance(models, X)

    attribution = {}
    for fk_name, col_indices in fk_to_cols.items():
        X_int = X.copy()
        # Intervention: set to mean (do operator)
        X_int[:, col_indices] = X[:, col_indices].mean(axis=0)
        int_unc = ensemble_variance(models, X_int)
        attribution[fk_name] = int_unc - base_unc

    # Normalize
    total = sum(max(0, v) for v in attribution.values())
    if total > 0:
        attribution = {k: max(0, v)/total for k, v in attribution.items()}

    return attribution


def main():
    print("\n" + "="*60)
    print("TEST 4: Causal vs Observational Attribution")
    print("="*60)

    dataset_name = 'rel-f1'
    task_name = 'driver-position'

    print(f"\nDataset: {dataset_name}")
    print(f"Task: {task_name}")
    print("\nDomain knowledge: F1 racing has causal structure:")
    print("  DRIVER (skill) → QUALIFYING (grid) → RESULTS (race) → STANDINGS")
    print("  Expected: DRIVER is root cause, RESULTS is downstream effect")

    # Load data
    print("\nLoading data...")
    dataset = get_dataset(dataset_name, download=True)
    task = get_task(dataset_name, task_name, download=True)

    # Extract features
    X, y, fk_to_cols, feature_names = extract_features_with_fk(dataset, task, sample_size=1500)

    print(f"\n✅ Extracted {len(feature_names)} features from {len(fk_to_cols)} FK groups")
    for fk_name, cols in fk_to_cols.items():
        print(f"   - {fk_name}: {len(cols)} features")

    # Train ensemble
    print("\nTraining ensemble...")
    models = train_ensemble(X, y, n_models=5)

    # Observational attribution
    print("\nComputing observational attribution (permutation)...")
    obs_attr = observational_attribution(models, X, fk_to_cols)

    print("\n📊 Observational Attribution:")
    for fk, attr in sorted(obs_attr.items(), key=lambda x: x[1], reverse=True):
        print(f"   {fk}: {attr*100:.1f}%")

    # Causal attribution
    print("\nComputing causal attribution (intervention)...")
    causal_attr = causal_attribution(models, X, fk_to_cols)

    print("\n📊 Causal Attribution:")
    for fk, attr in sorted(causal_attr.items(), key=lambda x: x[1], reverse=True):
        print(f"   {fk}: {attr*100:.1f}%")

    # Compare rankings
    obs_ranking = sorted(obs_attr.items(), key=lambda x: x[1], reverse=True)
    causal_ranking = sorted(causal_attr.items(), key=lambda x: x[1], reverse=True)

    obs_top = obs_ranking[0][0] if obs_ranking else None
    causal_top = causal_ranking[0][0] if causal_ranking else None

    print("\n" + "="*60)
    print("Comparison")
    print("="*60)

    print(f"\nObservational top FK: {obs_top}")
    print(f"Causal top FK: {causal_top}")

    rankings_differ = obs_top != causal_top

    # Domain validation
    # Expected: Upstream FKs (DRIVER, QUALIFYING) should have higher causal attribution
    # Downstream FKs (RESULTS, STANDINGS) should have lower causal attribution
    domain_makes_sense = False
    if causal_top in ['DRIVER', 'QUALIFYING']:
        domain_makes_sense = True
        explanation = f"{causal_top} is upstream in causal chain - makes sense!"
    elif causal_top in ['RESULTS', 'STANDINGS']:
        explanation = f"{causal_top} is downstream - might be confounded"
    else:
        explanation = "Unknown FK - cannot validate"

    print(f"Rankings differ: {rankings_differ}")
    print(f"Domain validation: {explanation}")

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    if rankings_differ and domain_makes_sense:
        verdict = "✅ PASS"
        recommendation = "Causal attribution provides meaningful insights - strong contribution for UAI/CLeaR"
    elif rankings_differ:
        verdict = "⚠️  MARGINAL"
        recommendation = "Rankings differ but domain validation unclear - needs more investigation"
    else:
        verdict = "❌ FAIL"
        recommendation = "Causal same as observational - no added value, drop this direction"

    print(f"\nResult: {verdict}")
    print(f"Rankings differ: {rankings_differ}")
    print(f"Domain makes sense: {domain_makes_sense}")
    print(f"\nRecommendation: {recommendation}")

    print("\n⚠️  NOTE: Full implementation would need:")
    print("  1. Formal DAG construction from schema")
    print("  2. Do-calculus for confounding adjustment")
    print("  3. Multiple domain validation")
    print("  4. Expert validation of causal rankings")
    print("\nThis test shows basic feasibility.")

    # Save results
    output_dir = Path(__file__).parent / 'test_results'
    output_dir.mkdir(exist_ok=True)

    output = {
        'test': 'causal_attribution',
        'dataset': dataset_name,
        'task': task_name,
        'verdict': verdict,
        'recommendation': recommendation,
        'results': {
            'observational': {k: float(v) for k, v in obs_attr.items()},
            'causal': {k: float(v) for k, v in causal_attr.items()},
            'obs_top': obs_top,
            'causal_top': causal_top,
            'rankings_differ': rankings_differ,
            'domain_makes_sense': domain_makes_sense,
        }
    }

    output_file = output_dir / 'test_4_causal.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return output


if __name__ == '__main__':
    result = main()
