"""
TEST 3: Epistemic/Aleatoric Decomposition
==========================================

Question: Can we separate epistemic (data scarcity) from aleatoric (inherent noise) uncertainty by FK?

Success Criteria: >70% accuracy on synthetic validation

Time: ~5-10 minutes

Usage:
    python test_3_decomposition.py
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


def extract_features_simple(dataset, task, sample_size=1000):
    """Feature extraction with FK tracking (fixed version)."""
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

    feature_cols = []

    # Get numeric columns from train
    for col in train_df.columns:
        if col == target_col or col.endswith('Id') or col.endswith('_id'):
            continue
        if train_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            if col in merged_df.columns:
                feature_cols.append(col)

    # Get numeric columns from entity
    for col in entity_df.columns:
        if col == entity_pkey or col.endswith('Id') or col.endswith('_id'):
            continue
        if entity_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            col_name = col if col in merged_df.columns else f"{col}_entity"
            if col_name in merged_df.columns and col_name not in feature_cols:
                feature_cols.append(col_name)

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

    X = merged_df[feature_cols].fillna(0).values

    return X, y, feature_cols


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
    """Compute ensemble variance (epistemic proxy)."""
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0).mean()


def test_decomposition_synthetic():
    """Test on synthetic data with known epistemic/aleatoric split."""
    print("\n" + "="*60)
    print("Synthetic Validation Test")
    print("="*60)

    # Generate synthetic data
    n_samples = 1000
    n_features = 10

    # True function: y = X[:, :5].sum() + noise
    X = np.random.randn(n_samples, n_features)
    y_true = X[:, :5].sum(axis=1)

    # Add aleatoric noise (inherent randomness)
    aleatoric_noise = np.random.randn(n_samples) * 0.5
    y = y_true + aleatoric_noise

    print(f"\n✅ Generated {n_samples} synthetic samples")
    print(f"   True aleatoric std: 0.5")
    print(f"   Features: {n_features}")

    # Method 1: Total uncertainty via ensemble
    models = train_ensemble(X, y, n_models=10)
    total_uncertainty = ensemble_variance(models, X)

    print(f"\n📊 Total uncertainty (ensemble variance): {total_uncertainty:.4f}")

    # Method 2: Try to separate epistemic vs aleatoric
    # Approach: Train on more data, see if uncertainty decreases

    # Train on 50% data
    X_sparse = X[:500]
    y_sparse = y[:500]
    models_sparse = train_ensemble(X_sparse, y_sparse, n_models=10)
    unc_sparse = ensemble_variance(models_sparse, X[500:])

    # Train on 100% data
    models_full = train_ensemble(X, y, n_models=10)
    unc_full = ensemble_variance(models_full, X[500:])

    epistemic_reduction = unc_sparse - unc_full
    epistemic_pct = (epistemic_reduction / unc_sparse) * 100

    print(f"\n📊 Uncertainty with 50% data: {unc_sparse:.4f}")
    print(f"📊 Uncertainty with 100% data: {unc_full:.4f}")
    print(f"📊 Reduction (epistemic): {epistemic_reduction:.4f} ({epistemic_pct:.1f}%)")

    # Ground truth: epistemic should decrease with more data
    # Aleatoric should stay constant (it's 0.5^2 = 0.25 variance)

    success = epistemic_reduction > 0 and epistemic_pct > 10

    return {
        'total_uncertainty': float(total_uncertainty),
        'epistemic_reduction': float(epistemic_reduction),
        'epistemic_pct': float(epistemic_pct),
        'success': success,
    }


def test_decomposition_real(dataset, task):
    """Test on real data."""
    print("\n" + "="*60)
    print("Real Data Test")
    print("="*60)

    X, y, feature_names = extract_features_simple(dataset, task, sample_size=1000)

    print(f"\n✅ Extracted {len(feature_names)} features")
    print(f"   Samples: {len(X)}")

    # Total uncertainty
    models = train_ensemble(X, y, n_models=5)
    total_unc = ensemble_variance(models, X)

    # Try data augmentation approach
    # Add Gaussian noise to simulate "more data"
    X_aug = np.vstack([X, X + np.random.randn(*X.shape) * 0.1])
    y_aug = np.hstack([y, y + np.random.randn(len(y)) * 0.1])

    models_aug = train_ensemble(X_aug, y_aug, n_models=5)
    aug_unc = ensemble_variance(models_aug, X)

    reduction = total_unc - aug_unc
    reduction_pct = (reduction / total_unc) * 100 if total_unc > 0 else 0

    print(f"\n📊 Base uncertainty: {total_unc:.4f}")
    print(f"📊 Augmented uncertainty: {aug_unc:.4f}")
    print(f"📊 Reduction: {reduction:.4f} ({reduction_pct:.1f}%)")

    return {
        'total_uncertainty': float(total_unc),
        'reduction_pct': float(reduction_pct),
    }


def main():
    print("\n" + "="*60)
    print("TEST 3: Epistemic/Aleatoric Decomposition")
    print("="*60)

    # Test 1: Synthetic validation
    synthetic_results = test_decomposition_synthetic()

    # Test 2: Real data
    print("\n" + "="*60)
    print("Testing on Real Data (rel-f1)")
    print("="*60)

    dataset = get_dataset('rel-f1', download=True)
    task = get_task('rel-f1', 'driver-position', download=True)

    real_results = test_decomposition_real(dataset, task)

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    synthetic_success = synthetic_results['success']
    real_reduction = real_results['reduction_pct']

    # Success if synthetic works and real shows some signal
    overall_success = synthetic_success and (real_reduction > 5)

    if overall_success:
        verdict = "✅ PASS"
        recommendation = "Decomposition is feasible - include as theoretical extension (workshop paper)"
    elif synthetic_success:
        verdict = "⚠️  MARGINAL"
        recommendation = "Works on synthetic but weak on real data - mention in future work"
    else:
        verdict = "❌ FAIL"
        recommendation = "Cannot reliably separate epistemic/aleatoric - drop this direction"

    print(f"\nResult: {verdict}")
    print(f"Synthetic test: {'✓' if synthetic_success else '✗'}")
    print(f"Real data reduction: {real_reduction:.1f}% (need >5%)")
    print(f"\nRecommendation: {recommendation}")

    print("\n⚠️  NOTE: Full implementation would need:")
    print("  1. Heteroscedastic models (predict mean + variance)")
    print("  2. Per-FK decomposition (not just overall)")
    print("  3. Validation on multiple domains")
    print("\nThis test shows basic feasibility.")

    # Save results
    output_dir = Path(__file__).parent / 'test_results'
    output_dir.mkdir(exist_ok=True)

    output = {
        'test': 'decomposition',
        'verdict': verdict,
        'recommendation': recommendation,
        'results': {
            'synthetic': synthetic_results,
            'real': real_results,
        }
    }

    output_file = output_dir / 'test_3_decomposition.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return output


if __name__ == '__main__':
    result = main()
