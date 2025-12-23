"""
Run FK Uncertainty Framework validation on rel-salt dataset

rel-salt is an ERP (SAP) dataset with the following structure:
- salesdocument: Sales orders
- salesdocumentitem: Line items (entity table for most tasks)
- customer: Customer info (with address FK)
- address: Geographic info

FK Classification:
- SALESDOCUMENT: Sales order context (causal - order determines item behavior)
- CUSTOMER (SOLDTOPARTY, SHIPTOPARTY, BILLTOPARTY, PAYERPARTY): Customer info (correlational)
- ADDRESS: Geographic info (correlational)

COVID Context:
- This dataset spans pre/during COVID period
- Distribution shift expected around Feb 2020
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from relbench.datasets import get_dataset
from relbench.tasks import get_task

# FK Classification for rel-salt based on domain knowledge
SALT_FK_CLASSIFICATION = {
    'SALESDOCUMENT': 'causal',      # Sales order context determines item behavior
    'SALESDOCUMENTITEM': 'causal',  # Entity features
    'SOLDTOPARTY': 'correlational', # Customer type doesn't cause plant assignment
    'SHIPTOPARTY': 'correlational', # Shipping customer is correlational
    'BILLTOPARTY': 'correlational', # Billing customer is correlational
    'PAYERPARTY': 'correlational',  # Payer is correlational
    'CUSTOMER': 'correlational',    # Generic customer info
    'ADDRESS': 'correlational',     # Geographic info is correlational
    'TRAIN': 'causal',              # Direct task features
}


def extract_features_with_fk_classification(dataset, task, sample_size=3000):
    """
    Extract features for classification tasks and track FK groups.
    Adapted for rel-salt schema.
    """
    db = dataset.get_db()
    train_table = task.get_table("train")

    entity_table_name = task.entity_table
    entity_table = db.table_dict[entity_table_name]
    entity_df = entity_table.df.copy()
    train_df = train_table.df.copy()

    if len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=42)

    # Get FK relationship
    fk_cols = list(train_table.fkey_col_to_pkey_table.keys())
    if not fk_cols:
        # Task table might reference entity directly
        fk_to_entity = task.entity_col
    else:
        fk_to_entity = fk_cols[0]

    entity_pkey = entity_table.pkey_col

    # Merge train with entity
    merged_df = train_df.merge(entity_df, how='left', left_on=fk_to_entity,
                                right_on=entity_pkey, suffixes=('', '_entity'))

    target_col = task.target_col
    y_raw = merged_df[target_col].values

    # For classification, encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    col_to_fk = {}
    feature_cols = []

    # Skip columns that are IDs or targets
    skip_cols = {target_col, entity_pkey, fk_to_entity, 'ID', 'CREATIONTIMESTAMP'}

    # Get entity table features
    for col in entity_df.columns:
        if col in skip_cols:
            continue
        if col.endswith('Id') or col.endswith('_id') or col.endswith('ID'):
            continue
        if entity_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            col_name = col if col in merged_df.columns else f"{col}_entity"
            if col_name in merged_df.columns and col_name not in feature_cols:
                feature_cols.append(col_name)
                # Map to FK based on column name
                fk_name = identify_fk_source(col, entity_table_name)
                col_to_fk[col_name] = fk_name

    # Get FK table features (aggregated)
    for table_name, table in db.table_dict.items():
        if table_name == entity_table_name:
            continue

        if hasattr(table, 'fkey_col_to_pkey_table'):
            for fk_col, ref_table in table.fkey_col_to_pkey_table.items():
                if ref_table == entity_table_name:
                    table_df = table.df
                    numeric_cols = [c for c in table_df.select_dtypes(include=[np.number]).columns
                                   if not c.endswith('Id') and not c.endswith('ID') and c != fk_col]

                    if numeric_cols:
                        agg_df = table_df.groupby(fk_col)[numeric_cols].mean().reset_index()
                        agg_df.columns = [fk_col] + [f'{table_name}_{c}_mean' for c in numeric_cols]

                        merged_df = merged_df.merge(agg_df, how='left', left_on=fk_to_entity,
                                                    right_on=fk_col, suffixes=('', f'_{table_name}'))

                        for col in agg_df.columns[1:]:
                            if col in merged_df.columns and col not in feature_cols:
                                feature_cols.append(col)
                                col_to_fk[col] = table_name.upper()

    # Also try to join via entity table's FKs
    if hasattr(entity_table, 'fkey_col_to_pkey_table'):
        for fk_col, ref_table_name in entity_table.fkey_col_to_pkey_table.items():
            if ref_table_name in db.table_dict:
                ref_table = db.table_dict[ref_table_name]
                ref_df = ref_table.df
                ref_pkey = ref_table.pkey_col

                # Get numeric columns from referenced table
                numeric_cols = [c for c in ref_df.select_dtypes(include=[np.number]).columns
                               if c != ref_pkey]

                if numeric_cols and fk_col in merged_df.columns:
                    for col in numeric_cols:
                        new_col = f'{ref_table_name}_{col}'
                        if new_col not in merged_df.columns:
                            # Create mapping
                            mapping = ref_df.set_index(ref_pkey)[col].to_dict()
                            merged_df[new_col] = merged_df[fk_col].map(mapping)

                            if new_col in merged_df.columns:
                                feature_cols.append(new_col)
                                col_to_fk[new_col] = fk_col.upper()

    # If still no features, use all numeric columns from entity
    if len(feature_cols) == 0:
        for col in merged_df.columns:
            if col in skip_cols:
                continue
            if merged_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                feature_cols.append(col)
                col_to_fk[col] = 'ENTITY'

    X = merged_df[feature_cols].fillna(0).values

    # Create FK column index mapping
    fk_to_cols = defaultdict(list)
    for i, col in enumerate(feature_cols):
        fk_name = col_to_fk.get(col, 'UNKNOWN')
        fk_to_cols[fk_name].append(i)

    return X, y, col_to_fk, feature_cols, dict(fk_to_cols), n_classes


def identify_fk_source(col_name, entity_table_name):
    """Identify which FK a column belongs to based on naming patterns."""
    col_upper = col_name.upper()

    # Check for specific FK patterns in rel-salt
    if 'SALESDOCUMENT' in col_upper:
        return 'SALESDOCUMENT'
    elif 'SOLDTO' in col_upper:
        return 'SOLDTOPARTY'
    elif 'SHIPTO' in col_upper:
        return 'SHIPTOPARTY'
    elif 'BILLTO' in col_upper:
        return 'BILLTOPARTY'
    elif 'PAYER' in col_upper:
        return 'PAYERPARTY'
    elif 'CUSTOMER' in col_upper:
        return 'CUSTOMER'
    elif 'ADDRESS' in col_upper:
        return 'ADDRESS'
    else:
        return entity_table_name.upper()


def train_ensemble_classifier(X, y, n_models=5, n_classes=2, seed=42):
    """Train LightGBM ensemble for classification."""
    models = []
    for i in range(n_models):
        idx = np.random.RandomState(seed+i).choice(len(X), int(0.8 * len(X)), replace=True)

        # Ensure all classes are represented in bootstrap sample
        # by adding at least one sample from each class if missing
        unique_classes = np.unique(y)
        bootstrap_classes = np.unique(y[idx])
        missing_classes = set(unique_classes) - set(bootstrap_classes)

        if missing_classes:
            # Add one sample from each missing class
            extra_idx = []
            for cls in missing_classes:
                cls_indices = np.where(y == cls)[0]
                if len(cls_indices) > 0:
                    extra_idx.append(cls_indices[0])
            if extra_idx:
                idx = np.concatenate([idx, extra_idx])

        model = lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed+i,
            verbose=-1,
            num_class=n_classes if n_classes > 2 else None,
            objective='multiclass' if n_classes > 2 else 'binary'
        )
        model.fit(X[idx], y[idx])
        models.append(model)

    return models


def ensemble_variance_classifier(models, X, n_classes=None):
    """Compute ensemble variance for classification (using probability predictions)."""
    # Get probability predictions from each model
    # Handle case where models might have different class subsets
    all_probs = []
    for m in models:
        prob = m.predict_proba(X)
        all_probs.append(prob)

    # Find the maximum number of classes across all models
    max_classes = max(p.shape[1] for p in all_probs)

    # Pad probabilities to same shape if needed
    padded_probs = []
    for prob in all_probs:
        if prob.shape[1] < max_classes:
            # Pad with zeros for missing classes
            padded = np.zeros((prob.shape[0], max_classes))
            padded[:, :prob.shape[1]] = prob
            padded_probs.append(padded)
        else:
            padded_probs.append(prob)

    probs = np.array(padded_probs)  # Shape: (n_models, n_samples, n_classes)

    # Variance across models for each sample (average across classes)
    var_per_class = probs.var(axis=0)  # Shape: (n_samples, n_classes)
    return var_per_class.mean(axis=1)  # Shape: (n_samples,)


def compute_fk_uncertainty_contribution(models, X, fk_to_cols, n_permutations=10, is_classifier=True):
    """Compute FK-level uncertainty contribution via permutation."""
    if is_classifier:
        base_uncertainty = np.mean(ensemble_variance_classifier(models, X))
    else:
        base_uncertainty = np.mean(ensemble_variance_regressor(models, X))

    fk_contributions = {}
    for fk_name, col_indices in fk_to_cols.items():
        if not col_indices:
            continue

        contributions = []
        for _ in range(n_permutations):
            X_perm = X.copy()
            for col_idx in col_indices:
                X_perm[:, col_idx] = np.random.permutation(X_perm[:, col_idx])

            if is_classifier:
                perm_uncertainty = np.mean(ensemble_variance_classifier(models, X_perm))
            else:
                perm_uncertainty = np.mean(ensemble_variance_regressor(models, X_perm))

            # Contribution = (base - permuted) / base * 100
            # Positive = FK was adding uncertainty (when we break it, uncertainty decreases)
            # Negative = FK was reducing uncertainty (when we break it, uncertainty increases)
            contribution = (base_uncertainty - perm_uncertainty) / base_uncertainty * 100
            contributions.append(contribution)

        fk_contributions[fk_name] = {
            'mean': np.mean(contributions),
            'std': np.std(contributions)
        }

    return fk_contributions, base_uncertainty


def ensemble_variance_regressor(models, X):
    """Compute ensemble variance for regression."""
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0)


def run_experiment(task_name, n_models=5, sample_size=2000, seed=42):
    """Run experiment for a single task."""
    print(f"\n{'='*60}")
    print(f"Running: rel-salt / {task_name}")
    print(f"{'='*60}")

    dataset = get_dataset('rel-salt', download=True)
    task = get_task('rel-salt', task_name, download=False)

    print(f"  Task type: {task.task_type}")

    print("Extracting features...")
    try:
        X, y, col_to_fk, feature_cols, fk_to_cols, n_classes = extract_features_with_fk_classification(
            dataset, task, sample_size=sample_size
        )
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None

    print(f"  Data shape: {X.shape}")
    print(f"  Target classes: {n_classes}")
    print(f"  FK groups: {list(fk_to_cols.keys())}")
    for fk, cols in fk_to_cols.items():
        print(f"    {fk}: {len(cols)} features")

    if X.shape[1] < 1:
        print("  ERROR: Not enough features")
        return None

    print(f"Training ensemble ({n_models} models)...")
    models = train_ensemble_classifier(X, y, n_models=n_models, n_classes=n_classes, seed=seed)

    print("Computing FK uncertainty contributions...")
    fk_contributions, base_uncertainty = compute_fk_uncertainty_contribution(
        models, X, fk_to_cols, is_classifier=True
    )

    print(f"\n  Base uncertainty: {base_uncertainty:.6f}")
    print("  FK contributions:")
    for fk_name, contrib in sorted(fk_contributions.items(), key=lambda x: -x[1]['mean']):
        fk_type = SALT_FK_CLASSIFICATION.get(fk_name, 'unknown')
        print(f"    {fk_name}: {contrib['mean']:+.2f}% ± {contrib['std']:.2f}% ({fk_type})")

    # Classify FKs and compute hypothesis support
    causal = [c['mean'] for fk, c in fk_contributions.items()
              if SALT_FK_CLASSIFICATION.get(fk) == 'causal']
    correlational = [c['mean'] for fk, c in fk_contributions.items()
                    if SALT_FK_CLASSIFICATION.get(fk) == 'correlational']

    hypothesis_supported = False
    if causal and correlational:
        causal_mean = np.mean(causal)
        corr_mean = np.mean(correlational)
        diff = causal_mean - corr_mean
        print(f"\n  Causal FKs mean: {causal_mean:+.2f}%")
        print(f"  Correlational FKs mean: {corr_mean:+.2f}%")
        print(f"  Difference: {diff:+.2f}%")
        hypothesis_supported = diff > 0
        print(f"  Original hypothesis supported: {'YES' if hypothesis_supported else 'NO'}")

    return {
        'task': task_name,
        'seed': seed,
        'task_type': str(task.task_type),
        'n_classes': n_classes,
        'data_shape': list(X.shape),
        'fk_contributions': {k: v['mean'] for k, v in fk_contributions.items()},
        'fk_contributions_std': {k: v['std'] for k, v in fk_contributions.items()},
        'base_uncertainty': float(base_uncertainty),
        'hypothesis_supported': hypothesis_supported
    }


def run_multi_seed(task_name, seeds=[42, 43, 44, 45, 46]):
    """Run with multiple seeds."""
    print(f"\n{'#'*60}")
    print(f"MULTI-SEED: rel-salt / {task_name}")
    print(f"{'#'*60}")

    results = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed)
        r = run_experiment(task_name, seed=seed)
        if r:
            results.append(r)

    if not results:
        return None

    # Aggregate
    print(f"\n{'='*60}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*60}")

    all_fks = set()
    for r in results:
        all_fks.update(r['fk_contributions'].keys())

    print("\nFK Contribution Across Seeds:")
    fk_summary = {}
    for fk in sorted(all_fks):
        contribs = [r['fk_contributions'].get(fk, 0) for r in results]
        mean_contrib = np.mean(contribs)
        std_contrib = np.std(contribs)
        fk_type = SALT_FK_CLASSIFICATION.get(fk, 'unknown')
        print(f"  {fk}: {mean_contrib:+.2f}% ± {std_contrib:.2f}% ({fk_type})")
        fk_summary[fk] = {'mean': mean_contrib, 'std': std_contrib, 'type': fk_type}

    # Hypothesis support rate
    support_count = sum(1 for r in results if r.get('hypothesis_supported', False))
    print(f"\nHypothesis supported: {support_count}/{len(results)} seeds")

    return {
        'task': task_name,
        'seeds': seeds,
        'fk_summary': fk_summary,
        'hypothesis_support_rate': f"{support_count}/{len(results)}",
        'results': results
    }


def main():
    # Select diverse tasks: 2 classification, 1 that's different
    tasks = ['item-plant', 'item-shippoint', 'sales-payterms']

    all_results = []
    for task in tasks:
        try:
            result = run_multi_seed(task)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error with {task}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    os.makedirs('test_results', exist_ok=True)
    output_file = 'test_results/salt_validation_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    for result in all_results:
        print(f"\n{result['task']}:")
        print(f"  Hypothesis support: {result['hypothesis_support_rate']}")
        print(f"  Top FKs by uncertainty:")
        sorted_fks = sorted(result['fk_summary'].items(), key=lambda x: -x[1]['mean'])[:3]
        for fk, data in sorted_fks:
            print(f"    {fk}: {data['mean']:+.2f}% ({data['type']})")

    print(f"\n\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
