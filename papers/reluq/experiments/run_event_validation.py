"""
Run FK Uncertainty Framework validation on rel-event dataset

rel-event is an event/meetup platform with:
- users: User profiles
- events: Event listings (110 features)
- event_attendees: Who attended which events
- user_friends: Social network
- event_interest: User interest in events

FK Classification:
- USERS: User profile features (correlational - user behavior patterns)
- EVENTS: Event features (causal - event characteristics drive attendance)
- EVENT_ATTENDEES: Attendance history (correlational - past behavior)
- USER_FRIENDS: Social network (correlational - social influence)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from relbench.datasets import get_dataset
from relbench.tasks import get_task

# FK Classification for rel-event
EVENT_FK_CLASSIFICATION = {
    'USERS': 'correlational',         # User behavior is correlational
    'EVENTS': 'causal',               # Event characteristics cause attendance
    'EVENT_ATTENDEES': 'correlational', # Past attendance is correlational
    'USER_FRIENDS': 'correlational',  # Social network is correlational
    'EVENT_INTEREST': 'correlational', # Interest signals are correlational
    'ENTITY': 'causal',
    'TRAIN': 'causal',
}


def extract_features_generic(dataset, task, sample_size=3000):
    """Generic feature extraction for rel-event."""
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
    fk_to_entity = fk_cols[0] if fk_cols else task.entity_col
    entity_pkey = entity_table.pkey_col

    # Merge
    merged_df = train_df.merge(entity_df, how='left', left_on=fk_to_entity,
                                right_on=entity_pkey, suffixes=('', '_entity'))

    target_col = task.target_col
    y_raw = merged_df[target_col].values

    # Determine task type
    is_classification = 'classification' in str(task.task_type).lower()

    if is_classification:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_clean = np.array([str(x) if pd.notna(x) else 'unknown' for x in y_raw])
        y = le.fit_transform(y_clean)
        n_classes = len(le.classes_)
    else:
        y = np.array(y_raw, dtype=float)
        y = np.nan_to_num(y, nan=np.nanmean(y[~np.isnan(y)]) if np.any(~np.isnan(y)) else 0)
        n_classes = 0

    col_to_fk = {}
    feature_cols = []
    skip_cols = {target_col, entity_pkey, fk_to_entity, 'timestamp', 'Timestamp'}

    # Get entity features (limit to avoid too many)
    entity_numeric = [c for c in entity_df.columns
                     if entity_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
                     and c not in skip_cols and not c.endswith('_id') and not c.endswith('ID')]

    for col in entity_numeric[:20]:  # Limit to 20 features
        col_name = col if col in merged_df.columns else f"{col}_entity"
        if col_name in merged_df.columns and col_name not in feature_cols:
            feature_cols.append(col_name)
            col_to_fk[col_name] = entity_table_name.upper()

    # Get FK table features
    if hasattr(entity_table, 'fkey_col_to_pkey_table'):
        for fk_col, ref_table_name in entity_table.fkey_col_to_pkey_table.items():
            if ref_table_name in db.table_dict:
                ref_table = db.table_dict[ref_table_name]
                ref_df = ref_table.df
                ref_pkey = ref_table.pkey_col

                numeric_cols = [c for c in ref_df.select_dtypes(include=[np.number]).columns
                               if c != ref_pkey][:5]

                if numeric_cols and fk_col in merged_df.columns:
                    for col in numeric_cols:
                        new_col = f'{ref_table_name}_{col}'
                        if new_col not in merged_df.columns:
                            mapping = ref_df.set_index(ref_pkey)[col].to_dict()
                            merged_df[new_col] = merged_df[fk_col].map(mapping)
                            if new_col in merged_df.columns:
                                feature_cols.append(new_col)
                                col_to_fk[new_col] = ref_table_name.upper()

    # Get referencing table features (aggregated)
    for table_name, table in db.table_dict.items():
        if table_name == entity_table_name:
            continue
        if hasattr(table, 'fkey_col_to_pkey_table'):
            for fk_col, ref_table in table.fkey_col_to_pkey_table.items():
                if ref_table == entity_table_name:
                    table_df = table.df
                    numeric_cols = [c for c in table_df.select_dtypes(include=[np.number]).columns
                                   if c != fk_col][:3]
                    if numeric_cols:
                        try:
                            agg_df = table_df.groupby(fk_col)[numeric_cols].agg(['mean', 'count']).reset_index()
                            agg_df.columns = [fk_col] + [f'{table_name}_{c[0]}_{c[1]}' for c in agg_df.columns[1:]]
                            merge_col = fk_to_entity if fk_to_entity in merged_df.columns else entity_pkey
                            merged_df = merged_df.merge(agg_df, how='left', left_on=merge_col,
                                                        right_on=fk_col, suffixes=('', f'_{table_name}'))
                            for col in agg_df.columns[1:]:
                                if col in merged_df.columns and col not in feature_cols:
                                    feature_cols.append(col)
                                    col_to_fk[col] = table_name.upper()
                        except Exception:
                            pass

    if len(feature_cols) == 0:
        for col in merged_df.columns:
            if merged_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                if col not in skip_cols:
                    feature_cols.append(col)
                    col_to_fk[col] = 'ENTITY'

    X = merged_df[feature_cols].fillna(0).values

    fk_to_cols = defaultdict(list)
    for i, col in enumerate(feature_cols):
        fk_name = col_to_fk.get(col, 'UNKNOWN')
        fk_to_cols[fk_name].append(i)

    return X, y, col_to_fk, feature_cols, dict(fk_to_cols), n_classes, is_classification


def train_ensemble(X, y, n_models=5, n_classes=0, is_classification=True, seed=42):
    """Train LightGBM ensemble."""
    models = []
    for i in range(n_models):
        idx = np.random.RandomState(seed+i).choice(len(X), int(0.8 * len(X)), replace=True)

        if is_classification:
            unique_classes = np.unique(y)
            bootstrap_classes = np.unique(y[idx])
            missing = set(unique_classes) - set(bootstrap_classes)
            if missing:
                extra = [np.where(y == c)[0][0] for c in missing if len(np.where(y == c)[0]) > 0]
                if extra:
                    idx = np.concatenate([idx, extra])

            model = lgb.LGBMClassifier(
                n_estimators=50, max_depth=6, learning_rate=0.1,
                random_state=seed+i, verbose=-1,
                num_class=n_classes if n_classes > 2 else None,
                objective='multiclass' if n_classes > 2 else 'binary'
            )
        else:
            model = lgb.LGBMRegressor(
                n_estimators=50, max_depth=6, learning_rate=0.1,
                random_state=seed+i, verbose=-1
            )

        model.fit(X[idx], y[idx])
        models.append(model)
    return models


def ensemble_variance(models, X, is_classification=True):
    """Compute ensemble variance."""
    if is_classification:
        all_probs = [m.predict_proba(X) for m in models]
        max_classes = max(p.shape[1] for p in all_probs)
        padded = []
        for p in all_probs:
            if p.shape[1] < max_classes:
                pad = np.zeros((p.shape[0], max_classes))
                pad[:, :p.shape[1]] = p
                padded.append(pad)
            else:
                padded.append(p)
        probs = np.array(padded)
        return probs.var(axis=0).mean(axis=1)
    else:
        preds = np.array([m.predict(X) for m in models])
        return preds.var(axis=0)


def compute_fk_uncertainty(models, X, fk_to_cols, is_classification=True, n_perm=10):
    """Compute FK uncertainty contribution."""
    base_unc = np.mean(ensemble_variance(models, X, is_classification))

    fk_contributions = {}
    for fk_name, col_indices in fk_to_cols.items():
        if not col_indices:
            continue
        contribs = []
        for _ in range(n_perm):
            X_perm = X.copy()
            for idx in col_indices:
                X_perm[:, idx] = np.random.permutation(X_perm[:, idx])
            perm_unc = np.mean(ensemble_variance(models, X_perm, is_classification))
            contribs.append((base_unc - perm_unc) / base_unc * 100)
        fk_contributions[fk_name] = {'mean': np.mean(contribs), 'std': np.std(contribs)}

    return fk_contributions, base_unc


def run_experiment(task_name, seed=42, sample_size=2000):
    """Run single experiment."""
    print(f"\n{'='*60}")
    print(f"Running: rel-event / {task_name}")
    print(f"{'='*60}")

    dataset = get_dataset('rel-event', download=False)
    task = get_task('rel-event', task_name, download=False)

    print(f"  Task type: {task.task_type}")
    print(f"  Entity: {task.entity_table}")

    try:
        X, y, col_to_fk, feature_cols, fk_to_cols, n_classes, is_class = extract_features_generic(
            dataset, task, sample_size=sample_size
        )
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None

    print(f"  Data shape: {X.shape}")
    print(f"  Classification: {is_class}")
    print(f"  FK groups: {list(fk_to_cols.keys())}")

    if X.shape[1] < 1:
        print("  ERROR: No features")
        return None

    models = train_ensemble(X, y, n_models=5, n_classes=n_classes, is_classification=is_class, seed=seed)
    fk_contribs, base_unc = compute_fk_uncertainty(models, X, fk_to_cols, is_class)

    print(f"\n  Base uncertainty: {base_unc:.6f}")
    for fk, c in sorted(fk_contribs.items(), key=lambda x: -x[1]['mean']):
        fk_type = EVENT_FK_CLASSIFICATION.get(fk, 'unknown')
        print(f"    {fk}: {c['mean']:+.2f}% ± {c['std']:.2f}% ({fk_type})")

    causal = [c['mean'] for fk, c in fk_contribs.items() if EVENT_FK_CLASSIFICATION.get(fk) == 'causal']
    corr = [c['mean'] for fk, c in fk_contribs.items() if EVENT_FK_CLASSIFICATION.get(fk) == 'correlational']

    hyp = False
    if causal and corr:
        diff = np.mean(causal) - np.mean(corr)
        hyp = diff > 0
        print(f"\n  Causal mean: {np.mean(causal):+.2f}%, Corr mean: {np.mean(corr):+.2f}%")
        print(f"  Hypothesis supported: {'YES' if hyp else 'NO'}")

    return {
        'task': task_name, 'seed': seed, 'task_type': str(task.task_type),
        'data_shape': list(X.shape),
        'fk_contributions': {k: v['mean'] for k, v in fk_contribs.items()},
        'base_uncertainty': float(base_unc), 'hypothesis_supported': hyp
    }


def run_multi_seed(task_name, seeds=[42, 43, 44, 45, 46]):
    """Run with multiple seeds."""
    print(f"\n{'#'*60}")
    print(f"MULTI-SEED: rel-event / {task_name}")
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

    print(f"\n{'='*60}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*60}")

    all_fks = set()
    for r in results:
        all_fks.update(r['fk_contributions'].keys())

    fk_summary = {}
    for fk in sorted(all_fks):
        contribs = [r['fk_contributions'].get(fk, 0) for r in results]
        mean_c, std_c = np.mean(contribs), np.std(contribs)
        fk_type = EVENT_FK_CLASSIFICATION.get(fk, 'unknown')
        print(f"  {fk}: {mean_c:+.2f}% ± {std_c:.2f}% ({fk_type})")
        fk_summary[fk] = {'mean': mean_c, 'std': std_c, 'type': fk_type}

    support = sum(1 for r in results if r.get('hypothesis_supported', False))
    print(f"\nHypothesis supported: {support}/{len(results)} seeds")

    return {
        'task': task_name, 'seeds': seeds, 'fk_summary': fk_summary,
        'hypothesis_support_rate': f"{support}/{len(results)}", 'results': results
    }


def main():
    tasks = ['user-attendance', 'user-repeat']

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

    os.makedirs('test_results', exist_ok=True)
    with open('test_results/event_validation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"\n{r['task']}:")
        print(f"  Hypothesis support: {r['hypothesis_support_rate']}")
        sorted_fks = sorted(r['fk_summary'].items(), key=lambda x: -x[1]['mean'])[:3]
        for fk, data in sorted_fks:
            print(f"    {fk}: {data['mean']:+.2f}% ({data['type']})")

    print(f"\nResults saved to test_results/event_validation_results.json")


if __name__ == '__main__':
    main()
