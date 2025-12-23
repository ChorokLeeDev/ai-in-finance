"""
RelUQ Intervention-Focused Experiments
=======================================

Pivoted approach: Focus on intervention effect as the main validation metric.
This works regardless of the number of FK groups.

Key metrics:
1. Intervention effect: Does fixing top FK group reduce error?
2. Stability: FK-level vs feature-level attribution stability across seeds
3. Sample-level: Does uncertainty correlate with error per-sample?
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import lightgbm as lgb
from relbench.datasets import get_dataset
from relbench.tasks import get_task, get_task_names


@dataclass
class DomainClassification:
    """Pre-registered domain classification."""
    dataset: str
    is_cep: bool
    rationale: str


# Pre-registered before experiments
DOMAIN_CLASSIFICATIONS = {
    'rel-f1': DomainClassification('rel-f1', True, 'Racing: driver->race->result causal chain'),
    'rel-trial': DomainClassification('rel-trial', True, 'Clinical: study->site->outcome'),
    'rel-avito': DomainClassification('rel-avito', True, 'Classifieds: user->ad->interaction'),
    'rel-stack': DomainClassification('rel-stack', False, 'Q&A: bidirectional user-post'),
    'rel-amazon': DomainClassification('rel-amazon', False, 'E-commerce: associative'),
    'rel-hm': DomainClassification('rel-hm', False, 'Fashion: user-item recommendations'),
}

# Tasks that exist in RelBench registry
VALID_TASKS = {
    'rel-f1': ['driver-position', 'driver-dnf', 'driver-top3'],
    'rel-trial': ['study-outcome', 'site-success', 'study-adverse'],
    'rel-avito': ['user-visits', 'user-clicks'],
    'rel-stack': ['user-engagement', 'post-votes'],
}


def extract_features(dataset, task) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, str]]:
    """Extract features with FK tracking."""
    db = dataset.get_db()
    train_table = task.get_table("train")

    entity_table_name = task.entity_table
    entity_table = db.table_dict[entity_table_name]
    entity_df = entity_table.df.copy()
    train_df = train_table.df.copy()

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

    # Get numeric columns
    for col in merged_df.columns:
        if col == target_col or col.endswith('Id') or col.endswith('_id'):
            continue
        if merged_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            feature_cols.append(col)
            if col in train_df.columns:
                col_to_fk[col] = 'TRAIN'
            else:
                col_to_fk[col] = entity_table_name.upper()

    # Join FK tables that reference entity
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
                        agg_df = table_df.groupby(fk_col)[numeric_cols].agg(['mean', 'count']).reset_index()
                        agg_df.columns = [fk_col] + [f'{table_name}_{c}_{s}' for c, s in
                                                      [col.split('_')[-2:] if '_' in str(col) else (col, 'val')
                                                       for col in agg_df.columns[1:]]]

                        # Simpler column naming
                        new_cols = [fk_col]
                        for i, orig_col in enumerate(numeric_cols):
                            new_cols.append(f'{table_name}_{orig_col}_mean')
                            new_cols.append(f'{table_name}_{orig_col}_count')

                        if len(new_cols) == len(agg_df.columns):
                            agg_df.columns = new_cols

                        merged_df = merged_df.merge(agg_df, how='left', left_on=fk_to_entity,
                                                    right_on=fk_col, suffixes=('', f'_{table_name}'))

                        for col in agg_df.columns[1:]:
                            if col in merged_df.columns and col not in feature_cols:
                                feature_cols.append(col)
                                col_to_fk[col] = table_name.upper()

    X = merged_df[feature_cols].copy() if feature_cols else pd.DataFrame()

    # Fill NaN
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)

    if X.shape[1] == 0:
        raise ValueError("No features extracted")

    return X, y, col_to_fk


def train_ensemble(X: pd.DataFrame, y: np.ndarray, n_models: int = 10, seed: int = 42) -> List:
    """Train bootstrap ensemble."""
    models = []
    np.random.seed(seed)

    for i in range(n_models):
        idx = np.random.choice(len(X), size=int(len(X) * 0.8), replace=True)
        model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, verbose=-1,
                                   random_state=seed + i, force_col_wise=True)
        model.fit(X.iloc[idx], y[idx])
        models.append(model)

    return models


def compute_predictions(models: List, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Get ensemble mean and std."""
    preds = np.array([m.predict(X) for m in models])
    return preds.mean(axis=0), preds.std(axis=0)


def compute_importance_attribution(models: List, X: pd.DataFrame,
                                   col_to_fk: Dict[str, str]) -> Dict[str, float]:
    """Use feature importance instead of permutation for attribution."""
    # Aggregate feature importance by FK group
    fk_importance = defaultdict(float)

    for model in models:
        importances = model.feature_importances_
        for i, col in enumerate(X.columns):
            fk = col_to_fk.get(col, 'OTHER')
            fk_importance[fk] += importances[i]

    # Normalize
    total = sum(fk_importance.values())
    if total > 0:
        fk_importance = {k: v / total * 100 for k, v in fk_importance.items()}

    return dict(fk_importance)


def intervention_experiment(models: List, X: pd.DataFrame, y: np.ndarray,
                           col_to_fk: Dict[str, str], target_fk: str) -> Dict:
    """Measure effect of fixing an FK group."""
    fk_cols = [c for c, fk in col_to_fk.items() if fk == target_fk and c in X.columns]

    if not fk_cols:
        return {'error': f'No columns for {target_fk}'}

    pred_mean, pred_std = compute_predictions(models, X)
    base_mae = np.abs(pred_mean - y).mean()
    base_rmse = np.sqrt(((pred_mean - y) ** 2).mean())

    # Fix by replacing with mean
    X_fixed = X.copy()
    for col in fk_cols:
        X_fixed[col] = X[col].mean()

    pred_fixed, std_fixed = compute_predictions(models, X_fixed)
    fixed_mae = np.abs(pred_fixed - y).mean()
    fixed_rmse = np.sqrt(((pred_fixed - y) ** 2).mean())

    return {
        'target_fk': target_fk,
        'n_cols_fixed': len(fk_cols),
        'base_mae': float(base_mae),
        'fixed_mae': float(fixed_mae),
        'mae_change_pct': float((fixed_mae - base_mae) / base_mae * 100),
        'base_rmse': float(base_rmse),
        'fixed_rmse': float(fixed_rmse),
        'rmse_change_pct': float((fixed_rmse - base_rmse) / base_rmse * 100),
        'base_uncertainty': float(pred_std.mean()),
        'fixed_uncertainty': float(std_fixed.mean()),
    }


def sample_level_correlation(models: List, X: pd.DataFrame, y: np.ndarray) -> Dict:
    """Compute per-sample uncertainty-error correlation."""
    pred_mean, pred_std = compute_predictions(models, X)
    abs_errors = np.abs(pred_mean - y)

    # Correlation between uncertainty and error
    rho, pval = stats.spearmanr(pred_std, abs_errors)

    # Binned analysis
    n_bins = 5
    bin_edges = np.percentile(pred_std, np.linspace(0, 100, n_bins + 1))
    bin_means = []
    bin_errors = []

    for i in range(n_bins):
        mask = (pred_std >= bin_edges[i]) & (pred_std < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_means.append(pred_std[mask].mean())
            bin_errors.append(abs_errors[mask].mean())

    return {
        'uncertainty_error_rho': float(rho),
        'uncertainty_error_pval': float(pval),
        'bin_uncertainties': bin_means,
        'bin_errors': bin_errors,
    }


def run_experiment(dataset_name: str, task_name: str, n_models: int = 10,
                   sample_size: int = 5000, seed: int = 42) -> Dict:
    """Run full experiment on one task."""
    print(f"\n{'='*50}")
    print(f"{dataset_name} / {task_name}")
    print(f"{'='*50}")

    try:
        dataset = get_dataset(dataset_name, download=True)
        task = get_task(dataset_name, task_name, download=True)

        X, y, col_to_fk = extract_features(dataset, task)

        # Sample if needed
        if len(X) > sample_size:
            idx = np.random.choice(len(X), sample_size, replace=False)
            X = X.iloc[idx].reset_index(drop=True)
            y = y[idx]

        print(f"  Features: {X.shape[1]}, Samples: {len(X)}")
        print(f"  FK groups: {set(col_to_fk.values())}")

        # Train
        models = train_ensemble(X, y, n_models=n_models, seed=seed)

        # Importance-based attribution
        fk_attribution = compute_importance_attribution(models, X, col_to_fk)
        print(f"  Attribution: {fk_attribution}")

        # Intervention on each FK group
        interventions = {}
        for fk in set(col_to_fk.values()):
            interventions[fk] = intervention_experiment(models, X, y, col_to_fk, fk)

        # Sample-level correlation
        sample_corr = sample_level_correlation(models, X, y)
        print(f"  Sample uncertainty-error ρ: {sample_corr['uncertainty_error_rho']:.3f}")

        # Get domain classification
        domain = DOMAIN_CLASSIFICATIONS.get(dataset_name)

        return {
            'dataset': dataset_name,
            'task': task_name,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_fk_groups': len(set(col_to_fk.values())),
            'fk_groups': list(set(col_to_fk.values())),
            'fk_attribution': fk_attribution,
            'interventions': interventions,
            'sample_correlation': sample_corr,
            'is_cep': domain.is_cep if domain else None,
            'seed': seed,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        return {'dataset': dataset_name, 'task': task_name, 'error': str(e)}


def run_all(output_path: str = 'results/intervention_results.json'):
    """Run on all valid tasks."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_results = {
        'metadata': {
            'domains': {k: asdict(v) for k, v in DOMAIN_CLASSIFICATIONS.items()},
            'timestamp': datetime.now().isoformat(),
        },
        'results': []
    }

    for dataset, tasks in VALID_TASKS.items():
        for task in tasks:
            for seed in [42, 43, 44]:
                result = run_experiment(dataset, task, seed=seed)
                all_results['results'].append(result)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved to {output_path}")
    return all_results


def summarize_results(results: Dict):
    """Print summary."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    cep_corrs = []
    non_cep_corrs = []

    for r in results['results']:
        if 'error' in r:
            continue

        corr = r.get('sample_correlation', {}).get('uncertainty_error_rho', np.nan)
        if np.isnan(corr):
            continue

        if r.get('is_cep'):
            cep_corrs.append(corr)
        else:
            non_cep_corrs.append(corr)

    if cep_corrs:
        print(f"CEP domains: uncertainty-error ρ = {np.mean(cep_corrs):.3f} ± {np.std(cep_corrs):.3f} (n={len(cep_corrs)})")
    if non_cep_corrs:
        print(f"Non-CEP domains: ρ = {np.mean(non_cep_corrs):.3f} ± {np.std(non_cep_corrs):.3f} (n={len(non_cep_corrs)})")

    # Intervention effects
    print("\nIntervention Effects (top FK group):")
    for r in results['results']:
        if 'error' in r or 'interventions' not in r:
            continue

        # Find top FK by attribution
        attr = r.get('fk_attribution', {})
        if not attr:
            continue
        top_fk = max(attr.keys(), key=lambda k: attr[k])
        interv = r['interventions'].get(top_fk, {})

        if 'mae_change_pct' in interv:
            print(f"  {r['dataset']}/{r['task']} seed={r.get('seed')}: "
                  f"Top FK={top_fk} ({attr[top_fk]:.1f}%), "
                  f"MAE change={interv['mae_change_pct']:+.1f}%")


if __name__ == '__main__':
    results = run_all()
    summarize_results(results)
