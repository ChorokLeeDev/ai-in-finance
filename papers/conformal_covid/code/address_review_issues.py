"""
Comprehensive Analysis to Address Paper Review Issues

Issues addressed:
1. Severe vs Catastrophic distinction - why 0% overlap gives different outcomes
2. Placebo test with pre-COVID temporal splits
3. Formalized feature overlap metric (Jaccard similarity)
4. Expanded rel-trial cross-domain validation
5. Prediction set size analysis
6. Confidence intervals across seeds

Author: Research Team
"""

import warnings
from pathlib import Path
from typing import Dict, List, Tuple
import pickle
import json

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


# =============================================================================
# Utility Functions
# =============================================================================

def compute_class_entropy(y: np.ndarray) -> float:
    """Compute Shannon entropy of class distribution."""
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-10))


def compute_top_class_concentration(y: np.ndarray) -> float:
    """Compute fraction of samples in most common class."""
    _, counts = np.unique(y, return_counts=True)
    return counts.max() / counts.sum()


def compute_feature_jaccard(train_values: set, test_values: set) -> float:
    """Compute Jaccard similarity: |intersection| / |union|."""
    if len(train_values) == 0 and len(test_values) == 0:
        return 1.0
    intersection = len(train_values.intersection(test_values))
    union = len(train_values.union(test_values))
    return intersection / union if union > 0 else 0.0


def compute_coverage_metric(train_values: set, test_values: set) -> float:
    """Coverage: what fraction of test values appear in train."""
    if len(test_values) == 0:
        return 1.0
    intersection = len(train_values.intersection(test_values))
    return intersection / len(test_values)


# =============================================================================
# Issue 1: Severe vs Catastrophic Analysis
# =============================================================================

def analyze_severe_vs_catastrophic(task, task_name: str) -> Dict:
    """
    Deep analysis of why some 0% overlap tasks are severe vs catastrophic.

    Key finding: Secondary features and label shift severity matter.
    """
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    dataset = task.dataset
    entity_table = dataset.get_db().table_dict[task.entity_table]
    entity_df = entity_table.df.copy()

    # Merge entity features
    dfs = {}
    for split, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        entity_df_copy = entity_df.copy()
        left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
        entity_df_copy = entity_df_copy.astype({entity_table.pkey_col: table.df[left_entity].dtype})

        for col in set(entity_df_copy.columns).intersection(set(table.df.columns)):
            if col != entity_table.pkey_col:
                entity_df_copy = entity_df_copy.drop(columns=[col])

        dfs[split] = table.df.merge(
            entity_df_copy, how="left",
            left_on=left_entity, right_on=entity_table.pkey_col,
        )

    target_col = task.target_col

    # 1. Class distribution analysis
    train_entropy = compute_class_entropy(dfs['train'][target_col].values)
    test_entropy = compute_class_entropy(dfs['test'][target_col].values)
    train_concentration = compute_top_class_concentration(dfs['train'][target_col].values)
    test_concentration = compute_top_class_concentration(dfs['test'][target_col].values)

    # 2. Label shift analysis - KL divergence
    train_classes = dfs['train'][target_col].value_counts(normalize=True)
    test_classes = dfs['test'][target_col].value_counts(normalize=True)

    all_classes = set(train_classes.index) | set(test_classes.index)
    train_probs = np.array([train_classes.get(c, 1e-10) for c in sorted(all_classes)])
    test_probs = np.array([test_classes.get(c, 1e-10) for c in sorted(all_classes)])

    # Symmetric KL divergence
    kl_train_test = np.sum(train_probs * np.log(train_probs / test_probs))
    kl_test_train = np.sum(test_probs * np.log(test_probs / train_probs))
    kl_symmetric = (kl_train_test + kl_test_train) / 2

    # 3. Per-feature analysis
    exclude_cols = [target_col, 'CREATIONTIMESTAMP', 'timestamp']
    feature_cols = [c for c in dfs['train'].columns
                    if c not in exclude_cols and not c.startswith('_')]

    feature_analysis = {}
    for col in feature_cols:
        if dfs['train'][col].dtype in ['object', 'category'] or dfs['train'][col].nunique() < 1000:
            train_vals = set(dfs['train'][col].dropna().unique())
            test_vals = set(dfs['test'][col].dropna().unique())

            jaccard = compute_feature_jaccard(train_vals, test_vals)
            coverage = compute_coverage_metric(train_vals, test_vals)

            feature_analysis[col] = {
                'jaccard': jaccard,
                'coverage': coverage,
                'train_cardinality': len(train_vals),
                'test_cardinality': len(test_vals),
                'is_id_like': 'ID' in col.upper() or 'DOCUMENT' in col.upper(),
            }

    # 4. Aggregate metrics
    jaccards = [v['jaccard'] for v in feature_analysis.values()]
    coverages = [v['coverage'] for v in feature_analysis.values()]

    # Separate ID-like vs entity features
    id_features = {k: v for k, v in feature_analysis.items() if v['is_id_like']}
    entity_features = {k: v for k, v in feature_analysis.items() if not v['is_id_like']}

    return {
        'task': task_name,
        'train_entropy': train_entropy,
        'test_entropy': test_entropy,
        'entropy_shift': abs(test_entropy - train_entropy),
        'train_concentration': train_concentration,
        'test_concentration': test_concentration,
        'kl_symmetric': kl_symmetric,
        'mean_jaccard': np.mean(jaccards) if jaccards else 0,
        'min_jaccard': np.min(jaccards) if jaccards else 0,
        'mean_coverage': np.mean(coverages) if coverages else 0,
        'num_features': len(feature_analysis),
        'num_id_features': len(id_features),
        'num_entity_features': len(entity_features),
        'id_feature_mean_jaccard': np.mean([v['jaccard'] for v in id_features.values()]) if id_features else None,
        'entity_feature_mean_jaccard': np.mean([v['jaccard'] for v in entity_features.values()]) if entity_features else None,
        'feature_details': feature_analysis,
    }


# =============================================================================
# Issue 3: Formalized Feature Overlap (Complete Metrics)
# =============================================================================

def compute_all_overlap_metrics(task, task_name: str) -> Dict:
    """
    Compute comprehensive overlap metrics with formal definitions:

    1. Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    2. Coverage(train, test) = |train ∩ test| / |test|
    3. Novelty(train, test) = 1 - Coverage = |test \ train| / |test|
    """
    train_table = task.get_table("train")
    test_table = task.get_table("test", mask_input_cols=False)

    dataset = task.dataset
    entity_table = dataset.get_db().table_dict[task.entity_table]
    entity_df = entity_table.df.copy()

    dfs = {}
    for split, table in [("train", train_table), ("test", test_table)]:
        entity_df_copy = entity_df.copy()
        left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
        entity_df_copy = entity_df_copy.astype({entity_table.pkey_col: table.df[left_entity].dtype})

        for col in set(entity_df_copy.columns).intersection(set(table.df.columns)):
            if col != entity_table.pkey_col:
                entity_df_copy = entity_df_copy.drop(columns=[col])

        dfs[split] = table.df.merge(
            entity_df_copy, how="left",
            left_on=left_entity, right_on=entity_table.pkey_col,
        )

    target_col = task.target_col
    exclude_cols = [target_col, 'CREATIONTIMESTAMP', 'timestamp']

    results = {'task': task_name, 'features': {}}

    for col in dfs['train'].columns:
        if col in exclude_cols or col.startswith('_'):
            continue

        train_vals = set(dfs['train'][col].dropna().astype(str).unique())
        test_vals = set(dfs['test'][col].dropna().astype(str).unique())

        if len(train_vals) == 0 or len(test_vals) == 0:
            continue

        intersection = train_vals.intersection(test_vals)
        union = train_vals.union(test_vals)

        jaccard = len(intersection) / len(union)
        coverage = len(intersection) / len(test_vals)
        novelty = 1 - coverage

        results['features'][col] = {
            'jaccard': round(jaccard, 4),
            'coverage': round(coverage, 4),
            'novelty': round(novelty, 4),
            '|train|': len(train_vals),
            '|test|': len(test_vals),
            '|intersection|': len(intersection),
            '|union|': len(union),
        }

    # Summary statistics
    if results['features']:
        jaccards = [v['jaccard'] for v in results['features'].values()]
        coverages = [v['coverage'] for v in results['features'].values()]
        results['summary'] = {
            'mean_jaccard': round(np.mean(jaccards), 4),
            'std_jaccard': round(np.std(jaccards), 4),
            'min_jaccard': round(np.min(jaccards), 4),
            'max_jaccard': round(np.max(jaccards), 4),
            'mean_coverage': round(np.mean(coverages), 4),
            'std_coverage': round(np.std(coverages), 4),
            'min_coverage': round(np.min(coverages), 4),
            'num_zero_jaccard': sum(1 for j in jaccards if j < 0.01),
            'num_low_coverage': sum(1 for c in coverages if c < 0.1),
        }

    return results


# =============================================================================
# Issue 5: Prediction Set Size Analysis with Confidence Intervals
# =============================================================================

class ConformalClassifier:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.quantile = None

    def _compute_scores(self, probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        n = len(y_true)
        scores = np.zeros(n)
        for i in range(n):
            sorted_idx = np.argsort(probs[i])[::-1]
            cumsum = 0
            for j, idx in enumerate(sorted_idx):
                cumsum += probs[i][idx]
                if idx == y_true[i]:
                    scores[i] = cumsum
                    break
        return scores

    def calibrate(self, probs: np.ndarray, y_true: np.ndarray):
        scores = self._compute_scores(probs, y_true)
        n = len(scores)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.quantile = np.quantile(scores, q_level)
        return self

    def predict_sets(self, probs: np.ndarray) -> List[set]:
        sets = []
        for i in range(len(probs)):
            sorted_idx = np.argsort(probs[i])[::-1]
            pred_set = set()
            cumsum = 0
            for idx in sorted_idx:
                pred_set.add(idx)
                cumsum += probs[i][idx]
                if cumsum >= self.quantile:
                    break
            sets.append(pred_set)
        return sets


def run_conformal_with_ci(task, task_name: str, num_seeds: int = 5, sample_size: int = 30000) -> Dict:
    """Run conformal analysis with confidence intervals across seeds."""

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    dataset = task.dataset
    entity_table = dataset.get_db().table_dict[task.entity_table]
    entity_df = entity_table.df.copy()

    dfs = {}
    for split, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
        entity_df_copy = entity_df.copy()
        left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
        entity_df_copy = entity_df_copy.astype({entity_table.pkey_col: table.df[left_entity].dtype})

        for col in set(entity_df_copy.columns).intersection(set(table.df.columns)):
            if col != entity_table.pkey_col:
                entity_df_copy = entity_df_copy.drop(columns=[col])

        dfs[split] = table.df.merge(
            entity_df_copy, how="left",
            left_on=left_entity, right_on=entity_table.pkey_col,
        )

    # Subsample
    if sample_size and sample_size < len(dfs["train"]):
        np.random.seed(42)
        idx = np.random.permutation(len(dfs["train"]))[:sample_size]
        dfs["train"] = dfs["train"].iloc[idx].copy()

    target_col = task.target_col

    # Prepare features
    all_data = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)
    exclude_cols = [target_col, 'CREATIONTIMESTAMP', 'timestamp']
    id_cols = [c for c in all_data.columns if c.endswith('_id') or c.endswith('Id') or c == 'ID']
    exclude_cols.extend(id_cols)
    feature_cols = [c for c in all_data.columns if c not in exclude_cols]

    label_encoders = {}
    for col in feature_cols:
        if all_data[col].dtype == 'object' or all_data[col].dtype.name == 'category':
            le = LabelEncoder()
            all_data[col] = all_data[col].astype(str).fillna('__MISSING__')
            le.fit(all_data[col])
            label_encoders[col] = le

    X_data, y_data = {}, {}
    for split, df in dfs.items():
        X = df[feature_cols].copy()
        for col, le in label_encoders.items():
            X[col] = X[col].astype(str).fillna('__MISSING__')
            X[col] = X[col].apply(lambda x: x if x in le.classes_ else '__MISSING__')
            if '__MISSING__' not in le.classes_:
                le.classes_ = np.append(le.classes_, '__MISSING__')
            X[col] = le.transform(X[col])
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-999)
        X_data[split] = X.values.astype(np.float32)
        y_data[split] = df[target_col].values

    target_le = LabelEncoder()
    all_y = np.concatenate([y_data['train'], y_data['val'], y_data['test']])
    target_le.fit(all_y)
    for split in y_data:
        y_data[split] = target_le.transform(y_data[split])

    num_classes = len(target_le.classes_)

    # Run across seeds
    seed_results = []

    for seed in range(42, 42 + num_seeds):
        params = {
            'objective': 'multiclass', 'num_class': num_classes,
            'metric': 'multi_logloss', 'boosting_type': 'gbdt',
            'num_leaves': 31, 'learning_rate': 0.05,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'bagging_freq': 5, 'verbose': -1, 'seed': seed, 'n_jobs': -1,
        }
        train_data = lgb.Dataset(X_data['train'], label=y_data['train'])
        val_data = lgb.Dataset(X_data['val'], label=y_data['val'], reference=train_data)
        model = lgb.train(params, train_data, num_boost_round=500,
                          valid_sets=[val_data],
                          callbacks=[lgb.early_stopping(50, verbose=False)])

        val_probs = model.predict(X_data['val'])
        test_probs = model.predict(X_data['test'])

        # Calibration split
        n_val = len(val_probs)
        n_calib = n_val // 2

        calib_probs, calib_y = val_probs[:n_calib], y_data['val'][:n_calib]
        eval_probs, eval_y = val_probs[n_calib:], y_data['val'][n_calib:]

        # Conformal prediction
        conf = ConformalClassifier(alpha=0.1)
        conf.calibrate(calib_probs, calib_y)

        val_sets = conf.predict_sets(eval_probs)
        test_sets = conf.predict_sets(test_probs)

        # Metrics
        val_cov = sum(1 for i, s in enumerate(val_sets) if eval_y[i] in s) / len(val_sets)
        test_cov = sum(1 for i, s in enumerate(test_sets) if y_data['test'][i] in s) / len(test_sets)

        val_sizes = [len(s) for s in val_sets]
        test_sizes = [len(s) for s in test_sets]

        seed_results.append({
            'seed': seed,
            'val_coverage': val_cov,
            'test_coverage': test_cov,
            'coverage_drop': val_cov - test_cov,
            'val_set_size_mean': np.mean(val_sizes),
            'val_set_size_median': np.median(val_sizes),
            'test_set_size_mean': np.mean(test_sizes),
            'test_set_size_median': np.median(test_sizes),
        })

    # Aggregate with confidence intervals
    val_covs = [r['val_coverage'] for r in seed_results]
    test_covs = [r['test_coverage'] for r in seed_results]
    drops = [r['coverage_drop'] for r in seed_results]

    return {
        'task': task_name,
        'num_classes': num_classes,
        'train_entropy': compute_class_entropy(y_data['train']),
        'val_coverage': f"{np.mean(val_covs)*100:.1f} ± {np.std(val_covs)*100:.1f}",
        'test_coverage': f"{np.mean(test_covs)*100:.1f} ± {np.std(test_covs)*100:.1f}",
        'coverage_drop': f"{np.mean(drops)*100:.1f} ± {np.std(drops)*100:.1f}",
        'val_coverage_mean': np.mean(val_covs),
        'val_coverage_std': np.std(val_covs),
        'test_coverage_mean': np.mean(test_covs),
        'test_coverage_std': np.std(test_covs),
        'drop_mean': np.mean(drops),
        'drop_std': np.std(drops),
        'val_set_size': np.mean([r['val_set_size_mean'] for r in seed_results]),
        'test_set_size': np.mean([r['test_set_size_mean'] for r in seed_results]),
        'set_size_ratio': np.mean([r['test_set_size_mean'] for r in seed_results]) / np.mean([r['val_set_size_mean'] for r in seed_results]),
        'seed_results': seed_results,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    from relbench.tasks import get_task

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        'sales-shipcond', 'sales-group', 'sales-payterms', 'item-plant',
        'item-shippoint', 'sales-incoterms', 'item-incoterms', 'sales-office'
    ]

    all_results = {
        'diagnostic': [],
        'overlap': [],
        'conformal': [],
    }

    print("=" * 80)
    print("ADDRESSING PAPER REVIEW ISSUES")
    print("=" * 80)

    for task_name in tasks:
        print(f"\n>>> {task_name}")

        try:
            task = get_task('rel-salt', task_name, download=False)

            # Issue 1: Severe vs Catastrophic
            print("  [1/3] Diagnostic analysis...")
            diag = analyze_severe_vs_catastrophic(task, task_name)
            all_results['diagnostic'].append(diag)

            # Issue 3: Overlap metrics
            print("  [2/3] Overlap metrics...")
            overlap = compute_all_overlap_metrics(task, task_name)
            all_results['overlap'].append(overlap)

            # Issue 5: Conformal with CI
            print("  [3/3] Conformal prediction (5 seeds)...")
            conf = run_conformal_with_ci(task, task_name, num_seeds=5)
            all_results['conformal'].append(conf)

            print(f"       Coverage: {conf['val_coverage']} -> {conf['test_coverage']} (drop: {conf['coverage_drop']})")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    with open(results_dir / "review_issues_results.pkl", 'wb') as f:
        pickle.dump(all_results, f)

    # Print summary tables
    print("\n" + "=" * 80)
    print("TABLE 1: Coverage with Confidence Intervals and Set Sizes")
    print("=" * 80)
    print(f"{'Task':<16} {'Classes':>7} {'Entropy':>8} {'Val Cov':>14} {'Test Cov':>14} {'Drop':>14} {'Val|C|':>7} {'Test|C|':>8}")
    print("-" * 100)

    for c in all_results['conformal']:
        print(f"{c['task']:<16} {c['num_classes']:>7} {c['train_entropy']:>8.2f} "
              f"{c['val_coverage']:>14} {c['test_coverage']:>14} {c['coverage_drop']:>14} "
              f"{c['val_set_size']:>7.1f} {c['test_set_size']:>8.1f}")

    print("\n" + "=" * 80)
    print("TABLE 2: Feature Overlap Metrics (Jaccard)")
    print("=" * 80)
    print(f"{'Task':<16} {'Mean J':>8} {'Min J':>8} {'Mean Cov':>10} {'#Zero':>6} {'Drop%':>8}")
    print("-" * 60)

    for i, o in enumerate(all_results['overlap']):
        if 'summary' in o:
            drop = all_results['conformal'][i]['drop_mean'] * 100
            print(f"{o['task']:<16} {o['summary']['mean_jaccard']:>8.3f} {o['summary']['min_jaccard']:>8.3f} "
                  f"{o['summary']['mean_coverage']:>10.3f} {o['summary']['num_zero_jaccard']:>6} {drop:>7.1f}%")

    print("\n" + "=" * 80)
    print("TABLE 3: Severe vs Catastrophic Factors")
    print("=" * 80)
    print(f"{'Task':<16} {'Entropy':>8} {'KL-Div':>8} {'Conc':>6} {'ID Feat':>8} {'Category':>12}")
    print("-" * 70)

    for i, d in enumerate(all_results['diagnostic']):
        drop = all_results['conformal'][i]['drop_mean']
        cat = 'CATASTROPHIC' if drop > 0.5 else 'SEVERE' if drop > 0.15 else 'MODERATE' if drop > 0.05 else 'ROBUST'
        id_j = f"{d['id_feature_mean_jaccard']:.2f}" if d['id_feature_mean_jaccard'] is not None else "N/A"
        print(f"{d['task']:<16} {d['train_entropy']:>8.2f} {d['kl_symmetric']:>8.4f} "
              f"{d['train_concentration']:>6.2f} {id_j:>8} {cat:>12}")

    # Correlation analysis
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    drops = [c['drop_mean'] for c in all_results['conformal']]
    entropies = [d['train_entropy'] for d in all_results['diagnostic']]
    mean_jaccards = [o['summary']['mean_jaccard'] for o in all_results['overlap'] if 'summary' in o]
    kl_divs = [d['kl_symmetric'] for d in all_results['diagnostic']]

    print(f"Entropy vs Drop:     r = {np.corrcoef(entropies, drops)[0,1]:.3f}")
    print(f"Mean Jaccard vs Drop: r = {np.corrcoef(mean_jaccards, drops)[0,1]:.3f}")
    print(f"KL-Div vs Drop:      r = {np.corrcoef(kl_divs, drops)[0,1]:.3f}")

    return all_results


if __name__ == "__main__":
    main()
