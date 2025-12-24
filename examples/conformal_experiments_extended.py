"""
Extended Conformal Prediction Experiments

Additional experiments to strengthen the paper:
1. Adaptive Conformal Inference (ACI) - does online adaptation help?
2. Feature Ablation - does removing ID features improve robustness?
3. rel-trial dataset - does COVID affect clinical trial predictions?
4. Coverage-Overlap Correlation - theoretical relationship
"""

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


# =============================================================================
# Conformal Predictors
# =============================================================================

class StandardConformal:
    """Standard split conformal prediction (baseline)."""

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
            for j, idx in enumerate(sorted_idx):
                pred_set.add(idx)
                cumsum += probs[i][idx]
                if cumsum >= self.quantile:
                    break
            sets.append(pred_set)
        return sets


class AdaptiveConformal:
    """
    Adaptive Conformal Inference (ACI) - Gibbs & Candes (2021)

    Key idea: Update quantile online based on observed coverage.
    If coverage is too low, increase quantile (larger sets).
    If coverage is too high, decrease quantile (smaller sets).
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.01):
        """
        Args:
            alpha: Target miscoverage rate
            gamma: Learning rate for quantile adaptation
        """
        self.alpha = alpha
        self.gamma = gamma
        self.quantile = None
        self.quantile_history = []

    def _compute_score(self, probs: np.ndarray, y_true: int) -> float:
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = 0
        for idx in sorted_idx:
            cumsum += probs[idx]
            if idx == y_true:
                return cumsum
        return 1.0

    def calibrate(self, probs: np.ndarray, y_true: np.ndarray):
        """Initial calibration on calibration set."""
        scores = np.array([self._compute_score(probs[i], y_true[i])
                          for i in range(len(y_true))])
        n = len(scores)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.quantile = np.quantile(scores, q_level)
        self.quantile_history = [self.quantile]
        return self

    def predict_and_update(self, probs: np.ndarray, y_true: np.ndarray) -> Tuple[List[set], List[float]]:
        """
        Make predictions and update quantile online.

        Returns:
            prediction_sets: List of prediction sets
            coverages: Running coverage after each prediction
        """
        sets = []
        coverages = []
        covered_count = 0

        for i in range(len(probs)):
            # Make prediction with current quantile
            sorted_idx = np.argsort(probs[i])[::-1]
            pred_set = set()
            cumsum = 0
            for idx in sorted_idx:
                pred_set.add(idx)
                cumsum += probs[i][idx]
                if cumsum >= self.quantile:
                    break
            sets.append(pred_set)

            # Check coverage
            covered = y_true[i] in pred_set
            covered_count += covered
            coverages.append(covered_count / (i + 1))

            # Update quantile (ACI update rule)
            # If we missed, increase quantile; if we covered, decrease slightly
            err = 1 - int(covered)  # 1 if missed, 0 if covered
            self.quantile = self.quantile + self.gamma * (self.alpha - err)
            self.quantile = np.clip(self.quantile, 0.01, 0.99)
            self.quantile_history.append(self.quantile)

        return sets, coverages


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_data_with_features(task, sample_size=30000, seed=42, exclude_id_features=False):
    """Prepare data with option to exclude ID-based features."""
    from relbench.tasks import get_task

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

        dfs[split] = table.df.merge(entity_df_copy, how="left",
            left_on=left_entity, right_on=entity_table.pkey_col)

    # Subsample training
    if sample_size and sample_size < len(dfs["train"]):
        np.random.seed(seed)
        idx = np.random.permutation(len(dfs["train"]))[:sample_size]
        dfs["train"] = dfs["train"].iloc[idx].copy()

    target_col = task.target_col

    # Identify feature columns
    all_data = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)
    exclude_cols = [target_col, 'CREATIONTIMESTAMP', 'timestamp']

    # Optionally exclude ID-based features
    if exclude_id_features:
        id_patterns = ['DOCUMENT', 'ID', '_id', 'Id']
        for col in all_data.columns:
            if any(p in col for p in id_patterns):
                exclude_cols.append(col)

    feature_cols = [c for c in all_data.columns if c not in exclude_cols and not c.startswith('_')]

    # Encode categoricals
    label_encoders = {}
    for col in feature_cols:
        if all_data[col].dtype == 'object' or all_data[col].dtype.name == 'category':
            le = LabelEncoder()
            all_data[col] = all_data[col].astype(str).fillna('__MISSING__')
            le.fit(all_data[col])
            label_encoders[col] = le

    X_data, y_data = {}, {}
    for split, df in dfs.items():
        X = df[feature_cols].copy() if feature_cols else pd.DataFrame(index=df.index)
        for col, le in label_encoders.items():
            if col in X.columns:
                X[col] = X[col].astype(str).fillna('__MISSING__')
                X[col] = X[col].apply(lambda x: x if x in le.classes_ else '__MISSING__')
                if '__MISSING__' not in le.classes_:
                    le.classes_ = np.append(le.classes_, '__MISSING__')
                X[col] = le.transform(X[col])
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-999)
        X_data[split] = X.values.astype(np.float32) if len(X.columns) > 0 else np.zeros((len(df), 1), dtype=np.float32)
        y_data[split] = df[target_col].values

    target_le = LabelEncoder()
    all_y = np.concatenate([y_data['train'], y_data['val'], y_data['test']])
    target_le.fit(all_y)
    for split in y_data:
        y_data[split] = target_le.transform(y_data[split])

    num_classes = len(target_le.classes_)

    return X_data, y_data, num_classes, feature_cols


def train_ensemble(X_train, y_train, X_val, y_val, num_classes, num_seeds=3, base_seed=42):
    """Train LightGBM ensemble."""
    all_val_probs, all_test_probs = [], []

    for seed in range(base_seed, base_seed + num_seeds):
        params = {
            'objective': 'multiclass', 'num_class': num_classes,
            'metric': 'multi_logloss', 'boosting_type': 'gbdt',
            'num_leaves': 31, 'learning_rate': 0.05,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8,
            'bagging_freq': 5, 'verbose': -1, 'seed': seed, 'n_jobs': -1,
        }
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model = lgb.train(params, train_data, num_boost_round=500,
                          valid_sets=[val_data],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        all_val_probs.append(model.predict(X_val))

    return np.mean(all_val_probs, axis=0)


# =============================================================================
# Experiment 1: Adaptive Conformal vs Standard
# =============================================================================

def experiment_adaptive_conformal(dataset='rel-salt', task_name='sales-shipcond'):
    """Compare standard vs adaptive conformal prediction."""
    from relbench.tasks import get_task

    print(f"\n{'='*70}")
    print(f"Experiment 1: Adaptive Conformal Inference")
    print(f"Dataset: {dataset}, Task: {task_name}")
    print('='*70)

    task = get_task(dataset, task_name, download=False)
    X_data, y_data, num_classes, _ = prepare_data_with_features(task, sample_size=30000)

    # Train model
    print("Training ensemble...")
    val_probs = train_ensemble(X_data['train'], y_data['train'],
                               X_data['val'], y_data['val'], num_classes)

    # Get test predictions
    params = {
        'objective': 'multiclass', 'num_class': num_classes,
        'metric': 'multi_logloss', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'learning_rate': 0.05, 'verbose': -1, 'seed': 42,
    }
    train_data = lgb.Dataset(X_data['train'], label=y_data['train'])
    val_data = lgb.Dataset(X_data['val'], label=y_data['val'])
    model = lgb.train(params, train_data, num_boost_round=200,
                      valid_sets=[val_data], callbacks=[lgb.early_stopping(50, verbose=False)])
    test_probs = model.predict(X_data['test'])

    # Split validation for calibration
    n_val = len(val_probs)
    n_calib = n_val // 2
    calib_probs, calib_y = val_probs[:n_calib], y_data['val'][:n_calib]

    # Standard Conformal
    print("\nStandard Conformal:")
    std_conf = StandardConformal(alpha=0.1)
    std_conf.calibrate(calib_probs, calib_y)
    std_sets = std_conf.predict_sets(test_probs)
    std_coverage = sum(y_data['test'][i] in std_sets[i] for i in range(len(std_sets))) / len(std_sets)
    print(f"  Test Coverage: {std_coverage*100:.1f}%")

    # Adaptive Conformal with different learning rates
    print("\nAdaptive Conformal:")
    results = {'standard': std_coverage}

    for gamma in [0.001, 0.005, 0.01, 0.05]:
        aci = AdaptiveConformal(alpha=0.1, gamma=gamma)
        aci.calibrate(calib_probs, calib_y)
        aci_sets, aci_coverages = aci.predict_and_update(test_probs, y_data['test'])
        final_coverage = aci_coverages[-1] if aci_coverages else 0
        print(f"  γ={gamma}: Coverage {final_coverage*100:.1f}%")
        results[f'aci_gamma_{gamma}'] = final_coverage

    return results


# =============================================================================
# Experiment 2: Feature Ablation
# =============================================================================

def experiment_feature_ablation(dataset='rel-salt', task_name='sales-shipcond'):
    """Test if removing ID features improves robustness."""
    from relbench.tasks import get_task

    print(f"\n{'='*70}")
    print(f"Experiment 2: Feature Ablation (Remove ID Features)")
    print(f"Dataset: {dataset}, Task: {task_name}")
    print('='*70)

    task = get_task(dataset, task_name, download=False)

    results = {}

    for exclude_ids, label in [(False, 'With IDs'), (True, 'Without IDs')]:
        print(f"\n{label}:")
        X_data, y_data, num_classes, features = prepare_data_with_features(
            task, sample_size=30000, exclude_id_features=exclude_ids
        )
        print(f"  Features: {features[:5]}..." if len(features) > 5 else f"  Features: {features}")
        print(f"  Num features: {X_data['train'].shape[1]}")

        if X_data['train'].shape[1] == 0:
            print("  No features left! Skipping...")
            results[label] = {'val_coverage': 0, 'test_coverage': 0}
            continue

        # Train and evaluate
        val_probs = train_ensemble(X_data['train'], y_data['train'],
                                   X_data['val'], y_data['val'], num_classes)

        # Test predictions
        params = {
            'objective': 'multiclass', 'num_class': num_classes,
            'metric': 'multi_logloss', 'verbose': -1, 'seed': 42,
        }
        model = lgb.train(params, lgb.Dataset(X_data['train'], label=y_data['train']),
                          num_boost_round=200,
                          valid_sets=[lgb.Dataset(X_data['val'], label=y_data['val'])],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        test_probs = model.predict(X_data['test'])

        # Conformal prediction
        n_calib = len(val_probs) // 2
        conf = StandardConformal(alpha=0.1)
        conf.calibrate(val_probs[:n_calib], y_data['val'][:n_calib])

        val_sets = conf.predict_sets(val_probs[n_calib:])
        test_sets = conf.predict_sets(test_probs)

        val_cov = sum(y_data['val'][n_calib+i] in val_sets[i] for i in range(len(val_sets))) / len(val_sets)
        test_cov = sum(y_data['test'][i] in test_sets[i] for i in range(len(test_sets))) / len(test_sets)

        print(f"  Val Coverage: {val_cov*100:.1f}%")
        print(f"  Test Coverage: {test_cov*100:.1f}%")
        print(f"  Drop: {(val_cov - test_cov)*100:.1f}%")

        results[label] = {'val_coverage': val_cov, 'test_coverage': test_cov}

    return results


# =============================================================================
# Experiment 3: rel-trial Dataset
# =============================================================================

def experiment_rel_trial():
    """Test COVID impact on rel-trial (clinical trials) dataset."""
    from relbench.tasks import get_task_names, get_task

    print(f"\n{'='*70}")
    print(f"Experiment 3: rel-trial Dataset (Clinical Trials)")
    print('='*70)

    try:
        tasks = get_task_names('rel-trial')
        print(f"Available tasks: {tasks}")
    except Exception as e:
        print(f"Error getting tasks: {e}")
        return {}

    results = []
    for task_name in tasks[:3]:  # Test first 3 tasks
        try:
            print(f"\nAnalyzing: {task_name}")
            task = get_task('rel-trial', task_name, download=False)

            # Check task type
            print(f"  Task type: {task.task_type}")

            X_data, y_data, num_classes, _ = prepare_data_with_features(task, sample_size=20000)
            print(f"  Classes: {num_classes}")

            val_probs = train_ensemble(X_data['train'], y_data['train'],
                                       X_data['val'], y_data['val'], num_classes, num_seeds=2)

            params = {'objective': 'multiclass', 'num_class': num_classes, 'verbose': -1}
            model = lgb.train(params, lgb.Dataset(X_data['train'], label=y_data['train']),
                              num_boost_round=100)
            test_probs = model.predict(X_data['test'])

            n_calib = len(val_probs) // 2
            conf = StandardConformal(alpha=0.1)
            conf.calibrate(val_probs[:n_calib], y_data['val'][:n_calib])

            val_sets = conf.predict_sets(val_probs[n_calib:])
            test_sets = conf.predict_sets(test_probs)

            val_cov = sum(y_data['val'][n_calib+i] in val_sets[i] for i in range(len(val_sets))) / len(val_sets)
            test_cov = sum(y_data['test'][i] in test_sets[i] for i in range(len(test_sets))) / len(test_sets)

            print(f"  Val Coverage: {val_cov*100:.1f}%")
            print(f"  Test Coverage: {test_cov*100:.1f}%")
            print(f"  Drop: {(val_cov - test_cov)*100:.1f}%")

            results.append({
                'task': task_name,
                'val_coverage': val_cov,
                'test_coverage': test_cov,
                'drop': val_cov - test_cov
            })
        except Exception as e:
            print(f"  Error: {e}")

    return results


# =============================================================================
# Experiment 4: Coverage-Overlap Correlation
# =============================================================================

def experiment_coverage_overlap_correlation():
    """Analyze correlation between feature overlap and coverage drop."""
    from relbench.tasks import get_task

    print(f"\n{'='*70}")
    print(f"Experiment 4: Feature Overlap vs Coverage Drop Correlation")
    print('='*70)

    # Data from our main experiments
    data = [
        {'task': 'sales-shipcond', 'overlap': 0.0, 'drop': 93.1},
        {'task': 'sales-group', 'overlap': 0.0, 'drop': 86.7},
        {'task': 'sales-payterms', 'overlap': 0.0, 'drop': 33.8},
        {'task': 'item-plant', 'overlap': 0.0, 'drop': 29.1},
        {'task': 'item-shippoint', 'overlap': 0.0, 'drop': 18.9},
        {'task': 'sales-incoterms', 'overlap': 50.0, 'drop': 3.6},
        {'task': 'item-incoterms', 'overlap': 60.0, 'drop': 0.5},
        {'task': 'sales-office', 'overlap': 50.0, 'drop': 0.1},
    ]

    overlaps = [d['overlap'] for d in data]
    drops = [d['drop'] for d in data]

    # Correlation
    corr = np.corrcoef(overlaps, drops)[0, 1]
    print(f"\nCorrelation (Feature Overlap vs Coverage Drop): {corr:.3f}")

    # For tasks with 0% overlap, check entropy
    print("\nFor 0% overlap tasks, entropy determines severity:")
    entropy_data = [
        ('sales-group', 7.61, 86.7),
        ('sales-shipcond', 3.16, 93.1),
        ('sales-payterms', 3.5, 33.8),
        ('item-plant', 2.8, 29.1),
        ('item-shippoint', 3.0, 18.9),
    ]

    entropies = [e[1] for e in entropy_data]
    drops_0overlap = [e[2] for e in entropy_data]
    corr_entropy = np.corrcoef(entropies, drops_0overlap)[0, 1]
    print(f"Correlation (Entropy vs Drop, 0% overlap only): {corr_entropy:.3f}")

    return {'overlap_corr': corr, 'entropy_corr_0overlap': corr_entropy}


# =============================================================================
# Main
# =============================================================================

def main():
    results = {}

    # Experiment 1: Adaptive Conformal
    print("\n" + "="*70)
    print("RUNNING ALL EXTENDED EXPERIMENTS")
    print("="*70)

    results['adaptive'] = experiment_adaptive_conformal('rel-salt', 'sales-shipcond')

    # Experiment 2: Feature Ablation
    results['ablation'] = experiment_feature_ablation('rel-salt', 'sales-shipcond')

    # Experiment 3: rel-trial
    results['rel_trial'] = experiment_rel_trial()

    # Experiment 4: Correlation Analysis
    results['correlation'] = experiment_coverage_overlap_correlation()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF EXTENDED EXPERIMENTS")
    print("="*70)

    print("\n1. Adaptive Conformal (ACI):")
    for k, v in results['adaptive'].items():
        print(f"   {k}: {v*100:.1f}%")

    print("\n2. Feature Ablation:")
    for k, v in results['ablation'].items():
        print(f"   {k}: Val={v['val_coverage']*100:.1f}%, Test={v['test_coverage']*100:.1f}%")

    print("\n3. rel-trial Results:")
    if results['rel_trial']:
        for r in results['rel_trial']:
            print(f"   {r['task']}: Drop={r['drop']*100:.1f}%")

    print("\n4. Correlation Analysis:")
    print(f"   Overlap-Drop Correlation: {results['correlation']['overlap_corr']:.3f}")
    print(f"   Entropy-Drop Correlation (0% overlap): {results['correlation']['entropy_corr_0overlap']:.3f}")

    # Save results
    results_dir = Path("results/conformal/extended")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "extended_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {results_dir / 'extended_results.pkl'}")

    return results


if __name__ == "__main__":
    main()
