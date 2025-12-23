"""
Run Conformal Prediction on ALL rel-salt tasks and compare COVID distribution shift effects.
"""

import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


class ConformalClassifier:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.quantile = None

    def _compute_scores(self, probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        n_samples = len(y_true)
        scores = np.zeros(n_samples)
        for i in range(n_samples):
            sorted_indices = np.argsort(probs[i])[::-1]
            sorted_probs = probs[i][sorted_indices]
            true_class = y_true[i]
            cumsum = 0
            for j, idx in enumerate(sorted_indices):
                cumsum += sorted_probs[j]
                if idx == true_class:
                    scores[i] = cumsum
                    break
        return scores

    def calibrate(self, probs: np.ndarray, y_true: np.ndarray):
        scores = self._compute_scores(probs, y_true)
        n = len(scores)
        quantile_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.quantile = np.quantile(scores, quantile_level)
        return self

    def predict_sets(self, probs: np.ndarray):
        prediction_sets = []
        for i in range(len(probs)):
            sorted_indices = np.argsort(probs[i])[::-1]
            sorted_probs = probs[i][sorted_indices]
            pred_set = set()
            cumsum = 0
            for j, idx in enumerate(sorted_indices):
                pred_set.add(idx)
                cumsum += sorted_probs[j]
                if cumsum >= self.quantile:
                    break
            prediction_sets.append(pred_set)
        return prediction_sets


def compute_coverage(prediction_sets, y_true):
    covered = sum(1 for i, ps in enumerate(prediction_sets) if y_true[i] in ps)
    return covered / len(y_true)


def compute_avg_set_size(prediction_sets):
    return np.mean([len(ps) for ps in prediction_sets])


def prepare_and_train(task, sample_size=30000, num_seeds=3, base_seed=42):
    """Prepare data and train ensemble for a task."""
    from relbench.tasks import get_task as rt_get_task

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

    # Subsample training
    if sample_size and sample_size < len(dfs["train"]):
        np.random.seed(base_seed)
        idx = np.random.permutation(len(dfs["train"]))[:sample_size]
        dfs["train"] = dfs["train"].iloc[idx].copy()

    target_col = task.target_col

    # Identify feature columns
    all_data = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)
    exclude_cols = [target_col, 'CREATIONTIMESTAMP', 'timestamp']
    id_cols = [c for c in all_data.columns if c.endswith('_id') or c.endswith('Id') or c == 'ID']
    exclude_cols.extend(id_cols)
    feature_cols = [c for c in all_data.columns if c not in exclude_cols]

    # Encode categoricals
    label_encoders = {}
    for col in feature_cols:
        if all_data[col].dtype == 'object' or all_data[col].dtype.name == 'category':
            le = LabelEncoder()
            all_data[col] = all_data[col].astype(str).fillna('__MISSING__')
            le.fit(all_data[col])
            label_encoders[col] = le

    # Process each split
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

    # Encode targets
    target_le = LabelEncoder()
    all_y = np.concatenate([y_data['train'], y_data['val'], y_data['test']])
    target_le.fit(all_y)
    for split in y_data:
        y_data[split] = target_le.transform(y_data[split])

    num_classes = len(target_le.classes_)

    # Train ensemble
    all_val_probs, all_test_probs = [], []
    for seed in range(base_seed, base_seed + num_seeds):
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
        all_val_probs.append(model.predict(X_data['val']))
        all_test_probs.append(model.predict(X_data['test']))

    val_probs = np.mean(all_val_probs, axis=0)
    test_probs = np.mean(all_test_probs, axis=0)

    return val_probs, test_probs, y_data['val'], y_data['test'], num_classes


def analyze_task(task_name, alpha=0.1, sample_size=30000, num_seeds=3):
    """Analyze a single task and return metrics."""
    from relbench.tasks import get_task

    print(f"\n{'='*50}")
    print(f"Analyzing: {task_name}")
    print('='*50)

    task = get_task('rel-salt', task_name, download=False)

    val_probs, test_probs, y_val, y_test, num_classes = prepare_and_train(
        task, sample_size=sample_size, num_seeds=num_seeds
    )

    # Split validation for calibration
    n_val = len(val_probs)
    n_calib = n_val // 2

    calib_probs, calib_labels = val_probs[:n_calib], y_val[:n_calib]
    eval_probs, eval_labels = val_probs[n_calib:], y_val[n_calib:]

    # Calibrate and predict
    conformal = ConformalClassifier(alpha=alpha)
    conformal.calibrate(calib_probs, calib_labels)

    val_sets = conformal.predict_sets(eval_probs)
    test_sets = conformal.predict_sets(test_probs)

    val_coverage = compute_coverage(val_sets, eval_labels)
    test_coverage = compute_coverage(test_sets, y_test)
    val_set_size = compute_avg_set_size(val_sets)
    test_set_size = compute_avg_set_size(test_sets)

    coverage_drop = val_coverage - test_coverage

    print(f"  Classes: {num_classes}")
    print(f"  Val coverage: {val_coverage*100:.1f}%")
    print(f"  Test coverage: {test_coverage*100:.1f}%")
    print(f"  Coverage drop: {coverage_drop*100:.1f}%")

    return {
        'task': task_name,
        'num_classes': num_classes,
        'val_coverage': val_coverage,
        'test_coverage': test_coverage,
        'coverage_drop': coverage_drop,
        'val_set_size': val_set_size,
        'test_set_size': test_set_size,
    }


def plot_comparison(results, save_path):
    """Plot comparison across all tasks."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    tasks = [r['task'] for r in results]
    val_coverages = [r['val_coverage']*100 for r in results]
    test_coverages = [r['test_coverage']*100 for r in results]
    coverage_drops = [r['coverage_drop']*100 for r in results]
    val_sizes = [r['val_set_size'] for r in results]
    test_sizes = [r['test_set_size'] for r in results]

    x = np.arange(len(tasks))
    width = 0.35

    # 1. Coverage comparison
    ax = axes[0, 0]
    bars1 = ax.bar(x - width/2, val_coverages, width, label='Validation (COVID onset)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_coverages, width, label='Test (COVID peak)', color='coral', alpha=0.8)
    ax.axhline(90, color='black', linestyle='--', lw=2, label='Target (90%)')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Coverage by Task')
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('-', '\n') for t in tasks], fontsize=8)
    ax.legend(loc='lower left')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Coverage drop
    ax = axes[0, 1]
    colors = ['red' if d > 10 else 'orange' if d > 5 else 'green' for d in coverage_drops]
    bars = ax.bar(tasks, coverage_drops, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(0, color='black', lw=1)
    ax.axhline(5, color='orange', linestyle='--', alpha=0.5, label='Moderate (5%)')
    ax.axhline(10, color='red', linestyle='--', alpha=0.5, label='Severe (10%)')
    ax.set_ylabel('Coverage Drop (%)')
    ax.set_title('Coverage Drop (Val → Test)')
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('-', '\n') for t in tasks], fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bar, drop in zip(bars, coverage_drops):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{drop:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 3. Set size comparison
    ax = axes[1, 0]
    bars1 = ax.bar(x - width/2, val_sizes, width, label='Validation', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_sizes, width, label='Test', color='coral', alpha=0.8)
    ax.set_ylabel('Avg Set Size')
    ax.set_title('Prediction Set Size by Task')
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('-', '\n') for t in tasks], fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')

    # Create summary table
    table_data = []
    for r in results:
        status = '🔴' if r['coverage_drop'] > 0.10 else '🟡' if r['coverage_drop'] > 0.05 else '🟢'
        table_data.append([
            r['task'],
            f"{r['val_coverage']*100:.1f}%",
            f"{r['test_coverage']*100:.1f}%",
            f"{r['coverage_drop']*100:.1f}%",
            status
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Task', 'Val Cov', 'Test Cov', 'Drop', 'Status'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title('Summary Table\n🟢 <5%  🟡 5-10%  🔴 >10%', fontsize=10)

    plt.suptitle('Conformal Prediction: rel-salt COVID Distribution Shift\nAll Tasks Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison figure: {save_path}")
    plt.close()


def main():
    tasks = [
        'item-plant', 'item-shippoint', 'item-incoterms',
        'sales-office', 'sales-group', 'sales-payterms',
        'sales-shipcond', 'sales-incoterms'
    ]

    results = []
    for task_name in tasks:
        try:
            result = analyze_task(task_name, alpha=0.1, sample_size=30000, num_seeds=3)
            results.append(result)
        except Exception as e:
            print(f"Error on {task_name}: {e}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: COVID Distribution Shift Impact")
    print('='*70)
    print(f"\n{'Task':<20} {'Val Cov':>10} {'Test Cov':>10} {'Drop':>10} {'Status':>10}")
    print('-'*60)

    for r in results:
        status = 'SEVERE' if r['coverage_drop'] > 0.10 else 'MODERATE' if r['coverage_drop'] > 0.05 else 'OK'
        print(f"{r['task']:<20} {r['val_coverage']*100:>9.1f}% {r['test_coverage']*100:>9.1f}% {r['coverage_drop']*100:>9.1f}% {status:>10}")

    avg_drop = np.mean([r['coverage_drop'] for r in results])
    print('-'*60)
    print(f"{'AVERAGE':<20} {'':<10} {'':<10} {avg_drop*100:>9.1f}%")

    # Save results
    results_dir = Path("results/conformal/rel-salt")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "all_tasks_results.pkl", 'wb') as f:
        pickle.dump(results, f)

    plot_comparison(results, results_dir / "all_tasks_comparison.png")

    return results


if __name__ == "__main__":
    main()
