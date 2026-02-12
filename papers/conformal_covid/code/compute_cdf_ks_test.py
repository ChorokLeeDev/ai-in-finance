"""
Compute APS conformity score CDFs and KS tests for stochastic dominance analysis.

For each of 8 SALT classification tasks:
- Train LightGBM (seed=42)
- Compute APS conformity scores for calibration and test sets
- Run 2-sample KS tests (one-sided and two-sided)
- Generate 2-panel CDF figure comparing catastrophic vs robust task

Output:
    results/ks_stochastic_dominance.json - KS test results per task
    uai_2026/figures/conformity_score_cdfs.pdf - 2-panel CDF figure

Usage:
    PYTHONPATH=/Users/i767700/Github/ai-in-finance:$PYTHONPATH \
    /usr/bin/python3 -u papers/conformal_covid/code/compute_cdf_ks_test.py
"""

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Configuration
SEED = 42
ALPHA = 0.1
SAMPLE_SIZE = 30000

ALL_TASKS = [
    'sales-shipcond',
    'sales-group',
    'sales-payterms',
    'item-plant',
    'item-shippoint',
    'sales-incoterms',
    'item-incoterms',
    'sales-office',
]

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
FIGURES_DIR = SCRIPT_DIR.parent / "uai_2026" / "figures"
OUTPUT_JSON = RESULTS_DIR / "ks_stochastic_dominance.json"


def compute_aps_scores(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Compute APS conformity scores for each instance (vectorized).

    For each (x, y_true):
        Sort classes by decreasing predicted probability.
        Accumulate probability mass until y_true is included.
        Score = cumulative probability at that point.

    Higher scores mean the true label was harder to reach (less confident).
    """
    n, k = probs.shape
    # Sort probabilities in descending order and get sorted indices
    sorted_idx = np.argsort(-probs, axis=1)  # (n, k)
    # Gather sorted probabilities
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)  # (n, k)
    # Cumulative sum along classes
    cumsum = np.cumsum(sorted_probs, axis=1)  # (n, k)
    # Find which position in the sorted order the true label occupies
    # For each row i, find j such that sorted_idx[i, j] == y_true[i]
    # Create a mask: sorted_idx == y_true[:, None]
    true_mask = (sorted_idx == y_true[:, None])  # (n, k) boolean
    # The score is cumsum at the position where the true label appears
    scores = np.sum(cumsum * true_mask, axis=1)  # (n,)
    return scores


def prepare_data_and_train(task_name: str):
    """
    Load data, train LightGBM, compute conformity scores for calibration and test.

    Returns:
        calib_scores, test_scores, num_classes
    """
    from relbench.tasks import get_task

    print(f"  Loading task: {task_name}")
    task = get_task('rel-salt', task_name, download=False)

    # Load tables
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
        entity_df_copy = entity_df_copy.astype(
            {entity_table.pkey_col: table.df[left_entity].dtype}
        )

        for col in set(entity_df_copy.columns).intersection(set(table.df.columns)):
            if col != entity_table.pkey_col:
                entity_df_copy = entity_df_copy.drop(columns=[col])

        dfs[split] = table.df.merge(
            entity_df_copy,
            how="left",
            left_on=left_entity,
            right_on=entity_table.pkey_col,
        )

    # Subsample training data
    np.random.seed(SEED)
    if SAMPLE_SIZE and SAMPLE_SIZE < len(dfs["train"]):
        idx = np.random.permutation(len(dfs["train"]))[:SAMPLE_SIZE]
        dfs["train"] = dfs["train"].iloc[idx].copy()

    target_col = task.target_col

    # Feature engineering
    all_data = pd.concat([dfs["train"], dfs["val"], dfs["test"]], ignore_index=True)
    exclude_cols = [target_col, 'CREATIONTIMESTAMP', 'timestamp']
    id_cols = [c for c in all_data.columns if c.endswith('_id') or c.endswith('Id') or c == 'ID']
    exclude_cols.extend(id_cols)
    feature_cols = [c for c in all_data.columns if c not in exclude_cols]

    # Encode categorical features
    label_encoders = {}
    for col in feature_cols:
        if all_data[col].dtype == 'object' or all_data[col].dtype.name == 'category':
            le = LabelEncoder()
            all_data[col] = all_data[col].astype(str).fillna('__MISSING__')
            le.fit(all_data[col])
            label_encoders[col] = le

    # Prepare datasets
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

    # Encode target
    target_le = LabelEncoder()
    all_y = np.concatenate([y_data['train'], y_data['val'], y_data['test']])
    target_le.fit(all_y)
    for split in y_data:
        y_data[split] = target_le.transform(y_data[split])

    num_classes = len(target_le.classes_)
    print(f"  Classes: {num_classes}, Train: {len(y_data['train'])}, "
          f"Val: {len(y_data['val'])}, Test: {len(y_data['test'])}")

    # Train LightGBM
    print(f"  Training LightGBM (seed={SEED})...")
    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': SEED,
        'n_jobs': -1,
    }

    train_data = lgb.Dataset(X_data['train'], label=y_data['train'])
    val_data = lgb.Dataset(X_data['val'], label=y_data['val'], reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Predict probabilities
    val_probs = model.predict(X_data['val'])
    test_probs = model.predict(X_data['test'])

    # Split validation 50/50 into calibration and evaluation
    n_val = len(val_probs)
    n_calib = n_val // 2

    calib_probs = val_probs[:n_calib]
    calib_y = y_data['val'][:n_calib]

    # Compute APS conformity scores
    print(f"  Computing APS conformity scores...")
    calib_scores = compute_aps_scores(calib_probs, calib_y)
    test_scores = compute_aps_scores(test_probs, y_data['test'])

    print(f"  Calib scores: mean={calib_scores.mean():.4f}, median={np.median(calib_scores):.4f}")
    print(f"  Test scores:  mean={test_scores.mean():.4f}, median={np.median(test_scores):.4f}")

    return calib_scores, test_scores, num_classes


def run_ks_tests(calib_scores, test_scores):
    """
    Run KS tests for stochastic dominance.

    One-sided (alternative='less'): Tests H1: CDF_calib(x) > CDF_test(x)
        i.e., test scores are stochastically larger (shifted right).
        This means test instances have higher conformity scores, indicating
        the model is less confident on test data.

    Two-sided: Tests if the distributions differ at all.
    """
    # One-sided: test if test scores stochastically dominate calib scores
    ks_stat_one, ks_pval_one = ks_2samp(calib_scores, test_scores, alternative='less')

    # Two-sided: test if distributions differ
    ks_stat_two, ks_pval_two = ks_2samp(calib_scores, test_scores, alternative='two-sided')

    return {
        'ks_statistic': float(ks_stat_one),
        'ks_pvalue': float(ks_pval_one),
        'ks_statistic_twosided': float(ks_stat_two),
        'ks_pvalue_twosided': float(ks_pval_two),
        'n_calib': int(len(calib_scores)),
        'n_test': int(len(test_scores)),
        'calib_score_mean': float(np.mean(calib_scores)),
        'test_score_mean': float(np.mean(test_scores)),
        'calib_score_median': float(np.median(calib_scores)),
        'test_score_median': float(np.median(test_scores)),
    }


def plot_cdf_panels(all_scores, all_results, save_path):
    """
    Generate 2-panel CDF figure:
    - Left: sales-shipcond (catastrophic task)
    - Right: sales-office (robust task)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    panel_tasks = [
        ('sales-shipcond', '(A) Catastrophic Task (sales-shipcond)'),
        ('sales-office', '(B) Robust Task (sales-office)'),
    ]

    for ax, (task_name, title) in zip(axes, panel_tasks):
        calib_scores = all_scores[task_name]['calib']
        test_scores = all_scores[task_name]['test']
        result = all_results[task_name]

        # Sort scores for CDF plotting
        calib_sorted = np.sort(calib_scores)
        test_sorted = np.sort(test_scores)
        calib_cdf = np.arange(1, len(calib_sorted) + 1) / len(calib_sorted)
        test_cdf = np.arange(1, len(test_sorted) + 1) / len(test_sorted)

        # Plot CDFs
        ax.step(calib_sorted, calib_cdf, where='post', color='#2166ac',
                linewidth=1.8, label='Calibration (pre-COVID)')
        ax.step(test_sorted, test_cdf, where='post', color='#b2182b',
                linewidth=1.8, label='Test (COVID-era)')

        # Annotate with KS results
        ks_stat = result['ks_statistic_twosided']
        ks_pval = result['ks_pvalue_twosided']
        if ks_pval < 1e-10:
            pval_str = f"p < 10$^{{-10}}$"
        elif ks_pval < 0.001:
            pval_str = f"p = {ks_pval:.1e}"
        else:
            pval_str = f"p = {ks_pval:.3f}"

        ax.text(0.97, 0.08, f"KS = {ks_stat:.3f}\n{pval_str}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='gray', alpha=0.9))

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('APS Conformity Score', fontsize=10)
        ax.set_ylabel('Cumulative Probability', fontsize=10)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.tick_params(labelsize=9)

        # Clean style: no grid
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {save_path}")
    print(f"Figure saved: {save_path.with_suffix('.png')}")


def main():
    print("=" * 70)
    print("CONFORMITY SCORE CDF AND KS TEST ANALYSIS")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_scores = {}

    for i, task_name in enumerate(ALL_TASKS):
        print(f"\n[{i+1}/{len(ALL_TASKS)}] Processing: {task_name}")
        print("-" * 50)

        try:
            calib_scores, test_scores, num_classes = prepare_data_and_train(task_name)

            # Run KS tests
            result = run_ks_tests(calib_scores, test_scores)
            result['num_classes'] = num_classes
            all_results[task_name] = result
            all_scores[task_name] = {
                'calib': calib_scores,
                'test': test_scores,
            }

            print(f"  KS (one-sided): D={result['ks_statistic']:.4f}, p={result['ks_pvalue']:.2e}")
            print(f"  KS (two-sided): D={result['ks_statistic_twosided']:.4f}, p={result['ks_pvalue_twosided']:.2e}")

            # Save incrementally (per-task JSON saving)
            with open(OUTPUT_JSON, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved intermediate results ({len(all_results)} tasks)")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Final save
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFinal results saved: {OUTPUT_JSON}")

    # Generate figure
    if 'sales-shipcond' in all_scores and 'sales-office' in all_scores:
        plot_cdf_panels(all_scores, all_results, FIGURES_DIR / "conformity_score_cdfs.pdf")
    else:
        print("\nWARNING: Cannot generate figure - missing sales-shipcond or sales-office data")

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY: KS STOCHASTIC DOMINANCE TESTS")
    print("=" * 100)
    print(f"{'Task':<18} {'KS(1s)':>8} {'p(1s)':>12} {'KS(2s)':>8} {'p(2s)':>12} "
          f"{'Mean cal':>9} {'Mean test':>9} {'Classes':>8}")
    print("-" * 100)

    for task_name in ALL_TASKS:
        if task_name in all_results:
            r = all_results[task_name]
            print(f"{task_name:<18} {r['ks_statistic']:>8.4f} {r['ks_pvalue']:>12.2e} "
                  f"{r['ks_statistic_twosided']:>8.4f} {r['ks_pvalue_twosided']:>12.2e} "
                  f"{r['calib_score_mean']:>9.4f} {r['test_score_mean']:>9.4f} "
                  f"{r['num_classes']:>8d}")


if __name__ == "__main__":
    main()
