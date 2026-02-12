"""
Generate 4-panel (2x2) CDF figure for APS conformity scores.

Panels:
    (A) sales-shipcond  — catastrophic (KS=0.956, drop=71.6%)
    (B) sales-payterms  — catastrophic (KS=0.748, drop=77.1%)
    (C) sales-group     — catastrophic (KS=0.676, drop=71.2%)
    (D) sales-office    — robust       (KS=0.994, drop=0.1%)

Recomputes APS conformity scores from scratch (LightGBM, seed=42),
then plots calibration vs test CDFs with KS annotations.

Output:
    uai_2026/figures/conformity_score_cdfs.pdf
    uai_2026/figures/conformity_score_cdfs.png

Usage:
    PYTHONPATH=/Users/i767700/Github/ai-in-finance:$PYTHONPATH \
    /usr/bin/python3 -u papers/conformal_covid/code/generate_4panel_cdf.py
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

# The 4 tasks we need (3 catastrophic + 1 robust)
PANEL_TASKS = [
    ('sales-shipcond', 'Catastrophic', 71.6),
    ('sales-payterms', 'Catastrophic', 77.1),
    ('sales-group',    'Catastrophic', 71.2),
    ('sales-office',   'Robust',       0.1),
]

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
FIGURES_DIR = SCRIPT_DIR.parent / "uai_2026" / "figures"
KS_JSON = RESULTS_DIR / "ks_stochastic_dominance.json"


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
    sorted_idx = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    true_mask = (sorted_idx == y_true[:, None])
    scores = np.sum(cumsum * true_mask, axis=1)
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


def plot_4panel_cdf(all_scores, all_ks_results, save_path):
    """
    Generate 2x2 CDF figure for 3 catastrophic + 1 robust task.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
    axes_flat = axes.flatten()

    panel_labels = ['(A)', '(B)', '(C)', '(D)']

    for idx, (task_name, category, cov_drop) in enumerate(PANEL_TASKS):
        ax = axes_flat[idx]
        label = panel_labels[idx]

        calib_scores = all_scores[task_name]['calib']
        test_scores = all_scores[task_name]['test']

        # Sort scores for CDF plotting
        calib_sorted = np.sort(calib_scores)
        test_sorted = np.sort(test_scores)
        calib_cdf = np.arange(1, len(calib_sorted) + 1) / len(calib_sorted)
        test_cdf = np.arange(1, len(test_sorted) + 1) / len(test_sorted)

        # Plot CDFs
        ax.step(calib_sorted, calib_cdf, where='post', color='#2166ac',
                linewidth=1.5, label='Calibration (pre-COVID)')
        ax.step(test_sorted, test_cdf, where='post', color='#b2182b',
                linewidth=1.5, label='Test (COVID-era)')

        # Get KS results
        ks_result = all_ks_results.get(task_name, {})
        ks_stat = ks_result.get('ks_statistic_twosided', 0.0)
        ks_pval = ks_result.get('ks_pvalue_twosided', 1.0)

        # Format p-value string
        if ks_pval == 0.0 or ks_pval < 1e-10:
            pval_str = "p < 10$^{-10}$"
        elif ks_pval < 0.001:
            pval_str = f"p = {ks_pval:.1e}"
        else:
            pval_str = f"p = {ks_pval:.3f}"

        # KS annotation box
        ax.text(0.97, 0.08, f"KS = {ks_stat:.3f}\n{pval_str}",
                transform=ax.transAxes, fontsize=8.5,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='gray', alpha=0.9))

        # Panel title: label + task name + category
        # Clean task name for display
        task_display = task_name.replace('sales-', '').replace('item-', '')
        ax.set_title(f"{label} {task_name} ({category.lower()}, $\\Delta$cov = {cov_drop:.1f}%)",
                     fontsize=9, fontweight='bold', pad=6)

        ax.set_xlabel('APS Conformity Score', fontsize=9)
        ax.set_ylabel('Cumulative Probability', fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=8)

        # Clean academic style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend only on first panel to avoid clutter
        if idx == 0:
            ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9)

    plt.tight_layout(h_pad=1.2, w_pad=1.0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {save_path}")
    print(f"Figure saved: {save_path.with_suffix('.png')}")


def main():
    print("=" * 70)
    print("GENERATE 4-PANEL CDF FIGURE")
    print("=" * 70)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing KS results for annotation
    if KS_JSON.exists():
        with open(KS_JSON, 'r') as f:
            ks_results = json.load(f)
        print(f"Loaded KS results from {KS_JSON}")
    else:
        ks_results = {}
        print("WARNING: No KS results JSON found; will compute fresh KS stats.")

    # Compute conformity scores for each task
    all_scores = {}
    task_names = [t[0] for t in PANEL_TASKS]

    for i, task_name in enumerate(task_names):
        print(f"\n[{i+1}/{len(task_names)}] Processing: {task_name}")
        print("-" * 50)

        try:
            calib_scores, test_scores, num_classes = prepare_data_and_train(task_name)
            all_scores[task_name] = {
                'calib': calib_scores,
                'test': test_scores,
            }

            # Compute fresh KS stats if not in loaded results
            if task_name not in ks_results:
                ks_stat_two, ks_pval_two = ks_2samp(calib_scores, test_scores,
                                                     alternative='two-sided')
                ks_results[task_name] = {
                    'ks_statistic_twosided': float(ks_stat_two),
                    'ks_pvalue_twosided': float(ks_pval_two),
                }
                print(f"  KS (two-sided): D={ks_stat_two:.4f}, p={ks_pval_two:.2e}")
            else:
                print(f"  KS (from JSON): D={ks_results[task_name]['ks_statistic_twosided']:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Check all 4 tasks computed
    missing = [t for t in task_names if t not in all_scores]
    if missing:
        print(f"\nERROR: Missing scores for: {missing}")
        print("Cannot generate figure.")
        return

    # Generate the 4-panel figure
    print("\n" + "=" * 70)
    print("GENERATING 4-PANEL FIGURE")
    print("=" * 70)
    output_path = FIGURES_DIR / "conformity_score_cdfs.pdf"
    plot_4panel_cdf(all_scores, ks_results, output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
