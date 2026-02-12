"""
Compute Baseline Diagnostics for Comparison Against SHAP Concentration

Computes three pre-deployment diagnostic metrics and their Spearman correlation
with coverage drop under distribution shift:

1. SHAP concentration: top-1 SHAP / total SHAP (from existing CSV)
2. Native LightGBM FI concentration: top-1 gain / total gain (zero SHAP cost)
3. Ensemble disagreement: std of validation coverage across 50 seeds

All correlations are computed against 50-seed mean coverage drops from
statistical_rigor.json.

Output: results/baseline_diagnostics.json
"""

import json
import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"

ALL_TASKS = [
    'sales-shipcond', 'sales-group', 'sales-payterms', 'item-plant',
    'item-shippoint', 'sales-incoterms', 'item-incoterms', 'sales-office'
]

SAMPLE_SIZE = 30000
SEED = 42


def load_coverage_drops():
    """Load 50-seed mean coverage drops from statistical_rigor.json."""
    with open(RESULTS_DIR / "statistical_rigor.json", 'r') as f:
        data = json.load(f)

    drops = {}
    for task in ALL_TASKS:
        drops[task] = data[task]["coverage_drop"]["mean"]
    return drops


def load_shap_concentration():
    """Load SHAP concentration from existing CSV."""
    csv_path = RESULTS_DIR / "shap" / "concentration_all_tasks.csv"
    df = pd.read_csv(csv_path)
    conc = {}
    for _, row in df.iterrows():
        conc[row['task']] = row['concentration_pct']
    return conc


def load_ensemble_disagreement():
    """Load ensemble disagreement (std of val coverage across 50 seeds)."""
    with open(RESULTS_DIR / "ensemble_50seeds.pkl", 'rb') as f:
        data = pickle.load(f)

    disagreement = {}
    for item in data:
        task = item['task']
        # Compute std directly from per-seed val coverages for accuracy
        val_covs = [sr['val_coverage'] for sr in item['seed_results']]
        disagreement[task] = float(np.std(val_covs))
    return disagreement


def train_single_model(task_name, seed=SEED):
    """
    Train a single LightGBM model for a task and return native feature importance.

    Returns:
        dict mapping feature_name -> gain importance
    """
    from relbench.tasks import get_task

    task = get_task('rel-salt', task_name, download=False)

    # Load data (same pipeline as run_50seed_ensemble.py)
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

        # Remove duplicate columns
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
    if SAMPLE_SIZE and SAMPLE_SIZE < len(dfs["train"]):
        np.random.seed(seed)
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

    # Train LightGBM
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
        'seed': seed,
        'n_jobs': -1,
    }

    train_data = lgb.Dataset(X_data['train'], label=y_data['train'])
    val_data = lgb.Dataset(X_data['val'], label=y_data['val'], reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    # Extract native feature importance (gain)
    importance = model.feature_importance(importance_type='gain')
    fi_dict = dict(zip(feature_cols, importance.tolist()))

    return fi_dict, feature_cols


def compute_native_fi_concentration():
    """Compute native LightGBM FI concentration for each task (1 seed)."""
    concentrations = {}

    for task_name in ALL_TASKS:
        print(f"  Training LightGBM for {task_name}...")
        fi_dict, feature_cols = train_single_model(task_name)

        # Compute top-1 gain / total gain
        gains = np.array([fi_dict[f] for f in feature_cols])
        total_gain = gains.sum()
        top1_gain = gains.max()

        if total_gain > 0:
            concentration = (top1_gain / total_gain) * 100.0
        else:
            concentration = 0.0

        top_feature = feature_cols[np.argmax(gains)]
        concentrations[task_name] = {
            'concentration_pct': concentration,
            'top_feature': top_feature,
            'top1_gain': float(top1_gain),
            'total_gain': float(total_gain),
            'all_gains': {f: float(fi_dict[f]) for f in feature_cols}
        }
        print(f"    -> {task_name}: top={top_feature}, concentration={concentration:.1f}%")

    return concentrations


def compute_spearman(metric_values, coverage_drops, metric_name):
    """Compute Spearman correlation between a metric and coverage drops."""
    tasks = sorted(coverage_drops.keys())
    x = [metric_values[t] for t in tasks]
    y = [coverage_drops[t] for t in tasks]

    rho, p = stats.spearmanr(x, y)

    return {
        'rho': round(float(rho), 4),
        'p_value': round(float(p), 6),
        'n': len(tasks),
        'values': {t: round(float(metric_values[t]), 6) for t in tasks},
        'metric_name': metric_name
    }


def main():
    print("=" * 70)
    print("BASELINE DIAGNOSTICS COMPARISON")
    print("=" * 70)

    # 1. Load coverage drops
    print("\n[1/4] Loading coverage drops...")
    coverage_drops = load_coverage_drops()
    print(f"  Loaded drops for {len(coverage_drops)} tasks")
    for t in sorted(coverage_drops.keys()):
        print(f"    {t}: {coverage_drops[t]*100:.1f}%")

    # 2. Load SHAP concentration
    print("\n[2/4] Loading SHAP concentration...")
    shap_conc = load_shap_concentration()
    for t in sorted(shap_conc.keys()):
        print(f"    {t}: {shap_conc[t]:.1f}%")

    # 3. Load ensemble disagreement
    print("\n[3/4] Loading ensemble disagreement...")
    ens_disagree = load_ensemble_disagreement()
    for t in sorted(ens_disagree.keys()):
        print(f"    {t}: {ens_disagree[t]:.6f}")

    # 4. Compute native FI concentration
    print("\n[4/4] Computing native LightGBM FI concentration (1 seed each)...")
    native_fi = compute_native_fi_concentration()
    native_fi_conc = {t: v['concentration_pct'] for t, v in native_fi.items()}

    # Compute Spearman correlations
    print("\n" + "=" * 70)
    print("SPEARMAN CORRELATIONS WITH COVERAGE DROP")
    print("=" * 70)

    results = {}

    # SHAP concentration
    shap_result = compute_spearman(shap_conc, coverage_drops, "SHAP concentration")
    results['shap_concentration'] = shap_result
    print(f"\n  SHAP concentration:      rho={shap_result['rho']:.4f}, p={shap_result['p_value']:.6f}")

    # Native FI concentration
    fi_result = compute_spearman(native_fi_conc, coverage_drops, "Native FI concentration")
    results['native_fi_concentration'] = fi_result
    print(f"  Native FI concentration: rho={fi_result['rho']:.4f}, p={fi_result['p_value']:.6f}")

    # Ensemble disagreement
    ens_result = compute_spearman(ens_disagree, coverage_drops, "Ensemble disagreement")
    results['ensemble_disagreement'] = ens_result
    print(f"  Ensemble disagreement:   rho={ens_result['rho']:.4f}, p={ens_result['p_value']:.6f}")

    # Build comparison table
    tasks = sorted(coverage_drops.keys())
    header = f"{'Task':<18} {'Drop%':>7} {'SHAP%':>8} {'NativeFI%':>10} {'EnsDisagree':>12}"
    sep = "-" * len(header)
    rows = [header, sep]
    for t in tasks:
        rows.append(
            f"{t:<18} {coverage_drops[t]*100:>6.1f}% {shap_conc[t]:>7.1f}% "
            f"{native_fi_conc[t]:>9.1f}% {ens_disagree[t]:>11.6f}"
        )
    rows.append(sep)
    rows.append(f"{'Spearman rho':<18} {'':>7} {shap_result['rho']:>8.4f} "
                f"{fi_result['rho']:>10.4f} {ens_result['rho']:>12.4f}")
    rows.append(f"{'p-value':<18} {'':>7} {shap_result['p_value']:>8.4f} "
                f"{fi_result['p_value']:>10.4f} {ens_result['p_value']:>12.4f}")
    comparison_table = "\n".join(rows)

    print(f"\n{comparison_table}")

    # Build output JSON
    output = {
        'metrics': results,
        'coverage_drops': {t: round(float(coverage_drops[t]), 6) for t in tasks},
        'native_fi_details': {
            t: {
                'top_feature': native_fi[t]['top_feature'],
                'concentration_pct': round(native_fi[t]['concentration_pct'], 2),
                'all_gains': native_fi[t]['all_gains']
            }
            for t in tasks
        },
        'comparison_table': comparison_table,
        'notes': {
            'shap_concentration': 'Top-1 SHAP importance / total importance (from existing 50-seed analysis)',
            'native_fi_concentration': 'Top-1 LightGBM gain / total gain (single seed=42, zero SHAP cost)',
            'ensemble_disagreement': 'Std of validation coverage across 50 seeds (pre-deployment metric)',
            'coverage_drops': '50-seed mean coverage drops from statistical_rigor.json'
        }
    }

    # Save
    output_path = RESULTS_DIR / "baseline_diagnostics.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Summary interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    best_metric = max(results.items(), key=lambda x: abs(x[1]['rho']))
    print(f"\n  Best predictor: {best_metric[0]} (rho={best_metric[1]['rho']:.4f})")

    sig_metrics = [(k, v) for k, v in results.items() if v['p_value'] < 0.05]
    print(f"  Significant (p<0.05): {len(sig_metrics)}/{len(results)}")
    for name, vals in sig_metrics:
        print(f"    - {name}: rho={vals['rho']:.4f}, p={vals['p_value']:.6f}")

    nonsig = [(k, v) for k, v in results.items() if v['p_value'] >= 0.05]
    for name, vals in nonsig:
        print(f"    - {name}: rho={vals['rho']:.4f}, p={vals['p_value']:.6f} (n.s.)")


if __name__ == "__main__":
    main()
