"""
Multi-Seed RAPS Validation for UAI 2026

Runs BOTH APS and RAPS with 10 seeds on all 8 SALT multiclass tasks.
Key question: Does SHAP concentration predict RAPS coverage drop across tasks?

The single-seed RAPS experiment showed weak correlations (rho=0.262, p=0.53).
Multi-seed averaging should reduce noise and reveal the true signal.

RAPS parameters: lambda_reg=0.01, k_reg=5 (Angelopoulos et al. 2021)

Usage:
    PYTHONPATH=/Users/i767700/Github/ai-in-finance:$PYTHONPATH \
    /usr/bin/python3 -u papers/conformal_covid/code/run_raps_multiseed.py

Output:
    papers/conformal_covid/results/raps_multiseed_validation.json
"""

import json
import time
import traceback
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Configuration
NUM_SEEDS = 10
SEED_START = 42  # Seeds: 42..51
ALPHA = 0.1
SAMPLE_SIZE = 30000

# RAPS hyperparameters (Angelopoulos et al. 2021)
LAMBDA_REG = 0.01
K_REG = 5

ALL_TASKS = [
    'sales-shipcond', 'sales-group', 'sales-payterms', 'item-plant',
    'item-shippoint', 'sales-incoterms', 'item-incoterms', 'sales-office',
]

# LightGBM SHAP concentrations (top-3 feature %)
LGB_CONCENTRATIONS = {
    'sales-shipcond': 50.70, 'sales-group': 47.30, 'sales-payterms': 54.18,
    'item-plant': 23.90, 'item-shippoint': 48.79, 'sales-incoterms': 23.66,
    'item-incoterms': 28.93, 'sales-office': 42.65,
}

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_FILE = RESULTS_DIR / "raps_multiseed_validation.json"


class APSClassifier:
    """Standard Adaptive Prediction Sets."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.quantile = None

    def _compute_scores(self, probs, y_true):
        n = len(y_true)
        scores = np.zeros(n)
        for i in range(n):
            sorted_idx = np.argsort(probs[i])[::-1]
            cumsum = 0.0
            for idx in sorted_idx:
                cumsum += probs[i][idx]
                if idx == y_true[i]:
                    scores[i] = cumsum
                    break
        return scores

    def calibrate(self, probs, y_true):
        scores = self._compute_scores(probs, y_true)
        n = len(scores)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.quantile = np.quantile(scores, q_level)
        return self

    def predict_sets(self, probs):
        sets = []
        for i in range(len(probs)):
            sorted_idx = np.argsort(probs[i])[::-1]
            pred_set = set()
            cumsum = 0.0
            for idx in sorted_idx:
                pred_set.add(idx)
                cumsum += probs[i][idx]
                if cumsum >= self.quantile:
                    break
            sets.append(pred_set)
        return sets


class RAPSClassifier:
    """Regularized Adaptive Prediction Sets (Angelopoulos et al. 2021)."""

    def __init__(self, alpha=0.1, lambda_reg=0.01, k_reg=5):
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.k_reg = k_reg
        self.quantile = None

    def _compute_scores(self, probs, y_true):
        n = len(y_true)
        scores = np.zeros(n)
        for i in range(n):
            sorted_idx = np.argsort(probs[i])[::-1]
            cumsum = 0.0
            for rank, idx in enumerate(sorted_idx):
                cumsum += probs[i][idx]
                reg_penalty = self.lambda_reg * max(rank + 1 - self.k_reg, 0)
                if idx == y_true[i]:
                    scores[i] = cumsum + reg_penalty
                    break
        return scores

    def calibrate(self, probs, y_true):
        scores = self._compute_scores(probs, y_true)
        n = len(scores)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.quantile = np.quantile(scores, q_level)
        return self

    def predict_sets(self, probs):
        sets = []
        for i in range(len(probs)):
            sorted_idx = np.argsort(probs[i])[::-1]
            pred_set = set()
            cumsum = 0.0
            for rank, idx in enumerate(sorted_idx):
                pred_set.add(idx)
                cumsum += probs[i][idx]
                reg_penalty = self.lambda_reg * max(rank + 1 - self.k_reg, 0)
                if cumsum + reg_penalty >= self.quantile:
                    break
            sets.append(pred_set)
        return sets


def load_and_preprocess_data(task, seed):
    """Load task data and preprocess features (same pipeline as 50-seed ensemble)."""
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
        entity_df_copy = entity_df_copy.astype(
            {entity_table.pkey_col: table.df[left_entity].dtype}
        )
        for col in set(entity_df_copy.columns).intersection(set(table.df.columns)):
            if col != entity_table.pkey_col:
                entity_df_copy = entity_df_copy.drop(columns=[col])
        dfs[split] = table.df.merge(
            entity_df_copy, how="left",
            left_on=left_entity, right_on=entity_table.pkey_col,
        )

    # Subsample training
    if SAMPLE_SIZE < len(dfs["train"]):
        np.random.seed(seed)
        idx = np.random.permutation(len(dfs["train"]))[:SAMPLE_SIZE]
        dfs["train"] = dfs["train"].iloc[idx].copy()

    target_col = task.target_col

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
    return X_data, y_data, feature_cols, num_classes


def run_single_seed(task, task_name, seed):
    """Run APS + RAPS for one seed. Returns dict with coverage metrics."""
    X_data, y_data, feature_cols, num_classes = load_and_preprocess_data(task, seed)

    # Train LightGBM (same config as run_50seed_ensemble.py)
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

    train_ds = lgb.Dataset(X_data['train'], label=y_data['train'])
    val_ds = lgb.Dataset(X_data['val'], label=y_data['val'], reference=train_ds)
    model = lgb.train(
        params, train_ds, num_boost_round=500,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    val_probs = model.predict(X_data['val'])
    test_probs = model.predict(X_data['test'])

    # Split validation 50/50 for calibration (randomized per seed)
    np.random.seed(seed)
    n_val = len(val_probs)
    perm = np.random.permutation(n_val)
    n_calib = n_val // 2

    calib_probs = val_probs[perm[:n_calib]]
    calib_y = y_data['val'][perm[:n_calib]]
    eval_probs = val_probs[perm[n_calib:]]
    eval_y = y_data['val'][perm[n_calib:]]

    results = {'seed': seed, 'num_classes': num_classes}

    # --- APS ---
    aps = APSClassifier(alpha=ALPHA)
    aps.calibrate(calib_probs, calib_y)

    aps_val_sets = aps.predict_sets(eval_probs)
    aps_test_sets = aps.predict_sets(test_probs)

    aps_val_cov = sum(1 for i, s in enumerate(aps_val_sets) if eval_y[i] in s) / len(aps_val_sets)
    aps_test_cov = sum(1 for i, s in enumerate(aps_test_sets) if y_data['test'][i] in s) / len(aps_test_sets)

    results['aps_val_coverage'] = aps_val_cov
    results['aps_test_coverage'] = aps_test_cov
    results['aps_coverage_drop'] = aps_val_cov - aps_test_cov
    results['aps_val_set_size'] = np.mean([len(s) for s in aps_val_sets])
    results['aps_test_set_size'] = np.mean([len(s) for s in aps_test_sets])

    # --- RAPS ---
    raps = RAPSClassifier(alpha=ALPHA, lambda_reg=LAMBDA_REG, k_reg=K_REG)
    raps.calibrate(calib_probs, calib_y)

    raps_val_sets = raps.predict_sets(eval_probs)
    raps_test_sets = raps.predict_sets(test_probs)

    raps_val_cov = sum(1 for i, s in enumerate(raps_val_sets) if eval_y[i] in s) / len(raps_val_sets)
    raps_test_cov = sum(1 for i, s in enumerate(raps_test_sets) if y_data['test'][i] in s) / len(raps_test_sets)

    results['raps_val_coverage'] = raps_val_cov
    results['raps_test_coverage'] = raps_test_cov
    results['raps_coverage_drop'] = raps_val_cov - raps_test_cov
    results['raps_val_set_size'] = np.mean([len(s) for s in raps_val_sets])
    results['raps_test_set_size'] = np.mean([len(s) for s in raps_test_sets])

    return results


def run_task(task_name, existing_results):
    """Run all seeds for one task. Returns aggregated result dict."""
    from relbench.tasks import get_task

    print(f"\n{'='*70}", flush=True)
    print(f"Task: {task_name}", flush=True)
    print(f"{'='*70}", flush=True)

    t_start = time.time()
    task = get_task('rel-salt', task_name, download=False)

    seeds = list(range(SEED_START, SEED_START + NUM_SEEDS))
    seed_results = []

    # Check which seeds are already done (for resume)
    done_seeds = set()
    if task_name in existing_results and 'seed_results' in existing_results[task_name]:
        for sr in existing_results[task_name]['seed_results']:
            done_seeds.add(sr['seed'])
            seed_results.append(sr)
        print(f"  Resuming: {len(done_seeds)} seeds already done", flush=True)

    for seed in seeds:
        if seed in done_seeds:
            continue
        st = time.time()
        print(f"  Seed {seed} ({len(seed_results)+1}/{NUM_SEEDS})...", end='', flush=True)
        try:
            res = run_single_seed(task, task_name, seed)
            seed_results.append(res)
            elapsed = time.time() - st
            print(f" APS drop={res['aps_coverage_drop']*100:.1f}%, "
                  f"RAPS drop={res['raps_coverage_drop']*100:.1f}% "
                  f"({elapsed:.0f}s)", flush=True)
        except Exception as e:
            print(f" ERROR: {e}", flush=True)
            traceback.print_exc()

    # Aggregate across seeds
    aps_drops = [r['aps_coverage_drop'] for r in seed_results]
    raps_drops = [r['raps_coverage_drop'] for r in seed_results]
    aps_val_covs = [r['aps_val_coverage'] for r in seed_results]
    aps_test_covs = [r['aps_test_coverage'] for r in seed_results]
    raps_val_covs = [r['raps_val_coverage'] for r in seed_results]
    raps_test_covs = [r['raps_test_coverage'] for r in seed_results]
    aps_val_sizes = [r['aps_val_set_size'] for r in seed_results]
    aps_test_sizes = [r['aps_test_set_size'] for r in seed_results]
    raps_val_sizes = [r['raps_val_set_size'] for r in seed_results]
    raps_test_sizes = [r['raps_test_set_size'] for r in seed_results]

    elapsed_total = time.time() - t_start

    agg = {
        'num_classes': seed_results[0]['num_classes'],
        'concentration': LGB_CONCENTRATIONS[task_name],
        'num_seeds': len(seed_results),
        # APS aggregated
        'aps_val_coverage_mean': round(float(np.mean(aps_val_covs) * 100), 2),
        'aps_val_coverage_std': round(float(np.std(aps_val_covs) * 100), 2),
        'aps_test_coverage_mean': round(float(np.mean(aps_test_covs) * 100), 2),
        'aps_test_coverage_std': round(float(np.std(aps_test_covs) * 100), 2),
        'aps_drop_mean': round(float(np.mean(aps_drops) * 100), 2),
        'aps_drop_std': round(float(np.std(aps_drops) * 100), 2),
        'aps_val_size_mean': round(float(np.mean(aps_val_sizes)), 2),
        'aps_test_size_mean': round(float(np.mean(aps_test_sizes)), 2),
        # RAPS aggregated
        'raps_val_coverage_mean': round(float(np.mean(raps_val_covs) * 100), 2),
        'raps_val_coverage_std': round(float(np.std(raps_val_covs) * 100), 2),
        'raps_test_coverage_mean': round(float(np.mean(raps_test_covs) * 100), 2),
        'raps_test_coverage_std': round(float(np.std(raps_test_covs) * 100), 2),
        'raps_drop_mean': round(float(np.mean(raps_drops) * 100), 2),
        'raps_drop_std': round(float(np.std(raps_drops) * 100), 2),
        'raps_val_size_mean': round(float(np.mean(raps_val_sizes)), 2),
        'raps_test_size_mean': round(float(np.mean(raps_test_sizes)), 2),
        # Per-seed raw data
        'seed_results': seed_results,
        'elapsed_s': round(elapsed_total, 1),
    }

    print(f"\n  Summary for {task_name}:", flush=True)
    print(f"    APS:  val={agg['aps_val_coverage_mean']:.1f}+-{agg['aps_val_coverage_std']:.1f}%, "
          f"test={agg['aps_test_coverage_mean']:.1f}+-{agg['aps_test_coverage_std']:.1f}%, "
          f"drop={agg['aps_drop_mean']:.1f}+-{agg['aps_drop_std']:.1f}%", flush=True)
    print(f"    RAPS: val={agg['raps_val_coverage_mean']:.1f}+-{agg['raps_val_coverage_std']:.1f}%, "
          f"test={agg['raps_test_coverage_mean']:.1f}+-{agg['raps_test_coverage_std']:.1f}%, "
          f"drop={agg['raps_drop_mean']:.1f}+-{agg['raps_drop_std']:.1f}%", flush=True)
    print(f"    Time: {elapsed_total:.0f}s", flush=True)

    return agg


def _save_results(task_results, correlation_summary=None):
    """Save results to JSON."""
    clean_tasks = {}
    for t, r in task_results.items():
        clean = {k: v for k, v in r.items() if k != 'seed_results'}
        clean['seed_results'] = r.get('seed_results', [])
        clean_tasks[t] = clean

    output = {
        'config': {
            'num_seeds': NUM_SEEDS,
            'seed_start': SEED_START,
            'alpha': ALPHA,
            'sample_size': SAMPLE_SIZE,
            'raps_lambda': LAMBDA_REG,
            'raps_k_reg': K_REG,
        },
        'tasks': clean_tasks,
        'completed_tasks': list(task_results.keys()),
        'n_completed': len(task_results),
    }

    if correlation_summary:
        output['correlation'] = correlation_summary

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)


def main():
    print(f"\n{'='*70}", flush=True)
    print("Multi-Seed APS vs RAPS Validation", flush=True)
    print(f"Seeds: {NUM_SEEDS} ({SEED_START}..{SEED_START + NUM_SEEDS - 1})", flush=True)
    print(f"RAPS params: lambda={LAMBDA_REG}, k_reg={K_REG}", flush=True)
    print(f"Tasks: {len(ALL_TASKS)}", flush=True)
    print(f"{'='*70}\n", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing partial results for resume
    existing_results = {}
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, 'r') as f:
                existing = json.load(f)
            if 'tasks' in existing:
                existing_results = existing['tasks']
                completed = [t for t in ALL_TASKS if t in existing_results
                             and existing_results[t].get('num_seeds', 0) == NUM_SEEDS]
                print(f"Resume: found {len(completed)} fully completed tasks", flush=True)
        except Exception:
            pass

    t_total = time.time()
    task_results = {}

    for task_name in ALL_TASKS:
        # Skip if fully completed
        if (task_name in existing_results
                and existing_results[task_name].get('num_seeds', 0) == NUM_SEEDS):
            print(f"\nSkipping {task_name} (already completed)", flush=True)
            task_results[task_name] = existing_results[task_name]
            continue

        try:
            agg = run_task(task_name, existing_results)
            task_results[task_name] = agg

            # Incremental save
            _save_results(task_results)
            print(f"  Saved ({len(task_results)}/{len(ALL_TASKS)})", flush=True)

        except Exception as e:
            print(f"\nERROR on {task_name}: {e}", flush=True)
            traceback.print_exc()

    # Final analysis
    print(f"\n{'='*70}", flush=True)
    print("FINAL ANALYSIS", flush=True)
    print(f"{'='*70}", flush=True)

    completed = [t for t in ALL_TASKS if t in task_results]
    if len(completed) < 3:
        print("Not enough tasks completed for correlation analysis.", flush=True)
        return

    concs = [task_results[t]['concentration'] for t in completed]
    aps_drops = [task_results[t]['aps_drop_mean'] for t in completed]
    raps_drops = [task_results[t]['raps_drop_mean'] for t in completed]

    # Spearman correlations: concentration vs drop
    aps_rho, aps_p = stats.spearmanr(concs, aps_drops)
    raps_rho, raps_p = stats.spearmanr(concs, raps_drops)

    # Kendall tau
    aps_tau, aps_tau_p = stats.kendalltau(concs, aps_drops)
    raps_tau, raps_tau_p = stats.kendalltau(concs, raps_drops)

    # APS vs RAPS drop correlation
    aps_raps_rho, aps_raps_p = stats.spearmanr(aps_drops, raps_drops)

    # Bootstrap CIs (1000 resamples)
    n_boot = 1000
    np.random.seed(42)
    aps_boot, raps_boot = [], []
    n = len(completed)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        c_b = [concs[i] for i in idx]
        a_b = [aps_drops[i] for i in idx]
        r_b = [raps_drops[i] for i in idx]
        if len(set(c_b)) > 1 and len(set(a_b)) > 1:
            aps_boot.append(stats.spearmanr(c_b, a_b).statistic)
        if len(set(c_b)) > 1 and len(set(r_b)) > 1:
            raps_boot.append(stats.spearmanr(c_b, r_b).statistic)

    aps_ci = (round(float(np.percentile(aps_boot, 2.5)), 3),
              round(float(np.percentile(aps_boot, 97.5)), 3)) if aps_boot else (None, None)
    raps_ci = (round(float(np.percentile(raps_boot, 2.5)), 3),
               round(float(np.percentile(raps_boot, 97.5)), 3)) if raps_boot else (None, None)

    print(f"\nConcentration vs Coverage Drop (n={n}):", flush=True)
    print(f"  APS:  rho={aps_rho:.3f} (p={aps_p:.4f}), tau={aps_tau:.3f} (p={aps_tau_p:.4f}), "
          f"95% CI [{aps_ci[0]}, {aps_ci[1]}]", flush=True)
    print(f"  RAPS: rho={raps_rho:.3f} (p={raps_p:.4f}), tau={raps_tau:.3f} (p={raps_tau_p:.4f}), "
          f"95% CI [{raps_ci[0]}, {raps_ci[1]}]", flush=True)
    print(f"  APS-RAPS drop correlation: rho={aps_raps_rho:.3f} (p={aps_raps_p:.4f})", flush=True)

    # Comparison table
    print(f"\n{'='*70}", flush=True)
    print(f"{'Task':<18} {'Conc':>6} {'APS Drop (mean+-std)':>22} {'RAPS Drop (mean+-std)':>23}", flush=True)
    print(f"{'-'*70}", flush=True)
    for t in ALL_TASKS:
        if t in task_results:
            r = task_results[t]
            print(f"{t:<18} {r['concentration']:>5.1f}% "
                  f"{r['aps_drop_mean']:>7.1f}+-{r['aps_drop_std']:>4.1f}%  "
                  f"{r['raps_drop_mean']:>7.1f}+-{r['raps_drop_std']:>4.1f}%", flush=True)

    total_time = time.time() - t_total
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    # Save final with correlation results
    correlation_summary = {
        'n_tasks': n,
        'aps_spearman_rho': round(float(aps_rho), 3),
        'aps_spearman_p': round(float(aps_p), 4),
        'aps_kendall_tau': round(float(aps_tau), 3),
        'aps_kendall_p': round(float(aps_tau_p), 4),
        'aps_boot_ci_95': list(aps_ci),
        'raps_spearman_rho': round(float(raps_rho), 3),
        'raps_spearman_p': round(float(raps_p), 4),
        'raps_kendall_tau': round(float(raps_tau), 3),
        'raps_kendall_p': round(float(raps_tau_p), 4),
        'raps_boot_ci_95': list(raps_ci),
        'aps_vs_raps_drop_rho': round(float(aps_raps_rho), 3),
        'aps_vs_raps_drop_p': round(float(aps_raps_p), 4),
    }

    _save_results(task_results, correlation_summary)
    print(f"\nResults saved to: {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
