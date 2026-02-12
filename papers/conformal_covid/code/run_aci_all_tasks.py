"""
ACI (Adaptive Conformal Inference) Experiment — All 8 SALT Tasks

Extends run_aci_experiment.py to cover all 8 tasks, not just sales-shipcond.
Key result: ACI helps catastrophic tasks but is unnecessary for robust ones.

Usage:
    python code/run_aci_all_tasks.py
    python code/run_aci_all_tasks.py --num_seeds 10 --gammas 0.001 0.01 0.05

Output:
    results/aci/aci_all_tasks_summary.json
    results/aci/aci_all_tasks_table.tex
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

ALPHA = 0.1
SAMPLE_SIZE = 30000
SEED_START = 42

ALL_TASKS = [
    'sales-shipcond', 'sales-group', 'sales-payterms',
    'item-plant', 'item-shippoint',
    'sales-incoterms', 'item-incoterms', 'sales-office'
]


class ConformalClassifier:
    """Adaptive Prediction Sets (APS) for classification."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.quantile = None

    def _compute_scores(self, probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        n = len(y_true)
        scores = np.zeros(n)
        for i in range(n):
            sorted_idx = np.argsort(probs[i])[::-1]
            cumsum = 0
            for idx in sorted_idx:
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

    def evaluate(self, probs: np.ndarray, y_true: np.ndarray) -> Dict:
        sets = self.predict_sets(probs)
        coverage = sum(1 for i, s in enumerate(sets) if y_true[i] in s) / len(sets)
        mean_size = np.mean([len(s) for s in sets])
        return {'coverage': coverage, 'mean_set_size': mean_size}


class AdaptiveConformalClassifier:
    """ACI (Gibbs & Candes, 2021) with online quantile updates."""

    def __init__(self, alpha: float = 0.1, gamma: float = 0.01):
        self.alpha = alpha
        self.gamma = gamma

    def run_online(self, calib_probs, calib_y, test_probs, test_y) -> Dict:
        n_calib = len(calib_y)
        calib_scores = np.zeros(n_calib)
        for i in range(n_calib):
            sorted_idx = np.argsort(calib_probs[i])[::-1]
            cumsum = 0
            for idx in sorted_idx:
                cumsum += calib_probs[i][idx]
                if idx == calib_y[i]:
                    calib_scores[i] = cumsum
                    break

        alpha_t = self.alpha
        coverages = []
        set_sizes = []

        for t in range(len(test_y)):
            q_level = np.clip(
                np.ceil((n_calib + 1) * (1 - alpha_t)) / n_calib, 0.0, 1.0
            )
            current_quantile = np.quantile(calib_scores, min(q_level, 1.0))

            sorted_idx = np.argsort(test_probs[t])[::-1]
            pred_set = set()
            cumsum = 0
            for idx in sorted_idx:
                pred_set.add(idx)
                cumsum += test_probs[t][idx]
                if cumsum >= current_quantile:
                    break
            set_sizes.append(len(pred_set))

            covered = test_y[t] in pred_set
            coverages.append(int(covered))

            err_t = 1 - int(covered)
            alpha_t = alpha_t + self.gamma * (self.alpha - err_t)
            alpha_t = np.clip(alpha_t, 0.001, 0.999)

        return {
            'coverage': np.mean(coverages),
            'mean_set_size': np.mean(set_sizes),
        }


def prepare_data(task, task_name: str, seed: int, sample_size: int = SAMPLE_SIZE):
    """Prepare data — matches run_50seed_ensemble.py exactly."""
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

    np.random.seed(seed)
    if sample_size and sample_size < len(dfs["train"]):
        idx = np.random.permutation(len(dfs["train"]))[:sample_size]
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


def run_single_seed(task, task_name, seed, gammas):
    """Run standard + ACI for one seed, return all metrics."""
    X_data, y_data, feature_cols, num_classes = prepare_data(task, task_name, seed)

    params = {
        'objective': 'multiclass', 'num_class': num_classes,
        'metric': 'multi_logloss', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'learning_rate': 0.05,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
        'verbose': -1, 'seed': seed, 'n_jobs': 1,
    }

    train_data = lgb.Dataset(X_data['train'], label=y_data['train'])
    val_data = lgb.Dataset(X_data['val'], label=y_data['val'], reference=train_data)

    model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    val_probs = model.predict(X_data['val'])
    test_probs = model.predict(X_data['test'])

    n_val = len(val_probs)
    n_calib = n_val // 2
    calib_probs, calib_y = val_probs[:n_calib], y_data['val'][:n_calib]
    eval_probs, eval_y = val_probs[n_calib:], y_data['val'][n_calib:]

    # --- Standard conformal ---
    conf = ConformalClassifier(alpha=ALPHA)
    conf.calibrate(calib_probs, calib_y)
    std_val = conf.evaluate(eval_probs, eval_y)
    std_test = conf.evaluate(test_probs, y_data['test'])

    result = {
        'seed': seed, 'task': task_name, 'num_classes': num_classes,
        'standard': {
            'val_coverage': std_val['coverage'],
            'test_coverage': std_test['coverage'],
            'val_set_size': std_val['mean_set_size'],
            'test_set_size': std_test['mean_set_size'],
        },
    }

    # --- ACI for each gamma ---
    for gamma in gammas:
        aci = AdaptiveConformalClassifier(alpha=ALPHA, gamma=gamma)
        aci_result = aci.run_online(calib_probs, calib_y, test_probs, y_data['test'])
        result[f'aci_{gamma}'] = {
            'test_coverage': aci_result['coverage'],
            'mean_set_size': aci_result['mean_set_size'],
        }

    # --- Entropy (post-hoc shift detection baseline) ---
    val_entropy = -np.sum(eval_probs * np.log(eval_probs + 1e-10), axis=1)
    test_entropy = -np.sum(test_probs * np.log(test_probs + 1e-10), axis=1)
    result['entropy'] = {
        'val_mean': float(np.mean(val_entropy)),
        'test_mean': float(np.mean(test_entropy)),
        'delta': float(np.mean(test_entropy) - np.mean(val_entropy)),
    }

    # --- ECE (post-hoc calibration error baseline) ---
    def compute_ece(probs, y_true, n_bins=15):
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == y_true).astype(float)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = accuracies[mask].mean()
                bin_conf = confidences[mask].mean()
                ece += mask.sum() * abs(bin_acc - bin_conf)
        return ece / len(y_true)

    val_ece = compute_ece(eval_probs, eval_y)
    test_ece = compute_ece(test_probs, y_data['test'])
    result['ece'] = {
        'val': float(val_ece),
        'test': float(test_ece),
        'delta': float(test_ece - val_ece),
    }

    return result


def compute_95ci(values):
    """Compute 95% confidence interval using t-distribution."""
    n = len(values)
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(n)
    ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    return mean, ci[0], ci[1]


def main():
    parser = argparse.ArgumentParser(description='ACI All Tasks Experiment')
    parser.add_argument('--tasks', nargs='+', default=ALL_TASKS)
    parser.add_argument('--num_seeds', type=int, default=10)
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.001, 0.01, 0.05])
    parser.add_argument('--output_dir', type=str, default='results/aci')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    from relbench.tasks import get_task

    # Merge with existing results (preserve previously computed tasks)
    output_file = output_dir / 'aci_all_tasks_summary.json'
    if output_file.exists():
        with open(output_file) as f:
            all_task_results = json.load(f)
        print(f"Loaded {len(all_task_results)} existing task results from {output_file}")
    else:
        all_task_results = {}

    for task_name in args.tasks:
        print(f"\n{'=' * 80}")
        print(f"Task: {task_name} ({args.num_seeds} seeds)")
        print(f"{'=' * 80}")

        task = get_task('rel-salt', task_name, download=False)
        seeds = list(range(SEED_START, SEED_START + args.num_seeds))

        seed_results = []
        for seed in seeds:
            try:
                result = run_single_seed(task, task_name, seed, args.gammas)
                std_cov = result['standard']['test_coverage']
                aci_cov = result.get('aci_0.01', {}).get('test_coverage', 'N/A')
                print(f"  Seed {seed}: Std={std_cov:.3f}, ACI(0.01)={aci_cov:.3f}")
                seed_results.append(result)
            except Exception as e:
                print(f"  ERROR seed {seed}: {e}")
                import traceback
                traceback.print_exc()

        if not seed_results:
            print(f"  SKIPPED {task_name} — no successful seeds")
            continue

        num_classes = seed_results[0]['num_classes']

        # --- Aggregate with 95% CI ---
        std_test_covs = [r['standard']['test_coverage'] for r in seed_results]
        std_val_covs = [r['standard']['val_coverage'] for r in seed_results]
        std_test_sizes = [r['standard']['test_set_size'] for r in seed_results]

        summary = {
            'task': task_name,
            'num_classes': num_classes,
            'num_seeds': len(seed_results),
            'standard': {
                'test_coverage_mean': float(np.mean(std_test_covs)),
                'test_coverage_std': float(np.std(std_test_covs)),
                'test_coverage_median': float(np.median(std_test_covs)),
                'test_coverage_95ci': list(compute_95ci(std_test_covs)),
                'test_set_size_mean': float(np.mean(std_test_sizes)),
                'test_coverages': [float(c) for c in std_test_covs],
                'fail_count': sum(1 for c in std_test_covs if c < 0.05),
            },
        }

        # ACI per gamma with paired tests
        for gamma in args.gammas:
            key = f'aci_{gamma}'
            aci_covs = [r[key]['test_coverage'] for r in seed_results]
            aci_sizes = [r[key]['mean_set_size'] for r in seed_results]

            # Paired Wilcoxon test (seed-level)
            try:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                    aci_covs, std_test_covs, alternative='greater'
                )
            except ValueError:
                wilcoxon_p = 1.0

            # Paired deltas
            deltas = [a - s for a, s in zip(aci_covs, std_test_covs)]
            sign_count = sum(1 for d in deltas if d > 0)
            sign_p = stats.binomtest(sign_count, len(deltas), 0.5, alternative='greater').pvalue if len(deltas) > 0 else 1.0

            summary[key] = {
                'test_coverage_mean': float(np.mean(aci_covs)),
                'test_coverage_std': float(np.std(aci_covs)),
                'test_coverage_median': float(np.median(aci_covs)),
                'test_coverage_95ci': list(compute_95ci(aci_covs)),
                'test_coverages': [float(c) for c in aci_covs],
                'mean_set_size': float(np.mean(aci_sizes)),
                'size_per_classes': float(np.mean(aci_sizes)) / num_classes,
                'fail_count': sum(1 for c in aci_covs if c < 0.05),
                'paired_wilcoxon_p': float(wilcoxon_p),
                'sign_test': f'{sign_count}/{len(deltas)}',
                'sign_test_p': float(sign_p),
                'paired_delta_mean': float(np.mean(deltas)),
                'paired_delta_95ci': list(compute_95ci(deltas)),
            }

        # Entropy & ECE baselines
        entropy_vals = [r['entropy']['val_mean'] for r in seed_results]
        entropy_tests = [r['entropy']['test_mean'] for r in seed_results]
        entropy_deltas = [r['entropy']['delta'] for r in seed_results]
        ece_vals = [r['ece']['val'] for r in seed_results]
        ece_tests = [r['ece']['test'] for r in seed_results]
        ece_deltas = [r['ece']['delta'] for r in seed_results]

        summary['entropy'] = {
            'val_mean': float(np.mean(entropy_vals)),
            'test_mean': float(np.mean(entropy_tests)),
            'delta_mean': float(np.mean(entropy_deltas)),
            'delta_95ci': list(compute_95ci(entropy_deltas)),
        }
        summary['ece'] = {
            'val_mean': float(np.mean(ece_vals)),
            'test_mean': float(np.mean(ece_tests)),
            'delta_mean': float(np.mean(ece_deltas)),
            'delta_95ci': list(compute_95ci(ece_deltas)),
        }

        all_task_results[task_name] = summary

        # Print
        print(f"\n  --- {task_name} Summary ---")
        s = summary['standard']
        print(f"  Standard: {s['test_coverage_mean']*100:.1f}% "
              f"[95%CI: {s['test_coverage_95ci'][1]*100:.1f}-{s['test_coverage_95ci'][2]*100:.1f}%] "
              f"| set_size={s['test_set_size_mean']:.1f} | fails={s['fail_count']}/{len(seed_results)}")
        for gamma in args.gammas:
            key = f'aci_{gamma}'
            a = summary[key]
            print(f"  ACI(γ={gamma}): {a['test_coverage_mean']*100:.1f}% "
                  f"[95%CI: {a['test_coverage_95ci'][1]*100:.1f}-{a['test_coverage_95ci'][2]*100:.1f}%] "
                  f"| set_size={a['mean_set_size']:.1f} ({a['size_per_classes']*100:.0f}% of classes) "
                  f"| Δ={a['paired_delta_mean']*100:+.1f}pp (p={a['paired_wilcoxon_p']:.3f})")
        print(f"  Entropy: val={summary['entropy']['val_mean']:.3f} → test={summary['entropy']['test_mean']:.3f} (Δ={summary['entropy']['delta_mean']:+.3f})")
        print(f"  ECE: val={summary['ece']['val_mean']:.4f} → test={summary['ece']['test_mean']:.4f} (Δ={summary['ece']['delta_mean']:+.4f})")

        # Save after each task to prevent data loss on interruption
        with open(output_file, 'w') as f:
            json.dump(all_task_results, f, indent=2)
        print(f"  Saved ({len(all_task_results)} tasks): {output_file}")

    print(f"\nFinal save: {output_file} ({len(all_task_results)} tasks)")

    # --- Generate LaTeX table ---
    print("\n\n=== LaTeX Table: ACI Results (All Tasks) ===\n")
    latex = generate_aci_latex_table(all_task_results, args.gammas)
    print(latex)
    with open(output_dir / 'aci_all_tasks_table.tex', 'w') as f:
        f.write(latex)

    # --- Generate Entropy/ECE comparison table ---
    print("\n\n=== LaTeX Table: Shift Detection Baselines ===\n")
    baseline_latex = generate_baseline_latex_table(all_task_results)
    print(baseline_latex)
    with open(output_dir / 'shift_detection_baselines_table.tex', 'w') as f:
        f.write(baseline_latex)


def generate_aci_latex_table(results, gammas):
    """Generate the ACI comparison table for the paper."""
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{ACI Under Distribution Shift: All 8 Tasks (10 seeds, $\alpha=0.1$). '
                 r'Paired $\Delta$ = ACI coverage $-$ Standard coverage (seed-level). '
                 r'$p$-values from Wilcoxon signed-rank test.}\label{tab:aci_all}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{@{}lccccccc@{}}')
    lines.append(r'\toprule')
    lines.append(r'Task & Cat & Standard & ACI ($\gamma$=0.01) & $\Delta$ & $p$ & Set Size & Size/\#cl \\')
    lines.append(r'\midrule')

    for task_name in ALL_TASKS:
        if task_name not in results:
            continue
        r = results[task_name]
        task_short = task_name.replace('sales-', 's-').replace('item-', 'i-')
        s = r['standard']
        a = r.get('aci_0.01', {})

        # Determine category
        drop = (s.get('test_coverage_mean', 0))
        if 'test_coverages' in s:
            val_mean = np.mean([sr for sr in [r['standard'].get('test_coverage_mean', 0)]])
        cat = 'SEV' if s['test_coverage_mean'] < 0.5 else 'ROB'

        std_str = f"{s['test_coverage_mean']*100:.1f}$\\pm${s['test_coverage_std']*100:.1f}\\%"
        if a:
            aci_str = f"{a['test_coverage_mean']*100:.1f}$\\pm${a['test_coverage_std']*100:.1f}\\%"
            delta_str = f"{a['paired_delta_mean']*100:+.1f}pp"
            p_val = a['paired_wilcoxon_p']
            p_str = f"{p_val:.3f}" if p_val >= 0.001 else "$<$0.001"
            size_str = f"{a['mean_set_size']:.1f}"
            pct_str = f"{a['size_per_classes']*100:.0f}\\%"
        else:
            aci_str = delta_str = p_str = size_str = pct_str = '--'

        lines.append(f'{task_short} & {cat} & {std_str} & {aci_str} & {delta_str} & {p_str} & {size_str} & {pct_str} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')
    return '\n'.join(lines)


def generate_baseline_latex_table(results):
    """Generate pre-deployment vs post-hoc shift detection comparison."""
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Shift Detection: Pre-Deployment (SHAP) vs Post-Hoc (Entropy, ECE). '
                 r'SHAP concentration is computed on validation data \textit{before} deployment. '
                 r'Entropy and ECE changes require test-time observations.}\label{tab:shift_detection}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{@{}lcccc@{}}')
    lines.append(r'\toprule')
    lines.append(r'Task & Cat & SHAP Conc.$^\dagger$ & $\Delta$Entropy & $\Delta$ECE \\')
    lines.append(r'\midrule')

    # Load SHAP concentration from CSV
    shap_csv = Path(__file__).parent.parent / 'results/shap/concentration_all_tasks.csv'
    shap_data = {}
    if shap_csv.exists():
        df = pd.read_csv(shap_csv)
        for _, row in df.iterrows():
            shap_data[row['task']] = row['concentration_pct']

    for task_name in ALL_TASKS:
        if task_name not in results:
            continue
        r = results[task_name]
        task_short = task_name.replace('sales-', 's-').replace('item-', 'i-')
        cat = 'SEV' if r['standard']['test_coverage_mean'] < 0.5 else 'ROB'
        conc = shap_data.get(task_name, '--')
        conc_str = f"{conc:.1f}\\%" if isinstance(conc, (int, float)) else '--'
        ent_delta = r['entropy']['delta_mean']
        ece_delta = r['ece']['delta_mean']

        lines.append(f'{task_short} & {cat} & {conc_str} & {ent_delta:+.3f} & {ece_delta:+.4f} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\vspace{1mm}')
    lines.append(r'')
    lines.append(r'\raggedright')
    lines.append(r'\footnotesize')
    lines.append(r'$^\dagger$Pre-deployment diagnostic (computed on validation data only).')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


if __name__ == "__main__":
    main()
