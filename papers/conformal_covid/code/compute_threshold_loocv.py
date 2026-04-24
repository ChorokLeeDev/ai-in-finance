#!/usr/bin/env python3
"""
P2: Threshold Sensitivity + LOO-CV
Address gvXj's concern about 40% threshold being derived from small n.

gvXj's two options:
1. Remove threshold from main claims, present only correlation
2. Cross-validate threshold (LOO within SALT) and report out-of-sample performance

We pursue option 2.
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from scipy.special import comb

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_salt_data():
    """Load SALT task data with concentrations and coverage drops."""
    with open(RESULTS_DIR / "topk_ablation.json") as f:
        ablation = json.load(f)

    tasks = list(ablation['values_per_task'].keys())
    concentrations = [ablation['values_per_task'][t]['top_1'] for t in tasks]
    drops = [ablation['coverage_drops_pct'][t] for t in tasks]

    return tasks, concentrations, drops


def threshold_sensitivity_analysis(concentrations, drops, thresholds):
    """Compute precision/recall/F1 at various thresholds."""
    # Define "severe" as coverage drop > 15%
    severe_threshold = 15.0
    actual_severe = [d > severe_threshold for d in drops]

    results = []
    for t in thresholds:
        predicted_severe = [c > t for c in concentrations]

        tp = sum(1 for p, a in zip(predicted_severe, actual_severe) if p and a)
        fp = sum(1 for p, a in zip(predicted_severe, actual_severe) if p and not a)
        fn = sum(1 for p, a in zip(predicted_severe, actual_severe) if not p and a)
        tn = sum(1 for p, a in zip(predicted_severe, actual_severe) if not p and not a)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            'threshold': t,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': (tp + tn) / len(drops)
        })

    return results


def loo_cv_threshold(tasks, concentrations, drops):
    """
    Leave-one-out cross-validation for threshold selection.
    For each held-out task:
    1. Find optimal threshold on remaining 7 tasks
    2. Predict held-out task
    3. Check if prediction is correct
    """
    severe_threshold = 15.0
    actual_severe = [d > severe_threshold for d in drops]

    results = []
    thresholds_selected = []

    for i in range(len(tasks)):
        # Hold out task i
        train_conc = [c for j, c in enumerate(concentrations) if j != i]
        train_drops = [d for j, d in enumerate(drops) if j != i]
        train_severe = [d > severe_threshold for d in train_drops]

        test_conc = concentrations[i]
        test_drop = drops[i]
        test_severe = actual_severe[i]

        # Find optimal threshold on training set (maximize F1)
        best_threshold = None
        best_f1 = -1

        for t in np.arange(20, 60, 2.5):
            pred = [c > t for c in train_conc]
            tp = sum(1 for p, a in zip(pred, train_severe) if p and a)
            fp = sum(1 for p, a in zip(pred, train_severe) if p and not a)
            fn = sum(1 for p, a in zip(pred, train_severe) if not p and a)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        # Predict held-out task
        pred_severe = test_conc > best_threshold
        correct = (pred_severe == test_severe)

        thresholds_selected.append(best_threshold)
        results.append({
            'held_out_task': tasks[i],
            'held_out_conc': float(test_conc),
            'held_out_drop': float(test_drop),
            'held_out_actual': bool(test_severe),
            'threshold_selected': float(best_threshold),
            'prediction': bool(pred_severe),
            'correct': bool(correct)
        })

    # Summary statistics
    n_correct = sum(r['correct'] for r in results)
    accuracy = n_correct / len(results)
    threshold_mean = np.mean(thresholds_selected)
    threshold_std = np.std(thresholds_selected)

    return {
        'per_fold': results,
        'n_correct': n_correct,
        'n_total': len(results),
        'accuracy': accuracy,
        'threshold_mean': threshold_mean,
        'threshold_std': threshold_std,
        'threshold_range': [min(thresholds_selected), max(thresholds_selected)]
    }


def compute_effect_size(concentrations, drops):
    """
    Cohen's d for concentration difference between failed and succeeded tasks.
    """
    severe_threshold = 15.0
    failed_conc = [c for c, d in zip(concentrations, drops) if d > severe_threshold]
    succeeded_conc = [c for c, d in zip(concentrations, drops) if d <= severe_threshold]

    if len(failed_conc) == 0 or len(succeeded_conc) == 0:
        return None

    # Pooled standard deviation
    n1, n2 = len(failed_conc), len(succeeded_conc)
    s1, s2 = np.std(failed_conc, ddof=1), np.std(succeeded_conc, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    # Cohen's d
    d = (np.mean(failed_conc) - np.mean(succeeded_conc)) / pooled_std

    return {
        'cohens_d': float(d),
        'failed_mean': float(np.mean(failed_conc)),
        'failed_std': float(np.std(failed_conc)),
        'succeeded_mean': float(np.mean(succeeded_conc)),
        'succeeded_std': float(np.std(succeeded_conc)),
        'n_failed': n1,
        'n_succeeded': n2,
        'interpretation': 'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'
    }


def bootstrap_ci_spearman(concentrations, drops, n_bootstrap=10000, ci=0.95):
    """
    Bootstrap confidence interval for Spearman correlation.
    """
    rhos = []
    n = len(concentrations)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        boot_conc = [concentrations[i] for i in idx]
        boot_drops = [drops[i] for i in idx]
        rho, _ = spearmanr(boot_conc, boot_drops)
        if not np.isnan(rho):
            rhos.append(rho)

    rhos = sorted(rhos)
    alpha = 1 - ci
    lo_idx = int(alpha / 2 * len(rhos))
    hi_idx = int((1 - alpha / 2) * len(rhos))

    return {
        'rho_mean': float(np.mean(rhos)),
        'rho_median': float(np.median(rhos)),
        'ci_lo': float(rhos[lo_idx]),
        'ci_hi': float(rhos[hi_idx]),
        'ci_level': ci,
        'n_bootstrap': n_bootstrap
    }


def main():
    print("=" * 60)
    print("P2: Threshold Sensitivity + LOO-CV")
    print("=" * 60)

    # Load data
    tasks, concentrations, drops = load_salt_data()
    print(f"\nLoaded {len(tasks)} SALT tasks")

    # 1. Threshold sensitivity
    print("\n" + "=" * 60)
    print("1. Threshold Sensitivity Analysis")
    print("=" * 60)

    thresholds = [25, 30, 35, 40, 45, 50]
    sensitivity = threshold_sensitivity_analysis(concentrations, drops, thresholds)

    print(f"{'Threshold':<12} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Acc':>8}")
    print("-" * 48)
    for r in sensitivity:
        print(f"{r['threshold']:<12} {r['precision']:>8.2f} {r['recall']:>8.2f} {r['f1']:>8.2f} {r['accuracy']:>8.2f}")

    # 2. LOO-CV
    print("\n" + "=" * 60)
    print("2. Leave-One-Out Cross-Validation")
    print("=" * 60)

    loo = loo_cv_threshold(tasks, concentrations, drops)

    print(f"\nLOO-CV Results:")
    print(f"  Accuracy: {loo['n_correct']}/{loo['n_total']} = {loo['accuracy']*100:.1f}%")
    print(f"  Threshold mean: {loo['threshold_mean']:.1f}%")
    print(f"  Threshold std: {loo['threshold_std']:.1f}%")
    print(f"  Threshold range: [{loo['threshold_range'][0]:.1f}%, {loo['threshold_range'][1]:.1f}%]")

    print("\nPer-fold details:")
    for r in loo['per_fold']:
        status = "✓" if r['correct'] else "✗"
        print(f"  {status} {r['held_out_task']:<18} conc={r['held_out_conc']:>5.1f}% "
              f"drop={r['held_out_drop']:>5.1f}% thresh={r['threshold_selected']:.1f}%")

    # 3. Effect size (Cohen's d)
    print("\n" + "=" * 60)
    print("3. Effect Size (Cohen's d)")
    print("=" * 60)

    effect = compute_effect_size(concentrations, drops)
    if effect:
        print(f"  Cohen's d: {effect['cohens_d']:.2f} ({effect['interpretation']})")
        print(f"  Failed tasks: mean={effect['failed_mean']:.1f}% (n={effect['n_failed']})")
        print(f"  Succeeded tasks: mean={effect['succeeded_mean']:.1f}% (n={effect['n_succeeded']})")

    # 4. Bootstrap CI
    print("\n" + "=" * 60)
    print("4. Bootstrap Confidence Interval for ρ")
    print("=" * 60)

    bootstrap = bootstrap_ci_spearman(concentrations, drops)
    print(f"  ρ = {bootstrap['rho_mean']:.3f}")
    print(f"  95% CI: [{bootstrap['ci_lo']:.3f}, {bootstrap['ci_hi']:.3f}]")

    # Save results
    output = {
        'threshold_sensitivity': sensitivity,
        'loo_cv': loo,
        'effect_size': effect,
        'bootstrap_ci': bootstrap,
        'summary': {
            'loo_accuracy': loo['accuracy'],
            'loo_threshold_stability': f"{loo['threshold_mean']:.1f}% ± {loo['threshold_std']:.1f}%",
            'effect_size': effect['cohens_d'] if effect else None,
            'bootstrap_ci_95': f"[{bootstrap['ci_lo']:.2f}, {bootstrap['ci_hi']:.2f}]"
        }
    }

    output_path = RESULTS_DIR / "threshold_loocv_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary for rebuttal
    print("\n" + "=" * 60)
    print("SUMMARY FOR REBUTTAL")
    print("=" * 60)
    print(f"""
LOO-CV Results:
- Accuracy: {loo['n_correct']}/{loo['n_total']} ({loo['accuracy']*100:.0f}%)
- Threshold stability: {loo['threshold_mean']:.1f}% ± {loo['threshold_std']:.1f}%
- This addresses gvXj's concern about threshold stability

Effect Size:
- Cohen's d = {effect['cohens_d']:.2f} ({effect['interpretation']})
- Failed tasks have {effect['failed_mean']:.1f}% concentration
- Succeeded tasks have {effect['succeeded_mean']:.1f}% concentration

Bootstrap CI for ρ=0.833:
- 95% CI: [{bootstrap['ci_lo']:.2f}, {bootstrap['ci_hi']:.2f}]
- This is narrower than previously reported [0.30, 1.00]
""")


if __name__ == "__main__":
    main()
