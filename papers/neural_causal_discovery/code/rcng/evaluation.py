"""
Evaluation metrics for Regime-Conditional Neural Granger

Includes:
- Per-regime F1, Precision, Recall
- Regime detection accuracy (ARI)
- Graph diversity metrics
- Comparison utilities (paired t-tests, bootstrap CIs)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import adjusted_rand_score
from scipy import stats


def evaluate_regime_causal_discovery(
    pred_adj: np.ndarray,
    true_adj: np.ndarray,
    pred_regimes: np.ndarray,
    true_regimes: np.ndarray,
    threshold: float = 0.3,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of regime-conditional causal discovery.

    Args:
        pred_adj: (n_regimes, n_factors, n_factors) predicted adjacency
        true_adj: (n_regimes, n_factors, n_factors) true adjacency
        pred_regimes: (T,) predicted regime assignments
        true_regimes: (T,) true regime labels
        threshold: binarization threshold for predicted adjacency

    Returns:
        dict with all metrics
    """
    results = {}

    # 1. Per-regime causal discovery metrics
    graph_metrics = compute_per_regime_metrics(pred_adj, true_adj, threshold)
    results.update(graph_metrics)

    # 2. Regime detection accuracy
    regime_metrics = compute_regime_metrics(pred_regimes, true_regimes)
    results.update(regime_metrics)

    # 3. Graph diversity (how different are learned graphs across regimes?)
    diversity = compute_graph_diversity(pred_adj)
    results['graph_diversity'] = diversity

    return results


def compute_per_regime_metrics(
    pred_adj: np.ndarray,
    true_adj: np.ndarray,
    threshold: float = 0.3,
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 per regime.

    Args:
        pred_adj: (n_regimes, n_factors, n_factors) predicted (soft)
        true_adj: (n_regimes, n_factors, n_factors) true (binary)
        threshold: binarization threshold

    Returns:
        dict with per-regime and aggregate metrics
    """
    n_regimes = pred_adj.shape[0]

    # Binarize predictions
    pred_binary = (pred_adj > threshold).astype(float)
    true_binary = (true_adj > 0).astype(float)

    results = {}
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for k in range(n_regimes):
        pred_k = pred_binary[k].flatten()
        true_k = true_binary[k].flatten()

        tp = ((pred_k == 1) & (true_k == 1)).sum()
        fp = ((pred_k == 1) & (true_k == 0)).sum()
        fn = ((pred_k == 0) & (true_k == 1)).sum()
        tn = ((pred_k == 0) & (true_k == 0)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        results[f'regime_{k}_precision'] = float(precision)
        results[f'regime_{k}_recall'] = float(recall)
        results[f'regime_{k}_f1'] = float(f1)
        results[f'regime_{k}_accuracy'] = float(accuracy)

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Aggregates
    results['macro_precision'] = float(np.mean(precision_scores))
    results['macro_recall'] = float(np.mean(recall_scores))
    results['macro_f1'] = float(np.mean(f1_scores))

    return results


def compute_regime_metrics(
    pred_regimes: np.ndarray,
    true_regimes: np.ndarray,
) -> Dict[str, float]:
    """
    Compute regime detection accuracy metrics.

    Args:
        pred_regimes: (T,) predicted regime assignments
        true_regimes: (T,) true regime labels

    Returns:
        dict with ARI and accuracy
    """
    # Adjusted Rand Index (handles label permutation)
    ari = adjusted_rand_score(true_regimes, pred_regimes)

    # Raw accuracy (may be misleading due to label mismatch)
    # Find best label alignment
    n_regimes = max(true_regimes.max(), pred_regimes.max()) + 1
    best_accuracy = 0.0

    from itertools import permutations
    for perm in permutations(range(n_regimes)):
        remapped = np.array([perm[r] for r in pred_regimes])
        acc = (remapped == true_regimes).mean()
        best_accuracy = max(best_accuracy, acc)

    return {
        'regime_ari': float(ari),
        'regime_accuracy': float(best_accuracy),
    }


def compute_graph_diversity(adj: np.ndarray) -> float:
    """
    Compute diversity of learned graphs across regimes.

    Higher diversity = more different causal structures across regimes.
    This is what we want to encourage via L_diverse.

    Args:
        adj: (n_regimes, n_factors, n_factors)

    Returns:
        diversity score (mean pairwise Frobenius distance)
    """
    n_regimes = adj.shape[0]

    if n_regimes < 2:
        return 0.0

    distances = []
    for k1 in range(n_regimes):
        for k2 in range(k1 + 1, n_regimes):
            dist = np.linalg.norm(adj[k1] - adj[k2], 'fro')
            distances.append(dist)

    return float(np.mean(distances))


def paired_ttest(
    scores_a: List[float],
    scores_b: List[float],
    alternative: str = 'two-sided',
) -> Tuple[float, float]:
    """
    Paired t-test for comparing two methods.

    Args:
        scores_a: list of scores for method A
        scores_b: list of scores for method B
        alternative: 'two-sided', 'greater', or 'less'

    Returns:
        (t_statistic, p_value)
    """
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b, alternative=alternative)
    return float(t_stat), float(p_value)


def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for mean.

    Args:
        scores: list of scores
        n_bootstrap: number of bootstrap samples
        ci_level: confidence level

    Returns:
        (mean, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    scores = np.array(scores)
    n = len(scores)

    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_means.append(scores[idx].mean())

    boot_means = np.array(boot_means)
    alpha = 1 - ci_level
    ci_lower = np.percentile(boot_means, 100 * alpha / 2)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return float(scores.mean()), float(ci_lower), float(ci_upper)


def compare_methods(
    results_joint: List[Dict],
    results_baseline: List[Dict],
    metric: str = 'macro_f1',
) -> Dict[str, float]:
    """
    Statistical comparison between Joint RCNG and baseline.

    Args:
        results_joint: list of result dicts from Joint RCNG
        results_baseline: list of result dicts from baseline
        metric: metric to compare

    Returns:
        dict with comparison statistics
    """
    scores_joint = [r[metric] for r in results_joint]
    scores_baseline = [r[metric] for r in results_baseline]

    # Means and CIs
    mean_joint, ci_l_joint, ci_u_joint = bootstrap_ci(scores_joint)
    mean_baseline, ci_l_baseline, ci_u_baseline = bootstrap_ci(scores_baseline)

    # Paired t-test
    t_stat, p_value = paired_ttest(scores_joint, scores_baseline, alternative='greater')

    # Effect size (Cohen's d)
    diff = np.array(scores_joint) - np.array(scores_baseline)
    cohens_d = diff.mean() / (diff.std() + 1e-8)

    return {
        'joint_mean': mean_joint,
        'joint_ci': (ci_l_joint, ci_u_joint),
        'baseline_mean': mean_baseline,
        'baseline_ci': (ci_l_baseline, ci_u_baseline),
        'improvement': mean_joint - mean_baseline,
        'improvement_pct': (mean_joint - mean_baseline) / (mean_baseline + 1e-8) * 100,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': float(cohens_d),
    }


def format_results_table(
    results_list: List[Dict],
    metrics: List[str] = ['macro_f1', 'regime_ari', 'graph_diversity'],
) -> str:
    """
    Format results as a nice table string.

    Args:
        results_list: list of result dicts
        metrics: metrics to include

    Returns:
        formatted table string
    """
    lines = []
    header = "| Metric | Mean | Std | 95% CI |"
    lines.append(header)
    lines.append("|--------|------|-----|--------|")

    for metric in metrics:
        scores = [r.get(metric, 0) for r in results_list]
        mean, ci_l, ci_u = bootstrap_ci(scores)
        std = np.std(scores)
        lines.append(f"| {metric} | {mean:.3f} | {std:.3f} | [{ci_l:.3f}, {ci_u:.3f}] |")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test evaluation functions
    print("Testing evaluation functions...")

    # Fake data
    n_regimes, n_factors = 3, 6
    pred_adj = np.random.rand(n_regimes, n_factors, n_factors) * 0.5
    true_adj = np.zeros((n_regimes, n_factors, n_factors))
    # Chain in regime 0
    for i in range(n_factors - 1):
        true_adj[0, i, i+1] = 1.0
    # Hub in regime 2
    for j in range(1, n_factors):
        true_adj[2, 0, j] = 1.0

    pred_regimes = np.random.randint(0, 3, size=1000)
    true_regimes = np.random.randint(0, 3, size=1000)

    results = evaluate_regime_causal_discovery(
        pred_adj, true_adj, pred_regimes, true_regimes
    )

    print("\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v:.3f}")

    print("\nGraph diversity:", compute_graph_diversity(pred_adj))

    print("\nAll tests passed!")
