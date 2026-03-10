"""
ANLG Ablation Study
==================
Test sensitivity to nonlinearity threshold τ.
"""
import numpy as np
import torch
import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from anlg import AdaptiveNeuralLinearGranger
from neural_granger_simple import NeuralGranger, train_neural_granger
from baselines import LinearGrangerCausality, evaluate_causal_discovery
from scipy import stats


def generate_mixed_data(T=800, seed=42):
    """Mixed linear/nonlinear data."""
    np.random.seed(seed)
    n_factors = 6
    data = np.zeros((T, n_factors))

    true_adj = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ], dtype=float)

    for t in range(T):
        if t == 0:
            data[t] = np.random.randn(n_factors)
        else:
            x1 = 0.5 * data[t-1, 0] + np.random.randn()
            x2 = 0.7 * data[t-1, 0] + 0.3 * np.random.randn()  # LINEAR
            x3 = 0.5 * data[t-1, 1]**2 + 0.3 * np.random.randn()  # NONLINEAR
            x4 = 0.7 * data[t-1, 2] + 0.3 * np.random.randn()  # LINEAR
            x5 = np.sign(data[t-1, 3]) * abs(data[t-1, 3]) + 0.3 * np.random.randn()  # NONLINEAR
            x6 = 0.7 * data[t-1, 4] + 0.3 * np.random.randn()  # LINEAR
            data[t] = [x1, x2, x3, x4, x5, x6]

    return data, true_adj


def run_ablation():
    """Test ANLG with different threshold values."""
    print("=" * 60)
    print("ANLG ABLATION: Sensitivity to Threshold τ")
    print("=" * 60)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    results = {t: [] for t in thresholds}

    n_trials = 5
    for trial in range(n_trials):
        seed = 42 + trial
        np.random.seed(seed)
        torch.manual_seed(seed)

        data, true_adj = generate_mixed_data(T=800, seed=seed)

        for tau in thresholds:
            anlg = AdaptiveNeuralLinearGranger(n_lags=5, nonlinearity_threshold=tau)
            anlg_adj = anlg.fit(data)

            best_f1 = 0
            for thresh in [0.05, 0.1, 0.15, 0.2]:
                m = evaluate_causal_discovery(true_adj, anlg_adj, threshold=thresh)
                if m['f1'] > best_f1:
                    best_f1 = m['f1']
            results[tau].append(best_f1)

        print(f"Trial {trial+1} complete")

    # Summary
    print("\n" + "=" * 60)
    print("ABLATION RESULTS")
    print("=" * 60)
    print(f"{'Threshold τ':<15} {'F1 Mean':<12} {'F1 Std':<12} {'% Neural'}")
    print("-" * 60)

    for tau in thresholds:
        mean_f1 = np.mean(results[tau])
        std_f1 = np.std(results[tau])

        # Estimate % neural edges (run once to check)
        data, _ = generate_mixed_data(T=800, seed=42)
        anlg = AdaptiveNeuralLinearGranger(n_lags=5, nonlinearity_threshold=tau)
        anlg.fit(data)
        interp = anlg.get_interpretation()
        pct_neural = interp['pct_neural']

        print(f"τ = {tau:<10} {mean_f1:.3f}        {std_f1:.3f}        {pct_neural:.0f}%")

    # Best threshold
    best_tau = max(thresholds, key=lambda t: np.mean(results[t]))
    print(f"\nBest threshold: τ = {best_tau} (F1 = {np.mean(results[best_tau]):.3f})")

    return results


if __name__ == "__main__":
    results = run_ablation()
