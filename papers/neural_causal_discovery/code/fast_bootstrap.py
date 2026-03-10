"""
Fast Bootstrap Validation (Neural vs Linear, 20 trials)
"""
import numpy as np
import torch
import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from neural_granger_simple import NeuralGranger, train_neural_granger
from baselines import LinearGrangerCausality, evaluate_causal_discovery


def generate_data(T=500, seed=42):
    np.random.seed(seed)
    data = np.zeros((T, 6))
    true_adj = np.array([
        [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0],
        [0,0,0,0,1,0], [0,0,0,0,0,1], [0,0,0,0,0,0]
    ], dtype=float)
    for t in range(T):
        if t == 0:
            data[t] = np.random.randn(6)
        else:
            data[t] = [
                0.5 * data[t-1,0] + np.random.randn(),
                np.tanh(2 * data[t-1,0]) + 0.3 * np.random.randn(),
                0.3 * data[t-1,1]**2 + 0.3 * np.random.randn(),
                np.sign(data[t-1,2]) * np.sqrt(np.abs(data[t-1,2])) + 0.3 * np.random.randn(),
                np.sin(data[t-1,3]) + 0.3 * np.random.randn(),
                np.abs(data[t-1,4]) + 0.3 * np.random.randn()
            ]
    return data, true_adj


if __name__ == "__main__":
    print("=" * 60)
    print("FAST BOOTSTRAP VALIDATION (Neural vs Linear, 20 trials)")
    print("=" * 60)

    n_trials = 20
    neural_f1s, linear_f1s = [], []

    for trial in range(n_trials):
        seed = 42 + trial
        np.random.seed(seed)
        torch.manual_seed(seed)
        data, true_adj = generate_data(T=500, seed=seed)

        # Linear
        gc = LinearGrangerCausality(n_lags=5)
        linear_f1 = evaluate_causal_discovery(true_adj, gc.fit(data))['f1']
        linear_f1s.append(linear_f1)

        # Neural
        model = NeuralGranger(n_factors=6, n_lags=5, hidden_dim=32)
        model = train_neural_granger(model, data, n_epochs=15, lr=1e-3)
        x = torch.FloatTensor(data).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            neural_adj = model.compute_granger_adjacency(x)
        neural_f1 = max(evaluate_causal_discovery(true_adj, neural_adj, threshold=t)['f1']
                        for t in [0.05, 0.1, 0.15, 0.2])
        neural_f1s.append(neural_f1)

        print(f"Trial {trial+1}: Neural={neural_f1:.3f}, Linear={linear_f1:.3f}")

    # Bootstrap CIs
    print("\n" + "=" * 60)
    print("RESULTS WITH 95% BOOTSTRAP CI (1000 resamples)")
    print("=" * 60)

    n_bootstrap = 1000
    neural_vals = np.array(neural_f1s)
    linear_vals = np.array(linear_f1s)

    # Neural CI
    neural_boot = [np.random.choice(neural_vals, len(neural_vals), replace=True).mean()
                   for _ in range(n_bootstrap)]
    print(f"Neural: {neural_vals.mean():.3f} [{np.percentile(neural_boot, 2.5):.3f}, {np.percentile(neural_boot, 97.5):.3f}]")

    # Linear CI
    linear_boot = [np.random.choice(linear_vals, len(linear_vals), replace=True).mean()
                   for _ in range(n_bootstrap)]
    print(f"Linear: {linear_vals.mean():.3f} [{np.percentile(linear_boot, 2.5):.3f}, {np.percentile(linear_boot, 97.5):.3f}]")

    # Difference
    diff = neural_vals - linear_vals
    diff_boot = [np.random.choice(diff, len(diff), replace=True).mean()
                 for _ in range(n_bootstrap)]
    p_value = 2 * min((np.array(diff_boot) < 0).mean(), (np.array(diff_boot) > 0).mean())
    print(f"\nNeural - Linear: {diff.mean():.3f} [{np.percentile(diff_boot, 2.5):.3f}, {np.percentile(diff_boot, 97.5):.3f}]")
    print(f"Bootstrap p-value: {p_value:.6f}")
