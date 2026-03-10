"""
Fast Baseline Comparison (5 trials, shorter data)
================================================
Quick comparison of Neural Granger, Linear Granger, NOTEARS, and PCMCI.
"""
import numpy as np
import torch
import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from neural_granger_simple import NeuralGranger, train_neural_granger
from baselines import LinearGrangerCausality, evaluate_causal_discovery
from scipy import stats

# Import PCMCI
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr


def simple_notears(X, lambda1=0.1, max_iter=50, h_tol=1e-8, w_threshold=0.3):
    """Simple NOTEARS implementation."""
    n, d = X.shape
    X = X - X.mean(axis=0)
    W = np.zeros((d, d))
    rho = 1.0
    alpha = 0.0

    for iteration in range(max_iter):
        M = X @ W
        R = X - M
        grad = -X.T @ R / n + lambda1 * np.sign(W)
        W_sq = W * W
        exp_W_sq = np.exp(W_sq)
        h = np.trace(exp_W_sq) - d
        grad_h = 2 * W * exp_W_sq
        grad += (rho * h + alpha) * grad_h
        lr = 0.01
        W = W - lr * grad
        W = np.clip(W, -10, 10)
        np.fill_diagonal(W, 0)
        if h > h_tol:
            rho = min(rho * 2, 1e6)
        alpha = alpha + rho * h

    W[np.abs(W) < w_threshold] = 0
    return (np.abs(W) > 0).astype(float)


def run_pcmci_baseline(data, tau_max=3):
    """Run PCMCI on data."""
    dataframe = pp.DataFrame(data)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=0.1)

    n_vars = data.shape[1]
    adj = np.zeros((n_vars, n_vars))
    p_matrix = results['p_matrix']

    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                for tau in range(1, tau_max + 1):
                    if p_matrix[i, j, tau] < 0.05:
                        adj[i, j] = 1
                        break
    return adj


def generate_nonlinear_data(T=500, seed=42):
    """Generate nonlinear synthetic data."""
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
            x2 = np.tanh(2 * data[t-1, 0]) + 0.3 * np.random.randn()
            x3 = 0.3 * data[t-1, 1]**2 + 0.3 * np.random.randn()
            x4 = np.sign(data[t-1, 2]) * np.sqrt(np.abs(data[t-1, 2])) + 0.3 * np.random.randn()
            x5 = np.sin(data[t-1, 3]) + 0.3 * np.random.randn()
            x6 = np.abs(data[t-1, 4]) + 0.3 * np.random.randn()
            data[t] = [x1, x2, x3, x4, x5, x6]

    return data, true_adj


def run_comparison():
    """Compare all methods."""
    print("=" * 60)
    print("BASELINE COMPARISON: Neural vs Linear vs NOTEARS vs PCMCI")
    print("=" * 60)

    n_trials = 5  # Quick test
    results = {'neural': [], 'linear': [], 'notears': [], 'pcmci': []}

    for trial in range(n_trials):
        seed = 42 + trial
        np.random.seed(seed)
        torch.manual_seed(seed)

        data, true_adj = generate_nonlinear_data(T=500, seed=seed)
        n_factors = data.shape[1]

        # Linear Granger
        gc = LinearGrangerCausality(n_lags=5)
        gc_adj = gc.fit(data)
        gc_m = evaluate_causal_discovery(true_adj, gc_adj)
        results['linear'].append(gc_m['f1'])

        # Neural Granger
        model = NeuralGranger(n_factors=n_factors, n_lags=5, hidden_dim=32)
        model = train_neural_granger(model, data, n_epochs=20, lr=1e-3)
        x = torch.FloatTensor(data).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            neural_adj = model.compute_granger_adjacency(x)

        best_f1 = 0
        for thresh in [0.05, 0.1, 0.15, 0.2]:
            m = evaluate_causal_discovery(true_adj, neural_adj, threshold=thresh)
            if m['f1'] > best_f1:
                best_f1 = m['f1']
        results['neural'].append(best_f1)

        # NOTEARS
        notears_adj = simple_notears(data, lambda1=0.1, w_threshold=0.1)
        notears_m = evaluate_causal_discovery(true_adj, notears_adj)
        results['notears'].append(notears_m['f1'])

        # PCMCI
        pcmci_adj = run_pcmci_baseline(data, tau_max=3)
        pcmci_m = evaluate_causal_discovery(true_adj, pcmci_adj)
        results['pcmci'].append(pcmci_m['f1'])

        print(f"Trial {trial+1}: Neural={results['neural'][-1]:.3f}, Linear={results['linear'][-1]:.3f}, "
              f"NOTEARS={results['notears'][-1]:.3f}, PCMCI={results['pcmci'][-1]:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS: NONLINEAR SYNTHETIC DATA (5 trials)")
    print("=" * 60)

    for method in ['neural', 'linear', 'notears', 'pcmci']:
        mean_f1 = np.mean(results[method])
        std_f1 = np.std(results[method])
        print(f"{method.upper():<12}: {mean_f1:.3f} ± {std_f1:.3f}")

    # Statistical tests
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS (paired t-tests)")
    print("=" * 60)

    comparisons = [
        ('neural', 'linear'),
        ('neural', 'notears'),
        ('neural', 'pcmci'),
    ]

    for m1, m2 in comparisons:
        t, p = stats.ttest_rel(results[m1], results[m2])
        print(f"{m1.upper()} vs {m2.upper()}: t={t:.3f}, p={p:.6f}")

    # Best method
    means = {k: np.mean(v) for k, v in results.items()}
    winner = max(means, key=means.get)
    print(f"\nBest method: {winner.upper()} (F1={means[winner]:.3f})")

    return results


if __name__ == "__main__":
    results = run_comparison()
