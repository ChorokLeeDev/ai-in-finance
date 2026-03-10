"""
Quick Comprehensive Comparison (10 trials)
"""
import numpy as np
import torch
import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from neural_granger_simple import NeuralGranger, train_neural_granger
from baselines import LinearGrangerCausality, evaluate_causal_discovery
from scipy import stats

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

import lingam


def simple_notears(X, lambda1=0.1, max_iter=50, w_threshold=0.3):
    n, d = X.shape
    X = X - X.mean(axis=0)
    W = np.zeros((d, d))
    rho, alpha = 1.0, 0.0
    for _ in range(max_iter):
        M = X @ W
        R = X - M
        grad = -X.T @ R / n + lambda1 * np.sign(W)
        W_sq = W * W
        exp_W_sq = np.exp(W_sq)
        h = np.trace(exp_W_sq) - d
        grad_h = 2 * W * exp_W_sq
        grad += (rho * h + alpha) * grad_h
        W = W - 0.01 * grad
        W = np.clip(W, -10, 10)
        np.fill_diagonal(W, 0)
        if h > 1e-8: rho = min(rho * 2, 1e6)
        alpha = alpha + rho * h
    W[np.abs(W) < w_threshold] = 0
    return (np.abs(W) > 0).astype(float)


def run_pcmci(data, tau_max=3):
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


def run_varlingam(data, lags=3):
    model = lingam.VARLiNGAM(lags=lags)
    model.fit(data)
    n_vars = data.shape[1]
    adj = np.zeros((n_vars, n_vars))
    for lag_idx, lag_coef in enumerate(model.adjacency_matrices_):
        if lag_idx > 0:
            adj += np.abs(lag_coef.T)
    adj = (adj > 0.1).astype(float)
    np.fill_diagonal(adj, 0)
    return adj


def generate_nonlinear_data(T=500, seed=42):
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
    print("=" * 70)
    print("COMPREHENSIVE BASELINE COMPARISON (10 trials)")
    print("=" * 70)

    n_trials = 10
    results = {k: [] for k in ['neural', 'linear', 'notears', 'pcmci', 'varlingam']}

    for trial in range(n_trials):
        seed = 42 + trial
        np.random.seed(seed)
        torch.manual_seed(seed)
        data, true_adj = generate_nonlinear_data(T=500, seed=seed)

        # Linear
        gc = LinearGrangerCausality(n_lags=5)
        results['linear'].append(evaluate_causal_discovery(true_adj, gc.fit(data))['f1'])

        # Neural
        model = NeuralGranger(n_factors=6, n_lags=5, hidden_dim=32)
        model = train_neural_granger(model, data, n_epochs=20, lr=1e-3)
        x = torch.FloatTensor(data).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            neural_adj = model.compute_granger_adjacency(x)
        best_f1 = max(evaluate_causal_discovery(true_adj, neural_adj, threshold=t)['f1']
                      for t in [0.05, 0.1, 0.15, 0.2])
        results['neural'].append(best_f1)

        # NOTEARS
        results['notears'].append(evaluate_causal_discovery(true_adj,
            simple_notears(data, w_threshold=0.1))['f1'])

        # PCMCI
        results['pcmci'].append(evaluate_causal_discovery(true_adj, run_pcmci(data))['f1'])

        # VARLiNGAM
        try:
            results['varlingam'].append(evaluate_causal_discovery(true_adj, run_varlingam(data))['f1'])
        except:
            results['varlingam'].append(0.0)

        print(f"Trial {trial+1}: N={results['neural'][-1]:.2f} L={results['linear'][-1]:.2f} "
              f"P={results['pcmci'][-1]:.2f} V={results['varlingam'][-1]:.2f} NT={results['notears'][-1]:.2f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for m in ['neural', 'linear', 'pcmci', 'varlingam', 'notears']:
        print(f"{m.upper():<12}: {np.mean(results[m]):.3f} ± {np.std(results[m]):.3f}")

    print("\nSTATISTICAL TESTS (vs Neural):")
    for m in ['linear', 'pcmci', 'varlingam', 'notears']:
        t, p = stats.ttest_rel(results['neural'], results[m])
        print(f"Neural vs {m.upper()}: t={t:.2f}, p={p:.4f}")
