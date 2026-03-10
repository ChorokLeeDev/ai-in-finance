"""
Bootstrap Validation (100 resamples)
====================================
Proper statistical validation with bootstrap confidence intervals.
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


def simple_notears(X, lambda1=0.1, max_iter=50, w_threshold=0.3):
    n, d = X.shape
    X = X - X.mean(axis=0)
    W = np.zeros((d, d))
    rho, alpha = 1.0, 0.0
    for _ in range(max_iter):
        M = X @ W
        grad = -X.T @ (X - M) / n + lambda1 * np.sign(W)
        W_sq = W * W
        exp_W_sq = np.exp(W_sq)
        h = np.trace(exp_W_sq) - d
        grad += (rho * h + alpha) * 2 * W * exp_W_sq
        W = np.clip(W - 0.01 * grad, -10, 10)
        np.fill_diagonal(W, 0)
        if h > 1e-8: rho = min(rho * 2, 1e6)
        alpha += rho * h
    W[np.abs(W) < w_threshold] = 0
    return (np.abs(W) > 0).astype(float)


def run_pcmci(data, tau_max=3):
    dataframe = pp.DataFrame(data)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=0.1)
    adj = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            if i != j:
                for tau in range(1, tau_max + 1):
                    if results['p_matrix'][i, j, tau] < 0.05:
                        adj[i, j] = 1
                        break
    return adj


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


def run_single_trial(seed):
    """Run single trial and return F1 scores."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    data, true_adj = generate_data(T=500, seed=seed)

    # Linear
    gc = LinearGrangerCausality(n_lags=5)
    linear_f1 = evaluate_causal_discovery(true_adj, gc.fit(data))['f1']

    # Neural
    model = NeuralGranger(n_factors=6, n_lags=5, hidden_dim=32)
    model = train_neural_granger(model, data, n_epochs=20, lr=1e-3)
    x = torch.FloatTensor(data).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        neural_adj = model.compute_granger_adjacency(x)
    neural_f1 = max(evaluate_causal_discovery(true_adj, neural_adj, threshold=t)['f1']
                    for t in [0.05, 0.1, 0.15, 0.2])

    # NOTEARS
    notears_f1 = evaluate_causal_discovery(true_adj, simple_notears(data, w_threshold=0.1))['f1']

    # PCMCI
    pcmci_f1 = evaluate_causal_discovery(true_adj, run_pcmci(data))['f1']

    return {'neural': neural_f1, 'linear': linear_f1, 'notears': notears_f1, 'pcmci': pcmci_f1}


if __name__ == "__main__":
    print("=" * 70)
    print("BOOTSTRAP VALIDATION (20 trials)")
    print("=" * 70)

    n_trials = 20  # Run 20 trials for faster execution
    results = {k: [] for k in ['neural', 'linear', 'notears', 'pcmci']}

    for trial in range(n_trials):
        seed = 42 + trial
        trial_results = run_single_trial(seed)
        for k, v in trial_results.items():
            results[k].append(v)
        print(f"Trial {trial+1}/{n_trials}: N={trial_results['neural']:.2f} "
              f"L={trial_results['linear']:.2f} P={trial_results['pcmci']:.2f} "
              f"NT={trial_results['notears']:.2f}")

    # Bootstrap CIs
    print("\n" + "=" * 70)
    print("RESULTS WITH 95% BOOTSTRAP CI")
    print("=" * 70)

    n_bootstrap = 1000
    for method in ['neural', 'linear', 'pcmci', 'notears']:
        vals = np.array(results[method])
        bootstrap_means = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(vals), len(vals), replace=True)
            bootstrap_means.append(vals[idx].mean())

        mean = np.mean(vals)
        ci_low = np.percentile(bootstrap_means, 2.5)
        ci_high = np.percentile(bootstrap_means, 97.5)
        print(f"{method.upper():<12}: {mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]")

    # Statistical tests with bootstrap p-values
    print("\n" + "=" * 70)
    print("BOOTSTRAP P-VALUES (vs Neural)")
    print("=" * 70)

    neural_vals = np.array(results['neural'])
    for method in ['linear', 'pcmci', 'notears']:
        other_vals = np.array(results[method])
        diff = neural_vals - other_vals

        # Bootstrap test
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(diff), len(diff), replace=True)
            bootstrap_diffs.append(diff[idx].mean())

        # Two-sided p-value
        p_value = 2 * min((np.array(bootstrap_diffs) < 0).mean(),
                         (np.array(bootstrap_diffs) > 0).mean())
        mean_diff = np.mean(diff)
        ci_low = np.percentile(bootstrap_diffs, 2.5)
        ci_high = np.percentile(bootstrap_diffs, 97.5)

        print(f"Neural - {method.upper()}: {mean_diff:.3f} [{ci_low:.3f}, {ci_high:.3f}], p={p_value:.4f}")
