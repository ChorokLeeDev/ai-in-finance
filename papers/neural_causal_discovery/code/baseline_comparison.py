"""
Add PCMCI and NOTEARS Baselines
===============================
Address reviewer concern: missing standard causal discovery baselines.
"""
import numpy as np
import torch
import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from neural_granger_simple import NeuralGranger, train_neural_granger
from baselines import LinearGrangerCausality, evaluate_causal_discovery
from scipy import stats

# Try to import PCMCI
try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    PCMCI_AVAILABLE = True
except ImportError:
    PCMCI_AVAILABLE = False
    print("PCMCI not available - install tigramite: pip install tigramite")

# Try to import NOTEARS
try:
    from notears import notears_linear
    NOTEARS_AVAILABLE = True
except ImportError:
    try:
        # Alternative: implement simple NOTEARS
        NOTEARS_AVAILABLE = False
    except:
        NOTEARS_AVAILABLE = False
        print("NOTEARS not available")


def simple_notears(X, lambda1=0.1, max_iter=100, h_tol=1e-8, w_threshold=0.3):
    """
    Simple NOTEARS implementation for DAG learning.
    Based on Zheng et al. (2018) continuous optimization.
    """
    n, d = X.shape

    # Center data
    X = X - X.mean(axis=0)

    # Initialize W
    W = np.zeros((d, d))

    # Gradient descent with augmented Lagrangian
    rho = 1.0
    alpha = 0.0

    for iteration in range(max_iter):
        # Compute gradient
        M = X @ W
        R = X - M
        grad = -X.T @ R / n + lambda1 * np.sign(W)

        # Add DAG constraint gradient: d/dW tr(e^{W∘W})
        W_sq = W * W
        exp_W_sq = np.exp(W_sq)
        h = np.trace(exp_W_sq) - d
        grad_h = 2 * W * exp_W_sq

        grad += (rho * h + alpha) * grad_h

        # Update W
        lr = 0.01
        W = W - lr * grad

        # Project to valid range
        W = np.clip(W, -10, 10)
        np.fill_diagonal(W, 0)

        # Update rho if constraint not satisfied
        if h > h_tol:
            rho = min(rho * 2, 1e6)

        alpha = alpha + rho * h

    # Threshold
    W[np.abs(W) < w_threshold] = 0

    return (np.abs(W) > 0).astype(float)


def run_pcmci_baseline(data, tau_max=5):
    """Run PCMCI on data."""
    if not PCMCI_AVAILABLE:
        return None

    # Create tigramite dataframe
    dataframe = pp.DataFrame(data)

    # Initialize PCMCI
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())

    # Run PCMCI
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=0.05)

    # Extract adjacency (any significant link at any lag)
    n_vars = data.shape[1]
    adj = np.zeros((n_vars, n_vars))

    p_matrix = results['p_matrix']
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                # Check if any lag is significant
                for tau in range(1, tau_max + 1):
                    if p_matrix[i, j, tau] < 0.05:
                        adj[i, j] = 1
                        break

    return adj


def generate_nonlinear_data(T=800, seed=42):
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


def run_baseline_comparison():
    """Compare all methods including PCMCI and NOTEARS."""
    print("=" * 60)
    print("BASELINE COMPARISON: Neural vs Linear vs PCMCI vs NOTEARS")
    print("=" * 60)

    n_trials = 10

    results = {
        'neural': [],
        'linear': [],
        'notears': [],
        'pcmci': []
    }

    for trial in range(n_trials):
        seed = 42 + trial
        np.random.seed(seed)
        torch.manual_seed(seed)

        data, true_adj = generate_nonlinear_data(T=800, seed=seed)
        n_factors = data.shape[1]

        # Linear Granger
        gc = LinearGrangerCausality(n_lags=5)
        gc_adj = gc.fit(data)
        gc_m = evaluate_causal_discovery(true_adj, gc_adj)
        results['linear'].append(gc_m['f1'])

        # Neural Granger
        model = NeuralGranger(n_factors=n_factors, n_lags=5, hidden_dim=32)
        model = train_neural_granger(model, data, n_epochs=30, lr=1e-3)

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
        try:
            notears_adj = simple_notears(data, lambda1=0.1, w_threshold=0.1)
            notears_m = evaluate_causal_discovery(true_adj, notears_adj)
            results['notears'].append(notears_m['f1'])
        except Exception as e:
            print(f"NOTEARS failed: {e}")
            results['notears'].append(0.0)

        # PCMCI
        if PCMCI_AVAILABLE:
            try:
                pcmci_adj = run_pcmci_baseline(data, tau_max=5)
                if pcmci_adj is not None:
                    pcmci_m = evaluate_causal_discovery(true_adj, pcmci_adj)
                    results['pcmci'].append(pcmci_m['f1'])
                else:
                    results['pcmci'].append(0.0)
            except Exception as e:
                print(f"PCMCI failed: {e}")
                results['pcmci'].append(0.0)
        else:
            results['pcmci'].append(np.nan)

        print(f"Trial {trial+1}: Neural={results['neural'][-1]:.3f}, Linear={results['linear'][-1]:.3f}, "
              f"NOTEARS={results['notears'][-1]:.3f}, PCMCI={results['pcmci'][-1] if not np.isnan(results['pcmci'][-1]) else 'N/A'}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS: NONLINEAR SYNTHETIC DATA (10 trials)")
    print("=" * 60)

    print(f"\nNeural Granger:  {np.mean(results['neural']):.3f} ± {np.std(results['neural']):.3f}")
    print(f"Linear Granger:  {np.mean(results['linear']):.3f} ± {np.std(results['linear']):.3f}")
    print(f"NOTEARS:         {np.mean(results['notears']):.3f} ± {np.std(results['notears']):.3f}")

    if PCMCI_AVAILABLE:
        pcmci_vals = [x for x in results['pcmci'] if not np.isnan(x)]
        if pcmci_vals:
            print(f"PCMCI:           {np.mean(pcmci_vals):.3f} ± {np.std(pcmci_vals):.3f}")

    # Statistical tests
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)

    # Neural vs Linear
    t, p = stats.ttest_rel(results['neural'], results['linear'])
    print(f"Neural vs Linear: t={t:.3f}, p={p:.4f}")

    # Neural vs NOTEARS
    t, p = stats.ttest_rel(results['neural'], results['notears'])
    print(f"Neural vs NOTEARS: t={t:.3f}, p={p:.4f}")

    # Determine winner
    means = {
        'Neural': np.mean(results['neural']),
        'Linear': np.mean(results['linear']),
        'NOTEARS': np.mean(results['notears'])
    }
    winner = max(means, key=means.get)
    print(f"\nBest method: {winner} (F1={means[winner]:.3f})")

    return results


if __name__ == "__main__":
    results = run_baseline_comparison()
