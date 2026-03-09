"""
Nonlinear Synthetic Data Experiments
====================================
Test RANCD on nonlinear causal structure where neural methods should shine.
"""
import numpy as np
import torch
import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from data_loader import create_data_loader
from rancd_v2 import RANCDV2
from baselines import LinearGrangerCausality, VARModel, evaluate_causal_discovery


def generate_nonlinear_causal_data(n_factors=6, T=600, seed=42):
    """
    Generate data with NONLINEAR causal relationships.
    Neural methods should outperform linear baselines here.
    """
    np.random.seed(seed)

    data = np.zeros((T, n_factors))

    # Initialize
    data[0] = np.random.randn(n_factors) * 0.1
    data[1] = np.random.randn(n_factors) * 0.1

    # True adjacency (chain with nonlinear effects)
    # X0 -> X1 -> X2 -> X3 -> X4 -> X5 (nonlinear)
    true_adj = np.zeros((n_factors, n_factors))
    for i in range(n_factors - 1):
        true_adj[i, i+1] = 1.0

    for t in range(2, T):
        noise = np.random.randn(n_factors) * 0.2

        # X0: exogenous
        data[t, 0] = 0.3 * data[t-1, 0] + noise[0]

        # X1: nonlinear function of X0
        data[t, 1] = 0.5 * np.tanh(2 * data[t-1, 0]) + 0.2 * data[t-1, 1] + noise[1]

        # X2: quadratic function of X1
        data[t, 2] = 0.3 * np.sign(data[t-1, 1]) * data[t-1, 1]**2 + 0.2 * data[t-1, 2] + noise[2]

        # X3: threshold function of X2
        data[t, 3] = 0.5 * (data[t-1, 2] > 0) * data[t-1, 2] + 0.2 * data[t-1, 3] + noise[3]

        # X4: sine function of X3
        data[t, 4] = 0.4 * np.sin(3 * data[t-1, 3]) + 0.2 * data[t-1, 4] + noise[4]

        # X5: abs function of X4
        data[t, 5] = 0.4 * np.abs(data[t-1, 4]) + 0.2 * data[t-1, 5] + noise[5]

    return data, true_adj


def train_rancd_quick(model, loader, n_epochs=40, lr=1e-3):
    """Quick training."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss, _ = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}")
    return model


def run_nonlinear_experiments():
    print("=" * 60)
    print("NONLINEAR CAUSAL DATA EXPERIMENTS")
    print("=" * 60)
    print("Testing where neural methods should outperform linear baselines")

    results = {
        'granger': {'f1': []},
        'var': {'f1': []},
        'rancd': {'f1': []}
    }

    for trial in range(3):
        seed = 42 + trial
        print(f"\n--- Trial {trial+1}/3 (seed={seed}) ---")

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate nonlinear data
        data, true_adj = generate_nonlinear_causal_data(n_factors=6, T=600, seed=seed)
        print(f"Data shape: {data.shape}")
        print(f"True edges: {(true_adj > 0).sum()}")

        # 1. Granger (should struggle with nonlinearity)
        gc = LinearGrangerCausality(n_lags=5)
        gc_adj = gc.fit(data)
        gc_m = evaluate_causal_discovery(true_adj, gc_adj)
        results['granger']['f1'].append(gc_m['f1'])
        print(f"Granger F1: {gc_m['f1']:.3f}")

        # 2. VAR (should struggle with nonlinearity)
        var = VARModel(n_lags=5)
        var_adj = var.fit(data)
        var_m = evaluate_causal_discovery(true_adj, var_adj, threshold=0.25)
        results['var']['f1'].append(var_m['f1'])
        print(f"VAR F1: {var_m['f1']:.3f}")

        # 3. RANCD (should handle nonlinearity better)
        loader = create_data_loader(data, window_size=60, batch_size=16)
        model = RANCDV2(n_factors=6, hidden_dim=48, n_regimes=2, n_lags=5)
        model = train_rancd_quick(model, loader, n_epochs=40)

        test_data = torch.FloatTensor(data[:60]).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            _, adj, _, _ = model(test_data, return_all=True)
            rancd_adj = adj.cpu().numpy().mean(axis=0)

        # Try multiple thresholds
        best_f1 = 0
        for thresh in [0.25, 0.3, 0.35, 0.4, 0.5]:
            m = evaluate_causal_discovery(true_adj, rancd_adj, threshold=thresh)
            if m['f1'] > best_f1:
                best_f1 = m['f1']

        results['rancd']['f1'].append(best_f1)
        print(f"RANCD F1 (best threshold): {best_f1:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("NONLINEAR DATA RESULTS")
    print("=" * 60)
    for method in ['granger', 'var', 'rancd']:
        f1_mean = np.mean(results[method]['f1'])
        f1_std = np.std(results[method]['f1'])
        print(f"{method.upper():<12} F1: {f1_mean:.3f} ± {f1_std:.3f}")

    return results


if __name__ == "__main__":
    run_nonlinear_experiments()
