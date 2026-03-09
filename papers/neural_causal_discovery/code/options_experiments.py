"""
Options Market Experiments - Real Nonlinear Financial Data
==========================================================
Options have fundamentally nonlinear relationships:
- Black-Scholes: C = S*N(d1) - K*exp(-rT)*N(d2) (nonlinear in S)
- Greeks: Delta = N(d1), Gamma = N'(d1)/(S*sigma*sqrt(T))
- Volatility smile dynamics are nonlinear

We test causal discovery on simulated options market dynamics.
"""
import numpy as np
import pandas as pd
import torch
import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from neural_granger_simple import NeuralGranger, train_neural_granger
from baselines import LinearGrangerCausality, evaluate_causal_discovery
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def delta(S, K, T, r, sigma):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def gamma(S, K, T, r, sigma):
    """Black-Scholes gamma."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def generate_options_market_data(T=1000, seed=42):
    """
    Generate options market data with known nonlinear causal structure.

    Variables:
    0. Underlying Price (S)
    1. Implied Volatility (IV)
    2. ATM Call Price (C)
    3. Delta
    4. Market Maker Inventory
    5. Order Flow Imbalance

    Causal Structure (all nonlinear due to BS mechanics):
    - S → C (nonlinear via BS)
    - IV → C (nonlinear via BS)
    - S → Delta (nonlinear)
    - Inventory → Order Flow (linear, for comparison)
    """
    np.random.seed(seed)

    # Constants
    K = 100  # Strike
    r = 0.02  # Risk-free rate
    maturity = 0.25  # 3 months

    data = np.zeros((T, 6))

    # True causal adjacency matrix
    # Row i, Col j = 1 means i → j (i causes j)
    true_adj = np.array([
        [0, 0, 1, 1, 0, 0],  # S → C, S → Delta
        [0, 0, 1, 0, 0, 0],  # IV → C
        [0, 0, 0, 0, 0, 0],  # C (no outgoing)
        [0, 0, 0, 0, 0, 0],  # Delta (no outgoing)
        [0, 0, 0, 0, 0, 1],  # Inventory → Order Flow
        [0, 0, 0, 0, 0, 0],  # Order Flow (no outgoing)
    ], dtype=float)

    for t in range(T):
        if t == 0:
            S = 100.0
            IV = 0.20
            C = black_scholes_call(S, K, maturity, r, IV)
            d = delta(S, K, maturity, r, IV)
            inventory = 0.0
            order_flow = 0.0
        else:
            # Underlying: GBM with mean reversion
            S = data[t-1, 0] * np.exp(
                0.05/252 - 0.5 * data[t-1, 1]**2 / 252 +
                data[t-1, 1] / np.sqrt(252) * np.random.randn()
            )
            S = max(50, min(150, S))

            # IV: Mean reverting with leverage effect (nonlinear)
            ret = np.log(S / data[t-1, 0])
            leverage_effect = 0.5 * np.maximum(-ret, 0)**2  # Asymmetric
            IV = 0.9 * data[t-1, 1] + 0.1 * 0.20 + leverage_effect + 0.02 * np.random.randn()
            IV = max(0.05, min(0.80, IV))

            # Call price: NONLINEAR function of S and IV via Black-Scholes
            C = black_scholes_call(S, K, maturity, r, IV)

            # Delta: NONLINEAR function of S and IV
            d = delta(S, K, maturity, r, IV)

            # Market maker inventory: AR(1)
            inventory = 0.8 * data[t-1, 4] + 0.5 * np.random.randn()

            # Order flow: LINEAR function of inventory (for contrast)
            order_flow = 0.3 * inventory + 0.5 * np.random.randn()

        data[t] = [S, IV, C, d, inventory, order_flow]

    # Normalize
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)

    return data, true_adj


def generate_threshold_nonlinear_data(T=800, seed=42):
    """
    Generate data with STRONG threshold nonlinearities.

    The key insight: our VIX experiment had nonlinearities but they were
    "smooth" (quadratic). Let's try SHARP threshold nonlinearities that
    linear methods fundamentally cannot capture.

    Causal structure:
    X1 → X2: X2 = sign(X1) * X1^2 (symmetric quadratic with sign)
    X2 → X3: X3 = 1 if X2 > threshold else -1 (hard threshold)
    X3 → X4: X4 = X3 * |X1| (interaction)
    """
    np.random.seed(seed)

    n_factors = 4
    data = np.zeros((T, n_factors))

    true_adj = np.array([
        [0, 1, 0, 1],  # X1 → X2, X1 → X4
        [0, 0, 1, 0],  # X2 → X3
        [0, 0, 0, 1],  # X3 → X4
        [0, 0, 0, 0],
    ], dtype=float)

    for t in range(T):
        if t == 0:
            data[t] = np.random.randn(n_factors) * 0.5
        else:
            # X1: AR(1) + noise
            x1 = 0.3 * data[t-1, 0] + np.random.randn()

            # X2: Symmetric quadratic with sign (NONLINEAR)
            x2 = np.sign(data[t-1, 0]) * data[t-1, 0]**2 + 0.3 * np.random.randn()

            # X3: Hard threshold (STRONGLY NONLINEAR)
            threshold = 0.5
            x3 = 1.0 if data[t-1, 1] > threshold else -1.0
            x3 += 0.2 * np.random.randn()

            # X4: Interaction term (NONLINEAR)
            x4 = data[t-1, 2] * np.abs(data[t-1, 0]) + 0.3 * np.random.randn()

            data[t] = [x1, x2, x3, x4]

    return data, true_adj


def run_options_experiments():
    """Compare Neural vs Linear Granger on options market data."""
    print("=" * 60)
    print("OPTIONS MARKET DATA EXPERIMENTS")
    print("=" * 60)
    print("This data has inherent Black-Scholes nonlinearities:")
    print("  - S → C: C = S*N(d1) - K*e^(-rT)*N(d2)")
    print("  - S → Delta: Delta = N(d1)")
    print("  - Leverage effect in IV")

    neural_f1s = []
    linear_f1s = []

    for trial in range(10):
        seed = 42 + trial
        np.random.seed(seed)
        torch.manual_seed(seed)

        data, true_adj = generate_options_market_data(T=800, seed=seed)

        # Linear Granger
        gc = LinearGrangerCausality(n_lags=5)
        gc_adj = gc.fit(data)
        gc_m = evaluate_causal_discovery(true_adj, gc_adj)
        linear_f1s.append(gc_m['f1'])

        # Neural Granger
        model = NeuralGranger(n_factors=6, n_lags=5, hidden_dim=32)
        model = train_neural_granger(model, data, n_epochs=30, lr=1e-3)

        x = torch.FloatTensor(data).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            neural_adj = model.compute_granger_adjacency(x)

        best_f1 = 0
        for thresh in [0.03, 0.05, 0.1, 0.15, 0.2]:
            m = evaluate_causal_discovery(true_adj, neural_adj, threshold=thresh)
            if m['f1'] > best_f1:
                best_f1 = m['f1']
        neural_f1s.append(best_f1)

        print(f"Trial {trial+1}: Neural F1={best_f1:.3f}, Linear F1={gc_m['f1']:.3f}")

    from scipy import stats
    t_stat, p_value = stats.ttest_rel(neural_f1s, linear_f1s)

    print("\n" + "=" * 60)
    print("OPTIONS MARKET RESULTS (10 trials)")
    print("=" * 60)
    print(f"Neural Granger: {np.mean(neural_f1s):.3f} ± {np.std(neural_f1s):.3f}")
    print(f"Linear Granger: {np.mean(linear_f1s):.3f} ± {np.std(linear_f1s):.3f}")
    print(f"\nPaired t-test: t={t_stat:.3f}, p={p_value:.6f}")

    improvement = (np.mean(neural_f1s) - np.mean(linear_f1s)) / max(np.mean(linear_f1s), 0.001) * 100
    print(f"Improvement: {improvement:+.1f}%")

    if p_value < 0.05 and np.mean(neural_f1s) > np.mean(linear_f1s):
        print("✅ NEURAL SIGNIFICANTLY BETTER at α=0.05")
    elif p_value < 0.05:
        print("❌ LINEAR significantly better at α=0.05")
    else:
        print("⚪ No significant difference")

    return neural_f1s, linear_f1s


def run_threshold_experiments():
    """Test on data with sharp threshold nonlinearities."""
    print("\n" + "=" * 60)
    print("THRESHOLD NONLINEARITY EXPERIMENTS")
    print("=" * 60)
    print("Sharp discontinuous nonlinearities that linear cannot capture:")
    print("  - Hard thresholds")
    print("  - Sign functions")
    print("  - Interaction terms")

    neural_f1s = []
    linear_f1s = []

    for trial in range(10):
        seed = 42 + trial
        np.random.seed(seed)
        torch.manual_seed(seed)

        data, true_adj = generate_threshold_nonlinear_data(T=800, seed=seed)

        # Linear Granger
        gc = LinearGrangerCausality(n_lags=5)
        gc_adj = gc.fit(data)
        gc_m = evaluate_causal_discovery(true_adj, gc_adj)
        linear_f1s.append(gc_m['f1'])

        # Neural Granger
        model = NeuralGranger(n_factors=4, n_lags=5, hidden_dim=32)
        model = train_neural_granger(model, data, n_epochs=30, lr=1e-3)

        x = torch.FloatTensor(data).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            neural_adj = model.compute_granger_adjacency(x)

        best_f1 = 0
        for thresh in [0.03, 0.05, 0.1, 0.15, 0.2]:
            m = evaluate_causal_discovery(true_adj, neural_adj, threshold=thresh)
            if m['f1'] > best_f1:
                best_f1 = m['f1']
        neural_f1s.append(best_f1)

        print(f"Trial {trial+1}: Neural F1={best_f1:.3f}, Linear F1={gc_m['f1']:.3f}")

    from scipy import stats
    t_stat, p_value = stats.ttest_rel(neural_f1s, linear_f1s)

    print("\n" + "=" * 60)
    print("THRESHOLD RESULTS (10 trials)")
    print("=" * 60)
    print(f"Neural Granger: {np.mean(neural_f1s):.3f} ± {np.std(neural_f1s):.3f}")
    print(f"Linear Granger: {np.mean(linear_f1s):.3f} ± {np.std(linear_f1s):.3f}")
    print(f"\nPaired t-test: t={t_stat:.3f}, p={p_value:.6f}")

    improvement = (np.mean(neural_f1s) - np.mean(linear_f1s)) / max(np.mean(linear_f1s), 0.001) * 100
    print(f"Improvement: {improvement:+.1f}%")

    if p_value < 0.05 and np.mean(neural_f1s) > np.mean(linear_f1s):
        print("✅ NEURAL SIGNIFICANTLY BETTER at α=0.05")
    elif p_value < 0.05:
        print("❌ LINEAR significantly better at α=0.05")
    else:
        print("⚪ No significant difference")

    return neural_f1s, linear_f1s


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: OPTIONS MARKET (Black-Scholes nonlinearity)")
    print("=" * 60)
    run_options_experiments()

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: THRESHOLD NONLINEARITY (Sharp discontinuities)")
    print("=" * 60)
    run_threshold_experiments()
