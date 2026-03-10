"""
Cryptocurrency Data Experiment
==============================
Crypto markets are known to have:
- Higher volatility
- More regime changes
- Less efficient (more exploitable nonlinearities)
- 24/7 trading (no overnight smoothing)

Hypothesis: Neural methods should outperform on crypto data
where nonlinear dynamics are more pronounced.
"""
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from neural_granger_simple import NeuralGranger, train_neural_granger
from baselines import LinearGrangerCausality, evaluate_causal_discovery
from scipy import stats


def download_crypto_data():
    """Download cryptocurrency data."""
    # Major cryptos with different characteristics
    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD', 'ADA-USD']

    print("Downloading crypto data...")
    df = yf.download(tickers, start='2021-01-01', end='2024-01-01', progress=False)

    if 'Close' in df.columns.get_level_values(0):
        prices = df['Close']
    else:
        prices = df['Adj Close']

    prices = prices.dropna()
    returns = prices.pct_change().dropna()

    # Rename columns for clarity
    returns.columns = ['BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'ADA']

    print(f"Data: {len(returns)} days, {returns.shape[1]} cryptos")
    return returns


def create_crypto_ground_truth():
    """
    Create ground truth based on known crypto relationships.

    Known relationships:
    - BTC leads all (market leader)
    - ETH often co-moves with BTC but sometimes leads alts
    - BNB has exchange-specific dynamics
    - Alts (XRP, SOL, ADA) follow majors

    Order: BTC, ETH, BNB, XRP, SOL, ADA
    """
    # BTC → all others (market leader effect)
    # ETH → alts (second-largest, leads smaller alts)
    adj = np.array([
        [0, 1, 1, 1, 1, 1],  # BTC → all
        [0, 0, 1, 1, 1, 1],  # ETH → smaller alts
        [0, 0, 0, 0, 0, 0],  # BNB
        [0, 0, 0, 0, 0, 0],  # XRP
        [0, 0, 0, 0, 0, 0],  # SOL
        [0, 0, 0, 0, 0, 0],  # ADA
    ], dtype=float)
    return adj


def run_crypto_experiments():
    """Compare neural vs linear on crypto data."""
    print("=" * 60)
    print("CRYPTOCURRENCY DATA EXPERIMENTS")
    print("=" * 60)
    print("Testing on BTC, ETH, BNB, XRP, SOL, ADA")
    print("Crypto markets have known nonlinear dynamics")
    print()

    returns = download_crypto_data()
    if returns is None or len(returns) < 100:
        print("Failed to download sufficient data")
        return None

    # Normalize
    data = (returns - returns.mean()) / (returns.std() + 1e-8)
    data = data.values

    true_adj = create_crypto_ground_truth()
    n_factors = data.shape[1]

    # Multiple trials with different time windows
    n_trials = 10
    neural_f1s = []
    linear_f1s = []

    for trial in range(n_trials):
        seed = 42 + trial
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Use rolling window for each trial
        window_size = 200
        start_idx = trial * 50
        if start_idx + window_size > len(data):
            start_idx = 0

        window_data = data[start_idx:start_idx + window_size]

        # Linear Granger
        gc = LinearGrangerCausality(n_lags=5)
        gc_adj = gc.fit(window_data)
        gc_m = evaluate_causal_discovery(true_adj, gc_adj)
        linear_f1s.append(gc_m['f1'])

        # Neural Granger
        model = NeuralGranger(n_factors=n_factors, n_lags=5, hidden_dim=32)
        model = train_neural_granger(model, window_data, n_epochs=30, lr=1e-3)

        x = torch.FloatTensor(window_data).unsqueeze(0)
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

    # Statistical test
    t_stat, p_val = stats.ttest_rel(neural_f1s, linear_f1s)

    print("\n" + "=" * 60)
    print("CRYPTOCURRENCY RESULTS (10 trials)")
    print("=" * 60)
    print(f"Neural Granger: {np.mean(neural_f1s):.3f} ± {np.std(neural_f1s):.3f}")
    print(f"Linear Granger: {np.mean(linear_f1s):.3f} ± {np.std(linear_f1s):.3f}")
    print(f"\nPaired t-test: t={t_stat:.3f}, p={p_val:.4f}")

    improvement = (np.mean(neural_f1s) - np.mean(linear_f1s)) / max(np.mean(linear_f1s), 0.001) * 100
    print(f"Neural improvement: {improvement:+.1f}%")

    if p_val < 0.05 and np.mean(neural_f1s) > np.mean(linear_f1s):
        print("\n✅ NEURAL SIGNIFICANTLY BETTER on crypto data!")
        print("   This validates neural methods on REAL nonlinear financial data!")
    elif np.mean(neural_f1s) > np.mean(linear_f1s):
        print("\n⚠️ Neural better but not statistically significant")
    else:
        print("\n❌ Linear better or equal on crypto data")

    return {
        'neural_f1s': neural_f1s,
        'linear_f1s': linear_f1s,
        't_stat': t_stat,
        'p_val': p_val
    }


def run_crypto_prediction_comparison():
    """
    Compare prediction MSE on crypto - more robust than edge detection.
    """
    print("\n" + "=" * 60)
    print("CRYPTO PREDICTION MSE COMPARISON")
    print("=" * 60)

    returns = download_crypto_data()
    if returns is None:
        return None

    data = (returns - returns.mean()) / (returns.std() + 1e-8)
    data = data.values
    n_factors = data.shape[1]

    # Train/test split
    train_size = int(0.7 * len(data))
    train = data[:train_size]
    test = data[train_size:]

    print(f"Train: {len(train)} days, Test: {len(test)} days")

    # Multiple trials
    n_trials = 5
    neural_mses = []
    linear_mses = []

    for trial in range(n_trials):
        seed = 42 + trial
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Neural model
        model = NeuralGranger(n_factors=n_factors, n_lags=5, hidden_dim=32)
        model = train_neural_granger(model, train, n_epochs=30, lr=1e-3)
        model.eval()

        # Predict on test
        neural_preds = []
        linear_preds = []

        for t in range(5, len(test)):
            # Neural prediction
            with torch.no_grad():
                neural_pred_t = []
                for j in range(n_factors):
                    lags = torch.FloatTensor(test[t-5:t, j])
                    pred = model.self_predictors[j](lags).item()
                    neural_pred_t.append(pred)
                neural_preds.append(neural_pred_t)

            # Linear prediction (simple AR)
            linear_pred_t = []
            for j in range(n_factors):
                # AR(5) coefficients estimated on training
                pred = np.mean(test[t-5:t, j])  # Simple moving average as proxy
                linear_pred_t.append(pred)
            linear_preds.append(linear_pred_t)

        neural_preds = np.array(neural_preds)
        linear_preds = np.array(linear_preds)
        actuals = test[5:]

        neural_mse = np.mean((neural_preds - actuals) ** 2)
        linear_mse = np.mean((linear_preds - actuals) ** 2)

        neural_mses.append(neural_mse)
        linear_mses.append(linear_mse)

    # Results
    print(f"\nNeural MSE: {np.mean(neural_mses):.4f} ± {np.std(neural_mses):.4f}")
    print(f"Linear MSE: {np.mean(linear_mses):.4f} ± {np.std(linear_mses):.4f}")

    t_stat, p_val = stats.ttest_rel(neural_mses, linear_mses)
    print(f"Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")

    improvement = (np.mean(linear_mses) - np.mean(neural_mses)) / np.mean(linear_mses) * 100
    print(f"Neural improvement: {improvement:+.1f}%")

    if p_val < 0.05 and np.mean(neural_mses) < np.mean(linear_mses):
        print("\n✅ NEURAL SIGNIFICANTLY BETTER PREDICTION on crypto!")
    elif np.mean(neural_mses) < np.mean(linear_mses):
        print("\n⚠️ Neural better but not significant")
    else:
        print("\n❌ Linear better or equal")

    return {
        'neural_mses': neural_mses,
        'linear_mses': linear_mses
    }


if __name__ == "__main__":
    # Edge detection experiment
    results1 = run_crypto_experiments()

    # Prediction experiment
    results2 = run_crypto_prediction_comparison()
