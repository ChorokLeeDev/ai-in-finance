"""
Real Financial Threshold Data: Prediction-Based Comparison
==========================================================
Compare neural vs linear Granger on PREDICTION ACCURACY
around circuit breaker events.

Key insight: We don't need ground truth causality.
Instead, we measure: Does neural predict better during threshold periods?

If threshold nonlinearities matter, neural should show LARGER
improvement over linear during circuit breaker periods.
"""
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from neural_granger_simple import NeuralGranger, train_neural_granger
from baselines import LinearGrangerCausality
from scipy import stats


def download_data():
    """Download ETF data around March 2020 circuit breaker events."""
    tickers = ['SPY', 'QQQ', 'IWM', 'XLF', 'TLT', 'GLD']
    start = '2019-06-01'  # More data for training
    end = '2020-12-31'

    print(f"Downloading {tickers}...")
    df = yf.download(tickers, start=start, end=end, progress=False)

    if 'Close' in df.columns.get_level_values(0):
        prices = df['Close']
    else:
        prices = df['Adj Close']

    prices = prices.dropna()
    returns = prices.pct_change().dropna()

    print(f"Data: {len(returns)} days, {returns.shape[1]} assets")
    return returns


def compute_prediction_mse(model_type, train_data, test_data, n_lags=5):
    """
    Train model on train_data, compute MSE on test_data.
    Returns MSE for each target variable.
    """
    n_factors = train_data.shape[1]

    if model_type == 'linear':
        # Linear VAR prediction
        from statsmodels.tsa.api import VAR
        try:
            model = VAR(train_data)
            results = model.fit(n_lags)

            # Predict on test data
            preds = []
            for t in range(n_lags, len(test_data)):
                history = test_data[t-n_lags:t]
                pred = results.forecast(history, steps=1)[0]
                preds.append(pred)

            preds = np.array(preds)
            actuals = test_data[n_lags:]
            mse = np.mean((preds - actuals) ** 2)
            return mse
        except Exception as e:
            print(f"Linear VAR failed: {e}")
            return None

    elif model_type == 'neural':
        # Neural Granger prediction
        np.random.seed(42)
        torch.manual_seed(42)

        model = NeuralGranger(n_factors=n_factors, n_lags=n_lags, hidden_dim=32)
        model = train_neural_granger(model, train_data, n_epochs=30, lr=1e-3)

        # Predict on test data
        model.eval()
        x = torch.FloatTensor(test_data).unsqueeze(0)

        with torch.no_grad():
            # Simple forward pass for prediction
            seq_len = x.shape[1]
            preds = []

            for t in range(n_lags, seq_len):
                # Get lagged inputs
                pred_t = []
                for j in range(n_factors):
                    lags = x[0, t-n_lags:t, j]
                    pred_j = model.self_predictors[j](lags).item()
                    pred_t.append(pred_j)
                preds.append(pred_t)

            preds = np.array(preds)
            actuals = test_data[n_lags:]
            mse = np.mean((preds - actuals) ** 2)
            return mse

    return None


def run_prediction_comparison():
    """
    Compare prediction accuracy during threshold vs normal periods.
    """
    print("=" * 60)
    print("PREDICTION-BASED COMPARISON ON REAL CIRCUIT BREAKER DATA")
    print("=" * 60)

    returns = download_data()
    if returns is None:
        return None

    # Normalize
    returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)
    data = returns_norm.values

    # Define periods
    # Pre-crisis: 2019-06-01 to 2020-02-15 (training)
    # Circuit breaker period: 2020-02-15 to 2020-04-15 (threshold test)
    # Post-crisis: 2020-04-15 to 2020-08-15 (normal test)

    dates = returns_norm.index

    # Find indices
    train_end = dates.get_indexer([pd.to_datetime('2020-02-15')], method='nearest')[0]
    threshold_end = dates.get_indexer([pd.to_datetime('2020-04-15')], method='nearest')[0]
    normal_end = dates.get_indexer([pd.to_datetime('2020-08-15')], method='nearest')[0]

    train_data = data[:train_end]
    threshold_data = data[train_end:threshold_end]
    normal_data = data[threshold_end:normal_end]

    print(f"\nTrain period: {len(train_data)} days (pre-crisis)")
    print(f"Threshold period: {len(threshold_data)} days (circuit breakers)")
    print(f"Normal period: {len(normal_data)} days (post-crisis)")

    results = {}

    # Threshold period
    print("\n" + "=" * 60)
    print("THRESHOLD PERIOD (Circuit Breakers Feb-Apr 2020)")
    print("=" * 60)

    linear_mse_thresh = compute_prediction_mse('linear', train_data, threshold_data)
    print(f"Linear VAR MSE: {linear_mse_thresh:.6f}")

    neural_mse_thresh = compute_prediction_mse('neural', train_data, threshold_data)
    print(f"Neural MSE: {neural_mse_thresh:.6f}")

    if linear_mse_thresh and neural_mse_thresh:
        thresh_improvement = (linear_mse_thresh - neural_mse_thresh) / linear_mse_thresh * 100
        print(f"Neural improvement: {thresh_improvement:+.1f}%")
        results['threshold'] = {
            'linear_mse': linear_mse_thresh,
            'neural_mse': neural_mse_thresh,
            'improvement': thresh_improvement
        }

    # Normal period
    print("\n" + "=" * 60)
    print("NORMAL PERIOD (Post-Crisis Apr-Aug 2020)")
    print("=" * 60)

    linear_mse_normal = compute_prediction_mse('linear', train_data, normal_data)
    print(f"Linear VAR MSE: {linear_mse_normal:.6f}")

    neural_mse_normal = compute_prediction_mse('neural', train_data, normal_data)
    print(f"Neural MSE: {neural_mse_normal:.6f}")

    if linear_mse_normal and neural_mse_normal:
        normal_improvement = (linear_mse_normal - neural_mse_normal) / linear_mse_normal * 100
        print(f"Neural improvement: {normal_improvement:+.1f}%")
        results['normal'] = {
            'linear_mse': linear_mse_normal,
            'neural_mse': neural_mse_normal,
            'improvement': normal_improvement
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: REAL FINANCIAL DATA COMPARISON")
    print("=" * 60)

    if 'threshold' in results and 'normal' in results:
        print(f"\nThreshold Period (Circuit Breakers):")
        print(f"  Linear MSE: {results['threshold']['linear_mse']:.6f}")
        print(f"  Neural MSE: {results['threshold']['neural_mse']:.6f}")
        print(f"  Neural improvement: {results['threshold']['improvement']:+.1f}%")

        print(f"\nNormal Period:")
        print(f"  Linear MSE: {results['normal']['linear_mse']:.6f}")
        print(f"  Neural MSE: {results['normal']['neural_mse']:.6f}")
        print(f"  Neural improvement: {results['normal']['improvement']:+.1f}%")

        diff = results['threshold']['improvement'] - results['normal']['improvement']
        print(f"\nDifference in Neural Advantage: {diff:+.1f}%")

        if diff > 5:
            print("\n✅ Neural methods show LARGER advantage during threshold period!")
            print("   This validates the threshold hypothesis on REAL financial data!")
        elif results['threshold']['improvement'] > 0:
            print("\n⚠️  Neural better in threshold period, but advantage not clearly larger")
        else:
            print("\n❌ Threshold hypothesis not validated on real data")

    return results


if __name__ == "__main__":
    results = run_prediction_comparison()
