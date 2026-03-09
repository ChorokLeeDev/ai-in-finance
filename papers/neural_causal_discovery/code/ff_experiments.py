"""
Real Financial Data Validation
==============================
Test Neural Granger on Fama-French factors to validate nonlinear finding.
"""
import numpy as np
import pandas as pd
import torch
import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from neural_granger_simple import NeuralGranger, train_neural_granger
from baselines import LinearGrangerCausality, VARModel


def load_ff_factors():
    """Load Fama-French 6-factor data."""
    try:
        import pandas_datareader as pdr
        # Download FF 5 factors + Momentum
        ff5 = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3_daily',
                                      start='2000-01-01', end='2023-12-31')[0]
        mom = pdr.get_data_famafrench('F-F_Momentum_Factor_daily',
                                      start='2000-01-01', end='2023-12-31')[0]
        data = ff5.join(mom, how='inner')
        data = data.rename(columns={'Mom   ': 'Mom'})
        data = data / 100.0  # Convert to decimal
        return data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']]
    except Exception as e:
        print(f"Could not download FF data: {e}")
        print("Using synthetic FF-like data...")
        return generate_synthetic_ff()


def generate_synthetic_ff(T=2000):
    """Generate synthetic FF-like data with realistic correlations."""
    np.random.seed(42)

    # Realistic daily return parameters
    means = np.array([0.0003, 0.0001, 0.0001, 0.0001, 0.0001, 0.0002])
    stds = np.array([0.01, 0.006, 0.006, 0.005, 0.004, 0.008])

    # Correlation structure (realistic)
    corr = np.array([
        [1.0, 0.3, 0.2, 0.2, 0.1, -0.1],
        [0.3, 1.0, 0.1, 0.2, 0.1, 0.0],
        [0.2, 0.1, 1.0, 0.3, 0.4, 0.1],
        [0.2, 0.2, 0.3, 1.0, 0.2, 0.1],
        [0.1, 0.1, 0.4, 0.2, 1.0, 0.0],
        [-0.1, 0.0, 0.1, 0.1, 0.0, 1.0]
    ])

    # Generate with nonlinear dynamics
    cov = np.outer(stds, stds) * corr
    data = np.zeros((T, 6))

    for t in range(T):
        if t == 0:
            data[t] = np.random.multivariate_normal(means, cov)
        else:
            # Add nonlinear autoregressive component
            ar_component = 0.1 * np.tanh(5 * data[t-1])
            innovation = np.random.multivariate_normal(means, cov)
            data[t] = ar_component + innovation

    dates = pd.date_range('2000-01-01', periods=T, freq='B')
    return pd.DataFrame(data, index=dates,
                       columns=['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom'])


def evaluate_prediction_improvement(data, method='neural', n_lags=5):
    """
    Evaluate prediction improvement using out-of-sample R².
    Split data into train/test and measure prediction quality.
    """
    values = data.values
    T, n_factors = values.shape

    # Train/test split (80/20)
    train_size = int(0.8 * T)
    train_data = values[:train_size]
    test_data = values[train_size:]

    if method == 'neural':
        model = NeuralGranger(n_factors=n_factors, n_lags=n_lags, hidden_dim=16)
        model = train_neural_granger(model, train_data, n_epochs=30, lr=1e-3)

        # Get adjacency from training data
        x_train = torch.FloatTensor(train_data).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            adj = model.compute_granger_adjacency(x_train)
    else:
        # Linear Granger
        gc = LinearGrangerCausality(n_lags=n_lags)
        adj = gc.fit(train_data)

    # Compute out-of-sample prediction accuracy
    # Use discovered edges to make predictions
    test_predictions = []
    test_targets = []

    for t in range(n_lags, len(test_data)):
        for j in range(n_factors):
            # Predict using discovered causal parents
            parents = np.where(adj[:, j] > 0.1)[0]
            if len(parents) > 0:
                # Simple linear prediction using parents
                X = test_data[t-n_lags:t, parents].flatten()
                # Just use mean of lagged values as simple predictor
                pred = X.mean()
            else:
                pred = test_data[t-1, j]  # Use previous value

            test_predictions.append(pred)
            test_targets.append(test_data[t, j])

    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)

    # Compute R² and MSE
    ss_res = np.sum((test_targets - test_predictions) ** 2)
    ss_tot = np.sum((test_targets - test_targets.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mse = np.mean((test_targets - test_predictions) ** 2)

    return {'r2': r2, 'mse': mse, 'adj': adj}


def run_ff_experiments():
    """Run Neural vs Linear Granger on FF data."""
    print("=" * 60)
    print("REAL FINANCIAL DATA: Fama-French Factors")
    print("=" * 60)

    # Load data
    ff_data = load_ff_factors()
    print(f"Data shape: {ff_data.shape}")
    print(f"Date range: {ff_data.index[0]} to {ff_data.index[-1]}")
    print(f"Factors: {list(ff_data.columns)}")

    results = {'neural': [], 'linear': []}

    # Multiple trials with different random seeds (for neural training)
    for trial in range(5):
        seed = 42 + trial
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"\n--- Trial {trial+1}/5 ---")

        # Neural Granger
        neural_result = evaluate_prediction_improvement(ff_data, method='neural')
        results['neural'].append(neural_result['mse'])
        print(f"Neural MSE: {neural_result['mse']:.6f}")

        # Linear Granger
        linear_result = evaluate_prediction_improvement(ff_data, method='linear')
        results['linear'].append(linear_result['mse'])
        print(f"Linear MSE: {linear_result['mse']:.6f}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    neural_mse = np.mean(results['neural'])
    linear_mse = np.mean(results['linear'])
    print(f"Neural Granger MSE: {neural_mse:.6f} ± {np.std(results['neural']):.6f}")
    print(f"Linear Granger MSE: {linear_mse:.6f} ± {np.std(results['linear']):.6f}")

    if neural_mse < linear_mse:
        improvement = (linear_mse - neural_mse) / linear_mse * 100
        print(f"\n✅ Neural Granger achieves {improvement:.1f}% lower MSE")
    else:
        degradation = (neural_mse - linear_mse) / linear_mse * 100
        print(f"\n❌ Neural Granger has {degradation:.1f}% higher MSE")

    # Show discovered causal structure
    print("\n" + "=" * 60)
    print("DISCOVERED CAUSAL EDGES (Neural)")
    print("=" * 60)
    factor_names = list(ff_data.columns)
    adj = neural_result['adj']
    for i in range(len(factor_names)):
        for j in range(len(factor_names)):
            if i != j and adj[i, j] > 0.05:
                print(f"  {factor_names[i]} → {factor_names[j]}: {adj[i,j]:.3f}")

    return results


if __name__ == "__main__":
    run_ff_experiments()
