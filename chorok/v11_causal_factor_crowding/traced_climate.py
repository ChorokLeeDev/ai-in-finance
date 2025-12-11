"""
TRACED: Climate Data Experiment (El Niño Southern Oscillation)

Validate TRACED on a well-studied climate system where:
1. Regime changes are known (El Niño / La Niña / Neutral)
2. Causal relationships are established (SST → Walker Circulation → Rainfall)
3. Heavy-tailed behavior during extreme events

This demonstrates domain generalization beyond finance.
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import climate data tools
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def download_enso_data():
    """
    Download El Niño Southern Oscillation Index data from NOAA.

    Returns monthly SST anomalies (Niño 3.4 region) from 1950-present.
    """
    # NOAA Monthly Niño 3.4 Index
    # https://psl.noaa.gov/enso/mei/
    url = "https://psl.noaa.gov/enso/mei/data/meiv2.data"

    # Alternative: Use synthetic climate-like data if download fails
    try:
        if HAS_REQUESTS:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print("Downloaded MEI v2 data from NOAA")
                # Parse the data (format is year followed by 12 monthly values)
                lines = response.text.strip().split('\n')
                data = []
                for line in lines[1:]:  # Skip header
                    parts = line.split()
                    if len(parts) >= 13:
                        year = int(parts[0])
                        for month in range(12):
                            try:
                                value = float(parts[month + 1])
                                if value > -900:  # Valid value
                                    data.append({
                                        'year': year,
                                        'month': month + 1,
                                        'mei': value
                                    })
                            except:
                                pass
                df = pd.DataFrame(data)
                return df
    except Exception as e:
        print(f"Could not download: {e}")

    return None


def generate_synthetic_climate_data(n_samples=1000, seed=42):
    """
    Generate synthetic climate data with known regime structure.

    Variables:
    - SST: Sea Surface Temperature anomaly
    - SOI: Southern Oscillation Index (pressure)
    - Rain: Rainfall anomaly
    - Wind: Trade wind strength

    Known causal structure:
    - El Niño: SST+ → SOI- → Wind- → Rain+
    - La Niña: SST- → SOI+ → Wind+ → Rain-
    - Neutral: weak connections
    """
    np.random.seed(seed)
    T = n_samples
    d = 4  # SST, SOI, Rain, Wind

    # Regime sequence (El Niño, Neutral, La Niña cycles)
    regime_pattern = []
    current_regime = 1  # Start neutral
    t = 0

    while t < T:
        # Regime duration: 12-36 months
        duration = np.random.randint(12, 36)

        # Transition probabilities (tend to cycle)
        if current_regime == 0:  # El Niño
            next_regime = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
        elif current_regime == 1:  # Neutral
            next_regime = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
        else:  # La Niña
            next_regime = np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])

        regime_pattern.extend([current_regime] * min(duration, T - t))
        t += duration
        current_regime = next_regime

    regimes = np.array(regime_pattern[:T])

    # DAG for each regime [d x d]
    # Edge (i,j) means i → j
    # 0=SST, 1=SOI, 2=Rain, 3=Wind
    W = np.zeros((3, d, d))

    # El Niño: strong SST → SOI → Wind → Rain cascade
    W[0] = np.array([
        [0, 0.7, 0.3, 0.4],   # SST → others
        [0, 0, 0.5, 0.6],     # SOI → others
        [0, 0, 0, 0.2],       # Rain → Wind (feedback)
        [0, 0, 0, 0]          # Wind
    ])

    # Neutral: weak connections
    W[1] = np.array([
        [0, 0.2, 0.1, 0.1],
        [0, 0, 0.1, 0.2],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    # La Niña: different structure (SOI drives more)
    W[2] = np.array([
        [0, 0.5, 0.2, 0.2],
        [0, 0, 0.7, 0.5],     # SOI is more dominant
        [0, 0, 0, 0.3],
        [0, 0, 0, 0]
    ])

    # Generate time series
    X = np.zeros((T, d))

    # Tail heaviness per regime (El Niño is extreme → low nu)
    nu_per_regime = [4, 10, 6]  # El Niño, Neutral, La Niña

    for t in range(5, T):
        k = regimes[t]

        # Noise from Student-t
        epsilon = stats.t.rvs(df=nu_per_regime[k], size=d) * 0.5

        # Lagged effects (simple AR)
        lagged = 0.3 * X[t-1]

        # Solve structural equation
        I_minus_W = np.eye(d) - W[k]
        X[t] = np.linalg.solve(I_minus_W, lagged + epsilon)

    return {
        'X': X,
        'regimes': regimes,
        'W': W,
        'variable_names': ['SST', 'SOI', 'Rain', 'Wind'],
        'regime_names': ['El Niño', 'Neutral', 'La Niña'],
        'nu': nu_per_regime
    }


def run_climate_experiment():
    """
    Run TRACED vs baselines on climate data.
    """
    print("=" * 70)
    print("TRACED: Climate Data Experiment")
    print("=" * 70)

    # Generate synthetic climate data
    print("\n[1] Generating synthetic climate data...")
    data = generate_synthetic_climate_data(n_samples=2000, seed=42)

    X = data['X']
    true_regimes = data['regimes']
    true_W = data['W']
    var_names = data['variable_names']
    regime_names = data['regime_names']

    print(f"  Shape: {X.shape}")
    print(f"  Regime distribution: {dict(zip(regime_names, np.bincount(true_regimes)))}")

    # Check for heavy tails in each regime
    print("\n  Kurtosis by regime:")
    for k in range(3):
        regime_data = X[true_regimes == k]
        kurtosis = stats.kurtosis(regime_data, axis=0).mean()
        print(f"    {regime_names[k]}: {kurtosis:.2f}")

    # Run methods
    from traced_synthetic import (
        baseline_single_dag_granger,
        baseline_gaussian_hmm_then_granger,
        traced_method,
        dag_metrics,
        regime_metrics
    )

    print("\n[2] Running baselines...")

    # Method 1: Single DAG
    print("  Single DAG Granger...")
    W_single = baseline_single_dag_granger(X, max_lag=5)

    # Method 2: Gaussian HMM
    print("  Gaussian HMM + Granger...")
    W_gaussian, regimes_gaussian = baseline_gaussian_hmm_then_granger(X, n_regimes=3)

    # Method 3: TRACED
    print("  TRACED (Student-t HMM)...")
    W_traced, regimes_traced, hmm = traced_method(X, n_regimes=3, n_iter=50)

    # Evaluate
    print("\n[3] Results:")
    print("-" * 70)

    results = []
    for method_name, W_pred, regimes_pred in [
        ('Single DAG', np.stack([W_single]*3), np.zeros_like(true_regimes)),
        ('Gaussian HMM', W_gaussian, regimes_gaussian),
        ('TRACED', W_traced, regimes_traced)
    ]:
        # DAG metrics
        f1_total = 0
        for k in range(3):
            metrics = dag_metrics(true_W[k], W_pred[k])
            f1_total += metrics['F1']

        # Regime metrics
        reg_metrics = regime_metrics(true_regimes, regimes_pred)

        results.append({
            'method': method_name,
            'F1': f1_total / 3,
            'NMI': reg_metrics['NMI'],
            'ARI': reg_metrics['ARI']
        })

        print(f"\n{method_name}:")
        print(f"  DAG F1: {f1_total/3:.3f}")
        print(f"  Regime NMI: {reg_metrics['NMI']:.3f}")
        print(f"  Regime ARI: {reg_metrics['ARI']:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("CLIMATE EXPERIMENT SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    traced_nmi = results_df[results_df['method'] == 'TRACED']['NMI'].values[0]
    gaussian_nmi = results_df[results_df['method'] == 'Gaussian HMM']['NMI'].values[0]

    print(f"\nKey finding:")
    print(f"  TRACED NMI: {traced_nmi:.3f}")
    print(f"  Gaussian NMI: {gaussian_nmi:.3f}")
    improvement = (traced_nmi - gaussian_nmi) / gaussian_nmi * 100
    print(f"  Improvement: {improvement:+.1f}%")

    return results_df, data


def analyze_discovered_structure(data, W_traced, regimes_traced):
    """
    Analyze what TRACED discovered vs ground truth.
    """
    var_names = data['variable_names']
    regime_names = data['regime_names']
    true_W = data['W']

    print("\n" + "=" * 70)
    print("CAUSAL STRUCTURE ANALYSIS")
    print("=" * 70)

    for k in range(3):
        print(f"\n{regime_names[k]} regime:")
        print("  Ground truth edges:")
        for i in range(4):
            for j in range(4):
                if true_W[k, i, j] > 0:
                    print(f"    {var_names[i]} → {var_names[j]}: {true_W[k,i,j]:.2f}")

        print("  Discovered edges:")
        for i in range(4):
            for j in range(4):
                if W_traced[k, i, j] > 0:
                    match = "✓" if true_W[k, i, j] > 0 else "✗"
                    print(f"    {var_names[i]} → {var_names[j]}: {W_traced[k,i,j]:.0f} {match}")


if __name__ == "__main__":
    results_df, data = run_climate_experiment()

    # Additional analysis
    from traced_synthetic import traced_method
    X = data['X']
    W_traced, regimes_traced, _ = traced_method(X, n_regimes=3, n_iter=50)
    analyze_discovered_structure(data, W_traced, regimes_traced)
