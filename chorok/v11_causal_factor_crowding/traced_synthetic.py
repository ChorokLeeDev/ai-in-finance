"""
TRACED: Synthetic Data Generator and Experiments

Generate regime-switching time series with known causal DAG
to validate our method against baselines.

Ground truth available → Can measure:
- DAG recovery (SHD, TPR, FDR)
- Regime detection (NMI, ARI)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.vq import kmeans2
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# SYNTHETIC DATA GENERATOR
# ==============================================================================

def generate_regime_switching_dag_data(
    n_samples=2000,
    n_variables=6,
    n_regimes=3,
    regime_lengths=None,
    sparsity=0.3,
    nu_per_regime=None,
    noise_scale=1.0,
    max_lag=3,
    seed=42
):
    """
    Generate synthetic time series with regime-switching causal structure.

    Parameters
    ----------
    n_samples : int
        Total number of time points
    n_variables : int
        Number of variables (d)
    n_regimes : int
        Number of regimes (K)
    regime_lengths : list or None
        Length of each regime. If None, equal lengths.
    sparsity : float
        Probability of edge existing in DAG
    nu_per_regime : list or None
        Degrees of freedom for Student-t noise per regime.
        Lower = heavier tails. If None, [4, 7, 15] (Crisis, Crowding, Normal)
    noise_scale : float
        Scale of noise
    max_lag : int
        Maximum lag for causal effects
    seed : int
        Random seed

    Returns
    -------
    data : dict with keys:
        'X': array [n_samples, n_variables] - the time series
        'regimes': array [n_samples] - true regime labels
        'W': array [n_regimes, n_variables, n_variables] - true DAGs
        'B': array [n_regimes, max_lag, n_variables, n_variables] - lagged effects
        'nu': array [n_regimes] - true degrees of freedom
    """
    np.random.seed(seed)
    d = n_variables
    K = n_regimes
    T = n_samples

    # Default regime lengths (equal)
    if regime_lengths is None:
        base_length = T // K
        regime_lengths = [base_length] * K
        regime_lengths[-1] = T - sum(regime_lengths[:-1])  # Adjust last

    # Default degrees of freedom (Crisis=4, Crowding=7, Normal=15)
    if nu_per_regime is None:
        nu_per_regime = [15, 7, 4]  # Normal, Crowding, Crisis

    # Generate DAGs for each regime
    W = np.zeros((K, d, d))
    B = np.zeros((K, max_lag, d, d))

    for k in range(K):
        # Random DAG (lower triangular to ensure acyclicity)
        W_k = np.tril(np.random.randn(d, d), k=-1)
        # Sparsify
        mask = np.random.rand(d, d) < sparsity
        W_k = W_k * mask
        # Scale
        W_k = W_k * 0.5
        W[k] = W_k

        # Lagged effects (sparser)
        for lag in range(max_lag):
            B_lag = np.random.randn(d, d) * 0.3
            mask = np.random.rand(d, d) < (sparsity * 0.5)
            B[k, lag] = B_lag * mask

    # Make regimes meaningfully different
    # Crisis regime: stronger edges
    W[2] = W[2] * 1.5
    # Normal regime: weaker edges
    W[0] = W[0] * 0.7

    # Generate regime sequence
    regimes = np.zeros(T, dtype=int)
    current_idx = 0
    for k in range(K):
        end_idx = current_idx + regime_lengths[k]
        regimes[current_idx:end_idx] = k
        current_idx = end_idx

    # Generate time series
    X = np.zeros((T, d))

    # Initialize first max_lag points
    for t in range(max_lag):
        k = regimes[t]
        X[t] = stats.t.rvs(df=nu_per_regime[k], size=d) * noise_scale

    # Generate rest using structural equations
    for t in range(max_lag, T):
        k = regimes[t]

        # Noise from Student-t
        epsilon = stats.t.rvs(df=nu_per_regime[k], size=d) * noise_scale

        # Lagged effects
        lagged = np.zeros(d)
        for lag in range(max_lag):
            lagged += B[k, lag] @ X[t - lag - 1]

        # Solve for X_t: X_t = W_k @ X_t + lagged + epsilon
        # => (I - W_k) @ X_t = lagged + epsilon
        # => X_t = (I - W_k)^{-1} @ (lagged + epsilon)
        I_minus_W = np.eye(d) - W[k]
        X[t] = np.linalg.solve(I_minus_W, lagged + epsilon)

    return {
        'X': X,
        'regimes': regimes,
        'W': W,
        'B': B,
        'nu': np.array(nu_per_regime),
        'regime_lengths': regime_lengths
    }


# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def structural_hamming_distance(W_true, W_pred, threshold=0.1):
    """
    Compute Structural Hamming Distance between DAGs.

    SHD = # edge additions + # edge deletions + # edge reversals
    """
    # Binarize
    A_true = (np.abs(W_true) > threshold).astype(int)
    A_pred = (np.abs(W_pred) > threshold).astype(int)

    # Count differences
    diff = A_true - A_pred

    # Additions (pred has edge, true doesn't)
    additions = np.sum((diff == -1))

    # Deletions (true has edge, pred doesn't)
    deletions = np.sum((diff == 1))

    # For reversals, check if (i,j) in true and (j,i) in pred
    reversals = 0
    d = W_true.shape[0]
    for i in range(d):
        for j in range(i+1, d):
            if A_true[i, j] == 1 and A_true[j, i] == 0:
                if A_pred[i, j] == 0 and A_pred[j, i] == 1:
                    reversals += 1
            elif A_true[i, j] == 0 and A_true[j, i] == 1:
                if A_pred[i, j] == 1 and A_pred[j, i] == 0:
                    reversals += 1

    return additions + deletions + reversals


def dag_metrics(W_true, W_pred, threshold=0.1):
    """
    Compute DAG recovery metrics.

    Returns: dict with SHD, TPR, FDR, F1
    """
    A_true = (np.abs(W_true) > threshold).astype(int)
    A_pred = (np.abs(W_pred) > threshold).astype(int)

    # True positives, false positives, false negatives
    TP = np.sum((A_true == 1) & (A_pred == 1))
    FP = np.sum((A_true == 0) & (A_pred == 1))
    FN = np.sum((A_true == 1) & (A_pred == 0))

    # Metrics
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall
    FDR = FP / (TP + FP) if (TP + FP) > 0 else 0  # False Discovery Rate
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    F1 = 2 * precision * TPR / (precision + TPR) if (precision + TPR) > 0 else 0

    SHD = structural_hamming_distance(W_true, W_pred, threshold)

    return {
        'SHD': SHD,
        'TPR': TPR,
        'FDR': FDR,
        'F1': F1,
        'TP': TP,
        'FP': FP,
        'FN': FN
    }


def regime_metrics(regimes_true, regimes_pred):
    """
    Compute regime detection metrics.

    Returns: dict with NMI, ARI
    """
    nmi = normalized_mutual_info_score(regimes_true, regimes_pred)
    ari = adjusted_rand_score(regimes_true, regimes_pred)

    return {
        'NMI': nmi,
        'ARI': ari
    }


# ==============================================================================
# BASELINES
# ==============================================================================

def baseline_single_dag_granger(X, max_lag=10, alpha=0.01):
    """
    Baseline: Single DAG via Granger causality (ignores regimes).
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    T, d = X.shape
    W = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            if i == j:
                continue

            try:
                # Test if i Granger-causes j
                test_data = np.column_stack([X[:, j], X[:, i]])
                results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

                # Get minimum p-value across lags
                min_pval = min(results[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1))

                if min_pval < alpha:
                    W[i, j] = 1.0  # Binary: edge exists
            except:
                pass

    return W


def baseline_gaussian_hmm_then_granger(X, n_regimes=3, max_lag=10, alpha=0.01):
    """
    Baseline: Gaussian HMM for regimes, then Granger per regime.
    Two-stage approach with Gaussian (not Student-t).
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    T, d = X.shape

    # Simple Gaussian regime detection via K-means on rolling features
    window = 60
    features = pd.DataFrame(X).rolling(window).std().dropna().values
    valid_idx = np.arange(window - 1, T)

    if len(features) < n_regimes * 10:
        # Not enough data
        return np.zeros((n_regimes, d, d)), np.zeros(T, dtype=int)

    # K-means clustering
    centroids, labels = kmeans2(features, n_regimes, minit='++')

    # Extend labels to full length
    full_regimes = np.zeros(T, dtype=int)
    full_regimes[valid_idx] = labels

    # Granger per regime
    W = np.zeros((n_regimes, d, d))

    for k in range(n_regimes):
        regime_mask = full_regimes == k
        regime_indices = np.where(regime_mask)[0]

        if len(regime_indices) < max_lag + 50:
            continue

        regime_data = X[regime_mask]

        for i in range(d):
            for j in range(d):
                if i == j:
                    continue

                try:
                    test_data = np.column_stack([regime_data[:, j], regime_data[:, i]])
                    results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
                    min_pval = min(results[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1))

                    if min_pval < alpha:
                        W[k, i, j] = 1.0
                except:
                    pass

    return W, full_regimes


# ==============================================================================
# OUR METHOD: TRACED
# ==============================================================================

def traced_method(X, n_regimes=3, max_lag=10, alpha=0.01, n_iter=50):
    """
    TRACED: Student-t HMM + per-regime Granger causality.

    This is our two-stage approach with Student-t emissions.
    """
    from gate2_regime_detection import StudentTHMM
    from statsmodels.tsa.stattools import grangercausalitytests

    T, d = X.shape

    # Stage 1: Student-t HMM for regime detection
    print("  TRACED Stage 1: Student-t HMM...")
    hmm = StudentTHMM(n_regimes=n_regimes, n_iter=n_iter)
    hmm.fit(X)
    regimes = hmm.predict(X)

    # Stage 2: Per-regime Granger causality
    print("  TRACED Stage 2: Per-regime Granger...")
    W = np.zeros((n_regimes, d, d))

    for k in range(n_regimes):
        regime_mask = regimes == k
        regime_data = X[regime_mask]

        if len(regime_data) < max_lag + 50:
            continue

        for i in range(d):
            for j in range(d):
                if i == j:
                    continue

                try:
                    test_data = np.column_stack([regime_data[:, j], regime_data[:, i]])
                    results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
                    min_pval = min(results[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1))

                    if min_pval < alpha:
                        W[k, i, j] = 1.0
                except:
                    pass

    return W, regimes, hmm


# ==============================================================================
# EXPERIMENTS
# ==============================================================================

def run_experiment(nu_values=[3, 5, 10, 30], n_trials=5, verbose=True):
    """
    Main experiment: Compare methods across different tail heaviness.

    Key hypothesis: TRACED (Student-t) should outperform Gaussian methods
    when nu is low (heavy tails).
    """
    results = []

    for nu in nu_values:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Experiment: nu = {nu} (tail heaviness)")
            print(f"{'='*60}")

        for trial in range(n_trials):
            if verbose:
                print(f"\n  Trial {trial + 1}/{n_trials}")

            # Generate data
            # All regimes have same nu for this experiment
            data = generate_regime_switching_dag_data(
                n_samples=3000,
                n_variables=6,
                n_regimes=3,
                nu_per_regime=[nu, nu, nu],  # Same tail for all regimes
                sparsity=0.3,
                seed=42 + trial
            )

            X = data['X']
            true_regimes = data['regimes']
            true_W = data['W']

            # Method 1: Single DAG (ignores regimes)
            if verbose:
                print("    Running: Single DAG Granger...")
            W_single = baseline_single_dag_granger(X, max_lag=5)

            # Method 2: Gaussian HMM + Granger
            if verbose:
                print("    Running: Gaussian HMM + Granger...")
            W_gaussian, regimes_gaussian = baseline_gaussian_hmm_then_granger(X, n_regimes=3)

            # Method 3: TRACED (Student-t HMM + Granger)
            if verbose:
                print("    Running: TRACED...")
            W_traced, regimes_traced, _ = traced_method(X, n_regimes=3)

            # Evaluate DAG recovery (average across regimes)
            for method_name, W_pred, regimes_pred in [
                ('Single DAG', np.stack([W_single]*3), np.zeros_like(true_regimes)),
                ('Gaussian HMM', W_gaussian, regimes_gaussian),
                ('TRACED', W_traced, regimes_traced)
            ]:
                # DAG metrics (average across regimes)
                shd_total = 0
                f1_total = 0
                for k in range(3):
                    metrics = dag_metrics(true_W[k], W_pred[k])
                    shd_total += metrics['SHD']
                    f1_total += metrics['F1']

                # Regime metrics
                reg_metrics = regime_metrics(true_regimes, regimes_pred)

                results.append({
                    'nu': nu,
                    'trial': trial,
                    'method': method_name,
                    'SHD': shd_total / 3,
                    'F1': f1_total / 3,
                    'NMI': reg_metrics['NMI'],
                    'ARI': reg_metrics['ARI']
                })

    return pd.DataFrame(results)


def summarize_results(results_df):
    """Summarize experiment results."""
    summary = results_df.groupby(['nu', 'method']).agg({
        'SHD': ['mean', 'std'],
        'F1': ['mean', 'std'],
        'NMI': ['mean', 'std'],
        'ARI': ['mean', 'std']
    }).round(3)

    return summary


if __name__ == "__main__":
    print("="*60)
    print("TRACED: Synthetic Experiments")
    print("="*60)

    # Quick test first
    print("\n[1] Generating test data...")
    data = generate_regime_switching_dag_data(
        n_samples=1000,
        n_variables=6,
        n_regimes=3,
        seed=42
    )

    print(f"  Data shape: {data['X'].shape}")
    print(f"  Regime distribution: {np.bincount(data['regimes'])}")
    print(f"  True nu values: {data['nu']}")

    # Check DAG properties
    for k in range(3):
        n_edges = (np.abs(data['W'][k]) > 0.1).sum()
        print(f"  Regime {k} edges: {n_edges}")

    # Run small experiment
    print("\n[2] Running experiment (nu=[5, 15], trials=2)...")
    results = run_experiment(nu_values=[5, 15], n_trials=2, verbose=True)

    print("\n[3] Results Summary:")
    print(summarize_results(results))

    print("\n[4] Key finding:")
    # Compare TRACED vs Gaussian at nu=5 (heavy tails)
    traced_nu5 = results[(results['method'] == 'TRACED') & (results['nu'] == 5)]['F1'].mean()
    gaussian_nu5 = results[(results['method'] == 'Gaussian HMM') & (results['nu'] == 5)]['F1'].mean()

    print(f"  At nu=5 (heavy tails):")
    print(f"    TRACED F1: {traced_nu5:.3f}")
    print(f"    Gaussian F1: {gaussian_nu5:.3f}")

    if traced_nu5 > gaussian_nu5:
        print(f"    → TRACED wins by {(traced_nu5 - gaussian_nu5):.3f}")
    else:
        print(f"    → Gaussian wins by {(gaussian_nu5 - traced_nu5):.3f}")
