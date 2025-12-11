"""
Debug: Why is Gaussian HMM winning on heavy-tailed data?

Possible issues:
1. Student-t HMM implementation problem
2. Data generation problem
3. Block regime structure too easy for Gaussian
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from gate2_regime_detection import StudentTHMM
from traced_gaussian_comparison import GaussianHMM


def diagnose_hmm_performance():
    """Diagnose why Student-t HMM is underperforming."""

    print("=" * 70)
    print("DIAGNOSTIC: Why is Gaussian winning?")
    print("=" * 70)

    # Generate simple heavy-tailed data
    np.random.seed(42)
    T, d = 1000, 6
    K = 3

    # Block regimes
    regimes = np.zeros(T, dtype=int)
    regimes[:333] = 0
    regimes[333:666] = 1
    regimes[666:] = 2

    # Different means per regime (easy to separate)
    means = np.array([
        [0, 0, 0, 0, 0, 0],
        [2, 2, 2, 2, 2, 2],
        [-2, -2, -2, -2, -2, -2]
    ], dtype=float)

    # Heavy-tailed noise (nu=3)
    X = np.zeros((T, d))
    for t in range(T):
        k = regimes[t]
        noise = stats.t.rvs(df=3, size=d)
        X[t] = means[k] + noise

    print("\n[1] Data characteristics:")
    print(f"  Shape: {X.shape}")
    print(f"  Overall kurtosis: {stats.kurtosis(X).mean():.2f}")

    # Per-regime kurtosis
    for k in range(K):
        regime_data = X[regimes == k]
        kurt = stats.kurtosis(regime_data).mean()
        print(f"  Regime {k} kurtosis: {kurt:.2f}")

    # Fit both models
    print("\n[2] Fitting models...")

    print("\n  Gaussian HMM:")
    gauss = GaussianHMM(n_regimes=3, n_iter=100)
    gauss.fit(X)
    gauss_pred = gauss.predict(X)

    print("\n  Student-t HMM:")
    stud = StudentTHMM(n_regimes=3, n_iter=100)
    stud.fit(X)
    stud_pred = stud.predict(X)

    # Check accuracy (with label matching)
    from scipy.optimize import linear_sum_assignment

    def compute_accuracy(true, pred, K=3):
        conf = np.zeros((K, K))
        for t in range(K):
            for p in range(K):
                conf[t, p] = ((true == t) & (pred == p)).sum()
        row_ind, col_ind = linear_sum_assignment(-conf)
        return conf[row_ind, col_ind].sum() / len(true)

    gauss_acc = compute_accuracy(regimes, gauss_pred)
    stud_acc = compute_accuracy(regimes, stud_pred)

    print(f"\n[3] Results:")
    print(f"  Gaussian Accuracy: {gauss_acc:.3f}")
    print(f"  Student-t Accuracy: {stud_acc:.3f}")

    # Check learned parameters
    print("\n[4] Learned means comparison:")
    print(f"\n  True means:\n{means}")
    print(f"\n  Gaussian learned means:\n{gauss.means_.round(2)}")
    print(f"\n  Student-t learned means:\n{stud.means_.round(2)}")

    # Check nu estimation
    if hasattr(stud, 'nu_'):
        print(f"\n[5] Student-t nu estimation:")
        print(f"  True nu: 3")
        print(f"  Learned nu: {stud.nu_}")

    # Log-likelihood comparison
    print(f"\n[6] Log-likelihood:")
    print(f"  Gaussian LL: {gauss.log_likelihood_:.2f}")
    print(f"  Student-t LL: {stud.log_likelihood_:.2f}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Panel A: True regimes
    ax = axes[0, 0]
    ax.plot(X[:, 0], 'k-', alpha=0.3)
    for k in range(K):
        mask = regimes == k
        ax.scatter(np.where(mask)[0], X[mask, 0], s=5, label=f'Regime {k}')
    ax.set_title('True Regimes')
    ax.legend()

    # Panel B: Gaussian predictions
    ax = axes[0, 1]
    ax.plot(X[:, 0], 'k-', alpha=0.3)
    for k in range(K):
        mask = gauss_pred == k
        ax.scatter(np.where(mask)[0], X[mask, 0], s=5, label=f'Regime {k}')
    ax.set_title(f'Gaussian HMM (Acc={gauss_acc:.2f})')
    ax.legend()

    # Panel C: Student-t predictions
    ax = axes[1, 0]
    ax.plot(X[:, 0], 'k-', alpha=0.3)
    for k in range(K):
        mask = stud_pred == k
        ax.scatter(np.where(mask)[0], X[mask, 0], s=5, label=f'Regime {k}')
    ax.set_title(f'Student-t HMM (Acc={stud_acc:.2f})')
    ax.legend()

    # Panel D: Prediction difference
    ax = axes[1, 1]
    diff = (gauss_pred != stud_pred).astype(int)
    ax.plot(diff, 'r-')
    ax.set_ylabel('Disagreement')
    ax.set_title(f'Disagreement ({diff.sum()} days, {diff.mean()*100:.1f}%)')

    plt.tight_layout()
    plt.savefig('/Users/i767700/Github/ai-in-finance/chorok/v11_causal_factor_crowding/debug_hmm.png', dpi=150)
    print("\nSaved: debug_hmm.png")
    plt.close()

    return gauss_acc, stud_acc


def test_different_scenarios():
    """Test different data scenarios."""

    print("\n" + "=" * 70)
    print("TESTING DIFFERENT SCENARIOS")
    print("=" * 70)

    scenarios = [
        # (n_samples, d, K, nu, regime_type, description)
        (1000, 6, 3, 3, 'block', 'Block regimes, heavy tail'),
        (1000, 6, 3, 30, 'block', 'Block regimes, light tail'),
        (1000, 6, 3, 3, 'random', 'Random switching, heavy tail'),
        (3000, 6, 3, 3, 'block', 'Longer data, heavy tail'),
    ]

    for n_samples, d, K, nu, regime_type, desc in scenarios:
        print(f"\n--- {desc} ---")

        np.random.seed(42)

        # Generate regimes
        if regime_type == 'block':
            regimes = np.zeros(n_samples, dtype=int)
            for k in range(K):
                start = k * (n_samples // K)
                end = (k + 1) * (n_samples // K) if k < K - 1 else n_samples
                regimes[start:end] = k
        else:
            # Random switching
            regimes = np.zeros(n_samples, dtype=int)
            current = 0
            t = 0
            while t < n_samples:
                duration = np.random.randint(50, 200)
                regimes[t:min(t+duration, n_samples)] = current
                current = (current + 1) % K
                t += duration

        # Different means
        means = np.random.randn(K, d) * 2

        # Generate data
        X = np.zeros((n_samples, d))
        for t in range(n_samples):
            k = regimes[t]
            noise = stats.t.rvs(df=nu, size=d)
            X[t] = means[k] + noise

        # Fit models
        gauss = GaussianHMM(n_regimes=K, n_iter=50)
        gauss.fit(X)
        gauss_pred = gauss.predict(X)

        stud = StudentTHMM(n_regimes=K, n_iter=50)
        stud.fit(X)
        stud_pred = stud.predict(X)

        # Accuracy
        from scipy.optimize import linear_sum_assignment

        def acc(true, pred, K):
            conf = np.zeros((K, K))
            for t in range(K):
                for p in range(K):
                    conf[t, p] = ((true == t) & (pred == p)).sum()
            row_ind, col_ind = linear_sum_assignment(-conf)
            return conf[row_ind, col_ind].sum() / len(true)

        gauss_acc = acc(regimes, gauss_pred, K)
        stud_acc = acc(regimes, stud_pred, K)

        winner = "TRACED" if stud_acc > gauss_acc else "Gaussian"
        print(f"  Gaussian: {gauss_acc:.3f}, Student-t: {stud_acc:.3f} → {winner}")


if __name__ == "__main__":
    gauss_acc, stud_acc = diagnose_hmm_performance()

    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    if gauss_acc > stud_acc:
        print("""
PROBLEM IDENTIFIED: Gaussian HMM is outperforming Student-t HMM.

Possible causes:
1. Student-t HMM EM not converging properly
2. Nu estimation is wrong
3. Regime separation by MEAN, not variance → Gaussian enough
4. Block regime structure too easy

Need to investigate:
- Check if nu is being estimated correctly
- Check if Student-t is using the right likelihood
- Try data where regimes differ by VARIANCE, not mean
""")
    else:
        print("Student-t HMM working as expected.")

    test_different_scenarios()
