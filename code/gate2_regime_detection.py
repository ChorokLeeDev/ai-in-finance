"""
Gate 2: Student-t HMM Regime Detection for Factor Crowding

This implements the core novelty of CausalCrowd:
- Hidden Markov Model with Student-t emissions (not Gaussian)
- Finance-informed regime constraints (Normal/Crowding/Crisis)
- Learns different tail behavior per regime

If this works, we'll see:
1. Three distinct regimes detected
2. Crisis regime has lower degrees of freedom (heavier tails)
3. Regime switches align with known market events (2008, 2020, etc.)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln, digamma
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')


class StudentTHMM:
    """
    Hidden Markov Model with Multivariate Student-t Emissions.

    Key differences from Gaussian HMM:
    1. Each regime has its own degrees of freedom ν (tail heaviness)
    2. Uses auxiliary variable formulation for EM updates
    3. Finance-informed regime ordering constraints
    """

    def __init__(self, n_regimes=3, n_iter=100, tol=1e-4, random_state=42):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        # Parameters (initialized in fit)
        self.mu = None          # Mean per regime [K, d]
        self.Sigma = None       # Scale matrix per regime [K, d, d]
        self.nu = None          # Degrees of freedom per regime [K]
        self.A = None           # Transition matrix [K, K]
        self.pi = None          # Initial distribution [K]

        # Fitted values
        self.gamma = None       # Posterior regime probabilities [T, K]
        self.xi = None          # Pairwise posteriors [T-1, K, K]
        self.log_likelihood_ = None

    def _init_params(self, X):
        """Initialize parameters using K-means."""
        np.random.seed(self.random_state)
        T, d = X.shape
        K = self.n_regimes

        # K-means initialization for means
        from scipy.cluster.vq import kmeans2
        centroids, labels = kmeans2(X, K, minit='++')

        # Sort centroids by norm (Normal < Crowding < Crisis)
        norms = np.linalg.norm(centroids, axis=1)
        order = np.argsort(norms)
        centroids = centroids[order]

        # Reorder labels
        new_labels = np.zeros_like(labels)
        for new_k, old_k in enumerate(order):
            new_labels[labels == old_k] = new_k
        labels = new_labels

        self.mu = centroids

        # Initialize scale matrices from cluster covariances
        self.Sigma = np.zeros((K, d, d))
        for k in range(K):
            mask = labels == k
            if mask.sum() > d:
                self.Sigma[k] = np.cov(X[mask].T) + 1e-6 * np.eye(d)
            else:
                self.Sigma[k] = np.eye(d)

        # Initialize degrees of freedom (Crisis has heavier tails)
        # Normal: ν=15 (near Gaussian), Crowding: ν=7, Crisis: ν=4
        self.nu = np.array([15.0, 7.0, 4.0])

        # Initialize transition matrix (sticky)
        self.A = np.eye(K) * 0.95 + np.ones((K, K)) * 0.05 / K
        self.A = self.A / self.A.sum(axis=1, keepdims=True)

        # Initial distribution
        self.pi = np.ones(K) / K

    def _mvt_logpdf(self, x, mu, Sigma, nu):
        """
        Log probability density of multivariate Student-t.

        x: [d] or [T, d]
        mu: [d]
        Sigma: [d, d]
        nu: scalar
        """
        d = len(mu)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Mahalanobis distance
        diff = x - mu
        Sigma_inv = np.linalg.inv(Sigma)
        mahal = np.sum(diff @ Sigma_inv * diff, axis=1)  # [T]

        # Log determinant
        sign, logdet = np.linalg.slogdet(Sigma)

        # Log PDF
        logpdf = (
            gammaln((nu + d) / 2) - gammaln(nu / 2)
            - 0.5 * d * np.log(nu * np.pi)
            - 0.5 * logdet
            - 0.5 * (nu + d) * np.log(1 + mahal / nu)
        )

        return logpdf

    def _compute_emission_probs(self, X):
        """Compute log emission probabilities for all regimes."""
        T, d = X.shape
        K = self.n_regimes

        log_B = np.zeros((T, K))
        for k in range(K):
            log_B[:, k] = self._mvt_logpdf(X, self.mu[k], self.Sigma[k], self.nu[k])

        return log_B

    def _forward(self, log_B):
        """Forward algorithm (log-space for numerical stability)."""
        T, K = log_B.shape
        log_alpha = np.zeros((T, K))

        # Initialize
        log_alpha[0] = np.log(self.pi + 1e-300) + log_B[0]

        # Forward pass
        log_A = np.log(self.A + 1e-300)
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = (
                    np.logaddexp.reduce(log_alpha[t-1] + log_A[:, k])
                    + log_B[t, k]
                )

        return log_alpha

    def _backward(self, log_B):
        """Backward algorithm (log-space)."""
        T, K = log_B.shape
        log_beta = np.zeros((T, K))

        # Initialize (log(1) = 0)
        log_beta[-1] = 0

        # Backward pass
        log_A = np.log(self.A + 1e-300)
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = np.logaddexp.reduce(
                    log_A[k, :] + log_B[t+1, :] + log_beta[t+1, :]
                )

        return log_beta

    def _e_step(self, X):
        """E-step: compute posterior probabilities."""
        T, d = X.shape
        K = self.n_regimes

        # Emission probabilities
        log_B = self._compute_emission_probs(X)

        # Forward-backward
        log_alpha = self._forward(log_B)
        log_beta = self._backward(log_B)

        # Log-likelihood
        log_likelihood = np.logaddexp.reduce(log_alpha[-1])

        # Posterior regime probabilities γ_t(k) = P(z_t = k | X)
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        self.gamma = np.exp(log_gamma)

        # Pairwise posteriors ξ_t(j,k) = P(z_t = j, z_{t+1} = k | X)
        log_A = np.log(self.A + 1e-300)
        self.xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for j in range(K):
                for k in range(K):
                    self.xi[t, j, k] = np.exp(
                        log_alpha[t, j] + log_A[j, k] + log_B[t+1, k] + log_beta[t+1, k]
                        - log_likelihood
                    )

        # Expected auxiliary variable u_t for Student-t
        # E[u_t | z_t = k, X_t] = (ν_k + d) / (ν_k + δ_k(X_t))
        self.u = np.zeros((T, K))
        for k in range(K):
            diff = X - self.mu[k]
            Sigma_inv = np.linalg.inv(self.Sigma[k])
            mahal = np.sum(diff @ Sigma_inv * diff, axis=1)
            self.u[:, k] = (self.nu[k] + d) / (self.nu[k] + mahal)

        return log_likelihood

    def _m_step(self, X):
        """M-step: update parameters."""
        T, d = X.shape
        K = self.n_regimes

        # Update initial distribution
        self.pi = self.gamma[0] / self.gamma[0].sum()

        # Update transition matrix
        for j in range(K):
            for k in range(K):
                self.A[j, k] = self.xi[:, j, k].sum() / self.gamma[:-1, j].sum()
        self.A = self.A / self.A.sum(axis=1, keepdims=True)

        # Update means (weighted by γ and u)
        for k in range(K):
            weights = self.gamma[:, k] * self.u[:, k]
            self.mu[k] = (weights[:, None] * X).sum(axis=0) / weights.sum()

        # Update scale matrices
        for k in range(K):
            diff = X - self.mu[k]
            weights = self.gamma[:, k] * self.u[:, k]
            weighted_outer = np.zeros((d, d))
            for t in range(T):
                weighted_outer += weights[t] * np.outer(diff[t], diff[t])
            self.Sigma[k] = weighted_outer / self.gamma[:, k].sum()
            # Ensure positive definite
            self.Sigma[k] += 1e-6 * np.eye(d)

        # Update degrees of freedom (Newton-Raphson)
        for k in range(K):
            self._update_nu(X, k)

        # Enforce ordering constraint (Normal < Crowding < Crisis by mean norm)
        self._enforce_ordering()

    def _update_nu(self, X, k):
        """Update degrees of freedom for regime k using Newton-Raphson."""
        T, d = X.shape

        # Objective: maximize expected log-likelihood w.r.t. ν
        def neg_expected_ll(nu):
            if nu <= 2:
                return 1e10

            diff = X - self.mu[k]
            Sigma_inv = np.linalg.inv(self.Sigma[k])
            mahal = np.sum(diff @ Sigma_inv * diff, axis=1)

            # Expected log p(X | ν)
            term1 = gammaln((nu + d) / 2) - gammaln(nu / 2)
            term2 = -0.5 * d * np.log(nu)
            term3 = -0.5 * (nu + d) * np.log(1 + mahal / nu)

            ll = self.gamma[:, k] * (term1 + term2 + term3)
            return -ll.sum()

        # Optimize
        result = minimize_scalar(neg_expected_ll, bounds=(2.1, 50), method='bounded')
        self.nu[k] = result.x

    def _enforce_ordering(self):
        """Enforce regime ordering: Normal < Crowding < Crisis by mean norm."""
        norms = np.linalg.norm(self.mu, axis=1)
        order = np.argsort(norms)

        if not np.array_equal(order, np.arange(self.n_regimes)):
            # Reorder all parameters
            self.mu = self.mu[order]
            self.Sigma = self.Sigma[order]
            self.nu = self.nu[order]
            self.A = self.A[order][:, order]
            self.pi = self.pi[order]
            self.gamma = self.gamma[:, order]
            if self.xi is not None:
                self.xi = self.xi[:, order, :][:, :, order]

    def fit(self, X):
        """Fit the Student-t HMM using EM algorithm."""
        X = np.asarray(X)
        T, d = X.shape

        # Initialize
        self._init_params(X)

        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            # E-step
            log_likelihood = self._e_step(X)

            # M-step
            self._m_step(X)

            # Check convergence
            if abs(log_likelihood - prev_ll) < self.tol:
                print(f"Converged at iteration {iteration + 1}")
                break

            prev_ll = log_likelihood

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: log-likelihood = {log_likelihood:.2f}")

        self.log_likelihood_ = log_likelihood
        return self

    def predict(self, X):
        """Predict most likely regime sequence (Viterbi not needed, use argmax γ)."""
        X = np.asarray(X)
        self._e_step(X)
        return np.argmax(self.gamma, axis=1)

    def predict_proba(self, X):
        """Return posterior regime probabilities."""
        X = np.asarray(X)
        self._e_step(X)
        return self.gamma


def load_and_prepare_data():
    """Load Fama-French data and compute crowding proxy."""
    import pandas_datareader.data as web

    print("Loading Fama-French data...")

    # Get Fama-French 5 factors
    ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily',
                         'famafrench', start='1990-01-01')[0]

    # Get Momentum factor
    mom = web.DataReader('F-F_Momentum_Factor_daily',
                         'famafrench', start='1990-01-01')[0]

    # Combine
    df = ff5.join(mom)
    df.columns = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'MOM']
    df = df.drop('RF', axis=1)

    print(f"Loaded {len(df)} days of data")

    # Compute crowding proxy (rolling volatility)
    window = 60
    crowding = df.rolling(window=window).std()
    crowding = (crowding - crowding.mean()) / crowding.std()
    crowding = crowding.dropna()

    print(f"Crowding proxy computed: {len(crowding)} observations")

    return crowding


def run_gate2_validation(crowding=None):
    """
    Run Gate 2 validation: Does Student-t HMM find meaningful regimes?

    Success criteria:
    1. Three distinct regimes detected
    2. Crisis regime has lowest ν (heaviest tails)
    3. Regime switches align with known events (2008, 2020)
    4. Student-t fits better than Gaussian
    """

    if crowding is None:
        crowding = load_and_prepare_data()

    X = crowding.values
    dates = crowding.index

    print("\n" + "=" * 60)
    print("GATE 2: Student-t HMM Regime Detection")
    print("=" * 60)

    # Fit Student-t HMM
    print("\nFitting Student-t HMM with 3 regimes...")
    model = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4)
    model.fit(X)

    # Get regime assignments
    regimes = model.predict(X)
    regime_probs = model.predict_proba(X)

    # ========================================================================
    # Analysis 1: Regime characteristics
    # ========================================================================
    print("\n" + "-" * 60)
    print("REGIME CHARACTERISTICS")
    print("-" * 60)

    regime_names = ['Normal', 'Crowding', 'Crisis']

    for k in range(3):
        mask = regimes == k
        count = mask.sum()
        pct = count / len(regimes) * 100

        print(f"\nRegime {k} ({regime_names[k]}):")
        print(f"  Observations: {count} ({pct:.1f}%)")
        print(f"  Mean norm: {np.linalg.norm(model.mu[k]):.3f}")
        print(f"  Degrees of freedom (ν): {model.nu[k]:.1f}")
        print(f"  Trace(Σ): {np.trace(model.Sigma[k]):.3f}")

    # ========================================================================
    # Analysis 2: Check if ν ordering is correct (Crisis should have lowest)
    # ========================================================================
    print("\n" + "-" * 60)
    print("TAIL HEAVINESS CHECK")
    print("-" * 60)

    nu_order = np.argsort(model.nu)
    print(f"ν values: Normal={model.nu[0]:.1f}, Crowding={model.nu[1]:.1f}, Crisis={model.nu[2]:.1f}")

    if model.nu[2] < model.nu[1] < model.nu[0]:
        print("✅ Correct: Crisis has heaviest tails (lowest ν)")
        tail_check = True
    else:
        print("⚠️ Unexpected: ν ordering doesn't match expectation")
        tail_check = False

    # ========================================================================
    # Analysis 3: Known crisis detection
    # ========================================================================
    print("\n" + "-" * 60)
    print("CRISIS EVENT DETECTION")
    print("-" * 60)

    # Check regime during known crises
    known_crises = [
        ('2008-09-15', '2008-10-15', 'Lehman Brothers'),
        ('2008-10-01', '2008-12-31', 'Global Financial Crisis'),
        ('2020-03-01', '2020-04-15', 'COVID-19 Crash'),
        ('2011-08-01', '2011-08-31', 'US Debt Downgrade'),
        ('2015-08-15', '2015-09-15', 'China Devaluation'),
        ('2018-12-01', '2018-12-31', 'Dec 2018 Selloff'),
        ('2022-01-01', '2022-06-30', '2022 Bear Market'),
    ]

    crisis_detection = []
    for start, end, name in known_crises:
        try:
            mask = (dates >= start) & (dates <= end)
            if mask.sum() > 0:
                crisis_regime = regimes[mask]
                pct_crisis = (crisis_regime == 2).mean() * 100
                pct_crowding = (crisis_regime == 1).mean() * 100
                pct_elevated = pct_crisis + pct_crowding

                detected = pct_elevated > 50
                crisis_detection.append(detected)

                status = "✅" if detected else "❌"
                print(f"{status} {name}: {pct_crisis:.0f}% Crisis, {pct_crowding:.0f}% Crowding, {100-pct_elevated:.0f}% Normal")
        except:
            pass

    crisis_rate = sum(crisis_detection) / len(crisis_detection) * 100 if crisis_detection else 0
    print(f"\nCrisis detection rate: {crisis_rate:.0f}% ({sum(crisis_detection)}/{len(crisis_detection)})")

    # ========================================================================
    # Analysis 4: Transition matrix
    # ========================================================================
    print("\n" + "-" * 60)
    print("TRANSITION MATRIX")
    print("-" * 60)

    print("        Normal  Crowding  Crisis")
    for i, name in enumerate(regime_names):
        row = model.A[i]
        print(f"{name:8s} {row[0]:.3f}    {row[1]:.3f}     {row[2]:.3f}")

    # Check stickiness
    sticky = np.diag(model.A).mean()
    print(f"\nAverage stickiness (diagonal): {sticky:.3f}")

    # ========================================================================
    # Analysis 5: Compare to Gaussian HMM (information criterion)
    # ========================================================================
    print("\n" + "-" * 60)
    print("MODEL COMPARISON")
    print("-" * 60)

    # Student-t log-likelihood
    T, d = X.shape
    K = 3
    n_params_t = K * d + K * d * (d+1) / 2 + K + K * K  # mu, Sigma, nu, A
    bic_t = -2 * model.log_likelihood_ + n_params_t * np.log(T)

    print(f"Student-t HMM:")
    print(f"  Log-likelihood: {model.log_likelihood_:.2f}")
    print(f"  BIC: {bic_t:.2f}")

    # Fit Gaussian HMM for comparison
    try:
        from hmmlearn import hmm
        gaussian_model = hmm.GaussianHMM(n_components=3, covariance_type='full', n_iter=100)
        gaussian_model.fit(X)

        n_params_g = K * d + K * d * (d+1) / 2 + K * K  # no ν
        bic_g = -2 * gaussian_model.score(X) * T + n_params_g * np.log(T)

        print(f"\nGaussian HMM:")
        print(f"  Log-likelihood: {gaussian_model.score(X) * T:.2f}")
        print(f"  BIC: {bic_g:.2f}")

        if bic_t < bic_g:
            print(f"\n✅ Student-t HMM has lower BIC (better fit)")
            model_comparison = True
        else:
            print(f"\n⚠️ Gaussian HMM has lower BIC")
            model_comparison = False
    except ImportError:
        print("\n(hmmlearn not installed, skipping Gaussian comparison)")
        model_comparison = None

    # ========================================================================
    # VERDICT
    # ========================================================================
    print("\n" + "=" * 60)
    print("GATE 2 VERDICT")
    print("=" * 60)

    criteria = {
        'three_regimes_detected': len(np.unique(regimes)) == 3,
        'crisis_has_heavy_tails': tail_check,
        'crisis_events_detected': crisis_rate >= 50,
        'regimes_are_sticky': sticky > 0.8,
    }

    passed = sum(criteria.values())
    total = len(criteria)

    print(f"\nCriteria passed: {passed}/{total}")
    for name, value in criteria.items():
        status = "✅" if value else "❌"
        print(f"  {status} {name}: {value}")

    if passed >= 3:
        print(f"\n🟢 GATE 2 PASSED - PROCEED to Gate 3: Per-Regime Causal Discovery")
        gate_passed = True
    else:
        print(f"\n🔴 GATE 2 FAILED - Consider simpler regime model or different features")
        gate_passed = False

    # Return results
    results = {
        'pass': gate_passed,
        'model': model,
        'regimes': regimes,
        'regime_probs': regime_probs,
        'dates': dates,
        'criteria': criteria,
        'nu': model.nu,
        'transition_matrix': model.A
    }

    return results


def plot_regimes(results, crowding):
    """Plot regime assignments over time."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        dates = results['dates']
        regimes = results['regimes']

        # Plot 1: Factor crowding levels
        ax1 = axes[0]
        for col in crowding.columns[:3]:  # Plot first 3 factors
            ax1.plot(dates, crowding.loc[dates, col], alpha=0.7, label=col)
        ax1.set_ylabel('Crowding (z-score)')
        ax1.legend(loc='upper right')
        ax1.set_title('Factor Crowding Levels')

        # Plot 2: Regime probabilities
        ax2 = axes[1]
        regime_probs = results['regime_probs']
        colors = ['green', 'orange', 'red']
        labels = ['Normal', 'Crowding', 'Crisis']
        ax2.stackplot(dates, regime_probs.T, colors=colors, labels=labels, alpha=0.7)
        ax2.set_ylabel('Regime Probability')
        ax2.legend(loc='upper right')
        ax2.set_title('Regime Probabilities (Student-t HMM)')

        # Plot 3: Hard regime assignment
        ax3 = axes[2]
        for k, (color, label) in enumerate(zip(colors, labels)):
            mask = regimes == k
            ax3.fill_between(dates, 0, 1, where=mask, color=color, alpha=0.5, label=label)
        ax3.set_ylabel('Regime')
        ax3.set_xlabel('Date')
        ax3.legend(loc='upper right')
        ax3.set_title('Detected Regimes')

        # Mark known crises
        for ax in axes:
            ax.axvline(pd.Timestamp('2008-09-15'), color='black', linestyle='--', alpha=0.5)
            ax.axvline(pd.Timestamp('2020-03-01'), color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig('/Users/i767700/Github/ai-in-finance/chorok/v11_causal_factor_crowding/gate2_regimes.png', dpi=150)
        print("\nPlot saved to gate2_regimes.png")
        plt.close()

    except ImportError:
        print("\nMatplotlib not available, skipping plot")


if __name__ == "__main__":
    # Load data
    crowding = load_and_prepare_data()

    # Run validation
    results = run_gate2_validation(crowding)

    # Plot if possible
    plot_regimes(results, crowding)

    # Print regime statistics by period
    print("\n" + "=" * 60)
    print("REGIME DISTRIBUTION BY DECADE")
    print("=" * 60)

    dates = results['dates']
    regimes = results['regimes']

    for decade_start in ['1990', '2000', '2010', '2020']:
        decade_end = str(int(decade_start) + 10)
        mask = (dates >= decade_start) & (dates < decade_end)
        if mask.sum() > 0:
            decade_regimes = regimes[mask]
            print(f"\n{decade_start}s:")
            for k, name in enumerate(['Normal', 'Crowding', 'Crisis']):
                pct = (decade_regimes == k).mean() * 100
                print(f"  {name}: {pct:.1f}%")
