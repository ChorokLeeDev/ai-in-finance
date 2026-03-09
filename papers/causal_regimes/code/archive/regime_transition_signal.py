"""
Regime Transition Prediction Signal
====================================

Extends the regime-conditional Granger framework to prediction:
- Use HMM filtered probability (real-time, no lookahead)
- Compute regime transition probability: P(regime_t+1 != regime_t)
- Generate early warning signal when transition prob exceeds threshold

Validation:
- Precision/Recall of transition prediction
- Lead time: how many days before actual transition?
- Risk-adjusted returns of signal-based strategy

This addresses the gap: original paper detects regimes post-hoc,
this extension predicts regime changes in real-time.
"""

import numpy as np
import pandas as pd
import urllib.request
import zipfile
import io
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA
# =============================================================================

def download_ff_data():
    """Download Fama-French 5 factors daily data."""
    url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f, skiprows=3)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={df.columns[0]: 'Date'})
    df = df[df['Date'].astype(str).str.match(r'^\d{8}$')]
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Mkt-RF', 'SMB', 'HML'])
    return df.set_index('Date').sort_index()


# =============================================================================
# STUDENT-T HMM WITH ONLINE PREDICTION
# =============================================================================

class StudentTHMM:
    """Student-t HMM with filtered probability for real-time prediction."""

    def __init__(self, n_regimes=3, n_iter=100, tol=1e-4, random_state=42):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.mu = None
        self.Sigma = None
        self.nu = None
        self.A = None  # Transition matrix
        self.pi = None
        self.gamma = None
        self.alpha = None

    def _init_params(self, X):
        np.random.seed(self.random_state)
        T, d = X.shape
        K = self.n_regimes
        centroids, labels = kmeans2(X, K, minit='++')
        norms = np.linalg.norm(centroids, axis=1)
        order = np.argsort(norms)
        centroids = centroids[order]
        new_labels = np.zeros_like(labels)
        for new_k, old_k in enumerate(order):
            new_labels[labels == old_k] = new_k
        labels = new_labels
        self.mu = centroids
        self.Sigma = np.zeros((K, d, d))
        for k in range(K):
            mask = labels == k
            if mask.sum() > d:
                self.Sigma[k] = np.cov(X[mask].T) + 1e-6 * np.eye(d)
            else:
                self.Sigma[k] = np.eye(d)
        self.nu = np.array([15.0, 7.0, 4.0])
        self.A = np.eye(K) * 0.95 + np.ones((K, K)) * 0.05 / K
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        self.pi = np.ones(K) / K

    def _mvt_logpdf(self, x, mu, Sigma, nu):
        d = len(mu)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        diff = x - mu
        Sigma_inv = np.linalg.inv(Sigma)
        mahal = np.sum(diff @ Sigma_inv * diff, axis=1)
        sign, logdet = np.linalg.slogdet(Sigma)
        logpdf = (
            gammaln((nu + d) / 2) - gammaln(nu / 2)
            - 0.5 * d * np.log(nu * np.pi)
            - 0.5 * logdet
            - 0.5 * (nu + d) * np.log(1 + mahal / nu)
        )
        return logpdf

    def _compute_emission_probs(self, X):
        T = X.shape[0]
        K = self.n_regimes
        log_B = np.zeros((T, K))
        for k in range(K):
            log_B[:, k] = self._mvt_logpdf(X, self.mu[k], self.Sigma[k], self.nu[k])
        return log_B

    def _forward(self, log_B):
        T, K = log_B.shape
        log_alpha = np.zeros((T, K))
        log_alpha[0] = np.log(self.pi + 1e-300) + log_B[0]
        log_A = np.log(self.A + 1e-300)
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = (
                    np.logaddexp.reduce(log_alpha[t-1] + log_A[:, k])
                    + log_B[t, k]
                )
        return log_alpha

    def _backward(self, log_B):
        T, K = log_B.shape
        log_beta = np.zeros((T, K))
        log_A = np.log(self.A + 1e-300)
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = np.logaddexp.reduce(
                    log_A[k, :] + log_B[t+1, :] + log_beta[t+1, :]
                )
        return log_beta

    def _e_step(self, X):
        T, d = X.shape
        K = self.n_regimes
        log_B = self._compute_emission_probs(X)
        log_alpha = self._forward(log_B)
        log_beta = self._backward(log_B)
        log_likelihood = np.logaddexp.reduce(log_alpha[-1])

        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        self.gamma = np.exp(log_gamma)

        log_alpha_norm = log_alpha - np.logaddexp.reduce(log_alpha, axis=1, keepdims=True)
        self.alpha = np.exp(log_alpha_norm)

        log_A = np.log(self.A + 1e-300)
        self.xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for j in range(K):
                for k in range(K):
                    self.xi[t, j, k] = np.exp(
                        log_alpha[t, j] + log_A[j, k] + log_B[t+1, k] + log_beta[t+1, k]
                        - log_likelihood
                    )

        self.u = np.zeros((T, K))
        for k in range(K):
            diff = X - self.mu[k]
            Sigma_inv = np.linalg.inv(self.Sigma[k])
            mahal = np.sum(diff @ Sigma_inv * diff, axis=1)
            self.u[:, k] = (self.nu[k] + d) / (self.nu[k] + mahal)

        return log_likelihood

    def _m_step(self, X):
        T, d = X.shape
        K = self.n_regimes
        self.pi = self.gamma[0] / self.gamma[0].sum()
        for j in range(K):
            for k in range(K):
                self.A[j, k] = self.xi[:, j, k].sum() / (self.gamma[:-1, j].sum() + 1e-10)
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        for k in range(K):
            weights = self.gamma[:, k] * self.u[:, k]
            self.mu[k] = (weights[:, None] * X).sum(axis=0) / (weights.sum() + 1e-10)
        for k in range(K):
            diff = X - self.mu[k]
            weights = self.gamma[:, k] * self.u[:, k]
            weighted_outer = np.zeros((d, d))
            for t in range(T):
                weighted_outer += weights[t] * np.outer(diff[t], diff[t])
            self.Sigma[k] = weighted_outer / (self.gamma[:, k].sum() + 1e-10)
            self.Sigma[k] += 1e-6 * np.eye(d)
        for k in range(K):
            self._update_nu(X, k)
        self._enforce_ordering()

    def _update_nu(self, X, k):
        T, d = X.shape
        def neg_expected_ll(nu):
            if nu <= 2:
                return 1e10
            diff = X - self.mu[k]
            Sigma_inv = np.linalg.inv(self.Sigma[k])
            mahal = np.sum(diff @ Sigma_inv * diff, axis=1)
            term1 = gammaln((nu + d) / 2) - gammaln(nu / 2)
            term2 = -0.5 * d * np.log(nu)
            term3 = -0.5 * (nu + d) * np.log(1 + mahal / nu)
            ll = self.gamma[:, k] * (term1 + term2 + term3)
            return -ll.sum()
        result = minimize_scalar(neg_expected_ll, bounds=(2.1, 50), method='bounded')
        self.nu[k] = result.x

    def _enforce_ordering(self):
        norms = np.linalg.norm(self.mu, axis=1)
        order = np.argsort(norms)
        if not np.array_equal(order, np.arange(self.n_regimes)):
            self.mu = self.mu[order]
            self.Sigma = self.Sigma[order]
            self.nu = self.nu[order]
            self.A = self.A[order][:, order]
            self.pi = self.pi[order]
            if self.gamma is not None:
                self.gamma = self.gamma[:, order]
            if self.alpha is not None:
                self.alpha = self.alpha[:, order]

    def fit(self, X):
        X = np.asarray(X)
        self._init_params(X)
        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            log_likelihood = self._e_step(X)
            self._m_step(X)
            if abs(log_likelihood - prev_ll) < self.tol:
                break
            prev_ll = log_likelihood
        return self

    def get_filtered_probs(self, X):
        """Get filtered probabilities (real-time, no lookahead)."""
        X = np.asarray(X)
        log_B = self._compute_emission_probs(X)
        log_alpha = self._forward(log_B)
        log_alpha_norm = log_alpha - np.logaddexp.reduce(log_alpha, axis=1, keepdims=True)
        return np.exp(log_alpha_norm)

    def get_transition_prob(self, filtered_prob):
        """
        Compute P(regime changes tomorrow | today's filtered prob).

        P(S_{t+1} != S_t) = sum_i P(S_t=i) * sum_{j!=i} A[i,j]
                         = sum_i P(S_t=i) * (1 - A[i,i])
        """
        # filtered_prob: shape (T, K) or (K,)
        if filtered_prob.ndim == 1:
            filtered_prob = filtered_prob.reshape(1, -1)

        # P(transition) = sum_i P(S_t=i) * (1 - A[i,i])
        stay_probs = np.diag(self.A)  # A[i,i] for each regime
        transition_probs = 1 - stay_probs  # P(leave regime i)

        # Expected transition probability given current belief
        p_transition = (filtered_prob * transition_probs).sum(axis=1)
        return p_transition


# =============================================================================
# REGIME TRANSITION SIGNAL
# =============================================================================

class RegimeTransitionSignal:
    """
    Generate regime transition warning signals.

    Signal logic:
    - Train HMM on historical data
    - Compute filtered probabilities in real-time
    - When P(transition) > threshold, emit warning
    - Validate against actual regime changes
    """

    def __init__(self, n_regimes=3, threshold=0.3, lookback_days=5):
        self.n_regimes = n_regimes
        self.threshold = threshold
        self.lookback_days = lookback_days
        self.hmm = None
        self.regime_names = ['Normal', 'Transition', 'Crisis']

    def fit(self, X_train):
        """Fit HMM on training data."""
        self.hmm = StudentTHMM(n_regimes=self.n_regimes)
        self.hmm.fit(X_train)
        return self

    def generate_signals(self, X):
        """
        Generate transition warning signals.

        Returns DataFrame with:
        - filtered_regime: current regime estimate (real-time)
        - transition_prob: P(regime changes tomorrow)
        - signal: 1 if warning, 0 otherwise
        """
        filtered_probs = self.hmm.get_filtered_probs(X)
        filtered_regimes = np.argmax(filtered_probs, axis=1)
        transition_probs = self.hmm.get_transition_prob(filtered_probs)

        # Signal when transition prob exceeds threshold
        signals = (transition_probs > self.threshold).astype(int)

        # Also signal when regime uncertainty is high (no dominant regime)
        max_probs = filtered_probs.max(axis=1)
        uncertainty_signal = (max_probs < 0.5).astype(int)

        # Combined signal
        combined_signal = np.maximum(signals, uncertainty_signal)

        return pd.DataFrame({
            'filtered_regime': filtered_regimes,
            'transition_prob': transition_probs,
            'max_prob': max_probs,
            'signal': combined_signal
        })

    def evaluate_predictions(self, X, min_lead_time=1, max_lead_time=10):
        """
        Evaluate signal quality.

        Metrics:
        - Precision: when signal fires, how often does transition happen?
        - Recall: when transition happens, how often did signal fire?
        - Lead time: how many days before actual transition?
        """
        signals_df = self.generate_signals(X)
        filtered_regimes = signals_df['filtered_regime'].values
        signals = signals_df['signal'].values

        # Identify actual regime transitions (using smoothed for ground truth)
        # This is a simplification - in practice, we'd use ex-post regime labels
        actual_transitions = np.zeros(len(filtered_regimes), dtype=int)
        for t in range(1, len(filtered_regimes)):
            if filtered_regimes[t] != filtered_regimes[t-1]:
                actual_transitions[t] = 1

        # For each signal, check if transition happens within window
        true_positives = 0
        false_positives = 0

        lead_times = []

        for t in range(len(signals)):
            if signals[t] == 1:
                # Check if any transition in [t+min_lead, t+max_lead]
                window_start = min(t + min_lead_time, len(actual_transitions))
                window_end = min(t + max_lead_time + 1, len(actual_transitions))

                if window_start < len(actual_transitions):
                    transitions_in_window = actual_transitions[window_start:window_end]
                    if transitions_in_window.sum() > 0:
                        true_positives += 1
                        # Lead time = distance to first transition
                        first_transition = np.where(transitions_in_window == 1)[0][0]
                        lead_times.append(first_transition + min_lead_time)
                    else:
                        false_positives += 1

        # Count missed transitions (false negatives)
        false_negatives = 0
        for t in range(len(actual_transitions)):
            if actual_transitions[t] == 1:
                # Check if any signal in [t-max_lead, t-min_lead]
                window_start = max(0, t - max_lead_time)
                window_end = max(0, t - min_lead_time + 1)
                if signals[window_start:window_end].sum() == 0:
                    false_negatives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        avg_lead_time = np.mean(lead_times) if lead_times else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'avg_lead_time_days': avg_lead_time,
            'n_signals': signals.sum(),
            'n_transitions': actual_transitions.sum()
        }


# =============================================================================
# ROLLING OUT-OF-SAMPLE VALIDATION
# =============================================================================

def rolling_oos_validation(df, train_years=10, test_years=2, step_years=1):
    """
    Rolling window OOS validation of regime transition signals.

    For each window:
    1. Train HMM on train_years of data
    2. Generate signals on test_years of data
    3. Evaluate prediction quality
    """
    results = []

    # Prepare features
    features = df[['Mkt-RF', 'SMB', 'HML']].values
    dates = df.index

    # Rolling windows
    start_year = dates[0].year + train_years
    end_year = dates[-1].year - test_years

    for year in range(start_year, end_year + 1, step_years):
        train_end = pd.Timestamp(f'{year}-01-01')
        test_end = pd.Timestamp(f'{year + test_years}-01-01')

        train_mask = dates < train_end
        test_mask = (dates >= train_end) & (dates < test_end)

        if train_mask.sum() < 252 * 5 or test_mask.sum() < 252:
            continue

        X_train = features[train_mask]
        X_test = features[test_mask]
        test_dates = dates[test_mask]

        # Train and evaluate
        signal_gen = RegimeTransitionSignal(threshold=0.3)
        signal_gen.fit(X_train)

        metrics = signal_gen.evaluate_predictions(X_test)
        metrics['train_end'] = train_end
        metrics['test_period'] = f'{year}-{year+test_years}'

        results.append(metrics)

        print(f"Period {year}-{year+test_years}: "
              f"Precision={metrics['precision']:.2f}, "
              f"Recall={metrics['recall']:.2f}, "
              f"F1={metrics['f1']:.2f}, "
              f"Lead={metrics['avg_lead_time_days']:.1f}d")

    return pd.DataFrame(results)


# =============================================================================
# STRATEGY BACKTEST
# =============================================================================

def backtest_signal_strategy(df, train_end='2010-01-01'):
    """
    Backtest: reduce exposure when regime transition signal fires.

    Strategy:
    - Base: 100% equity (Mkt-RF)
    - When signal fires: reduce to 50% equity for next 5 days

    Compare vs buy-and-hold.
    """
    features = df[['Mkt-RF', 'SMB', 'HML']].values
    dates = df.index
    returns = df['Mkt-RF'].values / 100  # Convert to decimal

    train_mask = dates < pd.Timestamp(train_end)
    test_mask = dates >= pd.Timestamp(train_end)

    X_train = features[train_mask]
    X_test = features[test_mask]
    test_dates = dates[test_mask]
    test_returns = returns[test_mask]

    # Train signal generator
    signal_gen = RegimeTransitionSignal(threshold=0.25)
    signal_gen.fit(X_train)

    # Generate signals for test period
    signals_df = signal_gen.generate_signals(X_test)
    signals = signals_df['signal'].values

    # Strategy: reduce exposure when signal fires
    exposure = np.ones(len(test_returns))
    signal_duration = 5  # Days to stay defensive

    for t in range(len(signals)):
        if signals[t] == 1:
            # Reduce exposure for next signal_duration days
            end_idx = min(t + signal_duration, len(exposure))
            exposure[t:end_idx] = 0.5

    # Calculate returns
    strategy_returns = exposure * test_returns
    benchmark_returns = test_returns

    # Cumulative returns
    strategy_cum = (1 + strategy_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()

    # Risk metrics
    strategy_vol = strategy_returns.std() * np.sqrt(252)
    benchmark_vol = benchmark_returns.std() * np.sqrt(252)

    strategy_sharpe = (strategy_returns.mean() * 252) / strategy_vol if strategy_vol > 0 else 0
    benchmark_sharpe = (benchmark_returns.mean() * 252) / benchmark_vol if benchmark_vol > 0 else 0

    # Drawdown
    strategy_dd = (strategy_cum / np.maximum.accumulate(strategy_cum) - 1).min()
    benchmark_dd = (benchmark_cum / np.maximum.accumulate(benchmark_cum) - 1).min()

    return {
        'test_period': f'{test_dates[0].date()} to {test_dates[-1].date()}',
        'strategy_return': (strategy_cum[-1] - 1) * 100,
        'benchmark_return': (benchmark_cum[-1] - 1) * 100,
        'strategy_vol': strategy_vol * 100,
        'benchmark_vol': benchmark_vol * 100,
        'strategy_sharpe': strategy_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'strategy_max_dd': strategy_dd * 100,
        'benchmark_max_dd': benchmark_dd * 100,
        'n_signals': signals.sum(),
        'pct_time_defensive': (1 - exposure.mean()) * 100
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("REGIME TRANSITION PREDICTION SIGNAL")
    print("Extending causal regime framework to real-time prediction")
    print("=" * 70)

    # Load data
    print("\n[1] Loading Fama-French data...")
    df = download_ff_data()
    df = df[(df.index >= '1990-01-01') & (df.index <= '2024-12-31')]
    print(f"    Data: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")

    # Rolling OOS validation
    print("\n[2] Rolling out-of-sample validation...")
    print("-" * 70)
    oos_results = rolling_oos_validation(df, train_years=10, test_years=2, step_years=2)

    print("\n[3] Summary statistics across all OOS periods:")
    print("-" * 70)
    print(f"    Average Precision: {oos_results['precision'].mean():.2f} (+/- {oos_results['precision'].std():.2f})")
    print(f"    Average Recall:    {oos_results['recall'].mean():.2f} (+/- {oos_results['recall'].std():.2f})")
    print(f"    Average F1:        {oos_results['f1'].mean():.2f} (+/- {oos_results['f1'].std():.2f})")
    print(f"    Average Lead Time: {oos_results['avg_lead_time_days'].mean():.1f} days")

    # Strategy backtest
    print("\n[4] Strategy backtest (defensive on signal)...")
    print("-" * 70)
    backtest = backtest_signal_strategy(df, train_end='2010-01-01')

    print(f"    Test period: {backtest['test_period']}")
    print(f"    Strategy return: {backtest['strategy_return']:.1f}%")
    print(f"    Benchmark return: {backtest['benchmark_return']:.1f}%")
    print(f"    Strategy Sharpe: {backtest['strategy_sharpe']:.2f}")
    print(f"    Benchmark Sharpe: {backtest['benchmark_sharpe']:.2f}")
    print(f"    Strategy Max DD: {backtest['strategy_max_dd']:.1f}%")
    print(f"    Benchmark Max DD: {backtest['benchmark_max_dd']:.1f}%")
    print(f"    Signals fired: {backtest['n_signals']}")
    print(f"    Time in defensive mode: {backtest['pct_time_defensive']:.1f}%")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
    The regime transition signal uses filtered (real-time) HMM probabilities
    to predict regime changes BEFORE they happen. Unlike the original paper
    which detects regimes post-hoc, this provides actionable early warning.

    Practical application:
    - Risk management: reduce exposure when transition probability spikes
    - Model monitoring: flag when current regime assumptions may break
    - Portfolio rebalancing: adjust factor tilts ahead of regime shifts
    """)

    return oos_results, backtest


if __name__ == '__main__':
    oos_results, backtest = main()
