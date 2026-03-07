"""
Improved Regime Transition Signal
==================================

Improvements over baseline:
1. Threshold optimization via grid search
2. Multiple signal combination:
   - HMM transition probability
   - Regime entropy (uncertainty)
   - Volatility spike detection
   - Momentum divergence
3. Ensemble signal with voting
"""

import numpy as np
import pandas as pd
import urllib.request
import zipfile
import io
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
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
# STUDENT-T HMM (same as before)
# =============================================================================

class StudentTHMM:
    def __init__(self, n_regimes=3, n_iter=100, tol=1e-4, random_state=42):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.mu = None
        self.Sigma = None
        self.nu = None
        self.A = None
        self.pi = None

    def _init_params(self, X):
        np.random.seed(self.random_state)
        T, d = X.shape
        K = self.n_regimes
        centroids, labels = kmeans2(X, K, minit='++')
        norms = np.linalg.norm(centroids, axis=1)
        order = np.argsort(norms)
        centroids = centroids[order]
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
        self.u = np.zeros((T, K))
        for k in range(K):
            diff = X - self.mu[k]
            Sigma_inv = np.linalg.inv(self.Sigma[k])
            mahal = np.sum(diff @ Sigma_inv * diff, axis=1)
            self.u[:, k] = (self.nu[k] + d) / (self.nu[k] + mahal)
        log_A = np.log(self.A + 1e-300)
        self.xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for j in range(K):
                for k in range(K):
                    self.xi[t, j, k] = np.exp(
                        log_alpha[t, j] + log_A[j, k] + log_B[t+1, k] + log_beta[t+1, k]
                        - log_likelihood
                    )
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
            if hasattr(self, 'gamma') and self.gamma is not None:
                self.gamma = self.gamma[:, order]

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
        X = np.asarray(X)
        log_B = self._compute_emission_probs(X)
        log_alpha = self._forward(log_B)
        log_alpha_norm = log_alpha - np.logaddexp.reduce(log_alpha, axis=1, keepdims=True)
        return np.exp(log_alpha_norm)

    def get_transition_prob(self, filtered_prob):
        if filtered_prob.ndim == 1:
            filtered_prob = filtered_prob.reshape(1, -1)
        stay_probs = np.diag(self.A)
        transition_probs = 1 - stay_probs
        p_transition = (filtered_prob * transition_probs).sum(axis=1)
        return p_transition


# =============================================================================
# MULTIPLE SIGNAL GENERATORS
# =============================================================================

def compute_regime_entropy(filtered_probs):
    """
    Shannon entropy of regime probabilities.
    High entropy = high uncertainty about current regime.
    """
    # Avoid log(0)
    probs = np.clip(filtered_probs, 1e-10, 1)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    # Normalize by max entropy (log K)
    max_entropy = np.log(probs.shape[1])
    return entropy / max_entropy


def compute_volatility_signal(returns, short_window=5, long_window=60):
    """
    Volatility regime shift signal.
    Signal fires when short-term vol exceeds long-term vol significantly.
    """
    short_vol = pd.Series(returns).rolling(short_window).std().values
    long_vol = pd.Series(returns).rolling(long_window).std().values

    # Ratio of short to long vol
    vol_ratio = short_vol / (long_vol + 1e-10)
    return vol_ratio


def compute_momentum_divergence(df, window=20):
    """
    Cross-factor momentum divergence.
    When factors that usually move together start diverging.
    """
    # Rolling correlation between SMB and HML
    smb = df['SMB'].values
    hml = df['HML'].values

    rolling_corr = pd.Series(smb).rolling(window).corr(pd.Series(hml)).values

    # Long-term average correlation
    long_corr = pd.Series(smb).rolling(252).corr(pd.Series(hml)).values

    # Divergence = deviation from long-term correlation
    divergence = np.abs(rolling_corr - long_corr)
    return divergence


# =============================================================================
# ENSEMBLE SIGNAL
# =============================================================================

class EnsembleRegimeSignal:
    """
    Combine multiple signals for better recall while maintaining precision.
    """

    def __init__(self, n_regimes=3,
                 trans_threshold=0.2,
                 entropy_threshold=0.7,
                 vol_threshold=1.5,
                 div_threshold=0.3,
                 min_votes=2):
        self.n_regimes = n_regimes
        self.trans_threshold = trans_threshold
        self.entropy_threshold = entropy_threshold
        self.vol_threshold = vol_threshold
        self.div_threshold = div_threshold
        self.min_votes = min_votes
        self.hmm = None

    def fit(self, X_train, df_train):
        self.hmm = StudentTHMM(n_regimes=self.n_regimes)
        self.hmm.fit(X_train)
        return self

    def generate_signals(self, X, df):
        """Generate ensemble signal from multiple indicators."""
        T = len(X)

        # Signal 1: HMM transition probability
        filtered_probs = self.hmm.get_filtered_probs(X)
        trans_probs = self.hmm.get_transition_prob(filtered_probs)
        signal_trans = (trans_probs > self.trans_threshold).astype(int)

        # Signal 2: Regime entropy (uncertainty)
        entropy = compute_regime_entropy(filtered_probs)
        signal_entropy = (entropy > self.entropy_threshold).astype(int)

        # Signal 3: Volatility spike
        mkt_returns = df['Mkt-RF'].values
        vol_ratio = compute_volatility_signal(mkt_returns)
        signal_vol = (vol_ratio > self.vol_threshold).astype(int)
        signal_vol = np.nan_to_num(signal_vol, nan=0)

        # Signal 4: Momentum divergence
        divergence = compute_momentum_divergence(df)
        signal_div = (divergence > self.div_threshold).astype(int)
        signal_div = np.nan_to_num(signal_div, nan=0)

        # Ensemble: vote counting
        votes = signal_trans + signal_entropy + signal_vol + signal_div
        ensemble_signal = (votes >= self.min_votes).astype(int)

        return pd.DataFrame({
            'trans_prob': trans_probs,
            'entropy': entropy,
            'vol_ratio': vol_ratio,
            'divergence': divergence,
            'signal_trans': signal_trans,
            'signal_entropy': signal_entropy,
            'signal_vol': signal_vol,
            'signal_div': signal_div,
            'votes': votes,
            'ensemble_signal': ensemble_signal,
            'filtered_regime': np.argmax(filtered_probs, axis=1)
        })


def evaluate_signal(signals, signal_col, min_lead=1, max_lead=10):
    """Evaluate a signal column for precision/recall."""
    filtered_regimes = signals['filtered_regime'].values
    signal_values = signals[signal_col].values

    # Actual transitions
    actual_transitions = np.zeros(len(filtered_regimes), dtype=int)
    for t in range(1, len(filtered_regimes)):
        if filtered_regimes[t] != filtered_regimes[t-1]:
            actual_transitions[t] = 1

    true_positives = 0
    false_positives = 0
    lead_times = []

    for t in range(len(signal_values)):
        if signal_values[t] == 1:
            window_start = min(t + min_lead, len(actual_transitions))
            window_end = min(t + max_lead + 1, len(actual_transitions))

            if window_start < len(actual_transitions):
                transitions_in_window = actual_transitions[window_start:window_end]
                if transitions_in_window.sum() > 0:
                    true_positives += 1
                    first_transition = np.where(transitions_in_window == 1)[0][0]
                    lead_times.append(first_transition + min_lead)
                else:
                    false_positives += 1

    false_negatives = 0
    for t in range(len(actual_transitions)):
        if actual_transitions[t] == 1:
            window_start = max(0, t - max_lead)
            window_end = max(0, t - min_lead + 1)
            if signal_values[window_start:window_end].sum() == 0:
                false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_lead = np.mean(lead_times) if lead_times else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_lead_time': avg_lead,
        'n_signals': int(signal_values.sum()),
        'n_transitions': int(actual_transitions.sum())
    }


# =============================================================================
# THRESHOLD OPTIMIZATION
# =============================================================================

def optimize_thresholds(X_train, df_train, X_val, df_val):
    """Grid search for optimal thresholds."""

    # Fit HMM once
    hmm = StudentTHMM(n_regimes=3)
    hmm.fit(X_train)

    best_f1 = 0
    best_params = {}

    # Grid search
    trans_thresholds = [0.15, 0.20, 0.25, 0.30]
    entropy_thresholds = [0.5, 0.6, 0.7, 0.8]
    min_votes_options = [1, 2, 3]

    results = []

    for trans_th in trans_thresholds:
        for entropy_th in entropy_thresholds:
            for min_votes in min_votes_options:
                signal_gen = EnsembleRegimeSignal(
                    trans_threshold=trans_th,
                    entropy_threshold=entropy_th,
                    min_votes=min_votes
                )
                signal_gen.hmm = hmm  # Use pre-fitted HMM

                signals = signal_gen.generate_signals(X_val, df_val)
                metrics = evaluate_signal(signals, 'ensemble_signal')

                results.append({
                    'trans_th': trans_th,
                    'entropy_th': entropy_th,
                    'min_votes': min_votes,
                    **metrics
                })

                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_params = {
                        'trans_threshold': trans_th,
                        'entropy_threshold': entropy_th,
                        'min_votes': min_votes
                    }

    return best_params, pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("IMPROVED REGIME TRANSITION SIGNAL")
    print("Multiple signals + Threshold optimization")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    df = download_ff_data()
    df = df[(df.index >= '1990-01-01') & (df.index <= '2024-12-31')]

    features = df[['Mkt-RF', 'SMB', 'HML']].values
    dates = df.index

    # Split: train / validation / test
    train_end = pd.Timestamp('2005-01-01')
    val_end = pd.Timestamp('2010-01-01')

    train_mask = dates < train_end
    val_mask = (dates >= train_end) & (dates < val_end)
    test_mask = dates >= val_end

    X_train, df_train = features[train_mask], df[train_mask]
    X_val, df_val = features[val_mask], df[val_mask]
    X_test, df_test = features[test_mask], df[test_mask]

    print(f"    Train: {df_train.index[0].date()} to {df_train.index[-1].date()} ({len(df_train)} days)")
    print(f"    Val:   {df_val.index[0].date()} to {df_val.index[-1].date()} ({len(df_val)} days)")
    print(f"    Test:  {df_test.index[0].date()} to {df_test.index[-1].date()} ({len(df_test)} days)")

    # Optimize thresholds on validation set
    print("\n[2] Optimizing thresholds on validation set...")
    best_params, grid_results = optimize_thresholds(X_train, df_train, X_val, df_val)
    print(f"    Best params: {best_params}")

    # Show top 5 configurations
    print("\n    Top 5 configurations:")
    top5 = grid_results.nlargest(5, 'f1')[['trans_th', 'entropy_th', 'min_votes', 'precision', 'recall', 'f1']]
    print(top5.to_string(index=False))

    # Train final model and evaluate on test set
    print("\n[3] Evaluating on test set with optimized params...")

    # Refit on train + val
    X_trainval = features[dates < val_end]
    df_trainval = df[dates < val_end]

    signal_gen = EnsembleRegimeSignal(**best_params)
    signal_gen.fit(X_trainval, df_trainval)

    signals = signal_gen.generate_signals(X_test, df_test)

    # Compare individual signals vs ensemble
    print("\n    Individual signal performance:")
    print("-" * 60)
    for sig_col in ['signal_trans', 'signal_entropy', 'signal_vol', 'signal_div', 'ensemble_signal']:
        metrics = evaluate_signal(signals, sig_col)
        print(f"    {sig_col:20s}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1']:.2f}, Lead={metrics['avg_lead_time']:.1f}d")

    # Final ensemble metrics
    print("\n[4] Final results (ensemble on test set):")
    print("-" * 60)
    final_metrics = evaluate_signal(signals, 'ensemble_signal')
    print(f"    Precision:     {final_metrics['precision']:.2f}")
    print(f"    Recall:        {final_metrics['recall']:.2f}")
    print(f"    F1 Score:      {final_metrics['f1']:.2f}")
    print(f"    Avg Lead Time: {final_metrics['avg_lead_time']:.1f} days")
    print(f"    Signals fired: {final_metrics['n_signals']}")
    print(f"    Transitions:   {final_metrics['n_transitions']}")

    return signals, final_metrics, best_params


if __name__ == '__main__':
    signals, metrics, params = main()
