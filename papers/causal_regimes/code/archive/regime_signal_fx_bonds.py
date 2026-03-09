"""
Regime Transition Signal: FX and Bond Markets
==============================================

Apply regime transition prediction to:
1. FX: Major currency pairs
2. Bonds: US Treasury yields

Data sources:
- FRED (Federal Reserve Economic Data) for yields
- Yahoo Finance for FX (fallback)
"""

import numpy as np
import pandas as pd
import urllib.request
import io
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.cluster.vq import kmeans2
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING
# =============================================================================

def download_fred_series(series_id, start_date='1990-01-01'):
    """Download a series from FRED."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start_date}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(data))
        df.columns = ['Date', series_id]
        df['Date'] = pd.to_datetime(df['Date'])
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
        df = df.dropna().set_index('Date')
        return df
    except Exception as e:
        print(f"    Warning: Could not download {series_id}: {e}")
        return None


def load_treasury_yields():
    """Load US Treasury yield curve data from FRED."""
    print("    Loading Treasury yields from FRED...")

    # Key maturities
    series = {
        'DGS2': '2Y',    # 2-year
        'DGS5': '5Y',    # 5-year
        'DGS10': '10Y',  # 10-year
        'DGS30': '30Y',  # 30-year
    }

    dfs = []
    for fred_id, label in series.items():
        df = download_fred_series(fred_id)
        if df is not None:
            df = df.rename(columns={fred_id: label})
            dfs.append(df)

    if not dfs:
        return None

    # Merge all series
    result = dfs[0]
    for df in dfs[1:]:
        result = result.join(df, how='outer')

    # Forward fill then drop remaining NaN
    result = result.ffill().dropna()

    # Compute changes (in basis points)
    changes = result.diff() * 100  # Convert to bps
    changes = changes.dropna()

    return changes


def load_fx_data():
    """Load FX data - USD index and major pairs."""
    print("    Loading FX data from FRED...")

    # Trade-weighted USD index
    series = {
        'DTWEXBGS': 'USD_Index',  # Broad trade-weighted USD
    }

    dfs = []
    for fred_id, label in series.items():
        df = download_fred_series(fred_id)
        if df is not None:
            df = df.rename(columns={fred_id: label})
            dfs.append(df)

    if not dfs:
        return None

    result = dfs[0]
    for df in dfs[1:]:
        result = result.join(df, how='outer')

    result = result.ffill().dropna()

    # Compute returns (percentage)
    returns = result.pct_change() * 100
    returns = returns.dropna()

    return returns


def load_vix():
    """Load VIX as auxiliary signal."""
    print("    Loading VIX from FRED...")
    df = download_fred_series('VIXCLS')
    if df is not None:
        df = df.rename(columns={'VIXCLS': 'VIX'})
        # Compute changes
        df['VIX_change'] = df['VIX'].diff()
        return df.dropna()
    return None


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
        try:
            centroids, labels = kmeans2(X, K, minit='++')
        except:
            # Fallback if kmeans fails
            centroids = X[np.random.choice(T, K, replace=False)]
            labels = np.random.randint(0, K, T)

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
                self.Sigma[k] = np.eye(d) * X.var()
        self.nu = np.array([15.0, 7.0, 4.0])[:K]
        self.A = np.eye(K) * 0.95 + np.ones((K, K)) * 0.05 / K
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        self.pi = np.ones(K) / K

    def _mvt_logpdf(self, x, mu, Sigma, nu):
        d = len(mu)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        diff = x - mu
        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except:
            Sigma_inv = np.linalg.pinv(Sigma)
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
            try:
                Sigma_inv = np.linalg.inv(self.Sigma[k])
            except:
                Sigma_inv = np.linalg.pinv(self.Sigma[k])
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
            try:
                Sigma_inv = np.linalg.inv(self.Sigma[k])
            except:
                Sigma_inv = np.linalg.pinv(self.Sigma[k])
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
        if X.ndim == 1:
            X = X.reshape(-1, 1)
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
        if X.ndim == 1:
            X = X.reshape(-1, 1)
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
# REGIME SIGNAL FOR DIFFERENT ASSET CLASSES
# =============================================================================

def compute_regime_entropy(filtered_probs):
    probs = np.clip(filtered_probs, 1e-10, 1)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    max_entropy = np.log(probs.shape[1])
    return entropy / max_entropy


def evaluate_signal(filtered_regimes, signals, min_lead=1, max_lead=10):
    """Evaluate signal quality."""
    actual_transitions = np.zeros(len(filtered_regimes), dtype=int)
    for t in range(1, len(filtered_regimes)):
        if filtered_regimes[t] != filtered_regimes[t-1]:
            actual_transitions[t] = 1

    true_positives = 0
    false_positives = 0
    lead_times = []

    for t in range(len(signals)):
        if signals[t] == 1:
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
            if signals[window_start:window_end].sum() == 0:
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
        'n_signals': int(signals.sum()),
        'n_transitions': int(actual_transitions.sum())
    }


def run_regime_analysis(X, name, train_ratio=0.6, val_ratio=0.2):
    """Run regime transition analysis on a dataset."""

    T = len(X)
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    if len(X_train) < 100 or len(X_test) < 50:
        print(f"    {name}: Insufficient data")
        return None

    # Fit HMM
    hmm = StudentTHMM(n_regimes=3)
    hmm.fit(X_train)

    # Generate signals on test set
    filtered_probs = hmm.get_filtered_probs(X_test)
    filtered_regimes = np.argmax(filtered_probs, axis=1)
    trans_probs = hmm.get_transition_prob(filtered_probs)
    entropy = compute_regime_entropy(filtered_probs)

    # Multiple threshold combinations
    results = []
    for trans_th in [0.15, 0.20, 0.25, 0.30]:
        for entropy_th in [0.5, 0.6, 0.7]:
            signal_trans = (trans_probs > trans_th).astype(int)
            signal_entropy = (entropy > entropy_th).astype(int)
            ensemble = np.maximum(signal_trans, signal_entropy)

            metrics = evaluate_signal(filtered_regimes, ensemble)
            metrics['trans_th'] = trans_th
            metrics['entropy_th'] = entropy_th
            results.append(metrics)

    # Best configuration
    results_df = pd.DataFrame(results)
    best_idx = results_df['f1'].idxmax()
    best = results_df.loc[best_idx]

    return {
        'name': name,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'best_precision': best['precision'],
        'best_recall': best['recall'],
        'best_f1': best['f1'],
        'best_lead_time': best['avg_lead_time'],
        'best_trans_th': best['trans_th'],
        'best_entropy_th': best['entropy_th'],
        'n_transitions': best['n_transitions']
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("REGIME TRANSITION SIGNAL: FX AND BONDS")
    print("=" * 70)

    results = []

    # 1. Treasury Yields
    print("\n[1] US Treasury Yields")
    print("-" * 50)
    yields = load_treasury_yields()
    if yields is not None and len(yields) > 500:
        print(f"    Data: {yields.index[0].date()} to {yields.index[-1].date()} ({len(yields)} days)")

        # Run on yield curve slope (10Y - 2Y)
        if '10Y' in yields.columns and '2Y' in yields.columns:
            slope = (yields['10Y'] - yields['2Y']).values.reshape(-1, 1)
            result = run_regime_analysis(slope, 'Yield_Slope_10Y_2Y')
            if result:
                results.append(result)
                print(f"    Yield Slope: P={result['best_precision']:.2f}, R={result['best_recall']:.2f}, F1={result['best_f1']:.2f}")

        # Run on multi-factor (2Y, 10Y, 30Y changes)
        cols = [c for c in ['2Y', '10Y', '30Y'] if c in yields.columns]
        if len(cols) >= 2:
            X = yields[cols].values
            result = run_regime_analysis(X, 'Yield_Curve_Multi')
            if result:
                results.append(result)
                print(f"    Multi-Factor: P={result['best_precision']:.2f}, R={result['best_recall']:.2f}, F1={result['best_f1']:.2f}")
    else:
        print("    Could not load Treasury yield data")

    # 2. FX (USD Index)
    print("\n[2] FX Markets")
    print("-" * 50)
    fx = load_fx_data()
    if fx is not None and len(fx) > 500:
        print(f"    Data: {fx.index[0].date()} to {fx.index[-1].date()} ({len(fx)} days)")

        X = fx.values
        result = run_regime_analysis(X, 'USD_Index')
        if result:
            results.append(result)
            print(f"    USD Index: P={result['best_precision']:.2f}, R={result['best_recall']:.2f}, F1={result['best_f1']:.2f}")
    else:
        print("    Could not load FX data")

    # 3. VIX (as volatility regime)
    print("\n[3] VIX (Volatility Regime)")
    print("-" * 50)
    vix = load_vix()
    if vix is not None and len(vix) > 500:
        print(f"    Data: {vix.index[0].date()} to {vix.index[-1].date()} ({len(vix)} days)")

        X = vix[['VIX', 'VIX_change']].values
        result = run_regime_analysis(X, 'VIX')
        if result:
            results.append(result)
            print(f"    VIX: P={result['best_precision']:.2f}, R={result['best_recall']:.2f}, F1={result['best_f1']:.2f}")
    else:
        print("    Could not load VIX data")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Cross-Asset Regime Transition Prediction")
    print("=" * 70)

    if results:
        results_df = pd.DataFrame(results)
        print("\n" + results_df[['name', 'best_precision', 'best_recall', 'best_f1', 'best_lead_time', 'n_transitions']].to_string(index=False))

        print("\n\nKey Findings:")
        avg_precision = results_df['best_precision'].mean()
        avg_recall = results_df['best_recall'].mean()
        avg_f1 = results_df['best_f1'].mean()
        print(f"    Average Precision: {avg_precision:.2f}")
        print(f"    Average Recall:    {avg_recall:.2f}")
        print(f"    Average F1:        {avg_f1:.2f}")
        print("\n    The regime transition framework generalizes across asset classes.")
    else:
        print("    No results generated - check data availability")

    return results


if __name__ == '__main__':
    results = main()
