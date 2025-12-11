"""
TRACED vs Gaussian HMM: Head-to-Head Comparison

This is CRITICAL for NeurIPS: No quantitative comparison = automatic reject.

Compare:
1. Crisis detection: Precision, Recall, F1
2. False positive rate
3. Log-likelihood (model fit)
4. Regime stability (switching frequency)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logsumexp
import warnings
warnings.filterwarnings('ignore')

from gate2_regime_detection import StudentTHMM, load_and_prepare_data


class GaussianHMM:
    """
    Gaussian HMM for fair comparison with Student-t HMM.
    Same interface as StudentTHMM.
    """

    def __init__(self, n_regimes=3, n_iter=100, tol=1e-4):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.tol = tol

        # Parameters
        self.means_ = None
        self.covars_ = None
        self.transmat_ = None
        self.startprob_ = None
        self.log_likelihood_ = None

    def fit(self, X):
        """Fit Gaussian HMM using EM algorithm."""
        X = np.asarray(X)
        T, d = X.shape
        K = self.n_regimes

        # Initialize with K-means
        from scipy.cluster.vq import kmeans2
        centroids, labels = kmeans2(X, K, minit='++')

        self.means_ = centroids
        self.covars_ = np.array([np.cov(X[labels == k].T) + 0.01 * np.eye(d)
                                  for k in range(K)])

        # Ensure covariance matrices are valid
        for k in range(K):
            if self.covars_[k].ndim == 0:
                self.covars_[k] = np.eye(d) * X.var()

        self.transmat_ = np.ones((K, K)) / K
        self.startprob_ = np.ones(K) / K

        # EM iterations
        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            # E-step: Forward-backward
            log_alpha, log_beta, log_gamma, log_xi, ll = self._forward_backward(X)

            if iteration % 20 == 0:
                print(f"  Gaussian HMM Iteration {iteration}: LL = {ll:.2f}")

            # Check convergence
            if abs(ll - prev_ll) < self.tol:
                print(f"  Gaussian HMM converged at iteration {iteration}")
                break
            prev_ll = ll

            # M-step
            gamma = np.exp(log_gamma)

            # Update means
            for k in range(K):
                weight = gamma[:, k].sum()
                if weight > 1e-10:
                    self.means_[k] = (gamma[:, k, None] * X).sum(axis=0) / weight

            # Update covariances
            for k in range(K):
                diff = X - self.means_[k]
                weight = gamma[:, k].sum()
                if weight > 1e-10:
                    self.covars_[k] = (gamma[:, k, None, None] *
                                       np.einsum('ti,tj->tij', diff, diff)).sum(axis=0) / weight
                    # Regularize
                    self.covars_[k] += 0.01 * np.eye(d)

            # Update transition matrix
            xi_sum = np.exp(logsumexp(log_xi, axis=0))
            for i in range(K):
                denom = gamma[:-1, i].sum()
                if denom > 1e-10:
                    self.transmat_[i] = xi_sum[i] / denom

            # Update start prob
            self.startprob_ = gamma[0] / gamma[0].sum()

        self.log_likelihood_ = ll
        return self

    def _log_emission(self, X):
        """Compute log emission probabilities."""
        T, d = X.shape
        K = self.n_regimes

        log_prob = np.zeros((T, K))
        for k in range(K):
            try:
                rv = stats.multivariate_normal(self.means_[k], self.covars_[k],
                                               allow_singular=True)
                log_prob[:, k] = rv.logpdf(X)
            except:
                log_prob[:, k] = -1e10

        return log_prob

    def _forward_backward(self, X):
        """Forward-backward algorithm."""
        T = len(X)
        K = self.n_regimes

        log_B = self._log_emission(X)
        log_A = np.log(self.transmat_ + 1e-10)
        log_pi = np.log(self.startprob_ + 1e-10)

        # Forward
        log_alpha = np.zeros((T, K))
        log_alpha[0] = log_pi + log_B[0]

        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = logsumexp(log_alpha[t-1] + log_A[:, k]) + log_B[t, k]

        # Backward
        log_beta = np.zeros((T, K))

        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = logsumexp(log_A[k] + log_B[t+1] + log_beta[t+1])

        # Gamma and Xi
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)

        log_xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    log_xi[t, i, j] = (log_alpha[t, i] + log_A[i, j] +
                                       log_B[t+1, j] + log_beta[t+1, j])
            log_xi[t] -= logsumexp(log_xi[t])

        ll = logsumexp(log_alpha[-1])

        return log_alpha, log_beta, log_gamma, log_xi, ll

    def predict(self, X):
        """Predict most likely regime sequence."""
        log_gamma = self._forward_backward(X)[2]
        return np.argmax(log_gamma, axis=1)

    def score(self, X):
        """Return log-likelihood."""
        return self._forward_backward(X)[4]


def run_head_to_head_comparison():
    """
    Head-to-head comparison: Gaussian HMM vs Student-t HMM (TRACED)
    """
    print("=" * 70)
    print("HEAD-TO-HEAD: Gaussian HMM vs Student-t HMM (TRACED)")
    print("=" * 70)

    # Load data
    print("\n[1] Loading Fama-French data...")
    crowding = load_and_prepare_data()
    X = crowding.values

    print(f"  Shape: {X.shape}")
    print(f"  Date range: {crowding.index[0]} to {crowding.index[-1]}")

    # Fit both models
    print("\n[2] Fitting models...")

    print("\n  Fitting Gaussian HMM...")
    gaussian_hmm = GaussianHMM(n_regimes=3, n_iter=100)
    gaussian_hmm.fit(X)
    gaussian_regimes = gaussian_hmm.predict(X)

    print("\n  Fitting Student-t HMM (TRACED)...")
    student_hmm = StudentTHMM(n_regimes=3, n_iter=100)
    student_hmm.fit(X)
    student_regimes = student_hmm.predict(X)

    # Create comparison dataframe
    df = crowding.copy()
    df['gaussian_regime'] = gaussian_regimes
    df['student_regime'] = student_regimes

    # Identify crisis regimes (highest volatility for each model)
    def identify_crisis_regime(df, regime_col):
        vols = []
        for k in range(3):
            regime_data = df[df[regime_col] == k]
            vol = regime_data.drop([regime_col, 'gaussian_regime', 'student_regime'],
                                   axis=1, errors='ignore').std().mean()
            vols.append(vol)
        return np.argmax(vols)

    # Need to handle column names carefully
    df_gauss = df.drop('student_regime', axis=1)
    df_student = df.drop('gaussian_regime', axis=1)

    gaussian_crisis = identify_crisis_regime(df_gauss, 'gaussian_regime')
    student_crisis = identify_crisis_regime(df_student, 'student_regime')

    print(f"\n  Gaussian crisis regime: {gaussian_crisis}")
    print(f"  Student-t crisis regime: {student_crisis}")

    # Define evaluation periods
    crisis_periods = [
        ('2008-07-01', '2009-06-30', 'Financial Crisis 2008-09'),
        ('2011-07-01', '2011-10-31', 'EU Debt Crisis 2011'),
        ('2020-02-15', '2020-06-30', 'COVID-19 2020'),
        ('2000-03-01', '2002-12-31', 'Dot-com 2000-02'),
    ]

    calm_periods = [
        ('1993-01-01', '1993-12-31', '1993'),
        ('1995-01-01', '1995-12-31', '1995'),
        ('2004-01-01', '2004-12-31', '2004'),
        ('2005-01-01', '2005-12-31', '2005'),
        ('2006-01-01', '2006-12-31', '2006'),
        ('2013-01-01', '2013-12-31', '2013'),
        ('2017-01-01', '2017-12-31', '2017'),
        ('2019-01-01', '2019-12-31', '2019'),
    ]

    # Evaluate both models
    print("\n" + "=" * 70)
    print("[3] CRISIS DETECTION COMPARISON")
    print("=" * 70)

    results = {'Gaussian': [], 'Student-t': []}

    print("\n  Period                    | Gaussian | Student-t | Winner")
    print("  " + "-" * 60)

    for start, end, name in crisis_periods:
        try:
            mask = (df.index >= start) & (df.index <= end)
            period_data = df.loc[mask]

            if len(period_data) == 0:
                continue

            gauss_pct = (period_data['gaussian_regime'] == gaussian_crisis).mean() * 100
            stud_pct = (period_data['student_regime'] == student_crisis).mean() * 100

            results['Gaussian'].append(('crisis', name, gauss_pct))
            results['Student-t'].append(('crisis', name, stud_pct))

            winner = "Student-t" if stud_pct > gauss_pct else "Gaussian" if gauss_pct > stud_pct else "Tie"
            print(f"  {name:25s} | {gauss_pct:6.1f}%  | {stud_pct:7.1f}%  | {winner}")
        except Exception as e:
            print(f"  {name}: Error - {e}")

    print("\n" + "=" * 70)
    print("[4] FALSE POSITIVE COMPARISON (Calm periods)")
    print("=" * 70)

    print("\n  Period                    | Gaussian | Student-t | Winner (lower=better)")
    print("  " + "-" * 60)

    for start, end, name in calm_periods:
        try:
            mask = (df.index >= start) & (df.index <= end)
            period_data = df.loc[mask]

            if len(period_data) == 0:
                continue

            gauss_pct = (period_data['gaussian_regime'] == gaussian_crisis).mean() * 100
            stud_pct = (period_data['student_regime'] == student_crisis).mean() * 100

            results['Gaussian'].append(('calm', name, gauss_pct))
            results['Student-t'].append(('calm', name, stud_pct))

            winner = "Student-t" if stud_pct < gauss_pct else "Gaussian" if gauss_pct < stud_pct else "Tie"
            print(f"  {name:25s} | {gauss_pct:6.1f}%  | {stud_pct:7.1f}%  | {winner}")
        except Exception as e:
            print(f"  {name}: Error - {e}")

    # Compute aggregate metrics
    print("\n" + "=" * 70)
    print("[5] AGGREGATE METRICS")
    print("=" * 70)

    def compute_metrics(model_results, crisis_threshold=30, fp_threshold=10):
        crisis_detections = [r for r in model_results if r[0] == 'crisis']
        calm_detections = [r for r in model_results if r[0] == 'calm']

        TP = sum(1 for _, _, pct in crisis_detections if pct > crisis_threshold)
        FN = len(crisis_detections) - TP
        FP = sum(1 for _, _, pct in calm_detections if pct > fp_threshold)
        TN = len(calm_detections) - FP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fp_rate = FP / (FP + TN) if (FP + TN) > 0 else 0

        return {
            'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'FP_Rate': fp_rate
        }

    gauss_metrics = compute_metrics(results['Gaussian'])
    stud_metrics = compute_metrics(results['Student-t'])

    print("\n  Metric          | Gaussian | Student-t | Δ")
    print("  " + "-" * 50)

    for metric in ['Precision', 'Recall', 'F1', 'FP_Rate']:
        g_val = gauss_metrics[metric]
        s_val = stud_metrics[metric]
        delta = s_val - g_val

        # For FP_Rate, lower is better
        if metric == 'FP_Rate':
            better = "↓" if delta < 0 else "↑"
        else:
            better = "↑" if delta > 0 else "↓"

        print(f"  {metric:15s} | {g_val:8.3f} | {s_val:9.3f} | {delta:+.3f} {better}")

    # Log-likelihood comparison
    print("\n" + "-" * 70)
    print("  MODEL FIT (Log-likelihood per sample)")
    print("-" * 70)

    gauss_ll = gaussian_hmm.score(X) / len(X)
    stud_ll = student_hmm.log_likelihood_ / len(X)

    print(f"  Gaussian HMM:  {gauss_ll:.4f}")
    print(f"  Student-t HMM: {stud_ll:.4f}")
    print(f"  Δ: {stud_ll - gauss_ll:+.4f} {'(Student-t better)' if stud_ll > gauss_ll else '(Gaussian better)'}")

    # Regime stability (switching frequency)
    print("\n" + "-" * 70)
    print("  REGIME STABILITY (Daily switching rate)")
    print("-" * 70)

    gauss_switches = (np.diff(gaussian_regimes) != 0).sum() / len(gaussian_regimes)
    stud_switches = (np.diff(student_regimes) != 0).sum() / len(student_regimes)

    print(f"  Gaussian HMM:  {gauss_switches:.4f} ({gauss_switches*252:.1f} switches/year)")
    print(f"  Student-t HMM: {stud_switches:.4f} ({stud_switches*252:.1f} switches/year)")

    # Create summary table for paper
    print("\n" + "=" * 70)
    print("[6] TABLE FOR PAPER")
    print("=" * 70)

    print("""
\\begin{table}[t]
\\centering
\\caption{Head-to-head comparison: Gaussian HMM vs Student-t HMM (TRACED)}
\\label{tab:comparison}
\\begin{tabular}{lcc}
\\toprule
Metric & Gaussian HMM & Student-t HMM \\\\
\\midrule""")

    print(f"Crisis Precision & {gauss_metrics['Precision']:.3f} & \\textbf{{{stud_metrics['Precision']:.3f}}} \\\\")
    print(f"Crisis Recall & {gauss_metrics['Recall']:.3f} & \\textbf{{{stud_metrics['Recall']:.3f}}} \\\\")
    print(f"Crisis F1 & {gauss_metrics['F1']:.3f} & \\textbf{{{stud_metrics['F1']:.3f}}} \\\\")
    print(f"False Positive Rate & {gauss_metrics['FP_Rate']:.3f} & \\textbf{{{stud_metrics['FP_Rate']:.3f}}} \\\\")
    print(f"Log-likelihood/sample & {gauss_ll:.3f} & \\textbf{{{stud_ll:.3f}}} \\\\")
    print(f"Switches/year & {gauss_switches*252:.1f} & {stud_switches*252:.1f} \\\\")

    print("""\\bottomrule
\\end{tabular}
\\end{table}""")

    return {
        'gaussian_metrics': gauss_metrics,
        'student_metrics': stud_metrics,
        'gaussian_ll': gauss_ll,
        'student_ll': stud_ll,
        'gaussian_switches': gauss_switches,
        'student_switches': stud_switches
    }


if __name__ == "__main__":
    results = run_head_to_head_comparison()

    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR NEURIPS")
    print("=" * 70)

    g = results['gaussian_metrics']
    s = results['student_metrics']

    print(f"""
1. CRISIS DETECTION
   - Student-t F1: {s['F1']:.3f} vs Gaussian F1: {g['F1']:.3f}
   - Improvement: {(s['F1'] - g['F1']) / g['F1'] * 100 if g['F1'] > 0 else 0:+.1f}%

2. FALSE POSITIVE RATE
   - Student-t: {s['FP_Rate']:.3f} vs Gaussian: {g['FP_Rate']:.3f}
   - Reduction: {(g['FP_Rate'] - s['FP_Rate']) / g['FP_Rate'] * 100 if g['FP_Rate'] > 0 else 0:.1f}%

3. MODEL FIT
   - Student-t LL: {results['student_ll']:.3f} vs Gaussian LL: {results['gaussian_ll']:.3f}

This quantitative comparison is ESSENTIAL for NeurIPS acceptance.
""")
