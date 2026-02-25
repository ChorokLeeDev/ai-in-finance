#!/usr/bin/env python3
"""
LSTM-Based Neural Granger Causality Test for HML -> SMB
========================================================
Regime-conditioned LSTM Granger test with permutation inference,
gradient-based feature importance, and regime-aware interaction model.

For ICAIF 2025 submission.

Author: Research pipeline
"""

import json
import os
import sys
import time
import warnings
import io
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
N_LAGS = 9
HIDDEN_SIZE = 32
N_EPOCHS = 100
BATCH_SIZE = 64
LR = 0.001
N_PERMUTATIONS = 100
N_CV_FOLDS = 5
RANDOM_SEED = 42
REGIME_NAMES = ["Normal", "Elevated", "Crisis"]

RESULTS_DIR = "/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "lstm_granger_results.json")

device = torch.device("cpu")  # daily factor data is small, CPU is fine


# ============================================================================
# DATA
# ============================================================================
def load_ff5_data():
    """Download Fama-French 5-factor daily data (1990-2024) without pandas_datareader."""
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            df = pd.read_csv(f, skiprows=3)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={df.columns[0]: "Date"})
    df = df[df["Date"].astype(str).str.match(r"^\d{8}$")]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    for col in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("Date").dropna()
    return df.loc["1990-01-01":"2024-12-31"]


# ============================================================================
# StudentTHMM — copied from critical_fixes_analysis.py for reproducibility
# ============================================================================
class StudentTHMM:
    """Student-t HMM with both filtered and smoothed probability outputs."""

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
        self.gamma = None
        self.alpha = None
        self.log_likelihood_ = None
        self.xi = None

    def _init_params(self, X):
        np.random.seed(self.random_state)
        T, d = X.shape
        K = self.n_regimes
        centroids, labels = kmeans2(X, K, minit="++")
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
        _, logdet = np.linalg.slogdet(Sigma)
        return (
            gammaln((nu + d) / 2)
            - gammaln(nu / 2)
            - 0.5 * d * np.log(nu * np.pi)
            - 0.5 * logdet
            - 0.5 * (nu + d) * np.log(1 + mahal / nu)
        )

    def _compute_emission_probs(self, X):
        T, d = X.shape
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
                    np.logaddexp.reduce(log_alpha[t - 1] + log_A[:, k]) + log_B[t, k]
                )
        return log_alpha

    def _backward(self, log_B):
        T, K = log_B.shape
        log_beta = np.zeros((T, K))
        log_A = np.log(self.A + 1e-300)
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = np.logaddexp.reduce(
                    log_A[k, :] + log_B[t + 1, :] + log_beta[t + 1, :]
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
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        self.gamma = np.exp(log_gamma)
        log_alpha_norm = log_alpha - np.logaddexp.reduce(
            log_alpha, axis=1, keepdims=True
        )
        self.alpha = np.exp(log_alpha_norm)
        log_A = np.log(self.A + 1e-300)
        self.xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for j in range(K):
                for k in range(K):
                    self.xi[t, j, k] = np.exp(
                        log_alpha[t, j]
                        + log_A[j, k]
                        + log_B[t + 1, k]
                        + log_beta[t + 1, k]
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
                self.A[j, k] = self.xi[:, j, k].sum() / self.gamma[:-1, j].sum()
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        for k in range(K):
            weights = self.gamma[:, k] * self.u[:, k]
            self.mu[k] = (weights[:, None] * X).sum(axis=0) / weights.sum()
        for k in range(K):
            diff = X - self.mu[k]
            weights = self.gamma[:, k] * self.u[:, k]
            weighted_outer = np.zeros((d, d))
            for t in range(T):
                weighted_outer += weights[t] * np.outer(diff[t], diff[t])
            self.Sigma[k] = weighted_outer / self.gamma[:, k].sum()
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
            return -(self.gamma[:, k] * (term1 + term2 + term3)).sum()

        result = minimize_scalar(neg_expected_ll, bounds=(2.1, 50), method="bounded")
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
            self.gamma = self.gamma[:, order]
            if self.alpha is not None:
                self.alpha = self.alpha[:, order]
            if self.xi is not None:
                self.xi = self.xi[:, order, :][:, :, order]

    def fit(self, X):
        X = np.asarray(X)
        self._init_params(X)
        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            log_likelihood = self._e_step(X)
            self._m_step(X)
            if abs(log_likelihood - prev_ll) < self.tol:
                print(f"  HMM converged at iteration {iteration + 1}")
                break
            prev_ll = log_likelihood
        self.log_likelihood_ = log_likelihood
        return self

    def predict(self, X, use_filtered=False):
        X = np.asarray(X)
        self._e_step(X)
        if use_filtered:
            return np.argmax(self.alpha, axis=1)
        return np.argmax(self.gamma, axis=1)


# ============================================================================
# LSTM MODEL DEFINITIONS
# ============================================================================
class GrangerLSTM(nn.Module):
    """LSTM for Granger causality test.

    Input: (batch, seq_len, n_features)
    Output: (batch, 1) — prediction of target at time t
    """

    def __init__(self, input_dim, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)  # h_n: (1, batch, hidden)
        out = self.fc(h_n.squeeze(0))  # (batch, 1)
        return out.squeeze(-1)


class RegimeAwareLSTM(nn.Module):
    """LSTM with regime embedding for regime-interaction Granger test.

    Takes time-series input + regime indicator.
    Regime is embedded and concatenated with LSTM hidden state before prediction.
    """

    def __init__(self, input_dim, n_regimes=3, hidden_size=32, regime_embed_dim=8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.regime_embed = nn.Embedding(n_regimes, regime_embed_dim)
        self.fc = nn.Linear(hidden_size + regime_embed_dim, 1)

    def forward(self, x, regime_ids):
        # x: (batch, seq_len, input_dim), regime_ids: (batch,) LongTensor
        _, (h_n, _) = self.lstm(x)
        h = h_n.squeeze(0)  # (batch, hidden)
        r = self.regime_embed(regime_ids)  # (batch, embed_dim)
        combined = torch.cat([h, r], dim=-1)
        return self.fc(combined).squeeze(-1)


# ============================================================================
# DATA PREPARATION
# ============================================================================
def create_lag_features(smb, hml, n_lags=N_LAGS):
    """Create lagged feature arrays for Granger test.

    Returns:
        X_restricted: (N, n_lags, 1) — only SMB lags
        X_unrestricted: (N, n_lags, 2) — SMB + HML lags
        y: (N,) — SMB target
        valid_indices: original time indices for each sample
    """
    N = len(smb)
    X_smb_lags = []
    X_hml_lags = []
    y = []
    valid_indices = []

    for t in range(n_lags, N):
        X_smb_lags.append(smb[t - n_lags : t])
        X_hml_lags.append(hml[t - n_lags : t])
        y.append(smb[t])
        valid_indices.append(t)

    X_smb_lags = np.array(X_smb_lags)  # (N_valid, n_lags)
    X_hml_lags = np.array(X_hml_lags)  # (N_valid, n_lags)
    y = np.array(y)
    valid_indices = np.array(valid_indices)

    # For LSTM: (N, seq_len, features)
    X_restricted = X_smb_lags[:, :, np.newaxis]  # (N, n_lags, 1)
    X_unrestricted = np.stack(
        [X_smb_lags, X_hml_lags], axis=-1
    )  # (N, n_lags, 2)

    return X_restricted, X_unrestricted, y, valid_indices


def expanding_window_cv_splits(n_samples, n_folds=N_CV_FOLDS, min_train_frac=0.3):
    """Generate expanding-window time-series CV splits.

    Each fold uses an expanding training window and the next chunk as test.
    This preserves temporal ordering (no look-ahead bias).
    """
    # Divide into n_folds + 1 chunks; use first chunk(s) as train, last as test
    chunk_size = n_samples // (n_folds + 1)
    splits = []
    for fold in range(n_folds):
        train_end = chunk_size * (fold + 1)
        test_start = train_end
        test_end = min(test_start + chunk_size, n_samples)
        if test_end <= test_start:
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
    return splits


# ============================================================================
# TRAINING UTILITIES
# ============================================================================
def train_lstm(model, X_train, y_train, n_epochs=N_EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    """Train an LSTM model on given data."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    n = len(X_train)

    for epoch in range(n_epochs):
        # Shuffle within training data (permute samples, not time steps)
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            X_batch = X_t[idx]
            y_batch = y_t[idx]

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

    return model


def train_regime_lstm(
    model, X_train, regime_train, y_train, n_epochs=N_EPOCHS, lr=LR, batch_size=BATCH_SIZE
):
    """Train a regime-aware LSTM model."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    r_t = torch.tensor(regime_train, dtype=torch.long, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    n = len(X_train)

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            pred = model(X_t[idx], r_t[idx])
            loss = criterion(pred, y_t[idx])
            loss.backward()
            optimizer.step()

    return model


def evaluate_lstm(model, X_test, y_test):
    """Evaluate LSTM model, return MSE."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_test, dtype=torch.float32, device=device)
        pred = model(X_t)
        mse = nn.MSELoss()(pred, y_t).item()
    return mse


def evaluate_regime_lstm(model, X_test, regime_test, y_test):
    """Evaluate regime-aware LSTM model, return MSE."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        r_t = torch.tensor(regime_test, dtype=torch.long, device=device)
        y_t = torch.tensor(y_test, dtype=torch.float32, device=device)
        pred = model(X_t, r_t)
        mse = nn.MSELoss()(pred, y_t).item()
    return mse


# ============================================================================
# GRADIENT-BASED FEATURE IMPORTANCE
# ============================================================================
def compute_gradient_importance(model, X_test, y_test):
    """Compute mean absolute gradient of output w.r.t. each input feature.

    For unrestricted model, X_test has shape (N, n_lags, 2):
      - channel 0 = SMB lags
      - channel 1 = HML lags

    Returns dict with importance per lag for SMB and HML.
    """
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32, device=device, requires_grad=True)
    y_t = torch.tensor(y_test, dtype=torch.float32, device=device)

    pred = model(X_t)
    # Compute gradient of sum of predictions w.r.t. input
    pred.sum().backward()

    grad = X_t.grad.detach().numpy()  # (N, n_lags, 2)
    # Mean absolute gradient per feature
    mean_abs_grad = np.mean(np.abs(grad), axis=0)  # (n_lags, 2)

    importance = {}
    for lag in range(mean_abs_grad.shape[0]):
        importance[f"SMB_lag_{lag+1}"] = float(mean_abs_grad[lag, 0])
        importance[f"HML_lag_{lag+1}"] = float(mean_abs_grad[lag, 1])

    # Also compute aggregate
    importance["SMB_total"] = float(mean_abs_grad[:, 0].sum())
    importance["HML_total"] = float(mean_abs_grad[:, 1].sum())
    importance["HML_fraction"] = float(
        mean_abs_grad[:, 1].sum()
        / (mean_abs_grad[:, 0].sum() + mean_abs_grad[:, 1].sum() + 1e-12)
    )

    return importance


# ============================================================================
# PER-REGIME GRANGER TEST WITH PERMUTATION INFERENCE
# ============================================================================
def run_regime_granger_test(
    smb_regime, hml_regime, regime_name, n_permutations=N_PERMUTATIONS
):
    """Run LSTM Granger causality test for one regime.

    1. Expanding-window CV for restricted (SMB only) and unrestricted (SMB+HML) LSTMs
    2. Permutation test: shuffle HML lags, retrain unrestricted, compare MSE
    3. Gradient-based feature importance on unrestricted model

    Returns dict with all results.
    """
    print(f"\n{'='*60}")
    print(f"  Regime: {regime_name} (n={len(smb_regime)})")
    print(f"{'='*60}")

    n_obs = len(smb_regime)

    # Standardize within regime for numerical stability
    smb_mean, smb_std = smb_regime.mean(), smb_regime.std()
    hml_mean, hml_std = hml_regime.mean(), hml_regime.std()
    smb_z = (smb_regime - smb_mean) / (smb_std + 1e-8)
    hml_z = (hml_regime - hml_mean) / (hml_std + 1e-8)

    X_r, X_u, y, valid_idx = create_lag_features(smb_z, hml_z)

    n_valid = len(y)
    if n_valid < 50:
        print(f"  WARNING: only {n_valid} valid samples, skipping")
        return None

    print(f"  Valid samples after lagging: {n_valid}")

    # --- Step 1: Expanding-window CV ---
    splits = expanding_window_cv_splits(n_valid, n_folds=N_CV_FOLDS)
    print(f"  CV folds: {len(splits)}")

    mse_restricted_folds = []
    mse_unrestricted_folds = []
    all_importance = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        torch.manual_seed(RANDOM_SEED + fold_i)
        np.random.seed(RANDOM_SEED + fold_i)

        # Restricted model (SMB lags only)
        model_r = GrangerLSTM(input_dim=1, hidden_size=HIDDEN_SIZE).to(device)
        model_r = train_lstm(model_r, X_r[train_idx], y[train_idx])
        mse_r = evaluate_lstm(model_r, X_r[test_idx], y[test_idx])
        mse_restricted_folds.append(mse_r)

        # Unrestricted model (SMB + HML lags)
        model_u = GrangerLSTM(input_dim=2, hidden_size=HIDDEN_SIZE).to(device)
        model_u = train_lstm(model_u, X_u[train_idx], y[train_idx])
        mse_u = evaluate_lstm(model_u, X_u[test_idx], y[test_idx])
        mse_unrestricted_folds.append(mse_u)

        # Gradient importance on this fold's test set
        imp = compute_gradient_importance(model_u, X_u[test_idx], y[test_idx])
        all_importance.append(imp)

        print(
            f"    Fold {fold_i+1}: MSE restricted={mse_r:.6f}, "
            f"unrestricted={mse_u:.6f}, improvement={100*(mse_r-mse_u)/mse_r:.2f}%"
        )

    mean_mse_r = np.mean(mse_restricted_folds)
    mean_mse_u = np.mean(mse_unrestricted_folds)
    improvement_pct = 100 * (mean_mse_r - mean_mse_u) / mean_mse_r
    print(
        f"\n  Mean MSE: restricted={mean_mse_r:.6f}, unrestricted={mean_mse_u:.6f}"
    )
    print(f"  MSE improvement: {improvement_pct:.2f}%")

    # Aggregate feature importance across folds
    avg_importance = {}
    for key in all_importance[0]:
        avg_importance[key] = float(np.mean([imp[key] for imp in all_importance]))

    # --- Step 2: Permutation test ---
    print(f"\n  Running permutation test ({n_permutations} permutations)...")
    observed_improvement = mean_mse_r - mean_mse_u

    perm_improvements = []
    for p in range(n_permutations):
        np.random.seed(RANDOM_SEED + 1000 + p)

        # Shuffle HML lags (channel 1) across samples — breaks temporal HML->SMB link
        X_u_perm = X_u.copy()
        perm_idx = np.random.permutation(len(X_u_perm))
        X_u_perm[:, :, 1] = X_u_perm[perm_idx, :, 1]

        # Retrain unrestricted model on permuted data using all CV folds
        perm_mse_u_folds = []
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            torch.manual_seed(RANDOM_SEED + 2000 + p * N_CV_FOLDS + fold_i)
            model_perm = GrangerLSTM(input_dim=2, hidden_size=HIDDEN_SIZE).to(device)
            model_perm = train_lstm(model_perm, X_u_perm[train_idx], y[train_idx])
            mse_perm = evaluate_lstm(model_perm, X_u_perm[test_idx], y[test_idx])
            perm_mse_u_folds.append(mse_perm)

        perm_mean_mse = np.mean(perm_mse_u_folds)
        perm_improvement = mean_mse_r - perm_mean_mse
        perm_improvements.append(perm_improvement)

        if (p + 1) % 20 == 0:
            print(f"    Permutation {p+1}/{n_permutations} done")

    perm_improvements = np.array(perm_improvements)
    # P-value: fraction of permuted improvements >= observed
    p_value = float(np.mean(perm_improvements >= observed_improvement))
    print(f"\n  Observed MSE improvement: {observed_improvement:.6f}")
    print(f"  Permutation p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  --> SIGNIFICANT at alpha=0.05: HML Granger-causes SMB in {regime_name}")
    else:
        print(f"  --> NOT significant at alpha=0.05 in {regime_name}")

    return {
        "regime": regime_name,
        "n_obs": int(n_obs),
        "n_valid_samples": int(n_valid),
        "n_cv_folds": len(splits),
        "mse_restricted": float(mean_mse_r),
        "mse_unrestricted": float(mean_mse_u),
        "mse_improvement_pct": float(improvement_pct),
        "mse_restricted_folds": [float(x) for x in mse_restricted_folds],
        "mse_unrestricted_folds": [float(x) for x in mse_unrestricted_folds],
        "observed_mse_improvement": float(observed_improvement),
        "permutation_p_value": float(p_value),
        "n_permutations": n_permutations,
        "permutation_improvements_mean": float(perm_improvements.mean()),
        "permutation_improvements_std": float(perm_improvements.std()),
        "feature_importance": avg_importance,
    }


# ============================================================================
# REGIME-AWARE INTERACTION MODEL
# ============================================================================
def run_regime_aware_model(smb_all, hml_all, regimes_for_samples):
    """Fit a single regime-aware LSTM on ALL data.

    Compares:
      Model A: LSTM(SMB_lags, regime) -> SMB
      Model B: LSTM(SMB_lags + HML_lags, regime) -> SMB

    Tests whether HML predictive power INTERACTS with regime.
    """
    print(f"\n{'='*60}")
    print(f"  REGIME-AWARE INTERACTION MODEL (all data)")
    print(f"{'='*60}")

    # Standardize
    smb_mean, smb_std = smb_all.mean(), smb_all.std()
    hml_mean, hml_std = hml_all.mean(), hml_all.std()
    smb_z = (smb_all - smb_mean) / (smb_std + 1e-8)
    hml_z = (hml_all - hml_mean) / (hml_std + 1e-8)

    X_r, X_u, y, valid_idx = create_lag_features(smb_z, hml_z)

    # Regime for each valid sample (use regime at time t, i.e., the target time)
    regime_arr = regimes_for_samples[valid_idx]

    n_valid = len(y)
    print(f"  Total valid samples: {n_valid}")
    for k, name in enumerate(REGIME_NAMES):
        n_k = (regime_arr == k).sum()
        print(f"    {name}: {n_k} ({100*n_k/n_valid:.1f}%)")

    # Expanding-window CV
    splits = expanding_window_cv_splits(n_valid, n_folds=N_CV_FOLDS)

    mse_A_folds = []
    mse_B_folds = []
    # Also track per-regime MSE within each fold
    per_regime_A = {k: [] for k in range(3)}
    per_regime_B = {k: [] for k in range(3)}

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        torch.manual_seed(RANDOM_SEED + 500 + fold_i)
        np.random.seed(RANDOM_SEED + 500 + fold_i)

        # Model A: restricted (SMB lags + regime)
        model_A = RegimeAwareLSTM(
            input_dim=1, n_regimes=3, hidden_size=HIDDEN_SIZE
        ).to(device)
        model_A = train_regime_lstm(
            model_A, X_r[train_idx], regime_arr[train_idx], y[train_idx]
        )
        mse_A = evaluate_regime_lstm(
            model_A, X_r[test_idx], regime_arr[test_idx], y[test_idx]
        )
        mse_A_folds.append(mse_A)

        # Model B: unrestricted (SMB + HML lags + regime)
        model_B = RegimeAwareLSTM(
            input_dim=2, n_regimes=3, hidden_size=HIDDEN_SIZE
        ).to(device)
        model_B = train_regime_lstm(
            model_B, X_u[train_idx], regime_arr[train_idx], y[train_idx]
        )
        mse_B = evaluate_regime_lstm(
            model_B, X_u[test_idx], regime_arr[test_idx], y[test_idx]
        )
        mse_B_folds.append(mse_B)

        # Per-regime MSE on test set
        for k in range(3):
            mask = regime_arr[test_idx] == k
            if mask.sum() > 0:
                mse_Ak = evaluate_regime_lstm(
                    model_A,
                    X_r[test_idx][mask],
                    regime_arr[test_idx][mask],
                    y[test_idx][mask],
                )
                mse_Bk = evaluate_regime_lstm(
                    model_B,
                    X_u[test_idx][mask],
                    regime_arr[test_idx][mask],
                    y[test_idx][mask],
                )
                per_regime_A[k].append(mse_Ak)
                per_regime_B[k].append(mse_Bk)

        print(
            f"    Fold {fold_i+1}: MSE_A={mse_A:.6f}, MSE_B={mse_B:.6f}, "
            f"improvement={100*(mse_A-mse_B)/mse_A:.2f}%"
        )

    mean_A = np.mean(mse_A_folds)
    mean_B = np.mean(mse_B_folds)
    overall_improvement = 100 * (mean_A - mean_B) / mean_A

    print(f"\n  Overall: MSE_A={mean_A:.6f}, MSE_B={mean_B:.6f}")
    print(f"  Overall improvement from adding HML: {overall_improvement:.2f}%")

    # Per-regime breakdown
    regime_interaction = {}
    print(f"\n  Per-regime improvement (regime-aware model):")
    for k, name in enumerate(REGIME_NAMES):
        if per_regime_A[k]:
            avg_A = np.mean(per_regime_A[k])
            avg_B = np.mean(per_regime_B[k])
            imp = 100 * (avg_A - avg_B) / avg_A
            regime_interaction[name] = {
                "mse_restricted": float(avg_A),
                "mse_unrestricted": float(avg_B),
                "improvement_pct": float(imp),
                "n_folds": len(per_regime_A[k]),
            }
            print(
                f"    {name}: MSE_A={avg_A:.6f}, MSE_B={avg_B:.6f}, "
                f"improvement={imp:.2f}%"
            )
        else:
            regime_interaction[name] = {"mse_restricted": None, "mse_unrestricted": None}

    # Permutation test on the regime-aware model
    print(f"\n  Permutation test for regime-aware model ({N_PERMUTATIONS} perms)...")
    observed_imp = mean_A - mean_B
    perm_imps = []

    for p in range(N_PERMUTATIONS):
        np.random.seed(RANDOM_SEED + 3000 + p)
        X_u_perm = X_u.copy()
        perm_order = np.random.permutation(len(X_u_perm))
        X_u_perm[:, :, 1] = X_u_perm[perm_order, :, 1]

        perm_B_folds = []
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            torch.manual_seed(RANDOM_SEED + 4000 + p * N_CV_FOLDS + fold_i)
            model_perm = RegimeAwareLSTM(
                input_dim=2, n_regimes=3, hidden_size=HIDDEN_SIZE
            ).to(device)
            model_perm = train_regime_lstm(
                model_perm, X_u_perm[train_idx], regime_arr[train_idx], y[train_idx]
            )
            mse_perm = evaluate_regime_lstm(
                model_perm, X_u_perm[test_idx], regime_arr[test_idx], y[test_idx]
            )
            perm_B_folds.append(mse_perm)

        perm_mean_B = np.mean(perm_B_folds)
        perm_imps.append(mean_A - perm_mean_B)

        if (p + 1) % 20 == 0:
            print(f"    Permutation {p+1}/{N_PERMUTATIONS} done")

    perm_imps = np.array(perm_imps)
    p_value = float(np.mean(perm_imps >= observed_imp))
    print(f"\n  Regime-aware model permutation p-value: {p_value:.4f}")

    return {
        "mse_model_A_restricted": float(mean_A),
        "mse_model_B_unrestricted": float(mean_B),
        "overall_improvement_pct": float(overall_improvement),
        "permutation_p_value": float(p_value),
        "n_permutations": N_PERMUTATIONS,
        "per_regime_interaction": regime_interaction,
        "mse_A_folds": [float(x) for x in mse_A_folds],
        "mse_B_folds": [float(x) for x in mse_B_folds],
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("LSTM-Based Neural Granger Causality: HML -> SMB")
    print("Regime-conditioned with permutation inference")
    print("=" * 70)

    t_start = time.time()

    # --- Load data ---
    print("\n[1/4] Loading Fama-French 5-factor daily data (1990-2024)...")
    ff5 = load_ff5_data()
    print(f"  Loaded {len(ff5)} daily observations")

    smb = ff5["SMB"].values
    hml = ff5["HML"].values

    # --- Fit HMM ---
    print("\n[2/4] Fitting Student-t HMM (3 regimes, random_state=42)...")
    obs = ff5[["Mkt-RF", "SMB", "HML"]].values
    hmm = StudentTHMM(n_regimes=3, random_state=42)
    hmm.fit(obs)
    regimes = hmm.predict(obs)

    print(f"\n  Regime distribution:")
    for k, name in enumerate(REGIME_NAMES):
        n_k = (regimes == k).sum()
        print(f"    {name}: {n_k} days ({100*n_k/len(regimes):.1f}%)")

    # --- Per-regime Granger tests ---
    print("\n[3/4] Per-regime LSTM Granger tests with permutation inference...")
    regime_results = {}

    for k, name in enumerate(REGIME_NAMES):
        mask = regimes == k
        smb_k = smb[mask]
        hml_k = hml[mask]

        result = run_regime_granger_test(smb_k, hml_k, name, n_permutations=N_PERMUTATIONS)
        if result is not None:
            regime_results[name] = result

    # --- Regime-aware interaction model ---
    print("\n[4/4] Regime-aware interaction model...")
    interaction_result = run_regime_aware_model(smb, hml, regimes)

    # --- Compile and save results ---
    elapsed = time.time() - t_start

    results = {
        "method": "LSTM Neural Granger Causality",
        "description": (
            "Regime-conditioned LSTM Granger test for HML->SMB with "
            "permutation-based inference, gradient feature importance, "
            "and regime-aware interaction model"
        ),
        "config": {
            "n_lags": N_LAGS,
            "hidden_size": HIDDEN_SIZE,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "n_permutations": N_PERMUTATIONS,
            "n_cv_folds": N_CV_FOLDS,
            "random_seed": RANDOM_SEED,
            "optimizer": "Adam",
            "loss": "MSE",
        },
        "data": {
            "source": "Fama-French 5-factor daily",
            "period": "1990-2024",
            "n_total_obs": int(len(smb)),
        },
        "regime_detection": {
            "method": "Student-t HMM (3 regimes)",
            "regime_counts": {
                name: int((regimes == k).sum()) for k, name in enumerate(REGIME_NAMES)
            },
        },
        "per_regime_granger": regime_results,
        "regime_aware_interaction": interaction_result,
        "runtime_seconds": float(elapsed),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nPer-regime LSTM Granger test (HML -> SMB):")
    print(f"{'Regime':<12} {'N':>6} {'MSE_r':>10} {'MSE_u':>10} {'Improv%':>9} {'P-value':>9}")
    print("-" * 60)
    for name in REGIME_NAMES:
        if name in regime_results:
            r = regime_results[name]
            sig = "*" if r["permutation_p_value"] < 0.05 else ""
            print(
                f"{name:<12} {r['n_obs']:>6} {r['mse_restricted']:>10.6f} "
                f"{r['mse_unrestricted']:>10.6f} {r['mse_improvement_pct']:>8.2f}% "
                f"{r['permutation_p_value']:>8.4f}{sig}"
            )

    print(f"\nRegime-aware interaction model:")
    ir = interaction_result
    print(f"  Overall: MSE_A={ir['mse_model_A_restricted']:.6f}, "
          f"MSE_B={ir['mse_model_B_unrestricted']:.6f}, "
          f"improvement={ir['overall_improvement_pct']:.2f}%")
    print(f"  Permutation p-value: {ir['permutation_p_value']:.4f}")
    print(f"\n  Per-regime interaction:")
    for name in REGIME_NAMES:
        ri = ir["per_regime_interaction"].get(name, {})
        if ri.get("mse_restricted") is not None:
            print(f"    {name}: improvement={ri['improvement_pct']:.2f}%")

    print(f"\nGradient-based feature importance (HML fraction of total):")
    for name in REGIME_NAMES:
        if name in regime_results:
            imp = regime_results[name]["feature_importance"]
            print(f"  {name}: HML fraction = {imp['HML_fraction']:.4f}")
            # Top HML lags
            hml_lags = {k: v for k, v in imp.items() if k.startswith("HML_lag")}
            top_lags = sorted(hml_lags.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join(f"{k}={v:.4f}" for k, v in top_lags)
            print(f"    Top HML lags: {top_str}")

    print(f"\nTotal runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
