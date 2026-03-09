#!/usr/bin/env python3
"""
Neural Granger Causality: PyTorch Implementation
=================================================
Implements Tank et al. (2022)-style Neural Granger Causality models.

Three architecture options:
  A. MLP with L1 penalty on input weights (component-wise MLP)
  B. LSTM with attention for causal discovery
  C. Linear Granger (OLS F-test) as baseline

Key features:
  - Regime-conditional analysis (Normal, Elevated, Crisis)
  - Causal graph extraction via attention weights or L1 sparsity
  - Expanding-window temporal CV (no look-ahead bias)
  - Permutation test for statistical significance

Focus: HML -> SMB relationship in Fama-French 5-factor data.

Reference: Tank et al. (2022) "Neural Granger Causality"
"""

import json
import os
import sys
import time
import warnings
import io
import zipfile
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import f as f_dist

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
N_LAGS = 9
HIDDEN_SIZE = 64
N_EPOCHS = 100
BATCH_SIZE = 64
LR = 0.001
L1_LAMBDA = 0.01  # L1 regularization strength for input weights
N_PERMUTATIONS = 50  # Reduced for faster execution
N_CV_FOLDS = 5
RANDOM_SEED = 42
REGIME_NAMES = ["Normal", "Elevated", "Crisis"]

RESULTS_DIR = "/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "ml_neural_granger_results.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================================
# DATA LOADING
# ============================================================================
def load_ff5_data():
    """Download Fama-French 5-factor daily data (1990-2024)."""
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    print(f"Downloading FF5 daily data from: {url}")

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as response:
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
    df = df.loc["1990-01-01":"2024-12-31"]
    print(f"  Loaded {len(df)} daily observations ({df.index[0].date()} to {df.index[-1].date()})")
    return df


# ============================================================================
# StudentTHMM for Regime Detection
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
# NEURAL NETWORK MODELS
# ============================================================================

class ComponentMLP(nn.Module):
    """
    Component-wise MLP for Neural Granger Causality (Tank et al. 2022 style).

    Architecture:
      - Input layer with learnable weights per variable (for L1 sparsity)
      - Shared hidden layers
      - Output: prediction of target at time t

    The L1 norm of input weights indicates causal importance.
    """

    def __init__(self, n_input_vars, n_lags, hidden_sizes=(64, 32), dropout=0.1):
        super().__init__()
        self.n_input_vars = n_input_vars
        self.n_lags = n_lags
        total_input = n_input_vars * n_lags

        # Separate input weights for each variable (for L1 regularization)
        self.input_weights = nn.Parameter(torch.ones(n_input_vars))

        # Build MLP layers
        layers = []
        prev_size = total_input
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = h_size
        layers.append(nn.Linear(prev_size, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (batch, n_lags, n_input_vars)
        """
        batch_size = x.shape[0]
        # Apply input weights (per-variable importance)
        weighted_x = x * self.input_weights.view(1, 1, -1)
        # Flatten for MLP
        flat_x = weighted_x.view(batch_size, -1)
        return self.mlp(flat_x).squeeze(-1)

    def get_variable_importance(self):
        """Return absolute input weights as causal importance measure."""
        return torch.abs(self.input_weights).detach().cpu().numpy()

    def l1_regularization(self):
        """L1 norm of input weights (excluding target variable's own lags)."""
        return torch.sum(torch.abs(self.input_weights[1:]))  # Skip index 0 (SMB own lags)


class AttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism for causal discovery.

    Architecture:
      - Per-variable LSTM encoders
      - Cross-attention to discover causal relationships
      - Attention weights indicate causal importance
    """

    def __init__(self, n_input_vars, n_lags, hidden_size=64, n_heads=4):
        super().__init__()
        self.n_input_vars = n_input_vars
        self.n_lags = n_lags
        self.hidden_size = hidden_size

        # Per-variable LSTM encoders
        self.var_lstms = nn.ModuleList([
            nn.LSTM(1, hidden_size, batch_first=True)
            for _ in range(n_input_vars)
        ])

        # Multi-head attention for causal discovery
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            batch_first=True
        )

        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * n_input_vars, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

        # Store attention weights for analysis
        self.last_attention_weights = None

    def forward(self, x):
        """
        x: (batch, n_lags, n_input_vars)
        """
        batch_size = x.shape[0]

        # Encode each variable separately
        var_hidden = []
        for i in range(self.n_input_vars):
            var_input = x[:, :, i:i+1]  # (batch, n_lags, 1)
            _, (h_n, _) = self.var_lstms[i](var_input)
            var_hidden.append(h_n.squeeze(0))  # (batch, hidden_size)

        # Stack variable representations: (batch, n_vars, hidden_size)
        var_stack = torch.stack(var_hidden, dim=1)

        # Self-attention across variables
        attn_out, attn_weights = self.attention(
            var_stack, var_stack, var_stack,
            need_weights=True
        )
        self.last_attention_weights = attn_weights.detach()

        # Flatten and predict
        flat = attn_out.reshape(batch_size, -1)
        return self.fc(flat).squeeze(-1)

    def get_causal_graph(self):
        """Extract causal graph from attention weights."""
        if self.last_attention_weights is None:
            return None
        # Average attention weights across batch
        return self.last_attention_weights.mean(dim=0).cpu().numpy()


class RegimeConditionedMLP(nn.Module):
    """
    MLP with regime embedding for regime-conditional Granger causality.
    """

    def __init__(self, n_input_vars, n_lags, n_regimes=3, hidden_sizes=(64, 32), regime_embed_dim=16):
        super().__init__()
        self.n_input_vars = n_input_vars
        self.n_lags = n_lags
        total_input = n_input_vars * n_lags

        # Regime embedding
        self.regime_embed = nn.Embedding(n_regimes, regime_embed_dim)

        # Input weights per variable
        self.input_weights = nn.Parameter(torch.ones(n_input_vars))

        # MLP with regime conditioning
        layers = []
        prev_size = total_input + regime_embed_dim
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = h_size
        layers.append(nn.Linear(prev_size, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, regime_ids):
        """
        x: (batch, n_lags, n_input_vars)
        regime_ids: (batch,) LongTensor
        """
        batch_size = x.shape[0]
        weighted_x = x * self.input_weights.view(1, 1, -1)
        flat_x = weighted_x.view(batch_size, -1)
        regime_emb = self.regime_embed(regime_ids)
        combined = torch.cat([flat_x, regime_emb], dim=-1)
        return self.mlp(combined).squeeze(-1)

    def get_variable_importance(self):
        return torch.abs(self.input_weights).detach().cpu().numpy()

    def l1_regularization(self):
        return torch.sum(torch.abs(self.input_weights[1:]))


# ============================================================================
# DATA PREPARATION
# ============================================================================

def create_lag_features(smb, hml, n_lags=N_LAGS):
    """
    Create lagged feature arrays for Granger test.

    Returns:
        X_restricted: (N, n_lags, 1) - only SMB lags
        X_unrestricted: (N, n_lags, 2) - SMB + HML lags
        y: (N,) - SMB target
        valid_indices: original time indices
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

    X_smb_lags = np.array(X_smb_lags)
    X_hml_lags = np.array(X_hml_lags)
    y = np.array(y)
    valid_indices = np.array(valid_indices)

    # Shape: (N, n_lags, n_vars)
    X_restricted = X_smb_lags[:, :, np.newaxis]
    X_unrestricted = np.stack([X_smb_lags, X_hml_lags], axis=-1)

    return X_restricted, X_unrestricted, y, valid_indices


def expanding_window_cv_splits(n_samples, n_folds=N_CV_FOLDS):
    """
    Generate expanding-window time-series CV splits.
    Preserves temporal ordering (no look-ahead bias).
    """
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

def train_mlp(model, X_train, y_train, n_epochs=N_EPOCHS, lr=LR, batch_size=BATCH_SIZE,
              l1_lambda=L1_LAMBDA, verbose=False):
    """Train ComponentMLP or standard MLP model."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    n = len(X_train)

    best_loss = float('inf')
    patience = 20
    no_improve = 0

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            X_batch = X_t[idx]
            y_batch = y_t[idx]

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)

            # L1 regularization on input weights
            if hasattr(model, 'l1_regularization'):
                loss = loss + l1_lambda * model.l1_regularization()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch+1}")
            break

    return model


def train_attention_lstm(model, X_train, y_train, n_epochs=N_EPOCHS, lr=LR,
                         batch_size=BATCH_SIZE, verbose=False):
    """Train AttentionLSTM model."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    n = len(X_train)

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            pred = model(X_t[idx])
            loss = criterion(pred, y_t[idx])
            loss.backward()
            optimizer.step()

    return model


def train_regime_mlp(model, X_train, regime_train, y_train, n_epochs=N_EPOCHS,
                     lr=LR, batch_size=BATCH_SIZE, l1_lambda=L1_LAMBDA):
    """Train RegimeConditionedMLP."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    r_t = torch.tensor(regime_train, dtype=torch.long, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    n = len(X_train)

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            pred = model(X_t[idx], r_t[idx])
            loss = criterion(pred, y_t[idx])
            if hasattr(model, 'l1_regularization'):
                loss = loss + l1_lambda * model.l1_regularization()
            loss.backward()
            optimizer.step()

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return MSE."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_test, dtype=torch.float32, device=device)
        pred = model(X_t)
        mse = nn.MSELoss()(pred, y_t).item()
    return mse


def evaluate_regime_model(model, X_test, regime_test, y_test):
    """Evaluate regime-conditioned model."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        r_t = torch.tensor(regime_test, dtype=torch.long, device=device)
        y_t = torch.tensor(y_test, dtype=torch.float32, device=device)
        pred = model(X_t, r_t)
        mse = nn.MSELoss()(pred, y_t).item()
    return mse


# ============================================================================
# LINEAR GRANGER (BASELINE)
# ============================================================================

def linear_granger_test(y, X_r, X_u, max_lag=N_LAGS):
    """
    Standard linear Granger causality via OLS F-test.

    X_r: restricted features (SMB lags only), shape (N, max_lag)
    X_u: unrestricted features (SMB + HML lags), shape (N, 2*max_lag)
    """
    n = len(y)

    # Flatten lag features
    X_r_flat = X_r.reshape(n, -1)
    X_u_flat = X_u.reshape(n, -1)

    X_r_i = np.column_stack([np.ones(n), X_r_flat])
    X_u_i = np.column_stack([np.ones(n), X_u_flat])

    beta_r = np.linalg.lstsq(X_r_i, y, rcond=None)[0]
    beta_u = np.linalg.lstsq(X_u_i, y, rcond=None)[0]

    rss_r = np.sum((y - X_r_i @ beta_r) ** 2)
    rss_u = np.sum((y - X_u_i @ beta_u) ** 2)

    df1 = max_lag  # Number of HML lag coefficients
    df2 = n - X_u_i.shape[1]

    if df2 > 0 and rss_u > 0:
        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
        p_value = 1 - f_dist.cdf(f_stat, df1, df2)
    else:
        f_stat = 0.0
        p_value = 1.0

    tss = np.sum((y - y.mean()) ** 2)
    r2_r = 1 - rss_r / tss if tss > 0 else 0.0
    r2_u = 1 - rss_u / tss if tss > 0 else 0.0

    mse_r = rss_r / n
    mse_u = rss_u / n
    mse_improvement = (mse_r - mse_u) / mse_r * 100 if mse_r > 0 else 0.0

    return {
        'f_stat': float(f_stat),
        'p_value': float(p_value),
        'mse_restricted': float(mse_r),
        'mse_unrestricted': float(mse_u),
        'mse_improvement_pct': float(mse_improvement),
        'r2_restricted': float(r2_r),
        'r2_unrestricted': float(r2_u),
        'delta_r2': float(r2_u - r2_r),
        'n_obs': int(n)
    }


# ============================================================================
# PER-REGIME GRANGER TEST WITH PERMUTATION INFERENCE
# ============================================================================

def run_neural_granger_test(smb_regime, hml_regime, regime_name,
                            n_permutations=N_PERMUTATIONS, verbose=True):
    """
    Run Neural Granger causality test for one regime.

    Compares:
      1. Linear Granger (OLS F-test)
      2. MLP Granger (ComponentMLP with L1 penalty)
      3. LSTM Granger (AttentionLSTM)

    Uses expanding-window temporal CV and permutation test.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Regime: {regime_name} (n={len(smb_regime)})")
        print(f"{'='*70}")

    n_obs = len(smb_regime)

    # Standardize
    smb_mean, smb_std = smb_regime.mean(), smb_regime.std()
    hml_mean, hml_std = hml_regime.mean(), hml_regime.std()
    smb_z = (smb_regime - smb_mean) / (smb_std + 1e-8)
    hml_z = (hml_regime - hml_mean) / (hml_std + 1e-8)

    X_r, X_u, y, valid_idx = create_lag_features(smb_z, hml_z)

    n_valid = len(y)
    if n_valid < 100:
        if verbose:
            print(f"  WARNING: only {n_valid} valid samples, skipping")
        return None

    if verbose:
        print(f"  Valid samples after lagging: {n_valid}")

    # --- Linear Granger (baseline) ---
    if verbose:
        print(f"\n  [1/3] Linear Granger (OLS F-test)...")
    linear_result = linear_granger_test(y, X_r, X_u)
    if verbose:
        sig = "***" if linear_result['p_value'] < 0.001 else "**" if linear_result['p_value'] < 0.01 else "*" if linear_result['p_value'] < 0.05 else "ns"
        print(f"    F={linear_result['f_stat']:.3f}, p={linear_result['p_value']:.6f} {sig}")
        print(f"    MSE improvement: {linear_result['mse_improvement_pct']:.4f}%")
        print(f"    R2: restricted={linear_result['r2_restricted']:.4f}, unrestricted={linear_result['r2_unrestricted']:.4f}")

    # --- MLP Granger (expanding-window CV) ---
    if verbose:
        print(f"\n  [2/3] MLP Granger (ComponentMLP with L1 penalty)...")

    splits = expanding_window_cv_splits(n_valid, n_folds=N_CV_FOLDS)

    mlp_mse_r_folds = []
    mlp_mse_u_folds = []
    mlp_importance_all = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        torch.manual_seed(RANDOM_SEED + fold_i)

        # Restricted MLP (SMB lags only)
        model_r = ComponentMLP(n_input_vars=1, n_lags=N_LAGS, hidden_sizes=(64, 32)).to(device)
        model_r = train_mlp(model_r, X_r[train_idx], y[train_idx])
        mse_r = evaluate_model(model_r, X_r[test_idx], y[test_idx])
        mlp_mse_r_folds.append(mse_r)

        # Unrestricted MLP (SMB + HML lags)
        model_u = ComponentMLP(n_input_vars=2, n_lags=N_LAGS, hidden_sizes=(64, 32)).to(device)
        model_u = train_mlp(model_u, X_u[train_idx], y[train_idx])
        mse_u = evaluate_model(model_u, X_u[test_idx], y[test_idx])
        mlp_mse_u_folds.append(mse_u)

        # Variable importance from L1 weights
        importance = model_u.get_variable_importance()
        mlp_importance_all.append(importance)

    mlp_mean_r = np.mean(mlp_mse_r_folds)
    mlp_mean_u = np.mean(mlp_mse_u_folds)
    mlp_improvement = (mlp_mean_r - mlp_mean_u) / mlp_mean_r * 100 if mlp_mean_r > 0 else 0.0
    mlp_avg_importance = np.mean(mlp_importance_all, axis=0)

    if verbose:
        print(f"    MSE restricted: {mlp_mean_r:.6f}")
        print(f"    MSE unrestricted: {mlp_mean_u:.6f}")
        print(f"    MSE improvement: {mlp_improvement:.4f}%")
        print(f"    Input weights: SMB={mlp_avg_importance[0]:.4f}, HML={mlp_avg_importance[1]:.4f}")

    # MLP Permutation test
    if verbose:
        print(f"    Running {n_permutations} permutations...")

    observed_mlp_imp = mlp_mean_r - mlp_mean_u
    perm_mlp_imps = []

    t0 = time.time()
    for p in range(n_permutations):
        np.random.seed(RANDOM_SEED + 1000 + p)
        X_u_perm = X_u.copy()
        perm_order = np.random.permutation(len(X_u_perm))
        X_u_perm[:, :, 1] = X_u_perm[perm_order, :, 1]

        perm_mse_folds = []
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            torch.manual_seed(RANDOM_SEED + 2000 + p * N_CV_FOLDS + fold_i)
            model_perm = ComponentMLP(n_input_vars=2, n_lags=N_LAGS, hidden_sizes=(64, 32)).to(device)
            model_perm = train_mlp(model_perm, X_u_perm[train_idx], y[train_idx], verbose=False)
            mse_perm = evaluate_model(model_perm, X_u_perm[test_idx], y[test_idx])
            perm_mse_folds.append(mse_perm)

        perm_mean = np.mean(perm_mse_folds)
        perm_mlp_imps.append(mlp_mean_r - perm_mean)

        if verbose and (p + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"      Permutation {p+1}/{n_permutations} ({elapsed:.0f}s)")

    perm_mlp_imps = np.array(perm_mlp_imps)
    mlp_p_value = float((np.sum(perm_mlp_imps >= observed_mlp_imp) + 1) / (n_permutations + 1))

    if verbose:
        sig = "***" if mlp_p_value < 0.001 else "**" if mlp_p_value < 0.01 else "*" if mlp_p_value < 0.05 else "ns"
        print(f"    Permutation p-value: {mlp_p_value:.4f} {sig}")

    # --- LSTM Granger (expanding-window CV) ---
    if verbose:
        print(f"\n  [3/3] LSTM Granger (AttentionLSTM)...")

    lstm_mse_r_folds = []
    lstm_mse_u_folds = []
    lstm_causal_graphs = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        torch.manual_seed(RANDOM_SEED + 100 + fold_i)

        # Simple LSTM for restricted
        model_r = nn.Sequential(
            nn.LSTM(1, HIDDEN_SIZE, batch_first=True),
        )
        # Use simpler LSTM for restricted
        class SimpleLSTM(nn.Module):
            def __init__(self, input_dim, hidden_size):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
            def forward(self, x):
                _, (h_n, _) = self.lstm(x)
                return self.fc(h_n.squeeze(0)).squeeze(-1)

        model_r = SimpleLSTM(1, HIDDEN_SIZE).to(device)
        model_r = train_attention_lstm(model_r, X_r[train_idx], y[train_idx])
        mse_r = evaluate_model(model_r, X_r[test_idx], y[test_idx])
        lstm_mse_r_folds.append(mse_r)

        # Attention LSTM for unrestricted
        model_u = AttentionLSTM(n_input_vars=2, n_lags=N_LAGS, hidden_size=HIDDEN_SIZE).to(device)
        model_u = train_attention_lstm(model_u, X_u[train_idx], y[train_idx])
        mse_u = evaluate_model(model_u, X_u[test_idx], y[test_idx])
        lstm_mse_u_folds.append(mse_u)

        # Get causal graph from attention
        _ = model_u(torch.tensor(X_u[test_idx], dtype=torch.float32, device=device))
        causal_graph = model_u.get_causal_graph()
        if causal_graph is not None:
            lstm_causal_graphs.append(causal_graph)

    lstm_mean_r = np.mean(lstm_mse_r_folds)
    lstm_mean_u = np.mean(lstm_mse_u_folds)
    lstm_improvement = (lstm_mean_r - lstm_mean_u) / lstm_mean_r * 100 if lstm_mean_r > 0 else 0.0

    # Average causal graph across folds
    avg_causal_graph = np.mean(lstm_causal_graphs, axis=0) if lstm_causal_graphs else None

    if verbose:
        print(f"    MSE restricted: {lstm_mean_r:.6f}")
        print(f"    MSE unrestricted: {lstm_mean_u:.6f}")
        print(f"    MSE improvement: {lstm_improvement:.4f}%")
        if avg_causal_graph is not None:
            print(f"    Attention graph (HML->SMB): {avg_causal_graph[1, 0]:.4f}")
            print(f"    Attention graph (SMB->HML): {avg_causal_graph[0, 1]:.4f}")

    # LSTM Permutation test
    if verbose:
        print(f"    Running {n_permutations} permutations...")

    observed_lstm_imp = lstm_mean_r - lstm_mean_u
    perm_lstm_imps = []

    t0 = time.time()
    for p in range(n_permutations):
        np.random.seed(RANDOM_SEED + 3000 + p)
        X_u_perm = X_u.copy()
        perm_order = np.random.permutation(len(X_u_perm))
        X_u_perm[:, :, 1] = X_u_perm[perm_order, :, 1]

        perm_mse_folds = []
        for fold_i, (train_idx, test_idx) in enumerate(splits):
            torch.manual_seed(RANDOM_SEED + 4000 + p * N_CV_FOLDS + fold_i)
            model_perm = AttentionLSTM(n_input_vars=2, n_lags=N_LAGS, hidden_size=HIDDEN_SIZE).to(device)
            model_perm = train_attention_lstm(model_perm, X_u_perm[train_idx], y[train_idx])
            mse_perm = evaluate_model(model_perm, X_u_perm[test_idx], y[test_idx])
            perm_mse_folds.append(mse_perm)

        perm_mean = np.mean(perm_mse_folds)
        perm_lstm_imps.append(lstm_mean_r - perm_mean)

        if verbose and (p + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"      Permutation {p+1}/{n_permutations} ({elapsed:.0f}s)")

    perm_lstm_imps = np.array(perm_lstm_imps)
    lstm_p_value = float((np.sum(perm_lstm_imps >= observed_lstm_imp) + 1) / (n_permutations + 1))

    if verbose:
        sig = "***" if lstm_p_value < 0.001 else "**" if lstm_p_value < 0.01 else "*" if lstm_p_value < 0.05 else "ns"
        print(f"    Permutation p-value: {lstm_p_value:.4f} {sig}")

    # Compile results
    result = {
        "regime": regime_name,
        "n_obs": int(n_obs),
        "n_valid_samples": int(n_valid),
        "n_cv_folds": len(splits),
        "linear_granger": linear_result,
        "mlp_granger": {
            "mse_restricted": float(mlp_mean_r),
            "mse_unrestricted": float(mlp_mean_u),
            "mse_improvement_pct": float(mlp_improvement),
            "permutation_p_value": float(mlp_p_value),
            "input_weight_smb": float(mlp_avg_importance[0]),
            "input_weight_hml": float(mlp_avg_importance[1]),
            "hml_importance_ratio": float(mlp_avg_importance[1] / (mlp_avg_importance[0] + 1e-8)),
        },
        "lstm_granger": {
            "mse_restricted": float(lstm_mean_r),
            "mse_unrestricted": float(lstm_mean_u),
            "mse_improvement_pct": float(lstm_improvement),
            "permutation_p_value": float(lstm_p_value),
            "attention_hml_to_smb": float(avg_causal_graph[1, 0]) if avg_causal_graph is not None else None,
            "attention_smb_to_hml": float(avg_causal_graph[0, 1]) if avg_causal_graph is not None else None,
        }
    }

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("NEURAL GRANGER CAUSALITY: PyTorch Implementation")
    print("Tank et al. (2022) style - HML -> SMB analysis")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  N_LAGS = {N_LAGS}")
    print(f"  HIDDEN_SIZE = {HIDDEN_SIZE}")
    print(f"  N_EPOCHS = {N_EPOCHS}")
    print(f"  L1_LAMBDA = {L1_LAMBDA}")
    print(f"  N_PERMUTATIONS = {N_PERMUTATIONS}")
    print(f"  N_CV_FOLDS = {N_CV_FOLDS}")

    t_start = time.time()

    # --- Load data ---
    print("\n[1/4] Loading Fama-French 5-factor daily data (1990-2024)...")
    ff5 = load_ff5_data()

    smb = ff5["SMB"].values
    hml = ff5["HML"].values

    # --- Fit HMM for regime detection ---
    print("\n[2/4] Fitting Student-t HMM (3 regimes, random_state=28)...")
    obs = ff5[["Mkt-RF", "SMB", "HML"]].values
    hmm = StudentTHMM(n_regimes=3, random_state=28)
    hmm.fit(obs)
    regimes = hmm.predict(obs)

    print(f"\n  Regime distribution:")
    for k, name in enumerate(REGIME_NAMES):
        n_k = (regimes == k).sum()
        print(f"    {name}: {n_k} days ({100*n_k/len(regimes):.1f}%)")

    # --- Per-regime Neural Granger tests ---
    print("\n[3/4] Per-regime Neural Granger tests...")
    regime_results = {}

    for k, name in enumerate(REGIME_NAMES):
        mask = regimes == k
        smb_k = smb[mask]
        hml_k = hml[mask]

        result = run_neural_granger_test(smb_k, hml_k, name, n_permutations=N_PERMUTATIONS)
        if result is not None:
            regime_results[name] = result

    # --- Full sample analysis ---
    print("\n[4/4] Full sample Neural Granger test...")
    full_result = run_neural_granger_test(smb, hml, "Full Sample", n_permutations=N_PERMUTATIONS)
    if full_result is not None:
        regime_results["Full Sample"] = full_result

    # --- Compile and save results ---
    elapsed = time.time() - t_start

    results = {
        "metadata": {
            "description": "Neural Granger Causality: HML -> SMB",
            "method": "Tank et al. (2022) style with MLP/LSTM",
            "models": {
                "linear": "OLS F-test",
                "mlp": "ComponentMLP with L1 penalty on input weights",
                "lstm": "AttentionLSTM with multi-head attention"
            },
            "config": {
                "n_lags": N_LAGS,
                "hidden_size": HIDDEN_SIZE,
                "n_epochs": N_EPOCHS,
                "l1_lambda": L1_LAMBDA,
                "n_permutations": N_PERMUTATIONS,
                "n_cv_folds": N_CV_FOLDS,
                "random_seed": RANDOM_SEED,
            },
            "data": {
                "source": "Fama-French 5-factor daily",
                "period": f"{ff5.index[0].date()} to {ff5.index[-1].date()}",
                "n_total_obs": int(len(ff5)),
            },
            "timestamp": datetime.now().isoformat(),
            "runtime_seconds": float(elapsed),
        },
        "regime_detection": {
            "method": "Student-t HMM (3 regimes)",
            "regime_counts": {
                name: int((regimes == k).sum()) for k, name in enumerate(REGIME_NAMES)
            },
        },
        "results": regime_results,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY: Linear Granger R2 vs Neural Granger R2 per Regime")
    print("=" * 80)

    print(f"\n{'Method':<10} | {'Regime':<12} | {'MSE Improv%':>12} | {'p-value':>10} | {'Significant':>12}")
    print("-" * 70)

    for regime_name in ["Normal", "Elevated", "Crisis", "Full Sample"]:
        if regime_name not in regime_results:
            continue
        r = regime_results[regime_name]

        # Linear
        lin = r['linear_granger']
        sig_lin = "Yes" if lin['p_value'] < 0.05 else "No"
        print(f"{'Linear':<10} | {regime_name:<12} | {lin['mse_improvement_pct']:>11.4f}% | {lin['p_value']:>10.6f} | {sig_lin:>12}")

        # MLP
        mlp = r['mlp_granger']
        sig_mlp = "Yes" if mlp['permutation_p_value'] < 0.05 else "No"
        print(f"{'MLP':<10} | {regime_name:<12} | {mlp['mse_improvement_pct']:>11.4f}% | {mlp['permutation_p_value']:>10.4f} | {sig_mlp:>12}")

        # LSTM
        lstm = r['lstm_granger']
        sig_lstm = "Yes" if lstm['permutation_p_value'] < 0.05 else "No"
        print(f"{'LSTM':<10} | {regime_name:<12} | {lstm['mse_improvement_pct']:>11.4f}% | {lstm['permutation_p_value']:>10.4f} | {sig_lstm:>12}")

        print("-" * 70)

    # --- Key findings ---
    print("\n" + "=" * 80)
    print("KEY FINDINGS: Where does nonlinearity matter?")
    print("=" * 80)

    for regime_name in ["Normal", "Elevated", "Crisis"]:
        if regime_name not in regime_results:
            continue
        r = regime_results[regime_name]

        lin_imp = r['linear_granger']['mse_improvement_pct']
        mlp_imp = r['mlp_granger']['mse_improvement_pct']
        lstm_imp = r['lstm_granger']['mse_improvement_pct']

        best_nl = max(mlp_imp, lstm_imp)
        best_method = "MLP" if mlp_imp >= lstm_imp else "LSTM"

        print(f"\n{regime_name}:")
        print(f"  Linear: {lin_imp:.4f}%")
        print(f"  MLP:    {mlp_imp:.4f}%")
        print(f"  LSTM:   {lstm_imp:.4f}%")

        if lin_imp > 0:
            nl_ratio = best_nl / lin_imp
            print(f"  Nonlinear/Linear ratio ({best_method}): {nl_ratio:.2f}x")
            if nl_ratio > 1.2:
                print(f"  --> NONLINEAR EFFECTS: {best_method} captures {(nl_ratio-1)*100:.0f}% more signal")
            elif nl_ratio > 0.8:
                print(f"  --> CONSISTENT: Nonlinear confirms linear finding")
            else:
                print(f"  --> LINEAR DOMINATES: Simpler model is better")
        else:
            if best_nl > 0:
                print(f"  --> NONLINEAR UNIQUE: {best_method} finds signal that linear misses")
            else:
                print(f"  --> NO CAUSAL SIGNAL detected")

    # --- Causal graph from attention ---
    print("\n" + "=" * 80)
    print("CAUSAL GRAPH (from LSTM attention weights)")
    print("=" * 80)

    for regime_name in ["Normal", "Elevated", "Crisis"]:
        if regime_name not in regime_results:
            continue
        lstm = regime_results[regime_name]['lstm_granger']
        if lstm['attention_hml_to_smb'] is not None:
            hml_smb = lstm['attention_hml_to_smb']
            smb_hml = lstm['attention_smb_to_hml']
            print(f"\n{regime_name}:")
            print(f"  HML -> SMB attention: {hml_smb:.4f}")
            print(f"  SMB -> HML attention: {smb_hml:.4f}")
            if hml_smb > smb_hml:
                print(f"  --> Direction: HML causes SMB (ratio: {hml_smb/smb_hml:.2f}x)")
            else:
                print(f"  --> Direction: SMB causes HML (ratio: {smb_hml/hml_smb:.2f}x)")

    print(f"\n\nTotal runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
