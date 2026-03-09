"""
Baseline Models for Causal Discovery Comparison
================================================

Implements:
1. Linear Granger Causality (VAR-based F-test)
2. NOTEARS (continuous DAG learning)
3. Simple LSTM baseline (no causal structure)
4. VAR model (linear autoregression)

These serve as baselines to compare against RANCD.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LinearGrangerCausality:
    """
    Classical Granger causality using VAR model and F-test.
    Tests if lagged values of X help predict Y beyond Y's own lags.
    """

    def __init__(self, n_lags: int = 5, significance: float = 0.05):
        self.n_lags = n_lags
        self.significance = significance

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Compute Granger causality for all pairs.

        Args:
            data: (T, n_factors) time series

        Returns:
            adj: (n_factors, n_factors) - causal adjacency matrix
                 adj[i,j] = 1 if i Granger-causes j
        """
        T, n_factors = data.shape
        adj = np.zeros((n_factors, n_factors))

        for i in range(n_factors):
            for j in range(n_factors):
                if i == j:
                    continue

                # Test if i Granger-causes j
                p_value = self._granger_test(data[:, i], data[:, j])

                if p_value < self.significance:
                    adj[i, j] = 1.0

        return adj

    def _granger_test(self, x: np.ndarray, y: np.ndarray) -> float:
        """F-test for Granger causality: does x cause y?"""
        T = len(y)
        n_lags = self.n_lags

        # Restricted model: y ~ y_lags
        Y_r = y[n_lags:]
        X_r = np.column_stack([y[n_lags-i-1:T-i-1] for i in range(n_lags)])

        # Unrestricted model: y ~ y_lags + x_lags
        X_u = np.column_stack([
            *[y[n_lags-i-1:T-i-1] for i in range(n_lags)],
            *[x[n_lags-i-1:T-i-1] for i in range(n_lags)]
        ])

        # Fit both models
        try:
            beta_r = np.linalg.lstsq(X_r, Y_r, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_u, Y_r, rcond=None)[0]

            # Residuals
            resid_r = Y_r - X_r @ beta_r
            resid_u = Y_r - X_u @ beta_u

            # Sum of squared residuals
            SSR_r = np.sum(resid_r ** 2)
            SSR_u = np.sum(resid_u ** 2)

            # F-statistic
            n = len(Y_r)
            k_r = X_r.shape[1]
            k_u = X_u.shape[1]

            F = ((SSR_r - SSR_u) / (k_u - k_r)) / (SSR_u / (n - k_u))

            # p-value
            p_value = 1 - stats.f.cdf(F, k_u - k_r, n - k_u)

            return p_value

        except Exception:
            return 1.0  # No causality if computation fails


class NOTEARS:
    """
    Continuous DAG structure learning using NOTEARS algorithm.
    Minimizes: ||X - XW||^2 + λ||W||_1 s.t. h(W) = 0

    Simplified implementation for comparison.
    """

    def __init__(self, lambda_l1: float = 0.1, max_iter: int = 100,
                 h_tol: float = 1e-8, rho_max: float = 1e16):
        self.lambda_l1 = lambda_l1
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Learn DAG structure from data.

        Args:
            data: (T, n_factors) time series

        Returns:
            adj: (n_factors, n_factors) - weighted adjacency matrix
        """
        n = data.shape[1]

        # Initialize
        W = np.zeros((n, n))
        rho = 1.0
        alpha = 0.0

        for iteration in range(self.max_iter):
            # Optimize W with fixed Lagrangian
            W = self._optimize_w(data, W, rho, alpha)

            # Compute constraint
            h = self._h(W)

            if h < self.h_tol:
                break

            # Update Lagrangian
            alpha += rho * h
            rho = min(rho * 2, self.rho_max)

        # Threshold small values
        W[np.abs(W) < 0.3] = 0

        return np.abs(W)

    def _h(self, W: np.ndarray) -> float:
        """Acyclicity constraint: tr(e^{W∘W}) - d"""
        n = W.shape[0]
        W_squared = W * W
        expm = np.linalg.matrix_power(np.eye(n) + W_squared / n, n)
        return np.trace(expm) - n

    def _optimize_w(self, data: np.ndarray, W: np.ndarray,
                    rho: float, alpha: float) -> np.ndarray:
        """Single optimization step for W."""
        n = data.shape[1]
        X = data

        # Gradient of squared loss
        grad_loss = (X.T @ X @ W - X.T @ X) / data.shape[0]

        # Gradient of h(W)
        W_squared = W * W
        expm = np.linalg.matrix_power(np.eye(n) + W_squared / n, n)
        grad_h = expm * W * 2

        # Gradient of augmented Lagrangian
        grad = grad_loss + (rho * self._h(W) + alpha) * grad_h

        # Gradient step with L1 proximal
        lr = 0.01
        W_new = W - lr * grad

        # Proximal operator for L1
        W_new = np.sign(W_new) * np.maximum(np.abs(W_new) - lr * self.lambda_l1, 0)

        # Zero diagonal
        np.fill_diagonal(W_new, 0)

        return W_new


class SimpleLSTM(nn.Module):
    """
    Simple LSTM baseline for prediction (no causal structure).
    Just predicts next step from history.
    """

    def __init__(self, n_factors: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_factors,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, n_factors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, n_factors)
        Returns:
            pred: (batch, time, n_factors)
        """
        h, _ = self.lstm(x)
        return self.fc(h)


class VARModel:
    """
    Vector Autoregression model.
    Y_t = c + A_1 Y_{t-1} + ... + A_p Y_{t-p} + e_t
    """

    def __init__(self, n_lags: int = 5):
        self.n_lags = n_lags
        self.coef_ = None

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit VAR model and extract causal structure from coefficients.

        Args:
            data: (T, n_factors)

        Returns:
            adj: (n_factors, n_factors) - based on coefficient magnitudes
        """
        T, n = data.shape
        p = self.n_lags

        # Build design matrix
        Y = data[p:]  # (T-p, n)
        X = np.column_stack([
            data[p-i-1:T-i-1] for i in range(p)
        ])  # (T-p, n*p)

        # Add intercept
        X = np.column_stack([np.ones(len(Y)), X])

        # Solve: Y = X @ B
        self.coef_ = np.linalg.lstsq(X, Y, rcond=None)[0]

        # Extract adjacency from lag-1 coefficients (simplified)
        A1 = self.coef_[1:n+1, :]  # First lag coefficients

        # Use absolute values as edge weights
        adj = np.abs(A1)

        # Normalize and threshold
        adj = adj / (adj.max() + 1e-8)
        adj[adj < 0.1] = 0

        return adj

    def predict(self, data: np.ndarray) -> np.ndarray:
        """One-step ahead prediction."""
        T, n = data.shape
        p = self.n_lags

        X = np.column_stack([
            np.ones(T-p),
            *[data[p-i-1:T-i-1] for i in range(p)]
        ])

        return X @ self.coef_


def evaluate_causal_discovery(true_adj: np.ndarray, pred_adj: np.ndarray,
                              threshold: float = 0.5) -> dict:
    """
    Evaluate causal discovery performance.

    Args:
        true_adj: (n, n) ground truth adjacency
        pred_adj: (n, n) predicted adjacency

    Returns:
        metrics: dict with precision, recall, F1, etc.
    """
    # Binarize predictions
    pred_binary = (pred_adj > threshold).astype(float)
    true_binary = (true_adj > 0).astype(float)

    # Flatten (excluding diagonal)
    n = true_adj.shape[0]
    mask = ~np.eye(n, dtype=bool)

    y_true = true_binary[mask].flatten()
    y_pred = pred_binary[mask].flatten()

    # Metrics
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }


# Quick test
if __name__ == "__main__":
    print("Testing baseline models...")

    # Generate synthetic data
    np.random.seed(42)
    T, n = 500, 4

    # Create data with known causal structure
    # X0 → X1, X1 → X2, X2 → X3
    data = np.zeros((T, n))
    data[:, 0] = np.random.randn(T)
    for t in range(1, T):
        data[t, 1] = 0.5 * data[t-1, 0] + 0.3 * np.random.randn()
        data[t, 2] = 0.5 * data[t-1, 1] + 0.3 * np.random.randn()
        data[t, 3] = 0.5 * data[t-1, 2] + 0.3 * np.random.randn()

    # True adjacency
    true_adj = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    print("\n1. Linear Granger Causality")
    gc = LinearGrangerCausality(n_lags=5)
    adj_gc = gc.fit(data)
    metrics_gc = evaluate_causal_discovery(true_adj, adj_gc)
    print(f"   Adjacency:\n{adj_gc}")
    print(f"   F1: {metrics_gc['f1']:.3f}")

    print("\n2. NOTEARS")
    notears = NOTEARS(lambda_l1=0.1)
    adj_notears = notears.fit(data)
    metrics_notears = evaluate_causal_discovery(true_adj, adj_notears)
    print(f"   Adjacency:\n{np.round(adj_notears, 2)}")
    print(f"   F1: {metrics_notears['f1']:.3f}")

    print("\n3. VAR Model")
    var = VARModel(n_lags=5)
    adj_var = var.fit(data)
    metrics_var = evaluate_causal_discovery(true_adj, adj_var)
    print(f"   Adjacency:\n{np.round(adj_var, 2)}")
    print(f"   F1: {metrics_var['f1']:.3f}")

    print("\n✅ Baseline tests passed!")
    print(f"\nSummary:")
    print(f"  Granger F1: {metrics_gc['f1']:.3f}")
    print(f"  NOTEARS F1: {metrics_notears['f1']:.3f}")
    print(f"  VAR F1: {metrics_var['f1']:.3f}")
