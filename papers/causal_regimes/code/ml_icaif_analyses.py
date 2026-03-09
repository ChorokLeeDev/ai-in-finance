#!/usr/bin/env python3
"""
ML ICAIF Positioning Analyses
==============================
Generates figures for the new main-body ML section:
1. Attention-LSTM heatmap showing per-lag importance by regime
2. RF feature importance (permutation-based) bar chart by regime
3. TE asymmetry bar chart (HML→SMB vs SMB→HML by regime)

Uses existing results where available; runs new attention LSTM analysis.
"""

import json
import os
import sys
import warnings
import io
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import torch
import torch.nn as nn
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

warnings.filterwarnings("ignore")

BASE_DIR = "/sessions/quirky-vibrant-faraday/mnt/causal_regimes"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

N_LAGS = 9
HIDDEN_SIZE = 32
N_EPOCHS = 150
BATCH_SIZE = 64
LR = 0.001
RANDOM_SEED = 28
REGIME_NAMES = ["Normal", "Elevated", "Crisis"]

device = torch.device("cpu")

# ============================================================================
# HMM + DATA (reuse from existing codebase patterns)
# ============================================================================
def download_ff_data():
    """Download Fama-French 5 factors + Momentum daily."""
    url5 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    urlm = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

    def _load(url, skip, target_col=None):
        resp = urllib.request.urlopen(url)
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        fname = [n for n in z.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        raw = z.read(fname).decode("utf-8")
        lines = raw.strip().split("\n")
        header_idx = None
        for i, line in enumerate(lines):
            if "Mkt-RF" in line or "Mom" in line:
                header_idx = i
                break
        if header_idx is None:
            header_idx = skip
        data_lines = [header_idx] + list(range(header_idx + 1, len(lines)))
        from io import StringIO
        subset = "\n".join([lines[i] for i in data_lines if i < len(lines)])
        df = pd.read_csv(StringIO(subset))
        df.columns = [c.strip() for c in df.columns]
        first_col = df.columns[0]
        df = df[df[first_col].apply(lambda x: str(x).strip().isdigit())]
        df[first_col] = pd.to_datetime(df[first_col], format="%Y%m%d")
        df = df.set_index(first_col)
        df = df.apply(pd.to_numeric, errors="coerce")
        return df

    ff5 = _load(url5, 3)
    mom = _load(urlm, 3)
    mom.columns = ["MOM"] if len(mom.columns) == 1 else [c if c != "Mom   " else "MOM" for c in mom.columns]
    if "Mom" in mom.columns:
        mom = mom.rename(columns={"Mom": "MOM"})

    merged = ff5.join(mom[["MOM"]], how="inner")
    merged = merged.loc["1990-01-02":"2024-12-31"]
    # percentage-unit convention
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    for c in cols:
        if c in merged.columns:
            merged[c] = merged[c].astype(float)
    merged = merged.rename(columns={"Mkt-RF": "MKT-RF"})
    return merged[["MKT-RF", "SMB", "HML", "RMW", "CMA", "MOM"]].dropna()


class StudentTHMM:
    """Student-t HMM with K states, fitted via EM."""

    def __init__(self, K=3, n_dim=6, max_iter=500, tol=1e-6, seed=28):
        self.K = K
        self.n_dim = n_dim
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def fit(self, X):
        np.random.seed(self.seed)
        T, D = X.shape
        self.n_dim = D
        K = self.K

        # K-means init
        centroids, labels = kmeans2(X, K, minit="points", seed=self.seed)
        self.means = centroids.copy()
        self.covs = np.array([np.cov(X[labels == k].T) + 1e-6 * np.eye(D)
                              for k in range(K)])
        self.dfs = np.full(K, 5.0)
        self.A = np.full((K, K), 1.0 / K)
        self.pi = np.full(K, 1.0 / K)

        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            # E-step
            log_B = self._log_emission(X)
            log_alpha, log_beta, ll = self._forward_backward(log_B)
            gamma = self._compute_gamma(log_alpha, log_beta)
            xi = self._compute_xi(log_alpha, log_beta, log_B)

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

            # M-step
            self.pi = gamma[0] / gamma[0].sum()
            self.A = xi.sum(axis=0)
            self.A /= self.A.sum(axis=1, keepdims=True) + 1e-15

            for k in range(K):
                wk = gamma[:, k]
                # Update df-dependent weights
                diff = X - self.means[k]
                maha = np.sum(diff @ np.linalg.pinv(self.covs[k]) * diff, axis=1)
                u = (self.dfs[k] + D) / (self.dfs[k] + maha)

                wku = wk * u
                denom = wk.sum() + 1e-15
                self.means[k] = (wku[:, None] * X).sum(axis=0) / (wku.sum() + 1e-15)

                diff2 = X - self.means[k]
                self.covs[k] = (wku[:, None, None] * (diff2[:, :, None] * diff2[:, None, :])).sum(axis=0) / denom
                self.covs[k] += 1e-6 * np.eye(D)

                # Update df via 1D optimization
                def neg_df_ll(nu):
                    nu = max(nu, 2.01)
                    return -(wk * (
                        gammaln((nu + D) / 2) - gammaln(nu / 2)
                        + (nu / 2) * np.log(nu) - ((nu + D) / 2) * np.log(nu + maha)
                    )).sum()
                res = minimize_scalar(neg_df_ll, bounds=(2.01, 50), method="bounded")
                self.dfs[k] = res.x

        self.ll = ll
        return self

    def _log_emission(self, X):
        T, D = X.shape
        K = self.K
        log_B = np.zeros((T, K))
        for k in range(K):
            diff = X - self.means[k]
            try:
                L = np.linalg.cholesky(self.covs[k])
                log_det = 2 * np.sum(np.log(np.diag(L)))
                solved = np.linalg.solve(L, diff.T).T
                maha = np.sum(solved ** 2, axis=1)
            except np.linalg.LinAlgError:
                eigvals = np.linalg.eigvalsh(self.covs[k])
                log_det = np.sum(np.log(np.maximum(eigvals, 1e-15)))
                maha = np.sum(diff @ np.linalg.pinv(self.covs[k]) * diff, axis=1)

            nu = self.dfs[k]
            log_B[:, k] = (
                gammaln((nu + D) / 2) - gammaln(nu / 2)
                - 0.5 * D * np.log(nu * np.pi) - 0.5 * log_det
                - ((nu + D) / 2) * np.log(1 + maha / nu)
            )
        return log_B

    def _forward_backward(self, log_B):
        T, K = log_B.shape
        log_A = np.log(self.A + 1e-15)
        log_pi = np.log(self.pi + 1e-15)

        log_alpha = np.zeros((T, K))
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = log_B[t, k] + np.logaddexp.reduce(log_alpha[t-1] + log_A[:, k])

        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = np.logaddexp.reduce(log_A[k] + log_B[t+1] + log_beta[t+1])

        ll = np.logaddexp.reduce(log_alpha[-1])
        return log_alpha, log_beta, ll

    def _compute_gamma(self, log_alpha, log_beta):
        log_gamma = log_alpha + log_beta
        log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def _compute_xi(self, log_alpha, log_beta, log_B):
        T, K = log_B.shape
        log_A = np.log(self.A + 1e-15)
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = log_alpha[t, i] + log_A[i, j] + log_B[t+1, j] + log_beta[t+1, j]
            xi[t] -= np.logaddexp.reduce(xi[t].flatten())
        return np.exp(xi)

    def decode(self, X):
        log_B = self._log_emission(X)
        gamma = self._compute_gamma(*self._forward_backward(log_B)[:2])
        return gamma.argmax(axis=1), gamma


def relabel_regimes(X, states, regime_names=REGIME_NAMES):
    """Relabel regimes by data norm (ascending volatility)."""
    K = len(regime_names)
    norms = []
    for k in range(K):
        mask = states == k
        if mask.sum() > 0:
            norms.append(np.linalg.norm(X[mask].std(axis=0)))
        else:
            norms.append(0)
    order = np.argsort(norms)
    mapping = {order[i]: i for i in range(K)}
    return np.array([mapping[s] for s in states])


# ============================================================================
# ATTENTION LSTM
# ============================================================================
class AttentionGrangerLSTM(nn.Module):
    """LSTM with temporal attention for interpretable lag importance."""

    def __init__(self, input_dim, hidden_size=32, n_lags=9):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, return_attention=False):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        attn_scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len)
        context = (lstm_out * attn_weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden)
        pred = self.fc(context).squeeze(-1)
        if return_attention:
            return pred, attn_weights
        return pred


def create_lag_features(smb, hml, n_lags=N_LAGS):
    """Create (y, X_restricted, X_unrestricted) from SMB, HML arrays."""
    T = len(smb)
    y = smb[n_lags:]
    X_r = np.column_stack([smb[n_lags - i - 1: T - i - 1] for i in range(n_lags)])
    X_u = np.column_stack([X_r] + [hml[n_lags - i - 1: T - i - 1] for i in range(n_lags)])
    return y, X_r, X_u


def create_seq_features(smb, hml, n_lags=N_LAGS):
    """Create sequential input for LSTM: (N, n_lags, 2) where channels are [SMB, HML]."""
    T = len(smb)
    N = T - n_lags
    X_r = np.zeros((N, n_lags, 1))
    X_u = np.zeros((N, n_lags, 2))
    y = smb[n_lags:]

    for i in range(N):
        for lag in range(n_lags):
            X_r[i, lag, 0] = smb[i + n_lags - lag - 1]
            X_u[i, lag, 0] = smb[i + n_lags - lag - 1]
            X_u[i, lag, 1] = hml[i + n_lags - lag - 1]
    return y, X_r, X_u


def train_attention_lstm(X, y, n_epochs=N_EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    """Train attention LSTM and extract attention weights."""
    input_dim = X.shape[2]
    model = AttentionGrangerLSTM(input_dim, HIDDEN_SIZE, N_LAGS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_t = torch.FloatTensor(X).to(device)
    y_t = torch.FloatTensor(y).to(device)

    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), batch_size):
            idx = perm[i:i + batch_size]
            pred = model(X_t[idx])
            loss = loss_fn(pred, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Extract attention weights
    model.eval()
    with torch.no_grad():
        _, attn = model(X_t, return_attention=True)

    return model, attn.cpu().numpy()


# ============================================================================
# FIGURE 1: COMBINED ATTENTION + FEATURE IMPORTANCE
# ============================================================================
def generate_ml_diagnostic_figure(regime_attention, regime_importance, regime_names):
    """Generate combined attention weights + RF importance figure."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
    colors_attn = ["#2E75B6", "#E67E22", "#C0392B"]
    colors_imp = ["#27AE60", "#E67E22", "#C0392B"]
    lag_labels = [f"Lag {i+1}" for i in range(N_LAGS)]

    for i, rname in enumerate(regime_names):
        # Top row: Attention weights
        ax = axes[0, i]
        attn = regime_attention[rname]
        mean_attn = attn.mean(axis=0)
        std_attn = attn.std(axis=0) / np.sqrt(len(attn))
        ax.bar(range(N_LAGS), mean_attn, color=colors_attn[i], alpha=0.8,
               yerr=std_attn, capsize=2, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{rname}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Attention Weight" if i == 0 else "", fontsize=9)
        ax.set_xticks(range(N_LAGS))
        ax.set_xticklabels([str(j+1) for j in range(N_LAGS)], fontsize=7)
        ax.set_ylim(0, max(mean_attn) * 1.4)
        ax.axhline(1/N_LAGS, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.tick_params(axis='y', labelsize=7)

        # Bottom row: RF feature importance
        ax2 = axes[1, i]
        imp = regime_importance[rname]
        smb_imp = [imp.get(f"SMB_lag{j+1}", 0) for j in range(N_LAGS)]
        hml_imp = [imp.get(f"HML_lag{j+1}", 0) for j in range(N_LAGS)]
        x = np.arange(N_LAGS)
        w = 0.35
        ax2.bar(x - w/2, smb_imp, w, label="SMB lags", color="#3498DB", alpha=0.8, edgecolor="white", linewidth=0.5)
        ax2.bar(x + w/2, hml_imp, w, label="HML lags", color="#E74C3C", alpha=0.8, edgecolor="white", linewidth=0.5)
        ax2.set_xlabel("Lag" if i == 1 else "", fontsize=9)
        ax2.set_ylabel("RF Importance" if i == 0 else "", fontsize=9)
        ax2.set_xticks(range(N_LAGS))
        ax2.set_xticklabels([str(j+1) for j in range(N_LAGS)], fontsize=7)
        ax2.tick_params(axis='y', labelsize=7)
        if i == 2:
            ax2.legend(fontsize=7, loc="upper right")

    axes[0, 0].set_ylabel("LSTM Attention\nWeight", fontsize=9)
    axes[1, 0].set_ylabel("RF Permutation\nImportance", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ml_attention_importance.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================================
# FIGURE 2: TE ASYMMETRY
# ============================================================================
def generate_te_asymmetry_figure():
    """Generate TE asymmetry bar chart from paper's reported values."""
    # Values from the paper (Table tab:te)
    regimes = ["Normal", "Elevated", "Crisis"]
    hml_to_smb_z = [2.45, 2.41, 1.01]
    smb_to_hml_z = [5.37, 1.65, 1.22]
    hml_to_smb_p = [0.007, 0.008, 0.157]
    smb_to_hml_p = [1e-6, 0.049, 0.111]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(regimes))
    w = 0.32

    bars1 = ax.bar(x - w/2, hml_to_smb_z, w, label="HML→SMB (forward)",
                   color="#2E75B6", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + w/2, smb_to_hml_z, w, label="SMB→HML (reverse)",
                   color="#E74C3C", alpha=0.85, edgecolor="white", linewidth=0.5)

    # Add significance stars
    for i in range(3):
        if hml_to_smb_p[i] < 0.01:
            ax.text(x[i] - w/2, hml_to_smb_z[i] + 0.15, "**", ha="center", fontsize=9, fontweight="bold")
        elif hml_to_smb_p[i] < 0.05:
            ax.text(x[i] - w/2, hml_to_smb_z[i] + 0.15, "*", ha="center", fontsize=9, fontweight="bold")
        if smb_to_hml_p[i] < 0.001:
            ax.text(x[i] + w/2, smb_to_hml_z[i] + 0.15, "***", ha="center", fontsize=9, fontweight="bold")
        elif smb_to_hml_p[i] < 0.01:
            ax.text(x[i] + w/2, smb_to_hml_z[i] + 0.15, "**", ha="center", fontsize=9, fontweight="bold")
        elif smb_to_hml_p[i] < 0.05:
            ax.text(x[i] + w/2, smb_to_hml_z[i] + 0.15, "*", ha="center", fontsize=9, fontweight="bold")

    ax.axhline(1.96, color="gray", ls="--", lw=0.8, alpha=0.6, label="$z = 1.96$ (5% threshold)")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=10)
    ax.set_ylabel("Transfer Entropy $z$-score", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 6.5)

    # Annotate the asymmetry
    ax.annotate("Nonlinear reverse\nchannel dominant",
                xy=(0 + w/2, 5.37), xytext=(0.8, 5.8),
                fontsize=8, fontstyle="italic", color="#C0392B",
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "te_asymmetry_by_regime.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("ML ICAIF Analyses")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading data...")
    df = download_ff_data()
    X = df.values
    smb = df["SMB"].values
    hml = df["HML"].values
    print(f"  Loaded {len(df)} days, {len(df.columns)} factors")

    # 2. Fit HMM
    print("\n[2/4] Fitting Student-t HMM (seed={})...".format(RANDOM_SEED))
    hmm = StudentTHMM(K=3, n_dim=X.shape[1], seed=RANDOM_SEED)
    hmm.fit(X)
    states_raw, gamma = hmm.decode(X)
    states = relabel_regimes(X, states_raw)
    for k, name in enumerate(REGIME_NAMES):
        print(f"  {name}: {(states == k).sum()} days ({(states == k).mean()*100:.1f}%)")

    # 3. Run attention LSTM per regime
    print("\n[3/4] Running Attention LSTM per regime...")
    regime_attention = {}
    regime_importance = {}

    # Also load existing RF importance from neural_granger_results.json
    with open(os.path.join(RESULTS_DIR, "neural_granger_results.json")) as f:
        ng_results = json.load(f)

    # Map neural_granger regime names to paper's regime names
    ng_regime_map = {"Calm": "Normal", "Transition": "Elevated", "Crisis": "Crisis"}

    for k, rname in enumerate(REGIME_NAMES):
        mask = states == k
        n_regime = mask.sum()
        print(f"\n  --- {rname} regime (n={n_regime}) ---")

        smb_r = smb[mask]
        hml_r = hml[mask]

        if n_regime < N_LAGS + 50:
            print(f"  Skipping (too few observations)")
            regime_attention[rname] = np.ones((10, N_LAGS)) / N_LAGS
            regime_importance[rname] = {f"SMB_lag{j+1}": 0 for j in range(N_LAGS)}
            regime_importance[rname].update({f"HML_lag{j+1}": 0 for j in range(N_LAGS)})
            continue

        # Create sequential features for attention LSTM (unrestricted: SMB + HML)
        y, X_r_seq, X_u_seq = create_seq_features(smb_r, hml_r)
        print(f"  Training Attention LSTM (unrestricted, {len(y)} samples)...")

        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        model, attn_weights = train_attention_lstm(X_u_seq, y)
        regime_attention[rname] = attn_weights
        print(f"  Attention weights shape: {attn_weights.shape}")
        print(f"  Mean attention by lag: {attn_weights.mean(axis=0).round(3)}")

        # Get RF importance from existing results
        ng_rname = [k for k, v in ng_regime_map.items() if v == rname]
        if ng_rname and ng_rname[0] in ng_results["regimes"]:
            rf_data = ng_results["regimes"][ng_rname[0]]["rf_granger"]
            regime_importance[rname] = rf_data["per_lag_importance"]
            print(f"  RF importance loaded from neural_granger_results.json")
        else:
            # Fallback: uniform
            regime_importance[rname] = {f"SMB_lag{j+1}": 0.01 for j in range(N_LAGS)}
            regime_importance[rname].update({f"HML_lag{j+1}": 0.01 for j in range(N_LAGS)})

    # 4. Generate figures
    print("\n[4/4] Generating figures...")
    fig1 = generate_ml_diagnostic_figure(regime_attention, regime_importance, REGIME_NAMES)
    fig2 = generate_te_asymmetry_figure()

    # Save results
    results = {
        "attention_by_regime": {
            rname: {
                "mean": regime_attention[rname].mean(axis=0).tolist(),
                "std": regime_attention[rname].std(axis=0).tolist(),
                "n_samples": len(regime_attention[rname]),
                "peak_lag": int(regime_attention[rname].mean(axis=0).argmax()) + 1,
                "peak_weight": float(regime_attention[rname].mean(axis=0).max()),
                "uniform_baseline": 1.0 / N_LAGS,
            }
            for rname in REGIME_NAMES
        },
        "rf_importance_by_regime": {
            rname: regime_importance[rname] for rname in REGIME_NAMES
        },
        "figures": [fig1, fig2],
    }

    out_path = os.path.join(RESULTS_DIR, "ml_icaif_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
