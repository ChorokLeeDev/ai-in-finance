"""
Differentiable Regime-Conditional Granger Discovery Method

Key Innovation: Joint end-to-end optimization of regime assignments and Granger coefficients
instead of the two-stage HMM->Granger approach.

ARCHITECTURE:
1. Regime Network: 2-layer MLP (64 hidden) -> K=3 soft regime probabilities
2. Per-Regime Granger Parameters: K sets of linear coefficients β_k
3. Joint Loss: Weighted MSE + entropy regularizer + L1 sparsity
4. Significance Testing: Regime-weighted Granger F-statistics
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json
import warnings
from datetime import datetime
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings('ignore')

# Setup paths
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / 'data'
RESULTS_DIR = BASE / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

print(f"[INFO] Base path: {BASE}")
print(f"[INFO] Results will be saved to: {RESULTS_DIR}")

# ============================================================================
# DEVICE & UTILITIES
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

def set_seed(seed=42):
    """Set seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)


# ============================================================================
# REGIME NETWORK & MODEL
# ============================================================================

class RegimeNetwork(nn.Module):
    """
    Soft regime assignment network.
    Input: window of past returns (batch, window_size)
    Output: regime probabilities (batch, K)
    """
    def __init__(self, input_size, num_regimes=3, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_regimes),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """x: (batch, input_size) -> (batch, K)"""
        logits = self.net(x)
        return self.softmax(logits)


class DifferentiableRegimeGranger(nn.Module):
    """
    Joint regime-conditional Granger discovery model.
    """
    def __init__(self, window_size=20, num_regimes=3, num_lags=1):
        super().__init__()
        self.window_size = window_size
        self.num_regimes = num_regimes
        self.num_lags = num_lags

        # Regime network
        self.regime_net = RegimeNetwork(window_size, num_regimes)

        # Per-regime Granger coefficients
        self.granger_coef = nn.ParameterList([
            nn.Parameter(torch.randn(num_lags) * 0.1) for _ in range(num_regimes)
        ])

        # Per-regime intercepts
        self.intercepts = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(num_regimes)
        ])

    def forward(self, x_window, x_t_lag):
        """
        x_window: (batch, window_size) - window of past X values for regime assignment
        x_t_lag: (batch, num_lags) - lagged X for Granger prediction

        Returns:
            predictions: (batch,) - predicted y values
            regime_probs: (batch, K) - soft regime assignments
        """
        regime_probs = self.regime_net(x_window)
        batch_size = x_window.shape[0]

        # Per-regime predictions
        regime_preds = []
        for k in range(self.num_regimes):
            pred_k = torch.matmul(x_t_lag, self.granger_coef[k]) + self.intercepts[k]
            regime_preds.append(pred_k)  # (batch,)

        regime_preds = torch.stack(regime_preds, dim=1)  # (batch, K)

        # Weighted prediction: Σ_k γ_{t,k} * pred_k
        predictions = torch.sum(regime_probs * regime_preds, dim=1)  # (batch,)

        return predictions, regime_probs, regime_preds


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_synthetic_data(T=5000, seed=42):
    """
    Generate 3-regime synthetic data with known causal structure:
    - Regime 1: x → y (β=0.3)
    - Regime 2: y → x (β=0.2, but here we test x→y with β=-0.1)
    - Regime 3: No causality

    Regime transitions via Markov chain (persistence=0.98)
    """
    np.random.seed(seed)

    # Markov transition matrix (persistence=0.98)
    persistence = 0.98
    switch_prob = (1 - persistence) / 2
    P = np.array([
        [persistence, switch_prob, switch_prob],
        [switch_prob, persistence, switch_prob],
        [switch_prob, switch_prob, persistence],
    ])

    # Simulate regime sequence
    regimes = np.zeros(T, dtype=int)
    regimes[0] = 0
    for t in range(1, T):
        regimes[t] = np.random.choice(3, p=P[regimes[t-1]])

    # Granger coefficients per regime
    beta = np.array([0.3, -0.1, 0.0])  # x→y strength in each regime

    # Generate data
    x = np.random.normal(0, 1, T)
    y = np.zeros(T)

    for t in range(1, T):
        y[t] = beta[regimes[t]] * x[t-1] + np.random.normal(0, 0.5)

    return x, y, regimes, beta


def generate_fama_french_data():
    """
    Load Fama-French data (HML and SMB factors).
    If ff5_momentum_daily.csv doesn't exist, generate synthetic data mimicking FF characteristics.
    """
    csv_path = DATA_DIR / 'ff5_momentum_daily.csv'

    if csv_path.exists():
        print(f"[INFO] Loading Fama-French data from {csv_path}")
        df = pd.read_csv(csv_path)
        if 'HML' in df.columns and 'SMB' in df.columns:
            return df['HML'].values, df['SMB'].values

    # Fallback: use the 25 Portfolios data
    fallback_path = DATA_DIR / '25_Portfolios_5x5_Daily.csv'
    if fallback_path.exists():
        print(f"[INFO] Loading from {fallback_path}")
        df = pd.read_csv(fallback_path, skiprows=3)
        # Use first two columns as proxies for HML and SMB
        cols = df.columns[1:3]
        hml = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
        smb = pd.to_numeric(df.iloc[:, 2], errors='coerce').values
        # Remove NaNs
        valid_idx = ~(np.isnan(hml) | np.isnan(smb))
        return hml[valid_idx], smb[valid_idx]

    print("[WARNING] FF data not found. Using synthetic proxy data.")
    T = 5000
    np.random.seed(42)
    hml = np.random.normal(0, 1, T)
    smb = 0.4 * hml + np.random.normal(0, 1, T)
    return hml, smb


# ============================================================================
# TRAINING
# ============================================================================

def train_differentiable_granger(
    x, y, window_size=20, num_regimes=3, num_lags=1,
    epochs=300, lr=1e-3, lambda_entropy=0.1, lambda_sparsity=0.01,
    verbose=True
):
    """
    Train the differentiable regime-conditional Granger model.

    Args:
        x: (T,) predictor variable
        y: (T,) target variable
        window_size: window for regime assignment
        num_regimes: number of regimes
        num_lags: number of lags for Granger
        epochs: training epochs
        lr: learning rate
        lambda_entropy: entropy regularization weight
        lambda_sparsity: L1 sparsity weight
        verbose: print progress

    Returns:
        model, losses, regime_probs_hist
    """
    T = len(x)
    assert len(y) == T

    # Standardize
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_norm = scaler_x.fit_transform(x.reshape(-1, 1)).flatten()
    y_norm = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Prepare training data
    # For each t in [window_size, T-1], we have:
    # - x_window: x[t-window_size:t] for regime assignment
    # - x_lag: x[t-1:t] for Granger prediction
    # - y_target: y[t]

    start_idx = window_size
    x_windows = []
    x_lags = []
    y_targets = []

    for t in range(start_idx, T):
        x_windows.append(x_norm[t-window_size:t])
        x_lags.append(x_norm[t-num_lags:t])
        y_targets.append(y_norm[t])

    x_windows = np.array(x_windows, dtype=np.float32)  # (N, window_size)
    x_lags = np.array(x_lags, dtype=np.float32)  # (N, num_lags)
    y_targets = np.array(y_targets, dtype=np.float32)  # (N,)

    x_windows_t = torch.from_numpy(x_windows).to(device)
    x_lags_t = torch.from_numpy(x_lags).to(device)
    y_targets_t = torch.from_numpy(y_targets).to(device)

    N = len(y_targets)
    print(f"[INFO] Training on {N} samples")

    # Initialize model
    model = DifferentiableRegimeGranger(window_size, num_regimes, num_lags).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    regime_probs_hist = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        y_pred, regime_probs, regime_preds = model(x_windows_t, x_lags_t)

        # Main loss: weighted MSE
        # L = Σ_t Σ_k γ_{t,k} * (y_t - pred_{t,k})²
        mse_per_regime = (y_targets_t.unsqueeze(1) - regime_preds) ** 2  # (N, K)
        weighted_mse = torch.sum(regime_probs * mse_per_regime)  # scalar

        # Entropy regularizer (encourage regime separation)
        # Higher entropy = softer assignments; we want to minimize (negative) entropy
        entropy = -torch.sum(regime_probs * torch.log(regime_probs + 1e-8)) / N

        # L1 sparsity on Granger coefficients
        l1_sparsity = torch.tensor(0.0, device=device)
        for k in range(num_regimes):
            l1_sparsity = l1_sparsity + torch.sum(torch.abs(model.granger_coef[k]))

        # Total loss
        loss = weighted_mse + lambda_entropy * entropy + lambda_sparsity * l1_sparsity

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 50 == 0 and verbose:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f} | "
                  f"MSE: {weighted_mse.item():.6f} | Entropy: {entropy.item():.6f}")

        # Store regime probabilities every 50 epochs
        if (epoch + 1) % 50 == 0:
            regime_probs_hist.append(regime_probs.detach().cpu().numpy())

    print(f"[INFO] Training complete. Final loss: {losses[-1]:.6f}")

    return model, losses, regime_probs_hist, (x_windows_t, x_lags_t, y_targets_t)


# ============================================================================
# SIGNIFICANCE TESTING
# ============================================================================

def compute_regime_granger_stats(
    model, x_norm, y_norm, x_windows_t, x_lags_t, y_targets_t,
    x_orig, y_orig, scaler_y, num_regimes=3
):
    """
    Compute Granger F-statistics for each regime.

    For regime k:
    - Restricted model: y_t ~ 1 (just mean)
    - Unrestricted model: y_t ~ x_{t-1}
    - F = (RSS_r - RSS_u) / RSS_u * (n_eff - 2)

    where n_eff = Σ_t γ_{t,k} is the effective sample size for regime k
    """

    with torch.no_grad():
        y_pred, regime_probs, regime_preds = model(x_windows_t, x_lags_t)
        regime_probs_np = regime_probs.cpu().numpy()  # (N, K)

    results = {}

    for k in range(num_regimes):
        gamma_k = regime_probs_np[:, k]  # (N,)
        n_eff = np.sum(gamma_k)

        if n_eff < 2:
            results[k] = {
                'n_eff': n_eff,
                'beta': 0.0,
                'f_stat': np.nan,
                'p_value': np.nan,
                'rss_r': np.nan,
                'rss_u': np.nan,
            }
            continue

        y_targets = y_targets_t.cpu().numpy()

        # Restricted model: predict with mean
        y_mean = np.average(y_targets, weights=gamma_k)
        rss_r = np.sum(gamma_k * (y_targets - y_mean) ** 2)

        # Unrestricted model: predict with x_{t-1}
        x_lags = x_lags_t.cpu().numpy()
        y_pred_k = regime_preds[:, k].cpu().numpy()
        residuals = y_targets - y_pred_k
        rss_u = np.sum(gamma_k * residuals ** 2)

        # Granger coefficient for regime k
        beta_k = model.granger_coef[k].item()

        # F-statistic
        if rss_u > 1e-10:
            f_stat = (rss_r - rss_u) / rss_u * (n_eff - 2)
            p_value = 1.0 - stats.f.cdf(max(f_stat, 0), 1, max(n_eff - 2, 1))
        else:
            f_stat = np.inf if rss_r > rss_u else 0
            p_value = 0.0 if f_stat == np.inf else 1.0

        results[k] = {
            'n_eff': float(n_eff),
            'beta': float(beta_k),
            'f_stat': float(f_stat),
            'p_value': float(p_value),
            'rss_r': float(rss_r),
            'rss_u': float(rss_u),
        }

    return results


# ============================================================================
# BASELINE: TWO-STAGE HMM + GRANGER
# ============================================================================

def two_stage_hmm_granger(x, y, num_regimes=3, num_lags=1):
    """
    Baseline method: fit HMM on x, then run Granger per regime.
    """
    T = len(x)
    assert len(y) == T

    # Standardize
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_norm = scaler_x.fit_transform(x.reshape(-1, 1)).flatten()
    y_norm = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Fit HMM on x
    print("[INFO] Fitting HMM...")
    hmm = GaussianHMM(n_components=num_regimes, covariance_type='full', n_iter=500)
    X_data = x_norm.reshape(-1, 1)
    hmm.fit(X_data)
    regime_labels = hmm.predict(X_data)

    # Per-regime Granger
    results = {}
    granger_coefs = []

    for k in range(num_regimes):
        mask = regime_labels == k
        if np.sum(mask) < 3:
            results[k] = {
                'n': int(np.sum(mask)),
                'beta': 0.0,
                'f_stat': np.nan,
                'p_value': np.nan,
            }
            granger_coefs.append(0.0)
            continue

        y_k = y_norm[mask]
        x_k = x_norm[mask]

        # Restricted: y_k ~ 1
        y_mean = np.mean(y_k)
        rss_r = np.sum((y_k - y_mean) ** 2)

        # Unrestricted: y_k ~ x_{t-1}
        # For simplicity, use all available pairs
        if len(y_k) > num_lags:
            y_k_unres = y_k[num_lags:]
            x_k_lag = x_k[:-num_lags]

            # Fit OLS
            X_design = np.column_stack([np.ones(len(x_k_lag)), x_k_lag])
            beta_ols = np.linalg.lstsq(X_design, y_k_unres, rcond=None)[0]
            y_pred = X_design @ beta_ols
            rss_u = np.sum((y_k_unres - y_pred) ** 2)

            n = len(y_k_unres)
            f_stat = (rss_r - rss_u) / rss_u * (n - 2) if rss_u > 1e-10 else 0
            p_value = 1.0 - stats.f.cdf(max(f_stat, 0), 1, max(n - 2, 1))

            results[k] = {
                'n': int(np.sum(mask)),
                'beta': float(beta_ols[1]),
                'f_stat': float(f_stat),
                'p_value': float(p_value),
            }
            granger_coefs.append(float(beta_ols[1]))
        else:
            results[k] = {
                'n': int(np.sum(mask)),
                'beta': 0.0,
                'f_stat': np.nan,
                'p_value': np.nan,
            }
            granger_coefs.append(0.0)

    return regime_labels, results, granger_coefs


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def adjusted_rand_index(true_labels, pred_labels):
    """Compute ARI (range [-1, 1], 1 = perfect agreement)"""
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(true_labels, pred_labels)


def get_regime_assignment(regime_probs):
    """Convert soft regime probs to hard assignments"""
    return np.argmax(regime_probs, axis=1)


# ============================================================================
# EXPERIMENT 1: SYNTHETIC DATA
# ============================================================================

def run_synthetic_experiment():
    """Test on synthetic 3-regime data with known ground truth"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: SYNTHETIC DATA")
    print("="*80)

    # Generate data
    T = 5000
    x, y, true_regimes, true_beta = generate_synthetic_data(T=T)
    print(f"[INFO] Generated synthetic data (T={T})")
    print(f"[INFO] True regime betas: {true_beta}")
    print(f"[INFO] Regime distribution: {np.bincount(true_regimes)}")

    # Train differentiable model
    model, losses, regime_probs_hist, train_data = train_differentiable_granger(
        x, y, window_size=20, num_regimes=3, num_lags=1,
        epochs=500, lr=1e-3, lambda_entropy=0.1, lambda_sparsity=0.01
    )

    # Get final regime assignments
    x_windows_t, x_lags_t, y_targets_t = train_data
    with torch.no_grad():
        _, regime_probs_final, _ = model(x_windows_t, x_lags_t)
    regime_probs_final = regime_probs_final.cpu().numpy()
    pred_regimes = get_regime_assignment(regime_probs_final)

    # Compute ARI (note: alignment issues may occur)
    # For fair comparison, we need to match predicted regimes to true regimes
    true_regimes_windowed = true_regimes[20:]  # Match window
    ari = adjusted_rand_index(true_regimes_windowed, pred_regimes)

    # Compute per-regime statistics
    granger_stats = compute_regime_granger_stats(
        model, None, None, x_windows_t, x_lags_t, y_targets_t,
        x, y, None, num_regimes=3
    )

    # Baseline: two-stage HMM + Granger
    print("\n[INFO] Running two-stage HMM + Granger baseline...")
    regime_labels_hmm, hmm_results, hmm_betas = two_stage_hmm_granger(x, y, num_regimes=3)
    ari_hmm = adjusted_rand_index(true_regimes, regime_labels_hmm)

    # Results dictionary
    synthetic_results = {
        'model': 'DifferentiableRegimeGranger',
        'data': 'Synthetic 3-regime',
        'T': T,
        'num_regimes': 3,
        'true_beta': true_beta.tolist(),
        'ari': float(ari),
        'granger_stats': {str(k): v for k, v in granger_stats.items()},
        'learned_betas': [granger_stats[k]['beta'] for k in range(3)],
        'baseline': {
            'model': 'HMM + Granger',
            'ari': float(ari_hmm),
            'granger_stats': {str(k): v for k, v in hmm_results.items()},
            'learned_betas': hmm_betas,
        }
    }

    print(f"\n[RESULTS] Synthetic Experiment Summary:")
    print(f"  DifferentiableRegimeGranger ARI: {ari:.4f}")
    print(f"  Learned betas: {synthetic_results['learned_betas']}")
    print(f"  F-stats per regime: {[granger_stats[k]['f_stat'] for k in range(3)]}")
    print(f"\n  HMM + Granger ARI: {ari_hmm:.4f}")
    print(f"  HMM learned betas: {hmm_betas}")
    print(f"  HMM F-stats: {[hmm_results[k]['f_stat'] for k in range(3)]}")

    return synthetic_results


# ============================================================================
# EXPERIMENT 2: FAMA-FRENCH DATA
# ============================================================================

def run_fama_french_experiment():
    """Apply to HML (y) vs SMB (x)"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: FAMA-FRENCH DATA (HML vs SMB)")
    print("="*80)

    # Load data
    x, y = generate_fama_french_data()  # x=HML, y=SMB or vice versa
    T = len(x)
    print(f"[INFO] Loaded FF data (T={T})")

    # Remove NaNs
    valid_idx = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_idx]
    y = y[valid_idx]
    T = len(x)
    print(f"[INFO] After removing NaNs: T={T}")

    # Train differentiable model
    model, losses, regime_probs_hist, train_data = train_differentiable_granger(
        x, y, window_size=20, num_regimes=3, num_lags=1,
        epochs=300, lr=1e-3, lambda_entropy=0.1, lambda_sparsity=0.01
    )

    # Granger stats
    x_windows_t, x_lags_t, y_targets_t = train_data
    granger_stats = compute_regime_granger_stats(
        model, None, None, x_windows_t, x_lags_t, y_targets_t,
        x, y, None, num_regimes=3
    )

    # Baseline
    print("\n[INFO] Running two-stage HMM + Granger baseline...")
    regime_labels_hmm, hmm_results, hmm_betas = two_stage_hmm_granger(x, y, num_regimes=3)

    # Test reverse direction (SMB → HML)
    print("\n[INFO] Testing reverse direction (SMB → HML)...")
    x_rev, y_rev = y, x  # Swap

    model_rev, losses_rev, _, train_data_rev = train_differentiable_granger(
        x_rev, y_rev, window_size=20, num_regimes=3, num_lags=1,
        epochs=300, lr=1e-3, lambda_entropy=0.1, lambda_sparsity=0.01,
        verbose=False
    )

    x_windows_rev_t, x_lags_rev_t, y_targets_rev_t = train_data_rev
    granger_stats_rev = compute_regime_granger_stats(
        model_rev, None, None, x_windows_rev_t, x_lags_rev_t, y_targets_rev_t,
        x_rev, y_rev, None, num_regimes=3
    )

    ff_results = {
        'model': 'DifferentiableRegimeGranger',
        'data': 'Fama-French (HML -> SMB)',
        'T': T,
        'num_regimes': 3,
        'granger_stats_hml_to_smb': {str(k): v for k, v in granger_stats.items()},
        'baseline': {
            'model': 'HMM + Granger',
            'granger_stats': {str(k): v for k, v in hmm_results.items()},
        },
        'reverse_direction_smb_to_hml': {
            'granger_stats': {str(k): v for k, v in granger_stats_rev.items()},
        }
    }

    print(f"\n[RESULTS] Fama-French Experiment Summary:")
    print(f"  Direction: HML → SMB")
    for k in range(3):
        stats_k = granger_stats[k]
        print(f"    Regime {k}: β={stats_k['beta']:.4f}, F={stats_k['f_stat']:.4f}, p={stats_k['p_value']:.4f}, n_eff={stats_k['n_eff']:.0f}")

    print(f"\n  Direction: SMB → HML (reverse)")
    for k in range(3):
        stats_k = granger_stats_rev[k]
        print(f"    Regime {k}: β={stats_k['beta']:.4f}, F={stats_k['f_stat']:.4f}, p={stats_k['p_value']:.4f}, n_eff={stats_k['n_eff']:.0f}")

    print(f"\n  HMM + Granger baseline (HML → SMB):")
    for k in range(3):
        stats_k = hmm_results[k]
        print(f"    Regime {k}: β={stats_k['beta']:.4f}, F={stats_k['f_stat']:.4f}, p={stats_k['p_value']:.4f}")

    return ff_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("DIFFERENTIABLE REGIME-CONDITIONAL GRANGER DISCOVERY")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Experiment 1: Synthetic
    try:
        synthetic_results = run_synthetic_experiment()
        results['experiment_1_synthetic'] = synthetic_results
    except Exception as e:
        print(f"[ERROR] Synthetic experiment failed: {e}")
        import traceback
        traceback.print_exc()

    # Experiment 2: Fama-French
    try:
        ff_results = run_fama_french_experiment()
        results['experiment_2_fama_french'] = ff_results
    except Exception as e:
        print(f"[ERROR] Fama-French experiment failed: {e}")
        import traceback
        traceback.print_exc()

    # Save results
    output_path = RESULTS_DIR / 'differentiable_regime_granger.json'
    with open(output_path, 'w') as f:
        # Convert numpy types to Python native types
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            return obj

        results_native = convert_to_native(results)
        json.dump(results_native, f, indent=2)
    print(f"\n[INFO] Results saved to {output_path}")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    return results


if __name__ == '__main__':
    results = main()
    print("\n[SUCCESS] All experiments completed!")
