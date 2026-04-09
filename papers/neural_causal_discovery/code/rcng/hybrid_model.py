"""
Hybrid RCNG: Two-Stage Warm-Start + Joint Fine-Tuning

Strategy:
1. Stage 1: Volatility-based regime detection (warm-start)
2. Stage 2: Per-regime neural Granger training
3. Stage 3: Joint fine-tuning (refine regime boundaries + adjust graphs)

This combines:
- Reliability of two-stage (good initialization)
- Flexibility of joint learning (boundary refinement)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.cluster import KMeans


class HybridRCNG(nn.Module):
    """
    Hybrid Regime-Conditional Neural Granger.

    Three phases:
    1. Initialize regimes via volatility clustering
    2. Train per-regime neural Granger (warm-start)
    3. Joint fine-tuning with soft regime boundaries
    """

    def __init__(
        self,
        n_factors: int,
        n_lags: int = 5,
        n_regimes: int = 3,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.n_factors = n_factors
        self.n_lags = n_lags
        self.n_regimes = n_regimes
        self.hidden_dim = hidden_dim

        # Per-regime adjacency matrices (learnable)
        self.adj_logits = nn.Parameter(torch.zeros(n_regimes, n_factors, n_factors))
        self.register_buffer('diag_mask', 1 - torch.eye(n_factors))

        # Per-regime predictors
        self.predictors = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(n_lags * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
                for _ in range(n_factors)
            ])
            for _ in range(n_regimes)
        ])

        # Regime classifier (for joint fine-tuning)
        self.regime_classifier = nn.Sequential(
            nn.Linear(n_factors * 2, hidden_dim),  # x + volatility features
            nn.ReLU(),
            nn.Linear(hidden_dim, n_regimes),
        )

        # Store initial regime assignments (from warm-start)
        self.register_buffer('initial_regimes', torch.zeros(1, dtype=torch.long))

    def get_adj(self, k: int) -> torch.Tensor:
        """Get soft adjacency for regime k."""
        return torch.sigmoid(self.adj_logits[k]) * self.diag_mask

    def create_lagged_data(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, time, n_factors = x.shape
        lagged = []
        for lag in range(1, self.n_lags + 1):
            lagged.append(x[:, self.n_lags - lag:-lag, :])
        x_lagged = torch.stack(lagged, dim=2)
        y_target = x[:, self.n_lags:, :]
        return x_lagged, y_target

    def predict_regime(self, x_lagged: torch.Tensor, k: int) -> torch.Tensor:
        """Predict using regime k's model."""
        batch, time, n_lags, n_factors = x_lagged.shape
        adj = self.get_adj(k)

        preds = []
        for j in range(n_factors):
            self_lags = x_lagged[:, :, :, j]
            cross_lags = (x_lagged * adj[:, j].view(1, 1, 1, n_factors)).sum(dim=-1)
            inputs = torch.cat([self_lags, cross_lags], dim=-1)
            pred_j = self.predictors[k][j](inputs)
            preds.append(pred_j)

        return torch.cat(preds, dim=-1)

    def forward(self, x: torch.Tensor, mode: str = 'joint') -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, time, n_factors)
            mode: 'warmstart' (use initial regimes) or 'joint' (use classifier)
        """
        x_lagged, y_target = self.create_lagged_data(x)
        batch, time, _, _ = x_lagged.shape

        # Compute volatility features for regime classification
        x_trimmed = x[:, self.n_lags:, :]
        vol_features = x_trimmed.abs()  # Simple volatility proxy
        classifier_input = torch.cat([x_trimmed, vol_features], dim=-1)

        # Get regime probabilities
        regime_logits = self.regime_classifier(classifier_input)
        regime_probs = F.softmax(regime_logits, dim=-1)

        # Get predictions from each regime
        regime_preds = []
        for k in range(self.n_regimes):
            pred_k = self.predict_regime(x_lagged, k)
            regime_preds.append(pred_k)
        regime_preds = torch.stack(regime_preds, dim=0)  # (K, batch, time, n_factors)

        # Weighted prediction
        predictions = torch.einsum('kbtf,btk->btf', regime_preds, regime_probs)

        # Get adjacency matrices
        adjs = torch.stack([self.get_adj(k) for k in range(self.n_regimes)], dim=0)

        return {
            'predictions': predictions,
            'regime_probs': regime_probs,
            'adj': adjs,
            'y_target': y_target,
        }

    def warmstart_init(self, data: np.ndarray, n_clusters: int = 3):
        """
        Phase 1: Initialize regimes using Gaussian HMM.

        Args:
            data: (T, n_factors) numpy array
        """
        from hmmlearn import hmm

        # Fit Gaussian HMM
        model_hmm = hmm.GaussianHMM(
            n_components=n_clusters,
            covariance_type='full',
            n_iter=100,
            random_state=42
        )
        model_hmm.fit(data)
        regime_labels = model_hmm.predict(data)

        # Order regimes by volatility (0=low, 2=high)
        regime_vols = []
        for k in range(n_clusters):
            mask = regime_labels == k
            if mask.sum() > 0:
                regime_vols.append((k, np.std(data[mask])))
            else:
                regime_vols.append((k, 0))
        regime_vols.sort(key=lambda x: x[1])
        remap = {regime_vols[i][0]: i for i in range(n_clusters)}
        regime_labels = np.array([remap[r] for r in regime_labels])

        self.initial_regimes = torch.tensor(regime_labels, dtype=torch.long)

        return regime_labels

    def warmstart_train(
        self,
        data: np.ndarray,
        regime_labels: np.ndarray,
        n_epochs: int = 50,
        lr: float = 1e-3,
    ):
        """
        Phase 2: Train per-regime models separately (warm-start).
        """
        device = next(self.parameters()).device
        T, n_factors = data.shape

        for k in range(self.n_regimes):
            # Get data for regime k
            mask = regime_labels[self.n_lags:] == k
            if mask.sum() < 50:
                print(f"  Regime {k}: insufficient data ({mask.sum()}), skipping")
                continue

            # Create training data for this regime
            x_all = torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(0)
            x_lagged, y_target = self.create_lagged_data(x_all)

            # Extract regime-specific indices
            regime_indices = torch.tensor(np.where(mask)[0], device=device)

            # Optimizer for regime k only
            params_k = list(self.predictors[k].parameters()) + [self.adj_logits]
            optimizer = torch.optim.Adam(params_k, lr=lr)

            for epoch in range(n_epochs):
                optimizer.zero_grad()

                # Predict using regime k's model
                pred_k = self.predict_regime(x_lagged, k)

                # Loss only on regime k's time points
                pred_regime = pred_k[:, regime_indices, :]
                target_regime = y_target[:, regime_indices, :]

                loss = F.mse_loss(pred_regime, target_regime)

                # Sparsity
                adj_k = self.get_adj(k)
                loss = loss + 0.01 * adj_k.abs().mean()

                loss.backward()
                optimizer.step()

            adj_k = self.get_adj(k)
            n_edges = (adj_k > 0.5).sum().item()
            print(f"  Regime {k}: {mask.sum()} samples, {n_edges} edges, loss={loss.item():.4f}")

    def joint_finetune(
        self,
        data: np.ndarray,
        n_epochs: int = 50,
        lr: float = 5e-4,
        regime_anchor_weight: float = 0.5,  # How much to trust warm-start regimes
    ):
        """
        Phase 3: Joint fine-tuning with soft regime boundaries.
        Uses warm-start regimes as anchor to prevent drift.
        """
        device = next(self.parameters()).device
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        T, n_factors = data.shape
        window_size = 100
        n_windows = T - window_size + 1

        windows = np.array([data[i:i+window_size] for i in range(n_windows)])
        windows_tensor = torch.tensor(windows, dtype=torch.float32, device=device)

        # Get warm-start regime labels for anchoring
        warmstart_regimes = self.initial_regimes.to(device)

        for epoch in range(n_epochs):
            perm = torch.randperm(len(windows_tensor))
            epoch_loss = []

            for i in range(0, len(windows_tensor), 32):
                batch_idx = perm[i:i+32]
                batch = windows_tensor[batch_idx]

                optimizer.zero_grad()

                out = self.forward(batch, mode='joint')
                predictions = out['predictions']
                regime_probs = out['regime_probs']
                adj = out['adj']
                y_target = out['y_target']

                # Prediction loss
                L_pred = F.mse_loss(predictions, y_target)

                # Sparsity
                L_sparse = adj.abs().mean()

                # Diversity
                L_diverse = torch.tensor(0.0, device=device)
                for k1 in range(self.n_regimes):
                    for k2 in range(k1+1, self.n_regimes):
                        L_diverse = L_diverse - (adj[k1] - adj[k2]).pow(2).sum().sqrt()
                L_diverse = L_diverse / max(1, self.n_regimes * (self.n_regimes - 1) // 2)

                # Entropy regularization
                avg_prob = regime_probs.mean(dim=(0, 1))
                L_entropy = ((avg_prob - 1/self.n_regimes) ** 2).sum()

                # ANCHOR LOSS: Encourage regime assignments to stay close to warm-start
                # Get warm-start labels for this batch's windows
                batch_warmstart = []
                for idx in batch_idx:
                    window_labels = warmstart_regimes[idx + self.n_lags : idx + window_size]
                    batch_warmstart.append(window_labels)
                batch_warmstart = torch.stack(batch_warmstart, dim=0)  # (batch, time-n_lags)

                # Cross-entropy loss to anchor to warm-start
                # regime_probs: (batch, time-n_lags, K)
                L_anchor = F.cross_entropy(
                    regime_probs.reshape(-1, self.n_regimes),
                    batch_warmstart.reshape(-1),
                )

                loss = L_pred + 0.01 * L_sparse + 0.1 * L_diverse + 0.1 * L_entropy + regime_anchor_weight * L_anchor
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                epoch_loss.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}: loss={np.mean(epoch_loss):.4f}")

    def get_adjacency_matrices(self) -> np.ndarray:
        adjs = []
        for k in range(self.n_regimes):
            adj_k = self.get_adj(k).detach().cpu().numpy()
            adjs.append(adj_k)
        return np.stack(adjs, axis=0)

    def get_regime_assignments(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            out = self.forward(x, mode='joint')
            return out['regime_probs'].argmax(dim=-1).cpu().numpy()


def train_hybrid_rcng(
    data: np.ndarray,
    n_regimes: int = 3,
    n_factors: int = 6,
    n_lags: int = 5,
    warmstart_epochs: int = 50,
    finetune_epochs: int = 50,
    verbose: bool = True,
) -> HybridRCNG:
    """
    Full hybrid training pipeline.
    """
    model = HybridRCNG(n_factors=n_factors, n_lags=n_lags, n_regimes=n_regimes)

    if verbose:
        print("Phase 1: Volatility-based regime initialization...")
    regime_labels = model.warmstart_init(data, n_clusters=n_regimes)

    if verbose:
        props = {k: (regime_labels == k).mean() for k in range(n_regimes)}
        print(f"  Regime proportions: {props}")

    if verbose:
        print("\nPhase 2: Per-regime warm-start training...")
    model.warmstart_train(data, regime_labels, n_epochs=warmstart_epochs)

    if verbose:
        print("\nPhase 3: Joint fine-tuning...")
    model.joint_finetune(data, n_epochs=finetune_epochs)

    return model


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    from synthetic_data import RegimeSwitchingDGP
    from sklearn.metrics import adjusted_rand_score

    print("=" * 60)
    print("Testing Hybrid RCNG")
    print("=" * 60)

    dgp = RegimeSwitchingDGP(seed=42)
    data, true_regimes, true_adj = dgp.generate(T=1500)

    print(f"\nTrue regime proportions: {dict(zip(range(3), [(true_regimes==k).mean() for k in range(3)]))}")
    print(f"True edge counts: {[int(true_adj[k].sum()) for k in range(3)]}")

    model = train_hybrid_rcng(data, n_regimes=3, n_factors=6, verbose=True)

    # Evaluate
    x_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    pred_regimes = model.get_regime_assignments(x_tensor).flatten()

    n_lags = model.n_lags
    true_aligned = true_regimes[n_lags:]

    # Phase 1 ARI (warm-start regimes)
    warmstart_regimes = model.initial_regimes.numpy()[n_lags:]
    warmstart_ari = adjusted_rand_score(true_aligned, warmstart_regimes)

    # Final ARI
    final_ari = adjusted_rand_score(true_aligned, pred_regimes)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Warm-start ARI: {warmstart_ari:.4f}")
    print(f"Final ARI (after fine-tuning): {final_ari:.4f}")
    print(f"Improvement: {final_ari - warmstart_ari:+.4f}")

    print("\nConfusion (true rows, pred cols):")
    for tk in range(3):
        row = [((true_aligned == tk) & (pred_regimes == pk)).sum() for pk in range(3)]
        print(f"  True {tk}: {row}")

    print("\nLearned adjacencies (binarized at 0.5):")
    adj = model.get_adjacency_matrices()
    for k in range(3):
        n_edges = (adj[k] > 0.5).sum()
        print(f"\nRegime {k} ({int(n_edges)} edges):")
        print((adj[k] > 0.5).astype(int))
