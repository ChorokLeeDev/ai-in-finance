"""
MC Dropout Ensemble for Uncertainty Quantification
===================================================

NOTE: This is NOT a true Bayesian Neural Network!
This is MC Dropout + Ensemble, which is an approximation.

For real BNN with weight distributions, see bnn_pyro.py

What this actually does:
- Train multiple MLPs with different initializations
- Use dropout at test time (MC Dropout)
- Epistemic uncertainty = variance across predictions

Author: ChorokLeeDev
Created: 2025-12-09
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MCDropoutMLP(nn.Module):
    """MLP with dropout for MC Dropout uncertainty estimation. NOT a true BNN."""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Input normalization
        self.input_bn = nn.BatchNorm1d(input_dim)

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_bn(x)
        return self.network(x).squeeze(-1)


class MCDropoutEnsemble:
    """
    MC Dropout Ensemble (NOT a true BNN!).

    Combines:
    1. Multiple independently trained networks (epistemic via initialization)
    2. MC Dropout at inference (epistemic via dropout)

    This is an APPROXIMATION to Bayesian inference, not the real thing.
    For real BNN, see bnn_pyro.py
    """

    def __init__(self, input_dim, n_networks=5, hidden_dims=[128, 64],
                 dropout_rate=0.1, mc_samples=10, device='cpu'):
        self.n_networks = n_networks
        self.mc_samples = mc_samples
        self.device = device
        self.networks = []

        for i in range(n_networks):
            torch.manual_seed(42 + i * 100)
            net = MCDropoutMLP(input_dim, hidden_dims, dropout_rate).to(device)
            self.networks.append(net)

    def fit(self, X, y, epochs=100, batch_size=256, lr=0.001, verbose=False):
        """Train all networks."""
        # Ensure numeric types
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Normalize target (important for stable training)
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-8
        y_normalized = (y - self.y_mean) / self.y_std

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y_normalized).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for idx, net in enumerate(self.networks):
            optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            criterion = nn.MSELoss()

            net.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    pred = net(batch_X)
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(loader)
                scheduler.step(avg_loss)

                if verbose and (epoch + 1) % 20 == 0:
                    print(f"  Network {idx+1}, Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    def predict(self, X):
        """Get predictions from all networks with MC sampling."""
        X = np.array(X, dtype=np.float32)
        X_tensor = torch.FloatTensor(X).to(self.device)

        all_preds = []

        for net in self.networks:
            net.train()  # Keep dropout active for MC sampling

            for _ in range(self.mc_samples):
                with torch.no_grad():
                    pred = net(X_tensor).cpu().numpy()
                    # Denormalize predictions
                    pred = pred * self.y_std + self.y_mean
                    all_preds.append(pred)

        return np.array(all_preds)  # Shape: (n_networks * mc_samples, n_samples)

    def get_uncertainty(self, X):
        """Get epistemic uncertainty (variance of predictions)."""
        preds = self.predict(X)
        return preds.var(axis=0)  # Variance across all predictions

    def get_mean_prediction(self, X):
        """Get mean prediction."""
        preds = self.predict(X)
        return preds.mean(axis=0)


def train_mc_dropout_ensemble(X, y, n_networks=5, mc_samples=10, epochs=100, seed=42):
    """
    Convenience function to train MC Dropout ensemble.

    NOTE: This is NOT a true BNN! For real BNN, use bnn_pyro.py

    Returns trained ensemble ready for uncertainty estimation.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_dim = X.shape[1]
    ensemble = MCDropoutEnsemble(
        input_dim=input_dim,
        n_networks=n_networks,
        mc_samples=mc_samples,
        device=device
    )

    ensemble.fit(X, y, epochs=epochs, verbose=False)

    return ensemble


if __name__ == "__main__":
    # Quick test
    print("Testing MC Dropout Ensemble (NOT a true BNN!)...")

    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(1000) * 0.1

    ensemble = train_mc_dropout_ensemble(X, y, n_networks=3, mc_samples=5, epochs=50)

    uncertainty = ensemble.get_uncertainty(X[:100])
    print(f"Mean uncertainty: {uncertainty.mean():.4f}")
    print(f"Uncertainty std: {uncertainty.std():.4f}")
    print("MC Dropout Ensemble test passed!")
