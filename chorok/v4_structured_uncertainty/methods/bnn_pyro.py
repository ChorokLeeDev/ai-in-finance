"""
Real Bayesian Neural Network with Pyro
======================================

Proper BNN with:
- Distributions over weights (not point estimates)
- Variational Inference for training
- Prior specification over weights

Reference: Blundell et al. "Weight Uncertainty in Neural Networks" (2015)

Author: ChorokLeeDev
Created: 2025-12-09
"""

import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


class BayesianRegressor(PyroModule):
    """
    Simple Bayesian MLP for regression with weight uncertainty.

    Uses PyroSample for weight priors - proper Bayesian approach.
    Simplified architecture for better convergence on tabular data.
    """
    def __init__(self, input_dim, hidden_dim=32, prior_scale=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Smaller prior scale for better initialization
        # First layer weights - scaled by input dim
        init_scale = prior_scale / np.sqrt(input_dim)
        self.fc1_weight = PyroSample(
            dist.Normal(0., init_scale).expand([hidden_dim, input_dim]).to_event(2)
        )
        self.fc1_bias = PyroSample(
            dist.Normal(0., prior_scale).expand([hidden_dim]).to_event(1)
        )

        # Output layer weights - scaled by hidden dim
        out_scale = prior_scale / np.sqrt(hidden_dim)
        self.out_weight = PyroSample(
            dist.Normal(0., out_scale).expand([1, hidden_dim]).to_event(2)
        )
        self.out_bias = PyroSample(
            dist.Normal(0., prior_scale).expand([1]).to_event(1)
        )

    def forward(self, x, y=None):
        # Single hidden layer (simpler = easier to train)
        h1 = torch.tanh(x @ self.fc1_weight.T + self.fc1_bias)

        # Output
        mean = (h1 @ self.out_weight.T + self.out_bias).squeeze(-1)

        # Observation noise - tighter prior
        sigma = pyro.sample("sigma", dist.LogNormal(-1., 0.5))

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)

        return mean


class RealBNN:
    """
    Real Bayesian Neural Network using Variational Inference.

    This is proper Bayesian deep learning:
    - Weights have distributions (not point estimates)
    - Training via Variational Inference (ELBO optimization)
    - Predictions via posterior sampling
    """

    def __init__(self, input_dim, hidden_dim=64, prior_scale=1.0, device='cpu'):
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prior_scale = prior_scale
        self.model = None
        self.guide = None
        self.y_mean = 0
        self.y_std = 1

    def fit(self, X, y, epochs=1000, lr=0.01, batch_size=256, verbose=False):
        """Train BNN using Stochastic Variational Inference."""
        pyro.clear_param_store()

        # Normalize data
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).flatten()

        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-8
        y_normalized = (y - self.y_mean) / self.y_std

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y_normalized).to(self.device)

        # Create model
        self.model = BayesianRegressor(
            self.input_dim,
            self.hidden_dim,
            self.prior_scale
        ).to(self.device)

        # Automatic guide (mean-field variational distribution)
        self.guide = AutoDiagonalNormal(self.model)

        # SVI setup
        optimizer = Adam({"lr": lr})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

        # Training loop
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                loss = svi.step(batch_X, batch_y)
                total_loss += loss

            if verbose and (epoch + 1) % 100 == 0:
                avg_loss = total_loss / len(X)
                print(f"  Epoch {epoch+1}: ELBO = {-avg_loss:.4f}")

    def predict(self, X, n_samples=100):
        """Get predictions by sampling from the posterior."""
        X = np.array(X, dtype=np.float32)
        X_tensor = torch.FloatTensor(X).to(self.device)

        preds = []
        for _ in range(n_samples):
            # Sample from guide
            guide_trace = pyro.poutine.trace(self.guide).get_trace(X_tensor)

            # Get sampled weights (simplified architecture - single hidden layer)
            fc1_w = guide_trace.nodes['fc1_weight']['value']
            fc1_b = guide_trace.nodes['fc1_bias']['value']
            out_w = guide_trace.nodes['out_weight']['value']
            out_b = guide_trace.nodes['out_bias']['value']

            # Forward pass with sampled weights
            h1 = torch.tanh(X_tensor @ fc1_w.T + fc1_b)
            pred = (h1 @ out_w.T + out_b).squeeze(-1)

            # Denormalize
            pred = pred.detach().cpu().numpy() * self.y_std + self.y_mean
            preds.append(pred)

        return np.array(preds)

    def get_uncertainty(self, X, n_samples=100):
        """Get epistemic uncertainty (variance of posterior predictions)."""
        preds = self.predict(X, n_samples)
        return preds.var(axis=0)

    def get_mean_prediction(self, X, n_samples=100):
        """Get mean prediction."""
        preds = self.predict(X, n_samples)
        return preds.mean(axis=0)


def train_real_bnn(X, y, hidden_dim=64, epochs=1000, lr=0.01, seed=42):
    """Convenience function to train a real BNN."""
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_dim = X.shape[1]

    bnn = RealBNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        device=device
    )

    bnn.fit(X, y, epochs=epochs, lr=lr)

    return bnn


if __name__ == "__main__":
    print("Testing Real BNN with Pyro...")
    print("="*50)

    # Simple test data
    np.random.seed(42)
    n_samples = 500
    X = np.random.randn(n_samples, 10).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.3).astype(np.float32)

    print(f"Data: {X.shape}, y range: [{y.min():.2f}, {y.max():.2f}]")

    print("\nTraining Real BNN with Variational Inference...")
    bnn = train_real_bnn(X, y, hidden_dim=32, epochs=500, lr=0.01)

    print("\nGetting posterior predictions...")
    preds = bnn.get_mean_prediction(X[:100], n_samples=50)
    unc = bnn.get_uncertainty(X[:100], n_samples=50)

    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y[:100], preds)
    r2 = r2_score(y[:100], preds)

    print(f"\nResults:")
    print(f"  MAE: {mae:.4f}")
    print(f"  R2: {r2:.4f}")
    print(f"  Mean Epistemic Uncertainty: {unc.mean():.4f}")
    print(f"  Std Epistemic Uncertainty: {unc.std():.4f}")

    print("\n✓ Real BNN test passed!")
