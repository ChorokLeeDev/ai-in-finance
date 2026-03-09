"""
Data Loaders for Causal Discovery Experiments
==============================================

Provides:
1. Synthetic data with known causal structure (for ground truth evaluation)
2. Fama-French 6-factor data (for real financial application)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import os


class SyntheticCausalData:
    """
    Generate synthetic multivariate time series with known causal structure.
    Supports regime switching for testing regime-aware methods.
    """

    def __init__(self, n_factors: int = 6, regime_lengths: list = None,
                 noise_std: float = 0.3, seed: int = 42):
        self.n_factors = n_factors
        self.regime_lengths = regime_lengths or [500, 500, 500]  # Default: 3 regimes
        self.noise_std = noise_std
        self.seed = seed

    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic data with regime-switching causal structure.

        Returns:
            data: (T, n_factors) time series
            true_adj: (n_regimes, n_factors, n_factors) true causal adjacency per regime
            regime_labels: (T,) regime assignment for each timestep
        """
        np.random.seed(self.seed)

        n_regimes = len(self.regime_lengths)
        T_total = sum(self.regime_lengths)

        # Define different causal structures for each regime
        true_adj = self._create_regime_graphs(n_regimes)

        # Generate data
        data = np.zeros((T_total, self.n_factors))
        regime_labels = np.zeros(T_total, dtype=int)

        t = 0
        for regime_idx, length in enumerate(self.regime_lengths):
            adj = true_adj[regime_idx]
            regime_labels[t:t+length] = regime_idx

            for i in range(length):
                if t + i == 0:
                    data[0] = np.random.randn(self.n_factors) * 0.1
                else:
                    # Generate based on causal structure
                    for j in range(self.n_factors):
                        # Sum contributions from causal parents
                        parent_contrib = 0
                        for k in range(self.n_factors):
                            if adj[k, j] > 0:
                                parent_contrib += adj[k, j] * data[t+i-1, k]

                        # Add autoregressive term and noise
                        data[t+i, j] = 0.3 * data[t+i-1, j] + parent_contrib + \
                                       self.noise_std * np.random.randn()
                t_current = t + i
            t += length

        return data, true_adj, regime_labels

    def _create_regime_graphs(self, n_regimes: int) -> np.ndarray:
        """Create distinct causal graphs for each regime."""
        n = self.n_factors
        true_adj = np.zeros((n_regimes, n, n))

        if n_regimes >= 1:
            # Regime 0: Chain structure (0 → 1 → 2 → 3 → 4 → 5)
            for i in range(n - 1):
                true_adj[0, i, i+1] = 0.5

        if n_regimes >= 2:
            # Regime 1: Hub structure (0 → all others)
            for i in range(1, n):
                true_adj[1, 0, i] = 0.4

        if n_regimes >= 3:
            # Regime 2: Sparse random + reverse chain
            true_adj[2, n-1, 0] = 0.5  # Reverse: 5 → 0
            true_adj[2, 2, 4] = 0.4    # 2 → 4
            true_adj[2, 1, 3] = 0.4    # 1 → 3

        return true_adj


class FamaFrenchLoader:
    """
    Load Fama-French factor data for real financial experiments.
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.expanduser("~/.cache/ff_factors")
        os.makedirs(self.data_dir, exist_ok=True)

    def load_factors(self, start_date: str = "1990-01-01",
                     end_date: str = "2024-12-31",
                     factors: list = None) -> pd.DataFrame:
        """
        Load Fama-French factor data.

        Args:
            start_date: Start date string
            end_date: End date string
            factors: List of factor names (default: 6 factors)

        Returns:
            DataFrame with factor returns
        """
        factors = factors or ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']

        try:
            import pandas_datareader as pdr

            # Download FF 5 factors
            ff5 = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3_daily',
                                          start=start_date, end=end_date)[0]

            # Download Momentum
            mom = pdr.get_data_famafrench('F-F_Momentum_Factor_daily',
                                          start=start_date, end=end_date)[0]

            # Merge and rename
            data = ff5.join(mom, how='inner')
            data = data.rename(columns={'Mom   ': 'Mom'})

            # Scale to decimal returns
            data = data / 100.0

            return data[factors]

        except Exception as e:
            print(f"Warning: Could not download FF data: {e}")
            print("Generating synthetic FF-like data instead...")
            return self._generate_synthetic_ff(start_date, end_date, factors)

    def _generate_synthetic_ff(self, start_date: str, end_date: str,
                               factors: list) -> pd.DataFrame:
        """Generate synthetic data resembling FF factors."""
        np.random.seed(42)

        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n_days = len(dates)
        n_factors = len(factors)

        # Generate correlated returns
        cov_matrix = np.eye(n_factors) * 0.0001  # Low variance
        cov_matrix[0, 1] = cov_matrix[1, 0] = 0.00005  # Some correlation

        returns = np.random.multivariate_normal(
            mean=np.zeros(n_factors),
            cov=cov_matrix,
            size=n_days
        )

        return pd.DataFrame(returns, index=dates, columns=factors)


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series windows."""

    def __init__(self, data: np.ndarray, window_size: int = 100, stride: int = 1):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
        self.stride = stride

        self.n_windows = (len(data) - window_size) // stride + 1

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        return self.data[start:end]


def create_data_loader(data: np.ndarray, window_size: int = 100,
                       batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """Create PyTorch DataLoader for training."""
    dataset = TimeSeriesDataset(data, window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Quick test
if __name__ == "__main__":
    print("Testing data loaders...")

    # Test synthetic data
    print("\n1. Synthetic Causal Data")
    synth = SyntheticCausalData(n_factors=6, regime_lengths=[300, 300, 300])
    data, true_adj, regimes = synth.generate()
    print(f"   Data shape: {data.shape}")
    print(f"   True adj shape: {true_adj.shape}")
    print(f"   Regimes: {np.unique(regimes, return_counts=True)}")
    print(f"   Regime 0 edges: {(true_adj[0] > 0).sum()}")
    print(f"   Regime 1 edges: {(true_adj[1] > 0).sum()}")
    print(f"   Regime 2 edges: {(true_adj[2] > 0).sum()}")

    # Test DataLoader
    print("\n2. DataLoader")
    loader = create_data_loader(data, window_size=50, batch_size=16)
    batch = next(iter(loader))
    print(f"   Batch shape: {batch.shape}")

    # Test FF loader (synthetic fallback)
    print("\n3. Fama-French Loader")
    ff_loader = FamaFrenchLoader()
    ff_data = ff_loader.load_factors(start_date="2020-01-01", end_date="2020-12-31")
    print(f"   FF data shape: {ff_data.shape}")
    print(f"   Columns: {list(ff_data.columns)}")

    print("\n✅ Data loader tests passed!")
