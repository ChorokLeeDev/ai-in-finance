"""
Synthetic Data Generator for Regime-Switching Causal Discovery

Generates time series with:
1. Markov-switching regimes
2. Different causal structures per regime
3. Mix of linear and nonlinear relationships

Key design: Each regime has a DIFFERENT causal graph, so methods must
discover both regimes AND per-regime causal structure.
"""

import numpy as np
from typing import Tuple, Dict, Optional


class RegimeSwitchingDGP:
    """
    Data Generating Process with regime-switching causal structure.

    Three regimes with different causal graphs:
    - Regime 0 (Normal): Linear chain X0 -> X1 -> X2 -> ... -> X5
    - Regime 1 (Elevated): Bidirectional weak connections
    - Regime 2 (Crisis): Nonlinear hub X0 -> all others via nonlinear functions

    This tests whether methods can discover:
    1. That there are different regimes
    2. That each regime has different causal structure
    3. Nonlinear relationships in Crisis regime
    """

    def __init__(
        self,
        n_factors: int = 6,
        n_regimes: int = 3,
        transition_probs: Optional[np.ndarray] = None,
        noise_std: float = 0.3,
        seed: int = 42,
    ):
        """
        Args:
            n_factors: number of time series variables
            n_regimes: number of latent regimes
            transition_probs: (n_regimes, n_regimes) transition matrix.
                              If None, uses default with sticky regimes.
            noise_std: standard deviation of noise
            seed: random seed
        """
        self.n_factors = n_factors
        self.n_regimes = n_regimes
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

        # Default transition matrix: more balanced regimes for learning
        # ~50% Normal, ~30% Elevated, ~20% Crisis (more balanced for joint learning)
        if transition_probs is None:
            self.transition_probs = np.array([
                [0.90, 0.07, 0.03],   # Normal -> Normal/Elevated/Crisis
                [0.10, 0.80, 0.10],   # Elevated -> ...
                [0.15, 0.15, 0.70],   # Crisis -> ...
            ])
        else:
            self.transition_probs = transition_probs

        # Build true adjacency matrices per regime
        self.true_adj = self._build_true_adjacency()

    def _build_true_adjacency(self) -> np.ndarray:
        """
        Build ground truth adjacency matrices for each regime.

        Returns:
            adj: (n_regimes, n_factors, n_factors) where adj[k,i,j]=1 means i->j in regime k
        """
        n = self.n_factors
        adj = np.zeros((self.n_regimes, n, n))

        # Regime 0 (Normal): Linear chain 0 -> 1 -> 2 -> 3 -> 4 -> 5
        for i in range(n - 1):
            adj[0, i, i + 1] = 1.0

        # Regime 1 (Elevated): Bidirectional weak (all pairs weakly connected)
        for i in range(n):
            for j in range(n):
                if i != j:
                    adj[1, i, j] = 0.5  # Weak edges

        # Regime 2 (Crisis): Hub structure (0 -> all others)
        for j in range(1, n):
            adj[2, 0, j] = 1.0

        return adj

    def generate(self, T: int = 1500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate time series data with regime-switching causal structure.

        Args:
            T: number of time points

        Returns:
            data: (T, n_factors) time series
            regimes: (T,) regime labels
            true_adj: (n_regimes, n_factors, n_factors) true adjacency
        """
        n = self.n_factors
        data = np.zeros((T, n))
        regimes = np.zeros(T, dtype=int)

        # Initialize
        data[0] = self.rng.randn(n) * 0.1
        regimes[0] = 0  # Start in Normal

        for t in range(1, T):
            # Regime transition
            prev_regime = regimes[t - 1]
            regimes[t] = self.rng.choice(
                self.n_regimes,
                p=self.transition_probs[prev_regime]
            )

            # Generate data based on current regime
            regime = regimes[t]
            data[t] = self._generate_step(data[t - 1], regime)

        return data, regimes, self.true_adj

    def _generate_step(self, x_prev: np.ndarray, regime: int) -> np.ndarray:
        """
        Generate one time step given previous values and current regime.

        Args:
            x_prev: (n_factors,) previous time step values
            regime: current regime (0, 1, or 2)

        Returns:
            x_curr: (n_factors,) current time step values
        """
        n = self.n_factors
        x_curr = np.zeros(n)

        # REGIME-SPECIFIC NOISE LEVELS (key for identifiability)
        regime_noise = [self.noise_std, self.noise_std * 2, self.noise_std * 4]
        noise = self.rng.randn(n) * regime_noise[regime]

        if regime == 0:
            # Normal: Linear chain with strong coefficients
            # x_j = 0.3 * x_j_prev + 0.6 * x_{j-1}_prev + noise
            x_curr[0] = 0.3 * x_prev[0] + noise[0]
            for j in range(1, n):
                x_curr[j] = 0.2 * x_prev[j] + 0.6 * x_prev[j - 1] + noise[j]

        elif regime == 1:
            # Elevated: Bidirectional weak (higher noise)
            # x_j = 0.3 * x_j_prev + 0.2 * sum(x_i_prev for i != j) + noise
            for j in range(n):
                ar_term = 0.3 * x_prev[j]
                cross_term = 0.2 * (x_prev.sum() - x_prev[j]) / (n - 1)
                x_curr[j] = ar_term + cross_term + noise[j]

        elif regime == 2:
            # Crisis: Nonlinear hub (X0 -> all via nonlinear functions, highest noise)
            x_curr[0] = 0.3 * x_prev[0] + noise[0]

            # Different nonlinear functions for each target - STRONGER coefficients
            nonlinear_funcs = [
                lambda x: np.tanh(3 * x),           # tanh (stronger)
                lambda x: np.sign(x) * x**2,         # signed quadratic
                lambda x: np.sin(4 * x),             # sinusoidal (stronger)
                lambda x: np.abs(x) * 1.5,           # absolute value (stronger)
                lambda x: np.where(x > 0, x * 2, 0), # ReLU-like (stronger)
            ]

            for j in range(1, n):
                func = nonlinear_funcs[(j - 1) % len(nonlinear_funcs)]
                x_curr[j] = 0.2 * x_prev[j] + 0.7 * func(x_prev[0]) + noise[j]

        return x_curr

    def get_regime_proportions(self, regimes: np.ndarray) -> Dict[int, float]:
        """Get proportion of time spent in each regime."""
        T = len(regimes)
        return {k: (regimes == k).sum() / T for k in range(self.n_regimes)}


class MixedLinearNonlinearDGP:
    """
    DGP with mixed linear/nonlinear edges.

    Some edges are linear, others are nonlinear.
    Tests whether neural methods can capture nonlinear edges
    without overfitting linear ones.
    """

    def __init__(
        self,
        n_factors: int = 6,
        noise_std: float = 0.3,
        seed: int = 42,
    ):
        self.n_factors = n_factors
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

        # Edge types: 0=none, 1=linear, 2=nonlinear
        self.edge_types = np.zeros((n_factors, n_factors), dtype=int)

        # Chain with alternating linear/nonlinear
        for i in range(n_factors - 1):
            self.edge_types[i, i + 1] = 1 if i % 2 == 0 else 2

        # True adjacency (binary)
        self.true_adj = (self.edge_types > 0).astype(float)

    def generate(self, T: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate mixed linear/nonlinear data.

        Returns:
            data: (T, n_factors)
            true_adj: (n_factors, n_factors)
        """
        n = self.n_factors
        data = np.zeros((T, n))
        data[0] = self.rng.randn(n) * 0.1

        for t in range(1, T):
            noise = self.rng.randn(n) * self.noise_std

            for j in range(n):
                # AR term
                val = 0.3 * data[t - 1, j]

                # Add contributions from parents
                for i in range(n):
                    if self.edge_types[i, j] == 1:
                        # Linear edge
                        val += 0.5 * data[t - 1, i]
                    elif self.edge_types[i, j] == 2:
                        # Nonlinear edge (tanh)
                        val += 0.5 * np.tanh(2 * data[t - 1, i])

                data[t, j] = val + noise[j]

        return data, self.true_adj


class ThresholdDGP:
    """
    DGP with threshold/discontinuous nonlinearities.

    Tests the key hypothesis: neural methods excel on discontinuities
    (margin calls, circuit breakers, etc.)
    """

    def __init__(
        self,
        n_factors: int = 6,
        noise_std: float = 0.3,
        seed: int = 42,
    ):
        self.n_factors = n_factors
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

        # True adjacency: specific threshold structure
        self.true_adj = np.zeros((n_factors, n_factors))
        self.true_adj[0, 1] = 1  # 0 -> 1
        self.true_adj[1, 2] = 1  # 1 -> 2
        self.true_adj[2, 3] = 1  # 2 -> 3
        self.true_adj[0, 4] = 1  # 0 -> 4 (interaction)
        self.true_adj[2, 4] = 1  # 2 -> 4 (interaction)

    def generate(self, T: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate threshold nonlinear data.

        Returns:
            data: (T, n_factors)
            true_adj: (n_factors, n_factors)
        """
        n = self.n_factors
        data = np.zeros((T, n))
        data[0] = self.rng.randn(n) * 0.1

        for t in range(1, T):
            noise = self.rng.randn(n) * self.noise_std
            x = data[t - 1]

            # Factor 0: AR only
            data[t, 0] = 0.3 * x[0] + noise[0]

            # Factor 1: 0 -> 1 via signed quadratic
            data[t, 1] = 0.3 * x[1] + 0.5 * np.sign(x[0]) * x[0]**2 + noise[1]

            # Factor 2: 1 -> 2 via hard threshold
            data[t, 2] = 0.3 * x[2] + 0.5 * (x[1] > 0.5).astype(float) + noise[2]

            # Factor 3: 2 -> 3 via soft threshold (sigmoid)
            data[t, 3] = 0.3 * x[3] + 0.5 / (1 + np.exp(-5 * x[2])) + noise[3]

            # Factor 4: 0, 2 -> 4 via interaction
            data[t, 4] = 0.3 * x[4] + 0.3 * x[2] * np.abs(x[0]) + noise[4]

            # Factor 5: AR only (isolated)
            data[t, 5] = 0.3 * x[5] + noise[5]

        return data, self.true_adj


def generate_crypto_like_data(
    T: int = 1000,
    n_factors: int = 6,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data mimicking crypto market dynamics.

    - High volatility regimes with regime switching
    - BTC (factor 0) leads others
    - Nonlinear contagion during high-volatility periods

    Returns:
        data: (T, n_factors)
        regimes: (T,) regime labels (0=low vol, 1=high vol)
        true_adj: (n_regimes, n_factors, n_factors)
    """
    rng = np.random.RandomState(seed)

    # Transition matrix: 80% stay in same regime
    trans = np.array([
        [0.95, 0.05],  # Low vol
        [0.10, 0.90],  # High vol
    ])

    # True adjacency per regime
    true_adj = np.zeros((2, n_factors, n_factors))

    # Low vol: Only BTC -> ETH (factors 0 -> 1)
    true_adj[0, 0, 1] = 1.0

    # High vol: BTC -> all (contagion)
    for j in range(1, n_factors):
        true_adj[1, 0, j] = 1.0

    # Generate
    data = np.zeros((T, n_factors))
    regimes = np.zeros(T, dtype=int)

    data[0] = rng.randn(n_factors) * 0.1
    regimes[0] = 0

    for t in range(1, T):
        # Regime transition
        regimes[t] = rng.choice(2, p=trans[regimes[t-1]])
        regime = regimes[t]

        # Noise (higher in high vol regime)
        noise_std = 0.3 if regime == 0 else 0.6
        noise = rng.randn(n_factors) * noise_std

        x = data[t - 1]

        if regime == 0:
            # Low vol: only BTC -> ETH
            data[t, 0] = 0.3 * x[0] + noise[0]
            data[t, 1] = 0.3 * x[1] + 0.4 * x[0] + noise[1]
            for j in range(2, n_factors):
                data[t, j] = 0.3 * x[j] + noise[j]
        else:
            # High vol: BTC -> all via nonlinear (panic selling)
            data[t, 0] = 0.3 * x[0] + noise[0]
            for j in range(1, n_factors):
                # Nonlinear contagion: amplified when BTC drops
                contagion = 0.5 * np.sign(x[0]) * np.abs(x[0])**1.5
                data[t, j] = 0.3 * x[j] + contagion + noise[j]

    return data, regimes, true_adj


if __name__ == "__main__":
    # Test the generators
    print("Testing RegimeSwitchingDGP...")
    dgp = RegimeSwitchingDGP(seed=42)
    data, regimes, true_adj = dgp.generate(T=1000)
    print(f"Data shape: {data.shape}")
    print(f"Regime proportions: {dgp.get_regime_proportions(regimes)}")
    print(f"True adj shape: {true_adj.shape}")

    print("\nRegime 0 (Normal) - Chain structure:")
    print(true_adj[0])

    print("\nRegime 2 (Crisis) - Hub structure:")
    print(true_adj[2])

    print("\nTesting MixedLinearNonlinearDGP...")
    dgp2 = MixedLinearNonlinearDGP(seed=42)
    data2, adj2 = dgp2.generate(T=500)
    print(f"Data shape: {data2.shape}")
    print(f"Edge types:\n{dgp2.edge_types}")

    print("\nAll tests passed!")
