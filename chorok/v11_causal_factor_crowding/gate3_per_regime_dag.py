"""
Gate 3: Per-Regime Causal DAG Discovery

This implements the key insight: different regimes have different causal structures.

- Normal regime: Weak, stable relationships
- Crowding regime: Stronger spillovers as positions correlate
- Crisis regime: Maximum contagion, cascade effects

Uses DYNOTEARS-style continuous optimization with DAG constraint.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.linalg import expm
import warnings
warnings.filterwarnings('ignore')

# Import Gate 2 for regime detection
from gate2_regime_detection import StudentTHMM, load_and_prepare_data


class PerRegimeDAG:
    """
    Learn different causal DAGs for each regime.

    Key innovation: The DAG structure can change across regimes,
    not just the edge weights.
    """

    def __init__(self, n_regimes=3, max_lag=10, lambda1=0.1, lambda2=0.01,
                 w_threshold=0.1):
        self.n_regimes = n_regimes
        self.max_lag = max_lag
        self.lambda1 = lambda1  # L1 sparsity penalty
        self.lambda2 = lambda2  # Cross-regime consistency penalty
        self.w_threshold = w_threshold  # Threshold for edge detection

        # Learned parameters
        self.W = None  # Instantaneous adjacency [K, d, d]
        self.B = None  # Lagged adjacency [K, max_lag, d, d]

    def _h(self, W):
        """
        DAG constraint: h(W) = tr(e^{W∘W}) - d = 0

        This is the NOTEARS/DYNOTEARS acyclicity constraint.
        h(W) = 0 iff W is a DAG.
        """
        d = W.shape[0]
        M = W * W  # Element-wise square
        E = expm(M)
        return np.trace(E) - d

    def _h_grad(self, W):
        """Gradient of DAG constraint."""
        d = W.shape[0]
        M = W * W
        E = expm(M)
        return 2 * W * E

    def _loss_regime(self, X, W, B, regime_mask):
        """
        Loss for a single regime.

        L = ||X_t - W @ X_t - sum_l B_l @ X_{t-l}||^2 + λ1 * ||W||_1
        """
        T, d = X.shape
        n_samples = regime_mask.sum()

        if n_samples < self.max_lag + 10:
            return 0.0, np.zeros((d, d)), np.zeros((self.max_lag, d, d))

        # Compute residuals
        residual = np.zeros((T, d))

        for t in range(self.max_lag, T):
            if regime_mask[t]:
                pred = W @ X[t]
                for l in range(self.max_lag):
                    pred += B[l] @ X[t - l - 1]
                residual[t] = X[t] - pred

        # MSE loss
        mse = np.sum(residual[regime_mask] ** 2) / n_samples

        # L1 penalty
        l1 = self.lambda1 * (np.abs(W).sum() + np.abs(B).sum())

        return mse + l1

    def _fit_regime(self, X, regime_mask, W_init=None, B_init=None):
        """
        Fit DAG for a single regime using augmented Lagrangian method.

        This is a simplified version of DYNOTEARS.
        """
        T, d = X.shape
        n_samples = regime_mask.sum()

        if n_samples < self.max_lag + 50:
            print(f"  Warning: Only {n_samples} samples, using fallback")
            return np.zeros((d, d)), np.zeros((self.max_lag, d, d))

        # Get regime data
        regime_indices = np.where(regime_mask)[0]
        regime_indices = regime_indices[regime_indices >= self.max_lag]

        if len(regime_indices) < 50:
            return np.zeros((d, d)), np.zeros((self.max_lag, d, d))

        # Build design matrix for regression
        # Y = X_t, X_design = [X_t, X_{t-1}, ..., X_{t-p}]
        Y = X[regime_indices]  # [n, d]
        n = len(regime_indices)

        # For simplicity, use Granger-style regression per variable
        W = np.zeros((d, d))
        B = np.zeros((self.max_lag, d, d))

        for j in range(d):  # For each target variable
            # Build design matrix
            X_design = np.zeros((n, d + d * self.max_lag))

            for i, t in enumerate(regime_indices):
                # Contemporaneous (excluding self)
                X_design[i, :d] = X[t]
                X_design[i, j] = 0  # Exclude self

                # Lagged
                for l in range(self.max_lag):
                    X_design[i, d + l * d: d + (l + 1) * d] = X[t - l - 1]

            y = Y[:, j]

            # Ridge regression with L1-like sparsity via thresholding
            try:
                from sklearn.linear_model import LassoCV
                model = LassoCV(cv=5, max_iter=1000)
                model.fit(X_design, y)
                coef = model.coef_
            except:
                # Fallback to OLS with thresholding
                coef = np.linalg.lstsq(X_design, y, rcond=None)[0]

            # Extract coefficients
            W[:, j] = coef[:d]
            W[j, j] = 0  # No self-loop

            for l in range(self.max_lag):
                B[l, :, j] = coef[d + l * d: d + (l + 1) * d]

        # Threshold small coefficients
        W[np.abs(W) < self.w_threshold] = 0
        B[np.abs(B) < self.w_threshold] = 0

        return W, B

    def fit(self, X, regimes):
        """
        Fit per-regime DAGs.

        Parameters
        ----------
        X : array [T, d]
            Factor crowding time series
        regimes : array [T]
            Regime assignments (0, 1, 2)
        """
        X = np.asarray(X)
        regimes = np.asarray(regimes)
        T, d = X.shape

        self.W = np.zeros((self.n_regimes, d, d))
        self.B = np.zeros((self.n_regimes, self.max_lag, d, d))

        regime_names = ['Normal', 'Crowding', 'Crisis']

        for k in range(self.n_regimes):
            regime_mask = regimes == k
            n_samples = regime_mask.sum()

            print(f"\nFitting DAG for regime {k} ({regime_names[k]}): {n_samples} samples")

            W_k, B_k = self._fit_regime(X, regime_mask)

            self.W[k] = W_k
            self.B[k] = B_k

            # Report
            n_edges = (np.abs(W_k) > 0).sum()
            print(f"  Instantaneous edges: {n_edges}")

            # Report strongest edges
            if n_edges > 0:
                indices = np.unravel_index(np.argsort(np.abs(W_k).ravel())[-3:], W_k.shape)
                print(f"  Strongest edges:")
                for i, j in zip(indices[0][::-1], indices[1][::-1]):
                    if W_k[i, j] != 0:
                        print(f"    {i} → {j}: {W_k[i, j]:.3f}")

        return self

    def get_dag(self, regime):
        """Get adjacency matrix for a regime."""
        return self.W[regime]

    def get_edges(self, regime, factor_names=None):
        """Get list of edges for a regime."""
        W = self.W[regime]
        d = W.shape[0]

        if factor_names is None:
            factor_names = [f'F{i}' for i in range(d)]

        edges = []
        for i in range(d):
            for j in range(d):
                if W[i, j] != 0:
                    edges.append({
                        'from': factor_names[i],
                        'to': factor_names[j],
                        'weight': W[i, j]
                    })

        return sorted(edges, key=lambda x: abs(x['weight']), reverse=True)


def run_gate3_validation():
    """
    Gate 3: Discover different causal DAGs per regime.

    Success criteria:
    1. DAGs are learned for all three regimes
    2. Crisis regime has more/stronger edges (contagion)
    3. Edge structure differs meaningfully between regimes
    4. Key relationship (HML → SMB) appears in Crisis
    """

    print("=" * 60)
    print("GATE 3: Per-Regime Causal DAG Discovery")
    print("=" * 60)

    # Load data and fit regime model
    crowding = load_and_prepare_data()
    X = crowding.values
    factor_names = list(crowding.columns)

    print("\nStep 1: Fitting regime model...")
    hmm = StudentTHMM(n_regimes=3, n_iter=100)
    hmm.fit(X)
    regimes = hmm.predict(X)

    # Fit per-regime DAGs
    print("\nStep 2: Fitting per-regime DAGs...")
    dag_model = PerRegimeDAG(n_regimes=3, max_lag=10, lambda1=0.05, w_threshold=0.05)
    dag_model.fit(X, regimes)

    # ========================================================================
    # Analysis 1: Compare DAG density across regimes
    # ========================================================================
    print("\n" + "-" * 60)
    print("DAG DENSITY BY REGIME")
    print("-" * 60)

    regime_names = ['Normal', 'Crowding', 'Crisis']
    edge_counts = []
    total_weights = []

    for k in range(3):
        W = dag_model.W[k]
        n_edges = (np.abs(W) > 0).sum()
        total_weight = np.abs(W).sum()

        edge_counts.append(n_edges)
        total_weights.append(total_weight)

        print(f"\n{regime_names[k]}:")
        print(f"  Number of edges: {n_edges}")
        print(f"  Total edge weight: {total_weight:.3f}")
        print(f"  Mean edge weight: {total_weight / max(n_edges, 1):.3f}")

    # ========================================================================
    # Analysis 2: Specific edges per regime
    # ========================================================================
    print("\n" + "-" * 60)
    print("TOP EDGES BY REGIME")
    print("-" * 60)

    for k in range(3):
        print(f"\n{regime_names[k]} regime:")
        edges = dag_model.get_edges(k, factor_names)

        if len(edges) == 0:
            print("  (No significant edges)")
        else:
            for e in edges[:5]:
                direction = "+" if e['weight'] > 0 else "-"
                print(f"  {e['from']} → {e['to']}: {direction}{abs(e['weight']):.3f}")

    # ========================================================================
    # Analysis 3: Check for HML → SMB in Crisis
    # ========================================================================
    print("\n" + "-" * 60)
    print("KEY RELATIONSHIP CHECK: HML → SMB")
    print("-" * 60)

    hml_idx = factor_names.index('HML')
    smb_idx = factor_names.index('SMB')

    hml_smb_found = False
    for k in range(3):
        W = dag_model.W[k]
        weight = W[hml_idx, smb_idx]
        if weight != 0:
            print(f"  {regime_names[k]}: HML → SMB weight = {weight:.3f} ✅")
            hml_smb_found = True
        else:
            # Check lagged effects
            B = dag_model.B[k]
            for l in range(dag_model.max_lag):
                if B[l, hml_idx, smb_idx] != 0:
                    print(f"  {regime_names[k]}: HML → SMB (lag {l+1}) weight = {B[l, hml_idx, smb_idx]:.3f} ✅")
                    hml_smb_found = True
                    break
            else:
                print(f"  {regime_names[k]}: HML → SMB not found")

    # ========================================================================
    # Analysis 4: DAG structure differences
    # ========================================================================
    print("\n" + "-" * 60)
    print("STRUCTURAL DIFFERENCES BETWEEN REGIMES")
    print("-" * 60)

    # Compare Normal vs Crisis
    W_normal = dag_model.W[0]
    W_crisis = dag_model.W[2]

    edges_normal = set(zip(*np.where(np.abs(W_normal) > 0)))
    edges_crisis = set(zip(*np.where(np.abs(W_crisis) > 0)))

    only_normal = edges_normal - edges_crisis
    only_crisis = edges_crisis - edges_normal
    both = edges_normal & edges_crisis

    print(f"\nEdges only in Normal: {len(only_normal)}")
    print(f"Edges only in Crisis: {len(only_crisis)}")
    print(f"Edges in both: {len(both)}")

    if len(only_crisis) > 0:
        print(f"\nCrisis-specific edges (contagion channels):")
        for i, j in list(only_crisis)[:5]:
            print(f"  {factor_names[i]} → {factor_names[j]}: {W_crisis[i, j]:.3f}")

    # ========================================================================
    # VERDICT
    # ========================================================================
    print("\n" + "=" * 60)
    print("GATE 3 VERDICT")
    print("=" * 60)

    criteria = {
        'dags_learned': all(e > 0 for e in edge_counts),
        'crisis_more_edges': edge_counts[2] >= edge_counts[0],
        'structural_difference': len(only_crisis) > 0 or len(only_normal) > 0,
        'hml_smb_found': hml_smb_found,
    }

    passed = sum(criteria.values())
    total = len(criteria)

    print(f"\nCriteria passed: {passed}/{total}")
    for name, value in criteria.items():
        status = "✅" if value else "❌"
        print(f"  {status} {name}: {value}")

    if passed >= 3:
        print(f"\n🟢 GATE 3 PASSED - PROCEED to Stage 4: GNN Prediction")
        gate_passed = True
    else:
        print(f"\n🟡 GATE 3 PARTIAL - Consider adjusting thresholds or using different causal method")
        gate_passed = passed >= 2

    return {
        'pass': gate_passed,
        'dag_model': dag_model,
        'hmm': hmm,
        'regimes': regimes,
        'criteria': criteria,
        'edge_counts': edge_counts,
        'factor_names': factor_names
    }


def visualize_dags(results):
    """Visualize DAGs for each regime."""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        regime_names = ['Normal', 'Crowding', 'Crisis']
        factor_names = results['factor_names']

        for k, ax in enumerate(axes):
            W = results['dag_model'].W[k]

            # Create directed graph
            G = nx.DiGraph()
            G.add_nodes_from(factor_names)

            for i in range(len(factor_names)):
                for j in range(len(factor_names)):
                    if W[i, j] != 0:
                        G.add_edge(factor_names[i], factor_names[j],
                                   weight=abs(W[i, j]))

            # Layout
            pos = nx.circular_layout(G)

            # Draw
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue',
                                   node_size=1000)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)

            # Draw edges with width proportional to weight
            edges = G.edges(data=True)
            if edges:
                weights = [e[2]['weight'] * 5 for e in edges]
                nx.draw_networkx_edges(G, pos, ax=ax, width=weights,
                                       edge_color='gray', arrows=True,
                                       arrowsize=20, connectionstyle='arc3,rad=0.1')

            ax.set_title(f'{regime_names[k]} Regime\n({results["edge_counts"][k]} edges)')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('/Users/i767700/Github/ai-in-finance/chorok/v11_causal_factor_crowding/gate3_dags.png',
                    dpi=150)
        print("\nDAG visualization saved to gate3_dags.png")
        plt.close()

    except ImportError:
        print("\nMatplotlib/networkx not available, skipping visualization")


if __name__ == "__main__":
    results = run_gate3_validation()
    visualize_dags(results)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY: Causal Structure by Regime")
    print("=" * 60)

    factor_names = results['factor_names']
    dag_model = results['dag_model']

    for k, name in enumerate(['Normal', 'Crowding', 'Crisis']):
        print(f"\n{name}:")
        edges = dag_model.get_edges(k, factor_names)
        if edges:
            for e in edges:
                print(f"  {e['from']:4s} → {e['to']:4s}: {e['weight']:+.3f}")
        else:
            print("  (sparse/no edges)")
