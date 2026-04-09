"""
Synthetic Experiments for Joint RCNG

Compares:
1. Joint RCNG (end-to-end, our method)
2. Two-Stage (HMM -> per-regime neural, baseline)
3. Single Neural (no regime awareness, baseline)

Key hypothesis: Joint RCNG discovers better regime-specific causal structure
because it learns regimes DEFINED BY causal differences.
"""

import sys
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rcng.joint_model import JointRCNG, train_joint_rcng, binarize_adjacency
from rcng.synthetic_data import RegimeSwitchingDGP
from rcng.evaluation import (
    evaluate_regime_causal_discovery,
    compare_methods,
    bootstrap_ci,
    paired_ttest,
)


def run_joint_rcng(
    data: np.ndarray,
    true_adj: np.ndarray,
    true_regimes: np.ndarray,
    n_regimes: int = 3,
    n_epochs: int = 100,
    seed: int = 42,
) -> dict:
    """Run Joint RCNG and return results."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = JointRCNG(
        n_factors=data.shape[1],
        n_lags=5,
        n_regimes=n_regimes,
        hidden_dim=32,
        lambda_sparse=0.01,
        lambda_smooth=0.1,
        lambda_diverse=0.1,
    ).to(device)

    history = train_joint_rcng(
        model, data,
        n_epochs=n_epochs,
        lr=1e-3,
        batch_size=32,
        window_size=100,
        verbose=False,
    )

    # Get predictions
    x_tensor = torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(0)
    pred_adj = model.get_adjacency_matrices()
    pred_regimes = model.get_regime_assignments(x_tensor).flatten()

    # Evaluate
    results = evaluate_regime_causal_discovery(
        pred_adj, true_adj, pred_regimes, true_regimes
    )
    results['final_loss'] = history['total'][-1]
    results['final_diverse'] = history['diverse'][-1]

    return results


def run_single_neural(
    data: np.ndarray,
    true_adj: np.ndarray,
    true_regimes: np.ndarray,
    n_epochs: int = 100,
    seed: int = 42,
) -> dict:
    """
    Run single neural Granger (no regime awareness).
    This is the baseline that ignores regimes entirely.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use Joint RCNG with n_regimes=1 (equivalent to no regimes)
    model = JointRCNG(
        n_factors=data.shape[1],
        n_lags=5,
        n_regimes=1,  # No regime awareness
        hidden_dim=32,
        lambda_sparse=0.01,
        lambda_smooth=0.0,  # No smoothness needed with 1 regime
        lambda_diverse=0.0,  # No diversity needed with 1 regime
    ).to(device)

    history = train_joint_rcng(
        model, data,
        n_epochs=n_epochs,
        lr=1e-3,
        batch_size=32,
        window_size=100,
        verbose=False,
    )

    # Get predictions - single graph for all regimes
    pred_adj_single = model.get_adjacency_matrices()  # (1, n_factors, n_factors)

    # Replicate to compare against true_adj which has multiple regimes
    n_regimes = true_adj.shape[0]
    pred_adj = np.tile(pred_adj_single, (n_regimes, 1, 1))

    # All time points assigned to "regime 0"
    pred_regimes = np.zeros(len(true_regimes), dtype=int)

    # Evaluate - note: regime metrics will be poor since we ignore regimes
    results = evaluate_regime_causal_discovery(
        pred_adj, true_adj, pred_regimes, true_regimes
    )
    results['final_loss'] = history['total'][-1]

    return results


def run_two_stage(
    data: np.ndarray,
    true_adj: np.ndarray,
    true_regimes: np.ndarray,
    n_regimes: int = 3,
    n_epochs: int = 50,
    seed: int = 42,
) -> dict:
    """
    Run two-stage approach:
    1. Cluster by volatility (simple proxy for HMM)
    2. Train separate neural Granger per cluster

    This baseline uses regimes but doesn't jointly optimize.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_factors = data.shape[1]

    # Stage 1: Simple regime detection via rolling volatility
    window = 20
    rolling_vol = np.zeros(len(data))
    for t in range(window, len(data)):
        rolling_vol[t] = data[t-window:t].std()

    # Assign regimes by volatility terciles
    thresholds = np.percentile(rolling_vol[window:], [33, 67])
    pred_regimes = np.zeros(len(data), dtype=int)
    pred_regimes[rolling_vol > thresholds[1]] = 2  # High vol = Crisis
    pred_regimes[(rolling_vol > thresholds[0]) & (rolling_vol <= thresholds[1])] = 1  # Elevated

    # Stage 2: Train separate neural Granger per regime
    pred_adj = np.zeros((n_regimes, n_factors, n_factors))

    for k in range(n_regimes):
        regime_mask = (pred_regimes == k)
        regime_data = data[regime_mask]

        if len(regime_data) < 50:
            # Not enough data for this regime
            continue

        # Train small neural model
        model_k = JointRCNG(
            n_factors=n_factors,
            n_lags=5,
            n_regimes=1,
            hidden_dim=16,
            lambda_sparse=0.01,
        ).to(device)

        train_joint_rcng(
            model_k, regime_data,
            n_epochs=n_epochs,
            lr=1e-3,
            batch_size=min(32, len(regime_data) // 4),
            window_size=min(50, len(regime_data) // 2),
            verbose=False,
        )

        pred_adj[k] = model_k.get_adjacency_matrices()[0]

    # Evaluate
    results = evaluate_regime_causal_discovery(
        pred_adj, true_adj, pred_regimes, true_regimes
    )

    return results


def run_experiment(n_trials: int = 20, T: int = 1500, n_epochs: int = 100):
    """
    Run full synthetic experiment.

    Args:
        n_trials: number of trials
        T: time series length
        n_epochs: training epochs
    """
    print(f"=" * 60)
    print(f"Joint RCNG Synthetic Experiment")
    print(f"Trials: {n_trials}, T: {T}, Epochs: {n_epochs}")
    print(f"=" * 60)

    results_joint = []
    results_single = []
    results_twostage = []

    for trial in range(n_trials):
        seed = 42 + trial
        print(f"\nTrial {trial + 1}/{n_trials} (seed={seed})")

        # Generate data
        dgp = RegimeSwitchingDGP(seed=seed)
        data, true_regimes, true_adj = dgp.generate(T=T)

        print(f"  Regime proportions: {dgp.get_regime_proportions(true_regimes)}")

        # Run methods
        print("  Running Joint RCNG...", end=" ", flush=True)
        res_joint = run_joint_rcng(data, true_adj, true_regimes, n_epochs=n_epochs, seed=seed)
        results_joint.append(res_joint)
        print(f"F1={res_joint['macro_f1']:.3f}")

        print("  Running Single Neural...", end=" ", flush=True)
        res_single = run_single_neural(data, true_adj, true_regimes, n_epochs=n_epochs, seed=seed)
        results_single.append(res_single)
        print(f"F1={res_single['macro_f1']:.3f}")

        print("  Running Two-Stage...", end=" ", flush=True)
        res_twostage = run_two_stage(data, true_adj, true_regimes, n_epochs=n_epochs//2, seed=seed)
        results_twostage.append(res_twostage)
        print(f"F1={res_twostage['macro_f1']:.3f}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for name, results in [
        ("Joint RCNG", results_joint),
        ("Two-Stage", results_twostage),
        ("Single Neural", results_single),
    ]:
        f1_scores = [r['macro_f1'] for r in results]
        ari_scores = [r['regime_ari'] for r in results]
        div_scores = [r.get('graph_diversity', 0) for r in results]

        mean_f1, ci_l, ci_u = bootstrap_ci(f1_scores)
        mean_ari, _, _ = bootstrap_ci(ari_scores)
        mean_div, _, _ = bootstrap_ci(div_scores)

        print(f"\n{name}:")
        print(f"  Macro F1: {mean_f1:.3f} [{ci_l:.3f}, {ci_u:.3f}]")
        print(f"  Regime ARI: {mean_ari:.3f}")
        print(f"  Graph Diversity: {mean_div:.3f}")

    # Statistical comparisons
    print("\n" + "-" * 60)
    print("STATISTICAL COMPARISONS")
    print("-" * 60)

    # Joint vs Single
    comp_single = compare_methods(results_joint, results_single, 'macro_f1')
    print(f"\nJoint vs Single Neural:")
    print(f"  Improvement: {comp_single['improvement']:.3f} ({comp_single['improvement_pct']:.1f}%)")
    print(f"  t-statistic: {comp_single['t_statistic']:.3f}")
    print(f"  p-value: {comp_single['p_value']:.4f}")
    print(f"  Cohen's d: {comp_single['cohens_d']:.3f}")

    # Joint vs Two-Stage
    comp_twostage = compare_methods(results_joint, results_twostage, 'macro_f1')
    print(f"\nJoint vs Two-Stage:")
    print(f"  Improvement: {comp_twostage['improvement']:.3f} ({comp_twostage['improvement_pct']:.1f}%)")
    print(f"  t-statistic: {comp_twostage['t_statistic']:.3f}")
    print(f"  p-value: {comp_twostage['p_value']:.4f}")
    print(f"  Cohen's d: {comp_twostage['cohens_d']:.3f}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {'n_trials': n_trials, 'T': T, 'n_epochs': n_epochs},
        'results_joint': results_joint,
        'results_single': results_single,
        'results_twostage': results_twostage,
        'comparisons': {
            'joint_vs_single': comp_single,
            'joint_vs_twostage': comp_twostage,
        }
    }

    output_path = Path(__file__).parent.parent.parent / 'results' / 'rcng_synthetic.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    # Quick test with fewer trials
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--T', type=int, default=1500)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--quick', action='store_true', help='Quick test with 5 trials')
    args = parser.parse_args()

    if args.quick:
        args.n_trials = 5
        args.n_epochs = 50
        args.T = 1000

    run_experiment(
        n_trials=args.n_trials,
        T=args.T,
        n_epochs=args.n_epochs,
    )
