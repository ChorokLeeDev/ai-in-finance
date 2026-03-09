"""
Experiment Runner: RANCD vs Baselines
======================================

Runs comprehensive experiments comparing RANCD against baselines:
1. Synthetic data with known ground truth
2. Fama-French 6-factor real data

Saves results to ../results/
"""

import numpy as np
import torch
import json
import os
from datetime import datetime
from typing import Dict, List

# Local imports
from data_loader import SyntheticCausalData, FamaFrenchLoader, create_data_loader
from model import RANCD, train_rancd
from baselines import (LinearGrangerCausality, NOTEARS, VARModel,
                       evaluate_causal_discovery)


def run_synthetic_experiments(n_trials: int = 5, seed_base: int = 42) -> Dict:
    """
    Run experiments on synthetic data with known ground truth.

    Returns:
        results: Dict with metrics for each method
    """
    print("=" * 60)
    print("SYNTHETIC DATA EXPERIMENTS")
    print("=" * 60)

    results = {
        'rancd': {'f1': [], 'precision': [], 'recall': []},
        'granger': {'f1': [], 'precision': [], 'recall': []},
        'notears': {'f1': [], 'precision': [], 'recall': []},
        'var': {'f1': [], 'precision': [], 'recall': []}
    }

    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")
        seed = seed_base + trial

        # Generate synthetic data
        synth = SyntheticCausalData(
            n_factors=6,
            regime_lengths=[400, 400, 400],
            noise_std=0.3,
            seed=seed
        )
        data, true_adj, regimes = synth.generate()

        # Use regime 0 ground truth for evaluation (chain structure)
        true_adj_eval = true_adj[0]

        print(f"Data shape: {data.shape}")
        print(f"True edges (regime 0): {(true_adj_eval > 0).sum()}")

        # 1. RANCD
        print("\nTraining RANCD...")
        try:
            rancd_adj = run_rancd(data, n_epochs=50, seed=seed)
            rancd_metrics = evaluate_causal_discovery(true_adj_eval, rancd_adj)
            results['rancd']['f1'].append(rancd_metrics['f1'])
            results['rancd']['precision'].append(rancd_metrics['precision'])
            results['rancd']['recall'].append(rancd_metrics['recall'])
            print(f"RANCD F1: {rancd_metrics['f1']:.3f}")
        except Exception as e:
            print(f"RANCD failed: {e}")
            results['rancd']['f1'].append(0)
            results['rancd']['precision'].append(0)
            results['rancd']['recall'].append(0)

        # 2. Linear Granger
        print("\nRunning Linear Granger...")
        gc = LinearGrangerCausality(n_lags=5)
        gc_adj = gc.fit(data)
        gc_metrics = evaluate_causal_discovery(true_adj_eval, gc_adj)
        results['granger']['f1'].append(gc_metrics['f1'])
        results['granger']['precision'].append(gc_metrics['precision'])
        results['granger']['recall'].append(gc_metrics['recall'])
        print(f"Granger F1: {gc_metrics['f1']:.3f}")

        # 3. NOTEARS
        print("\nRunning NOTEARS...")
        notears = NOTEARS(lambda_l1=0.1, max_iter=100)
        notears_adj = notears.fit(data)
        notears_metrics = evaluate_causal_discovery(true_adj_eval, notears_adj)
        results['notears']['f1'].append(notears_metrics['f1'])
        results['notears']['precision'].append(notears_metrics['precision'])
        results['notears']['recall'].append(notears_metrics['recall'])
        print(f"NOTEARS F1: {notears_metrics['f1']:.3f}")

        # 4. VAR
        print("\nRunning VAR...")
        var = VARModel(n_lags=5)
        var_adj = var.fit(data)
        var_metrics = evaluate_causal_discovery(true_adj_eval, var_adj)
        results['var']['f1'].append(var_metrics['f1'])
        results['var']['precision'].append(var_metrics['precision'])
        results['var']['recall'].append(var_metrics['recall'])
        print(f"VAR F1: {var_metrics['f1']:.3f}")

    # Compute means and stds
    summary = {}
    for method, metrics in results.items():
        summary[method] = {
            'f1_mean': np.mean(metrics['f1']),
            'f1_std': np.std(metrics['f1']),
            'precision_mean': np.mean(metrics['precision']),
            'recall_mean': np.mean(metrics['recall'])
        }

    return {'trials': results, 'summary': summary}


def run_rancd(data: np.ndarray, n_epochs: int = 50, seed: int = 42) -> np.ndarray:
    """Train RANCD and extract causal graph."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_factors = data.shape[1]

    # Create data loader
    loader = create_data_loader(data, window_size=100, batch_size=16, shuffle=True)

    # Initialize model
    model = RANCD(
        n_factors=n_factors,
        hidden_dim=32,
        n_regimes=3,
        n_lags=5,
        temperature=0.5
    )

    # Train
    history = train_rancd(model, loader, n_epochs=n_epochs, lr=1e-3, device='cpu')

    # Extract graph
    test_data = torch.FloatTensor(data[:100]).unsqueeze(0)  # Single batch
    adj = model.get_causal_graph(test_data)

    return adj.mean(axis=0)  # Average over batch


def run_regime_detection_experiment(n_trials: int = 3, seed_base: int = 42) -> Dict:
    """
    Evaluate regime detection accuracy using Adjusted Rand Index.
    """
    from sklearn.metrics import adjusted_rand_score

    print("\n" + "=" * 60)
    print("REGIME DETECTION EXPERIMENTS")
    print("=" * 60)

    results = {'rancd_ari': []}

    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")
        seed = seed_base + trial

        # Generate data
        synth = SyntheticCausalData(
            n_factors=6,
            regime_lengths=[400, 400, 400],
            seed=seed
        )
        data, _, true_regimes = synth.generate()

        # Train RANCD
        torch.manual_seed(seed)
        loader = create_data_loader(data, window_size=100, batch_size=16)
        model = RANCD(n_factors=6, hidden_dim=32, n_regimes=3, n_lags=5)
        train_rancd(model, loader, n_epochs=30, lr=1e-3, device='cpu')

        # Get regime assignments
        test_data = torch.FloatTensor(data).unsqueeze(0)
        pred_regimes = model.get_regime_assignments(test_data).flatten()

        # Compute ARI
        ari = adjusted_rand_score(true_regimes, pred_regimes)
        results['rancd_ari'].append(ari)
        print(f"ARI: {ari:.3f}")

    summary = {
        'ari_mean': np.mean(results['rancd_ari']),
        'ari_std': np.std(results['rancd_ari'])
    }

    return {'trials': results, 'summary': summary}


def print_summary_table(synthetic_results: Dict, regime_results: Dict):
    """Print results as formatted table."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\n### Causal Discovery (Synthetic Data)")
    print("-" * 50)
    print(f"{'Method':<15} {'F1':>10} {'Precision':>12} {'Recall':>10}")
    print("-" * 50)

    summary = synthetic_results['summary']
    for method in ['rancd', 'granger', 'notears', 'var']:
        m = summary[method]
        print(f"{method.upper():<15} "
              f"{m['f1_mean']:.3f}±{m['f1_std']:.3f}  "
              f"{m['precision_mean']:.3f}        "
              f"{m['recall_mean']:.3f}")

    print("\n### Regime Detection (RANCD)")
    print("-" * 50)
    rs = regime_results['summary']
    print(f"ARI: {rs['ari_mean']:.3f} ± {rs['ari_std']:.3f}")


def save_results(results: Dict, output_dir: str):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"results_{timestamp}.json")

    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    results_json = convert_types(results)

    with open(filepath, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    print("=" * 60)
    print("RANCD vs Baselines: Experiment Suite")
    print("=" * 60)

    # Results directory
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")

    # Run experiments
    synthetic_results = run_synthetic_experiments(n_trials=3, seed_base=42)
    regime_results = run_regime_detection_experiment(n_trials=3, seed_base=42)

    # Print summary
    print_summary_table(synthetic_results, regime_results)

    # Save results
    all_results = {
        'synthetic': synthetic_results,
        'regime_detection': regime_results,
        'timestamp': datetime.now().isoformat()
    }
    save_results(all_results, results_dir)

    print("\n✅ All experiments completed!")
