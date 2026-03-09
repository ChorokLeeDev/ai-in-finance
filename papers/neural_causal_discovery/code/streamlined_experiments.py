"""
Streamlined Experiments for RANCD Paper
========================================
Faster experiments with reduced epochs but sufficient for demonstration.
"""
import numpy as np
import torch
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from data_loader import SyntheticCausalData, create_data_loader
from model import RANCD
from baselines import LinearGrangerCausality, NOTEARS, VARModel, evaluate_causal_discovery

def train_rancd_fast(model, loader, n_epochs=30, lr=1e-3, verbose=False):
    """Fast training loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss, _ = model.compute_loss(batch)
            loss.backward()
            optimizer.step()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}")

    return model


def run_experiments():
    print("=" * 60)
    print("RANCD vs Baselines: Streamlined Experiments")
    print("=" * 60)

    results = {
        'granger': {'f1': [], 'precision': [], 'recall': []},
        'notears': {'f1': [], 'precision': [], 'recall': []},
        'var': {'f1': [], 'precision': [], 'recall': []},
        'rancd': {'f1': [], 'precision': [], 'recall': []}
    }

    n_trials = 3

    for trial in range(n_trials):
        seed = 42 + trial
        print(f"\n--- Trial {trial+1}/{n_trials} (seed={seed}) ---")

        # Generate data
        np.random.seed(seed)
        synth = SyntheticCausalData(n_factors=6, regime_lengths=[300, 300, 300], seed=seed)
        data, true_adj, regimes = synth.generate()
        true_adj_eval = true_adj[0]  # Chain structure for evaluation

        # 1. Granger
        gc = LinearGrangerCausality(n_lags=5)
        gc_adj = gc.fit(data)
        gc_m = evaluate_causal_discovery(true_adj_eval, gc_adj)
        results['granger']['f1'].append(gc_m['f1'])
        results['granger']['precision'].append(gc_m['precision'])
        results['granger']['recall'].append(gc_m['recall'])
        print(f"Granger: F1={gc_m['f1']:.3f}")

        # 2. NOTEARS
        notears = NOTEARS(lambda_l1=0.05, max_iter=80)
        notears_adj = notears.fit(data)
        notears_m = evaluate_causal_discovery(true_adj_eval, notears_adj, threshold=0.2)
        results['notears']['f1'].append(notears_m['f1'])
        results['notears']['precision'].append(notears_m['precision'])
        results['notears']['recall'].append(notears_m['recall'])
        print(f"NOTEARS: F1={notears_m['f1']:.3f}")

        # 3. VAR
        var = VARModel(n_lags=5)
        var_adj = var.fit(data)
        var_m = evaluate_causal_discovery(true_adj_eval, var_adj, threshold=0.25)
        results['var']['f1'].append(var_m['f1'])
        results['var']['precision'].append(var_m['precision'])
        results['var']['recall'].append(var_m['recall'])
        print(f"VAR: F1={var_m['f1']:.3f}")

        # 4. RANCD
        torch.manual_seed(seed)
        loader = create_data_loader(data, window_size=80, batch_size=16)
        model = RANCD(n_factors=6, hidden_dim=48, n_regimes=3, n_lags=5)
        model = train_rancd_fast(model, loader, n_epochs=30, verbose=True)

        test_data = torch.FloatTensor(data[:80]).unsqueeze(0)
        rancd_adj = model.get_causal_graph(test_data).mean(axis=0)
        rancd_m = evaluate_causal_discovery(true_adj_eval, rancd_adj, threshold=0.35)
        results['rancd']['f1'].append(rancd_m['f1'])
        results['rancd']['precision'].append(rancd_m['precision'])
        results['rancd']['recall'].append(rancd_m['recall'])
        print(f"RANCD: F1={rancd_m['f1']:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (mean ± std)")
    print("=" * 60)
    print(f"{'Method':<12} {'F1':>15} {'Precision':>12} {'Recall':>10}")
    print("-" * 55)

    for method in ['granger', 'notears', 'var', 'rancd']:
        m = results[method]
        f1_mean, f1_std = np.mean(m['f1']), np.std(m['f1'])
        prec = np.mean(m['precision'])
        rec = np.mean(m['recall'])
        print(f"{method.upper():<12} {f1_mean:.3f} ± {f1_std:.3f}    {prec:.3f}        {rec:.3f}")

    return results


def run_regime_detection():
    """Evaluate regime detection using Adjusted Rand Index."""
    from sklearn.metrics import adjusted_rand_score

    print("\n" + "=" * 60)
    print("REGIME DETECTION EVALUATION")
    print("=" * 60)

    ari_scores = []

    for trial in range(3):
        seed = 42 + trial
        np.random.seed(seed)
        torch.manual_seed(seed)

        synth = SyntheticCausalData(n_factors=6, regime_lengths=[300, 300, 300], seed=seed)
        data, _, true_regimes = synth.generate()

        loader = create_data_loader(data, window_size=80, batch_size=16)
        model = RANCD(n_factors=6, hidden_dim=48, n_regimes=3, n_lags=5)
        model = train_rancd_fast(model, loader, n_epochs=30)

        test_data = torch.FloatTensor(data).unsqueeze(0)
        pred_regimes = model.get_regime_assignments(test_data).flatten()

        ari = adjusted_rand_score(true_regimes, pred_regimes)
        ari_scores.append(ari)
        print(f"Trial {trial+1}: ARI = {ari:.3f}")

    print(f"\nMean ARI: {np.mean(ari_scores):.3f} ± {np.std(ari_scores):.3f}")

    return {'ari_scores': ari_scores}


if __name__ == "__main__":
    # Run experiments
    causal_results = run_experiments()
    regime_results = run_regime_detection()

    # Save results
    results_dir = '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/results'
    os.makedirs(results_dir, exist_ok=True)

    all_results = {
        'causal_discovery': causal_results,
        'regime_detection': regime_results,
        'timestamp': datetime.now().isoformat()
    }

    # Convert numpy to list for JSON
    def to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_list(i) for i in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj

    with open(f'{results_dir}/streamlined_results.json', 'w') as f:
        json.dump(to_list(all_results), f, indent=2)

    print(f"\n✅ Results saved to {results_dir}/streamlined_results.json")
