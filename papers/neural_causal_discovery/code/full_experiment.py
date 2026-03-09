"""
Full Experiment: RANCD vs Baselines
More epochs and proper evaluation
"""
import numpy as np
import torch
import json
import os
import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from data_loader import SyntheticCausalData, create_data_loader
from model import RANCD, train_rancd
from baselines import LinearGrangerCausality, NOTEARS, VARModel, evaluate_causal_discovery

print("=" * 60)
print("Full Experiment: RANCD vs Baselines (3 trials)")
print("=" * 60)

results = {
    'granger': {'f1': [], 'precision': [], 'recall': []},
    'notears': {'f1': [], 'precision': [], 'recall': []},
    'var': {'f1': [], 'precision': [], 'recall': []},
    'rancd': {'f1': [], 'precision': [], 'recall': []}
}

for trial in range(3):
    seed = 42 + trial
    print(f"\n{'='*20} Trial {trial+1}/3 (seed={seed}) {'='*20}")

    # Generate data
    np.random.seed(seed)
    synth = SyntheticCausalData(n_factors=6, regime_lengths=[400, 400, 400], seed=seed)
    data, true_adj, regimes = synth.generate()
    true_adj_eval = true_adj[0]

    print(f"Data: {data.shape}, True edges: {(true_adj_eval > 0).sum()}")

    # 1. Granger
    gc = LinearGrangerCausality(n_lags=5)
    gc_adj = gc.fit(data)
    gc_m = evaluate_causal_discovery(true_adj_eval, gc_adj)
    results['granger']['f1'].append(gc_m['f1'])
    results['granger']['precision'].append(gc_m['precision'])
    results['granger']['recall'].append(gc_m['recall'])
    print(f"Granger: F1={gc_m['f1']:.3f}")

    # 2. NOTEARS (with lower threshold)
    notears = NOTEARS(lambda_l1=0.05, max_iter=100)
    notears_adj = notears.fit(data)
    notears_m = evaluate_causal_discovery(true_adj_eval, notears_adj, threshold=0.2)
    results['notears']['f1'].append(notears_m['f1'])
    results['notears']['precision'].append(notears_m['precision'])
    results['notears']['recall'].append(notears_m['recall'])
    print(f"NOTEARS: F1={notears_m['f1']:.3f}")

    # 3. VAR
    var = VARModel(n_lags=5)
    var_adj = var.fit(data)
    var_m = evaluate_causal_discovery(true_adj_eval, var_adj, threshold=0.3)
    results['var']['f1'].append(var_m['f1'])
    results['var']['precision'].append(var_m['precision'])
    results['var']['recall'].append(var_m['recall'])
    print(f"VAR: F1={var_m['f1']:.3f}")

    # 4. RANCD (more epochs)
    torch.manual_seed(seed)
    loader = create_data_loader(data, window_size=100, batch_size=32)
    model = RANCD(n_factors=6, hidden_dim=64, n_regimes=3, n_lags=5)
    train_rancd(model, loader, n_epochs=50, lr=1e-3, device='cpu')

    # Get graph using full data
    test_data = torch.FloatTensor(data[:100]).unsqueeze(0)
    rancd_adj = model.get_causal_graph(test_data).mean(axis=0)
    rancd_m = evaluate_causal_discovery(true_adj_eval, rancd_adj, threshold=0.3)
    results['rancd']['f1'].append(rancd_m['f1'])
    results['rancd']['precision'].append(rancd_m['precision'])
    results['rancd']['recall'].append(rancd_m['recall'])
    print(f"RANCD: F1={rancd_m['f1']:.3f}")

# Summary
print("\n" + "=" * 60)
print("FINAL RESULTS (mean ± std)")
print("=" * 60)
print(f"{'Method':<15} {'F1':>15} {'Precision':>12} {'Recall':>10}")
print("-" * 55)

for method in ['granger', 'notears', 'var', 'rancd']:
    m = results[method]
    f1_mean = np.mean(m['f1'])
    f1_std = np.std(m['f1'])
    prec_mean = np.mean(m['precision'])
    rec_mean = np.mean(m['recall'])
    print(f"{method.upper():<15} {f1_mean:.3f}±{f1_std:.3f}    {prec_mean:.3f}        {rec_mean:.3f}")

# Save results
results_dir = '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/results'
os.makedirs(results_dir, exist_ok=True)

summary = {
    'method': ['granger', 'notears', 'var', 'rancd'],
    'f1_mean': [np.mean(results[m]['f1']) for m in ['granger', 'notears', 'var', 'rancd']],
    'f1_std': [np.std(results[m]['f1']) for m in ['granger', 'notears', 'var', 'rancd']],
    'precision_mean': [np.mean(results[m]['precision']) for m in ['granger', 'notears', 'var', 'rancd']],
    'recall_mean': [np.mean(results[m]['recall']) for m in ['granger', 'notears', 'var', 'rancd']],
}

with open(f'{results_dir}/synthetic_results.json', 'w') as f:
    json.dump({'trials': results, 'summary': summary}, f, indent=2)

print(f"\nResults saved to {results_dir}/synthetic_results.json")
print("\n✅ Full experiment completed!")
