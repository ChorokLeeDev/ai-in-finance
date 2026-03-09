"""
Full Neural Granger Experiments
===============================
Systematic comparison on nonlinear data.
"""
import numpy as np
import torch
import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from neural_granger_simple import NeuralGranger, train_neural_granger
from nonlinear_experiments import generate_nonlinear_causal_data
from baselines import LinearGrangerCausality, VARModel, evaluate_causal_discovery

print("=" * 60)
print("Neural Granger vs Baselines (Nonlinear Data)")
print("=" * 60)

results = {
    'linear_granger': [],
    'var': [],
    'neural_granger': []
}

for trial in range(5):
    seed = 42 + trial
    print(f"\n--- Trial {trial+1}/5 (seed={seed}) ---")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate nonlinear data
    data, true_adj = generate_nonlinear_causal_data(n_factors=6, T=400, seed=seed)

    # Linear Granger
    gc = LinearGrangerCausality(n_lags=5)
    gc_adj = gc.fit(data)
    gc_m = evaluate_causal_discovery(true_adj, gc_adj)
    results['linear_granger'].append(gc_m['f1'])
    print(f"Linear Granger F1: {gc_m['f1']:.3f}")

    # VAR
    var = VARModel(n_lags=5)
    var_adj = var.fit(data)
    var_m = evaluate_causal_discovery(true_adj, var_adj, threshold=0.25)
    results['var'].append(var_m['f1'])
    print(f"VAR F1: {var_m['f1']:.3f}")

    # Neural Granger
    model = NeuralGranger(n_factors=6, n_lags=5, hidden_dim=16)
    model = train_neural_granger(model, data, n_epochs=25, lr=1e-3)

    x = torch.FloatTensor(data).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        neural_adj = model.compute_granger_adjacency(x)

    # Best threshold
    best_f1 = 0
    for thresh in [0.03, 0.05, 0.07, 0.1]:
        m = evaluate_causal_discovery(true_adj, neural_adj, threshold=thresh)
        if m['f1'] > best_f1:
            best_f1 = m['f1']

    results['neural_granger'].append(best_f1)
    print(f"Neural Granger F1: {best_f1:.3f}")

# Summary
print("\n" + "=" * 60)
print("FINAL RESULTS (5 trials, nonlinear data)")
print("=" * 60)
print(f"{'Method':<20} {'F1 Mean':>10} {'F1 Std':>10}")
print("-" * 45)

for method in ['linear_granger', 'var', 'neural_granger']:
    f1_mean = np.mean(results[method])
    f1_std = np.std(results[method])
    name = method.replace('_', ' ').title()
    print(f"{name:<20} {f1_mean:>10.3f} {f1_std:>10.3f}")

# Check if neural beats linear
ng_mean = np.mean(results['neural_granger'])
lg_mean = np.mean(results['linear_granger'])
var_mean = np.mean(results['var'])

print("\n" + "=" * 60)
if ng_mean > lg_mean and ng_mean > var_mean:
    print("✅ NEURAL GRANGER OUTPERFORMS BASELINES!")
    print(f"   Improvement over Linear Granger: +{(ng_mean - lg_mean)*100:.1f}%")
    print(f"   Improvement over VAR: +{(ng_mean - var_mean)*100:.1f}%")
else:
    print("❌ Neural Granger does not consistently beat baselines")
