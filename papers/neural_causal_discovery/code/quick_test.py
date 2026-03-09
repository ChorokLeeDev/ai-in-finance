"""Quick test of RANCD vs Baselines on synthetic data"""
import numpy as np
import torch
import sys
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance/papers/neural_causal_discovery/code')

from data_loader import SyntheticCausalData, create_data_loader
from model import RANCD, train_rancd
from baselines import LinearGrangerCausality, NOTEARS, VARModel, evaluate_causal_discovery

print("=" * 50)
print("Quick Experiment: RANCD vs Baselines")
print("=" * 50)

# Generate synthetic data
np.random.seed(42)
synth = SyntheticCausalData(n_factors=6, regime_lengths=[300, 300, 300])
data, true_adj, regimes = synth.generate()
true_adj_eval = true_adj[0]  # Chain structure

print(f"\nData: {data.shape}, True edges: {(true_adj_eval > 0).sum()}")

# 1. Granger
print("\n1. Linear Granger Causality...")
gc = LinearGrangerCausality(n_lags=5)
gc_adj = gc.fit(data)
gc_m = evaluate_causal_discovery(true_adj_eval, gc_adj)
print(f"   F1={gc_m['f1']:.3f}, Prec={gc_m['precision']:.3f}, Rec={gc_m['recall']:.3f}")

# 2. NOTEARS
print("\n2. NOTEARS...")
notears = NOTEARS(lambda_l1=0.1, max_iter=50)
notears_adj = notears.fit(data)
notears_m = evaluate_causal_discovery(true_adj_eval, notears_adj)
print(f"   F1={notears_m['f1']:.3f}, Prec={notears_m['precision']:.3f}, Rec={notears_m['recall']:.3f}")

# 3. VAR
print("\n3. VAR Model...")
var = VARModel(n_lags=5)
var_adj = var.fit(data)
var_m = evaluate_causal_discovery(true_adj_eval, var_adj)
print(f"   F1={var_m['f1']:.3f}, Prec={var_m['precision']:.3f}, Rec={var_m['recall']:.3f}")

# 4. RANCD (quick)
print("\n4. RANCD (20 epochs)...")
torch.manual_seed(42)
loader = create_data_loader(data, window_size=50, batch_size=16)
model = RANCD(n_factors=6, hidden_dim=32, n_regimes=3, n_lags=5)
train_rancd(model, loader, n_epochs=20, lr=1e-3, device='cpu')

test_data = torch.FloatTensor(data[:50]).unsqueeze(0)
rancd_adj = model.get_causal_graph(test_data).mean(axis=0)
rancd_m = evaluate_causal_discovery(true_adj_eval, rancd_adj)
print(f"   F1={rancd_m['f1']:.3f}, Prec={rancd_m['precision']:.3f}, Rec={rancd_m['recall']:.3f}")

# Summary
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"{'Method':<15} {'F1':>8} {'Precision':>10} {'Recall':>8}")
print("-" * 45)
print(f"{'Granger':<15} {gc_m['f1']:>8.3f} {gc_m['precision']:>10.3f} {gc_m['recall']:>8.3f}")
print(f"{'NOTEARS':<15} {notears_m['f1']:>8.3f} {notears_m['precision']:>10.3f} {notears_m['recall']:>8.3f}")
print(f"{'VAR':<15} {var_m['f1']:>8.3f} {var_m['precision']:>10.3f} {var_m['recall']:>8.3f}")
print(f"{'RANCD':<15} {rancd_m['f1']:>8.3f} {rancd_m['precision']:>10.3f} {rancd_m['recall']:>8.3f}")
print("\n✅ Quick experiment completed!")
