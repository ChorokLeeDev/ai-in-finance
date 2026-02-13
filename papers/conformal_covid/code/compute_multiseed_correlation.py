import json
import numpy as np
from scipy import stats

# 11 multiclass tasks with multi-seed means for external datasets
tasks_11 = [
    ("gas_sensor",    7.27,  -3.78),
    ("kddcup99",     21.1,   15.9),
    ("s-incoterms",  23.7,    8.5),
    ("i-plant",      23.9,   10.6),
    ("i-incoterms",  28.9,   11.3),
    ("s-office",     42.6,    0.1),
    ("s-group",      47.3,   71.2),
    ("i-shippoint",  48.8,   18.5),
    ("covertype",    49.78,  81.8),
    ("s-shipcond",   50.7,   71.6),
    ("s-payterms",   54.2,   77.1),
]

names_11 = [t[0] for t in tasks_11]
C_11 = np.array([t[1] for t in tasks_11])
drop_11 = np.array([t[2] for t in tasks_11])

# Spearman and Kendall for n=11
sp_11 = stats.spearmanr(C_11, drop_11)
kt_11 = stats.kendalltau(C_11, drop_11)

print("=" * 60)
print(f"n=11 MULTICLASS (multi-seed means)")
print(f"  Spearman rho = {sp_11.statistic:.4f}, p = {sp_11.pvalue:.6f}")
print(f"  Kendall  tau = {kt_11.statistic:.4f}, p = {kt_11.pvalue:.6f}")
print("=" * 60)

# n=12: add Stack Overflow
tasks_12 = tasks_11 + [("stackoverflow", 48.9, -7.0)]
C_12 = np.array([t[1] for t in tasks_12])
drop_12 = np.array([t[2] for t in tasks_12])

sp_12 = stats.spearmanr(C_12, drop_12)
kt_12 = stats.kendalltau(C_12, drop_12)

print(f"\nn=12 MULTICLASS + StackOverflow (multi-seed means)")
print(f"  Spearman rho = {sp_12.statistic:.4f}, p = {sp_12.pvalue:.6f}")
print(f"  Kendall  tau = {kt_12.statistic:.4f}, p = {kt_12.pvalue:.6f}")
print("=" * 60)

# Print ranked data for inspection
print("\nRanked by concentration (n=11):")
order = np.argsort(C_11)
for i in order:
    print(f"  {names_11[i]:15s}  C={C_11[i]:5.1f}%  drop={drop_11[i]:+6.1f}pp")

# Save results
results = {
    "n11_multiclass_multiseed": {
        "n": 11,
        "tasks": [{"name": t[0], "concentration_pct": t[1], "coverage_drop_pp": t[2]} for t in tasks_11],
        "spearman_rho": round(sp_11.statistic, 4),
        "spearman_p": round(sp_11.pvalue, 6),
        "kendall_tau": round(kt_11.statistic, 4),
        "kendall_p": round(kt_11.pvalue, 6),
    },
    "n12_with_stackoverflow": {
        "n": 12,
        "tasks": [{"name": t[0], "concentration_pct": t[1], "coverage_drop_pp": t[2]} for t in tasks_12],
        "spearman_rho": round(sp_12.statistic, 4),
        "spearman_p": round(sp_12.pvalue, 6),
        "kendall_tau": round(kt_12.statistic, 4),
        "kendall_p": round(kt_12.pvalue, 6),
    },
    "note": "External datasets (gas_sensor, kddcup99, covertype) use multi-seed means. KDDCup99 changed from single-seed drop=-0.83 to multi-seed drop=+15.9."
}

outpath = "/Users/i767700/Github/ai-in-finance/papers/conformal_covid/results/multiseed_robustness_correlation.json"
with open(outpath, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {outpath}")
