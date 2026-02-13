import json
import numpy as np
from scipy import stats

# Data from paper
tasks = [
    ("s-shipcond",  50.7, 0.956, 0.54, 0.98),
    ("s-payterms",  54.2, 0.748, 0.65, 0.92),
    ("s-group",     47.3, 0.676, 0.98, 1.00),
    ("i-plant",     23.9, 0.741, 0.73, 0.86),
    ("s-incoterms", 23.7, 0.633, 0.75, 0.87),
    ("s-office",    42.6, 0.994, 1.00, 1.00),
    ("i-incoterms", 28.9, 0.559, 0.78, 0.66),
    ("i-shippoint", 48.8, 0.505, 0.71, 0.58),
]

names = [t[0] for t in tasks]
conc = np.array([t[1] for t in tasks])
ks_raw = np.array([t[2] for t in tasks])
mean_cal = np.array([t[3] for t in tasks])
mean_test = np.array([t[4] for t in tasks])

# Signed KS: positive if test > cal (rightward/catastrophic), negative if test < cal
sign = np.sign(mean_test - mean_cal)
# s-office: both 1.00, sign=0 -> KS becomes 0
ks_signed = sign * ks_raw

print("=" * 65)
print(f"{'Task':<14} {'Conc%':>6} {'KS_raw':>7} {'sign':>5} {'KS_signed':>10}")
print("-" * 65)
for i, name in enumerate(names):
    print(f"{name:<14} {conc[i]:6.1f} {ks_raw[i]:7.3f} {sign[i]:5.0f} {ks_signed[i]:10.3f}")
print("=" * 65)

# 1. Concentration vs raw KS
rho_raw, p_raw = stats.spearmanr(conc, ks_raw)
tau_raw, p_tau_raw = stats.kendalltau(conc, ks_raw)
print(f"\n--- Concentration vs RAW KS (n=8) ---")
print(f"Spearman rho = {rho_raw:.3f}, p = {p_raw:.4f}")
print(f"Kendall  tau = {tau_raw:.3f}, p = {p_tau_raw:.4f}")

# 2. Concentration vs signed KS
rho_signed, p_signed = stats.spearmanr(conc, ks_signed)
tau_signed, p_tau_signed = stats.kendalltau(conc, ks_signed)
print(f"\n--- Concentration vs SIGNED KS (n=8) ---")
print(f"Spearman rho = {rho_signed:.3f}, p = {p_signed:.4f}")
print(f"Kendall  tau = {tau_signed:.3f}, p = {p_tau_signed:.4f}")

# 3. Exclude s-office (degenerate) for robustness
mask = np.array([n != "s-office" for n in names])
rho_ex, p_ex = stats.spearmanr(conc[mask], ks_signed[mask])
tau_ex, p_tau_ex = stats.kendalltau(conc[mask], ks_signed[mask])
print(f"\n--- Concentration vs SIGNED KS, excl s-office (n=7) ---")
print(f"Spearman rho = {rho_ex:.3f}, p = {p_ex:.4f}")
print(f"Kendall  tau = {tau_ex:.3f}, p = {p_tau_ex:.4f}")

# Save results
results = {
    "n_tasks": 8,
    "tasks": {name: {"concentration": float(conc[i]), "ks_raw": float(ks_raw[i]),
                      "mean_cal": float(mean_cal[i]), "mean_test": float(mean_test[i]),
                      "ks_signed": float(ks_signed[i])}
              for i, name in enumerate(names)},
    "raw_ks": {
        "spearman_rho": round(float(rho_raw), 4),
        "spearman_p": round(float(p_raw), 4),
        "kendall_tau": round(float(tau_raw), 4),
        "kendall_p": round(float(p_tau_raw), 4),
    },
    "signed_ks": {
        "spearman_rho": round(float(rho_signed), 4),
        "spearman_p": round(float(p_signed), 4),
        "kendall_tau": round(float(tau_signed), 4),
        "kendall_p": round(float(p_tau_signed), 4),
    },
    "signed_ks_excl_office": {
        "n": 7,
        "spearman_rho": round(float(rho_ex), 4),
        "spearman_p": round(float(p_ex), 4),
        "kendall_tau": round(float(tau_ex), 4),
        "kendall_p": round(float(p_tau_ex), 4),
    },
}

outpath = "/Users/i767700/Github/ai-in-finance/papers/conformal_covid/results/concentration_ks_correlation.json"
with open(outpath, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {outpath}")
