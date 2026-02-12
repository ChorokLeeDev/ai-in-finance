#!/usr/bin/env python3
"""
Compute Cross-Domain Statistics for SHAP Concentration Analysis

Combines SALT (n=8) + cross-domain tasks into unified correlation analysis.
Computes Spearman rho, bootstrap CI, LOO stability, threshold transfer test.

Usage:
    python3 compute_cross_domain_statistics.py
    python3 compute_cross_domain_statistics.py --include-new  # after Phase 2 runs complete

Output: results/cross_domain_statistics.json
"""

import argparse
import json
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
SHAP_DIR = RESULTS_DIR / "shap"
CONFORMAL_DIR = RESULTS_DIR / "conformal"
STAT_RIGOR = RESULTS_DIR / "statistical_rigor.json"
OUTPUT_FILE = RESULTS_DIR / "cross_domain_statistics.json"


# --- SALT task data (from statistical_rigor.json) ---

SALT_CONCENTRATION = {
    "sales-group": 47.3,
    "sales-payterms": 54.2,
    "sales-shipcond": 50.7,
    "item-shippoint": 48.8,
    "item-incoterms": 28.9,
    "item-plant": 23.9,
    "sales-incoterms": 23.7,
    "sales-office": 42.6,
}

# Severity labels for threshold test
SALT_SEVERITY = {
    "sales-group": "catastrophic",
    "sales-payterms": "catastrophic",
    "sales-shipcond": "catastrophic",
    "item-shippoint": "severe",
    "item-incoterms": "robust",
    "item-plant": "severe",
    "sales-incoterms": "robust",
    "sales-office": "robust",
}


def load_salt_data():
    """Load SALT task data from statistical_rigor.json."""
    with open(STAT_RIGOR) as f:
        data = json.load(f)

    tasks = []
    for task_name, conc in SALT_CONCENTRATION.items():
        if task_name not in data:
            print(f"  WARNING: {task_name} not in statistical_rigor.json")
            continue
        task_data = data[task_name]
        drop = task_data["coverage_drop"]["mean"] * 100  # Convert to percentage
        val_cov = task_data["val_coverage"]["mean"] * 100
        test_cov = task_data["test_coverage"]["mean"] * 100
        tasks.append({
            "dataset": "rel-salt",
            "task": task_name,
            "type": "multiclass",
            "concentration": conc,
            "coverage_drop": drop,
            "val_coverage": val_cov,
            "test_coverage": test_cov,
            "domain": "supply-chain",
            "shift": "COVID",
            "source": "salt",
        })
    return tasks


def load_shap_pickle(dataset, task):
    """Load SHAP concentration from pickle file."""
    pkl_path = SHAP_DIR / f"shap_{dataset}_{task}.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data.get("concentration_val", None)


def load_aps_pickle(dataset, task):
    """Load APS results from pickle file."""
    pkl_path = CONFORMAL_DIR / f"aps_{dataset}_{task}.pkl"
    if not pkl_path.exists():
        # Also try results root
        pkl_path = RESULTS_DIR / f"aps_{dataset}_{task}.pkl"
        if not pkl_path.exists():
            return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    val_cov = data.get("val_coverage_mean", None)
    test_cov = data.get("test_coverage_mean", None)
    if val_cov is None or test_cov is None:
        return None

    return {
        "val_coverage": val_cov * 100,
        "test_coverage": test_cov * 100,
        "coverage_drop": (val_cov - test_cov) * 100,
        "val_set_size": data.get("val_size_mean", None),
        "test_set_size": data.get("test_size_mean", None),
        "num_seeds": data.get("num_seeds", None),
    }


# Cross-domain task definitions
CROSS_DOMAIN_TASKS = [
    # Already computed
    {"dataset": "rel-trial", "task": "study-outcome", "type": "binary",
     "domain": "clinical-trials", "shift": "COVID"},
    {"dataset": "rel-f1", "task": "driver-dnf", "type": "binary",
     "domain": "motorsport", "shift": "none"},
    # New tasks (Phase 2)
    {"dataset": "rel-f1", "task": "driver-top3", "type": "binary",
     "domain": "motorsport", "shift": "none"},
    {"dataset": "rel-stack", "task": "user-engagement", "type": "binary",
     "domain": "tech-community", "shift": "COVID"},
    {"dataset": "rel-stack", "task": "user-badge", "type": "binary",
     "domain": "tech-community", "shift": "COVID"},
    {"dataset": "rel-amazon", "task": "user-churn", "type": "binary",
     "domain": "e-commerce", "shift": "none"},
    {"dataset": "rel-amazon", "task": "item-churn", "type": "binary",
     "domain": "e-commerce", "shift": "none"},
]


def load_cross_domain_data(include_new=False):
    """Load cross-domain task data from pickles."""
    tasks = []
    for task_def in CROSS_DOMAIN_TASKS:
        dataset = task_def["dataset"]
        task_name = task_def["task"]

        # Skip new tasks unless requested
        if not include_new and task_name not in ("study-outcome", "driver-dnf", "driver-top3"):
            continue

        # Load SHAP concentration
        conc = load_shap_pickle(dataset, task_name)
        if conc is None:
            print(f"  SKIP {dataset}/{task_name}: no SHAP pickle")
            continue

        # Load APS results
        aps = load_aps_pickle(dataset, task_name)
        if aps is None:
            print(f"  SKIP {dataset}/{task_name}: no APS pickle")
            continue

        tasks.append({
            "dataset": dataset,
            "task": task_name,
            "type": task_def["type"],
            "concentration": conc,
            "coverage_drop": aps["coverage_drop"],
            "val_coverage": aps["val_coverage"],
            "test_coverage": aps["test_coverage"],
            "val_set_size": aps.get("val_set_size"),
            "test_set_size": aps.get("test_set_size"),
            "num_seeds": aps.get("num_seeds"),
            "domain": task_def["domain"],
            "shift": task_def["shift"],
            "source": "cross-domain",
        })
        print(f"  LOADED {dataset}/{task_name}: conc={conc:.1f}%, drop={aps['coverage_drop']:.1f}%")

    return tasks


def spearman_exact_permutation(x, y, n_perm=100000):
    """Compute Spearman rho with exact permutation p-value."""
    rho, scipy_p = stats.spearmanr(x, y)

    # Permutation test
    np.random.seed(42)
    n = len(x)
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(n)
        rho_perm, _ = stats.spearmanr(x[perm], y)
        if abs(rho_perm) >= abs(rho):
            count += 1
    perm_p = count / n_perm

    return rho, scipy_p, perm_p


def bootstrap_ci(x, y, n_bootstrap=10000, alpha=0.05):
    """Bootstrap 95% CI for Spearman rho."""
    np.random.seed(42)
    n = len(x)
    rhos = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        r, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            rhos.append(r)
    rhos = np.array(rhos)
    lo = np.percentile(rhos, 100 * alpha / 2)
    hi = np.percentile(rhos, 100 * (1 - alpha / 2))
    return round(lo, 2), round(hi, 2)


def loo_stability(x, y):
    """Leave-one-out stability analysis."""
    n = len(x)
    results = []
    for i in range(n):
        mask = np.arange(n) != i
        rho, p = stats.spearmanr(x[mask], y[mask])
        results.append({"removed_idx": int(i), "rho": round(rho, 4), "p_value": round(p, 4), "n": n - 1})
    return results


def threshold_test(concentrations, drops, threshold=40.0, severe_threshold=15.0):
    """
    Test if threshold transfers to new data.

    A task is 'severe' if coverage_drop > severe_threshold (default 15%).
    A task is 'flagged' if concentration > threshold.
    """
    tp = fp = tn = fn = 0
    for conc, drop in zip(concentrations, drops):
        flagged = conc >= threshold
        severe = drop >= severe_threshold
        if flagged and severe:
            tp += 1
        elif flagged and not severe:
            fp += 1
        elif not flagged and severe:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "threshold": threshold,
        "severe_threshold": severe_threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def analyze(all_tasks, label="combined"):
    """Run full statistical analysis on a set of tasks."""
    conc = np.array([t["concentration"] for t in all_tasks])
    drop = np.array([t["coverage_drop"] for t in all_tasks])
    n = len(all_tasks)

    print(f"\n{'='*60}")
    print(f"Analysis: {label} (n={n})")
    print(f"{'='*60}")

    # Spearman correlation
    rho, scipy_p, perm_p = spearman_exact_permutation(conc, drop)
    print(f"  Spearman rho = {rho:.3f}, scipy p = {scipy_p:.4f}, perm p = {perm_p:.4f}")

    # Bootstrap CI
    ci_lo, ci_hi = bootstrap_ci(conc, drop)
    print(f"  Bootstrap 95% CI: [{ci_lo}, {ci_hi}]")

    # LOO stability
    loo = loo_stability(conc, drop)
    loo_rhos = [r["rho"] for r in loo]
    loo_sig = sum(1 for r in loo if r["p_value"] < 0.05)
    print(f"  LOO rho range: [{min(loo_rhos):.3f}, {max(loo_rhos):.3f}], {loo_sig}/{n} significant")

    # Threshold test at multiple thresholds
    thresholds = {}
    for thr in [30, 35, 40, 45, 50]:
        t = threshold_test(conc, drop, threshold=thr)
        thresholds[str(thr)] = t
        print(f"  Threshold {thr}%: P={t['precision']:.2f} R={t['recall']:.2f} F1={t['f1']:.2f}")

    # Per-task details
    task_details = []
    for t in all_tasks:
        task_details.append({
            "dataset": t["dataset"],
            "task": t["task"],
            "type": t["type"],
            "concentration": round(t["concentration"], 1),
            "coverage_drop": round(t["coverage_drop"], 1),
            "domain": t["domain"],
            "shift": t["shift"],
            "source": t["source"],
        })

    return {
        "label": label,
        "n": n,
        "spearman_rho": round(rho, 4),
        "scipy_p": round(scipy_p, 4),
        "permutation_p": round(perm_p, 4),
        "bootstrap_95ci": [ci_lo, ci_hi],
        "loo_stability": {
            "results": loo,
            "rho_range": [round(min(loo_rhos), 3), round(max(loo_rhos), 3)],
            "n_significant": loo_sig,
        },
        "threshold_tests": thresholds,
        "tasks": task_details,
    }


def binary_ceiling_analysis(all_tasks):
    """Analyze binary APS ceiling effect."""
    binary = [t for t in all_tasks if t["type"] == "binary"]
    multiclass = [t for t in all_tasks if t["type"] == "multiclass"]

    if not binary:
        return None

    binary_drops = [t["coverage_drop"] for t in binary]
    multi_drops = [t["coverage_drop"] for t in multiclass] if multiclass else []

    result = {
        "n_binary": len(binary),
        "n_multiclass": len(multiclass),
        "binary_drop_mean": round(np.mean(binary_drops), 2),
        "binary_drop_std": round(np.std(binary_drops), 2),
        "binary_drop_range": [round(min(binary_drops), 2), round(max(binary_drops), 2)],
    }

    if multiclass:
        result["multiclass_drop_mean"] = round(np.mean(multi_drops), 2)
        result["multiclass_drop_std"] = round(np.std(multi_drops), 2)
        result["multiclass_drop_range"] = [round(min(multi_drops), 2), round(max(multi_drops), 2)]

        # Mann-Whitney U test
        if len(binary_drops) >= 2 and len(multi_drops) >= 2:
            u_stat, u_p = stats.mannwhitneyu(multi_drops, binary_drops, alternative="greater")
            result["mannwhitney_u"] = round(u_stat, 2)
            result["mannwhitney_p"] = round(u_p, 4)

    # Per-task set size info
    binary_details = []
    for t in binary:
        detail = {
            "dataset": t["dataset"],
            "task": t["task"],
            "concentration": round(t["concentration"], 1),
            "coverage_drop": round(t["coverage_drop"], 1),
        }
        if t.get("val_set_size") is not None:
            detail["val_set_size"] = round(t["val_set_size"], 3)
            detail["test_set_size"] = round(t["test_set_size"], 3)
        binary_details.append(detail)
    result["binary_tasks"] = binary_details

    return result


def main():
    parser = argparse.ArgumentParser(description="Compute cross-domain statistics")
    parser.add_argument("--include-new", action="store_true",
                        help="Include new Phase 2 tasks (stack, amazon, f1/driver-top3)")
    args = parser.parse_args()

    print("Loading SALT data...")
    salt_tasks = load_salt_data()
    print(f"  Loaded {len(salt_tasks)} SALT tasks")

    print("\nLoading cross-domain data...")
    cross_tasks = load_cross_domain_data(include_new=args.include_new)
    print(f"  Loaded {len(cross_tasks)} cross-domain tasks")

    all_tasks = salt_tasks + cross_tasks
    n = len(all_tasks)
    print(f"\nTotal: {n} tasks")

    # Count domains
    domains = set(t["domain"] for t in all_tasks)
    print(f"Domains: {domains}")

    # --- Combined analysis ---
    combined = analyze(all_tasks, label=f"combined_n{n}")

    # --- SALT-only analysis (sanity check) ---
    salt_only = analyze(salt_tasks, label="salt_only_n8")

    # --- Cross-domain only (if enough tasks) ---
    cross_only = None
    if len(cross_tasks) >= 3:
        cross_only = analyze(cross_tasks, label=f"cross_only_n{len(cross_tasks)}")

    # --- COVID-era vs non-COVID split ---
    covid_tasks = [t for t in all_tasks if t["shift"] == "COVID"]
    non_covid_tasks = [t for t in all_tasks if t["shift"] != "COVID"]

    covid_analysis = None
    non_covid_analysis = None
    if len(covid_tasks) >= 4:
        covid_analysis = analyze(covid_tasks, label=f"covid_n{len(covid_tasks)}")
    if len(non_covid_tasks) >= 3:
        non_covid_analysis = analyze(non_covid_tasks, label=f"non_covid_n{len(non_covid_tasks)}")

    # --- Binary ceiling effect ---
    ceiling = binary_ceiling_analysis(all_tasks)
    if ceiling:
        print(f"\nBinary ceiling effect:")
        print(f"  Binary drops: mean={ceiling['binary_drop_mean']:.1f}%, range={ceiling['binary_drop_range']}")
        if "multiclass_drop_mean" in ceiling:
            print(f"  Multiclass drops: mean={ceiling['multiclass_drop_mean']:.1f}%, range={ceiling['multiclass_drop_range']}")
            if "mannwhitney_p" in ceiling:
                print(f"  Mann-Whitney p={ceiling['mannwhitney_p']:.4f}")

    # --- Save results ---
    output = {
        "combined": combined,
        "salt_only": salt_only,
        "cross_domain_only": cross_only,
        "covid_era": covid_analysis,
        "non_covid": non_covid_analysis,
        "binary_ceiling": ceiling,
        "n_total": n,
        "n_domains": len(domains),
        "domains": sorted(domains),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    import os
    os.chdir("/Users/i767700/Github/ai-in-finance")
    main()
