#!/usr/bin/env python3
"""
Build a canonical manuscript claims snapshot from existing result artifacts.

This script does not retrain models. It reads cached JSON results and
recomputes paper-facing derived metrics under fixed policies:
  - strict significance threshold: p < 0.05
  - at-risk rule: coverage_drop_pp > 15
  - threshold flag rule: concentration_pct > threshold
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

STAT_RIGOR = RESULTS_DIR / "statistical_rigor.json"
EXTERNAL_MULTI = RESULTS_DIR / "external_multiseed_validation.json"
EXTERNAL_PHASE2 = RESULTS_DIR / "external_phase2_validation.json"
PENDIGITS = RESULTS_DIR / "pendigits_validation.json"
SATIMAGE = RESULTS_DIR / "satimage_validation.json"

OUTPUT_DEFAULT = RESULTS_DIR / "manuscript_claims_snapshot.json"


SALT_CONCENTRATION = {
    "sales-payterms": 54.2,
    "sales-shipcond": 50.7,
    "sales-group": 47.3,
    "item-shippoint": 48.8,
    "sales-office": 42.6,
    "item-incoterms": 28.9,
    "item-plant": 23.9,
    "sales-incoterms": 23.7,
}

SALT_NAME_MAP = {
    "sales-payterms": "s-payterms",
    "sales-shipcond": "s-shipcond",
    "sales-group": "s-group",
    "item-shippoint": "i-shippoint",
    "sales-office": "s-office",
    "item-incoterms": "i-incoterms",
    "item-plant": "i-plant",
    "sales-incoterms": "s-incoterms",
}


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _round(x: float, nd: int = 3) -> float:
    quant = Decimal("1").scaleb(-nd)
    return float(Decimal(str(float(x))).quantize(quant, rounding=ROUND_HALF_UP))


def _bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rho, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(rho):
            rhos.append(float(rho))

    lo = np.percentile(rhos, 100 * alpha / 2)
    hi = np.percentile(rhos, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def load_multiclass_n16_tasks() -> List[Dict]:
    stat = _read_json(STAT_RIGOR)
    ext_multi = _read_json(EXTERNAL_MULTI)
    ext_p2 = _read_json(EXTERNAL_PHASE2)
    pend = _read_json(PENDIGITS)
    sat = _read_json(SATIMAGE)

    tasks: List[Dict] = []

    # SALT 8 multiclass
    for task_key, conc in SALT_CONCENTRATION.items():
        drop_pp = stat[task_key]["coverage_drop"]["mean"] * 100.0
        tasks.append(
            {
                "task": SALT_NAME_MAP[task_key],
                "dataset": "rel-salt",
                "domain": "Supply Chain",
                "source": "salt_50seed",
                "num_seeds": int(stat[task_key]["num_seeds"]),
                "concentration_pct": _round(conc, 1),
                "coverage_drop_pp": _round(drop_pp, 1),
            }
        )

    # External multiclass from external_multiseed_validation.json
    ext_map = {
        "covertype": ("Covertype", "Ecology"),
        "kddcup99": ("KDDCup99", "Security"),
        "gas_sensor": ("Gas Sensor", "Chemical"),
    }
    for key, (name, domain) in ext_map.items():
        d = ext_multi["datasets"][key]
        tasks.append(
            {
                "task": name,
                "dataset": name,
                "domain": domain,
                "source": "external_10seed",
                "num_seeds": int(d["num_seeds"]),
                "concentration_pct": _round(d["concentration_mean"], 1),
                "coverage_drop_pp": _round(d["coverage_drop_mean"], 1),
            }
        )

    # External multiclass from phase2
    p2_map = {
        "avila": ("Avila Bible", "Humanities"),
        "shuttle": ("Shuttle", "Aerospace"),
        "pamap2": ("PAMAP2", "Health"),
    }
    for key, (name, domain) in p2_map.items():
        d = ext_p2["datasets"][key]
        tasks.append(
            {
                "task": name,
                "dataset": name,
                "domain": domain,
                "source": "external_10seed",
                "num_seeds": int(d["num_seeds"]),
                "concentration_pct": _round(d["concentration_mean"], 1),
                "coverage_drop_pp": _round(d["coverage_drop_mean"], 1),
            }
        )

    # Pendigits and Satimage (standalone result files)
    pend_r = pend["result"]
    tasks.append(
        {
            "task": "Pendigits",
            "dataset": "Pendigits",
            "domain": "HCI",
            "source": "external_10seed",
            "num_seeds": int(pend_r["num_seeds"]),
            "concentration_pct": _round(pend_r["concentration_mean"], 1),
            "coverage_drop_pp": _round(pend_r["coverage_drop_mean"], 1),
        }
    )
    tasks.append(
        {
            "task": "Satimage",
            "dataset": "Satimage",
            "domain": "Remote Sensing",
            "source": "external_10seed",
            "num_seeds": int(sat["num_seeds"]),
            "concentration_pct": _round(sat["concentration_mean"], 1),
            "coverage_drop_pp": _round(sat["coverage_drop_mean"], 1),
        }
    )

    if len(tasks) != 16:
        raise RuntimeError(f"Expected 16 multiclass tasks, got {len(tasks)}")

    for t in tasks:
        t["at_risk_gt15"] = t["coverage_drop_pp"] > 15.0

    return tasks


def compute_n16_stats(tasks: List[Dict]) -> Dict:
    x = np.array([t["concentration_pct"] for t in tasks], dtype=float)
    y = np.array([t["coverage_drop_pp"] for t in tasks], dtype=float)

    rho, rho_p = stats.spearmanr(x, y)
    tau, tau_p = stats.kendalltau(x, y)
    ci_lo, ci_hi = _bootstrap_spearman_ci(x, y, n_bootstrap=10000, seed=42)

    return {
        "n": len(tasks),
        "spearman_rho": _round(rho, 3),
        "spearman_p": _round(rho_p, 6),
        "kendall_tau": _round(tau, 3),
        "kendall_p": _round(tau_p, 6),
        "bootstrap_95ci_spearman": [_round(ci_lo, 2), _round(ci_hi, 2)],
        "significant_at_0p05": bool(rho_p < 0.05 and tau_p < 0.05),
        "n_bootstrap": 10000,
        "bootstrap_seed": 42,
    }


def compute_threshold_sensitivity(
    tasks: List[Dict],
    thresholds: List[int] | Tuple[int, ...] = (25, 30, 35, 40, 45, 50),
) -> List[Dict]:
    rows: List[Dict] = []
    for thr in thresholds:
        tp = fp = tn = fn = 0
        for t in tasks:
            pred = t["concentration_pct"] > thr
            truth = t["coverage_drop_pp"] > 15.0
            if pred and truth:
                tp += 1
            elif pred and not truth:
                fp += 1
            elif (not pred) and truth:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold_pct": thr,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": _round(precision, 2),
                "recall": _round(recall, 2),
                "f1": _round(f1, 2),
            }
        )
    return rows


def compute_external_stability() -> Dict:
    ext_multi = _read_json(EXTERNAL_MULTI)
    ext_p2 = _read_json(EXTERNAL_PHASE2)
    pend = _read_json(PENDIGITS)
    sat = _read_json(SATIMAGE)

    ds = []

    def add_dataset(name: str, rec: dict, per_seed_correct: int, num_seeds: int) -> None:
        if per_seed_correct == 10:
            status = "deterministic"
        elif per_seed_correct == 9:
            status = "near-deterministic"
        else:
            status = "variable"
        ds.append(
            {
                "dataset": name,
                "num_seeds": num_seeds,
                "per_seed_threshold_correct": per_seed_correct,
                "stability_status": status,
                "concentration_mean_pct": _round(rec["concentration_mean"], 1),
                "coverage_drop_mean_pp": _round(rec["coverage_drop_mean"], 1),
            }
        )

    # 4 datasets in external_multiseed (includes StackOverflow)
    add_dataset(
        "Covertype",
        ext_multi["datasets"]["covertype"],
        int(ext_multi["datasets"]["covertype"]["per_seed_threshold_correct"]),
        int(ext_multi["datasets"]["covertype"]["num_seeds"]),
    )
    add_dataset(
        "KDDCup99",
        ext_multi["datasets"]["kddcup99"],
        int(ext_multi["datasets"]["kddcup99"]["per_seed_threshold_correct"]),
        int(ext_multi["datasets"]["kddcup99"]["num_seeds"]),
    )
    add_dataset(
        "Gas Sensor",
        ext_multi["datasets"]["gas_sensor"],
        int(ext_multi["datasets"]["gas_sensor"]["per_seed_threshold_correct"]),
        int(ext_multi["datasets"]["gas_sensor"]["num_seeds"]),
    )
    add_dataset(
        "Stack Overflow",
        ext_multi["datasets"]["stackoverflow"],
        int(ext_multi["datasets"]["stackoverflow"]["per_seed_threshold_correct"]),
        int(ext_multi["datasets"]["stackoverflow"]["num_seeds"]),
    )

    # 3 datasets in phase2
    add_dataset(
        "Avila Bible",
        ext_p2["datasets"]["avila"],
        int(ext_p2["datasets"]["avila"]["per_seed_threshold_correct"]),
        int(ext_p2["datasets"]["avila"]["num_seeds"]),
    )
    add_dataset(
        "Shuttle",
        ext_p2["datasets"]["shuttle"],
        int(ext_p2["datasets"]["shuttle"]["per_seed_threshold_correct"]),
        int(ext_p2["datasets"]["shuttle"]["num_seeds"]),
    )
    add_dataset(
        "PAMAP2",
        ext_p2["datasets"]["pamap2"],
        int(ext_p2["datasets"]["pamap2"]["per_seed_threshold_correct"]),
        int(ext_p2["datasets"]["pamap2"]["num_seeds"]),
    )

    # Pendigits and Satimage standalone
    add_dataset(
        "Pendigits",
        pend["result"],
        int(pend["result"]["per_seed_threshold_correct"]),
        int(pend["result"]["num_seeds"]),
    )
    add_dataset(
        "Satimage",
        sat,
        int(sat["per_seed_threshold_correct"]),
        int(sat["num_seeds"]),
    )

    counts = {"deterministic": 0, "near-deterministic": 0, "variable": 0}
    for row in ds:
        counts[row["stability_status"]] += 1

    return {
        "rule": "deterministic=10/10, near-deterministic=9/10, else variable",
        "datasets": ds,
        "counts": counts,
        "n_external_domains": len(ds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    tasks = load_multiclass_n16_tasks()
    n16_stats = compute_n16_stats(tasks)
    threshold_rows = compute_threshold_sensitivity(tasks)
    stability = compute_external_stability()

    snapshot = {
        "metadata": {
            "strict_alpha": 0.05,
            "at_risk_rule": "coverage_drop_pp > 15",
            "threshold_pred_rule": "concentration_pct > threshold_pct",
            "inputs": {
                "statistical_rigor": str(STAT_RIGOR),
                "external_multiseed_validation": str(EXTERNAL_MULTI),
                "external_phase2_validation": str(EXTERNAL_PHASE2),
                "pendigits_validation": str(PENDIGITS),
                "satimage_validation": str(SATIMAGE),
            },
        },
        "multiclass_n16": {
            "tasks": tasks,
            "correlation": n16_stats,
            "threshold_sensitivity": threshold_rows,
        },
        "external_stability": stability,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"Wrote snapshot to: {args.output}")
    print(
        "n16 Spearman:",
        snapshot["multiclass_n16"]["correlation"]["spearman_rho"],
        "p=",
        snapshot["multiclass_n16"]["correlation"]["spearman_p"],
    )


if __name__ == "__main__":
    main()
