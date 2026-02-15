"""
Build a claim ledger linking manuscript claims to recomputed JSON metrics.

Output:
  results/claim_ledger.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(obj: dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for key in path.split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur


def evaluate_numeric_claim(
    *,
    claim_id: str,
    manuscript_targets: list[str],
    description: str,
    source_file: str,
    metric_path: str,
    tier: str,
    comparator: str,
    threshold: float,
    value: Any,
) -> dict[str, Any]:
    status = "unresolved"
    passed = None
    if value is None:
        status = "unresolved"
    else:
        try:
            numeric = float(value)
            if comparator == "<":
                passed = numeric < threshold
            elif comparator == "<=":
                passed = numeric <= threshold
            elif comparator == ">":
                passed = numeric > threshold
            elif comparator == ">=":
                passed = numeric >= threshold
            else:
                raise ValueError(f"Unknown comparator: {comparator}")
            status = "pass" if passed else "fail"
        except Exception:
            status = "unresolved"

    return {
        "claim_id": claim_id,
        "manuscript_targets": manuscript_targets,
        "description": description,
        "tier": tier,
        "source": {
            "file": source_file,
            "metric_path": metric_path,
            "value": value,
        },
        "rule": {
            "comparator": comparator,
            "threshold": threshold,
        },
        "status": status,
    }


def evaluate_derived_claim(
    *,
    claim_id: str,
    manuscript_targets: list[str],
    description: str,
    tier: str,
    source_files: list[str],
    evaluator: Callable[[], tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    status, details = evaluator()
    return {
        "claim_id": claim_id,
        "manuscript_targets": manuscript_targets,
        "description": description,
        "tier": tier,
        "source": {
            "files": source_files,
            "details": details,
        },
        "status": status,
    }


def build_ledger(results_dir: Path) -> dict[str, Any]:
    hmm = read_json(results_dir / "multistart_hmm_results.json")
    ff25 = read_json(results_dir / "ff25_overlap_results.json")
    trading = read_json(results_dir / "trading_selected.json")
    hybrid = read_json(results_dir / "hybrid_detector_results.json")

    claims: list[dict[str, Any]] = []
    targets = ["main_icaif.tex", "main_arxiv.tex"]

    claims.append(
        evaluate_numeric_claim(
            claim_id="C01",
            manuscript_targets=targets,
            description="Normal regime HML->SMB meets confirmatory Bonferroni threshold.",
            source_file="multistart_hmm_results.json",
            metric_path="selected_fit.granger.Normal.hml_to_smb.f_p_value",
            tier="confirmatory",
            comparator="<",
            threshold=0.01 / 30.0,
            value=get_nested(hmm, "selected_fit.granger.Normal.hml_to_smb.f_p_value"),
        )
    )
    claims.append(
        evaluate_derived_claim(
            claim_id="C02",
            manuscript_targets=targets,
            description="Elevated regime HML->SMB is nominally significant at 1% but non-confirmatory under Bonferroni.",
            tier="confirmatory",
            source_files=["multistart_hmm_results.json"],
            evaluator=lambda: (
                (
                    "pass"
                    if (
                        (p := get_nested(hmm, "selected_fit.granger.Elevated.hml_to_smb.f_p_value")) is not None
                        and float(p) < 0.01
                        and float(p) >= (0.01 / 30.0)
                    )
                    else ("unresolved" if get_nested(hmm, "selected_fit.granger.Elevated.hml_to_smb.f_p_value") is None else "fail")
                ),
                {
                    "p_value": get_nested(hmm, "selected_fit.granger.Elevated.hml_to_smb.f_p_value"),
                    "alpha_nominal": 0.01,
                    "alpha_bonferroni": 0.01 / 30.0,
                },
            ),
        )
    )
    claims.append(
        evaluate_numeric_claim(
            claim_id="C03",
            manuscript_targets=targets,
            description="Frozen OOS aggregate crisis-level test is significant at conventional 5% level.",
            source_file="multistart_hmm_results.json",
            metric_path="frozen_oos.aggregate.p_value",
            tier="confirmatory",
            comparator="<",
            threshold=0.05,
            value=get_nested(hmm, "frozen_oos.aggregate.p_value"),
        )
    )
    claims.append(
        evaluate_numeric_claim(
            claim_id="C04",
            manuscript_targets=targets,
            description="FF25 spatial gradient permutation p-value is below 0.05.",
            source_file="ff25_overlap_results.json",
            metric_path="primary_test.permutation_p_value",
            tier="confirmatory",
            comparator="<",
            threshold=0.05,
            value=get_nested(ff25, "primary_test.permutation_p_value"),
        )
    )
    claims.append(
        evaluate_numeric_claim(
            claim_id="C05",
            manuscript_targets=targets,
            description="Trading strategy Sharpe is non-positive (no evidence of tradable alpha).",
            source_file="trading_selected.json",
            metric_path="strategy.sharpe_ratio",
            tier="confirmatory",
            comparator="<=",
            threshold=0.0,
            value=get_nested(trading, "strategy.sharpe_ratio"),
        )
    )
    claims.append(
        evaluate_numeric_claim(
            claim_id="C06",
            manuscript_targets=targets,
            description="Hybrid detector conditional coverage (CC) passes at alpha=0.05.",
            source_file="hybrid_detector_results.json",
            metric_path="var_results.hybrid.christoffersen_p_cc",
            tier="confirmatory",
            comparator=">",
            threshold=0.05,
            value=get_nested(hybrid, "var_results.hybrid.christoffersen_p_cc"),
        )
    )
    claims.append(
        evaluate_numeric_claim(
            claim_id="C07",
            manuscript_targets=targets,
            description="Hybrid detector improves COVID detection over HMM-only.",
            source_file="hybrid_detector_results.json",
            metric_path="covid_detection.hybrid_detection_rate_pct",
            tier="exploratory",
            comparator=">",
            threshold=float(get_nested(hybrid, "covid_detection.hmm_only_crisis_pct") or 0.0),
            value=get_nested(hybrid, "covid_detection.hybrid_detection_rate_pct"),
        )
    )

    def eval_event_validation() -> tuple[str, dict[str, Any]]:
        events = get_nested(hmm, "selected_fit.events")
        if not isinstance(events, dict):
            return "unresolved", {"reason": "selected_fit.events not found"}
        sig_count = 0
        dir_count = 0
        for _, row in events.items():
            p_h2s = row.get("hml_to_smb_p")
            p_s2h = row.get("smb_to_hml_p")
            if p_h2s is None or p_s2h is None:
                continue
            if p_h2s < 0.10 and p_s2h > 0.10:
                sig_count += 1
            if p_h2s < p_s2h:
                dir_count += 1
        status = "pass" if sig_count >= 2 else "fail"
        return status, {"significant_count": sig_count, "directional_count": dir_count, "total_events": len(events)}

    claims.append(
        evaluate_derived_claim(
            claim_id="C08",
            manuscript_targets=targets,
            description="Event validation reports significance count separately from directional count.",
            tier="confirmatory",
            source_files=["multistart_hmm_results.json"],
            evaluator=eval_event_validation,
        )
    )

    def eval_hybrid_best() -> tuple[str, dict[str, Any]]:
        models = get_nested(hybrid, "var_results")
        if not isinstance(models, dict) or not models:
            return "unresolved", {"reason": "hybrid var_results not found"}
        distances = {}
        for key, row in models.items():
            if not isinstance(row, dict):
                continue
            label = row.get("model", key)
            dev = row.get("deviation_from_target_pct")
            if dev is not None:
                distances[str(label)] = abs(float(dev))
        if not distances:
            return "unresolved", {"reason": "no model deviations found"}
        best_model = min(distances, key=distances.get)
        status = "pass" if best_model == "Hybrid (HMM+Vol)" else "fail"
        return status, {"best_model": best_model, "distances": distances}

    claims.append(
        evaluate_derived_claim(
            claim_id="C09",
            manuscript_targets=targets,
            description="Hybrid model is closest to 5% target by absolute deviation among hybrid-detector VaR models.",
            tier="exploratory",
            source_files=["hybrid_detector_results.json"],
            evaluator=eval_hybrid_best,
        )
    )

    summary = {
        "total": len(claims),
        "pass": sum(1 for c in claims if c["status"] == "pass"),
        "fail": sum(1 for c in claims if c["status"] == "fail"),
        "unresolved": sum(1 for c in claims if c["status"] == "unresolved"),
    }

    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "results_dir": str(results_dir),
            "description": "Claim ledger linking manuscript-facing claims to recomputed artifacts.",
        },
        "summary": summary,
        "claims": claims,
    }


def main():
    parser = argparse.ArgumentParser(description="Build claim ledger from recomputed result artifacts.")
    parser.add_argument(
        "--results-dir",
        default="/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results",
        help="Directory containing recomputed JSON artifacts.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to <results-dir>/claim_ledger.json",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ledger = build_ledger(results_dir)
    output_path = Path(args.output) if args.output else results_dir / "claim_ledger.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)
    print(f"Wrote claim ledger: {output_path}")
    print(f"Summary: {ledger['summary']}")


if __name__ == "__main__":
    main()
