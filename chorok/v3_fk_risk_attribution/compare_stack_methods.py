"""
Compare LOO vs SHAP Attribution for Stack Dataset
==================================================

Compute Spearman correlation between LOO and SHAP to validate
that FK Uncertainty Attribution captures different information
than SHAP feature importance on a second dataset.
"""

import json
from pathlib import Path
from scipy.stats import spearmanr
import numpy as np

def load_results():
    """Load LOO and SHAP results for Stack."""
    results_dir = Path("chorok/results")

    with open(results_dir / "fk_attribution_stack.json") as f:
        loo_results = json.load(f)

    with open(results_dir / "shap_attribution_stack.json") as f:
        shap_results = json.load(f)

    return loo_results, shap_results


def compute_correlation():
    """Compute Spearman correlation between LOO and SHAP delta rankings."""
    loo_results, shap_results = load_results()

    print("="*60)
    print("LOO vs SHAP Correlation Analysis (Stack)")
    print("="*60)

    # Get post-votes task results
    loo_task = loo_results.get("post-votes", {})
    shap_task = shap_results  # Already single task

    loo_deltas = loo_task.get("deltas", {})
    shap_deltas = shap_task.get("deltas", {})

    if not loo_deltas or not shap_deltas:
        print("Missing results. Run fk_attribution_stack.py and shap_attribution_stack.py first.")
        return

    # Get common FK groups
    common_groups = set(loo_deltas.keys()) & set(shap_deltas.keys())

    print(f"\nCommon FK groups: {sorted(common_groups)}")

    # Extract delta values
    loo_values = []
    shap_values = []

    print("\n--- Delta Values ---")
    print(f"{'FK Group':<25} {'LOO Delta':>12} {'SHAP Delta':>12}")
    print("-" * 50)

    for group in sorted(common_groups):
        loo_delta = loo_deltas[group].get("delta", 0)
        shap_delta = shap_deltas[group].get("delta", 0)
        loo_values.append(loo_delta)
        shap_values.append(shap_delta)
        print(f"{group:<25} {loo_delta:>+12.6f} {shap_delta:>+12.4f}")

    # Compute Spearman correlation
    if len(loo_values) >= 3:
        rho, p_value = spearmanr(loo_values, shap_values)
        print(f"\n--- Spearman Correlation ---")
        print(f"ρ = {rho:.4f}")
        print(f"p-value = {p_value:.4f}")

        # Interpretation
        if abs(rho) < 0.3:
            interpretation = "Weak/No correlation"
        elif abs(rho) < 0.6:
            interpretation = "Moderate correlation"
        else:
            interpretation = "Strong correlation"
        print(f"Interpretation: {interpretation}")
    else:
        print("\nNot enough data points for correlation (need >= 3)")
        rho = None

    # Top-1 comparison
    loo_top = max(loo_deltas.keys(), key=lambda k: abs(loo_deltas[k].get("delta", 0)))
    shap_top = max(shap_deltas.keys(), key=lambda k: abs(shap_deltas[k].get("delta", 0)))

    print(f"\n--- Top FK by Change ---")
    print(f"LOO:  {loo_top}")
    print(f"SHAP: {shap_top}")
    print(f"Match: {'Yes' if loo_top == shap_top else 'No'}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Stack Dataset Validation")
    print("="*60)

    if rho is not None:
        print(f"\nSpearman ρ (LOO vs SHAP): {rho:.4f}")

        if abs(rho) < 0.3:
            print("\n✓ VALIDATES HYPOTHESIS: LOO and SHAP measure different things!")
            print("  FK Uncertainty Attribution is orthogonal to SHAP feature importance")
            print("  on Stack dataset, consistent with SALT findings (ρ=0.064).")
        else:
            print("\n⚠ Higher correlation than SALT. Investigate further.")

    return rho


def compare_with_salt():
    """Compare Stack correlation with SALT correlation."""
    print("\n" + "="*60)
    print("Cross-Dataset Comparison: SALT vs Stack")
    print("="*60)

    # SALT result from previous analysis
    salt_rho = 0.064  # From SESSION_SUMMARY.md

    # Stack result
    stack_rho = compute_correlation()

    if stack_rho is not None:
        print(f"\n{'Dataset':<15} {'LOO vs SHAP ρ':>15}")
        print("-" * 30)
        print(f"{'SALT':<15} {salt_rho:>+15.4f}")
        print(f"{'Stack':<15} {stack_rho:>+15.4f}")

        avg_rho = (salt_rho + stack_rho) / 2
        print(f"{'Average':<15} {avg_rho:>+15.4f}")

        print("\n--- Conclusion ---")
        if abs(avg_rho) < 0.3:
            print("✓ Consistent low correlation across both datasets!")
            print("  This strongly supports the paper's claim that")
            print("  FK Uncertainty Attribution ≠ SHAP Feature Importance.")


if __name__ == "__main__":
    compare_with_salt()
