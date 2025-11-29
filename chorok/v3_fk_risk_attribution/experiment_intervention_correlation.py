"""
Attribution-Intervention Correlation Experiment
================================================

핵심 질문: FK Attribution 순위가 실제 개선 효과 순위와 일치하는가?

예시:
- Attribution: ITEM 31%, SALESGROUP 21%, SALESDOCUMENT 20%...
- 검증: ITEM 개선 시 불확실성 감소가 가장 큰가?

이 실험은 "actionable" 주장의 정당성을 검증합니다.
FK Attribution이 높은 그룹을 개선하면 실제로 불확실성이 더 많이 감소해야 합니다.

=== METHODOLOGY ===

1. Attribution 계산: 각 FK 그룹의 불확실성 기여도 측정
2. Intervention 시뮬레이션: 각 FK 그룹을 "개선"했을 때 불확실성 감소 측정
   - "개선" = 해당 FK 그룹 피처를 low-uncertainty 샘플의 값으로 교체
3. Correlation 계산: Attribution 순위 vs Intervention 효과 순위의 상관관계

=== EXPECTED RESULTS ===

Strong correlation (>0.7) = FK Attribution이 actionable함
Weak correlation (<0.3) = Attribution이 실제 개선과 무관 -> 문제
"""

import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from data_loader_salt import load_salt_data
from data_loader_amazon import load_amazon_data
from data_loader_stack import load_stack_data

RESULTS_DIR = '/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/results'


def train_ensemble(X, y, n_models=5, base_seed=42):
    """Train ensemble for uncertainty estimation."""
    models = []
    for i in range(n_models):
        model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=base_seed + i, verbose=-1
        )
        model.fit(X, y)
        models.append(model)
    return models


def get_uncertainty(models, X):
    """Compute ensemble uncertainty (variance of predictions)."""
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0)


def get_fk_grouping(col_to_fk):
    """Convert column->FK mapping to FK->columns mapping."""
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)
    return dict(fk_to_cols)


def compute_attribution(models, X, fk_grouping, n_permute=10):
    """
    Compute FK group attribution using permutation importance.

    Returns: Dict[FK_group: attribution_percentage]
    """
    base_unc = get_uncertainty(models, X).mean()
    results = {}

    for fk_group, cols in fk_grouping.items():
        col_indices = [list(X.columns).index(c) for c in cols if c in X.columns]
        if not col_indices:
            continue

        deltas = []
        for _ in range(n_permute):
            X_perm = X.copy()
            for idx in col_indices:
                col_name = X.columns[idx]
                X_perm[col_name] = np.random.permutation(X_perm[col_name].values)
            perm_unc = get_uncertainty(models, X_perm).mean()
            deltas.append(perm_unc - base_unc)

        results[fk_group] = np.mean(deltas)

    # Normalize to percentages
    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def compute_intervention_effect(models, X, y, fk_grouping, n_samples=100):
    """
    Compute the effect of "improving" each FK group.

    Intervention simulation:
    1. Find low-uncertainty samples (bottom 20%)
    2. For each FK group, replace high-uncertainty samples' values
       with values from low-uncertainty samples
    3. Measure uncertainty reduction

    Returns: Dict[FK_group: uncertainty_reduction_percentage]
    """
    base_unc = get_uncertainty(models, X)
    base_mean = base_unc.mean()

    # Find low-uncertainty samples (these represent "good" patterns)
    low_unc_mask = base_unc <= np.percentile(base_unc, 20)
    high_unc_mask = base_unc >= np.percentile(base_unc, 80)

    X_low = X[low_unc_mask]
    X_high = X[high_unc_mask]

    if len(X_low) < 10 or len(X_high) < 10:
        print("  [WARNING] Insufficient samples for intervention simulation")
        return {}

    results = {}

    for fk_group, cols in fk_grouping.items():
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            continue

        # Intervention: replace high-uncertainty samples' FK group values
        # with values sampled from low-uncertainty samples
        X_intervened = X_high.copy()

        for col in valid_cols:
            # Sample values from low-uncertainty samples
            low_values = X_low[col].values
            sampled = np.random.choice(low_values, size=len(X_high), replace=True)
            X_intervened[col] = sampled

        # Measure new uncertainty
        new_unc = get_uncertainty(models, X_intervened).mean()
        high_unc_baseline = get_uncertainty(models, X_high).mean()

        # Reduction as percentage
        reduction_pct = (high_unc_baseline - new_unc) / high_unc_baseline * 100
        results[fk_group] = max(0, reduction_pct)

    return results


def run_intervention_correlation_test(X, y, col_to_fk, domain_name, n_runs=3):
    """
    Run Attribution-Intervention Correlation test for one domain.

    This validates: Does higher attribution → larger intervention effect?
    """
    print(f"\n{'='*60}")
    print(f"ATTRIBUTION-INTERVENTION CORRELATION: {domain_name}")
    print(f"{'='*60}")

    fk_grouping = get_fk_grouping(col_to_fk)
    n_fk_groups = len(fk_grouping)

    print(f"Features: {len(X.columns)}, FK groups: {n_fk_groups}")
    print(f"FK groups: {list(fk_grouping.keys())}")

    all_attributions = []
    all_interventions = []

    for run in range(n_runs):
        print(f"\n  Run {run+1}/{n_runs}...")

        # Train ensemble with different seed
        models = train_ensemble(X, y, n_models=5, base_seed=42 + run * 10)

        # Compute attribution
        attribution = compute_attribution(models, X, fk_grouping, n_permute=5)
        print(f"    Attribution: {attribution}")

        # Compute intervention effect
        intervention = compute_intervention_effect(models, X, y, fk_grouping)
        print(f"    Intervention: {intervention}")

        all_attributions.append(attribution)
        all_interventions.append(intervention)

    # Average across runs
    avg_attribution = {}
    avg_intervention = {}

    for fk in fk_grouping.keys():
        attr_vals = [a.get(fk, 0) for a in all_attributions]
        int_vals = [i.get(fk, 0) for i in all_interventions]
        avg_attribution[fk] = np.mean(attr_vals)
        avg_intervention[fk] = np.mean(int_vals)

    # Compute correlation
    fk_list = list(fk_grouping.keys())
    attr_values = [avg_attribution.get(fk, 0) for fk in fk_list]
    int_values = [avg_intervention.get(fk, 0) for fk in fk_list]

    if len(fk_list) >= 3:
        spearman_corr, spearman_p = spearmanr(attr_values, int_values)
        pearson_corr, pearson_p = pearsonr(attr_values, int_values)
    else:
        spearman_corr, spearman_p = 0.0, 1.0
        pearson_corr, pearson_p = 0.0, 1.0

    # Results summary
    print(f"\n  --- Results ---")
    print(f"  {'FK Group':<15} {'Attribution':<15} {'Intervention':<15}")
    print(f"  {'-'*45}")
    for fk in fk_list:
        attr = avg_attribution.get(fk, 0)
        intv = avg_intervention.get(fk, 0)
        print(f"  {fk:<15} {attr:>12.1f}% {intv:>12.1f}%")

    print(f"\n  Spearman Correlation: {spearman_corr:.3f} (p={spearman_p:.3f})")
    print(f"  Pearson Correlation:  {pearson_corr:.3f} (p={pearson_p:.3f})")

    # Interpretation
    if spearman_corr > 0.7:
        verdict = "STRONG: Attribution is highly actionable"
    elif spearman_corr > 0.4:
        verdict = "MODERATE: Attribution is reasonably actionable"
    elif spearman_corr > 0.0:
        verdict = "WEAK: Attribution has limited actionability"
    else:
        verdict = "NEGATIVE: Attribution does not predict intervention effect"

    print(f"  Verdict: {verdict}")

    return {
        'domain': domain_name,
        'n_fk_groups': n_fk_groups,
        'attribution': avg_attribution,
        'intervention': avg_intervention,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'verdict': verdict
    }


def run_all_domains():
    """Run Attribution-Intervention Correlation test across all domains."""
    print("="*70)
    print("ATTRIBUTION-INTERVENTION CORRELATION VALIDATION")
    print("Does FK Attribution predict actual intervention effectiveness?")
    print("="*70)

    all_results = {}

    # SALT
    print("\n[1/3] Loading SALT data...")
    X_salt, y_salt, _, col_to_fk_salt = load_salt_data(sample_size=3000)
    salt_results = run_intervention_correlation_test(
        X_salt, y_salt, col_to_fk_salt, "SALT (ERP)"
    )
    all_results['salt'] = salt_results

    # Amazon
    print("\n[2/3] Loading Amazon data...")
    X_amazon, y_amazon, _, col_to_fk_amazon = load_amazon_data(sample_size=3000)
    amazon_results = run_intervention_correlation_test(
        X_amazon, y_amazon, col_to_fk_amazon, "Amazon (E-commerce)"
    )
    all_results['amazon'] = amazon_results

    # Stack
    print("\n[3/3] Loading Stack data...")
    X_stack, y_stack, _, col_to_fk_stack = load_stack_data(sample_size=3000)
    stack_results = run_intervention_correlation_test(
        X_stack, y_stack, col_to_fk_stack, "Stack (Q&A)"
    )
    all_results['stack'] = stack_results

    # Final Summary
    print("\n" + "="*70)
    print("ATTRIBUTION-INTERVENTION CORRELATION SUMMARY")
    print("="*70)
    print(f"\n{'Domain':<25} {'Spearman':<12} {'p-value':<12} {'Verdict':<30}")
    print("-"*70)
    for domain, results in all_results.items():
        print(f"{results['domain']:<25} {results['spearman_corr']:>10.3f} "
              f"{results['spearman_p']:>10.3f} {results['verdict']:<30}")

    # Overall assessment
    avg_corr = np.mean([r['spearman_corr'] for r in all_results.values()])
    print(f"\n{'='*70}")
    print(f"OVERALL AVERAGE CORRELATION: {avg_corr:.3f}")

    if avg_corr > 0.5:
        overall = "FK Attribution is ACTIONABLE across domains"
    elif avg_corr > 0.2:
        overall = "FK Attribution is PARTIALLY ACTIONABLE"
    else:
        overall = "WARNING: FK Attribution may not be actionable"

    print(f"CONCLUSION: {overall}")
    print(f"{'='*70}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = f"{RESULTS_DIR}/intervention_correlation.json"

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        return obj

    serializable_results = convert_to_serializable(all_results)

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n[SAVED] {output_path}")

    return all_results


if __name__ == "__main__":
    results = run_all_domains()
