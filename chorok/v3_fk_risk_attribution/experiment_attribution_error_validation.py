"""
Attribution-Error Validation: The Most Important Test
======================================================

핵심 질문: FK Attribution이 실제 예측 오차와 연결되는가?

현재까지의 한계:
- Attribution = FK permute → 불확실성 증가
- 검증 = FK permute → 불확실성 증가 확인 (순환!)

진짜 검증:
- Attribution (불확실성 기반) vs Error Impact (예측 오차 기반)
- 이 둘이 상관되면 → Attribution이 실제 예측 성능과 연결됨

방법:
1. 각 FK 그룹별 Attribution 계산 (불확실성 증가량)
2. 각 FK 그룹별 Error Impact 계산 (예측 오차 증가량)
3. 두 순위의 상관관계 측정

기대:
- Attribution 높은 FK → Error Impact도 높음
- Spearman correlation > 0.7 이면 PASS
"""

import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from data_loader_salt import load_salt_data
from data_loader_amazon import load_amazon_data
from data_loader_stack import load_stack_data
from data_loader_trial import load_trial_data

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


def get_prediction(models, X):
    """Get ensemble mean prediction."""
    preds = np.array([m.predict(X) for m in models])
    return preds.mean(axis=0)


def get_fk_grouping(col_to_fk):
    """Convert column->FK mapping to FK->columns mapping."""
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)
    return dict(fk_to_cols)


def compute_uncertainty_attribution(models, X, fk_grouping, n_permute=10):
    """
    Compute FK attribution based on UNCERTAINTY increase when FK is permuted.
    This is our current method.
    """
    base_unc = get_uncertainty(models, X).mean()
    results = {}

    for fk_group, cols in fk_grouping.items():
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            continue

        deltas = []
        for _ in range(n_permute):
            X_perm = X.copy()
            for col in valid_cols:
                X_perm[col] = np.random.permutation(X_perm[col].values)
            perm_unc = get_uncertainty(models, X_perm).mean()
            deltas.append(perm_unc - base_unc)

        results[fk_group] = np.mean(deltas)

    # Normalize to percentages
    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def compute_error_impact(models, X, y, fk_grouping, n_permute=10):
    """
    Compute FK impact based on PREDICTION ERROR increase when FK is permuted.
    This is the ground truth we're validating against.
    """
    # Base prediction error
    base_pred = get_prediction(models, X)
    base_mae = mean_absolute_error(y, base_pred)

    results = {}

    for fk_group, cols in fk_grouping.items():
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            continue

        deltas = []
        for _ in range(n_permute):
            X_perm = X.copy()
            for col in valid_cols:
                X_perm[col] = np.random.permutation(X_perm[col].values)
            perm_pred = get_prediction(models, X_perm)
            perm_mae = mean_absolute_error(y, perm_pred)
            deltas.append(perm_mae - base_mae)

        results[fk_group] = np.mean(deltas)

    # Normalize to percentages
    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def run_attribution_error_validation(X, y, col_to_fk, domain_name, n_runs=3):
    """
    The most important validation test:
    Does uncertainty-based attribution match error-based impact?
    """
    print(f"\n{'='*60}")
    print(f"ATTRIBUTION-ERROR VALIDATION: {domain_name}")
    print(f"{'='*60}")

    fk_grouping = get_fk_grouping(col_to_fk)
    n_fk_groups = len(fk_grouping)

    print(f"Features: {len(X.columns)}, FK groups: {n_fk_groups}")
    print(f"FK groups: {list(fk_grouping.keys())}")

    all_unc_attr = []
    all_error_impact = []

    for run in range(n_runs):
        print(f"\n  Run {run+1}/{n_runs}...")

        # Train ensemble
        models = train_ensemble(X, y, n_models=5, base_seed=42 + run * 10)

        # Uncertainty-based attribution (our method)
        unc_attr = compute_uncertainty_attribution(models, X, fk_grouping, n_permute=5)
        all_unc_attr.append(unc_attr)

        # Error-based impact (ground truth)
        error_impact = compute_error_impact(models, X, y, fk_grouping, n_permute=5)
        all_error_impact.append(error_impact)

        print(f"    Unc Attribution: {[f'{k}:{v:.1f}%' for k,v in unc_attr.items()]}")
        print(f"    Error Impact:    {[f'{k}:{v:.1f}%' for k,v in error_impact.items()]}")

    # Average across runs
    avg_unc_attr = {}
    avg_error_impact = {}

    for fk in fk_grouping.keys():
        unc_vals = [a.get(fk, 0) for a in all_unc_attr]
        err_vals = [e.get(fk, 0) for e in all_error_impact]
        avg_unc_attr[fk] = np.mean(unc_vals)
        avg_error_impact[fk] = np.mean(err_vals)

    # Compute correlation
    fk_list = list(fk_grouping.keys())
    unc_values = [avg_unc_attr.get(fk, 0) for fk in fk_list]
    err_values = [avg_error_impact.get(fk, 0) for fk in fk_list]

    if len(fk_list) >= 3:
        spearman_corr, spearman_p = spearmanr(unc_values, err_values)
        pearson_corr, pearson_p = pearsonr(unc_values, err_values)
    else:
        spearman_corr, spearman_p = float('nan'), float('nan')
        pearson_corr, pearson_p = float('nan'), float('nan')

    # Results summary
    print(f"\n  --- Results ---")
    print(f"  {'FK Group':<15} {'Unc Attr':<12} {'Error Impact':<12} {'Match?':<10}")
    print(f"  {'-'*50}")

    # Check ranking match
    unc_ranking = sorted(fk_list, key=lambda k: -avg_unc_attr.get(k, 0))
    err_ranking = sorted(fk_list, key=lambda k: -avg_error_impact.get(k, 0))

    for fk in fk_list:
        unc = avg_unc_attr.get(fk, 0)
        err = avg_error_impact.get(fk, 0)
        unc_rank = unc_ranking.index(fk) + 1
        err_rank = err_ranking.index(fk) + 1
        match = "Yes" if abs(unc_rank - err_rank) <= 1 else "No"
        print(f"  {fk:<15} {unc:>10.1f}% {err:>10.1f}% {match:<10}")

    print(f"\n  Ranking Comparison:")
    print(f"    Unc Attribution: {unc_ranking}")
    print(f"    Error Impact:    {err_ranking}")

    print(f"\n  Correlation:")
    print(f"    Spearman: {spearman_corr:.3f} (p={spearman_p:.3f})")
    print(f"    Pearson:  {pearson_corr:.3f} (p={pearson_p:.3f})")

    # Verdict
    if spearman_corr > 0.7:
        verdict = "STRONG MATCH: Attribution reflects true error impact"
    elif spearman_corr > 0.4:
        verdict = "MODERATE MATCH: Attribution partially reflects error impact"
    elif spearman_corr > 0.0:
        verdict = "WEAK MATCH: Attribution has limited connection to error"
    else:
        verdict = "NO MATCH: Attribution does not reflect error impact"

    print(f"\n  Verdict: {verdict}")

    return {
        'domain': domain_name,
        'n_fk_groups': n_fk_groups,
        'unc_attribution': avg_unc_attr,
        'error_impact': avg_error_impact,
        'unc_ranking': unc_ranking,
        'error_ranking': err_ranking,
        'spearman_corr': spearman_corr if not np.isnan(spearman_corr) else None,
        'spearman_p': spearman_p if not np.isnan(spearman_p) else None,
        'pearson_corr': pearson_corr if not np.isnan(pearson_corr) else None,
        'verdict': verdict
    }


def run_all_domains():
    """Run Attribution-Error validation across all domains."""
    print("="*70)
    print("ATTRIBUTION-ERROR VALIDATION")
    print("The Most Important Test: Does Attribution Match Error Impact?")
    print("="*70)

    all_results = {}

    # SALT (ERP - transactional, error propagation expected)
    print("\n[1/4] Loading SALT data...")
    X_salt, y_salt, _, col_to_fk_salt = load_salt_data(sample_size=3000)
    salt_results = run_attribution_error_validation(
        X_salt, y_salt, col_to_fk_salt, "SALT (ERP)"
    )
    all_results['salt'] = salt_results

    # Trial (Clinical Trials - error propagation expected)
    print("\n[2/4] Loading Trial data...")
    X_trial, y_trial, _, col_to_fk_trial = load_trial_data(sample_size=3000)
    trial_results = run_attribution_error_validation(
        X_trial, y_trial, col_to_fk_trial, "Trial (Clinical)"
    )
    all_results['trial'] = trial_results

    # Amazon (E-commerce)
    print("\n[3/4] Loading Amazon data...")
    X_amazon, y_amazon, _, col_to_fk_amazon = load_amazon_data(sample_size=3000)
    amazon_results = run_attribution_error_validation(
        X_amazon, y_amazon, col_to_fk_amazon, "Amazon (E-commerce)"
    )
    all_results['amazon'] = amazon_results

    # Stack (Q&A - no error propagation expected)
    print("\n[4/4] Loading Stack data...")
    X_stack, y_stack, _, col_to_fk_stack = load_stack_data(sample_size=3000)
    stack_results = run_attribution_error_validation(
        X_stack, y_stack, col_to_fk_stack, "Stack (Q&A)"
    )
    all_results['stack'] = stack_results

    # Final Summary
    print("\n" + "="*70)
    print("ATTRIBUTION-ERROR VALIDATION SUMMARY")
    print("="*70)
    print(f"\n{'Domain':<25} {'Spearman':<12} {'p-value':<12} {'Verdict':<30}")
    print("-"*70)
    for domain, results in all_results.items():
        corr = results['spearman_corr']
        corr_str = f"{corr:.3f}" if corr is not None else "N/A"
        p_val = results['spearman_p']
        p_str = f"{p_val:.3f}" if p_val is not None else "N/A"
        print(f"{results['domain']:<25} {corr_str:<12} {p_str:<12} {results['verdict']:<30}")

    # Overall assessment
    valid_corrs = [r['spearman_corr'] for r in all_results.values() if r['spearman_corr'] is not None]
    avg_corr = np.mean(valid_corrs) if valid_corrs else 0

    print(f"\n{'='*70}")
    print(f"OVERALL AVERAGE CORRELATION: {avg_corr:.3f}")

    if avg_corr > 0.7:
        overall = "VALIDATED: FK Attribution accurately reflects prediction error impact"
    elif avg_corr > 0.4:
        overall = "PARTIALLY VALIDATED: Attribution has moderate connection to error"
    else:
        overall = "NOT VALIDATED: Attribution does not match error impact"

    print(f"CONCLUSION: {overall}")
    print(f"{'='*70}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = f"{RESULTS_DIR}/attribution_error_validation.json"

    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if obj is None:
            return None
        return obj

    serializable_results = convert_to_serializable(all_results)

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n[SAVED] {output_path}")

    return all_results


if __name__ == "__main__":
    results = run_all_domains()
