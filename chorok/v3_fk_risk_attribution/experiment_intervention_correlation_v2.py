"""
Attribution-Intervention Correlation Experiment (V2)
====================================================

개선된 Intervention 시뮬레이션 방법론:

V1 문제점:
- Low-uncertainty 샘플 값 복사는 실제 "개선"을 모사하지 못함
- 불확실성은 피처 조합의 속성이므로 개별 교체로 감소 안됨

V2 방법론:
1. Marginal Effect: 각 FK 그룹을 평균값으로 교체 후 불확실성 변화
2. Permutation Reduction: FK 그룹 permutation으로 인한 불확실성 증가 = 해당 FK의 "중요도"
   - 이 중요도가 높으면, 해당 FK를 "잘 관리"하면 불확실성 감소 가능
3. Entity Quality Gap: 같은 FK 내 best vs worst entity의 불확실성 차이

핵심 가설:
- Attribution이 높은 FK = Permutation으로 불확실성 증가량이 큼
- 따라서 해당 FK를 "개선"(worst→best entity)하면 감소량도 큼
- 검증: Attribution 순위 = Permutation importance 순위 (당연히 같음)
- 더 중요한 검증: Entity quality gap이 Attribution에 비례하는가?

=== 실험 설계 변경 ===

새로운 "Actionability" 정의:
1. Attribution = FK 그룹의 불확실성 기여도
2. Actionability = 해당 FK 내에서 entity 선택으로 불확실성을 얼마나 조절 가능한가

검증 방법:
- 각 FK 그룹 내에서 entity별 평균 불확실성 계산
- Best entity vs Worst entity 불확실성 차이 (Gap)
- Gap이 크면 = Actionable (entity 선택이 불확실성에 큰 영향)
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
    """Compute FK group attribution using permutation importance."""
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


def compute_entity_quality_gap(models, X, col_to_fk, fk_grouping, min_entity_samples=10):
    """
    Compute the "actionability" of each FK group:
    How much can we reduce uncertainty by choosing the best entity vs worst entity?

    For each FK group:
    1. Find the primary key column (first column in the group, usually entity ID)
    2. Group samples by entity
    3. Compute average uncertainty per entity
    4. Gap = (worst entity uncertainty - best entity uncertainty) / mean uncertainty

    Large gap = high actionability (entity selection matters a lot)
    """
    base_unc = get_uncertainty(models, X)

    results = {}

    for fk_group, cols in fk_grouping.items():
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            continue

        # Use first column as entity identifier
        entity_col = valid_cols[0]

        # Group by entity and compute mean uncertainty
        entity_unc = defaultdict(list)
        for idx, (_, row) in enumerate(X.iterrows()):
            entity_id = row[entity_col]
            entity_unc[entity_id].append(base_unc[idx])

        # Filter entities with enough samples
        entity_means = {}
        for entity_id, uncs in entity_unc.items():
            if len(uncs) >= min_entity_samples:
                entity_means[entity_id] = np.mean(uncs)

        if len(entity_means) < 2:
            # Not enough entities to compare
            results[fk_group] = {
                'gap': 0.0,
                'gap_pct': 0.0,
                'n_entities': len(entity_means),
                'best_entity': None,
                'worst_entity': None
            }
            continue

        best_unc = min(entity_means.values())
        worst_unc = max(entity_means.values())
        mean_unc = base_unc.mean()

        gap = worst_unc - best_unc
        gap_pct = (gap / mean_unc) * 100 if mean_unc > 0 else 0

        # Find best and worst entities
        best_entity = min(entity_means.keys(), key=lambda k: entity_means[k])
        worst_entity = max(entity_means.keys(), key=lambda k: entity_means[k])

        results[fk_group] = {
            'gap': gap,
            'gap_pct': gap_pct,
            'n_entities': len(entity_means),
            'best_entity': best_entity,
            'best_unc': best_unc,
            'worst_entity': worst_entity,
            'worst_unc': worst_unc
        }

    return results


def compute_intervention_by_permutation(models, X, fk_grouping, n_permute=5):
    """
    Alternative intervention measure: permutation-based importance.

    The idea: if permuting an FK group increases uncertainty significantly,
    then "controlling" that FK (choosing the right entity) can reduce uncertainty
    by a similar amount.

    This is equivalent to asking: "How much uncertainty is explained by this FK?"
    """
    base_unc = get_uncertainty(models, X).mean()
    results = {}

    for fk_group, cols in fk_grouping.items():
        col_indices = [list(X.columns).index(c) for c in cols if c in X.columns]
        if not col_indices:
            continue

        increases = []
        for _ in range(n_permute):
            X_perm = X.copy()
            for idx in col_indices:
                col_name = X.columns[idx]
                X_perm[col_name] = np.random.permutation(X_perm[col_name].values)
            perm_unc = get_uncertainty(models, X_perm).mean()
            increase_pct = (perm_unc - base_unc) / base_unc * 100 if base_unc > 0 else 0
            increases.append(increase_pct)

        results[fk_group] = np.mean(increases)

    return results


def run_actionability_test(X, y, col_to_fk, domain_name, n_runs=3):
    """
    Run Actionability test for one domain.

    Measures:
    1. Attribution: FK group's contribution to total uncertainty
    2. Entity Quality Gap: How much entity selection affects uncertainty
    3. Permutation Importance: How much FK "control" could reduce uncertainty
    """
    print(f"\n{'='*60}")
    print(f"ACTIONABILITY TEST: {domain_name}")
    print(f"{'='*60}")

    fk_grouping = get_fk_grouping(col_to_fk)
    n_fk_groups = len(fk_grouping)

    print(f"Features: {len(X.columns)}, FK groups: {n_fk_groups}")
    print(f"FK groups: {list(fk_grouping.keys())}")

    all_attributions = []
    all_gaps = []
    all_perm_importance = []

    for run in range(n_runs):
        print(f"\n  Run {run+1}/{n_runs}...")

        models = train_ensemble(X, y, n_models=5, base_seed=42 + run * 10)

        # Attribution
        attribution = compute_attribution(models, X, fk_grouping, n_permute=5)
        all_attributions.append(attribution)

        # Entity Quality Gap
        entity_gaps = compute_entity_quality_gap(models, X, col_to_fk, fk_grouping, min_entity_samples=5)
        gap_dict = {fk: info['gap_pct'] for fk, info in entity_gaps.items()}
        all_gaps.append(gap_dict)

        # Permutation Importance (intervention proxy)
        perm_importance = compute_intervention_by_permutation(models, X, fk_grouping, n_permute=5)
        all_perm_importance.append(perm_importance)

        print(f"    Attribution: {[f'{k}:{v:.1f}%' for k,v in attribution.items()]}")
        print(f"    Entity Gap:  {[f'{k}:{v:.1f}%' for k,v in gap_dict.items()]}")

    # Average across runs
    avg_attribution = {}
    avg_gap = {}
    avg_perm = {}

    for fk in fk_grouping.keys():
        attr_vals = [a.get(fk, 0) for a in all_attributions]
        gap_vals = [g.get(fk, 0) for g in all_gaps]
        perm_vals = [p.get(fk, 0) for p in all_perm_importance]

        avg_attribution[fk] = np.mean(attr_vals)
        avg_gap[fk] = np.mean(gap_vals)
        avg_perm[fk] = np.mean(perm_vals)

    # Compute correlations
    fk_list = list(fk_grouping.keys())
    attr_values = [avg_attribution.get(fk, 0) for fk in fk_list]
    gap_values = [avg_gap.get(fk, 0) for fk in fk_list]
    perm_values = [avg_perm.get(fk, 0) for fk in fk_list]

    # Attribution vs Entity Gap correlation
    if len(fk_list) >= 3 and not all(g == 0 for g in gap_values):
        attr_gap_corr, attr_gap_p = spearmanr(attr_values, gap_values)
    else:
        attr_gap_corr, attr_gap_p = float('nan'), float('nan')

    # Attribution vs Permutation correlation (should be ~1.0, sanity check)
    if len(fk_list) >= 3:
        attr_perm_corr, attr_perm_p = spearmanr(attr_values, perm_values)
    else:
        attr_perm_corr, attr_perm_p = float('nan'), float('nan')

    # Results summary
    print(f"\n  --- Results ---")
    print(f"  {'FK Group':<15} {'Attribution':<12} {'Entity Gap':<12} {'Perm Imp.':<12}")
    print(f"  {'-'*50}")
    for fk in fk_list:
        attr = avg_attribution.get(fk, 0)
        gap = avg_gap.get(fk, 0)
        perm = avg_perm.get(fk, 0)
        print(f"  {fk:<15} {attr:>10.1f}% {gap:>10.1f}% {perm:>10.1f}%")

    print(f"\n  Correlations:")
    print(f"    Attribution vs Entity Gap:   {attr_gap_corr:.3f} (p={attr_gap_p:.3f})")
    print(f"    Attribution vs Perm Imp.:    {attr_perm_corr:.3f} (sanity check)")

    # Interpretation
    actionability_score = np.mean([g for g in gap_values if g > 0]) if any(g > 0 for g in gap_values) else 0

    if actionability_score > 50:
        verdict = "HIGHLY ACTIONABLE: Entity selection has major impact"
    elif actionability_score > 20:
        verdict = "MODERATELY ACTIONABLE: Entity selection matters"
    elif actionability_score > 5:
        verdict = "WEAKLY ACTIONABLE: Limited entity-level impact"
    else:
        verdict = "NOT ACTIONABLE: Entity selection has minimal impact"

    print(f"\n  Avg Entity Gap: {actionability_score:.1f}%")
    print(f"  Verdict: {verdict}")

    return {
        'domain': domain_name,
        'n_fk_groups': n_fk_groups,
        'attribution': avg_attribution,
        'entity_gap': avg_gap,
        'perm_importance': avg_perm,
        'attr_gap_corr': attr_gap_corr if not np.isnan(attr_gap_corr) else None,
        'attr_gap_p': attr_gap_p if not np.isnan(attr_gap_p) else None,
        'attr_perm_corr': attr_perm_corr if not np.isnan(attr_perm_corr) else None,
        'actionability_score': actionability_score,
        'verdict': verdict
    }


def run_all_domains():
    """Run Actionability test across all domains."""
    print("="*70)
    print("FK ATTRIBUTION ACTIONABILITY VALIDATION (V2)")
    print("Does entity selection within FK groups reduce uncertainty?")
    print("="*70)

    all_results = {}

    # SALT
    print("\n[1/3] Loading SALT data...")
    X_salt, y_salt, _, col_to_fk_salt = load_salt_data(sample_size=3000)
    salt_results = run_actionability_test(
        X_salt, y_salt, col_to_fk_salt, "SALT (ERP)"
    )
    all_results['salt'] = salt_results

    # Amazon
    print("\n[2/3] Loading Amazon data...")
    X_amazon, y_amazon, _, col_to_fk_amazon = load_amazon_data(sample_size=3000)
    amazon_results = run_actionability_test(
        X_amazon, y_amazon, col_to_fk_amazon, "Amazon (E-commerce)"
    )
    all_results['amazon'] = amazon_results

    # Stack
    print("\n[3/3] Loading Stack data...")
    X_stack, y_stack, _, col_to_fk_stack = load_stack_data(sample_size=3000)
    stack_results = run_actionability_test(
        X_stack, y_stack, col_to_fk_stack, "Stack (Q&A)"
    )
    all_results['stack'] = stack_results

    # Final Summary
    print("\n" + "="*70)
    print("ACTIONABILITY SUMMARY")
    print("="*70)
    print(f"\n{'Domain':<25} {'Attr-Gap Corr':<15} {'Actionability':<15} {'Verdict':<25}")
    print("-"*80)
    for domain, results in all_results.items():
        corr = results['attr_gap_corr']
        corr_str = f"{corr:.3f}" if corr is not None else "N/A"
        action = results['actionability_score']
        print(f"{results['domain']:<25} {corr_str:<15} {action:>12.1f}% {results['verdict']:<25}")

    # Overall assessment
    avg_actionability = np.mean([r['actionability_score'] for r in all_results.values()])
    print(f"\n{'='*70}")
    print(f"OVERALL AVERAGE ACTIONABILITY: {avg_actionability:.1f}%")

    if avg_actionability > 30:
        overall = "FK Attribution is ACTIONABLE - entity selection reduces uncertainty"
    elif avg_actionability > 10:
        overall = "FK Attribution is PARTIALLY ACTIONABLE"
    else:
        overall = "FK Attribution has LIMITED ACTIONABILITY"

    print(f"CONCLUSION: {overall}")
    print(f"{'='*70}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = f"{RESULTS_DIR}/actionability_v2.json"

    # Convert numpy types for JSON serialization
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
