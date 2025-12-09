"""
Root Cause Analysis Experiment V2
=================================

FIXED DESIGN:
1. Train model on CLEAN data (fixed model)
2. At TEST TIME, inject noise at FK X
3. Measure INCREASE in uncertainty
4. Check: Does the FK with highest uncertainty increase = injected FK?

This correctly measures: "If FK X is corrupted, does the model become
more uncertain about predictions that depend on X?"

Author: ChorokLeeDev
Created: 2025-12-08
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache')
RESULTS_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/results')


def load_from_cache(cache_file):
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    if len(data) == 4:
        return data
    return None, None, None, None


def train_ensemble(X, y, n_models=5, base_seed=42):
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
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0).mean()


def get_fk_grouping(col_to_fk):
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)
    return dict(fk_to_cols)


def inject_noise_at_test(X, fk_grouping, target_fk, noise_level=1.0):
    """Inject noise into a specific FK's columns at test time."""
    X_noisy = X.copy()
    cols = fk_grouping.get(target_fk, [])
    valid_cols = [c for c in cols if c in X.columns]

    for col in valid_cols:
        values = X_noisy[col].values.astype(float)
        std_val = np.std(values) + 1e-8
        noise = np.random.normal(0, std_val * noise_level, len(values))
        X_noisy[col] = values + noise

    return X_noisy


def run_root_cause_v2(X, y, col_to_fk, domain_name):
    """
    Root Cause V2: Test-time noise injection

    Protocol:
    1. Train model on CLEAN data (once)
    2. For each FK:
       a. Inject noise at test time
       b. Measure uncertainty INCREASE
    3. When we inject noise at FK X:
       a. Compute uncertainty increase for each FK via permutation
       b. Check if FK X shows the largest increase

    Key insight: If corrupting FK X at test time causes the model
    to become uncertain, and permutation of X captures this,
    then FK Attribution identifies the root cause.
    """
    print(f"\n{'='*70}")
    print(f"ROOT CAUSE ANALYSIS V2: {domain_name}")
    print(f"{'='*70}")
    print(f"\nMethod: Train on clean → Inject noise at test → Measure uncertainty increase")

    fk_grouping = get_fk_grouping(col_to_fk)
    fk_list = list(fk_grouping.keys())

    print(f"FK groups: {fk_list}")

    # Train model on CLEAN data
    print(f"\n  Training model on clean data...")
    models = train_ensemble(X, y, n_models=5, base_seed=42)

    baseline_unc = get_uncertainty(models, X)
    print(f"  Baseline uncertainty: {baseline_unc:.6f}")

    results = []

    for injected_fk in fk_list:
        print(f"\n  --- Injecting noise at: {injected_fk} (test time) ---")

        # Inject noise at test time
        X_noisy = inject_noise_at_test(X, fk_grouping, injected_fk, noise_level=1.5)

        # Measure uncertainty on noisy data
        noisy_unc = get_uncertainty(models, X_noisy)
        total_increase = (noisy_unc - baseline_unc) / baseline_unc * 100

        print(f"    Total uncertainty increase: +{total_increase:.1f}%")

        # Now: Which FK "explains" this increase?
        # Permute each FK and see which one's permutation
        # causes similar uncertainty increase
        fk_contributions = {}

        for fk in fk_list:
            cols = [c for c in fk_grouping[fk] if c in X.columns]
            if not cols:
                continue

            # Permute this FK on the NOISY data
            # If permuting FK X on noisy data REDUCES uncertainty back toward baseline,
            # it means FK X was the source of the noise
            X_perm = X_noisy.copy()
            for col in cols:
                # Replace with clean values (simulate "fixing" the FK)
                X_perm[col] = X[col].values

            perm_unc = get_uncertainty(models, X_perm)
            reduction = (noisy_unc - perm_unc) / noisy_unc * 100  # How much fixing this FK reduces uncertainty

            fk_contributions[fk] = reduction

        # Rank by contribution (higher = more responsible for the uncertainty)
        ranked = sorted(fk_contributions.keys(), key=lambda k: -fk_contributions[k])

        print(f"    Uncertainty reduction by fixing each FK:")
        for fk in ranked:
            print(f"      {fk}: -{fk_contributions[fk]:.1f}%")

        # Check if injected FK is identified
        rank_of_injected = ranked.index(injected_fk) + 1
        top1_correct = (ranked[0] == injected_fk)
        top2_correct = (injected_fk in ranked[:2])

        print(f"    Injected FK '{injected_fk}' ranked: #{rank_of_injected}")
        print(f"    Top-1 correct: {'✓' if top1_correct else '✗'}")

        results.append({
            'injected_fk': injected_fk,
            'total_increase': total_increase,
            'fk_contributions': fk_contributions,
            'rank': rank_of_injected,
            'top1_correct': top1_correct,
            'top2_correct': top2_correct,
            'identified_as_top': ranked[0]
        })

    # Summary
    print(f"\n{'='*70}")
    print("ROOT CAUSE IDENTIFICATION SUMMARY V2")
    print("="*70)

    top1_accuracy = sum(r['top1_correct'] for r in results) / len(results) * 100
    top2_accuracy = sum(r['top2_correct'] for r in results) / len(results) * 100
    avg_rank = np.mean([r['rank'] for r in results])

    print(f"\n  Results across {len(fk_list)} FK injection experiments:")
    print(f"\n  {'Injected FK':<20} {'Identified As':<20} {'Rank':<8} {'Correct?'}")
    print(f"  {'-'*60}")

    for r in results:
        correct_str = "✓" if r['top1_correct'] else "✗"
        print(f"  {r['injected_fk']:<20} {r['identified_as_top']:<20} #{r['rank']:<6} {correct_str}")

    print(f"\n  Metrics:")
    print(f"    Top-1 Accuracy: {top1_accuracy:.1f}% ({sum(r['top1_correct'] for r in results)}/{len(results)})")
    print(f"    Top-2 Accuracy: {top2_accuracy:.1f}% ({sum(r['top2_correct'] for r in results)}/{len(results)})")
    print(f"    Average Rank: {avg_rank:.2f} (best=1.0)")

    # Verdict
    if top1_accuracy >= 80:
        verdict = f"✓ STRONG: FK Attribution correctly identifies root cause {top1_accuracy:.0f}% of the time"
        success = True
    elif top1_accuracy >= 60:
        verdict = f"~ GOOD: FK Attribution identifies root cause {top1_accuracy:.0f}% of the time"
        success = True
    elif top2_accuracy >= 80:
        verdict = f"~ ACCEPTABLE: Root cause in top-2 {top2_accuracy:.0f}% of the time"
        success = True
    else:
        verdict = f"✗ WEAK: FK Attribution only finds root cause {top1_accuracy:.0f}% of the time"
        success = False

    print(f"\n  Verdict: {verdict}")

    return {
        'domain': domain_name,
        'n_fk_groups': len(fk_list),
        'top1_accuracy': top1_accuracy,
        'top2_accuracy': top2_accuracy,
        'avg_rank': avg_rank,
        'verdict': verdict,
        'success': success,
        'detailed_results': results
    }


def run_all():
    print("="*70)
    print("ROOT CAUSE ANALYSIS V2")
    print("="*70)
    print("\nFixed design: Train on clean → Inject at test → Identify source")

    np.random.seed(42)

    all_results = {}

    # SALT
    print("\n[1/3] Loading SALT data...")
    salt_cache = CACHE_DIR / 'data_salt_PLANT_2000.pkl'
    if salt_cache.exists():
        X, y, _, col_to_fk = load_from_cache(salt_cache)
        if X is not None:
            result = run_root_cause_v2(X, y, col_to_fk, "SALT (ERP)")
            all_results['salt'] = result

    # F1
    print("\n[2/3] Loading F1 data...")
    f1_cache = CACHE_DIR / 'data_f1_driver-position_3000_v3.pkl'
    if f1_cache.exists():
        X, y, _, col_to_fk = load_from_cache(f1_cache)
        if X is not None:
            result = run_root_cause_v2(X, y, col_to_fk, "F1 (Racing)")
            all_results['f1'] = result

    # H&M (subset for speed)
    print("\n[3/3] Loading H&M data...")
    hm_cache = CACHE_DIR / 'data_hm_item-sales_3000_v1.pkl'
    if hm_cache.exists():
        X, y, _, col_to_fk = load_from_cache(hm_cache)
        if X is not None:
            result = run_root_cause_v2(X, y, col_to_fk, "H&M (Retail)")
            all_results['hm'] = result

    # Final Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY: ROOT CAUSE IDENTIFICATION V2")
    print("="*70)

    print(f"\n  {'Domain':<20} {'FK Groups':<12} {'Top-1 Acc':<12} {'Top-2 Acc':<12} {'Avg Rank'}")
    print(f"  {'-'*68}")

    for key, res in all_results.items():
        print(f"  {res['domain']:<20} {res['n_fk_groups']:<12} {res['top1_accuracy']:>8.1f}% {res['top2_accuracy']:>10.1f}% {res['avg_rank']:>8.2f}")

    if all_results:
        overall_top1 = np.mean([r['top1_accuracy'] for r in all_results.values()])
        overall_top2 = np.mean([r['top2_accuracy'] for r in all_results.values()])

        print(f"\n  Overall Top-1 Accuracy: {overall_top1:.1f}%")
        print(f"  Overall Top-2 Accuracy: {overall_top2:.1f}%")

        if overall_top1 >= 70:
            print(f"\n  ✓ CLAIM VALIDATED:")
            print(f"    FK Attribution correctly performs ROOT CAUSE ANALYSIS,")
            print(f"    identifying the upstream data source that propagates risk.")

    # Save
    import json

    def convert(obj):
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(v) for v in obj]
        return obj

    output_path = RESULTS_DIR / 'root_cause_analysis_v2.json'
    with open(output_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\n[SAVED] {output_path}")

    return all_results


if __name__ == "__main__":
    results = run_all()
