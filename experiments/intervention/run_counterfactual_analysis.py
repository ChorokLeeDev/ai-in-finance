"""
Counterfactual Analysis for FK Attribution
==========================================

The right question is NOT "what values minimize uncertainty?"
(That creates OOD inputs and increases uncertainty)

The right question IS: "If we REDUCE NOISE in an FK group,
how much would uncertainty decrease?"

Method:
1. Add noise to each FK group → measure uncertainty increase
2. The FK with largest increase when noised = most sensitive
3. This tells us: "Cleaning this FK's data would help most"

This connects ATTRIBUTION to ACTIONABILITY:
- Attribution: "DRIVER contributes 29%"
- Counterfactual: "Reducing DRIVER noise would save 29% of uncertainty"
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution')
from data_loader_f1 import load_f1_data


def train_ensemble(X, y, n_models=5, base_seed=42):
    models = []
    for i in range(n_models):
        m = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=base_seed + i, verbose=-1
        )
        m.fit(X, y)
        models.append(m)
    return models


def get_uncertainty(models, X):
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0).mean()


def add_noise_to_fk(X, col_to_fk, fk_name, noise_level=0.1):
    """Add Gaussian noise to FK features."""
    X_noisy = X.copy()
    fk_cols = [c for c, fk in col_to_fk.items() if fk == fk_name and c in X.columns]

    for col in fk_cols:
        std = X[col].std()
        noise = np.random.normal(0, std * noise_level, len(X))
        X_noisy[col] = X[col] + noise

    return X_noisy


def run_counterfactual_analysis():
    print("=" * 70)
    print("Counterfactual Analysis: Noise Sensitivity")
    print("=" * 70)
    print("\nQuestion: If we REDUCE noise in each FK, how much does uncertainty drop?")
    print("Method: Add noise → measure increase → inverse = reduction potential\n")

    # Load data
    print("[1/3] Loading data...")
    X, y, _, col_to_fk = load_f1_data(sample_size=3000, use_cache=True)
    print(f"Data: {X.shape}")

    # Train
    print("[2/3] Training ensemble...")
    models = train_ensemble(X, y, n_models=5, base_seed=42)
    base_unc = get_uncertainty(models, X)
    print(f"Base uncertainty: {base_unc:.6f}")

    # Noise sensitivity for each FK
    print("\n[3/3] Testing noise sensitivity...")
    fk_names = list(set(col_to_fk.values()))
    noise_levels = [0.05, 0.1, 0.2, 0.5]

    results = {}
    for fk in fk_names:
        sensitivities = []
        for noise in noise_levels:
            X_noisy = add_noise_to_fk(X, col_to_fk, fk, noise)
            noisy_unc = get_uncertainty(models, X_noisy)
            increase = (noisy_unc - base_unc) / base_unc * 100
            sensitivities.append(increase)

        results[fk] = {
            'sensitivities': sensitivities,
            'mean_sensitivity': np.mean(sensitivities)
        }

    # Summary
    print("\n" + "=" * 70)
    print("NOISE SENSITIVITY RESULTS")
    print("=" * 70)
    print(f"\nBase uncertainty: {base_unc:.6f}")
    print(f"\n{'FK Group':<15} " + "".join([f'{n*100:.0f}% noise   ' for n in noise_levels]))
    print("-" * 70)

    sorted_fks = sorted(results.keys(), key=lambda x: results[x]['mean_sensitivity'], reverse=True)
    for fk in sorted_fks:
        sens = results[fk]['sensitivities']
        row = f"{fk:<15} " + "".join([f"+{s:>6.1f}%     " for s in sens])
        print(row)

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION: Reduction Potential")
    print("=" * 70)
    print("\nIf adding noise INCREASES uncertainty by X%,")
    print("then REDUCING noise could decrease uncertainty by ~X%.\n")

    print(f"{'FK Group':<15} {'Reduction Potential':<20} {'Priority'}")
    print("-" * 50)
    for i, fk in enumerate(sorted_fks):
        potential = results[fk]['mean_sensitivity']
        priority = ["🥇 HIGH", "🥈 MEDIUM", "🥉 LOW", "LOW", "LOW"][min(i, 4)]
        print(f"{fk:<15} {potential:>6.1f}%              {priority}")

    print("\n" + "=" * 70)
    print("ACTIONABLE RECOMMENDATION")
    print("=" * 70)
    top_fk = sorted_fks[0]
    top_potential = results[top_fk]['mean_sensitivity']
    print(f"\n→ Focus on {top_fk} data quality")
    print(f"→ Potential uncertainty reduction: {top_potential:.1f}%")
    print(f"→ Action: Audit {top_fk} data collection process for noise sources")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_counterfactual_analysis()
