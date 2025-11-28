"""
Semi-Synthetic Validation: Inject Known Shifts into Real FK Structure
======================================================================

Validation strategy:
1. Load real SALT data with FK structure
2. Inject KNOWN distribution shift into ONE specific FK
3. Train model on original data, test on shifted data
4. Check: Does our method correctly identify the shifted FK?

This provides ground truth on real data structure (not purely synthetic).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import entropy, spearmanr

from relbench.datasets import get_dataset
from relbench.tasks import get_task

warnings.filterwarnings('ignore')


def load_salt_data(task_name: str = 'sales-group', sample_size: int = 5000):
    """Load SALT dataset."""
    task = get_task("rel-salt", task_name, download=False)
    dataset = get_dataset("rel-salt", download=False)
    db = dataset.get_db(upto_test_timestamp=False)

    entity_table = db.table_dict[task.entity_table]
    df = entity_table.df.copy()

    # Sample
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    target_col = task.target_col

    # Prepare features
    exclude = {'CREATIONTIMESTAMP', target_col, entity_table.pkey_col}
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category').cat.codes
        X[col] = X[col].fillna(-999)

    y = df[target_col].copy()
    if y.dtype == 'object':
        y = y.astype('category').cat.codes

    return X, y, feature_cols


def inject_shift(X: pd.DataFrame, shifted_col: str, shift_type: str = 'permute') -> pd.DataFrame:
    """
    Inject distribution shift into a specific column.

    shift_type:
    - 'permute': Randomly permute values (breaks correlation)
    - 'noise': Add random noise
    - 'constant': Replace with constant value
    - 'category_swap': Swap category labels
    """
    X_shifted = X.copy()

    if shift_type == 'permute':
        # Random permutation - breaks all correlations
        X_shifted[shifted_col] = np.random.permutation(X_shifted[shifted_col].values)

    elif shift_type == 'noise':
        # Add Gaussian noise (for numeric columns)
        std = X_shifted[shifted_col].std()
        X_shifted[shifted_col] = X_shifted[shifted_col] + np.random.randn(len(X_shifted)) * std

    elif shift_type == 'constant':
        # Replace with most common value (extreme shift)
        mode_val = X_shifted[shifted_col].mode()[0]
        X_shifted[shifted_col] = mode_val

    elif shift_type == 'category_swap':
        # Swap category codes
        unique_vals = X_shifted[shifted_col].unique()
        if len(unique_vals) > 1:
            swap_map = dict(zip(unique_vals, np.roll(unique_vals, 1)))
            X_shifted[shifted_col] = X_shifted[shifted_col].map(swap_map)

    return X_shifted


def compute_loo_attribution(model, X_train, y_train, X_test, feature_cols: List[str]) -> Dict[str, float]:
    """Compute LOO attribution on test data using model trained on train data."""
    # Get base uncertainty on test data
    proba_base = model.predict_proba(X_test)
    entropy_base = np.mean(entropy(proba_base, axis=1))

    attributions = {}
    for col in feature_cols:
        # Remove this column and retrain
        remaining = [c for c in feature_cols if c != col]
        if not remaining:
            attributions[col] = 0.0
            continue

        model_loo = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        model_loo.fit(X_train[remaining], y_train)

        proba_loo = model_loo.predict_proba(X_test[remaining])
        entropy_loo = np.mean(entropy(proba_loo, axis=1))

        # Attribution: how much uncertainty increases without this feature
        attributions[col] = entropy_loo - entropy_base

    return attributions


def compute_causal_attribution(model, X_test, feature_cols: List[str]) -> Dict[str, float]:
    """
    Compute causal attribution via interventional approach.

    For each feature, intervene (randomize) and measure uncertainty change.
    This doesn't require retraining - uses same model.
    """
    proba_base = model.predict_proba(X_test)
    entropy_base = np.mean(entropy(proba_base, axis=1))

    attributions = {}
    n_interventions = 20

    for col in feature_cols:
        intervention_entropies = []

        for _ in range(n_interventions):
            # Intervene: randomize this column
            X_intervened = X_test.copy()
            X_intervened[col] = np.random.permutation(X_intervened[col].values)

            proba_int = model.predict_proba(X_intervened)
            entropy_int = np.mean(entropy(proba_int, axis=1))
            intervention_entropies.append(entropy_int)

        avg_entropy = np.mean(intervention_entropies)
        # Positive = intervention increases uncertainty (feature helps)
        attributions[col] = avg_entropy - entropy_base

    return attributions


def run_semi_synthetic_experiment(
    task_name: str = 'sales-group',
    shifted_col: str = 'CUSTOMERPAYMENTTERMS',
    shift_type: str = 'permute',
    sample_size: int = 3000
):
    """
    Run semi-synthetic validation experiment.

    1. Load real data
    2. Split into train/test
    3. Inject shift into test set for ONE column
    4. Train on original train, test on shifted test
    5. Check if method identifies the shifted column
    """
    print("="*60)
    print(f"Semi-Synthetic Validation: {task_name}")
    print(f"Shifted column: {shifted_col}")
    print(f"Shift type: {shift_type}")
    print("="*60)

    # Load data
    X, y, feature_cols = load_salt_data(task_name, sample_size)
    print(f"Loaded {len(X)} samples, {len(feature_cols)} features")

    if shifted_col not in feature_cols:
        print(f"ERROR: {shifted_col} not in features. Available: {feature_cols[:5]}...")
        return None

    # Split train/test
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train model on original data
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    # Inject shift into TEST set only
    X_test_shifted = inject_shift(X_test, shifted_col, shift_type)

    # Ground truth: shifted_col should have highest attribution
    ground_truth = {col: (1.0 if col == shifted_col else 0.0) for col in feature_cols}

    # =================================================================
    # Method 1: LOO Attribution (Correlational)
    # =================================================================
    print("\n--- LOO Attribution (Correlational) ---")
    loo_original = compute_loo_attribution(model, X_train, y_train, X_test, feature_cols)
    loo_shifted = compute_loo_attribution(model, X_train, y_train, X_test_shifted, feature_cols)

    # Change in attribution
    loo_delta = {col: loo_shifted[col] - loo_original[col] for col in feature_cols}

    print("Top 5 by |delta|:")
    for col, delta in sorted(loo_delta.items(), key=lambda x: -abs(x[1]))[:5]:
        marker = " ← SHIFTED" if col == shifted_col else ""
        print(f"  {col}: {delta:+.4f}{marker}")

    # Did LOO identify the shifted column?
    loo_top = max(loo_delta.items(), key=lambda x: abs(x[1]))[0]
    loo_correct = (loo_top == shifted_col)
    print(f"LOO identified: {loo_top} ({'CORRECT' if loo_correct else 'WRONG'})")

    # =================================================================
    # Method 2: Causal Attribution (Interventional)
    # =================================================================
    print("\n--- Causal Attribution (Interventional) ---")
    causal_original = compute_causal_attribution(model, X_test, feature_cols)
    causal_shifted = compute_causal_attribution(model, X_test_shifted, feature_cols)

    # Change in attribution
    causal_delta = {col: causal_shifted[col] - causal_original[col] for col in feature_cols}

    print("Top 5 by |delta|:")
    for col, delta in sorted(causal_delta.items(), key=lambda x: -abs(x[1]))[:5]:
        marker = " ← SHIFTED" if col == shifted_col else ""
        print(f"  {col}: {delta:+.4f}{marker}")

    # Did Causal identify the shifted column?
    causal_top = max(causal_delta.items(), key=lambda x: abs(x[1]))[0]
    causal_correct = (causal_top == shifted_col)
    print(f"Causal identified: {causal_top} ({'CORRECT' if causal_correct else 'WRONG'})")

    # =================================================================
    # Rank Correlation with Ground Truth
    # =================================================================
    print("\n--- Correlation with Ground Truth ---")

    gt_vals = [ground_truth[col] for col in feature_cols]
    loo_vals = [abs(loo_delta[col]) for col in feature_cols]
    causal_vals = [abs(causal_delta[col]) for col in feature_cols]

    rho_loo, p_loo = spearmanr(gt_vals, loo_vals)
    rho_causal, p_causal = spearmanr(gt_vals, causal_vals)

    print(f"LOO vs Ground Truth: ρ = {rho_loo:.3f}, p = {p_loo:.4f}")
    print(f"Causal vs Ground Truth: ρ = {rho_causal:.3f}, p = {p_causal:.4f}")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Ground truth shifted column: {shifted_col}")
    print(f"LOO identified: {loo_top} ({'✓' if loo_correct else '✗'})")
    print(f"Causal identified: {causal_top} ({'✓' if causal_correct else '✗'})")

    if causal_correct and not loo_correct:
        print("\n→ CAUSAL method correctly identified shift, LOO failed!")
    elif causal_correct and loo_correct:
        print("\n→ Both methods correctly identified the shift")
    elif not causal_correct and loo_correct:
        print("\n→ LOO correct, CAUSAL failed (unexpected)")
    else:
        print("\n→ Neither method identified the shift correctly")

    results = {
        'task': task_name,
        'shifted_col': shifted_col,
        'shift_type': shift_type,
        'loo_top': loo_top,
        'loo_correct': loo_correct,
        'causal_top': causal_top,
        'causal_correct': causal_correct,
        'rho_loo': float(rho_loo),
        'rho_causal': float(rho_causal),
        'loo_delta': {k: float(v) for k, v in loo_delta.items()},
        'causal_delta': {k: float(v) for k, v in causal_delta.items()}
    }

    return results


def run_multiple_experiments():
    """Run experiments with different shifted columns."""

    # Columns to test
    test_columns = [
        'CUSTOMERPAYMENTTERMS',
        'SALESORGANIZATION',
        'TRANSACTIONCURRENCY',
        'DISTRIBUTIONCHANNEL',
        'HEADERINCOTERMSCLASSIFICATION'
    ]

    all_results = []

    for col in test_columns:
        print(f"\n{'#'*60}")
        print(f"# Testing shift in: {col}")
        print(f"{'#'*60}")

        result = run_semi_synthetic_experiment(
            task_name='sales-group',
            shifted_col=col,
            shift_type='permute',
            sample_size=3000
        )

        if result:
            all_results.append(result)

    # Summary table
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    print(f"{'Shifted Column':<30} {'LOO':<10} {'Causal':<10}")
    print("-"*60)

    loo_correct_count = 0
    causal_correct_count = 0

    for r in all_results:
        loo_mark = '✓' if r['loo_correct'] else '✗'
        causal_mark = '✓' if r['causal_correct'] else '✗'
        print(f"{r['shifted_col']:<30} {loo_mark:<10} {causal_mark:<10}")
        loo_correct_count += r['loo_correct']
        causal_correct_count += r['causal_correct']

    print("-"*60)
    print(f"{'Accuracy':<30} {loo_correct_count}/{len(all_results):<10} {causal_correct_count}/{len(all_results):<10}")

    # Save results
    output_path = Path('chorok/results/semi_synthetic_validation.json')
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='sales-group')
    parser.add_argument('--shifted_col', type=str, default='CUSTOMERPAYMENTTERMS')
    parser.add_argument('--shift_type', type=str, default='permute',
                        choices=['permute', 'noise', 'constant', 'category_swap'])
    parser.add_argument('--sample_size', type=int, default=3000)
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    args = parser.parse_args()

    if args.all:
        run_multiple_experiments()
    else:
        run_semi_synthetic_experiment(
            task_name=args.task,
            shifted_col=args.shifted_col,
            shift_type=args.shift_type,
            sample_size=args.sample_size
        )


if __name__ == '__main__':
    main()
