"""
Synthetic Data Generator for FK-Causal-UQ Validation
=====================================================

Generate data where:
1. True causal graph is known
2. True interventional effects are computable
3. Can compare: LOO vs Causal vs Ground Truth

Key design:
- Simulate FK-like structure with known causal effects
- Each "FK" has known contribution to target uncertainty
- Ground truth is analytically computable
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import entropy
from dataclasses import dataclass


@dataclass
class CausalFK:
    """A synthetic FK relationship with known causal effect on uncertainty."""
    name: str
    n_categories: int
    causal_effect: float  # True effect on uncertainty (positive = reduces uncertainty)
    is_confounder: bool = False  # If True, affects both other FKs and target


def generate_synthetic_data(
    n_samples: int = 5000,
    n_classes: int = 4,
    fk_specs: List[CausalFK] = None,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, float], str]:
    """
    Generate synthetic data with known causal FK structure.

    Returns:
        - DataFrame with FK columns and target
        - Ground truth causal effects {fk_name: true_effect}
        - Target column name
    """
    np.random.seed(seed)

    if fk_specs is None:
        # Default: 3 FKs with different causal effects
        fk_specs = [
            CausalFK("FK_STRONG", n_categories=10, causal_effect=0.3),
            CausalFK("FK_MEDIUM", n_categories=5, causal_effect=0.1),
            CausalFK("FK_WEAK", n_categories=3, causal_effect=0.02),
            CausalFK("FK_CONFOUNDER", n_categories=8, causal_effect=0.15, is_confounder=True),
        ]

    # Initialize data
    data = {}
    ground_truth = {}

    # Generate confounder first (affects other FKs)
    confounder_values = {}
    for fk in fk_specs:
        if fk.is_confounder:
            confounder_values[fk.name] = np.random.randint(0, fk.n_categories, n_samples)
            data[fk.name] = confounder_values[fk.name]
            ground_truth[fk.name] = fk.causal_effect

    # Generate other FKs (some correlated with confounder)
    for fk in fk_specs:
        if not fk.is_confounder:
            if confounder_values and np.random.random() > 0.5:
                # Add correlation with confounder
                conf_name = list(confounder_values.keys())[0]
                conf_vals = confounder_values[conf_name]
                # FK values influenced by confounder
                base = np.random.randint(0, fk.n_categories, n_samples)
                noise = (conf_vals % fk.n_categories)
                data[fk.name] = (base + noise) % fk.n_categories
            else:
                data[fk.name] = np.random.randint(0, fk.n_categories, n_samples)

            ground_truth[fk.name] = fk.causal_effect

    # Generate target with known causal structure
    # P(Y | FK1, FK2, ...) depends on causal effects
    logits = np.zeros((n_samples, n_classes))

    for fk in fk_specs:
        fk_values = data[fk.name]
        # Causal effect: how much this FK determines the class
        # Higher causal_effect = more deterministic relationship
        effect_matrix = np.random.randn(fk.n_categories, n_classes) * fk.causal_effect
        logits += effect_matrix[fk_values]

    # Add noise
    logits += np.random.randn(n_samples, n_classes) * 0.5

    # Convert to probabilities
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # Sample target
    target = np.array([np.random.choice(n_classes, p=p) for p in probs])
    data['target'] = target

    # Add non-causal noise columns
    data['NOISE_1'] = np.random.randint(0, 20, n_samples)
    data['NOISE_2'] = np.random.randn(n_samples)
    ground_truth['NOISE_1'] = 0.0
    ground_truth['NOISE_2'] = 0.0

    df = pd.DataFrame(data)

    return df, ground_truth, 'target'


def compute_true_interventional_uq(
    df: pd.DataFrame,
    fk_name: str,
    target_col: str,
    fk_columns: List[str],
    n_interventions: int = 50
) -> float:
    """
    Compute true interventional UQ by actually intervening.

    do(FK = x): Replace FK values with intervention, retrain, measure uncertainty.

    This is the ground truth for comparison.
    """
    from sklearn.ensemble import RandomForestClassifier

    X = df[fk_columns]
    y = df[target_col]

    # Train model on original data
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    proba_original = model.predict_proba(X)
    entropy_original = np.mean(entropy(proba_original, axis=1))

    # Intervene: randomize FK values (breaks causal relationship)
    n_unique = df[fk_name].nunique()

    intervention_entropies = []
    for seed in range(n_interventions):
        np.random.seed(seed)
        df_intervened = df.copy()
        df_intervened[fk_name] = np.random.randint(0, n_unique, len(df))

        X_intervened = df_intervened[fk_columns]

        # Retrain model on intervened data (true intervention)
        model_int = RandomForestClassifier(n_estimators=50, random_state=42)
        model_int.fit(X_intervened, y)
        proba_int = model_int.predict_proba(X_intervened)
        entropy_int = np.mean(entropy(proba_int, axis=1))
        intervention_entropies.append(entropy_int)

    avg_intervention_entropy = np.mean(intervention_entropies)

    # Causal UQ = change in entropy under intervention
    # Positive = original has less uncertainty (FK helps)
    return avg_intervention_entropy - entropy_original


def compute_empirical_entropy(df: pd.DataFrame, target_col: str) -> float:
    """Compute empirical entropy of target distribution."""
    counts = df[target_col].value_counts(normalize=True)
    return entropy(counts)


def compute_loo_attribution_synthetic(
    df: pd.DataFrame,
    fk_columns: List[str],
    target_col: str
) -> Dict[str, float]:
    """
    Compute LOO attribution on synthetic data.
    """
    from sklearn.ensemble import RandomForestClassifier
    from scipy.stats import entropy

    X = df[fk_columns]
    y = df[target_col]

    # Full model
    model_full = RandomForestClassifier(n_estimators=50, random_state=42)
    model_full.fit(X, y)
    proba_full = model_full.predict_proba(X)
    entropy_full = np.mean(entropy(proba_full, axis=1))

    attributions = {}
    for fk in fk_columns:
        # Remove this FK
        remaining = [c for c in fk_columns if c != fk]
        if not remaining:
            attributions[fk] = 0.0
            continue

        model_loo = RandomForestClassifier(n_estimators=50, random_state=42)
        model_loo.fit(X[remaining], y)
        proba_loo = model_loo.predict_proba(X[remaining])
        entropy_loo = np.mean(entropy(proba_loo, axis=1))

        attributions[fk] = entropy_loo - entropy_full

    return attributions


def run_synthetic_validation():
    """
    Main validation: Compare methods against ground truth.
    """
    print("="*60)
    print("Synthetic Data Validation: FK-Causal-UQ")
    print("="*60)

    # Generate data with known structure
    fk_specs = [
        CausalFK("FK_STRONG", n_categories=10, causal_effect=0.5),
        CausalFK("FK_MEDIUM", n_categories=8, causal_effect=0.2),
        CausalFK("FK_WEAK", n_categories=5, causal_effect=0.05),
        CausalFK("FK_NOISE", n_categories=3, causal_effect=0.0),  # No effect
        CausalFK("FK_CONFOUNDER", n_categories=6, causal_effect=0.3, is_confounder=True),
    ]

    df, ground_truth, target_col = generate_synthetic_data(
        n_samples=5000,
        n_classes=4,
        fk_specs=fk_specs,
        seed=42
    )

    fk_columns = [fk.name for fk in fk_specs] + ['NOISE_1', 'NOISE_2']

    print(f"\nGenerated {len(df)} samples with {len(fk_columns)} FK columns")
    print(f"Target classes: {df[target_col].nunique()}")

    # Ground Truth (known causal effects)
    print("\n--- Ground Truth Causal Effects ---")
    for fk, effect in sorted(ground_truth.items(), key=lambda x: -x[1]):
        print(f"  {fk}: {effect:.3f}")

    # LOO Attribution
    print("\n--- LOO Attribution (Observational) ---")
    loo_attr = compute_loo_attribution_synthetic(df, fk_columns, target_col)
    for fk, attr in sorted(loo_attr.items(), key=lambda x: -x[1]):
        print(f"  {fk}: {attr:.4f}")

    # True Interventional (Ground Truth Validation)
    print("\n--- True Interventional UQ ---")
    interventional_attr = {}
    for fk in fk_columns:
        true_effect = compute_true_interventional_uq(
            df, fk, target_col, fk_columns, n_interventions=20
        )
        interventional_attr[fk] = true_effect
        print(f"  {fk}: {true_effect:.4f}")

    # Compare
    print("\n--- Comparison ---")
    from scipy.stats import spearmanr

    common_keys = list(ground_truth.keys())
    gt_vals = [ground_truth[k] for k in common_keys]

    # LOO vs Ground Truth
    loo_vals = [loo_attr.get(k, 0) for k in common_keys]
    rho_loo, p_loo = spearmanr(gt_vals, loo_vals)
    print(f"LOO vs Ground Truth: ρ = {rho_loo:.3f}, p = {p_loo:.4f}")

    # Interventional vs Ground Truth
    int_vals = [interventional_attr.get(k, 0) for k in common_keys]
    rho_int, p_int = spearmanr(gt_vals, int_vals)
    print(f"Interventional vs Ground Truth: ρ = {rho_int:.3f}, p = {p_int:.4f}")

    # LOO vs Interventional
    rho_loo_int, p_loo_int = spearmanr(loo_vals, int_vals)
    print(f"LOO vs Interventional: ρ = {rho_loo_int:.3f}, p = {p_loo_int:.4f}")

    print("\n--- Summary ---")
    if rho_int > rho_loo:
        print("✓ Interventional method closer to ground truth than LOO")
    else:
        print("✗ LOO method closer to ground truth (unexpected)")

    if abs(rho_loo_int) < 0.5:
        print("✓ LOO ≠ Interventional (as theoretically expected)")
    else:
        print("  LOO ≈ Interventional (methods agree)")

    return {
        'ground_truth': ground_truth,
        'loo': loo_attr,
        'interventional': interventional_attr,
        'rho_loo_gt': rho_loo,
        'rho_int_gt': rho_int,
        'rho_loo_int': rho_loo_int
    }


if __name__ == '__main__':
    results = run_synthetic_validation()

    # Save results
    import json
    from pathlib import Path

    output_path = Path('chorok/results/synthetic_validation.json')
    output_path.parent.mkdir(exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")
