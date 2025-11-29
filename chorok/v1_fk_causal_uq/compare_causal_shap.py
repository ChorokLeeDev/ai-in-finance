"""
Compare Our FK-Causal-UQ with Causal SHAP Applied to Uncertainty
================================================================

Key Question: If we apply Causal SHAP (Heskes 2020) to uncertainty U(f(x)),
does it perform as well as our FK-Causal-UQ method?

This is the critical comparison that determines our novelty:
- If Causal SHAP on U(f(x)) ≈ our method → weak novelty (just applying existing)
- If our method > Causal SHAP on U(f(x)) → strong novelty (FK structure helps)

Methods compared:
1. Standard SHAP on Uncertainty (observational, independence assumption)
2. Causal SHAP on Uncertainty (interventional, no FK structure)
3. Our FK-Causal-UQ (interventional, uses FK structure as DAG)
4. True Interventional (ground truth, requires retraining)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import entropy, spearmanr
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CausalFK:
    """A synthetic FK with known causal effect on uncertainty."""
    name: str
    n_categories: int
    causal_effect: float
    parent: str = None  # FK structure: which FK is parent of this one


def generate_fk_structured_data(
    n_samples: int = 5000,
    n_classes: int = 4,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, float], List[str], Dict]:
    """
    Generate data with EXPLICIT FK structure (DAG).

    DAG structure:
        FK_ROOT (no parent)
           ↓
        FK_CHILD1 (parent: FK_ROOT)
           ↓
        FK_GRANDCHILD (parent: FK_CHILD1)

        FK_INDEPENDENT (no parent, uncorrelated)

    This tests whether FK structure helps in attribution.
    """
    np.random.seed(seed)

    # Define FK hierarchy
    fk_specs = [
        CausalFK("FK_ROOT", n_categories=10, causal_effect=0.4, parent=None),
        CausalFK("FK_CHILD1", n_categories=8, causal_effect=0.25, parent="FK_ROOT"),
        CausalFK("FK_GRANDCHILD", n_categories=6, causal_effect=0.1, parent="FK_CHILD1"),
        CausalFK("FK_INDEPENDENT", n_categories=5, causal_effect=0.15, parent=None),
        CausalFK("FK_NOISE", n_categories=4, causal_effect=0.0, parent=None),
    ]

    # Build DAG structure info
    dag_structure = {}
    for fk in fk_specs:
        dag_structure[fk.name] = {
            'parent': fk.parent,
            'children': [f.name for f in fk_specs if f.parent == fk.name]
        }

    data = {}
    ground_truth = {}

    # Generate in topological order (parents first)
    # FK_ROOT
    data["FK_ROOT"] = np.random.randint(0, 10, n_samples)
    ground_truth["FK_ROOT"] = 0.4

    # FK_CHILD1 (depends on FK_ROOT)
    base = np.random.randint(0, 8, n_samples)
    parent_effect = (data["FK_ROOT"] % 8)
    data["FK_CHILD1"] = (base + parent_effect) % 8
    ground_truth["FK_CHILD1"] = 0.25

    # FK_GRANDCHILD (depends on FK_CHILD1)
    base = np.random.randint(0, 6, n_samples)
    parent_effect = (data["FK_CHILD1"] % 6)
    data["FK_GRANDCHILD"] = (base + parent_effect) % 6
    ground_truth["FK_GRANDCHILD"] = 0.1

    # FK_INDEPENDENT (no parent)
    data["FK_INDEPENDENT"] = np.random.randint(0, 5, n_samples)
    ground_truth["FK_INDEPENDENT"] = 0.15

    # FK_NOISE (no effect)
    data["FK_NOISE"] = np.random.randint(0, 4, n_samples)
    ground_truth["FK_NOISE"] = 0.0

    # Generate target based on causal effects
    logits = np.zeros((n_samples, n_classes))
    for fk in fk_specs:
        effect_matrix = np.random.randn(fk.n_categories, n_classes) * fk.causal_effect
        logits += effect_matrix[data[fk.name]]

    logits += np.random.randn(n_samples, n_classes) * 0.3
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    target = np.array([np.random.choice(n_classes, p=p) for p in probs])

    data['target'] = target
    df = pd.DataFrame(data)
    fk_columns = [fk.name for fk in fk_specs]

    return df, ground_truth, fk_columns, dag_structure


def compute_uncertainty(proba: np.ndarray) -> float:
    """Compute mean entropy as uncertainty measure."""
    return np.mean(entropy(proba + 1e-10, axis=1))


def method_1_standard_shap_uq(
    df: pd.DataFrame,
    fk_columns: List[str],
    target_col: str
) -> Dict[str, float]:
    """
    Standard SHAP on uncertainty (observational, independence assumption).
    This is what regular SHAP does - ignores causal structure.
    """
    X = df[fk_columns]
    y = df[target_col]

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    proba = model.predict_proba(X)
    entropy_full = compute_uncertainty(proba)

    # Approximate SHAP via marginal contribution (permutation)
    attributions = {}
    n_perm = 20

    for fk in fk_columns:
        entropy_diff = 0
        for _ in range(n_perm):
            X_perm = X.copy()
            X_perm[fk] = np.random.permutation(X_perm[fk].values)
            proba_perm = model.predict_proba(X_perm)
            entropy_perm = compute_uncertainty(proba_perm)
            entropy_diff += (entropy_perm - entropy_full)
        attributions[fk] = entropy_diff / n_perm

    return attributions


def method_2_causal_shap_uq(
    df: pd.DataFrame,
    fk_columns: List[str],
    target_col: str,
    dag_structure: Dict
) -> Dict[str, float]:
    """
    Causal SHAP on uncertainty (interventional, uses DAG).

    Key difference from standard SHAP: respects causal structure.
    When intervening on X_i, don't break correlations with parents,
    only break correlations with descendants.

    This is what Heskes 2020 does for predictions - we apply it to uncertainty.
    """
    X = df[fk_columns]
    y = df[target_col]

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    proba = model.predict_proba(X)
    entropy_full = compute_uncertainty(proba)

    attributions = {}
    n_interventions = 20

    for fk in fk_columns:
        # Get DAG info
        fk_info = dag_structure.get(fk, {'parent': None, 'children': []})
        parent = fk_info['parent']

        entropy_diffs = []
        for seed in range(n_interventions):
            np.random.seed(seed)
            X_int = X.copy()

            if parent is not None:
                # Interventional: sample conditioned on parent (causal)
                # Group by parent value and sample within groups
                parent_vals = X_int[parent].values
                unique_parents = np.unique(parent_vals)

                new_vals = np.zeros(len(X_int), dtype=int)
                for pv in unique_parents:
                    mask = (parent_vals == pv)
                    # Sample from conditional distribution P(FK | parent=pv)
                    pool = X_int.loc[mask, fk].values
                    if len(pool) > 0:
                        new_vals[mask] = np.random.choice(pool, size=mask.sum(), replace=True)
                X_int[fk] = new_vals
            else:
                # No parent: just permute (same as observational)
                X_int[fk] = np.random.permutation(X_int[fk].values)

            proba_int = model.predict_proba(X_int)
            entropy_int = compute_uncertainty(proba_int)
            entropy_diffs.append(entropy_int - entropy_full)

        attributions[fk] = np.mean(entropy_diffs)

    return attributions


def method_3_fk_causal_uq(
    df: pd.DataFrame,
    fk_columns: List[str],
    target_col: str,
    dag_structure: Dict
) -> Dict[str, float]:
    """
    Our FK-Causal-UQ method (interventional, uses FK structure).

    FIXED implementation: Uses stratified backdoor adjustment.
    Only adjust for PARENTS (minimal adjustment set), not all non-descendants.
    """
    X = df[fk_columns]
    y = df[target_col]

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    proba = model.predict_proba(X)
    entropy_full = compute_uncertainty(proba)

    attributions = {}
    n_perm = 20

    for fk in fk_columns:
        fk_info = dag_structure.get(fk, {'parent': None, 'children': []})
        parent = fk_info['parent']

        if parent is None:
            # No parent: simple permutation (same as standard SHAP)
            entropy_diffs = []
            for seed in range(n_perm):
                np.random.seed(seed)
                X_int = X.copy()
                X_int[fk] = np.random.permutation(X_int[fk].values)
                proba_int = model.predict_proba(X_int)
                entropy_diffs.append(compute_uncertainty(proba_int) - entropy_full)
            attributions[fk] = np.mean(entropy_diffs)
        else:
            # Has parent: stratified intervention (backdoor adjustment)
            # do(FK=x) while preserving correlation with parent
            entropy_diffs = []
            for seed in range(n_perm):
                np.random.seed(seed)
                X_int = X.copy()

                # Stratified permutation: permute FK WITHIN each parent stratum
                parent_vals = X_int[parent].values
                fk_vals = X_int[fk].values.copy()

                for pv in np.unique(parent_vals):
                    mask = (parent_vals == pv)
                    indices = np.where(mask)[0]
                    # Permute FK values within this parent stratum
                    fk_vals[indices] = np.random.permutation(fk_vals[indices])

                X_int[fk] = fk_vals
                proba_int = model.predict_proba(X_int)
                entropy_diffs.append(compute_uncertainty(proba_int) - entropy_full)

            attributions[fk] = np.mean(entropy_diffs)

    return attributions


def method_4_true_interventional(
    df: pd.DataFrame,
    fk_columns: List[str],
    target_col: str,
    n_interventions: int = 10
) -> Dict[str, float]:
    """
    True interventional (ground truth) - requires retraining.
    This is the gold standard but computationally expensive.
    """
    X = df[fk_columns]
    y = df[target_col]

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    proba = model.predict_proba(X)
    entropy_full = compute_uncertainty(proba)

    attributions = {}

    for fk in fk_columns:
        n_unique = df[fk].nunique()
        intervention_entropies = []

        for seed in range(n_interventions):
            np.random.seed(seed)
            df_int = df.copy()
            df_int[fk] = np.random.randint(0, n_unique, len(df))

            X_int = df_int[fk_columns]
            # Retrain on intervened data (true intervention)
            model_int = RandomForestClassifier(n_estimators=50, random_state=42)
            model_int.fit(X_int, y)
            proba_int = model_int.predict_proba(X_int)
            intervention_entropies.append(compute_uncertainty(proba_int))

        attributions[fk] = np.mean(intervention_entropies) - entropy_full

    return attributions


def run_comparison():
    """Run the critical comparison of all methods."""
    print("="*70)
    print("CRITICAL COMPARISON: Causal SHAP vs FK-Causal-UQ")
    print("="*70)

    # Generate FK-structured data
    df, ground_truth, fk_columns, dag_structure = generate_fk_structured_data(
        n_samples=3000,
        seed=42
    )

    print(f"\nGenerated {len(df)} samples with FK DAG structure:")
    print("  FK_ROOT → FK_CHILD1 → FK_GRANDCHILD")
    print("  FK_INDEPENDENT (no parents)")
    print("  FK_NOISE (no effect)")

    print("\n--- Ground Truth Causal Effects ---")
    for fk in fk_columns:
        print(f"  {fk}: {ground_truth[fk]:.3f}")

    # Run all methods
    print("\n--- Method 1: Standard SHAP on Uncertainty ---")
    shap_attr = method_1_standard_shap_uq(df, fk_columns, 'target')
    for fk in fk_columns:
        print(f"  {fk}: {shap_attr[fk]:.4f}")

    print("\n--- Method 2: Causal SHAP on Uncertainty ---")
    causal_shap_attr = method_2_causal_shap_uq(df, fk_columns, 'target', dag_structure)
    for fk in fk_columns:
        print(f"  {fk}: {causal_shap_attr[fk]:.4f}")

    print("\n--- Method 3: FK-Causal-UQ (Ours) ---")
    fk_uq_attr = method_3_fk_causal_uq(df, fk_columns, 'target', dag_structure)
    for fk in fk_columns:
        print(f"  {fk}: {fk_uq_attr[fk]:.4f}")

    print("\n--- Method 4: True Interventional (Ground Truth Validation) ---")
    true_attr = method_4_true_interventional(df, fk_columns, 'target', n_interventions=10)
    for fk in fk_columns:
        print(f"  {fk}: {true_attr[fk]:.4f}")

    # Compute correlations with ground truth
    print("\n" + "="*70)
    print("CORRELATION WITH GROUND TRUTH")
    print("="*70)

    gt_vals = [ground_truth[fk] for fk in fk_columns]

    methods = {
        'Standard SHAP': shap_attr,
        'Causal SHAP': causal_shap_attr,
        'FK-Causal-UQ (Ours)': fk_uq_attr,
        'True Interventional': true_attr
    }

    results = {}
    for method_name, attr in methods.items():
        method_vals = [attr[fk] for fk in fk_columns]
        rho, p = spearmanr(gt_vals, method_vals)
        results[method_name] = {'rho': rho, 'p': p}
        sig = "**" if p < 0.05 else ""
        print(f"  {method_name}: ρ = {rho:.3f}, p = {p:.4f} {sig}")

    # Key comparison: Causal SHAP vs FK-Causal-UQ
    print("\n" + "="*70)
    print("KEY COMPARISON: CAUSAL SHAP vs FK-CAUSAL-UQ")
    print("="*70)

    rho_causal_shap = results['Causal SHAP']['rho']
    rho_fk_uq = results['FK-Causal-UQ (Ours)']['rho']

    if rho_fk_uq > rho_causal_shap + 0.1:
        verdict = "STRONG NOVELTY: FK structure provides significant advantage"
        novelty = "strong"
    elif rho_fk_uq > rho_causal_shap:
        verdict = "MODERATE NOVELTY: FK structure provides some advantage"
        novelty = "moderate"
    elif abs(rho_fk_uq - rho_causal_shap) < 0.1:
        verdict = "WEAK NOVELTY: Methods perform similarly"
        novelty = "weak"
    else:
        verdict = "NO NOVELTY: Causal SHAP performs better"
        novelty = "none"

    print(f"\n  Causal SHAP:      ρ = {rho_causal_shap:.3f}")
    print(f"  FK-Causal-UQ:     ρ = {rho_fk_uq:.3f}")
    print(f"  Difference:       Δρ = {rho_fk_uq - rho_causal_shap:.3f}")
    print(f"\n  >>> {verdict} <<<")

    # Save results
    output = {
        'ground_truth': ground_truth,
        'attributions': {
            'standard_shap': {k: float(v) for k, v in shap_attr.items()},
            'causal_shap': {k: float(v) for k, v in causal_shap_attr.items()},
            'fk_causal_uq': {k: float(v) for k, v in fk_uq_attr.items()},
            'true_interventional': {k: float(v) for k, v in true_attr.items()}
        },
        'correlations': {k: {'rho': float(v['rho']), 'p': float(v['p'])}
                        for k, v in results.items()},
        'novelty_assessment': novelty,
        'dag_structure': dag_structure
    }

    output_path = Path('chorok/results/causal_shap_comparison.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    run_comparison()
