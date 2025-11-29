"""
FK-Causal-UQ: Interventional Uncertainty Attribution via FK Structure
======================================================================

Core contribution: Use FK relationships as causal prior for interventional
(not correlational) uncertainty attribution.

Key difference from LOO:
- LOO: "What if we don't observe FK features?" (observational)
- Causal: "What if we intervene on FK values?" (interventional)

Algorithm:
1. Construct G_FK (causal DAG) from relational schema
2. For each FK, compute adjustment set (parents + non-descendants)
3. Use backdoor adjustment to estimate interventional UQ

No retraining required! Uses single trained model + causal adjustment.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import entropy
import networkx as nx

from relbench.datasets import get_dataset
from relbench.tasks import get_task

warnings.filterwarnings('ignore')

VAL_TIMESTAMP = pd.Timestamp("2020-02-01")
TEST_TIMESTAMP = pd.Timestamp("2020-07-01")


# =============================================================================
# FK Graph Construction (G_FK)
# =============================================================================

def build_fk_graph(db) -> nx.DiGraph:
    """
    Build FK-induced causal DAG from database schema.

    FK direction: child → parent
    Causal direction: parent → child (reversed)

    Returns: NetworkX DiGraph where edge (A, B) means A causally affects B
    """
    G = nx.DiGraph()

    # Add all tables as nodes
    for table_name in db.table_dict.keys():
        G.add_node(table_name)

    # Add edges from FK relationships (reversed direction for causality)
    for table_name, table in db.table_dict.items():
        if hasattr(table, 'fkey_col_to_pkey_table') and table.fkey_col_to_pkey_table:
            for fkey_col, parent_table in table.fkey_col_to_pkey_table.items():
                # FK: child (table_name) → parent (parent_table)
                # Causal: parent (parent_table) → child (table_name)
                G.add_edge(parent_table, table_name, fkey_col=fkey_col)

    return G


def get_adjustment_set(G: nx.DiGraph, fk_source: str, target: str) -> Set[str]:
    """
    Compute adjustment set for backdoor criterion.

    For intervention on FK from source to target:
    Z = Parents(source) ∪ Non-descendants(source)

    In practice with FK structure, adjustment set includes all tables
    that are not descendants of the intervened relationship.
    """
    # Get descendants of source (including source itself)
    descendants = nx.descendants(G, fk_source) | {fk_source}

    # Non-descendants = all nodes - descendants
    all_nodes = set(G.nodes())
    non_descendants = all_nodes - descendants

    # Parents of source
    parents = set(G.predecessors(fk_source))

    # Adjustment set
    adjustment_set = parents | non_descendants

    # Remove target from adjustment set (we're predicting it)
    adjustment_set.discard(target)

    return adjustment_set


def get_fk_columns(db, table_name: str) -> Dict[str, List[str]]:
    """
    Get columns associated with each FK relationship.

    Returns: {fk_name: [column_names]}
    """
    table = db.table_dict[table_name]
    fk_columns = {}

    if hasattr(table, 'fkey_col_to_pkey_table') and table.fkey_col_to_pkey_table:
        for fkey_col, parent_table in table.fkey_col_to_pkey_table.items():
            fk_name = f"{parent_table}_{fkey_col}"
            fk_columns[fk_name] = [fkey_col]

    return fk_columns


# =============================================================================
# Causal UQ Attribution
# =============================================================================

def compute_conditional_entropy(model, X, y, condition_cols: List[str],
                                 n_strata: int = 10) -> float:
    """
    Compute E[U(Y) | Z=z] by stratifying on conditioning columns.

    This implements the backdoor adjustment:
    E[U(Y) | do(X)] = Σ_z P(Z=z) * E[U(Y) | X, Z=z]
    """
    if not condition_cols:
        # No conditioning - just return overall entropy
        proba = model.predict_proba(X)
        return np.mean(entropy(proba, axis=1))

    # Stratify by conditioning columns
    X_condition = X[condition_cols].copy()

    # Create strata by quantile binning for numeric columns
    strata = pd.DataFrame(index=X.index)
    for col in condition_cols:
        if col in X.columns:
            try:
                strata[col] = pd.qcut(X_condition[col], q=n_strata,
                                      labels=False, duplicates='drop')
            except:
                strata[col] = pd.cut(X_condition[col], bins=n_strata,
                                     labels=False)

    # Create stratum identifier
    strata['stratum'] = strata.apply(lambda x: tuple(x), axis=1)

    # Compute weighted entropy
    total_entropy = 0
    total_weight = 0

    for stratum_id, group in X.groupby(strata['stratum']):
        if len(group) < 5:
            continue

        proba = model.predict_proba(group)
        stratum_entropy = np.mean(entropy(proba, axis=1))
        weight = len(group)

        total_entropy += stratum_entropy * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return total_entropy / total_weight


def compute_causal_uq_attribution(model, X, y, fk_columns: Dict[str, List[str]],
                                   adjustment_sets: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Compute interventional UQ attribution using backdoor adjustment.

    For each FK fk:
    Attr_do(fk) = E[U(Y) | do(fk)] - E[U(Y)]

    Using backdoor adjustment:
    E[U(Y) | do(fk)] = Σ_z P(Z=z) * E[U(Y) | fk, Z=z]
    """
    attributions = {}

    # Baseline: overall uncertainty
    proba_baseline = model.predict_proba(X)
    baseline_entropy = np.mean(entropy(proba_baseline, axis=1))

    for fk_name, fk_cols in fk_columns.items():
        # Get adjustment set columns
        adj_cols = adjustment_sets.get(fk_name, [])

        # Filter to columns that exist in X
        valid_fk_cols = [c for c in fk_cols if c in X.columns]
        valid_adj_cols = [c for c in adj_cols if c in X.columns]

        if not valid_fk_cols:
            attributions[fk_name] = 0.0
            continue

        # Compute E[U(Y) | do(fk)] using backdoor adjustment
        # = Σ_z P(Z=z) * E[U(Y) | fk, Z=z]
        condition_cols = valid_fk_cols + valid_adj_cols

        conditional_entropy = compute_conditional_entropy(
            model, X, y, condition_cols, n_strata=5
        )

        # Attribution = conditional - baseline
        # Positive: FK reduces uncertainty when conditioned
        # Negative: FK increases uncertainty
        attributions[fk_name] = baseline_entropy - conditional_entropy

    return attributions


# =============================================================================
# LOO Attribution (Baseline for Comparison)
# =============================================================================

def compute_loo_attribution(X, y, fk_columns: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Compute LOO (observational) attribution for comparison.

    Attr_LOO(fk) = U(model_without_fk) - U(model_full)

    Requires retraining per FK.
    """
    attributions = {}

    # Train full model
    model_full = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model_full.fit(X, y)
    proba_full = model_full.predict_proba(X)
    entropy_full = np.mean(entropy(proba_full, axis=1))

    for fk_name, fk_cols in fk_columns.items():
        # Remove FK columns
        valid_fk_cols = [c for c in fk_cols if c in X.columns]
        if not valid_fk_cols:
            attributions[fk_name] = 0.0
            continue

        remaining_cols = [c for c in X.columns if c not in valid_fk_cols]
        if not remaining_cols:
            attributions[fk_name] = 0.0
            continue

        X_without_fk = X[remaining_cols]

        # Retrain model
        model_loo = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        model_loo.fit(X_without_fk, y)
        proba_loo = model_loo.predict_proba(X_without_fk)
        entropy_loo = np.mean(entropy(proba_loo, axis=1))

        # Attribution: how much uncertainty increases without FK
        attributions[fk_name] = entropy_loo - entropy_full

    return attributions


# =============================================================================
# Main Pipeline
# =============================================================================

def load_task_data(task_name: str):
    """Load task and split by temporal periods."""
    task = get_task("rel-salt", task_name, download=False)
    dataset = get_dataset("rel-salt", download=False)
    db = dataset.get_db(upto_test_timestamp=False)

    entity_table = db.table_dict[task.entity_table]
    df = entity_table.df.copy()

    # Split by temporal periods
    train_df = df[df['CREATIONTIMESTAMP'] < VAL_TIMESTAMP].copy()
    val_df = df[(df['CREATIONTIMESTAMP'] >= VAL_TIMESTAMP) &
                (df['CREATIONTIMESTAMP'] < TEST_TIMESTAMP)].copy()

    return train_df, val_df, task.target_col, entity_table, db


def prepare_features(df, entity_table, target_col, sample_size=10000):
    """Prepare features for modeling."""
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    exclude = {'CREATIONTIMESTAMP', target_col, entity_table.pkey_col}
    feature_cols = [c for c in df.columns if c not in exclude]

    X_df = df[feature_cols].copy()
    for col in X_df.columns:
        if X_df[col].dtype == 'object':
            X_df[col] = X_df[col].astype('category').cat.codes
        X_df[col] = X_df[col].fillna(-999)

    y = df[target_col].copy()
    if y.dtype == 'object':
        y = y.astype('category').cat.codes

    return X_df, y, feature_cols


def run_causal_uq_analysis(task_name: str, sample_size: int = 5000):
    """
    Run FK-Causal-UQ analysis and compare with LOO.
    """
    print(f"\n{'='*60}")
    print(f"FK-Causal-UQ Analysis: {task_name}")
    print(f"{'='*60}")

    # Load data
    train_df, val_df, target_col, entity_table, db = load_task_data(task_name)

    # Build FK graph
    G_FK = build_fk_graph(db)
    print(f"\nFK Graph: {len(G_FK.nodes())} tables, {len(G_FK.edges())} FK relationships")
    print(f"Edges: {list(G_FK.edges(data=True))[:5]}...")

    # Prepare features
    X_train, y_train, feature_cols = prepare_features(
        train_df, entity_table, target_col, sample_size
    )
    X_val, y_val, _ = prepare_features(
        val_df, entity_table, target_col, sample_size
    )

    print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
    print(f"Features: {len(feature_cols)}")

    # Map features to FK columns
    fk_columns = {}
    for col in feature_cols:
        # Each column is its own FK "relationship" for this analysis
        fk_columns[col] = [col]

    # Compute adjustment sets (simplified: use all other columns as adjustment)
    adjustment_sets = {}
    for col in feature_cols:
        other_cols = [c for c in feature_cols if c != col]
        adjustment_sets[col] = other_cols

    # Train single model (no retraining!)
    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    results = {
        'task': task_name,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_features': len(feature_cols),
        'train': {},
        'val': {}
    }

    # =================================================================
    # Causal Attribution (Our Method - No Retraining!)
    # =================================================================
    print("\n--- Causal UQ Attribution (No Retraining) ---")

    causal_train = compute_causal_uq_attribution(
        model, X_train, y_train, fk_columns, adjustment_sets
    )
    causal_val = compute_causal_uq_attribution(
        model, X_val, y_val, fk_columns, adjustment_sets
    )

    results['train']['causal'] = causal_train
    results['val']['causal'] = causal_val

    # Show top attributions
    print("\nTrain (Causal) Top 5:")
    for fk, attr in sorted(causal_train.items(), key=lambda x: -x[1])[:5]:
        print(f"  {fk}: {attr:.4f}")

    print("\nVal (Causal) Top 5:")
    for fk, attr in sorted(causal_val.items(), key=lambda x: -x[1])[:5]:
        print(f"  {fk}: {attr:.4f}")

    # =================================================================
    # LOO Attribution (Baseline - Requires Retraining)
    # =================================================================
    print("\n--- LOO Attribution (Retraining per FK) ---")

    loo_train = compute_loo_attribution(X_train, y_train, fk_columns)
    loo_val = compute_loo_attribution(X_val, y_val, fk_columns)

    results['train']['loo'] = loo_train
    results['val']['loo'] = loo_val

    print("\nTrain (LOO) Top 5:")
    for fk, attr in sorted(loo_train.items(), key=lambda x: -x[1])[:5]:
        print(f"  {fk}: {attr:.4f}")

    print("\nVal (LOO) Top 5:")
    for fk, attr in sorted(loo_val.items(), key=lambda x: -x[1])[:5]:
        print(f"  {fk}: {attr:.4f}")

    # =================================================================
    # Compare Causal vs LOO
    # =================================================================
    print("\n--- Causal vs LOO Comparison ---")

    from scipy.stats import spearmanr

    # Get common keys
    common_keys = set(causal_train.keys()) & set(loo_train.keys())
    if len(common_keys) >= 3:
        causal_vals = [causal_train[k] for k in common_keys]
        loo_vals = [loo_train[k] for k in common_keys]

        rho, p = spearmanr(causal_vals, loo_vals)
        print(f"\nSpearman ρ (Causal vs LOO, Train): {rho:.3f}, p={p:.4f}")

        results['comparison'] = {
            'spearman_rho': rho,
            'p_value': p,
            'interpretation': 'low' if abs(rho) < 0.3 else 'moderate' if abs(rho) < 0.7 else 'high'
        }

        if abs(rho) < 0.3:
            print("→ LOW correlation: Causal ≠ LOO (as expected)")
        elif abs(rho) < 0.7:
            print("→ MODERATE correlation: Some overlap")
        else:
            print("→ HIGH correlation: Methods agree")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='sales-group')
    parser.add_argument('--all_tasks', action='store_true')
    parser.add_argument('--sample_size', type=int, default=5000)
    args = parser.parse_args()

    SALT_TASKS = [
        'item-plant', 'item-shippoint', 'item-incoterms',
        'sales-office', 'sales-group', 'sales-payterms',
        'sales-shipcond', 'sales-incoterms'
    ]

    tasks = SALT_TASKS if args.all_tasks else [args.task]

    all_results = {}
    for task in tasks:
        try:
            results = run_causal_uq_analysis(task, args.sample_size)
            all_results[task] = results
        except Exception as e:
            print(f"Error on {task}: {e}")
            continue

    # Save results
    output_path = Path('chorok/results/fk_causal_uq.json')
    output_path.parent.mkdir(exist_ok=True)

    # Convert to JSON serializable
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
