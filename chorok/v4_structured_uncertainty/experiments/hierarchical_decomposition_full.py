"""
Full 3-Level Hierarchical Decomposition
========================================

Show the ACTUAL contribution: Error propagation through FK → Column → Value

For each domain:
  Level 1: FK Table importance
  Level 2: Column importance WITHIN top FK
  Level 3: Value range importance WITHIN top column

Author: ChorokLeeDev
Created: 2025-12-09
"""

import sys
import numpy as np
import pandas as pd
import pickle
import pyro
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'methods'))

from methods.ensemble_lgbm import train_lgbm_ensemble
from methods.hierarchical_bayesian import HierarchicalBayesianAnalyzer

CACHE_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache')
OUTPUT_DIR = Path(__file__).parent.parent / 'results'
OUTPUT_DIR.mkdir(exist_ok=True)


def load_cached_data(dataset_name, sample_size=3000):
    """Load cached data."""
    patterns = {
        'salt': f'data_salt_PLANT_{sample_size}.pkl',
        'f1': f'data_f1_driver-position_{sample_size}.pkl',
    }

    cache_file = CACHE_DIR / patterns.get(dataset_name, '')

    if not cache_file.exists():
        for size in [2000, 1000, 500]:
            alt_pattern = patterns[dataset_name].replace(str(sample_size), str(size))
            alt_file = CACHE_DIR / alt_pattern
            if alt_file.exists():
                cache_file = alt_file
                break

    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        X = data.get('X')
        y = data.get('y')
        col_to_fk = data.get('col_to_fk', {})
        if isinstance(X, dict):
            X = pd.DataFrame(X)
    else:
        X, y, _, col_to_fk = data
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

    return X, np.array(y).flatten(), col_to_fk


def compute_column_importance(X, col_to_fk, get_unc_fn, target_fk, n_permute=10):
    """Compute importance of columns WITHIN a specific FK."""
    base_unc = get_unc_fn(X)

    # Get columns belonging to target FK
    fk_columns = [col for col, fk in col_to_fk.items() if fk == target_fk and col in X.columns]

    scores = {}
    for col in fk_columns:
        deltas = []
        for _ in range(n_permute):
            X_perm = X.copy()
            X_perm[col] = np.random.permutation(X_perm[col].values)
            perm_unc = get_unc_fn(X_perm)
            deltas.append((perm_unc - base_unc) / base_unc)
        scores[col] = {
            'mean': np.mean(deltas),
            'std': np.std(deltas),
            'ci_lower': np.percentile(deltas, 2.5),
            'ci_upper': np.percentile(deltas, 97.5)
        }

    return scores


def compute_value_range_importance(X, col_to_fk, get_unc_fn, target_col, n_bins=4, n_permute=10):
    """Compute importance of value ranges WITHIN a specific column."""
    base_unc = get_unc_fn(X)

    col_data = X[target_col]

    # Create bins
    if col_data.dtype in ['object', 'category'] or col_data.nunique() < n_bins:
        # Categorical or low cardinality
        unique_vals = col_data.unique()
        bins = {str(v): col_data == v for v in unique_vals[:n_bins]}
    else:
        # Numerical - use quantile bins
        try:
            quantiles = [0, 0.25, 0.5, 0.75, 1.0]
            edges = col_data.quantile(quantiles).values
            bins = {}
            for i in range(len(edges) - 1):
                low, high = edges[i], edges[i+1]
                mask = (col_data >= low) & (col_data <= high)
                bins[f'[{low:.1f}, {high:.1f}]'] = mask
        except:
            return {}

    scores = {}
    for bin_name, mask in bins.items():
        if mask.sum() < 10:  # Skip small bins
            continue

        deltas = []
        for _ in range(n_permute):
            X_perm = X.copy()
            # Permute only values in this bin
            bin_values = X_perm.loc[mask, target_col].values
            X_perm.loc[mask, target_col] = np.random.permutation(bin_values)
            perm_unc = get_unc_fn(X_perm)
            deltas.append((perm_unc - base_unc) / base_unc)

        scores[bin_name] = {
            'mean': np.mean(deltas),
            'std': np.std(deltas),
            'ci_lower': np.percentile(deltas, 2.5),
            'ci_upper': np.percentile(deltas, 97.5),
            'n_samples': int(mask.sum())
        }

    return scores


def run_full_hierarchical_decomposition(domain_name, X, y, col_to_fk, n_samples=800):
    """Run complete 3-level hierarchical decomposition."""

    print(f"\n{'='*70}")
    print(f"HIERARCHICAL ERROR PROPAGATION: {domain_name.upper()}")
    print(f"{'='*70}")

    # Subsample
    if len(X) > n_samples:
        idx = np.random.permutation(len(X))[:n_samples]
        X_sub = X.iloc[idx].reset_index(drop=True)
        y_sub = y[idx]
    else:
        X_sub = X.reset_index(drop=True)
        y_sub = y

    print(f"\nData: {X_sub.shape[0]} samples, {X_sub.shape[1]} features")

    # Train ensemble
    print("\n[Training ensemble...]")
    ensemble = train_lgbm_ensemble(X_sub.values, y_sub, n_models=10, seed=42)

    def get_unc(X_in):
        if hasattr(X_in, 'values'):
            X_in = X_in.values
        return ensemble.get_uncertainty(X_in).mean()

    base_unc = get_unc(X_sub)
    print(f"Baseline uncertainty: {base_unc:.6f}")

    # Build FK structure
    fk_structure = defaultdict(list)
    for col, fk in col_to_fk.items():
        if col in X_sub.columns:
            fk_structure[fk].append(col)

    results = {
        'domain': domain_name,
        'base_uncertainty': base_unc,
        'n_samples': len(X_sub),
        'levels': {}
    }

    # ========== LEVEL 1: FK Tables ==========
    print(f"\n{'─'*70}")
    print("LEVEL 1: FK TABLE IMPORTANCE")
    print(f"{'─'*70}")

    analyzer = HierarchicalBayesianAnalyzer(get_unc)
    fk_scores = {}

    for fk_name in fk_structure.keys():
        fk_cols = fk_structure[fk_name]
        deltas = []
        for _ in range(10):
            X_perm = X_sub.copy()
            for col in fk_cols:
                X_perm[col] = np.random.permutation(X_perm[col].values)
            perm_unc = get_unc(X_perm)
            deltas.append((perm_unc - base_unc) / base_unc)

        fk_scores[fk_name] = {
            'mean': np.mean(deltas),
            'std': np.std(deltas),
            'ci_lower': np.percentile(deltas, 2.5),
            'ci_upper': np.percentile(deltas, 97.5),
            'n_columns': len(fk_cols)
        }

    # Sort by importance
    sorted_fks = sorted(fk_scores.items(), key=lambda x: -x[1]['mean'])

    print(f"\n{'FK Table':<20} │ {'Importance':>12} │ {'95% CI':>25} │ {'Cols':>5}")
    print(f"{'─'*70}")
    for fk_name, score in sorted_fks:
        ci_str = f"[{score['ci_lower']:>7.1%}, {score['ci_upper']:>7.1%}]"
        print(f"{fk_name:<20} │ {score['mean']:>11.1%} │ {ci_str:>25} │ {score['n_columns']:>5}")

    results['levels']['L1_FK'] = dict(sorted_fks)

    # ========== LEVEL 2: Columns within Top FK ==========
    top_fk = sorted_fks[0][0]
    print(f"\n{'─'*70}")
    print(f"LEVEL 2: COLUMN IMPORTANCE within {top_fk}")
    print(f"{'─'*70}")

    col_scores = compute_column_importance(X_sub, col_to_fk, get_unc, top_fk, n_permute=10)
    sorted_cols = sorted(col_scores.items(), key=lambda x: -x[1]['mean'])

    print(f"\n{'Column':<25} │ {'Importance':>12} │ {'95% CI':>25}")
    print(f"{'─'*70}")
    for col_name, score in sorted_cols:
        ci_str = f"[{score['ci_lower']:>7.1%}, {score['ci_upper']:>7.1%}]"
        print(f"{col_name:<25} │ {score['mean']:>11.1%} │ {ci_str:>25}")

    results['levels']['L2_Column'] = {
        'parent_fk': top_fk,
        'columns': dict(sorted_cols)
    }

    # ========== LEVEL 3: Value Ranges within Top Column ==========
    if sorted_cols:
        top_col = sorted_cols[0][0]
        print(f"\n{'─'*70}")
        print(f"LEVEL 3: VALUE RANGE IMPORTANCE within {top_col}")
        print(f"{'─'*70}")

        value_scores = compute_value_range_importance(X_sub, col_to_fk, get_unc, top_col, n_permute=10)
        sorted_values = sorted(value_scores.items(), key=lambda x: -x[1]['mean'])

        print(f"\n{'Value Range':<25} │ {'Importance':>12} │ {'95% CI':>25} │ {'N':>6}")
        print(f"{'─'*70}")
        for val_name, score in sorted_values:
            ci_str = f"[{score['ci_lower']:>7.1%}, {score['ci_upper']:>7.1%}]"
            print(f"{val_name:<25} │ {score['mean']:>11.1%} │ {ci_str:>25} │ {score['n_samples']:>6}")

        results['levels']['L3_Value'] = {
            'parent_fk': top_fk,
            'parent_col': top_col,
            'values': dict(sorted_values)
        }

    # ========== SUMMARY: Full Hierarchy Tree ==========
    print(f"\n{'='*70}")
    print("HIERARCHICAL ERROR PROPAGATION TREE")
    print(f"{'='*70}")

    print(f"\n{domain_name.upper()} (base uncertainty: {base_unc:.6f})")
    print("│")

    for i, (fk_name, fk_score) in enumerate(sorted_fks[:3]):
        prefix = "├──" if i < 2 else "└──"
        print(f"{prefix} L1: {fk_name}: {fk_score['mean']:.1%} [{fk_score['ci_lower']:.1%}, {fk_score['ci_upper']:.1%}]")

        if fk_name == top_fk and sorted_cols:
            for j, (col_name, col_score) in enumerate(sorted_cols[:3]):
                col_prefix = "│   ├──" if i < 2 else "    ├──"
                if j == len(sorted_cols[:3]) - 1:
                    col_prefix = "│   └──" if i < 2 else "    └──"
                print(f"{col_prefix} L2: {col_name}: {col_score['mean']:.1%} [{col_score['ci_lower']:.1%}, {col_score['ci_upper']:.1%}]")

                if col_name == top_col and sorted_values:
                    for k, (val_name, val_score) in enumerate(sorted_values[:3]):
                        val_prefix = "│   │   ├──" if i < 2 and j < len(sorted_cols[:3])-1 else "│       ├──"
                        if k == len(sorted_values[:3]) - 1:
                            val_prefix = "│   │   └──" if i < 2 and j < len(sorted_cols[:3])-1 else "│       └──"
                        print(f"{val_prefix} L3: {val_name}: {val_score['mean']:.1%} (n={val_score['n_samples']})")

    return results


def print_ascii_visualization(results_list):
    """Print ASCII visualization comparing domains."""

    print(f"\n{'='*70}")
    print("CROSS-DOMAIN COMPARISON: HIERARCHICAL ERROR PROPAGATION")
    print(f"{'='*70}")

    for results in results_list:
        domain = results['domain'].upper()
        l1 = results['levels'].get('L1_FK', {})
        l2 = results['levels'].get('L2_Column', {})
        l3 = results['levels'].get('L3_Value', {})

        print(f"\n┌─ {domain} {'─'*(65-len(domain))}")

        # Level 1
        if l1:
            top_fk = list(l1.keys())[0]
            top_fk_score = l1[top_fk]
            bar_len = int(min(40, max(1, top_fk_score['mean'] * 30)))
            bar = '█' * bar_len
            print(f"│ L1: {top_fk:<15} {bar} {top_fk_score['mean']:.0%}")

        # Level 2
        if l2 and l2.get('columns'):
            top_col = list(l2['columns'].keys())[0]
            top_col_score = l2['columns'][top_col]
            bar_len = int(min(40, max(1, top_col_score['mean'] * 30)))
            bar = '█' * bar_len
            print(f"│ └─L2: {top_col:<13} {bar} {top_col_score['mean']:.0%}")

        # Level 3
        if l3 and l3.get('values'):
            top_val = list(l3['values'].keys())[0]
            top_val_score = l3['values'][top_val]
            bar_len = int(min(40, max(1, top_val_score['mean'] * 30)))
            bar = '█' * bar_len
            print(f"│   └─L3: {top_val:<11} {bar} {top_val_score['mean']:.0%}")

        print(f"└{'─'*68}")


def main():
    print("="*70)
    print("FULL 3-LEVEL HIERARCHICAL DECOMPOSITION")
    print("Demonstrating: FK → Column → Value Error Propagation")
    print("="*70)

    np.random.seed(42)
    pyro.set_rng_seed(42)

    all_results = []

    # F1 Domain
    print("\n>>> Loading F1 data...")
    X, y, col_to_fk = load_cached_data('f1')
    results_f1 = run_full_hierarchical_decomposition('F1', X, y, col_to_fk)
    all_results.append(results_f1)

    # SALT Domain
    print("\n>>> Loading SALT data...")
    X, y, col_to_fk = load_cached_data('salt')
    results_salt = run_full_hierarchical_decomposition('SALT', X, y, col_to_fk)
    all_results.append(results_salt)

    # Cross-domain visualization
    print_ascii_visualization(all_results)

    # Save results
    output_file = OUTPUT_DIR / 'hierarchical_decomposition_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = main()
