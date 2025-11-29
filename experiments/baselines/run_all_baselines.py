"""
Baseline Comparison for RelUQ
=============================

Implements and compares:
1. Feature-level Permutation (our method at feature granularity)
2. SHAP Variance Attribution (TreeSHAP-based)
3. Correlation Clustering (data-driven grouping)
4. Random Grouping (control)

All compared against RelUQ (FK-level grouping)
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution')
from data_loader_f1 import load_f1_data


# =============================================================================
# Ensemble Training
# =============================================================================

def train_ensemble(X, y, n_models=5, base_seed=42):
    """Train LightGBM ensemble with subsampling for diversity."""
    models = []
    for i in range(n_models):
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=base_seed + i,
            verbose=-1,
            force_col_wise=True
        )
        model.fit(X, y)
        models.append(model)
    return models


def get_uncertainty(models, X):
    """Compute ensemble variance (epistemic uncertainty)."""
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0).mean()


# =============================================================================
# Method 1: Feature-Level Permutation
# =============================================================================

def feature_level_permutation(models, X, n_perm=5):
    """
    Attribute uncertainty to individual features via permutation.
    This is the "unstable" baseline - too many groups.
    """
    base_unc = get_uncertainty(models, X)

    feature_deltas = {}
    for col in X.columns:
        deltas = []
        for _ in range(n_perm):
            X_perm = X.copy()
            X_perm[col] = np.random.permutation(X_perm[col].values)
            perm_unc = get_uncertainty(models, X_perm)
            deltas.append(perm_unc - base_unc)
        feature_deltas[col] = np.mean(deltas)

    # Normalize
    total = sum(max(0, d) for d in feature_deltas.values())
    if total > 0:
        return {f: max(0, d) / total * 100 for f, d in feature_deltas.items()}
    return {f: 0.0 for f in feature_deltas}


# =============================================================================
# Method 2: SHAP Variance Attribution
# =============================================================================

def shap_variance_attribution(models, X, n_samples=500):
    """
    Use SHAP values to attribute uncertainty.
    Compute SHAP for each model, then measure variance across models.

    Interpretation: Features with high SHAP variance across ensemble
    are sources of epistemic uncertainty.
    """
    try:
        import shap
    except ImportError:
        print("[WARN] shap not installed, skipping SHAP baseline")
        return None

    # Sample for speed
    if len(X) > n_samples:
        X_sample = X.sample(n=n_samples, random_state=42)
    else:
        X_sample = X

    # Get SHAP values for each model
    all_shap_values = []
    for model in models:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        all_shap_values.append(shap_values)

    # Stack: (n_models, n_samples, n_features)
    shap_stack = np.array(all_shap_values)

    # Variance across models for each feature
    # Shape: (n_samples, n_features)
    shap_variance = shap_stack.var(axis=0)

    # Mean variance per feature
    feature_variance = shap_variance.mean(axis=0)

    # Normalize to percentages
    total = feature_variance.sum()
    if total > 0:
        return {col: feature_variance[i] / total * 100
                for i, col in enumerate(X.columns)}
    return {col: 0.0 for col in X.columns}


# =============================================================================
# Method 3: Correlation Clustering
# =============================================================================

def correlation_clustering(X, n_clusters=5):
    """
    Group features by correlation structure.
    Data-driven alternative to FK grouping.
    """
    # Compute correlation matrix
    corr_matrix = X.corr().abs()

    # Convert to distance matrix
    distance_matrix = 1 - corr_matrix
    np.fill_diagonal(distance_matrix.values, 0)

    # Hierarchical clustering
    condensed = squareform(distance_matrix.values)
    Z = linkage(condensed, method='average')

    # Cut tree to get clusters
    labels = fcluster(Z, n_clusters, criterion='maxclust')

    # Create mapping
    col_to_cluster = {col: f"CORR_GROUP_{labels[i]}"
                      for i, col in enumerate(X.columns)}

    return col_to_cluster


def group_level_permutation(models, X, col_to_group, n_perm=5):
    """
    Permutation at group level (works for any grouping).
    """
    base_unc = get_uncertainty(models, X)

    # Group columns
    group_to_cols = defaultdict(list)
    for col, grp in col_to_group.items():
        if col in X.columns:
            group_to_cols[grp].append(col)

    # Permute each group
    group_deltas = {}
    for grp, cols in group_to_cols.items():
        deltas = []
        for _ in range(n_perm):
            X_perm = X.copy()
            for col in cols:
                X_perm[col] = np.random.permutation(X_perm[col].values)
            perm_unc = get_uncertainty(models, X_perm)
            deltas.append(perm_unc - base_unc)
        group_deltas[grp] = np.mean(deltas)

    # Normalize
    total = sum(max(0, d) for d in group_deltas.values())
    if total > 0:
        return {g: max(0, d) / total * 100 for g, d in group_deltas.items()}
    return {g: 0.0 for g in group_deltas}


# =============================================================================
# Method 4: Random Grouping
# =============================================================================

def random_grouping(columns, n_groups=5, seed=None):
    """Randomly assign features to groups (control)."""
    if seed is not None:
        np.random.seed(seed)
    labels = np.random.randint(0, n_groups, len(columns))
    return {col: f"RANDOM_{labels[i]}" for i, col in enumerate(columns)}


# =============================================================================
# Stability Test
# =============================================================================

def compute_stability(method_fn, models_fn, X, y, n_seeds=3, **kwargs):
    """
    Run method multiple times with different seeds.
    Return average Spearman correlation (stability).
    """
    results = []

    for seed in range(n_seeds):
        # Train new ensemble
        models = models_fn(X, y, base_seed=100 * seed)

        # Run method
        attr = method_fn(models, X, **kwargs)
        results.append(attr)

    # Compute pairwise Spearman correlations
    keys = list(results[0].keys())
    correlations = []

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            vals_i = [results[i].get(k, 0) for k in keys]
            vals_j = [results[j].get(k, 0) for k in keys]

            # Handle constant arrays
            if len(set(vals_i)) > 1 and len(set(vals_j)) > 1:
                rho, _ = stats.spearmanr(vals_i, vals_j)
                if not np.isnan(rho):
                    correlations.append(rho)

    return np.mean(correlations) if correlations else np.nan


# =============================================================================
# Main Experiment
# =============================================================================

def run_baseline_comparison():
    print("=" * 70)
    print("Baseline Comparison for RelUQ")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data...")
    X, y, feature_cols, col_to_fk = load_f1_data(sample_size=3000, use_cache=True)
    print(f"Data shape: {X.shape}")
    print(f"FK groups: {set(col_to_fk.values())}")

    # Train base ensemble
    print("\n[2/5] Training ensemble...")
    models = train_ensemble(X, y)
    base_unc = get_uncertainty(models, X)
    print(f"Baseline uncertainty: {base_unc:.4f}")

    # Groupings
    corr_grouping = correlation_clustering(X, n_clusters=5)
    rand_grouping = random_grouping(list(X.columns), n_groups=5)

    results = {}

    # Method 1: Feature-level
    print("\n[3/5] Running baselines...")
    print("  - Feature-level permutation...")
    feature_attr = feature_level_permutation(models, X, n_perm=3)
    results['Feature-level'] = {
        'n_groups': len(feature_attr),
        'top_3': sorted(feature_attr.items(), key=lambda x: -x[1])[:3]
    }

    # Method 2: SHAP variance
    print("  - SHAP variance...")
    shap_attr = shap_variance_attribution(models, X, n_samples=500)
    if shap_attr:
        results['SHAP Variance'] = {
            'n_groups': len(shap_attr),
            'top_3': sorted(shap_attr.items(), key=lambda x: -x[1])[:3]
        }

    # Method 3: Correlation clustering
    print("  - Correlation clustering...")
    corr_attr = group_level_permutation(models, X, corr_grouping, n_perm=5)
    results['Correlation'] = {
        'n_groups': len(corr_attr),
        'top_3': sorted(corr_attr.items(), key=lambda x: -x[1])[:3]
    }

    # Method 4: Random grouping
    print("  - Random grouping...")
    rand_attr = group_level_permutation(models, X, rand_grouping, n_perm=5)
    results['Random'] = {
        'n_groups': len(rand_attr),
        'top_3': sorted(rand_attr.items(), key=lambda x: -x[1])[:3]
    }

    # Method 5: FK grouping (RelUQ)
    print("  - FK grouping (RelUQ)...")
    fk_attr = group_level_permutation(models, X, col_to_fk, n_perm=5)
    results['RelUQ (FK)'] = {
        'n_groups': len(fk_attr),
        'top_3': sorted(fk_attr.items(), key=lambda x: -x[1])[:3]
    }

    # Stability tests
    print("\n[4/5] Running stability tests...")

    stability = {}

    # Feature-level stability
    print("  - Feature-level stability...")
    stability['Feature-level'] = compute_stability(
        feature_level_permutation, train_ensemble, X, y, n_seeds=3, n_perm=3
    )

    # Correlation stability
    print("  - Correlation stability...")
    stability['Correlation'] = compute_stability(
        lambda m, x, **kw: group_level_permutation(m, x, corr_grouping, n_perm=3),
        train_ensemble, X, y, n_seeds=3
    )

    # Random stability (use different random grouping each time)
    print("  - Random stability...")
    def random_method(models, X, seed=0, **kw):
        rg = random_grouping(list(X.columns), n_groups=5, seed=seed)
        return group_level_permutation(models, X, rg, n_perm=3)

    # Custom stability for random (different grouping per seed)
    random_results = []
    for seed in range(3):
        models = train_ensemble(X, y, base_seed=100 * seed)
        rg = random_grouping(list(X.columns), n_groups=5, seed=200 + seed)
        attr = group_level_permutation(models, X, rg, n_perm=3)
        random_results.append(attr)

    # Compute stability for random
    random_corrs = []
    for i in range(len(random_results)):
        for j in range(i + 1, len(random_results)):
            # Get common keys (different groupings have same group names)
            keys = list(random_results[i].keys())
            vals_i = [random_results[i].get(k, 0) for k in keys]
            vals_j = [random_results[j].get(k, 0) for k in keys]
            if len(set(vals_i)) > 1 and len(set(vals_j)) > 1:
                rho, _ = stats.spearmanr(vals_i, vals_j)
                if not np.isnan(rho):
                    random_corrs.append(rho)
    stability['Random'] = np.mean(random_corrs) if random_corrs else np.nan

    # FK stability
    print("  - FK stability...")
    stability['RelUQ (FK)'] = compute_stability(
        lambda m, x, **kw: group_level_permutation(m, x, col_to_fk, n_perm=3),
        train_ensemble, X, y, n_seeds=3
    )

    # Print results
    print("\n[5/5] Results")
    print("=" * 70)
    print("\nAttribution Results:")
    print("-" * 70)
    for method, data in results.items():
        print(f"\n{method} ({data['n_groups']} groups):")
        for name, pct in data['top_3']:
            print(f"  {name}: {pct:.1f}%")

    print("\n" + "=" * 70)
    print("Stability Comparison (Spearman ρ across 3 seeds):")
    print("-" * 70)
    for method, stab in sorted(stability.items(), key=lambda x: -x[1] if not np.isnan(x[1]) else -999):
        print(f"  {method:20s}: {stab:.3f}")

    print("\n" + "=" * 70)
    print("SUMMARY TABLE (for paper)")
    print("-" * 70)
    print(f"{'Method':<20} {'Groups':<10} {'Stability':<12} {'Actionable':<12}")
    print("-" * 70)

    actionability = {
        'Feature-level': 'No',
        'SHAP Variance': 'No',
        'Correlation': 'No',
        'Random': 'No',
        'RelUQ (FK)': 'Yes'
    }

    for method in ['Feature-level', 'Correlation', 'Random', 'RelUQ (FK)']:
        n_grp = results[method]['n_groups']
        stab = stability.get(method, np.nan)
        act = actionability[method]
        print(f"{method:<20} {n_grp:<10} {stab:<12.3f} {act:<12}")

    print("=" * 70)

    return results, stability


if __name__ == "__main__":
    results, stability = run_baseline_comparison()
