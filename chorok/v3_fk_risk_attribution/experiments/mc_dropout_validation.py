"""
MC Dropout Validation: UQ Method-Agnostic Test
===============================================

Goal: Prove that FK Attribution works regardless of UQ method.

Comparison:
- LightGBM Ensemble (current method)
- MLP with MC Dropout (neural network)

If both achieve similar ρ with error impact → "UQ method-agnostic" claim valid.

Author: ChorokLeeDev
Created: 2025-12-08
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# For comparison
import lightgbm as lgb

CACHE_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/cache')
RESULTS_DIR = Path('/Users/i767700/Github/ai-in-finance/chorok/v3_fk_risk_attribution/results')
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# MC Dropout MLP
# =============================================================================

class MCDropoutMLP(nn.Module):
    """MLP with dropout for MC Dropout uncertainty estimation."""

    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def train_mc_dropout_model(X, y, hidden_dims=[128, 64, 32], dropout_rate=0.2,
                           epochs=100, batch_size=256, lr=0.001, seed=42):
    """Train a single MC Dropout model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y.values if hasattr(y, 'values') else y)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = MCDropoutMLP(X.shape[1], hidden_dims, dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    return model, scaler


def mc_dropout_predict(model, X, scaler, n_samples=50):
    """Get predictions with MC Dropout (multiple forward passes with dropout enabled)."""
    model.train()  # Keep dropout active!

    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X_tensor).numpy()
            predictions.append(pred)

    return np.array(predictions)  # (n_samples, n_data)


def mc_dropout_uncertainty(model, X, scaler, n_samples=50):
    """Compute uncertainty as variance of MC Dropout predictions."""
    preds = mc_dropout_predict(model, X, scaler, n_samples)
    return preds.var(axis=0)  # Variance across samples


def mc_dropout_mean_prediction(model, X, scaler, n_samples=50):
    """Compute mean prediction from MC Dropout."""
    preds = mc_dropout_predict(model, X, scaler, n_samples)
    return preds.mean(axis=0)


# =============================================================================
# LightGBM Ensemble (for comparison)
# =============================================================================

def train_lgbm_ensemble(X, y, n_models=5, base_seed=42):
    """Train LightGBM ensemble."""
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


def lgbm_uncertainty(models, X):
    """Compute uncertainty as variance of ensemble predictions."""
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0)


def lgbm_mean_prediction(models, X):
    """Compute mean prediction from ensemble."""
    preds = np.array([m.predict(X) for m in models])
    return preds.mean(axis=0)


# =============================================================================
# FK Attribution (works with any UQ method)
# =============================================================================

def get_fk_grouping(col_to_fk):
    """Convert column->FK mapping to FK->columns mapping."""
    fk_to_cols = defaultdict(list)
    for col, fk in col_to_fk.items():
        fk_to_cols[fk].append(col)
    return dict(fk_to_cols)


def compute_fk_attribution(uncertainty_fn, X, fk_grouping, n_permute=5):
    """
    Generic FK attribution that works with any uncertainty function.

    Args:
        uncertainty_fn: Function that takes X and returns uncertainty array
        X: Feature DataFrame
        fk_grouping: Dict mapping FK group -> list of columns
        n_permute: Number of permutation runs

    Returns:
        Dict mapping FK group -> attribution percentage
    """
    base_unc = uncertainty_fn(X).mean()
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
            perm_unc = uncertainty_fn(X_perm).mean()
            deltas.append(perm_unc - base_unc)

        results[fk_group] = np.mean(deltas)

    # Normalize to percentages
    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


def compute_error_impact(prediction_fn, X, y, fk_grouping, n_permute=5):
    """Compute ground truth error impact for each FK group."""
    base_pred = prediction_fn(X)
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
            perm_pred = prediction_fn(X_perm)
            perm_mae = mean_absolute_error(y, perm_pred)
            deltas.append(perm_mae - base_mae)

        results[fk_group] = np.mean(deltas)

    # Normalize to percentages
    total = sum(max(0, v) for v in results.values())
    if total > 0:
        return {g: max(0, v) / total * 100 for g, v in results.items()}
    return {g: 0 for g in results}


# =============================================================================
# Main Experiment
# =============================================================================

def load_from_cache(cache_file):
    """Load data from cached pickle file."""
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    if len(data) == 4:
        X, y, feature_cols, col_to_fk = data
        if isinstance(col_to_fk, dict):
            return X, y, feature_cols, col_to_fk
    return None, None, None, None


def run_comparison(X, y, col_to_fk, domain_name, n_runs=3):
    """
    Compare LightGBM Ensemble vs MC Dropout for FK Attribution.
    """
    print(f"\n{'='*70}")
    print(f"MC DROPOUT vs LGBM ENSEMBLE: {domain_name}")
    print(f"{'='*70}")

    fk_grouping = get_fk_grouping(col_to_fk)
    fk_list = list(fk_grouping.keys())

    print(f"Features: {len(X.columns)}, FK groups: {len(fk_list)}")
    print(f"FK groups: {fk_list}")

    all_results = {
        'lgbm_attr': [],
        'mcdropout_attr': [],
        'lgbm_error': [],
        'mcdropout_error': []
    }

    for run in range(n_runs):
        print(f"\n  Run {run+1}/{n_runs}...")
        seed = 42 + run * 10

        # === LightGBM Ensemble ===
        print(f"    Training LightGBM ensemble...")
        lgbm_models = train_lgbm_ensemble(X, y, n_models=5, base_seed=seed)

        # LightGBM uncertainty function
        def lgbm_unc_fn(X_input):
            return lgbm_uncertainty(lgbm_models, X_input)

        def lgbm_pred_fn(X_input):
            return lgbm_mean_prediction(lgbm_models, X_input)

        lgbm_attr = compute_fk_attribution(lgbm_unc_fn, X, fk_grouping, n_permute=5)
        lgbm_error = compute_error_impact(lgbm_pred_fn, X, y, fk_grouping, n_permute=5)

        all_results['lgbm_attr'].append(lgbm_attr)
        all_results['lgbm_error'].append(lgbm_error)

        # === MC Dropout MLP ===
        print(f"    Training MC Dropout MLP...")
        mc_model, scaler = train_mc_dropout_model(
            X, y, hidden_dims=[128, 64, 32], dropout_rate=0.2,
            epochs=100, batch_size=256, seed=seed
        )

        # MC Dropout uncertainty function
        def mc_unc_fn(X_input):
            return mc_dropout_uncertainty(mc_model, X_input, scaler, n_samples=50)

        def mc_pred_fn(X_input):
            return mc_dropout_mean_prediction(mc_model, X_input, scaler, n_samples=50)

        mc_attr = compute_fk_attribution(mc_unc_fn, X, fk_grouping, n_permute=5)
        mc_error = compute_error_impact(mc_pred_fn, X, y, fk_grouping, n_permute=5)

        all_results['mcdropout_attr'].append(mc_attr)
        all_results['mcdropout_error'].append(mc_error)

        print(f"    LGBM Attribution: {[f'{k}:{v:.1f}%' for k,v in lgbm_attr.items()]}")
        print(f"    MC Dropout Attr:  {[f'{k}:{v:.1f}%' for k,v in mc_attr.items()]}")

    # Average across runs
    avg_results = {}
    for method in all_results:
        avg_results[method] = {}
        for fk in fk_list:
            values = [r.get(fk, 0) for r in all_results[method]]
            avg_results[method][fk] = np.mean(values)

    # Compute correlations with error impact
    lgbm_attr_vals = [avg_results['lgbm_attr'].get(fk, 0) for fk in fk_list]
    lgbm_error_vals = [avg_results['lgbm_error'].get(fk, 0) for fk in fk_list]
    mc_attr_vals = [avg_results['mcdropout_attr'].get(fk, 0) for fk in fk_list]
    mc_error_vals = [avg_results['mcdropout_error'].get(fk, 0) for fk in fk_list]

    if len(fk_list) >= 3:
        lgbm_corr, lgbm_p = spearmanr(lgbm_attr_vals, lgbm_error_vals)
        mc_corr, mc_p = spearmanr(mc_attr_vals, mc_error_vals)
    else:
        lgbm_corr, lgbm_p = float('nan'), float('nan')
        mc_corr, mc_p = float('nan'), float('nan')

    # Display results
    print(f"\n  --- Results ---")
    print(f"  {'FK Group':<15} {'LGBM Attr':<12} {'LGBM Err':<12} {'MC Attr':<12} {'MC Err':<12}")
    print(f"  {'-'*60}")

    for fk in fk_list:
        lgbm_a = avg_results['lgbm_attr'].get(fk, 0)
        lgbm_e = avg_results['lgbm_error'].get(fk, 0)
        mc_a = avg_results['mcdropout_attr'].get(fk, 0)
        mc_e = avg_results['mcdropout_error'].get(fk, 0)
        print(f"  {fk:<15} {lgbm_a:>10.1f}% {lgbm_e:>10.1f}% {mc_a:>10.1f}% {mc_e:>10.1f}%")

    print(f"\n  --- Correlation with Error Impact (Spearman ρ) ---")
    lgbm_str = f"{lgbm_corr:.3f}" if not np.isnan(lgbm_corr) else "N/A"
    mc_str = f"{mc_corr:.3f}" if not np.isnan(mc_corr) else "N/A"
    print(f"  LightGBM Ensemble: ρ = {lgbm_str}")
    print(f"  MC Dropout MLP:    ρ = {mc_str}")

    # Verdict
    if not np.isnan(lgbm_corr) and not np.isnan(mc_corr):
        if lgbm_corr > 0.7 and mc_corr > 0.7:
            verdict = "✓ BOTH METHODS WORK - FK Attribution is UQ-agnostic!"
        elif abs(lgbm_corr - mc_corr) < 0.2:
            verdict = "~ Similar performance - method doesn't matter much"
        else:
            verdict = "✗ Methods differ significantly"
    else:
        verdict = "Insufficient FK groups for correlation"

    print(f"\n  Verdict: {verdict}")

    return {
        'domain': domain_name,
        'n_fk_groups': len(fk_list),
        'lgbm_attribution': avg_results['lgbm_attr'],
        'lgbm_error': avg_results['lgbm_error'],
        'mcdropout_attribution': avg_results['mcdropout_attr'],
        'mcdropout_error': avg_results['mcdropout_error'],
        'lgbm_correlation': lgbm_corr if not np.isnan(lgbm_corr) else None,
        'mcdropout_correlation': mc_corr if not np.isnan(mc_corr) else None,
        'verdict': verdict
    }


def run_all_domains():
    """Run MC Dropout validation across all domains."""
    print("="*70)
    print("MC DROPOUT VALIDATION")
    print("Goal: Prove FK Attribution is UQ Method-Agnostic")
    print("="*70)

    all_results = {}

    # SALT
    print("\n[1/2] Loading SALT data...")
    salt_cache = CACHE_DIR / 'data_salt_PLANT_2000.pkl'
    if salt_cache.exists():
        result = load_from_cache(salt_cache)
        if result[0] is not None:
            X, y, _, col_to_fk = result
            salt_results = run_comparison(X, y, col_to_fk, "SALT (ERP)")
            all_results['salt'] = salt_results

    # Avito
    print("\n[2/2] Loading Avito data...")
    avito_cache = CACHE_DIR / 'data_avito_ad-ctr_3000.pkl'
    if avito_cache.exists():
        result = load_from_cache(avito_cache)
        if result[0] is not None:
            X, y, _, col_to_fk = result
            if len(X) > 2000:
                idx = X.sample(2000, random_state=42).index
                X = X.loc[idx]
                y = y.loc[idx]
            avito_results = run_comparison(X, y, col_to_fk, "Avito (Classifieds)")
            all_results['avito'] = avito_results

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: MC Dropout vs LightGBM Ensemble")
    print("="*70)
    print(f"\n{'Domain':<25} {'LGBM ρ':<12} {'MC Dropout ρ':<15} {'Verdict'}")
    print("-"*70)

    for domain_key, results in all_results.items():
        domain = results['domain']
        lgbm = results['lgbm_correlation']
        mc = results['mcdropout_correlation']
        lgbm_str = f"{lgbm:.3f}" if lgbm is not None else "N/A"
        mc_str = f"{mc:.3f}" if mc is not None else "N/A"

        if lgbm is not None and mc is not None and lgbm > 0.7 and mc > 0.7:
            status = "✓ Both work"
        else:
            status = "?"

        print(f"{domain:<25} {lgbm_str:<12} {mc_str:<15} {status}")

    # Save results
    import json

    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if obj is None or (isinstance(obj, float) and np.isnan(obj)):
            return None
        return obj

    output_path = RESULTS_DIR / 'mc_dropout_validation.json'
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\n[SAVED] {output_path}")

    # Final verdict
    lgbm_corrs = [r['lgbm_correlation'] for r in all_results.values() if r['lgbm_correlation'] is not None]
    mc_corrs = [r['mcdropout_correlation'] for r in all_results.values() if r['mcdropout_correlation'] is not None]

    if lgbm_corrs and mc_corrs:
        avg_lgbm = np.mean(lgbm_corrs)
        avg_mc = np.mean(mc_corrs)

        print(f"\n{'='*70}")
        print(f"CONCLUSION")
        print(f"{'='*70}")
        print(f"Average LGBM Ensemble ρ:  {avg_lgbm:.3f}")
        print(f"Average MC Dropout ρ:     {avg_mc:.3f}")

        if avg_lgbm > 0.7 and avg_mc > 0.7:
            print(f"\n✓ FK ATTRIBUTION IS UQ METHOD-AGNOSTIC")
            print(f"  Both tree ensembles and neural networks with MC Dropout")
            print(f"  achieve strong correlation (ρ > 0.7) with error impact.")
        elif abs(avg_lgbm - avg_mc) < 0.2:
            print(f"\n~ Methods perform similarly")
        else:
            print(f"\n✗ Methods show different behavior")

    return all_results


if __name__ == "__main__":
    results = run_all_domains()
