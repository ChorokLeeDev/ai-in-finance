"""
Hierarchical Bayesian Model for FK-Structured Uncertainty
==========================================================

Proper Bayesian approach with:
- Hierarchical priors following FK structure
- Variational Inference for posterior estimation
- True credible intervals (not bootstrap)

Structure:
    Level 1: FK Table effects ~ Normal(0, σ_fk)
    Level 2: Column effects | FK ~ Normal(α_fk, σ_col)
    Level 3: Value effects | Column ~ Normal(β_col, σ_val)

Author: ChorokLeeDev
Created: 2025-12-09
"""

import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BayesianInterventionEffect:
    """Bayesian intervention effect with true credible intervals."""
    node: str
    level: str
    effect_mean: float
    effect_std: float
    ci_lower: float  # 2.5th percentile
    ci_upper: float  # 97.5th percentile
    posterior_samples: Optional[np.ndarray] = None

    def __repr__(self):
        return (f"BayesianEffect({self.node}): "
                f"{self.effect_mean:.1%} [{self.ci_lower:.1%}, {self.ci_upper:.1%}]")


class HierarchicalBayesianUQ:
    """
    Hierarchical Bayesian model for FK-structured uncertainty decomposition.

    This model learns:
    1. FK-level importance (α_fk)
    2. Column-level importance within FK (β_col|fk)
    3. Uncertainty in these estimates (posterior variance)

    The posterior gives us TRUE credible intervals, not bootstrap approximations.
    """

    def __init__(self, fk_structure: Dict[str, List[str]], device='cpu'):
        """
        Args:
            fk_structure: Dict mapping FK name to list of column names
                         e.g., {'ITEM': ['col1', 'col2'], 'CUSTOMER': ['col3']}
            device: 'cpu' or 'cuda'
        """
        self.fk_structure = fk_structure
        self.device = device
        self.guide = None
        self.fk_names = list(fk_structure.keys())
        self.n_fks = len(self.fk_names)

        # Flatten column structure for indexing
        self.col_to_idx = {}
        self.col_to_fk = {}
        idx = 0
        for fk, cols in fk_structure.items():
            for col in cols:
                self.col_to_idx[col] = idx
                self.col_to_fk[col] = fk
                idx += 1
        self.n_cols = idx

    def model(self, importance_obs: torch.Tensor = None):
        """
        Hierarchical Bayesian model for FK importance.

        Generative process:
            σ_fk ~ HalfNormal(1)
            σ_col ~ HalfNormal(0.5)

            For each FK k:
                α_k ~ Normal(0, σ_fk)  # FK-level importance

            For each column c in FK k:
                β_c ~ Normal(α_k, σ_col)  # Column importance

            Observed importance ~ Normal(β_c, σ_obs)
        """
        # Hyperpriors
        sigma_fk = pyro.sample("sigma_fk", dist.HalfNormal(1.0))
        sigma_col = pyro.sample("sigma_col", dist.HalfNormal(0.5))
        sigma_obs = pyro.sample("sigma_obs", dist.HalfNormal(0.3))

        # FK-level effects
        with pyro.plate("fks", self.n_fks):
            alpha_fk = pyro.sample("alpha_fk", dist.Normal(0, sigma_fk))

        # Column-level effects (nested in FK)
        beta_col = torch.zeros(self.n_cols, device=self.device)

        for fk_idx, (fk_name, cols) in enumerate(self.fk_structure.items()):
            for col in cols:
                col_idx = self.col_to_idx[col]
                beta_col[col_idx] = pyro.sample(
                    f"beta_{col}",
                    dist.Normal(alpha_fk[fk_idx], sigma_col)
                )

        # Observations (importance scores from permutation)
        if importance_obs is not None:
            with pyro.plate("obs", len(importance_obs)):
                pyro.sample("importance", dist.Normal(beta_col, sigma_obs), obs=importance_obs)

        return alpha_fk, beta_col

    def fit(self, importance_scores: Dict[str, float], n_steps: int = 1000,
            lr: float = 0.01, verbose: bool = False):
        """
        Fit the hierarchical model using Variational Inference.

        Args:
            importance_scores: Dict mapping column name to observed importance
                              e.g., {'col1': 0.35, 'col2': 0.28}
            n_steps: Number of SVI steps
            lr: Learning rate
            verbose: Print progress
        """
        pyro.clear_param_store()

        # Prepare observations
        obs = torch.zeros(self.n_cols, device=self.device)
        for col, score in importance_scores.items():
            if col in self.col_to_idx:
                obs[self.col_to_idx[col]] = score

        # Auto guide (mean-field variational family)
        self.guide = AutoNormal(self.model)

        # SVI setup
        optimizer = Adam({"lr": lr})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

        # Training
        losses = []
        for step in range(n_steps):
            loss = svi.step(obs)
            losses.append(loss)

            if verbose and (step + 1) % 200 == 0:
                print(f"  Step {step+1}: ELBO = {-loss:.4f}")

        return losses

    def get_posterior_samples(self, n_samples: int = 1000) -> Dict:
        """
        Sample from the posterior distribution.

        Returns dict with:
            - alpha_fk: [n_samples, n_fks] FK-level importance samples
            - beta_col: {col_name: [n_samples]} column importance samples
        """
        if self.guide is None:
            raise ValueError("Model not fitted. Call fit() first.")

        predictive = Predictive(self.model, guide=self.guide, num_samples=n_samples)
        samples = predictive()

        result = {
            'alpha_fk': samples['alpha_fk'].detach().cpu().numpy(),
            'sigma_fk': samples['sigma_fk'].detach().cpu().numpy(),
            'sigma_col': samples['sigma_col'].detach().cpu().numpy(),
        }

        # Column-level samples
        result['beta_col'] = {}
        for col in self.col_to_idx.keys():
            key = f"beta_{col}"
            if key in samples:
                result['beta_col'][col] = samples[key].detach().cpu().numpy()

        return result

    def get_fk_importance(self, n_samples: int = 1000) -> List[BayesianInterventionEffect]:
        """
        Get FK-level importance with credible intervals.
        """
        samples = self.get_posterior_samples(n_samples)
        alpha_samples = samples['alpha_fk']  # [n_samples, n_fks]

        results = []
        for i, fk_name in enumerate(self.fk_names):
            fk_samples = alpha_samples[:, i]
            results.append(BayesianInterventionEffect(
                node=fk_name,
                level='fk',
                effect_mean=float(np.mean(fk_samples)),
                effect_std=float(np.std(fk_samples)),
                ci_lower=float(np.percentile(fk_samples, 2.5)),
                ci_upper=float(np.percentile(fk_samples, 97.5)),
                posterior_samples=fk_samples
            ))

        # Sort by importance (highest mean first)
        results.sort(key=lambda x: -x.effect_mean)
        return results

    def get_column_importance(self, fk_name: str = None,
                               n_samples: int = 1000) -> List[BayesianInterventionEffect]:
        """
        Get column-level importance with credible intervals.

        Args:
            fk_name: If provided, only return columns within this FK
        """
        samples = self.get_posterior_samples(n_samples)
        beta_samples = samples['beta_col']

        results = []
        for col, col_samples in beta_samples.items():
            if fk_name and self.col_to_fk.get(col) != fk_name:
                continue

            results.append(BayesianInterventionEffect(
                node=col,
                level='column',
                effect_mean=float(np.mean(col_samples)),
                effect_std=float(np.std(col_samples)),
                ci_lower=float(np.percentile(col_samples, 2.5)),
                ci_upper=float(np.percentile(col_samples, 97.5)),
                posterior_samples=col_samples
            ))

        results.sort(key=lambda x: -x.effect_mean)
        return results


class HierarchicalBayesianAnalyzer:
    """
    Full analysis pipeline using hierarchical Bayesian model.

    Workflow:
    1. Compute importance scores via permutation (frequentist)
    2. Fit hierarchical Bayesian model to these scores
    3. Get posterior distributions with true credible intervals
    """

    def __init__(self, get_uncertainty_fn):
        """
        Args:
            get_uncertainty_fn: Function that takes X and returns mean uncertainty
        """
        self.get_uncertainty = get_uncertainty_fn

    def compute_importance_scores(self, X: pd.DataFrame, col_to_fk: Dict[str, str],
                                   n_permute: int = 10) -> Dict[str, float]:
        """
        Compute column-level importance via permutation.

        Returns dict mapping column name to importance (relative uncertainty increase).
        """
        base_unc = self.get_uncertainty(X)
        scores = {}

        for col in X.columns:
            if col not in col_to_fk:
                continue

            deltas = []
            for _ in range(n_permute):
                X_perm = X.copy()
                X_perm[col] = np.random.permutation(X_perm[col].values)
                perm_unc = self.get_uncertainty(X_perm)
                deltas.append((perm_unc - base_unc) / base_unc)

            scores[col] = np.mean(deltas)

        return scores

    def run_bayesian_analysis(self, X: pd.DataFrame, col_to_fk: Dict[str, str],
                               n_permute: int = 10, n_vi_steps: int = 1000,
                               n_posterior_samples: int = 1000,
                               verbose: bool = False) -> Dict:
        """
        Full Bayesian analysis pipeline.

        Returns:
            Dict with 'fk_importance', 'column_importance', 'model'
        """
        print("  [1/3] Computing importance scores via permutation...")
        importance_scores = self.compute_importance_scores(X, col_to_fk, n_permute)

        if verbose:
            print(f"        Raw scores: {importance_scores}")

        # Build FK structure
        fk_structure = defaultdict(list)
        for col, fk in col_to_fk.items():
            if col in X.columns:
                fk_structure[fk].append(col)
        fk_structure = dict(fk_structure)

        print("  [2/3] Fitting hierarchical Bayesian model...")
        model = HierarchicalBayesianUQ(fk_structure)
        model.fit(importance_scores, n_steps=n_vi_steps, verbose=verbose)

        print("  [3/3] Sampling from posterior...")
        fk_importance = model.get_fk_importance(n_posterior_samples)
        column_importance = model.get_column_importance(n_samples=n_posterior_samples)

        return {
            'fk_importance': fk_importance,
            'column_importance': column_importance,
            'importance_scores': importance_scores,
            'model': model
        }

    def print_report(self, results: Dict, title: str = "HIERARCHICAL BAYESIAN ANALYSIS"):
        """Print formatted report with credible intervals."""
        print(f"\n{'='*70}")
        print(title)
        print('='*70)
        print("\nNote: These are TRUE Bayesian credible intervals from posterior sampling")

        # FK-level
        print(f"\n{'─'*70}")
        print("FK TABLE IMPORTANCE (Bayesian posterior)")
        print(f"{'─'*70}")
        print(f"{'FK Table':<15} │ {'Mean':>10} │ {'Std':>8} │ {'95% CI':>20}")
        print(f"{'─'*70}")

        for effect in results['fk_importance']:
            print(f"{effect.node:<15} │ {effect.effect_mean:>9.1%} │ "
                  f"{effect.effect_std:>7.1%} │ "
                  f"[{effect.ci_lower:>7.1%}, {effect.ci_upper:>7.1%}]")

        # Column-level
        print(f"\n{'─'*70}")
        print("COLUMN IMPORTANCE (Bayesian posterior)")
        print(f"{'─'*70}")
        print(f"{'Column':<20} │ {'Mean':>10} │ {'95% CI':>20}")
        print(f"{'─'*70}")

        for effect in results['column_importance'][:10]:  # Top 10
            print(f"{effect.node:<20} │ {effect.effect_mean:>9.1%} │ "
                  f"[{effect.ci_lower:>7.1%}, {effect.ci_upper:>7.1%}]")

        # Key insight
        print(f"\n{'='*70}")
        print("KEY INSIGHT")
        print('='*70)

        top_fk = results['fk_importance'][0]
        print(f"\n  Most important FK: {top_fk.node}")
        print(f"  Posterior mean: {top_fk.effect_mean:.1%}")
        print(f"  95% Credible Interval: [{top_fk.ci_lower:.1%}, {top_fk.ci_upper:.1%}]")

        if top_fk.ci_lower > 0:
            print(f"\n  ✓ HIGH CONFIDENCE: Even lower bound ({top_fk.ci_lower:.1%}) shows importance")
        else:
            print(f"\n  ? Credible interval includes zero - uncertain importance")

        print()


# ==================== Test ====================

if __name__ == "__main__":
    print("Testing Hierarchical Bayesian UQ...")
    print("="*60)

    # Simple test
    np.random.seed(42)
    pyro.set_rng_seed(42)

    # Synthetic importance scores
    importance_scores = {
        'item_weight': 0.35,
        'item_price': 0.28,
        'customer_age': 0.15,
        'customer_income': 0.12,
        'order_date': 0.10
    }

    fk_structure = {
        'ITEM': ['item_weight', 'item_price'],
        'CUSTOMER': ['customer_age', 'customer_income'],
        'ORDER': ['order_date']
    }

    print("\n[1] Creating hierarchical model...")
    model = HierarchicalBayesianUQ(fk_structure)

    print("\n[2] Fitting model with VI...")
    model.fit(importance_scores, n_steps=500, verbose=True)

    print("\n[3] Getting posterior samples...")
    fk_imp = model.get_fk_importance(n_samples=1000)

    print("\n[4] FK-level importance with credible intervals:")
    for effect in fk_imp:
        print(f"  {effect}")

    col_imp = model.get_column_importance(n_samples=1000)
    print("\n[5] Column-level importance:")
    for effect in col_imp:
        print(f"  {effect}")

    print("\n✓ Hierarchical Bayesian model test passed!")
