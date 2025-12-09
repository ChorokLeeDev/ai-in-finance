"""
Hierarchical Bayesian Intervention Analysis
============================================

Core implementation for:
- Intervention simulation at each hierarchy level
- Effect estimation with bootstrap confidence intervals
- Full hierarchical analysis pipeline

Author: ChorokLeeDev
Created: 2025-12-09
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')


@dataclass
class InterventionEffect:
    """Result of an intervention effect estimation."""
    node: str
    level: str  # 'fk', 'column', 'value_range'
    intervention_type: str
    effect_mean: float  # Expected uncertainty reduction (negative = good)
    effect_std: float
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    n_samples: int
    details: Optional[Dict] = None

    def __repr__(self):
        return (f"InterventionEffect({self.node}, {self.level}): "
                f"{self.effect_mean:.1%} [{self.ci_lower:.1%}, {self.ci_upper:.1%}]")


class HierarchicalInterventionAnalyzer:
    """
    Analyze intervention effects at each level of the FK hierarchy.

    Hierarchy:
        Level 1: FK Table (e.g., PLANT, ITEM, CUSTOMER)
        Level 2: Column within FK (e.g., PLANT.capacity)
        Level 3: Value range within column (e.g., capacity in [0, 100])

    For each node, estimate:
        - Current uncertainty contribution
        - Expected effect of intervention
        - 95% confidence interval on effect
    """

    def __init__(self, get_uncertainty_fn: Callable, get_prediction_fn: Callable = None):
        """
        Args:
            get_uncertainty_fn: Function that takes X (DataFrame or array) and returns
                               mean epistemic uncertainty (scalar)
            get_prediction_fn: Optional function for prediction (for error impact)
        """
        self.get_uncertainty = get_uncertainty_fn
        self.get_prediction = get_prediction_fn

    def _get_fk_grouping(self, col_to_fk: Dict[str, str]) -> Dict[str, List[str]]:
        """Convert column-to-FK mapping to FK-to-columns mapping."""
        fk_to_cols = defaultdict(list)
        for col, fk in col_to_fk.items():
            fk_to_cols[fk].append(col)
        return dict(fk_to_cols)

    # ==================== Intervention Simulation ====================

    def _simulate_intervention(self, X: pd.DataFrame, columns: List[str],
                                intervention_type: str) -> pd.DataFrame:
        """
        Simulate an intervention on specified columns.

        Key insight: We want to measure IMPORTANCE via permutation.
        - If permuting a column INCREASES uncertainty a lot → that column is IMPORTANT
        - Important columns = candidates for data quality improvement

        Intervention types:
            - 'permute': Random permutation - breaks feature-target relationship
                         High Δ uncertainty = high importance = prioritize this
            - 'add_noise': Add noise (alternative importance measure)
            - 'impute_mean': Replace with mean (not recommended - removes signal)
            - 'reduce_variance': Shrink toward mean by 50%
            - 'remove_outliers': Clip to [Q1-1.5*IQR, Q3+1.5*IQR]
        """
        X_new = X.copy()

        for col in columns:
            if col not in X_new.columns:
                continue

            values = X_new[col].values

            if intervention_type == 'permute':
                # Permute - breaks relationship, measures importance
                # High increase in uncertainty = important feature
                X_new[col] = np.random.permutation(values)

            elif intervention_type == 'add_noise':
                # Add significant noise to degrade signal
                noise = np.random.normal(0, np.nanstd(values), len(values))
                X_new[col] = values + noise

            elif intervention_type == 'impute_mean':
                # Replace all with mean
                X_new[col] = np.nanmean(values)

            elif intervention_type == 'reduce_variance':
                # Shrink toward mean by 50%
                mean = np.nanmean(values)
                X_new[col] = mean + 0.5 * (values - mean)

            elif intervention_type == 'remove_outliers':
                # Clip to IQR bounds
                Q1, Q3 = np.nanpercentile(values, [25, 75])
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                X_new[col] = np.clip(values, lower, upper)

        return X_new

    # ==================== Effect Estimation ====================

    def estimate_effect_bootstrap(self, X: pd.DataFrame, columns: List[str],
                                   intervention_type: str, n_bootstrap: int = 100,
                                   sample_frac: float = 0.8) -> InterventionEffect:
        """
        Estimate intervention effect with bootstrap confidence intervals.

        Process:
            1. Get baseline uncertainty
            2. For each bootstrap sample:
                a. Sample data with replacement
                b. Apply intervention
                c. Compute uncertainty change
            3. Return mean effect and 95% CI
        """
        # Baseline uncertainty (on full data)
        base_unc = self.get_uncertainty(X)

        effects = []
        n_samples = len(X)
        sample_size = int(n_samples * sample_frac)

        for _ in range(n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(n_samples, size=sample_size, replace=True)
            X_sample = X.iloc[idx].reset_index(drop=True)

            # Baseline uncertainty on sample
            sample_base_unc = self.get_uncertainty(X_sample)

            # Apply intervention
            X_intervened = self._simulate_intervention(X_sample, columns, intervention_type)

            # Uncertainty after intervention
            intervened_unc = self.get_uncertainty(X_intervened)

            # Effect = (after - before) / before
            # Negative effect = uncertainty reduced (good)
            if sample_base_unc > 0:
                relative_effect = (intervened_unc - sample_base_unc) / sample_base_unc
            else:
                relative_effect = 0

            effects.append(relative_effect)

        effects = np.array(effects)

        return InterventionEffect(
            node='+'.join(columns) if len(columns) <= 3 else f"{columns[0]}+{len(columns)-1}more",
            level='custom',
            intervention_type=intervention_type,
            effect_mean=np.mean(effects),
            effect_std=np.std(effects),
            ci_lower=np.percentile(effects, 2.5),
            ci_upper=np.percentile(effects, 97.5),
            n_samples=n_bootstrap
        )

    # ==================== Hierarchical Analysis ====================

    def analyze_level1_fk(self, X: pd.DataFrame, col_to_fk: Dict[str, str],
                          intervention_type: str = 'impute_mean',
                          n_bootstrap: int = 50) -> List[InterventionEffect]:
        """
        Level 1: Analyze intervention effects at FK (table) level.

        For each FK group, estimate effect of improving all columns in that group.
        """
        fk_grouping = self._get_fk_grouping(col_to_fk)
        results = []

        for fk_name, columns in fk_grouping.items():
            valid_cols = [c for c in columns if c in X.columns]
            if not valid_cols:
                continue

            effect = self.estimate_effect_bootstrap(
                X, valid_cols, intervention_type, n_bootstrap
            )
            effect.node = fk_name
            effect.level = 'fk'
            results.append(effect)

        # Sort by effect (most negative = best improvement)
        results.sort(key=lambda x: x.effect_mean)
        return results

    def analyze_level2_column(self, X: pd.DataFrame, col_to_fk: Dict[str, str],
                               target_fk: str, intervention_type: str = 'impute_mean',
                               n_bootstrap: int = 50) -> List[InterventionEffect]:
        """
        Level 2: Analyze intervention effects at column level within a FK.
        """
        fk_grouping = self._get_fk_grouping(col_to_fk)
        columns = fk_grouping.get(target_fk, [])
        valid_cols = [c for c in columns if c in X.columns]

        results = []
        for col in valid_cols:
            effect = self.estimate_effect_bootstrap(
                X, [col], intervention_type, n_bootstrap
            )
            effect.node = f"{target_fk}.{col}"
            effect.level = 'column'
            results.append(effect)

        results.sort(key=lambda x: x.effect_mean)
        return results

    def analyze_level3_value_range(self, X: pd.DataFrame, target_column: str,
                                    n_bins: int = 4, intervention_type: str = 'impute_mean',
                                    n_bootstrap: int = 30) -> List[InterventionEffect]:
        """
        Level 3: Analyze intervention effects at value range level.

        Split data by target_column values, estimate effect for each bin.
        """
        if target_column not in X.columns:
            return []

        values = X[target_column].values

        # Create bins
        try:
            bins = pd.qcut(values, q=n_bins, labels=False, duplicates='drop')
        except:
            bins = pd.cut(values, bins=n_bins, labels=False)

        results = []
        unique_bins = np.unique(bins[~np.isnan(bins)])

        for b in unique_bins:
            mask = bins == b
            if mask.sum() < 20:  # Need minimum samples
                continue

            # Get value range for this bin
            bin_values = values[mask]
            val_min, val_max = bin_values.min(), bin_values.max()

            # Subset data
            X_subset = X[mask].reset_index(drop=True)

            # Estimate effect on this subset
            effect = self.estimate_effect_bootstrap(
                X_subset, [target_column], intervention_type, n_bootstrap
            )
            effect.node = f"{target_column}:[{val_min:.2f},{val_max:.2f}]"
            effect.level = 'value_range'
            effect.details = {
                'bin': int(b),
                'value_min': float(val_min),
                'value_max': float(val_max),
                'n_samples': int(mask.sum())
            }
            results.append(effect)

        results.sort(key=lambda x: x.effect_mean)
        return results

    # ==================== Full Hierarchical Analysis ====================

    def run_full_analysis(self, X: pd.DataFrame, col_to_fk: Dict[str, str],
                          intervention_type: str = 'impute_mean',
                          n_bootstrap: int = 50,
                          top_k: int = 3) -> Dict:
        """
        Run full hierarchical intervention analysis.

        Returns analysis at all 3 levels with intervention effects and CIs.
        """
        results = {
            'level1_fk': [],
            'level2_column': [],
            'level3_value_range': [],
            'summary': {}
        }

        # Level 1: FK tables
        print("  Analyzing Level 1 (FK Tables)...")
        level1 = self.analyze_level1_fk(X, col_to_fk, intervention_type, n_bootstrap)
        results['level1_fk'] = level1

        if not level1:
            return results

        # Top FK for drill-down (sort by importance = highest effect)
        sorted_level1 = sorted(level1, key=lambda x: -x.effect_mean)
        top_fk = sorted_level1[0].node

        # Level 2: Columns within top FK
        print(f"  Analyzing Level 2 (Columns within {top_fk})...")
        level2 = self.analyze_level2_column(X, col_to_fk, top_fk, intervention_type, n_bootstrap)
        results['level2_column'] = level2

        if not level2:
            return results

        # Top column for drill-down (sort by importance)
        sorted_level2 = sorted(level2, key=lambda x: -x.effect_mean)
        top_col = sorted_level2[0].node.split('.')[-1]  # Get column name without FK prefix

        # Level 3: Value ranges within top column
        print(f"  Analyzing Level 3 (Value ranges within {top_col})...")
        level3 = self.analyze_level3_value_range(X, top_col, n_bins=4,
                                                  intervention_type=intervention_type,
                                                  n_bootstrap=n_bootstrap // 2)
        results['level3_value_range'] = level3

        # Summary (use sorted lists)
        sorted_l3 = sorted(level3, key=lambda x: -x.effect_mean) if level3 else []

        results['summary'] = {
            'top_fk': {
                'name': top_fk,
                'effect': sorted_level1[0].effect_mean,
                'ci': (sorted_level1[0].ci_lower, sorted_level1[0].ci_upper)
            },
            'top_column': {
                'name': f"{top_fk}.{top_col}",
                'effect': sorted_level2[0].effect_mean if sorted_level2 else None,
                'ci': (sorted_level2[0].ci_lower, sorted_level2[0].ci_upper) if sorted_level2 else None
            },
            'top_value_range': {
                'name': sorted_l3[0].node if sorted_l3 else None,
                'effect': sorted_l3[0].effect_mean if sorted_l3 else None,
                'ci': (sorted_l3[0].ci_lower, sorted_l3[0].ci_upper) if sorted_l3 else None
            }
        }

        return results

    def print_report(self, results: Dict, title: str = "HIERARCHICAL INTERVENTION ANALYSIS"):
        """
        Print formatted report.

        Interpretation:
        - Effect = uncertainty increase when feature is permuted
        - HIGHER effect = MORE important feature = PRIORITIZE for data quality improvement
        - Credible interval tells us confidence in the importance ranking
        """
        print(f"\n{'='*70}")
        print(title)
        print('='*70)

        print("\nInterpretation: Higher % = More important = Higher priority for improvement")

        # Level 1
        print(f"\n{'─'*70}")
        print("LEVEL 1: FK TABLE IMPORTANCE (uncertainty increase when permuted)")
        print(f"{'─'*70}")
        print(f"{'FK Table':<15} │ {'Importance':>10} │ {'95% CI':>20} │ Priority")
        print(f"{'─'*70}")

        # Sort by importance (highest first)
        sorted_l1 = sorted(results.get('level1_fk', []), key=lambda x: -x.effect_mean)
        for i, effect in enumerate(sorted_l1):
            priority = "★★★ HIGH" if i == 0 else ("★★ MEDIUM" if i == 1 else "★ LOW")
            print(f"{effect.node:<15} │ {effect.effect_mean:>9.1%} │ "
                  f"[{effect.ci_lower:>7.1%}, {effect.ci_upper:>7.1%}] │ {priority}")

        # Level 2
        if results.get('level2_column'):
            print(f"\n{'─'*70}")
            top_fk = results['summary']['top_fk']['name']
            print(f"LEVEL 2: COLUMN IMPORTANCE (within {top_fk})")
            print(f"{'─'*70}")
            print(f"{'Column':<25} │ {'Importance':>10} │ {'95% CI':>20}")
            print(f"{'─'*70}")

            sorted_l2 = sorted(results.get('level2_column', []), key=lambda x: -x.effect_mean)
            for effect in sorted_l2[:5]:  # Top 5
                col_name = effect.node.split('.')[-1][:20]
                print(f"{col_name:<25} │ {effect.effect_mean:>9.1%} │ "
                      f"[{effect.ci_lower:>7.1%}, {effect.ci_upper:>7.1%}]")

        # Level 3
        if results.get('level3_value_range'):
            print(f"\n{'─'*70}")
            top_col = results['summary']['top_column']['name']
            print(f"LEVEL 3: VALUE RANGE IMPORTANCE (within {top_col})")
            print(f"{'─'*70}")
            print(f"{'Value Range':<30} │ {'Importance':>10} │ {'N':>6}")
            print(f"{'─'*70}")

            sorted_l3 = sorted(results.get('level3_value_range', []), key=lambda x: -x.effect_mean)
            for effect in sorted_l3:
                n = effect.details.get('n_samples', 0) if effect.details else 0
                print(f"{effect.node:<30} │ {effect.effect_mean:>9.1%} │ {n:>6}")

        # Summary - ACTIONABLE RECOMMENDATIONS
        print(f"\n{'='*70}")
        print("ACTIONABLE RECOMMENDATIONS (prioritized by importance)")
        print('='*70)

        # Get top FK from sorted list
        sorted_l1 = sorted(results.get('level1_fk', []), key=lambda x: -x.effect_mean)
        if sorted_l1:
            top = sorted_l1[0]
            print(f"\n  ★★★ TOP PRIORITY: Improve {top.node} data quality")
            print(f"      This FK contributes {top.effect_mean:.1%} to model uncertainty")
            print(f"      95% CI: [{top.ci_lower:.1%}, {top.ci_upper:.1%}]")

            # What this means
            print(f"\n      What this means:")
            print(f"      - If {top.node} data were perfect, uncertainty would drop ~{top.effect_mean:.0%}")
            print(f"      - Even conservative estimate: ~{top.ci_lower:.0%} reduction")

        sorted_l2 = sorted(results.get('level2_column', []), key=lambda x: -x.effect_mean)
        if sorted_l2:
            top_col = sorted_l2[0]
            col_name = top_col.node.split('.')[-1]
            print(f"\n  ★★ DRILL DOWN: Focus on '{col_name}' column")
            print(f"      Within the top FK, this column is most critical")
            print(f"      Importance: {top_col.effect_mean:.1%} [{top_col.ci_lower:.1%}, {top_col.ci_upper:.1%}]")

        sorted_l3 = sorted(results.get('level3_value_range', []), key=lambda x: -x.effect_mean)
        if sorted_l3:
            top_vr = sorted_l3[0]
            print(f"\n  ★ SPECIFIC ACTION: {top_vr.node}")
            print(f"      Data in this value range needs most attention")
            if top_vr.details:
                print(f"      Affected samples: {top_vr.details.get('n_samples', 'N/A')}")

        print()


# ==================== Convenience Function ====================

def run_hierarchical_intervention_analysis(X, y, col_to_fk, model_fn,
                                            intervention_type='impute_mean',
                                            n_bootstrap=50):
    """
    Convenience function to run full hierarchical intervention analysis.

    Args:
        X: Feature DataFrame
        y: Target array
        col_to_fk: Dict mapping column names to FK names
        model_fn: Function that takes (X, y) and returns a model with
                  get_uncertainty(X) method
        intervention_type: Type of intervention to simulate
        n_bootstrap: Number of bootstrap samples for CI estimation

    Returns:
        Dict with analysis results at all 3 levels
    """
    # Train model
    model = model_fn(X.values if hasattr(X, 'values') else X, y)

    # Create uncertainty function
    def get_unc(X_in):
        if hasattr(X_in, 'values'):
            X_in = X_in.values
        return model.get_uncertainty(X_in).mean()

    # Run analysis
    analyzer = HierarchicalInterventionAnalyzer(get_unc)
    results = analyzer.run_full_analysis(
        X if isinstance(X, pd.DataFrame) else pd.DataFrame(X),
        col_to_fk,
        intervention_type,
        n_bootstrap
    )

    analyzer.print_report(results)

    return results


if __name__ == "__main__":
    print("Testing Hierarchical Intervention Analysis...")
    print("="*60)

    # Simple test with synthetic data
    np.random.seed(42)
    n = 500

    # Synthetic features from 3 "tables"
    X = pd.DataFrame({
        'plant_capacity': np.random.normal(100, 30, n),
        'plant_efficiency': np.random.normal(0.8, 0.1, n),
        'item_weight': np.random.exponential(5, n),
        'item_price': np.random.lognormal(3, 0.5, n),
        'customer_age': np.random.normal(40, 15, n),
    })

    # Target depends mainly on plant_capacity
    y = (X['plant_capacity'] * 0.5 +
         X['item_weight'] * 2 +
         np.random.normal(0, 10, n))

    col_to_fk = {
        'plant_capacity': 'PLANT',
        'plant_efficiency': 'PLANT',
        'item_weight': 'ITEM',
        'item_price': 'ITEM',
        'customer_age': 'CUSTOMER'
    }

    # Simple ensemble for uncertainty
    from ensemble_lgbm import train_lgbm_ensemble

    ensemble = train_lgbm_ensemble(X.values, y, n_models=5)

    def get_unc(X_in):
        if hasattr(X_in, 'values'):
            X_in = X_in.values
        return ensemble.get_uncertainty(X_in).mean()

    # Run analysis
    analyzer = HierarchicalInterventionAnalyzer(get_unc)
    results = analyzer.run_full_analysis(X, col_to_fk, n_bootstrap=30)
    analyzer.print_report(results, "TEST: Synthetic Data")

    print("\n✓ Test passed!")
