"""
Hierarchical Uncertainty Attribution Framework
===============================================

Three-level drill-down for actionable insights:
- Level 1: FK (Process) - "SHIPPING is 27% of uncertainty"
- Level 2: Feature (Attribute) - "SHIPPINGCONDITION is 60% within SHIPPING"
- Level 3: Entity (Value) - "EXPRESS shipments are most uncertain"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from data_loader import load_salt_data
from ensemble import train_ensemble, compute_uncertainty


@dataclass
class AttributionResult:
    """Result of attribution at any level."""
    level: str  # 'fk', 'feature', 'entity'
    parent: Optional[str]  # Parent group (None for Level 1)
    contributions: Dict[str, float]  # name -> percentage
    details: Dict  # Additional info


class HierarchicalAttribution:
    """
    Hierarchical uncertainty attribution framework.

    Usage:
        ha = HierarchicalAttribution(models, X, col_to_fk)

        # Level 1: FK-level
        fk_result = ha.attribute_level1()

        # Level 2: Drill into SHIPPING
        feature_result = ha.attribute_level2(fk='SHIPPING')

        # Level 3: Drill into SHIPPINGCONDITION
        entity_result = ha.attribute_level3(feature='SHIPPINGCONDITION')
    """

    def __init__(
        self,
        models: List,
        X: pd.DataFrame,
        col_to_fk: Dict[str, str],
        n_perturbations: int = 10
    ):
        self.models = models
        self.X = X
        self.X_np = X.values
        self.col_to_fk = col_to_fk
        self.feature_cols = list(X.columns)
        self.n_perturbations = n_perturbations

        # Precompute FK groups
        self.fk_to_cols = defaultdict(list)
        for col, fk in col_to_fk.items():
            self.fk_to_cols[fk].append(col)

        # Baseline uncertainty
        self.baseline = compute_uncertainty(
            models, self.X_np, method='entropy'
        ).mean()

    def _compute_delta(self, col_indices: List[int]) -> float:
        """Compute uncertainty increase when columns are perturbed."""
        deltas = []
        for _ in range(self.n_perturbations):
            X_noisy = self.X_np.copy()
            for idx in col_indices:
                X_noisy[:, idx] = np.random.permutation(X_noisy[:, idx])
            noisy = compute_uncertainty(self.models, X_noisy, method='entropy').mean()
            deltas.append(noisy - self.baseline)
        return np.mean(deltas)

    def _normalize(self, results: Dict[str, float]) -> Dict[str, float]:
        """Normalize to percentages."""
        total = sum(max(0, v) for v in results.values())
        if total > 0:
            return {k: max(0, v) / total * 100 for k, v in results.items()}
        return {k: 0 for k in results}

    # =========================================================================
    # Level 1: FK (Process) Attribution
    # =========================================================================

    def attribute_level1(self) -> AttributionResult:
        """
        Level 1: Attribute uncertainty to FK groups (processes).

        Returns:
            AttributionResult with FK contributions
        """
        print("\n[Level 1] FK Attribution")
        print("-" * 40)
        print(f"Baseline uncertainty: {self.baseline:.4f}")

        results = {}
        for fk, cols in self.fk_to_cols.items():
            col_indices = [self.feature_cols.index(c) for c in cols if c in self.feature_cols]
            if not col_indices:
                continue

            delta = self._compute_delta(col_indices)
            results[fk] = delta
            print(f"  {fk}: delta = {delta:+.4f}")

        contributions = self._normalize(results)

        print(f"\nContributions:")
        for fk, pct in sorted(contributions.items(), key=lambda x: -x[1]):
            print(f"  {fk}: {pct:.1f}%")

        return AttributionResult(
            level='fk',
            parent=None,
            contributions=contributions,
            details={'raw_deltas': results, 'baseline': self.baseline}
        )

    # =========================================================================
    # Level 2: Feature Attribution (within FK)
    # =========================================================================

    def attribute_level2(self, fk: str) -> AttributionResult:
        """
        Level 2: Attribute uncertainty to features within a specific FK.

        Args:
            fk: FK group to drill into

        Returns:
            AttributionResult with feature contributions within FK
        """
        if fk not in self.fk_to_cols:
            raise ValueError(f"Unknown FK: {fk}")

        cols = self.fk_to_cols[fk]
        print(f"\n[Level 2] Feature Attribution within {fk}")
        print("-" * 40)
        print(f"Features in {fk}: {cols}")

        if len(cols) == 1:
            print(f"  Only one feature in {fk}, attribution = 100%")
            return AttributionResult(
                level='feature',
                parent=fk,
                contributions={cols[0]: 100.0},
                details={'single_feature': True}
            )

        results = {}
        for col in cols:
            col_idx = self.feature_cols.index(col)
            delta = self._compute_delta([col_idx])
            results[col] = delta
            print(f"  {col}: delta = {delta:+.4f}")

        contributions = self._normalize(results)

        print(f"\nContributions within {fk}:")
        for col, pct in sorted(contributions.items(), key=lambda x: -x[1]):
            print(f"  {col}: {pct:.1f}%")

        return AttributionResult(
            level='feature',
            parent=fk,
            contributions=contributions,
            details={'raw_deltas': results}
        )

    # =========================================================================
    # Level 3: Entity Attribution (within Feature)
    # =========================================================================

    def attribute_level3(
        self,
        feature: str,
        top_k: int = 5
    ) -> AttributionResult:
        """
        Level 3: Analyze which entity values cause high uncertainty.

        Args:
            feature: Feature column to analyze
            top_k: Number of top uncertain values to return

        Returns:
            AttributionResult with entity-level insights
        """
        if feature not in self.feature_cols:
            raise ValueError(f"Unknown feature: {feature}")

        print(f"\n[Level 3] Entity Attribution for {feature}")
        print("-" * 40)

        col_idx = self.feature_cols.index(feature)

        # Get unique values
        unique_vals = self.X[feature].unique()
        print(f"Unique values: {len(unique_vals)}")

        # Compute per-sample uncertainty
        sample_uncertainty = compute_uncertainty(
            self.models, self.X_np, method='entropy'
        )

        # Group by entity value
        entity_uncertainty = {}
        entity_counts = {}

        for val in unique_vals:
            mask = self.X[feature] == val
            if mask.sum() == 0:
                continue

            mean_unc = sample_uncertainty[mask].mean()
            entity_uncertainty[val] = mean_unc
            entity_counts[val] = mask.sum()

        # Sort by uncertainty
        sorted_entities = sorted(
            entity_uncertainty.items(),
            key=lambda x: -x[1]
        )

        print(f"\nTop {top_k} high-uncertainty values:")
        for val, unc in sorted_entities[:top_k]:
            count = entity_counts[val]
            print(f"  {feature}={val}: uncertainty={unc:.4f} (n={count})")

        print(f"\nBottom {top_k} low-uncertainty values:")
        for val, unc in sorted_entities[-top_k:]:
            count = entity_counts[val]
            print(f"  {feature}={val}: uncertainty={unc:.4f} (n={count})")

        # Compute relative contribution
        baseline_unc = np.mean(list(entity_uncertainty.values()))
        contributions = {
            val: ((unc - baseline_unc) / baseline_unc * 100) if baseline_unc > 0 else 0
            for val, unc in entity_uncertainty.items()
        }

        return AttributionResult(
            level='entity',
            parent=feature,
            contributions=contributions,
            details={
                'entity_uncertainty': entity_uncertainty,
                'entity_counts': entity_counts,
                'baseline_uncertainty': baseline_unc,
                'high_uncertainty': sorted_entities[:top_k],
                'low_uncertainty': sorted_entities[-top_k:]
            }
        )

    # =========================================================================
    # Full Drill-Down
    # =========================================================================

    def full_drilldown(self, top_k_fk: int = 3) -> Dict:
        """
        Perform full hierarchical drill-down.

        Args:
            top_k_fk: Number of top FKs to drill into

        Returns:
            Nested dict with all attribution results
        """
        print("=" * 60)
        print("HIERARCHICAL UNCERTAINTY ATTRIBUTION")
        print("=" * 60)

        results = {'level1': {}, 'level2': {}, 'level3': {}}

        # Level 1
        l1_result = self.attribute_level1()
        results['level1'] = l1_result.contributions

        # Get top FKs
        top_fks = sorted(
            l1_result.contributions.items(),
            key=lambda x: -x[1]
        )[:top_k_fk]

        # Level 2 for each top FK
        for fk, pct in top_fks:
            if pct <= 0:
                continue

            l2_result = self.attribute_level2(fk)
            results['level2'][fk] = l2_result.contributions

            # Level 3 for top feature in this FK
            if l2_result.contributions:
                top_feature = max(
                    l2_result.contributions.items(),
                    key=lambda x: x[1]
                )[0]

                l3_result = self.attribute_level3(top_feature)
                results['level3'][top_feature] = {
                    'high': l3_result.details['high_uncertainty'],
                    'low': l3_result.details['low_uncertainty']
                }

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        for fk, pct in top_fks:
            print(f"\n{fk} ({pct:.1f}% of total uncertainty):")

            if fk in results['level2']:
                for feat, feat_pct in sorted(
                    results['level2'][fk].items(),
                    key=lambda x: -x[1]
                )[:3]:
                    print(f"  └─ {feat}: {feat_pct:.1f}%")

                    if feat in results['level3']:
                        high = results['level3'][feat]['high'][:2]
                        for val, unc in high:
                            print(f"      └─ {feat}={val}: high uncertainty ({unc:.3f})")

        return results


def run_hierarchical_analysis(
    task_name: str = 'sales-group',
    sample_size: int = 500,
    n_models: int = 5
):
    """Run full hierarchical analysis."""

    print("Loading data...")
    X, y, feature_cols, col_to_fk = load_salt_data(task_name, sample_size)

    print("Training ensemble...")
    models, X_test, y_test = train_ensemble(X, y, n_models=n_models)

    # Create hierarchical attribution
    ha = HierarchicalAttribution(models, X, col_to_fk)

    # Full drill-down
    results = ha.full_drilldown(top_k_fk=3)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='sales-group')
    parser.add_argument('--sample_size', type=int, default=500)
    parser.add_argument('--n_models', type=int, default=5)
    args = parser.parse_args()

    run_hierarchical_analysis(
        task_name=args.task,
        sample_size=args.sample_size,
        n_models=args.n_models
    )
