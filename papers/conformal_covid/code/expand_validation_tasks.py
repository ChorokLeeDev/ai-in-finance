#!/usr/bin/env python3
"""
Expand SHAP concentration analysis to 20+ tasks to address n=8 statistical power issue.

This script:
1. Runs conformal prediction on all temporal tasks from rel-trial, rel-f1, rel-amazon, rel-stack
2. Computes SHAP concentration for each task
3. Tests correlation with coverage degradation on larger sample (n=20+)
4. Validates whether 40% threshold holds across domains
"""

from relbench.tasks import get_task_names, get_task
from relbench.datasets import get_dataset
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
from tqdm import tqdm

# Priority datasets with temporal structure
DATASETS_TO_TEST = {
    'rel-trial': ['study-outcome', 'study-adverse', 'site-success',
                  'condition-sponsor-run', 'site-sponsor-run'],
    'rel-f1': ['driver-position', 'driver-dnf', 'driver-top3',
               'results-position', 'qualifying-position'],
    'rel-amazon': ['user-churn', 'user-ltv', 'item-churn', 'item-ltv'],  # Start with 4
    'rel-stack': ['user-engagement', 'post-votes', 'user-badge']  # Start with 3
}

def compute_shap_concentration(model, X_val, num_samples=10000):
    """
    Compute SHAP importance concentration (top feature / total importance).

    Returns:
        concentration: float, percentage of importance in top feature
        top_feature: str, name of top feature
        all_importances: dict, SHAP values for all features
    """
    # Subsample to reduce computation
    if len(X_val) > num_samples:
        X_subsample = X_val.sample(num_samples, random_state=42)
    else:
        X_subsample = X_val

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_subsample)

    # For multiclass, average across classes
    if isinstance(shap_values, list):
        shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_values = np.abs(shap_values)

    # Compute mean importance per feature
    feature_importance = np.mean(shap_values, axis=0)
    total_importance = np.sum(feature_importance)

    # Find top feature
    top_idx = np.argmax(feature_importance)
    top_feature = X_val.columns[top_idx]
    concentration = (feature_importance[top_idx] / total_importance) * 100

    # All importances
    all_importances = {
        col: feature_importance[i]
        for i, col in enumerate(X_val.columns)
    }

    return concentration, top_feature, all_importances


def run_conformal_experiment(dataset_name, task_name, alpha=0.1, n_seeds=5):
    """
    Run conformal prediction experiment on one task.

    Returns:
        result: dict with coverage, concentration, etc.
    """
    print(f"\n{'='*60}")
    print(f"Running: {dataset_name} / {task_name}")
    print(f"{'='*60}")

    try:
        # Load data
        dataset = get_dataset(dataset_name, download=True)
        task = get_task(dataset_name, task_name, download=True)
        db = dataset.get_db()

        train_table = task.get_table("train")
        val_table = task.get_table("val")
        test_table = task.get_table("test")

        # Basic feature extraction (simplified - should use proper feature engineering)
        # This is a placeholder - real implementation needs proper feature extraction
        print("Feature extraction not implemented - need to add proper feature engineering")

        return None

    except Exception as e:
        print(f"ERROR on {dataset_name}/{task_name}: {e}")
        return None


def main():
    """Run expanded validation across 20+ tasks."""

    results = []

    for dataset_name, task_names in DATASETS_TO_TEST.items():
        for task_name in task_names:
            result = run_conformal_experiment(dataset_name, task_name)
            if result is not None:
                results.append(result)

    # Analyze results
    if len(results) > 0:
        df_results = pd.DataFrame(results)
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS: n={len(results)} tasks")
        print(f"{'='*60}")
        print(df_results)

        # Test correlation
        from scipy.stats import spearmanr
        if len(results) >= 8:
            rho, pval = spearmanr(df_results['concentration'], df_results['coverage_drop'])
            print(f"\nSpearman correlation:")
            print(f"  n={len(results)}")
            print(f"  ρ={rho:.3f}, p={pval:.4f}")

            # Compare to original n=8
            print(f"\nComparison to original (n=8, ρ=0.71, p=0.047):")
            if len(results) > 8:
                print(f"  Improved statistical power: {len(results)} tasks")
            if pval < 0.01:
                print(f"  Stronger significance: p={pval:.4f} < 0.01")

        # Save results
        df_results.to_csv('expanded_validation_results.csv', index=False)
        print(f"\nResults saved to: expanded_validation_results.csv")

    else:
        print("No results collected - feature engineering needed")


if __name__ == "__main__":
    print("""
    ========================================================================
    ADDRESSING ISSUE #1: Statistical Power (n=8)
    ========================================================================

    Current state: n=8 tasks, ρ=0.71, p=0.047 (barely significant)
    Goal: Expand to n=20+ tasks to strengthen statistical power

    This script will:
    1. Run conformal prediction on 17 additional tasks
    2. Compute SHAP concentration for each
    3. Test if 40% threshold generalizes
    4. Report updated correlation with stronger power

    Status: PLACEHOLDER - Needs feature engineering implementation
    TODO: Add proper feature extraction for each dataset
    ========================================================================
    """)

    # Count available tasks
    total_tasks = sum(len(tasks) for tasks in DATASETS_TO_TEST.values())
    print(f"Target: {total_tasks} tasks across {len(DATASETS_TO_TEST)} datasets")
    print(f"Current: 8 tasks from rel-salt only")
    print(f"Improvement: {total_tasks - 8} additional tasks (+{(total_tasks/8 - 1)*100:.0f}%)\n")

    # Uncomment to run (needs feature engineering first)
    # main()
