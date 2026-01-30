#!/usr/bin/env python3
"""
Analyze Continuous vs Categorical Features for Issue #4

This addresses Issue #4: Validate 40% concentration threshold on continuous features.

For regression tasks, we:
1. Load task data and examine feature types
2. Classify features as continuous vs categorical
3. Compute SHAP concentration separately by feature type
4. Test if 40% threshold holds for continuous features
5. Compare with categorical feature tasks

Target: At least 3 continuous-dominant tasks validated
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pickle

# Add relbench to path
sys.path.insert(0, '/Users/i767700/Github/ai-in-finance')

from relbench.datasets import get_dataset
from relbench.tasks import get_task

# Regression tasks to analyze
# Note: Only including tasks that are in the registry
REGRESSION_TASKS = [
    ('rel-f1', 'driver-position'),
    # ('rel-f1', 'results-position'),  # Not in registry
    # ('rel-f1', 'qualifying-position'),  # Not in registry
    ('rel-trial', 'study-adverse'),
    ('rel-trial', 'site-success'),
]


def classify_feature_type(series: pd.Series, unique_threshold: int = 20) -> str:
    """
    Classify feature as categorical or continuous.

    Heuristic:
    - If dtype is object or category → categorical
    - If numeric and unique values > threshold → continuous
    - If numeric and unique values ≤ threshold → categorical (ordinal)

    Args:
        series: Feature column
        unique_threshold: Max unique values for categorical (default: 20)

    Returns:
        'categorical' or 'continuous'
    """
    dtype = series.dtype

    # Object or category dtype → categorical
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
        return 'categorical'

    # Boolean → categorical
    if pd.api.types.is_bool_dtype(dtype):
        return 'categorical'

    # Numeric
    if pd.api.types.is_numeric_dtype(dtype):
        n_unique = series.nunique()

        # Many unique values → continuous
        if n_unique > unique_threshold:
            return 'continuous'
        else:
            # Few unique values → categorical (ordinal)
            return 'categorical'

    # Default
    return 'categorical'


def analyze_task_features(dataset_name: str, task_name: str) -> Dict:
    """
    Analyze feature types for one task.

    Returns:
        Dictionary with:
        - dataset: dataset name
        - task: task name
        - n_features: total number of features
        - n_categorical: number of categorical features
        - n_continuous: number of continuous features
        - categorical_features: list of categorical feature names
        - continuous_features: list of continuous feature names
        - feature_types: dict mapping feature → type
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {dataset_name} / {task_name}")
    print(f"{'='*70}")

    try:
        # Load dataset and task
        dataset = get_dataset(dataset_name, download=True)
        task = get_task(dataset_name, task_name, download=True)

        # Get training data
        train_table = task.get_table('train')
        db = dataset.get_db()

        # Get features (exclude target and timestamp columns)
        target_col = task.target_col
        time_col = task.time_col
        entity_col = task.entity_col

        exclude_cols = {target_col, time_col, entity_col}
        feature_cols = [col for col in train_table.df.columns if col not in exclude_cols]

        print(f"\nTotal columns: {len(train_table.df.columns)}")
        print(f"Excluded: {exclude_cols}")
        print(f"Features: {len(feature_cols)}")

        # Classify each feature
        feature_types = {}
        categorical_features = []
        continuous_features = []

        for col in feature_cols:
            series = train_table.df[col]
            feat_type = classify_feature_type(series)
            feature_types[col] = feat_type

            if feat_type == 'categorical':
                categorical_features.append(col)
            else:
                continuous_features.append(col)

            # Print feature info
            n_unique = series.nunique()
            n_null = series.isna().sum()
            print(f"  {col:30} {feat_type:12} {series.dtype!s:15} unique={n_unique:5} null={n_null:5}")

        print(f"\n✓ Categorical features: {len(categorical_features)}")
        print(f"✓ Continuous features: {len(continuous_features)}")

        dominant_type = 'categorical' if len(categorical_features) > len(continuous_features) else 'continuous'
        print(f"✓ Dominant type: {dominant_type}")

        return {
            'dataset': dataset_name,
            'task': task_name,
            'n_features': len(feature_cols),
            'n_categorical': len(categorical_features),
            'n_continuous': len(continuous_features),
            'categorical_features': categorical_features,
            'continuous_features': continuous_features,
            'feature_types': feature_types,
            'dominant_type': dominant_type
        }

    except Exception as e:
        print(f"❌ Error analyzing {dataset_name}/{task_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_shap_concentration(dataset_name: str, task_name: str, results_dir: Path) -> Dict:
    """
    Load SHAP concentration data if available.

    Returns:
        Dictionary with concentration data or None if not available
    """
    # Check if SHAP results exist
    shap_file = results_dir / 'shap' / f'shap_{dataset_name}_{task_name}.pkl'

    if not shap_file.exists():
        return None

    try:
        with open(shap_file, 'rb') as f:
            shap_data = pickle.load(f)

        # Extract concentration
        top_features_val = shap_data.get('top_features_val', [])
        if not top_features_val:
            return None

        total_importance = sum(imp for _, imp in top_features_val)
        top_importance = top_features_val[0][1]
        concentration = (top_importance / total_importance) * 100

        return {
            'concentration': concentration,
            'top_feature': top_features_val[0][0],
            'top_importance': top_importance,
            'total_importance': total_importance
        }

    except Exception as e:
        print(f"  ⚠️ Could not load SHAP data: {e}")
        return None


def generate_summary_table(results: List[Dict]) -> str:
    """Generate LaTeX summary table."""

    latex = r"""\begin{table}[t]
\centering
\caption{Feature Type Analysis for Regression Tasks}
\label{tab:feature_types}
\small
\begin{tabular}{lcccp{3cm}}
\toprule
Task & Cat & Cont & Dominant & Conc (\%) \\
\midrule
"""

    for res in results:
        if res is None:
            continue

        task_short = res['task']
        n_cat = res['n_categorical']
        n_cont = res['n_continuous']
        dominant = res['dominant_type'][:4]  # Cat or Cont

        latex += f"{task_short} & {n_cat} & {n_cont} & {dominant} & "
        latex += "TBD"  # Concentration to be computed
        latex += " \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\vspace{2mm}

\raggedright
\footnotesize
Cat = categorical features, Cont = continuous features.
Concentration computed after SHAP analysis.
\end{table}
"""

    return latex


def main():
    """Main execution."""
    os_chdir = False
    try:
        import os
        os.chdir('/Users/i767700/Github/ai-in-finance/papers/conformal_covid')
        os_chdir = True
    except:
        pass

    print("="*80)
    print("ISSUE #4: CONTINUOUS FEATURE VALIDATION")
    print("="*80)
    print("\nStep 1: Identify feature types (categorical vs continuous)")
    print("Target: At least 3 continuous-dominant tasks\n")

    # Analyze all regression tasks
    all_results = []
    for dataset_name, task_name in REGRESSION_TASKS:
        result = analyze_task_features(dataset_name, task_name)
        if result:
            all_results.append(result)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nTotal tasks analyzed: {len(all_results)}")

    categorical_dominant = [r for r in all_results if r['dominant_type'] == 'categorical']
    continuous_dominant = [r for r in all_results if r['dominant_type'] == 'continuous']

    print(f"Categorical-dominant: {len(categorical_dominant)}")
    for r in categorical_dominant:
        print(f"  - {r['task']:25} ({r['n_categorical']} cat, {r['n_continuous']} cont)")

    print(f"\nContinuous-dominant: {len(continuous_dominant)}")
    for r in continuous_dominant:
        print(f"  - {r['task']:25} ({r['n_categorical']} cat, {r['n_continuous']} cont)")

    # Check if we have enough continuous-dominant tasks
    if len(continuous_dominant) >= 3:
        print(f"\n✓ SUCCESS: Found {len(continuous_dominant)} continuous-dominant tasks (target: ≥3)")
    else:
        print(f"\n⚠️ WARNING: Only {len(continuous_dominant)} continuous-dominant tasks (target: ≥3)")
        print("  May need to include mixed tasks for analysis")

    # Save results
    if os_chdir:
        output_dir = Path('results/feature_types')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        csv_file = output_dir / 'feature_type_analysis.csv'
        rows = []
        for res in all_results:
            rows.append({
                'dataset': res['dataset'],
                'task': res['task'],
                'n_features': res['n_features'],
                'n_categorical': res['n_categorical'],
                'n_continuous': res['n_continuous'],
                'dominant_type': res['dominant_type']
            })

        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        print(f"\n✓ CSV saved to: {csv_file}")

        # Save detailed results as pickle
        pkl_file = output_dir / 'feature_type_analysis.pkl'
        with open(pkl_file, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"✓ Detailed results saved to: {pkl_file}")

        # Generate LaTeX table
        latex_table = generate_summary_table(all_results)
        latex_file = output_dir / 'table_feature_types.tex'
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        print(f"✓ LaTeX table saved to: {latex_file}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Run SHAP analysis on these regression tasks")
    print("2. Compute concentration separately for categorical vs continuous features")
    print("3. Test if 40% threshold holds for continuous features")
    print("4. Update paper with findings")

    return all_results


if __name__ == "__main__":
    results = main()
