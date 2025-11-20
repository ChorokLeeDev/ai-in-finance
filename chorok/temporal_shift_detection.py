"""
Temporal Distribution Shift Detection for SALT Dataset
========================================================

**Research Goal**: Detect distribution shift in raw data WITHOUT using ML predictions.
This is Phase 1 of the UQ-based shift detection research.

**Key Insight**: Predictions are NOT the goal. We first quantify distribution shift
using traditional statistical methods on raw label distributions. Later (Phase 3),
we compare this with epistemic uncertainty to validate our hypothesis:
    "Epistemic uncertainty increases when distribution shifts"

**Metrics Used**:
1. Population Stability Index (PSI) - Overall distribution drift magnitude
   - PSI < 0.1: No significant shift
   - 0.1 ≤ PSI < 0.2: Moderate shift (investigate)
   - PSI ≥ 0.2: Significant shift (model needs retraining)

2. Chi-square test - Statistical significance of categorical distribution change
   - p < 0.05: Statistically significant shift

3. Jensen-Shannon divergence - Symmetric measure of distribution distance
   - Range [0, 1]: 0 = identical, 1 = completely different

**Dataset**: SALT (SAP Autocomplete Logging) with COVID-19 natural experiment
- Train: 2018-01 to 2020-02 (pre-COVID baseline)
- Val: 2020-02 to 2020-07 (COVID onset)
- Test: 2020-07 to 2021-01 (COVID impact)

**Output**: Quantified shift metrics that will be correlated with epistemic uncertainty
in Phase 4 (compare_shift_uncertainty.py)

Author: ChorokLeeDev
Created: 2025-01-18
Last Updated: 2025-01-20
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
from tqdm import tqdm

from relbench.tasks import get_task

warnings.filterwarnings('ignore')

# SALT task names
SALT_TASKS = [
    "item-plant",
    "item-shippoint",
    "item-incoterms",
    "sales-office",
    "sales-group",
    "sales-payterms",
    "sales-shipcond",
    "sales-incoterms",
]

# COVID-19 temporal markers
VAL_TIMESTAMP = pd.Timestamp("2020-02-01")
TEST_TIMESTAMP = pd.Timestamp("2020-07-01")


def load_salt_task_data(task_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Load train/val/test splits and determine target column."""
    from relbench.datasets import get_dataset

    # Load task to get target column and entity table
    task = get_task("rel-salt", task_name, download=False)
    target_col = task.target_col
    entity_table_name = task.entity_table

    # Load dataset to get the full database
    dataset = get_dataset("rel-salt", download=False)
    db = dataset.get_db(upto_test_timestamp=False)  # Include test period data

    # Get the entity table (salesdocument or salesdocumentitem)
    entity_table = db.table_dict[entity_table_name]
    df_full = entity_table.df.copy()

    # Add temporal column if needed
    if 'CREATIONTIMESTAMP' not in df_full.columns:
        # For salesdocument, CREATIONTIMESTAMP should already exist
        # For salesdocumentitem, we might need to join
        print(f"Warning: CREATIONTIMESTAMP not in {entity_table_name}")

    # Split by temporal periods
    train_df = df_full[df_full['CREATIONTIMESTAMP'] < VAL_TIMESTAMP].copy()
    val_df = df_full[(df_full['CREATIONTIMESTAMP'] >= VAL_TIMESTAMP) &
                     (df_full['CREATIONTIMESTAMP'] < TEST_TIMESTAMP)].copy()
    test_df = df_full[df_full['CREATIONTIMESTAMP'] >= TEST_TIMESTAMP].copy()

    # Verify target column exists
    if target_col not in df_full.columns:
        print(f"ERROR: Target column {target_col} not found in {entity_table_name}")
        print(f"Available columns: {df_full.columns.tolist()}")

    return train_df, val_df, test_df, target_col


def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).

    PSI measures the shift in a variable's distribution between two samples.
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate change
    PSI >= 0.2: Significant change

    Args:
        expected: Reference distribution (e.g., train)
        actual: Comparison distribution (e.g., test)
        bins: Number of bins for discretization

    Returns:
        PSI value
    """
    # For categorical variables, use value counts
    if expected.dtype == 'object' or pd.api.types.is_categorical_dtype(expected):
        expected_pct = expected.value_counts(normalize=True)
        actual_pct = actual.value_counts(normalize=True)

        # Align indices (handle missing categories)
        all_categories = set(expected_pct.index) | set(actual_pct.index)
        expected_pct = expected_pct.reindex(all_categories, fill_value=1e-6)
        actual_pct = actual_pct.reindex(all_categories, fill_value=1e-6)

    else:
        # For numerical variables, bin them
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates

        expected_binned = pd.cut(expected, bins=breakpoints, include_lowest=True, duplicates='drop')
        actual_binned = pd.cut(actual, bins=breakpoints, include_lowest=True, duplicates='drop')

        expected_pct = expected_binned.value_counts(normalize=True)
        actual_pct = actual_binned.value_counts(normalize=True)

        # Align indices
        all_bins = set(expected_pct.index) | set(actual_pct.index)
        expected_pct = expected_pct.reindex(all_bins, fill_value=1e-6)
        actual_pct = actual_pct.reindex(all_bins, fill_value=1e-6)

    # Calculate PSI
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return float(psi_value)


def calculate_chi_square(expected: pd.Series, actual: pd.Series) -> Tuple[float, float]:
    """
    Calculate Chi-square test for categorical distribution shift.

    Returns:
        (chi2_statistic, p_value)
    """
    # Get value counts
    expected_counts = expected.value_counts()
    actual_counts = actual.value_counts()

    # Align categories
    all_categories = sorted(set(expected_counts.index) | set(actual_counts.index))
    expected_counts = expected_counts.reindex(all_categories, fill_value=0)
    actual_counts = actual_counts.reindex(all_categories, fill_value=0)

    # Create contingency table
    contingency = pd.DataFrame({
        'expected': expected_counts,
        'actual': actual_counts
    }).T

    # Chi-square test
    chi2, p_value, dof, expected_freq = stats.chi2_contingency(contingency)

    return float(chi2), float(p_value)


def calculate_ks_test(expected: pd.Series, actual: pd.Series) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov test for continuous distribution shift.

    Returns:
        (ks_statistic, p_value)
    """
    ks_stat, p_value = stats.ks_2samp(expected, actual)
    return float(ks_stat), float(p_value)


def calculate_js_divergence(p: pd.Series, q: pd.Series) -> float:
    """
    Calculate Jensen-Shannon divergence between two distributions.

    JSD is bounded [0, 1], where 0 = identical, 1 = completely different.

    Args:
        p: First distribution (e.g., train label distribution)
        q: Second distribution (e.g., test label distribution)

    Returns:
        JS divergence value
    """
    # Normalize to probability distributions
    p_dist = p.value_counts(normalize=True)
    q_dist = q.value_counts(normalize=True)

    # Align categories
    all_categories = sorted(set(p_dist.index) | set(q_dist.index))
    p_dist = p_dist.reindex(all_categories, fill_value=1e-10)
    q_dist = q_dist.reindex(all_categories, fill_value=1e-10)

    # Calculate JS divergence
    js_div = distance.jensenshannon(p_dist, q_dist)

    return float(js_div)


def detect_label_shift(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                       target_col: str) -> Dict:
    """Detect label distribution shift across splits."""
    if target_col not in train_df.columns:
        return {"error": f"{target_col} not found in train_df. Available: {train_df.columns.tolist()}"}
    if target_col not in val_df.columns:
        return {"error": f"{target_col} not found in val_df. Available: {val_df.columns.tolist()}"}
    if target_col not in test_df.columns:
        return {"error": f"{target_col} not found in test_df. Available: {test_df.columns.tolist()}"}

    results = {
        "target_column": target_col,
        "train_classes": int(train_df[target_col].nunique()),
        "val_classes": int(val_df[target_col].nunique()),
        "test_classes": int(test_df[target_col].nunique()),
    }

    # PSI: Train vs Val
    try:
        psi_train_val = calculate_psi(train_df[target_col], val_df[target_col])
        results["psi_train_val"] = psi_train_val
        results["psi_train_val_interpretation"] = interpret_psi(psi_train_val)
    except Exception as e:
        results["psi_train_val"] = None
        results["psi_train_val_error"] = str(e)

    # PSI: Train vs Test
    try:
        psi_train_test = calculate_psi(train_df[target_col], test_df[target_col])
        results["psi_train_test"] = psi_train_test
        results["psi_train_test_interpretation"] = interpret_psi(psi_train_test)
    except Exception as e:
        results["psi_train_test"] = None
        results["psi_train_test_error"] = str(e)

    # Chi-square: Train vs Val
    try:
        chi2_train_val, p_val_train_val = calculate_chi_square(train_df[target_col], val_df[target_col])
        results["chi2_train_val"] = chi2_train_val
        results["chi2_train_val_pvalue"] = p_val_train_val
        results["chi2_train_val_significant"] = p_val_train_val < 0.05
    except Exception as e:
        results["chi2_train_val"] = None
        results["chi2_train_val_error"] = str(e)

    # Chi-square: Train vs Test
    try:
        chi2_train_test, p_val_train_test = calculate_chi_square(train_df[target_col], test_df[target_col])
        results["chi2_train_test"] = chi2_train_test
        results["chi2_train_test_pvalue"] = p_val_train_test
        results["chi2_train_test_significant"] = p_val_train_test < 0.05
    except Exception as e:
        results["chi2_train_test"] = None
        results["chi2_train_test_error"] = str(e)

    # JS Divergence: Train vs Val
    try:
        js_train_val = calculate_js_divergence(train_df[target_col], val_df[target_col])
        results["js_train_val"] = js_train_val
    except Exception as e:
        results["js_train_val"] = None
        results["js_train_val_error"] = str(e)

    # JS Divergence: Train vs Test
    try:
        js_train_test = calculate_js_divergence(train_df[target_col], test_df[target_col])
        results["js_train_test"] = js_train_test
    except Exception as e:
        results["js_train_test"] = None
        results["js_train_test_error"] = str(e)

    return results


def interpret_psi(psi_value: float) -> str:
    """Interpret PSI value."""
    if psi_value < 0.1:
        return "No significant shift"
    elif psi_value < 0.2:
        return "Moderate shift"
    else:
        return "Significant shift (Action required)"


def analyze_task_shift(task_name: str) -> Dict:
    """Analyze distribution shift for a single task."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {task_name}")
    print(f"{'='*60}")

    # Load data
    train_df, val_df, test_df, target_col = load_salt_task_data(task_name)

    print(f"Target column: {target_col}")
    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples: {len(val_df):,}")
    print(f"Test samples: {len(test_df):,}")

    # Detect label shift
    shift_metrics = detect_label_shift(train_df, val_df, test_df, target_col)

    # Print summary
    print("\n## Label Shift Detection ##")
    if 'error' in shift_metrics:
        print(f"ERROR: {shift_metrics['error']}")
    else:
        psi_tv = shift_metrics.get('psi_train_val')
        psi_tt = shift_metrics.get('psi_train_test')
        psi_tv_str = f"{psi_tv:.4f}" if psi_tv is not None else 'N/A'
        psi_tt_str = f"{psi_tt:.4f}" if psi_tt is not None else 'N/A'
        print(f"PSI (Train->Val): {psi_tv_str} - {shift_metrics.get('psi_train_val_interpretation', 'N/A')}")
        print(f"PSI (Train->Test): {psi_tt_str} - {shift_metrics.get('psi_train_test_interpretation', 'N/A')}")

        chi2_tv = shift_metrics.get('chi2_train_val')
        chi2_tt = shift_metrics.get('chi2_train_test')
        pval_tv = shift_metrics.get('chi2_train_val_pvalue')
        pval_tt = shift_metrics.get('chi2_train_test_pvalue')
        chi2_tv_str = f"{chi2_tv:.2f}" if chi2_tv is not None else 'N/A'
        chi2_tt_str = f"{chi2_tt:.2f}" if chi2_tt is not None else 'N/A'
        pval_tv_str = f"{pval_tv:.4e}" if pval_tv is not None else 'N/A'
        pval_tt_str = f"{pval_tt:.4e}" if pval_tt is not None else 'N/A'
        print(f"Chi^2 (Train vs Val): {chi2_tv_str}, p={pval_tv_str} {'*' if shift_metrics.get('chi2_train_val_significant') else ''}")
        print(f"Chi^2 (Train vs Test): {chi2_tt_str}, p={pval_tt_str} {'*' if shift_metrics.get('chi2_train_test_significant') else ''}")

        js_tv = shift_metrics.get('js_train_val')
        js_tt = shift_metrics.get('js_train_test')
        js_tv_str = f"{js_tv:.4f}" if js_tv is not None else 'N/A'
        js_tt_str = f"{js_tt:.4f}" if js_tt is not None else 'N/A'
        print(f"JS Div (Train vs Val): {js_tv_str}")
        print(f"JS Div (Train vs Test): {js_tt_str}")

    results = {
        "task_name": task_name,
        "shift_metrics": shift_metrics,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Temporal distribution shift detection for SALT dataset")
    parser.add_argument("--tasks", nargs="+", default=SALT_TASKS,
                       help="List of tasks to analyze (default: all 8 tasks)")
    parser.add_argument("--output", type=str, default="chorok/results/shift_detection.json",
                       help="Output JSON file path")

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# SALT Dataset Temporal Shift Detection")
    print(f"{'#'*60}")
    print(f"Tasks to analyze: {args.tasks}")
    print(f"Output file: {args.output}")

    # Analyze each task
    all_results = {}
    for task_name in tqdm(args.tasks, desc="Analyzing tasks"):
        try:
            results = analyze_task_shift(task_name)
            all_results[task_name] = results
        except Exception as e:
            print(f"Error analyzing {task_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[task_name] = {"error": str(e)}

    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Shift detection complete! Results saved to: {args.output}")
    print(f"{'='*60}")

    # Summary comparison
    print("\n## COVID-19 Impact Summary (Train→Test Shift) ##")
    print(f"{'Task':<20} {'PSI':>8} {'Chi²':>12} {'JS Div':>8} {'Severity'}")
    print("=" * 70)

    # Sort tasks by shift severity (using PSI as primary metric)
    task_shifts = []
    for task_name, results in all_results.items():
        if "error" in results:
            continue
        metrics = results['shift_metrics']
        psi = metrics.get('psi_train_test', 0)
        chi2 = metrics.get('chi2_train_test', 0)
        js = metrics.get('js_train_test', 0)

        task_shifts.append((task_name, psi, chi2, js))

    task_shifts.sort(key=lambda x: x[1], reverse=True)  # Sort by PSI

    for task_name, psi, chi2, js in task_shifts:
        severity = interpret_psi(psi)
        print(f"{task_name:<20} {psi:>8.4f} {chi2:>12.2f} {js:>8.4f}  {severity}")

    print("\n* Chi² significant at p < 0.05")
    print("PSI Interpretation: <0.1 (No shift), 0.1-0.2 (Moderate), >=0.2 (Significant)")


if __name__ == "__main__":
    main()
