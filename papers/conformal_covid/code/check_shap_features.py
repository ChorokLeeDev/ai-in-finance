#!/usr/bin/env python3
"""
Check SHAP Features - Phase 1 (Accurate Version)

Instead of looking at train_table (which only has entity/target),
look at SHAP pickle files to see what features were ACTUALLY used.

This is the ground truth - these are the features the model saw.
"""

import pickle
from pathlib import Path
import pandas as pd
import numpy as np

# SHAP pickle locations (actual naming: shap_dataset_task.pkl)
SHAP_FILES = [
    # rel-salt (n=8)
    'results/shap/shap_rel-salt_sales-group.pkl',
    'results/shap/shap_rel-salt_sales-payterms.pkl',
    'results/shap/shap_rel-salt_sales-shipcond.pkl',
    'results/shap/shap_rel-salt_item-shippoint.pkl',
    'results/shap/shap_rel-salt_item-incoterms.pkl',
    'results/shap/shap_rel-salt_item-plant.pkl',
    'results/shap/shap_rel-salt_sales-incoterms.pkl',
    'results/shap/shap_rel-salt_sales-office.pkl',

    # rel-trial (classification)
    'results/shap/shap_rel-trial_study-outcome.pkl',
    'results/shap/shap_rel-trial_study-adverse.pkl',
    'results/shap/shap_rel-trial_site-success.pkl',

    # rel-f1 (classification)
    'results/shap/shap_rel-f1_driver-dnf.pkl',
]


def classify_feature_from_name(feature_name: str) -> str:
    """
    Heuristic: classify feature by name patterns.

    Categorical indicators:
    - ALL_CAPS (e.g., PRODUCT, PLANT)
    - Ends with _ID, _CODE
    - Contains words: GROUP, OFFICE, TYPE, CATEGORY

    Continuous indicators:
    - Contains: AMOUNT, COUNT, PRICE, RATE, SCORE, VALUE
    - Ends with _NUM, _QTY
    """
    name_upper = feature_name.upper()

    # Categorical patterns
    categorical_patterns = [
        '_ID', '_CODE', 'GROUP', 'OFFICE', 'TYPE', 'CATEGORY',
        'DOCUMENT', 'ORGANIZATION', 'PARTY', 'PLANT', 'POINT',
        'CONDITION', 'TERMS', 'INCOTERMS', 'SPONSOR', 'SITE',
        'DRIVER', 'CONSTRUCTOR', 'CIRCUIT'
    ]

    # Continuous patterns
    continuous_patterns = [
        'AMOUNT', 'COUNT', 'PRICE', 'RATE', 'SCORE', 'VALUE',
        '_NUM', '_QTY', 'QUANTITY', 'TOTAL', 'AVG', 'MEAN',
        'POINTS', 'POSITION', 'LAP', 'TIME', 'SECONDS',
        'ENROLLMENT', 'DURATION', 'AGE', 'DISTANCE'
    ]

    # Check continuous first
    for pattern in continuous_patterns:
        if pattern in name_upper:
            return 'continuous'

    # Check categorical
    for pattern in categorical_patterns:
        if pattern in name_upper:
            return 'categorical'

    # Default: if all caps or short → categorical
    if feature_name.isupper() or len(feature_name) <= 15:
        return 'categorical'

    # Otherwise → continuous
    return 'continuous'


def analyze_shap_file(shap_path: Path) -> dict:
    """Load SHAP pickle and analyze features."""
    task_name = shap_path.stem.replace('_shap', '')

    try:
        with open(shap_path, 'rb') as f:
            shap_data = pickle.load(f)

        # Get feature names
        if 'feature_names' in shap_data:
            features = shap_data['feature_names']
        elif 'shap_values' in shap_data and hasattr(shap_data['shap_values'], 'feature_names'):
            features = shap_data['shap_values'].feature_names
        else:
            # Try to infer from shap_values shape
            if 'shap_values' in shap_data:
                n_features = shap_data['shap_values'].shape[1]
                features = [f'feature_{i}' for i in range(n_features)]
            else:
                return {'task': task_name, 'error': 'No features found'}

        # Classify each feature
        categorical = []
        continuous = []

        for feat in features:
            feat_type = classify_feature_from_name(feat)
            if feat_type == 'continuous':
                continuous.append(feat)
            else:
                categorical.append(feat)

        n_total = len(features)
        n_cat = len(categorical)
        n_cont = len(continuous)
        pct_cont = (n_cont / n_total * 100) if n_total > 0 else 0

        return {
            'task': task_name,
            'n_total': n_total,
            'n_categorical': n_cat,
            'n_continuous': n_cont,
            'pct_continuous': pct_cont,
            'continuous_features': continuous[:10],  # First 10
            'categorical_features': categorical[:10],
            'dominant': 'continuous' if n_cont > n_cat else 'categorical'
        }

    except FileNotFoundError:
        return {'task': task_name, 'error': 'File not found'}
    except Exception as e:
        return {'task': task_name, 'error': str(e)}


def main():
    print("="*80)
    print("SHAP FEATURES ANALYSIS - Phase 1 (Ground Truth)")
    print("="*80)
    print("\nAnalyzing ACTUAL features used in models (from SHAP pickles)")
    print()

    base_path = Path('/Users/i767700/Github/ai-in-finance/papers/conformal_covid')
    results = []

    for shap_file in SHAP_FILES:
        shap_path = base_path / shap_file
        print(f"\n{shap_file:45}", end=" ... ")

        result = analyze_shap_file(shap_path)
        results.append(result)

        if 'error' in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"Features: {result['n_total']:3} = {result['n_categorical']:3} cat + {result['n_continuous']:3} cont ({result['pct_continuous']:5.1f}% cont)")
            if result['n_continuous'] > 5:
                print(f"      Sample continuous: {result['continuous_features'][:3]}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Summary statistics
    df = pd.DataFrame([r for r in results if 'error' not in r])

    if len(df) == 0:
        print("\n❌ NO DATA - All SHAP files had errors")
        return

    total_tasks = len(df)
    tasks_with_continuous = len(df[df['n_continuous'] > 0])
    continuous_dominated = len(df[df['dominant'] == 'continuous'])

    # Tasks with significant continuous features (>30%)
    significant_continuous = len(df[df['pct_continuous'] > 30])

    print(f"\nTotal tasks analyzed: {total_tasks}")
    print(f"Tasks with ANY continuous features: {tasks_with_continuous} ({tasks_with_continuous/total_tasks*100:.1f}%)")
    print(f"Tasks with >30% continuous: {significant_continuous} ({significant_continuous/total_tasks*100:.1f}%)")
    print(f"Tasks DOMINATED by continuous (>50%): {continuous_dominated} ({continuous_dominated/total_tasks*100:.1f}%)")
    print()

    # Show distribution
    print("Distribution by % continuous:")
    for pct_range in [(0, 10), (10, 30), (30, 50), (50, 70), (70, 100)]:
        count = len(df[(df['pct_continuous'] >= pct_range[0]) & (df['pct_continuous'] < pct_range[1])])
        if count > 0:
            tasks = df[(df['pct_continuous'] >= pct_range[0]) & (df['pct_continuous'] < pct_range[1])]['task'].tolist()
            print(f"  {pct_range[0]:3}%-{pct_range[1]:3}%: {count:2} tasks - {tasks}")

    # Show top continuous tasks
    if continuous_dominated > 0:
        print("\nContinuous-dominated tasks:")
        top_cont = df[df['dominant'] == 'continuous'].sort_values('pct_continuous', ascending=False)
        for _, row in top_cont.iterrows():
            print(f"  {row['task']:20} {row['pct_continuous']:5.1f}% continuous ({row['n_continuous']}/{row['n_total']} features)")

    print("\n" + "="*80)
    print("DECISION")
    print("="*80)

    if tasks_with_continuous == 0:
        print("\n❌ NO HOPE - All tasks are 100% categorical")
        print("   Continuous features validation is IMPOSSIBLE")
        print("   Recommendation: STOP, submit paper as-is")

    elif significant_continuous == 0:
        print("\n⚠️  WEAK HOPE - Some continuous but all <30%")
        print(f"   {tasks_with_continuous} tasks have continuous but minority")
        print("   Unlikely to validate concentration threshold")
        print("   Recommendation: STOP, not worth the effort")

    elif continuous_dominated == 0:
        print(f"\n⚠️  MODERATE HOPE - {significant_continuous} tasks have 30-50% continuous")
        print("   Mixed categorical-continuous tasks")
        print("   Could try separating continuous vs categorical SHAP importance")
        print("   Recommendation: RISKY - might work but uncertain")

    else:
        print(f"\n✓ STRONG HOPE - {continuous_dominated} tasks are continuous-dominated (>50%)")
        print("   Can compute SHAP concentration for continuous features")
        print("   Likely to find different pattern/threshold")
        print("   Recommendation: PROCEED to Phase 2")

    # Save results
    output_dir = base_path / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / 'shap_feature_type_analysis.csv', index=False)
    print(f"\n✓ Results saved to {output_dir}/shap_feature_type_analysis.csv")

    return df


if __name__ == '__main__':
    df = main()
