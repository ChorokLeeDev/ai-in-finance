"""
Out-of-Sample Validation for ICAIF 2025 Submission

This script implements rigorous OOS validation:
1. Train/Test split (default: 1990-2014 train, 2015-2024 test)
2. Fit HMM on training data only
3. Discover Granger relationships on training data
4. Validate both regime detection AND Granger relationships on test data

Key question: Do the regime-dependent predictive relationships generalize?
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

from gate2_regime_detection import StudentTHMM, load_and_prepare_data


def granger_test_pair(data, cause_col, effect_col, maxlag=15, alpha=0.01):
    """
    Test if cause_col Granger-causes effect_col.

    Returns:
        dict with 'significant', 'pvalue', 'lag', 'fstat'
    """
    try:
        test_data = data[[effect_col, cause_col]].dropna()

        if len(test_data) < maxlag * 3:
            return {'significant': False, 'pvalue': 1.0, 'lag': None, 'fstat': None}

        results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)

        # Find best lag (minimum p-value)
        best_pval = 1.0
        best_lag = 1
        best_fstat = 0

        for lag in range(1, maxlag + 1):
            pval = results[lag][0]['ssr_ftest'][1]
            fstat = results[lag][0]['ssr_ftest'][0]
            if pval < best_pval:
                best_pval = pval
                best_lag = lag
                best_fstat = fstat

        return {
            'significant': best_pval < alpha,
            'pvalue': best_pval,
            'lag': best_lag,
            'fstat': best_fstat
        }
    except Exception as e:
        return {'significant': False, 'pvalue': 1.0, 'lag': None, 'fstat': None, 'error': str(e)}


def run_oos_validation(train_end='2014-12-31', test_start='2015-01-01', alpha=0.01):
    """
    Main OOS validation function.

    Args:
        train_end: Last date for training
        test_start: First date for testing
        alpha: Significance threshold
    """
    print("=" * 70)
    print("OUT-OF-SAMPLE VALIDATION")
    print("=" * 70)
    print(f"\nTrain period: 1990-01-01 to {train_end}")
    print(f"Test period:  {test_start} to present")
    print(f"Alpha: {alpha}")

    # =========================================================================
    # Step 1: Load and split data
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 1: Data Split")
    print("-" * 70)

    crowding = load_and_prepare_data()
    factor_names = list(crowding.columns)

    train_mask = crowding.index <= train_end
    test_mask = crowding.index >= test_start

    train_data = crowding[train_mask]
    test_data = crowding[test_mask]

    print(f"  Training samples: {len(train_data)} ({train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')})")
    print(f"  Test samples:     {len(test_data)} ({test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')})")

    # =========================================================================
    # Step 2: Fit HMM on training data ONLY
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 2: Fit Student-t HMM (Training Data Only)")
    print("-" * 70)

    hmm = StudentTHMM(n_regimes=3, n_iter=100)
    hmm.fit(train_data.values)

    # Get regime assignments
    train_regimes = hmm.predict(train_data.values)

    # Identify regimes by volatility (Crisis = highest vol)
    vol_by_regime = []
    for k in range(3):
        regime_data = train_data.values[train_regimes == k]
        vol = np.std(regime_data)
        vol_by_regime.append(vol)

    crisis_regime = np.argmax(vol_by_regime)
    normal_regime = np.argmin(vol_by_regime)
    crowding_regime = 3 - crisis_regime - normal_regime

    regime_map = {
        normal_regime: 'Normal',
        crowding_regime: 'Crowding',
        crisis_regime: 'Crisis'
    }

    print(f"\n  Fitted degrees of freedom:")
    for k in range(3):
        print(f"    {regime_map.get(k, f'Regime {k}')}: nu = {hmm.nu[k]:.1f}")

    print(f"\n  Training regime distribution:")
    for k in range(3):
        pct = (train_regimes == k).mean() * 100
        print(f"    {regime_map.get(k, f'Regime {k}')}: {pct:.1f}%")

    # =========================================================================
    # Step 3: Discover Granger relationships on training data
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 3: Discover Granger Relationships (Training Data)")
    print("-" * 70)

    train_df = train_data.copy()
    train_df['regime'] = train_regimes

    discovered_relationships = {}

    for regime_id, regime_name in regime_map.items():
        regime_subset = train_df[train_df['regime'] == regime_id].drop('regime', axis=1)

        print(f"\n  {regime_name} regime ({len(regime_subset)} samples):")

        if len(regime_subset) < 100:
            print(f"    Skipping: too few samples")
            discovered_relationships[regime_name] = []
            continue

        regime_relationships = []

        # Test all pairs
        for cause in factor_names:
            for effect in factor_names:
                if cause == effect:
                    continue

                result = granger_test_pair(regime_subset, cause, effect, maxlag=15, alpha=alpha)

                if result['significant']:
                    regime_relationships.append({
                        'cause': cause,
                        'effect': effect,
                        'pvalue': result['pvalue'],
                        'lag': result['lag'],
                        'fstat': result['fstat']
                    })

        # Sort by p-value
        regime_relationships = sorted(regime_relationships, key=lambda x: x['pvalue'])
        discovered_relationships[regime_name] = regime_relationships

        print(f"    Found {len(regime_relationships)} significant relationships")
        for rel in regime_relationships[:5]:
            print(f"      {rel['cause']} -> {rel['effect']}: p={rel['pvalue']:.2e}, lag={rel['lag']}")

    # =========================================================================
    # Step 4: Apply HMM to test data (FROZEN parameters)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 4: Apply Frozen HMM to Test Data")
    print("-" * 70)

    # Use predict with frozen parameters
    test_regimes = hmm.predict(test_data.values)

    print(f"\n  Test regime distribution:")
    for k in range(3):
        pct = (test_regimes == k).mean() * 100
        print(f"    {regime_map.get(k, f'Regime {k}')}: {pct:.1f}%")

    # Check crisis detection on test events
    test_crises = [
        ('2015-08-24', '2015-09-15', 'China Crash 2015'),
        ('2018-12-01', '2018-12-31', 'Dec 2018 Selloff'),
        ('2020-02-20', '2020-04-01', 'COVID-19'),
        ('2022-01-01', '2022-06-30', '2022 Bear Market'),
    ]

    print(f"\n  Crisis detection on test period events:")
    test_crisis_detection = []

    for start, end, name in test_crises:
        try:
            mask = (test_data.index >= start) & (test_data.index <= end)
            if mask.sum() > 0:
                # Convert mask to numpy array for indexing
                mask_arr = np.array(mask)
                event_regimes = test_regimes[mask_arr]
                pct_crisis = (event_regimes == crisis_regime).mean() * 100
                pct_elevated = ((event_regimes == crisis_regime) | (event_regimes == crowding_regime)).mean() * 100

                detected = pct_elevated > 50
                test_crisis_detection.append(detected)

                status = "✅" if detected else "❌"
                print(f"    {status} {name}: {pct_crisis:.0f}% Crisis, {pct_elevated:.0f}% Elevated")
        except Exception as e:
            print(f"    ⚠️ {name}: Error - {e}")

    regime_oos_rate = sum(test_crisis_detection) / len(test_crisis_detection) * 100 if test_crisis_detection else 0
    print(f"\n  OOS crisis detection rate: {regime_oos_rate:.0f}%")

    # =========================================================================
    # Step 5: Validate Granger relationships on test data
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 5: Validate Granger Relationships (Test Data)")
    print("-" * 70)

    test_df = test_data.copy()
    test_df['regime'] = test_regimes

    validation_results = {}

    for regime_id, regime_name in regime_map.items():
        train_rels = discovered_relationships.get(regime_name, [])

        if not train_rels:
            validation_results[regime_name] = {'discovered': 0, 'validated': 0, 'rate': 0}
            continue

        test_subset = test_df[test_df['regime'] == regime_id].drop('regime', axis=1)

        print(f"\n  {regime_name} regime ({len(test_subset)} test samples):")

        if len(test_subset) < 50:
            print(f"    Skipping: too few test samples")
            validation_results[regime_name] = {'discovered': len(train_rels), 'validated': 0, 'rate': 0}
            continue

        validated = 0

        for rel in train_rels:
            result = granger_test_pair(
                test_subset,
                rel['cause'],
                rel['effect'],
                maxlag=15,
                alpha=0.05  # Slightly relaxed for OOS
            )

            if result['significant']:
                validated += 1
                status = "✅"
            else:
                status = "❌"

            # Only print top relationships
            if rel in train_rels[:3]:
                print(f"    {status} {rel['cause']} -> {rel['effect']}: train p={rel['pvalue']:.2e}, test p={result['pvalue']:.2e}")

        rate = validated / len(train_rels) * 100 if train_rels else 0
        validation_results[regime_name] = {
            'discovered': len(train_rels),
            'validated': validated,
            'rate': rate
        }

        print(f"    Validation rate: {validated}/{len(train_rels)} = {rate:.0f}%")

    # =========================================================================
    # Step 6: Key relationship validation (HML <-> SMB)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 6: Key Relationship Validation (HML <-> SMB)")
    print("-" * 70)

    key_results = {}

    for regime_id, regime_name in regime_map.items():
        test_subset = test_df[test_df['regime'] == regime_id].drop('regime', axis=1)

        if len(test_subset) < 50:
            continue

        # HML -> SMB
        hml_smb = granger_test_pair(test_subset, 'HML', 'SMB', maxlag=15, alpha=0.05)

        # SMB -> HML
        smb_hml = granger_test_pair(test_subset, 'SMB', 'HML', maxlag=15, alpha=0.05)

        key_results[regime_name] = {
            'HML->SMB': hml_smb,
            'SMB->HML': smb_hml
        }

        print(f"\n  {regime_name}:")
        print(f"    HML -> SMB: p={hml_smb['pvalue']:.2e}, lag={hml_smb['lag']}, sig={hml_smb['significant']}")
        print(f"    SMB -> HML: p={smb_hml['pvalue']:.2e}, lag={smb_hml['lag']}, sig={smb_hml['significant']}")

    # =========================================================================
    # Summary Table for Paper
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY TABLE FOR PAPER")
    print("=" * 70)

    print("\nTable: Out-of-Sample Validation Results")
    print("-" * 70)
    print(f"{'Metric':<40} {'Train':<15} {'Test':<15}")
    print("-" * 70)
    print(f"{'Period':<40} {'1990-2014':<15} {'2015-2024':<15}")
    print(f"{'Samples':<40} {len(train_data):<15} {len(test_data):<15}")
    print("-" * 70)

    # Regime detection
    print(f"{'Crisis Event Detection Rate':<40} {'N/A':<15} {regime_oos_rate:.0f}%")

    # Granger validation
    total_discovered = sum(v['discovered'] for v in validation_results.values())
    total_validated = sum(v['validated'] for v in validation_results.values())
    overall_rate = total_validated / total_discovered * 100 if total_discovered > 0 else 0

    print(f"{'Granger Relationships (discovered)':<40} {total_discovered:<15} {'N/A':<15}")
    print(f"{'Granger Relationships (validated OOS)':<40} {'N/A':<15} {total_validated:<15}")
    print(f"{'Validation Rate':<40} {'N/A':<15} {overall_rate:.0f}%")

    print("-" * 70)
    print("\nKey Finding (HML <-> SMB):")

    for regime_name, rels in key_results.items():
        hml_smb_sig = "✓" if rels['HML->SMB']['significant'] else "✗"
        smb_hml_sig = "✓" if rels['SMB->HML']['significant'] else "✗"
        print(f"  {regime_name}: HML->SMB [{hml_smb_sig}] SMB->HML [{smb_hml_sig}]")

    # =========================================================================
    # Final Verdict
    # =========================================================================
    print("\n" + "=" * 70)
    print("OOS VALIDATION VERDICT")
    print("=" * 70)

    criteria = {
        'regime_detection_generalizes': regime_oos_rate >= 50,
        'granger_relationships_replicate': overall_rate >= 30,
        'key_relationship_holds': any(
            rels['HML->SMB']['significant'] or rels['SMB->HML']['significant']
            for rels in key_results.values()
        ),
    }

    passed = sum(criteria.values())
    total = len(criteria)

    print(f"\nCriteria passed: {passed}/{total}")
    for name, value in criteria.items():
        status = "✅" if value else "❌"
        print(f"  {status} {name}")

    if passed >= 2:
        print("\n🟢 OOS VALIDATION PASSED")
        print("   Results generalize to unseen data.")
    else:
        print("\n🔴 OOS VALIDATION FAILED")
        print("   Results may not generalize. Investigate overfitting.")

    return {
        'train_data': train_data,
        'test_data': test_data,
        'hmm': hmm,
        'discovered_relationships': discovered_relationships,
        'validation_results': validation_results,
        'key_results': key_results,
        'regime_oos_rate': regime_oos_rate,
        'overall_granger_rate': overall_rate,
        'criteria': criteria
    }


def run_multiple_splits():
    """
    Run OOS validation with multiple train/test splits for robustness.
    """
    print("=" * 70)
    print("ROBUSTNESS CHECK: Multiple Train/Test Splits")
    print("=" * 70)

    splits = [
        ('2010-12-31', '2011-01-01', '2011+'),
        ('2014-12-31', '2015-01-01', '2015+'),
        ('2017-12-31', '2018-01-01', '2018+'),
    ]

    results = []

    for train_end, test_start, name in splits:
        print(f"\n{'=' * 70}")
        print(f"Split: Train until {train_end}, Test from {test_start}")
        print(f"{'=' * 70}")

        try:
            result = run_oos_validation(train_end=train_end, test_start=test_start)
            results.append({
                'split': name,
                'regime_oos': result['regime_oos_rate'],
                'granger_oos': result['overall_granger_rate'],
                'passed': sum(result['criteria'].values())
            })
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'split': name,
                'regime_oos': 0,
                'granger_oos': 0,
                'passed': 0
            })

    # Summary table
    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)

    print(f"\n{'Split':<10} {'Regime OOS':<15} {'Granger OOS':<15} {'Criteria':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['split']:<10} {r['regime_oos']:.0f}%{'':<10} {r['granger_oos']:.0f}%{'':<10} {r['passed']}/3")

    avg_regime = np.mean([r['regime_oos'] for r in results])
    avg_granger = np.mean([r['granger_oos'] for r in results])

    print("-" * 50)
    print(f"{'Average':<10} {avg_regime:.0f}%{'':<10} {avg_granger:.0f}%")

    return results


def generate_paper_table(results):
    """
    Generate a markdown table formatted for the paper.
    """
    print("\n" + "=" * 70)
    print("PAPER-READY TABLE (Markdown)")
    print("=" * 70)

    # Table 1: OOS Validation Summary
    print("""
### Table X: Out-of-Sample Validation Results

| Metric | Training (1990-2014) | Test (2015-2024) |
|--------|---------------------|------------------|
| Sample Size | {train_n:,} days | {test_n:,} days |
| Crisis Events | N/A | {crisis_rate:.0f}% detected |
| Granger Relationships | {n_discovered} discovered | {n_validated} validated ({val_rate:.0f}%) |
""".format(
        train_n=len(results['train_data']),
        test_n=len(results['test_data']),
        crisis_rate=results['regime_oos_rate'],
        n_discovered=sum(len(v) for v in results['discovered_relationships'].values()),
        n_validated=sum(v['validated'] for v in results['validation_results'].values()),
        val_rate=results['overall_granger_rate']
    ))

    # Table 2: Per-regime validation
    print("""
### Table Y: Per-Regime Granger Causality Validation

| Regime | Train Relationships | Test Validated | Rate |
|--------|--------------------:|---------------:|-----:|""")

    for regime_name, val in results['validation_results'].items():
        if val['discovered'] > 0:
            print(f"| {regime_name} | {val['discovered']} | {val['validated']} | {val['rate']:.0f}% |")

    # Table 3: Key relationship
    print("""
### Table Z: HML↔SMB Direction by Regime (OOS Test Period)

| Regime | HML→SMB | SMB→HML | Direction |
|--------|---------|---------|-----------|""")

    for regime_name, rels in results['key_results'].items():
        hml_smb = "✓" if rels['HML->SMB']['significant'] else "✗"
        smb_hml = "✓" if rels['SMB->HML']['significant'] else "✗"

        if rels['HML->SMB']['significant'] and not rels['SMB->HML']['significant']:
            direction = "HML→SMB"
        elif rels['SMB->HML']['significant'] and not rels['HML->SMB']['significant']:
            direction = "SMB→HML"
        elif rels['HML->SMB']['significant'] and rels['SMB->HML']['significant']:
            direction = "Bidirectional"
        else:
            direction = "None"

        print(f"| {regime_name} | {hml_smb} (p={rels['HML->SMB']['pvalue']:.1e}) | {smb_hml} (p={rels['SMB->HML']['pvalue']:.1e}) | {direction} |")

    print("""
**Note**: Training period 1990-2014, Test period 2015-2024.
Significance at α=0.05 for OOS validation.
""")


if __name__ == "__main__":
    # Run main OOS validation (2015+ test)
    results = run_oos_validation(train_end='2014-12-31', test_start='2015-01-01')

    # Generate paper tables
    generate_paper_table(results)

    print("\n\n")

    # Run robustness check with multiple splits
    # robustness = run_multiple_splits()
