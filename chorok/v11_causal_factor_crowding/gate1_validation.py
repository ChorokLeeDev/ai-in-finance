"""
Gate 1 Validation: Out-of-Sample Test for HML → SMB Causal Relationship

Critical Question: Does the HML → SMB Granger causality (p=1.3e-27, lag=9)
discovered on full sample hold when:
1. Trained on 1990-2015
2. Tested on 2015-2024

If this fails, we pivot. If it passes, we proceed to Gate 2.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING
# ============================================================================

def load_fama_french_factors(filepath=None):
    """
    Load Fama-French 6 factors from CSV or download from Kenneth French's website.

    Returns DataFrame with columns: MKT, SMB, HML, RMW, CMA, MOM
    """
    if filepath:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    else:
        # Try to load from Kenneth French's website
        try:
            import pandas_datareader.data as web

            # Get Fama-French 5 factors
            ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily',
                                 'famafrench', start='1990-01-01')[0]

            # Get Momentum factor
            mom = web.DataReader('F-F_Momentum_Factor_daily',
                                 'famafrench', start='1990-01-01')[0]

            # Combine
            df = ff5.join(mom)
            df.columns = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'MOM']
            df = df.drop('RF', axis=1)

        except Exception as e:
            print(f"Could not download data: {e}")
            print("Please provide filepath to Fama-French data CSV")
            return None

    return df


def compute_crowding_proxy(returns, window=60):
    """
    Simple crowding proxy based on rolling volatility.

    More sophisticated proxies would include:
    - Pairwise correlation among factor-exposed stocks
    - Short interest concentration
    - ETF flow data

    For validation, we use volatility as it's available from factor returns alone.
    """
    crowding = returns.rolling(window=window).std()

    # Normalize to z-scores
    crowding = (crowding - crowding.mean()) / crowding.std()

    return crowding.dropna()


# ============================================================================
# VALIDATION TESTS
# ============================================================================

def granger_test(data, cause, effect, maxlag=20):
    """
    Run Granger causality test and return best lag and p-value.
    """
    test_data = data[[effect, cause]].dropna()

    if len(test_data) < maxlag * 2:
        return None, None, None

    try:
        results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)

        # Find lag with minimum p-value
        best_lag = None
        best_pval = 1.0

        for lag in range(1, maxlag + 1):
            # Use F-test p-value
            pval = results[lag][0]['ssr_ftest'][1]
            if pval < best_pval:
                best_pval = pval
                best_lag = lag

        return best_lag, best_pval, results

    except Exception as e:
        print(f"Granger test failed: {e}")
        return None, None, None


def permutation_test(data, cause, effect, n_permutations=1000, maxlag=20):
    """
    Permutation test to establish null distribution of p-values.

    Shuffle the cause variable and re-run Granger test to see what
    p-values we'd get by chance.
    """
    null_pvals = []

    test_data = data[[effect, cause]].dropna()

    for i in range(n_permutations):
        # Shuffle the cause variable (break temporal structure)
        shuffled = test_data.copy()
        shuffled[cause] = np.random.permutation(shuffled[cause].values)

        try:
            results = grangercausalitytests(shuffled, maxlag=maxlag, verbose=False)

            # Get minimum p-value across lags
            min_pval = min(results[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag + 1))
            null_pvals.append(min_pval)
        except:
            continue

    return np.array(null_pvals)


def out_of_sample_prediction(train_data, test_data, cause, effect, lag):
    """
    Out-of-sample prediction test.

    1. Fit VAR model on training data
    2. Use it to predict test data
    3. Compare MSE with and without the causal variable
    """
    from statsmodels.tsa.ar_model import AutoReg

    # Model with cause (VAR)
    var_full = VAR(train_data[[effect, cause]])
    results_full = var_full.fit(maxlags=lag)

    # Model without cause (AR model) - use AutoReg instead of VAR
    ar_model = AutoReg(train_data[effect], lags=lag)
    results_ar = ar_model.fit()

    # Predictions on test data
    # We'll do rolling 1-step ahead predictions

    test_effect = test_data[effect].values
    test_cause = test_data[cause].values

    predictions_full = []
    predictions_ar = []
    actuals = []

    for t in range(lag, len(test_data) - 1):
        # History for VAR
        hist_full = np.column_stack([test_effect[t-lag:t], test_cause[t-lag:t]])

        try:
            # Predict next value with VAR (uses both effect and cause)
            pred_full = results_full.forecast(hist_full, steps=1)[0, 0]

            # Predict with AR (uses only effect history)
            # AutoReg predict needs start/end indices
            pred_ar = results_ar.predict(start=len(train_data) + t,
                                         end=len(train_data) + t,
                                         dynamic=False).values[0]

            predictions_full.append(pred_full)
            predictions_ar.append(pred_ar)
            actuals.append(test_effect[t])
        except Exception as e:
            continue

    # Compute MSE
    predictions_full = np.array(predictions_full)
    predictions_ar = np.array(predictions_ar)
    actuals = np.array(actuals)

    mse_full = np.mean((predictions_full - actuals) ** 2)
    mse_ar = np.mean((predictions_ar - actuals) ** 2)

    # Improvement ratio
    improvement = (mse_ar - mse_full) / mse_ar * 100

    return {
        'mse_with_cause': mse_full,
        'mse_without_cause': mse_ar,
        'improvement_pct': improvement,
        'n_predictions': len(actuals)
    }


# ============================================================================
# MAIN VALIDATION
# ============================================================================

def run_gate1_validation(data=None, verbose=True):
    """
    Run complete Gate 1 validation.

    Returns dict with:
    - pass: bool (True if validation passes)
    - in_sample: dict of in-sample results
    - out_of_sample: dict of out-of-sample results
    - permutation: dict of permutation test results
    - bonferroni: dict of multiple testing correction
    """

    results = {
        'pass': False,
        'in_sample': {},
        'out_of_sample': {},
        'permutation': {},
        'bonferroni': {},
        'recommendation': ''
    }

    # Load data
    if data is None:
        print("Loading Fama-French data...")
        data = load_fama_french_factors()
        if data is None:
            print("ERROR: Could not load data")
            return results

    # Compute crowding proxy
    print("Computing crowding proxy...")
    crowding = compute_crowding_proxy(data, window=60)

    # Split data
    train_end = '2015-12-31'

    train_data = crowding[crowding.index <= train_end]
    test_data = crowding[crowding.index > train_end]

    print(f"Train period: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} obs)")
    print(f"Test period: {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} obs)")
    print()

    # ========================================================================
    # TEST 1: In-sample Granger (replication of original finding)
    # ========================================================================
    print("=" * 60)
    print("TEST 1: In-Sample Granger Causality (Full Sample)")
    print("=" * 60)

    lag, pval, _ = granger_test(crowding, 'HML', 'SMB', maxlag=20)

    results['in_sample'] = {
        'lag': lag,
        'pval': pval,
        'significant': pval < 0.05 if pval else False
    }

    print(f"HML → SMB: lag={lag}, p-value={pval:.2e}")
    print()

    # ========================================================================
    # TEST 2: Out-of-sample Granger
    # ========================================================================
    print("=" * 60)
    print("TEST 2: Out-of-Sample Granger Causality")
    print("=" * 60)

    # Discover lag on training data
    train_lag, train_pval, _ = granger_test(train_data, 'HML', 'SMB', maxlag=20)
    print(f"Training set (1990-2015): HML → SMB lag={train_lag}, p={train_pval:.2e}")

    # Test on held-out data
    test_lag, test_pval, _ = granger_test(test_data, 'HML', 'SMB', maxlag=20)
    print(f"Test set (2015-2024): HML → SMB lag={test_lag}, p={test_pval:.2e}")

    results['out_of_sample'] = {
        'train_lag': train_lag,
        'train_pval': train_pval,
        'test_lag': test_lag,
        'test_pval': test_pval,
        'lag_consistent': abs(train_lag - test_lag) <= 3 if (train_lag and test_lag) else False,
        'significant': test_pval < 0.05 if test_pval else False
    }
    print()

    # ========================================================================
    # TEST 3: Prediction improvement
    # ========================================================================
    print("=" * 60)
    print("TEST 3: Out-of-Sample Prediction Improvement")
    print("=" * 60)

    if train_lag:
        pred_results = out_of_sample_prediction(train_data, test_data, 'HML', 'SMB', train_lag)

        print(f"MSE with HML: {pred_results['mse_with_cause']:.6f}")
        print(f"MSE without HML: {pred_results['mse_without_cause']:.6f}")
        print(f"Improvement: {pred_results['improvement_pct']:.2f}%")
        print(f"N predictions: {pred_results['n_predictions']}")

        results['prediction'] = pred_results
    print()

    # ========================================================================
    # TEST 4: Permutation test
    # ========================================================================
    print("=" * 60)
    print("TEST 4: Permutation Test (establishing null distribution)")
    print("=" * 60)

    print("Running 500 permutations (this may take a minute)...")
    null_pvals = permutation_test(train_data, 'HML', 'SMB', n_permutations=500, maxlag=20)

    # What percentile is our observed p-value?
    if train_pval:
        percentile = np.mean(null_pvals <= train_pval) * 100
        print(f"Observed p-value: {train_pval:.2e}")
        print(f"Percentile in null distribution: {percentile:.1f}%")
        print(f"5th percentile of null: {np.percentile(null_pvals, 5):.2e}")

        results['permutation'] = {
            'observed_pval': train_pval,
            'null_5th_percentile': np.percentile(null_pvals, 5),
            'null_median': np.median(null_pvals),
            'percentile_rank': percentile,
            'significant': percentile < 5
        }
    print()

    # ========================================================================
    # TEST 5: Multiple testing correction
    # ========================================================================
    print("=" * 60)
    print("TEST 5: Multiple Testing Correction (Bonferroni)")
    print("=" * 60)

    # We're testing 30 pairs (6 factors × 5 potential causes)
    n_tests = 30
    alpha = 0.05
    bonferroni_alpha = alpha / n_tests

    print(f"Number of tests: {n_tests}")
    print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.4f}")

    if test_pval:
        print(f"Test p-value: {test_pval:.2e}")
        print(f"Significant after correction: {test_pval < bonferroni_alpha}")

        results['bonferroni'] = {
            'n_tests': n_tests,
            'corrected_alpha': bonferroni_alpha,
            'significant': test_pval < bonferroni_alpha
        }
    print()

    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    print("=" * 60)
    print("GATE 1 VERDICT")
    print("=" * 60)

    # Criteria for passing:
    # 1. Out-of-sample p-value < 0.05
    # 2. Lag is consistent (±3 days) between train and test
    # 3. Prediction improvement > 0%
    # 4. Survives permutation test

    criteria = {
        'out_of_sample_significant': results['out_of_sample'].get('significant', False),
        'lag_consistent': results['out_of_sample'].get('lag_consistent', False),
        'prediction_improvement': results.get('prediction', {}).get('improvement_pct', 0) > 0,
        'permutation_significant': results['permutation'].get('significant', False)
    }

    passed = sum(criteria.values())
    total = len(criteria)

    print(f"Criteria passed: {passed}/{total}")
    for name, value in criteria.items():
        status = "✅" if value else "❌"
        print(f"  {status} {name}: {value}")

    if passed >= 3:
        results['pass'] = True
        results['recommendation'] = "PROCEED to Gate 2: Regime Detection"
        print(f"\n🟢 GATE 1 PASSED - {results['recommendation']}")
    elif passed >= 2:
        results['pass'] = False
        results['recommendation'] = "BORDERLINE - Consider simplifying to single-regime model"
        print(f"\n🟡 GATE 1 BORDERLINE - {results['recommendation']}")
    else:
        results['pass'] = False
        results['recommendation'] = "PIVOT - Core finding does not replicate"
        print(f"\n🔴 GATE 1 FAILED - {results['recommendation']}")

    return results


# ============================================================================
# ALL FACTOR PAIRS (for comprehensive analysis)
# ============================================================================

def run_all_pairs_validation(data=None):
    """
    Run Granger causality for all factor pairs and report top relationships
    that survive out-of-sample testing.
    """
    if data is None:
        data = load_fama_french_factors()
        if data is None:
            return None

    crowding = compute_crowding_proxy(data, window=60)

    train_end = '2015-12-31'
    train_data = crowding[crowding.index <= train_end]
    test_data = crowding[crowding.index > train_end]

    factors = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']

    results = []

    print("Testing all factor pairs...")
    print()

    for cause in factors:
        for effect in factors:
            if cause == effect:
                continue

            # Train
            train_lag, train_pval, _ = granger_test(train_data, cause, effect, maxlag=20)

            # Test
            test_lag, test_pval, _ = granger_test(test_data, cause, effect, maxlag=20)

            if train_pval and test_pval:
                results.append({
                    'cause': cause,
                    'effect': effect,
                    'train_lag': train_lag,
                    'train_pval': train_pval,
                    'test_lag': test_lag,
                    'test_pval': test_pval,
                    'lag_diff': abs(train_lag - test_lag) if (train_lag and test_lag) else None,
                    'replicates': test_pval < 0.05 and train_pval < 0.05
                })

    df = pd.DataFrame(results)
    df = df.sort_values('test_pval')

    print("Top 10 causal relationships (sorted by test p-value):")
    print()
    print(df.head(10).to_string(index=False))

    print()
    print(f"Relationships that replicate (train & test p < 0.05): {df['replicates'].sum()}/{len(df)}")

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("GATE 1 VALIDATION: Out-of-Sample Causal Test")
    print("=" * 60)
    print()

    # Run main validation
    results = run_gate1_validation()

    print()
    print("=" * 60)
    print("COMPREHENSIVE ANALYSIS: All Factor Pairs")
    print("=" * 60)
    print()

    # Run all pairs
    all_pairs = run_all_pairs_validation()
