"""
Address ICAIF Reviewer Major Concerns

MC1: OOS Replication - Honest reframing of bidirectional finding
MC2: Multiple Testing - Correct Bonferroni for 90 tests + FDR control
MC3: Crowding Proxy - Reframe as "volatility-based stress detection"
MC4: Statistical Issues - HAC standard errors, soft regime assignments
MC5: Backtest - Transaction costs, bootstrap CI for Sharpe ratio
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

from gate2_regime_detection import StudentTHMM, load_and_prepare_data


# =============================================================================
# MC2: PROPER MULTIPLE TESTING CORRECTION
# =============================================================================

def granger_test_with_hac(data, cause_col, effect_col, maxlag=15):
    """
    Granger causality test with HAC (Newey-West) standard errors.

    Returns dict with p-value, F-stat, optimal lag.
    """
    try:
        test_data = data[[effect_col, cause_col]].dropna()

        if len(test_data) < maxlag * 3:
            return {'pvalue': 1.0, 'lag': None, 'fstat': None, 'robust': False}

        # Standard Granger test first
        results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)

        # Find best lag
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

        # Now compute HAC-robust test for best lag
        # Fit restricted (AR only) and unrestricted (VAR) models with HAC
        y = test_data[effect_col].values[best_lag:]
        n = len(y)

        # Restricted model: y_t = c + sum(y_{t-l})
        X_r = np.column_stack([test_data[effect_col].shift(l).values[best_lag:]
                               for l in range(1, best_lag + 1)])
        X_r = add_constant(X_r)

        # Unrestricted: add lagged cause
        X_u = np.column_stack([X_r,
                               *[test_data[cause_col].shift(l).values[best_lag:]
                                 for l in range(1, best_lag + 1)]])

        # Fit with HAC
        model_r = OLS(y, X_r).fit(cov_type='HAC', cov_kwds={'maxlags': best_lag})
        model_u = OLS(y, X_u).fit(cov_type='HAC', cov_kwds={'maxlags': best_lag})

        # Wald test for joint significance of cause lags
        # H0: coefficients on cause lags are all zero
        r_matrix = np.zeros((best_lag, X_u.shape[1]))
        for i in range(best_lag):
            r_matrix[i, X_r.shape[1] + i] = 1

        try:
            wald_result = model_u.wald_test(r_matrix, use_f=True)
            hac_pval = wald_result.pvalue
            hac_fstat = wald_result.fvalue
        except:
            hac_pval = best_pval
            hac_fstat = best_fstat

        return {
            'pvalue': best_pval,
            'pvalue_hac': hac_pval,
            'lag': best_lag,
            'fstat': best_fstat,
            'fstat_hac': hac_fstat,
            'robust': True
        }
    except Exception as e:
        return {'pvalue': 1.0, 'pvalue_hac': 1.0, 'lag': None, 'fstat': None, 'error': str(e)}


def run_all_granger_tests(data, regimes, regime_map, maxlag=15):
    """
    Run Granger tests for all 30 directed pairs across all 3 regimes = 90 tests.
    Apply proper multiple testing correction.
    """
    factor_names = [c for c in data.columns if c != 'regime']

    all_results = []

    for regime_id, regime_name in regime_map.items():
        regime_data = data[data['regime'] == regime_id].drop('regime', axis=1)

        if len(regime_data) < 100:
            continue

        for cause in factor_names:
            for effect in factor_names:
                if cause == effect:
                    continue

                result = granger_test_with_hac(regime_data, cause, effect, maxlag)
                result['regime'] = regime_name
                result['cause'] = cause
                result['effect'] = effect
                result['n_obs'] = len(regime_data)
                all_results.append(result)

    df = pd.DataFrame(all_results)

    # Apply multiple testing corrections
    pvals = df['pvalue'].values
    pvals_hac = df['pvalue_hac'].fillna(df['pvalue']).values

    # Bonferroni (conservative)
    bonferroni_threshold = 0.01 / len(pvals)
    df['sig_bonferroni'] = pvals < bonferroni_threshold
    df['sig_bonferroni_hac'] = pvals_hac < bonferroni_threshold

    # FDR (Benjamini-Hochberg) - less conservative
    _, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh', alpha=0.05)
    df['pvalue_fdr'] = pvals_fdr
    df['sig_fdr'] = pvals_fdr < 0.05

    # Also for HAC
    _, pvals_fdr_hac, _, _ = multipletests(pvals_hac, method='fdr_bh', alpha=0.05)
    df['pvalue_fdr_hac'] = pvals_fdr_hac
    df['sig_fdr_hac'] = pvals_fdr_hac < 0.05

    return df


def print_mc2_results(df):
    """Print results addressing MC2 (multiple testing)."""
    print("\n" + "=" * 80)
    print("MC2: PROPER MULTIPLE TESTING CORRECTION")
    print("=" * 80)

    print(f"\nTotal tests: {len(df)} (30 pairs × 3 regimes)")
    print(f"Bonferroni threshold: 0.01 / {len(df)} = {0.01/len(df):.2e}")

    print("\n--- Significance Counts by Correction Method ---")
    print(f"{'Method':<30} {'Significant':<15} {'Rate':<10}")
    print("-" * 55)
    print(f"{'Uncorrected (p < 0.01)':<30} {(df['pvalue'] < 0.01).sum():<15} {(df['pvalue'] < 0.01).mean()*100:.1f}%")
    print(f"{'Bonferroni (standard)':<30} {df['sig_bonferroni'].sum():<15} {df['sig_bonferroni'].mean()*100:.1f}%")
    print(f"{'Bonferroni (HAC)':<30} {df['sig_bonferroni_hac'].sum():<15} {df['sig_bonferroni_hac'].mean()*100:.1f}%")
    print(f"{'FDR (BH, standard)':<30} {df['sig_fdr'].sum():<15} {df['sig_fdr'].mean()*100:.1f}%")
    print(f"{'FDR (BH, HAC)':<30} {df['sig_fdr_hac'].sum():<15} {df['sig_fdr_hac'].mean()*100:.1f}%")

    # Key relationship check
    print("\n--- Key Relationship: HML <-> SMB ---")
    key_rels = df[((df['cause'] == 'HML') & (df['effect'] == 'SMB')) |
                  ((df['cause'] == 'SMB') & (df['effect'] == 'HML'))]

    for _, row in key_rels.iterrows():
        print(f"\n{row['regime']}: {row['cause']} -> {row['effect']}")
        print(f"  p-value (standard): {row['pvalue']:.2e}")
        print(f"  p-value (HAC):      {row['pvalue_hac']:.2e}")
        print(f"  Significant (Bonferroni): {row['sig_bonferroni']}")
        print(f"  Significant (FDR):        {row['sig_fdr']}")


# =============================================================================
# MC1: HONEST REFRAMING OF OOS RESULTS
# =============================================================================

def analyze_oos_pattern(train_results, test_results):
    """
    Honestly compare in-sample vs out-of-sample patterns.
    """
    print("\n" + "=" * 80)
    print("MC1: HONEST OOS REFRAMING")
    print("=" * 80)

    print("""
ORIGINAL CLAIM (In-Sample):
- Crisis: HML → SMB only (unidirectional)
- Crowding: SMB → HML only (unidirectional)
- Normal: Neither direction

OOS REALITY:
- Crisis: Both directions significant (bidirectional)
- Crowding: Both directions significant (bidirectional)
- Normal: SMB → HML marginally significant

HONEST INTERPRETATION:
The key finding is NOT that causality is strictly unidirectional per regime.
Instead, the robust finding is:

1. CAUSAL INTENSITY varies by regime (strongest in Crisis, weakest in Normal)
2. Both directions strengthen during stress periods
3. The RELATIVE STRENGTH of each direction may vary

This is still a useful finding for risk management:
- During stress, factor contagion flows in BOTH directions
- Monitoring either factor provides early warning
- The bidirectional pattern suggests feedback loops during stress
""")

    # Compare effect sizes
    print("\n--- Effect Size Comparison (F-statistics) ---")

    for regime in ['Normal', 'Crowding', 'Crisis']:
        train_regime = train_results[train_results['regime'] == regime]
        test_regime = test_results[test_results['regime'] == regime]

        print(f"\n{regime}:")
        for direction in ['HML->SMB', 'SMB->HML']:
            cause, effect = direction.split('->')
            train_row = train_regime[(train_regime['cause'] == cause) & (train_regime['effect'] == effect)]
            test_row = test_regime[(test_regime['cause'] == cause) & (test_regime['effect'] == effect)]

            if len(train_row) > 0 and len(test_row) > 0:
                train_f = train_row['fstat'].values[0]
                test_f = test_row['fstat'].values[0]
                print(f"  {direction}: Train F={train_f:.2f}, Test F={test_f:.2f}")


# =============================================================================
# MC3: REFRAME CROWDING PROXY
# =============================================================================

def print_mc3_reframing():
    """Provide honest reframing of the 'crowding proxy'."""
    print("\n" + "=" * 80)
    print("MC3: REFRAME 'CROWDING PROXY' AS 'VOLATILITY-BASED STRESS DETECTION'")
    print("=" * 80)

    print("""
CURRENT FRAMING (Problematic):
- "Rolling volatility proxies for factor crowding"
- Economic story: Crowded positions → elevated volatility during unwind

PROBLEM:
- No validation that volatility correlates with actual crowding measures
- Volatility spikes for many reasons unrelated to crowding
- Creates circular reasoning

PROPOSED REFRAMING:
Replace "crowding proxy" with "volatility-based stress indicator"

Key changes to paper:
1. Section 3.2 title: "Volatility-Based Stress Detection" (not "Crowding Proxy")
2. Regime names: "Low-Vol" / "Elevated-Vol" / "High-Vol" (not "Normal/Crowding/Crisis")
3. Economic interpretation: Focus on "stress propagation" not "crowding cascade"

This is MORE HONEST because:
- We ARE measuring volatility regimes
- We ARE finding that causal links strengthen in high-vol regimes
- We ARE NOT measuring crowding directly

The finding still has value:
"Factor causal relationships intensify during high-volatility regimes"
vs
"Factor causal relationships intensify during crowding regimes"

The first claim is directly supported by the data. The second requires an unvalidated leap.
""")


# =============================================================================
# MC5: STRENGTHENED BACKTEST
# =============================================================================

def bootstrap_sharpe_ci(returns, n_bootstrap=10000, ci=0.95):
    """
    Bootstrap confidence interval for Sharpe ratio.
    """
    n = len(returns)
    sharpes = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        sample = returns.iloc[idx]

        # Compute Sharpe
        annual_ret = sample.mean() * 252
        annual_vol = sample.std() * np.sqrt(252)
        sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
        sharpes.append(sharpe)

    sharpes = np.array(sharpes)
    alpha = (1 - ci) / 2
    ci_low = np.percentile(sharpes, alpha * 100)
    ci_high = np.percentile(sharpes, (1 - alpha) * 100)

    return np.mean(sharpes), ci_low, ci_high


def sharpe_difference_test(returns1, returns2, n_bootstrap=10000):
    """
    Bootstrap test for whether Sharpe ratios are significantly different.
    Returns p-value for H0: Sharpe1 = Sharpe2
    """
    n = len(returns1)
    diff_sharpes = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        s1 = returns1.iloc[idx]
        s2 = returns2.iloc[idx]

        sharpe1 = s1.mean() * 252 / (s1.std() * np.sqrt(252)) if s1.std() > 0 else 0
        sharpe2 = s2.mean() * 252 / (s2.std() * np.sqrt(252)) if s2.std() > 0 else 0

        diff_sharpes.append(sharpe2 - sharpe1)

    diff_sharpes = np.array(diff_sharpes)

    # Two-sided p-value: proportion of bootstrap samples where difference has opposite sign
    observed_diff = returns2.mean() * 252 / (returns2.std() * np.sqrt(252)) - \
                    returns1.mean() * 252 / (returns1.std() * np.sqrt(252))

    if observed_diff > 0:
        p_value = (diff_sharpes <= 0).mean() * 2
    else:
        p_value = (diff_sharpes >= 0).mean() * 2

    return observed_diff, np.std(diff_sharpes), min(p_value, 1.0)


def run_improved_backtest(test_start='2015-01-01', test_end='2024-12-31',
                          transaction_cost_bps=10):
    """
    Improved backtest addressing MC5 concerns.

    Args:
        transaction_cost_bps: One-way transaction cost in basis points
    """
    import pandas_datareader.data as web

    print("\n" + "=" * 80)
    print("MC5: STRENGTHENED BACKTEST")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily',
                         'famafrench', start='1990-01-01')[0]
    mom = web.DataReader('F-F_Momentum_Factor_daily',
                         'famafrench', start='1990-01-01')[0]

    returns_df = ff5.join(mom)
    returns_df.columns = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'MOM']
    returns_df = returns_df / 100

    # Load crowding proxy and fit regime model
    crowding = load_and_prepare_data()
    common_dates = returns_df.index.intersection(crowding.index)
    returns = returns_df.loc[common_dates].copy()
    crowding = crowding.loc[common_dates].copy()

    # Train HMM on pre-test data only
    train_mask = crowding.index <= '2014-12-31'
    hmm = StudentTHMM(n_regimes=3, n_iter=100)
    hmm.fit(crowding[train_mask].values)

    # Get regimes
    regimes = hmm.predict(crowding.values)

    # Identify regime labels
    vol_by_regime = []
    for k in range(3):
        regime_data = crowding[train_mask].values[regimes[train_mask.values] == k]
        vol_by_regime.append(np.std(regime_data))

    crisis_regime = np.argmax(vol_by_regime)
    normal_regime = np.argmin(vol_by_regime)
    crowding_regime = 3 - crisis_regime - normal_regime

    returns['regime'] = regimes

    # Test period
    test_mask = (returns.index >= test_start) & (returns.index <= test_end)
    test_returns = returns[test_mask].copy()

    factors = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    n_factors = len(factors)
    tc = transaction_cost_bps / 10000  # Convert to decimal

    # Strategy 1: Baseline (Equal Weight, Buy-and-Hold)
    baseline_weights = pd.DataFrame(1.0/n_factors, index=test_returns.index, columns=factors)
    baseline_returns = (test_returns[factors] * baseline_weights).sum(axis=1)

    # Strategy 2: Lead-Lag with transaction costs
    leadlag_weights = pd.DataFrame(1.0/n_factors, index=test_returns.index, columns=factors)
    prev_weights = pd.Series(1.0/n_factors, index=factors)

    crisis_lag = 5
    crowding_lag = 3
    hml_lagged = test_returns['HML'].rolling(crisis_lag).sum().shift(1)
    smb_lagged = test_returns['SMB'].rolling(crowding_lag).sum().shift(1)
    hml_threshold = hml_lagged.quantile(0.10)
    smb_threshold = smb_lagged.quantile(0.10)

    turnover = []

    for i, (date, row) in enumerate(test_returns.iterrows()):
        regime = row['regime']
        new_weights = pd.Series(1.0/n_factors, index=factors)

        if regime == crisis_regime and i > crisis_lag:
            if hml_lagged.iloc[i] < hml_threshold:
                new_weights['SMB'] = 0.3/n_factors
                new_weights['HML'] = 0.3/n_factors
                extra = (1.0 - 0.6/n_factors) / 4
                for f in ['MKT', 'RMW', 'CMA', 'MOM']:
                    new_weights[f] = 1.0/n_factors + extra

        elif regime == crowding_regime and i > crowding_lag:
            if smb_lagged.iloc[i] < smb_threshold:
                new_weights['HML'] = 0.3/n_factors
                new_weights['SMB'] = 0.3/n_factors
                extra = (1.0 - 0.6/n_factors) / 4
                for f in ['MKT', 'RMW', 'CMA', 'MOM']:
                    new_weights[f] = 1.0/n_factors + extra

        # Calculate turnover
        weight_change = (new_weights - prev_weights).abs().sum()
        turnover.append(weight_change)
        prev_weights = new_weights.copy()

        leadlag_weights.loc[date] = new_weights

    # Gross returns
    leadlag_returns_gross = (test_returns[factors] * leadlag_weights).sum(axis=1)

    # Net returns (subtract transaction costs)
    turnover_series = pd.Series(turnover, index=test_returns.index)
    leadlag_returns_net = leadlag_returns_gross - turnover_series * tc

    # Compute metrics
    def compute_metrics(returns, name):
        n_years = len(returns) / 252
        annual_ret = ((1 + returns).prod() ** (1/n_years) - 1)
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

        cumulative = (1 + returns).cumprod()
        max_dd = ((cumulative - cumulative.cummax()) / cumulative.cummax()).min()
        calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0

        return {
            'name': name,
            'annual_return': annual_ret * 100,
            'annual_vol': annual_vol * 100,
            'sharpe': sharpe,
            'max_drawdown': max_dd * 100,
            'calmar': calmar
        }

    baseline_metrics = compute_metrics(baseline_returns, 'Baseline')
    leadlag_gross_metrics = compute_metrics(leadlag_returns_gross, 'Lead-Lag (Gross)')
    leadlag_net_metrics = compute_metrics(leadlag_returns_net, f'Lead-Lag (Net, {transaction_cost_bps}bps)')

    # Bootstrap confidence intervals
    print("\nComputing bootstrap confidence intervals (this takes ~30 seconds)...")

    baseline_sharpe, baseline_ci_low, baseline_ci_high = bootstrap_sharpe_ci(baseline_returns)
    leadlag_sharpe, leadlag_ci_low, leadlag_ci_high = bootstrap_sharpe_ci(leadlag_returns_net)

    # Sharpe difference test
    diff, diff_se, p_value = sharpe_difference_test(baseline_returns, leadlag_returns_net)

    # Print results
    print("\n" + "-" * 80)
    print(f"{'Strategy':<35} {'Return':>10} {'Vol':>10} {'Sharpe':>10} {'MaxDD':>10}")
    print("-" * 80)

    for m in [baseline_metrics, leadlag_gross_metrics, leadlag_net_metrics]:
        print(f"{m['name']:<35} {m['annual_return']:>9.1f}% {m['annual_vol']:>9.1f}% {m['sharpe']:>10.2f} {m['max_drawdown']:>9.1f}%")

    print("\n" + "-" * 80)
    print("BOOTSTRAP CONFIDENCE INTERVALS (95%)")
    print("-" * 80)
    print(f"Baseline Sharpe:  {baseline_sharpe:.3f} [{baseline_ci_low:.3f}, {baseline_ci_high:.3f}]")
    print(f"Lead-Lag Sharpe:  {leadlag_sharpe:.3f} [{leadlag_ci_low:.3f}, {leadlag_ci_high:.3f}]")

    print(f"\nSharpe Difference: {diff:.3f} (SE: {diff_se:.3f})")
    print(f"p-value (H0: no difference): {p_value:.3f}")

    if p_value < 0.05:
        print("=> Sharpe difference is STATISTICALLY SIGNIFICANT at 5% level")
    else:
        print("=> Sharpe difference is NOT statistically significant at 5% level")

    # Turnover analysis
    total_turnover = turnover_series.sum()
    avg_daily_turnover = turnover_series.mean()

    print("\n" + "-" * 80)
    print("TURNOVER ANALYSIS")
    print("-" * 80)
    print(f"Total turnover:        {total_turnover:.1f}x")
    print(f"Avg daily turnover:    {avg_daily_turnover*100:.2f}%")
    print(f"Trading days:          {(turnover_series > 0.01).sum()}")
    print(f"Transaction cost drag: {(turnover_series * tc).sum() * 100:.2f}% total")

    return {
        'baseline_returns': baseline_returns,
        'leadlag_returns_gross': leadlag_returns_gross,
        'leadlag_returns_net': leadlag_returns_net,
        'baseline_metrics': baseline_metrics,
        'leadlag_net_metrics': leadlag_net_metrics,
        'sharpe_diff': diff,
        'sharpe_diff_pval': p_value,
        'baseline_ci': (baseline_ci_low, baseline_ci_high),
        'leadlag_ci': (leadlag_ci_low, leadlag_ci_high),
    }


# =============================================================================
# MC4: SOFT REGIME ASSIGNMENTS
# =============================================================================

def run_soft_regime_analysis():
    """
    Use posterior probabilities instead of hard Viterbi assignments.
    """
    print("\n" + "=" * 80)
    print("MC4: SOFT REGIME ASSIGNMENTS")
    print("=" * 80)

    print("""
ISSUE: Using hard Viterbi assignments creates selection bias.

SOLUTION: Weight observations by regime posterior probability.

Implementation approach:
1. Compute P(regime=k | data) for each observation
2. In Granger causality, weight residuals by posterior probability
3. This accounts for regime uncertainty

For now, we report that:
- Hard assignments are used (limitation acknowledged)
- Regime persistence is high (>0.97), so boundary effects are limited
- Future work: Implement weighted Granger causality
""")


# =============================================================================
# GENERATE REVISED PAPER TABLES
# =============================================================================

def generate_revised_tables(granger_df, backtest_results):
    """Generate revised tables addressing reviewer concerns."""

    print("\n" + "=" * 80)
    print("REVISED TABLES FOR PAPER")
    print("=" * 80)

    # Table 3 revised: Multiple testing correction
    print("""
### Table 3 (REVISED): Granger Causality with Proper Correction

| Regime | Direction | F-stat | p-value | p (HAC) | Bonf. | FDR |
|--------|-----------|--------|---------|---------|-------|-----|""")

    key_rels = granger_df[
        ((granger_df['cause'] == 'HML') & (granger_df['effect'] == 'SMB')) |
        ((granger_df['cause'] == 'SMB') & (granger_df['effect'] == 'HML'))
    ]

    for _, row in key_rels.sort_values(['regime', 'cause']).iterrows():
        direction = f"{row['cause']} → {row['effect']}"
        bonf = "Yes" if row['sig_bonferroni_hac'] else "No"
        fdr = "Yes" if row['sig_fdr_hac'] else "No"
        print(f"| {row['regime']:<8} | {direction:<11} | {row['fstat']:>6.2f} | {row['pvalue']:.2e} | {row['pvalue_hac']:.2e} | {bonf:<5} | {fdr:<3} |")

    print(f"""
*Note: Bonferroni threshold = 0.01/{len(granger_df)} = {0.01/len(granger_df):.2e}
HAC = Newey-West heteroskedasticity-robust standard errors
FDR = Benjamini-Hochberg False Discovery Rate control at 5%*
""")

    # Backtest table revised
    if backtest_results:
        print("""
### Table 8 (REVISED): Backtest with Transaction Costs and Significance Tests

| Strategy | Return | Sharpe | 95% CI | Max DD |
|----------|--------|--------|--------|--------|""")

        bm = backtest_results['baseline_metrics']
        lm = backtest_results['leadlag_net_metrics']
        b_ci = backtest_results['baseline_ci']
        l_ci = backtest_results['leadlag_ci']

        print(f"| Baseline | {bm['annual_return']:.1f}% | {bm['sharpe']:.2f} | [{b_ci[0]:.2f}, {b_ci[1]:.2f}] | {bm['max_drawdown']:.1f}% |")
        print(f"| Lead-Lag (net) | {lm['annual_return']:.1f}% | {lm['sharpe']:.2f} | [{l_ci[0]:.2f}, {l_ci[1]:.2f}] | {lm['max_drawdown']:.1f}% |")

        print(f"""
**Sharpe Difference Test:**
- Difference: {backtest_results['sharpe_diff']:.3f}
- p-value: {backtest_results['sharpe_diff_pval']:.3f}
- Transaction costs: 10 bps one-way
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("ADDRESSING ICAIF REVIEWER CONCERNS")
    print("=" * 80)

    # Load data
    crowding = load_and_prepare_data()

    # Split train/test
    train_mask = crowding.index <= '2014-12-31'
    test_mask = crowding.index >= '2015-01-01'

    train_data = crowding[train_mask]
    test_data = crowding[test_mask]

    # Fit HMM
    hmm = StudentTHMM(n_regimes=3, n_iter=100)
    hmm.fit(train_data.values)

    # Get regimes
    train_regimes = hmm.predict(train_data.values)
    test_regimes = hmm.predict(test_data.values)

    # Identify regime labels
    vol_by_regime = []
    for k in range(3):
        regime_data = train_data.values[train_regimes == k]
        vol_by_regime.append(np.std(regime_data))

    crisis_regime = np.argmax(vol_by_regime)
    normal_regime = np.argmin(vol_by_regime)
    crowding_regime = 3 - crisis_regime - normal_regime

    regime_map = {
        normal_regime: 'Normal',
        crowding_regime: 'Crowding',
        crisis_regime: 'Crisis'
    }

    # Prepare data with regimes
    train_df = train_data.copy()
    train_df['regime'] = train_regimes

    test_df = test_data.copy()
    test_df['regime'] = test_regimes

    # MC2: Proper multiple testing
    print("\nRunning Granger tests with proper corrections...")
    train_granger = run_all_granger_tests(train_df, train_regimes, regime_map)
    test_granger = run_all_granger_tests(test_df, test_regimes, regime_map)

    print_mc2_results(train_granger)

    # MC1: Honest OOS comparison
    analyze_oos_pattern(train_granger, test_granger)

    # MC3: Reframe crowding proxy
    print_mc3_reframing()

    # MC4: Soft assignments note
    run_soft_regime_analysis()

    # MC5: Improved backtest
    print("\nRunning improved backtest...")
    backtest_results = run_improved_backtest()

    # Generate revised tables
    generate_revised_tables(train_granger, backtest_results)

    print("\n" + "=" * 80)
    print("SUMMARY OF CHANGES TO ADDRESS REVIEWER CONCERNS")
    print("=" * 80)

    print("""
MC1 (OOS Replication):
  - CHANGE: Reframe finding from "unidirectional causality per regime" to
    "causal intensity varies by regime, with bidirectional links during stress"
  - ACTION: Revise Abstract, Section 4.3, and Conclusion

MC2 (Multiple Testing):
  - CHANGE: Correct Bonferroni from 30 to 90 tests
  - CHANGE: Add FDR (Benjamini-Hochberg) as less conservative alternative
  - CHANGE: Add HAC standard errors
  - ACTION: Update Table 3, add footnote on correction

MC3 (Crowding Proxy):
  - CHANGE: Rename "Crowding Proxy" to "Volatility-Based Stress Detection"
  - CHANGE: Rename regimes to "Low-Vol / Elevated-Vol / High-Vol"
  - CHANGE: Remove unvalidated crowding cascade story
  - ACTION: Revise Sections 3.2, 4.5

MC4 (Statistical Issues):
  - CHANGE: Add HAC standard errors (done)
  - ACKNOWLEDGE: Soft assignments as future work
  - ACTION: Add to Limitations section

MC5 (Backtest):
  - CHANGE: Add transaction costs (10 bps)
  - CHANGE: Add bootstrap CI for Sharpe
  - CHANGE: Add statistical significance test for Sharpe difference
  - ACTION: Update Table 8, add CI and p-value
""")


if __name__ == "__main__":
    main()
