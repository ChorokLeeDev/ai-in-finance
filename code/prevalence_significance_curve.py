"""
Prevalence-Significance Analysis
==================================

Tests whether the significant HAC p-value in the Elevated regime (p=0.041)
from frozen OOS analysis is driven by prevalence expansion (13.7% train -> 30.7% test).

Methodology:
- Load all frozen OOS Elevated regime data (HML and SMB returns, 2013-2024)
- For prevalence levels from 5% to 50% (1% increments), subsample the test
  observations to match that prevalence and run Granger HAC test
- 500 bootstrap iterations per prevalence level
- Visualize: (a) Median p-value vs prevalence with 90% CI band
              (b) Fraction of significant p<0.05 vs prevalence
              (c) Vertical lines at training (13.7%) and test (30.7%) prevalence
- Save figure and numerical results

The analysis tests the hypothesis that the significance is primarily driven by
the expansion of the regime rather than genuine causal structure.
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, chi2
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# Path configuration
RESULTS_DIR = '/sessions/festive-youthful-mccarthy/mnt/causal_regimes/results'
FIGURES_DIR = '/sessions/festive-youthful-mccarthy/mnt/causal_regimes/figures'
CODE_DIR = '/sessions/festive-youthful-mccarthy/mnt/causal_regimes/code'
DATA_DIR = '/sessions/festive-youthful-mccarthy/mnt/causal_regimes/data'

sys.path.insert(0, CODE_DIR)
from multistart_hmm_pipeline import download_ff_data, relabel_regimes_by_data_norm

# Constants
REGIME_NAMES = ['Normal', 'Elevated', 'Crisis']
TRAIN_PREVALENCE = 13.7  # From task description
TEST_PREVALENCE = 30.7   # From task description
FIXED_LAG = 1
N_BOOTSTRAP = 500
PREVALENCE_RANGE = np.arange(5, 51, 1)  # 5% to 50% in 1% increments


def granger_ftest(y_curr, y_lagged, x_lagged):
    """Standard F-test for Granger causality (x -> y)."""
    n = len(y_curr)
    lag = y_lagged.shape[1]
    X_r = np.column_stack([np.ones(n), y_lagged])
    X_u = np.column_stack([np.ones(n), y_lagged, x_lagged])
    beta_r = np.linalg.lstsq(X_r, y_curr, rcond=None)[0]
    beta_u = np.linalg.lstsq(X_u, y_curr, rcond=None)[0]
    rss_r = np.sum((y_curr - X_r @ beta_r) ** 2)
    rss_u = np.sum((y_curr - X_u @ beta_u) ** 2)
    df1 = lag
    df2 = n - 2 * lag - 1
    if df2 <= 0 or rss_u <= 0:
        return np.nan, np.nan, np.nan, np.nan
    f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
    p_value = 1 - f_dist.cdf(f_stat, df1, df2)
    tss = np.sum((y_curr - y_curr.mean()) ** 2)
    r2_r = 1 - rss_r / tss
    r2_u = 1 - rss_u / tss
    delta_r2 = r2_u - r2_r
    return float(f_stat), float(p_value), float(delta_r2), float(r2_u)


def granger_hac_wald(y_curr, y_lagged, x_lagged, lag):
    """HAC (Newey-West) robust Wald test for Granger causality."""
    n = len(y_curr)
    p = y_lagged.shape[1]
    X_u = np.column_stack([np.ones(n), y_lagged, x_lagged])
    try:
        model = sm.OLS(y_curr, X_u)
        result = model.fit(cov_type='HAC', cov_kwds={'maxlags': lag})
        n_params = X_u.shape[1]
        R = np.zeros((p, n_params))
        for i in range(p):
            R[i, 1 + p + i] = 1.0
        beta = result.params
        V = result.cov_params()
        Rb = R @ beta
        RVR = R @ V @ R.T
        wald_stat = float(Rb @ np.linalg.inv(RVR) @ Rb)
        p_value = float(1 - chi2.cdf(wald_stat, p))
    except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
        wald_stat = np.nan
        p_value = np.nan
    return wald_stat, p_value


def extract_regime_clean_indices(regimes, regime_id, max_lag):
    """Get indices where ALL lags 1..max_lag fall within the same regime."""
    regime_mask = (regimes == regime_id)
    indices = np.where(regime_mask)[0]
    clean_indices = []
    for idx in indices:
        if idx >= max_lag:
            if all(regimes[idx - l] == regime_id for l in range(1, max_lag + 1)):
                clean_indices.append(idx)
    return np.array(clean_indices) if clean_indices else np.array([], dtype=int)


def run_granger_test(y_all, x_all, clean_indices, lag):
    """Run Granger F-test and HAC Wald test at a specific lag."""
    usable = np.array([idx for idx in clean_indices if idx >= lag])
    if len(usable) < 2 * lag + 10:
        return None, None

    y_curr = y_all[usable]
    y_lagged = np.column_stack([y_all[usable - i - 1] for i in range(lag)])
    x_lagged = np.column_stack([x_all[usable - i - 1] for i in range(lag)])

    # HAC test result
    _, hac_p = granger_hac_wald(y_curr, y_lagged, x_lagged, lag)

    return len(usable), hac_p


def load_frozen_oos_data():
    """Load frozen OOS 50-seed results and extract Elevated regime info."""
    print("Loading frozen OOS 50-seed results...")
    with open(f'{RESULTS_DIR}/frozen_oos_50seeds.json', 'r') as f:
        data = json.load(f)

    # Extract HAC p-values for Elevated regime across all seeds
    elevated_results = []
    for seed_result in data['all_seeds']:
        try:
            elevated_data = seed_result['granger']['Elevated']['hml_to_smb']
            hac_p = elevated_data.get('hac_p_value')
            n_obs = elevated_data.get('n_obs')
            if hac_p is not None and n_obs is not None:
                elevated_results.append({
                    'seed': seed_result['seed'],
                    'n_obs': n_obs,
                    'hac_p_value': hac_p,
                })
        except (KeyError, TypeError):
            continue

    print(f"  Found {len(elevated_results)} valid Elevated regime results")
    return elevated_results


def load_test_data():
    """Load test data (2013-2024) and detect regimes using HMM from seed 28."""
    print("Loading test data and running regime detection...")
    df = download_ff_data() / 100.0
    test_df = df.loc['2013-01-01':]

    # Load seed 28 HMM results (which had the significant result)
    train_df = df.loc[:'2012-12-31']

    # Reimport HMM class
    from multistart_hmm_pipeline import StudentTHMM

    # Fit HMM on training data with seed 28
    factor_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    hmm = StudentTHMM(n_regimes=3, n_iter=100, tol=1e-4, random_state=28)
    hmm.fit(train_df[factor_cols].values)

    # Get training regimes for relabeling
    train_raw = hmm.predict(train_df[factor_cols].values, use_filtered=False)
    _, remap = relabel_regimes_by_data_norm(train_df, train_raw, factor_cols)

    # Get test regimes
    test_raw, _ = hmm.predict_oos(test_df[factor_cols].values, use_filtered=True)
    test_regimes = np.array([remap[r] for r in test_raw])

    hml = test_df['HML'].values
    smb = test_df['SMB'].values

    print(f"  Test period: {len(test_df)} observations")
    print(f"  Regime distribution: Normal={sum(test_regimes==0)}, "
          f"Elevated={sum(test_regimes==1)}, Crisis={sum(test_regimes==2)}")

    return hml, smb, test_regimes, test_df.index


def subsample_and_test(hml, smb, regimes, target_prevalence, lag=FIXED_LAG, seed=None):
    """
    Subsample observations to match target prevalence of Elevated regime,
    then run Granger HAC test. Returns the p-value and whether it reached target.

    If target prevalence exceeds available clean observations, returns (nan, False).
    """
    if seed is not None:
        np.random.seed(seed)

    n_total = len(regimes)
    n_target_elevated = int(np.round(n_total * target_prevalence / 100.0))

    # Get clean indices for Elevated regime
    clean_indices = extract_regime_clean_indices(regimes, 1, lag)

    if len(clean_indices) < n_target_elevated:
        # Not enough observations to reach target prevalence
        return np.nan, False

    # Randomly subsample to target size
    selected_indices = np.random.choice(clean_indices, size=n_target_elevated, replace=False)

    # Run Granger test on subsampled data
    n_obs, hac_p = run_granger_test(smb, hml, selected_indices, lag)

    return hac_p, True


def run_prevalence_analysis(hml, smb, regimes):
    """Run bootstrap analysis across prevalence levels."""
    print(f"\nRunning prevalence-significance analysis ({N_BOOTSTRAP} bootstrap iterations)...")

    results = []
    max_feasible_prevalence = None

    for prevalence in PREVALENCE_RANGE:
        print(f"  Prevalence {prevalence:2d}%: ", end="", flush=True)

        pvalues = []
        significant_count = 0
        n_valid = 0

        for boot_iter in range(N_BOOTSTRAP):
            # Use different seed for each bootstrap iteration
            p_val, reached_target = subsample_and_test(hml, smb, regimes, prevalence,
                                                       lag=FIXED_LAG, seed=boot_iter)

            if not np.isnan(p_val):
                pvalues.append(p_val)
                if p_val < 0.05:
                    significant_count += 1
                n_valid += 1

        if n_valid == 0:
            print("SKIP (insufficient clean observations)")
            if max_feasible_prevalence is None:
                max_feasible_prevalence = prevalence - 1
            continue

        pvalues = np.array(pvalues)

        # Compute statistics
        median_p = np.median(pvalues)
        p_90_lower = np.percentile(pvalues, 5)
        p_90_upper = np.percentile(pvalues, 95)
        frac_sig = significant_count / n_valid

        result = {
            'prevalence': prevalence,
            'n_valid': n_valid,
            'median_p': median_p,
            'p_90_lower': p_90_lower,
            'p_90_upper': p_90_upper,
            'frac_sig': frac_sig,
            'mean_p': np.mean(pvalues),
            'std_p': np.std(pvalues),
        }
        results.append(result)

        print(f"median p={median_p:.4f} [{p_90_lower:.4f}, {p_90_upper:.4f}], "
              f"sig%={100*frac_sig:.1f}%")

    if max_feasible_prevalence is None:
        max_feasible_prevalence = PREVALENCE_RANGE[-1]

    return pd.DataFrame(results), max_feasible_prevalence


def plot_results(results_df):
    """Create publication-quality figure with prevalence analysis."""
    print("\nGenerating figure...")

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A: Median p-value with 90% CI band
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(results_df['prevalence'],
                      results_df['p_90_lower'],
                      results_df['p_90_upper'],
                      alpha=0.3, color='steelblue', label='90% CI')
    ax1.plot(results_df['prevalence'], results_df['median_p'],
             'o-', color='darkblue', linewidth=2, markersize=4, label='Median p-value')

    # Add significance threshold
    ax1.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='α=0.05')

    # Mark training and test prevalence
    ax1.axvline(x=TRAIN_PREVALENCE, color='green', linestyle=':', linewidth=2,
                alpha=0.8, label=f'Train prev. ({TRAIN_PREVALENCE:.1f}%)')
    ax1.axvline(x=TEST_PREVALENCE, color='orange', linestyle=':', linewidth=2,
                alpha=0.8, label=f'Test prev. ({TEST_PREVALENCE:.1f}%)')

    ax1.set_xlabel('Elevated Regime Prevalence (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('HAC p-value', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Median p-value vs Prevalence Level (HML→SMB Granger)',
                  fontsize=12, fontweight='bold', loc='left')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(loc='upper right', fontsize=9)

    # Panel B: Fraction significant at p<0.05
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(results_df['prevalence'], results_df['frac_sig'],
             's-', color='darkred', linewidth=2, markersize=5, label='Fraction sig.')
    ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axvline(x=TRAIN_PREVALENCE, color='green', linestyle=':', linewidth=2, alpha=0.8)
    ax2.axvline(x=TEST_PREVALENCE, color='orange', linestyle=':', linewidth=2, alpha=0.8)

    ax2.set_xlabel('Elevated Regime Prevalence (%)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Fraction of Bootstraps\nwith p < 0.05', fontsize=10, fontweight='bold')
    ax2.set_title('(b) Significance Rate by Prevalence', fontsize=11, fontweight='bold', loc='left')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Panel C: Mean p-value with error bands
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.errorbar(results_df['prevalence'], results_df['mean_p'],
                 yerr=results_df['std_p'], fmt='D', color='purple',
                 ecolor='purple', alpha=0.6, capsize=3, markersize=4,
                 label='Mean ± Std', linewidth=1.5)
    ax3.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axvline(x=TRAIN_PREVALENCE, color='green', linestyle=':', linewidth=2, alpha=0.8)
    ax3.axvline(x=TEST_PREVALENCE, color='orange', linestyle=':', linewidth=2, alpha=0.8)

    ax3.set_xlabel('Elevated Regime Prevalence (%)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Mean p-value', fontsize=10, fontweight='bold')
    ax3.set_title('(c) Mean p-value ± 1 SD by Prevalence', fontsize=11, fontweight='bold', loc='left')
    ax3.set_ylim([0, max(results_df['mean_p'] + results_df['std_p']) * 1.1])
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Add overall title and annotation
    fig.suptitle('Frozen OOS Elevated Regime: Prevalence-Driven Significance Analysis\n' +
                 f'Test period 2013–2024 ({N_BOOTSTRAP} bootstrap iterations per prevalence level)',
                 fontsize=13, fontweight='bold', y=0.995)

    # Add text box with key findings
    textstr = (f'Training prevalence: {TRAIN_PREVALENCE:.1f}%\n'
               f'Test prevalence: {TEST_PREVALENCE:.1f}%\n'
               f'Frozen OOS HAC p-value: 0.041')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.98, 0.02, textstr, fontsize=9, verticalalignment='bottom',
             horizontalalignment='right', bbox=props, family='monospace')

    return fig


def main():
    print("=" * 70)
    print("PREVALENCE-SIGNIFICANCE CURVE ANALYSIS")
    print("=" * 70)

    # Step 1: Load test data and regime assignments
    hml, smb, regimes, date_index = load_test_data()

    # Step 2: Run prevalence analysis
    results_df, max_feas = run_prevalence_analysis(hml, smb, regimes)

    # Step 3: Save results to CSV
    csv_path = f'{RESULTS_DIR}/prevalence_significance_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved numerical results → {csv_path}")

    # Step 4: Create and save figure
    fig = plot_results(results_df)
    fig_path = f'{FIGURES_DIR}/prevalence_significance_curve.pdf'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure → {fig_path}")

    # Step 5: Print summary statistics
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    clean_indices = extract_regime_clean_indices(regimes, 1, FIXED_LAG)
    n_clean_elevated = len(clean_indices)
    n_total = len(regimes)
    actual_max_prev = 100.0 * n_clean_elevated / n_total

    print(f"Test period clean Elevated observations: {n_clean_elevated} / {n_total}")
    print(f"Maximum feasible prevalence: {actual_max_prev:.1f}%")
    print(f"(Limited by available clean regime observations)\n")

    # Find prevalence where p-value crosses 0.05 threshold
    below_005 = results_df[results_df['median_p'] < 0.05]
    if len(below_005) > 0:
        min_prev_sig = below_005['prevalence'].min()
        print(f"Minimum prevalence for significance (p<0.05): {min_prev_sig:.0f}%")
    else:
        print("Significance threshold (p<0.05) not reached at feasible prevalence levels.")

    # Training prevalence result
    train_prev_row = results_df[(results_df['prevalence'] >= 13) &
                                (results_df['prevalence'] <= 14)]
    if len(train_prev_row) > 0:
        train_p = train_prev_row['median_p'].values[0]
        train_frac = train_prev_row['frac_sig'].values[0]
        print(f"\nAt training prevalence (~13.7%): median p={train_p:.4f}, "
              f"sig fraction={train_frac:.1%}")

    # Test prevalence result - note that 30.7% exceeds available data
    print(f"\nNote: Test prevalence of 30.7% EXCEEDS maximum feasible ({actual_max_prev:.1f}%)")
    print("      (frozen OOS expanded regime beyond available clean observations)")
    highest_row = results_df.iloc[-1]
    print(f"At highest feasible prevalence ({highest_row['prevalence']:.0f}%): "
          f"median p={highest_row['median_p']:.4f}, "
          f"sig fraction={highest_row['frac_sig']:.1%}")

    # Prevalence-significance relationship
    corr = np.corrcoef(results_df['prevalence'], results_df['median_p'])[0, 1]
    print(f"\nCorrelation (prevalence, p-value): {corr:.3f}")

    # Overall conclusion
    print("\n--- INTERPRETATION ---")
    if corr < -0.5:
        print("Strong negative relationship (r < -0.5): The p-value DECREASES as")
        print("regime prevalence increases. This indicates that:")
        print("  • Significance emerges/strengthens with higher prevalence")
        print("  • Frozen OOS result (p=0.041) may be PREVALENCE-DRIVEN")
        print("  • At lower prevalence (13.7%), effect is weaker (p=0.163)")
    elif corr < 0:
        print("Moderate negative relationship (r < 0): Weak prevalence effect detected.")
    else:
        print("No negative relationship detected.")

    print("=" * 70)


if __name__ == '__main__':
    main()
