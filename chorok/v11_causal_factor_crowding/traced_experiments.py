"""
TRACED: Comprehensive Synthetic Experiments for NeurIPS

Run systematic experiments comparing TRACED against baselines:
1. Single DAG Granger (no regimes)
2. Gaussian HMM + Granger (two-stage, Gaussian)
3. TRACED (Student-t HMM + Granger)

Key hypothesis: TRACED outperforms when data has heavy tails (low nu).
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from traced_synthetic import (
    generate_regime_switching_dag_data,
    dag_metrics,
    regime_metrics,
    baseline_single_dag_granger,
    baseline_gaussian_hmm_then_granger,
    traced_method
)


def run_comprehensive_experiment():
    """
    Comprehensive experiment for NeurIPS submission.

    Variables:
    - nu: [3, 5, 7, 10, 15, 30] (tail heaviness)
    - n_trials: 10 per setting
    - n_samples: 3000
    """

    nu_values = [3, 5, 7, 10, 15, 30]
    n_trials = 10
    n_samples = 3000
    n_variables = 6

    results = []

    print("=" * 70)
    print("TRACED: Comprehensive Synthetic Experiments")
    print("=" * 70)
    print(f"Settings: nu={nu_values}, trials={n_trials}, samples={n_samples}")

    for nu in nu_values:
        print(f"\n{'='*70}")
        print(f"NU = {nu} (lower = heavier tails)")
        print(f"{'='*70}")

        for trial in range(n_trials):
            print(f"  Trial {trial + 1}/{n_trials}...", end=" ", flush=True)

            # Generate data
            data = generate_regime_switching_dag_data(
                n_samples=n_samples,
                n_variables=n_variables,
                n_regimes=3,
                nu_per_regime=[nu, nu, nu],
                sparsity=0.3,
                seed=42 + trial * 100 + nu  # Different seed per (nu, trial)
            )

            X = data['X']
            true_regimes = data['regimes']
            true_W = data['W']

            # Check for NaN/Inf
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                print("SKIPPED (NaN/Inf in data)")
                continue

            # Method 1: Single DAG
            try:
                W_single = baseline_single_dag_granger(X, max_lag=5)
                single_ok = True
            except Exception as e:
                W_single = np.zeros((n_variables, n_variables))
                single_ok = False

            # Method 2: Gaussian HMM
            try:
                W_gaussian, regimes_gaussian = baseline_gaussian_hmm_then_granger(X, n_regimes=3)
                gaussian_ok = True
            except Exception as e:
                W_gaussian = np.zeros((3, n_variables, n_variables))
                regimes_gaussian = np.zeros(len(X), dtype=int)
                gaussian_ok = False

            # Method 3: TRACED
            try:
                W_traced, regimes_traced, hmm = traced_method(X, n_regimes=3, n_iter=30)
                traced_ok = True
                traced_ll = hmm.log_likelihood_
            except Exception as e:
                W_traced = np.zeros((3, n_variables, n_variables))
                regimes_traced = np.zeros(len(X), dtype=int)
                traced_ok = False
                traced_ll = np.nan

            # Evaluate
            for method_name, W_pred, regimes_pred, ok in [
                ('Single DAG', np.stack([W_single]*3), np.zeros_like(true_regimes), single_ok),
                ('Gaussian HMM', W_gaussian, regimes_gaussian, gaussian_ok),
                ('TRACED', W_traced, regimes_traced, traced_ok)
            ]:
                if not ok:
                    continue

                # DAG metrics (average across regimes)
                shd_list = []
                f1_list = []
                tpr_list = []
                fdr_list = []

                for k in range(3):
                    metrics = dag_metrics(true_W[k], W_pred[k])
                    shd_list.append(metrics['SHD'])
                    f1_list.append(metrics['F1'])
                    tpr_list.append(metrics['TPR'])
                    fdr_list.append(metrics['FDR'])

                # Regime metrics
                reg_metrics = regime_metrics(true_regimes, regimes_pred)

                results.append({
                    'nu': nu,
                    'trial': trial,
                    'method': method_name,
                    'SHD': np.mean(shd_list),
                    'F1': np.mean(f1_list),
                    'TPR': np.mean(tpr_list),
                    'FDR': np.mean(fdr_list),
                    'NMI': reg_metrics['NMI'],
                    'ARI': reg_metrics['ARI']
                })

            print("OK")

    return pd.DataFrame(results)


def analyze_results(df):
    """Analyze and display results."""

    print("\n" + "=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)

    # Summary statistics
    summary = df.groupby(['nu', 'method']).agg({
        'SHD': ['mean', 'std'],
        'F1': ['mean', 'std'],
        'TPR': ['mean', 'std'],
        'NMI': ['mean', 'std'],
        'ARI': ['mean', 'std']
    }).round(3)

    print("\n1. Full Summary Table:")
    print(summary)

    # Key comparison: TRACED vs Gaussian at different nu
    print("\n" + "-" * 70)
    print("2. TRACED vs Gaussian HMM (F1 Score)")
    print("-" * 70)

    for nu in sorted(df['nu'].unique()):
        traced = df[(df['method'] == 'TRACED') & (df['nu'] == nu)]
        gaussian = df[(df['method'] == 'Gaussian HMM') & (df['nu'] == nu)]

        if len(traced) == 0 or len(gaussian) == 0:
            continue

        traced_f1 = traced['F1'].mean()
        gaussian_f1 = gaussian['F1'].mean()
        diff = traced_f1 - gaussian_f1

        # Statistical test
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(traced['F1'].values, gaussian['F1'].values)

        winner = "TRACED" if diff > 0 else "Gaussian"
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        print(f"  nu={nu:2d}: TRACED={traced_f1:.3f}, Gaussian={gaussian_f1:.3f}, "
              f"diff={diff:+.3f} ({winner}) p={p_val:.3f}{sig}")

    # Regime detection
    print("\n" + "-" * 70)
    print("3. Regime Detection (NMI)")
    print("-" * 70)

    for nu in sorted(df['nu'].unique()):
        traced = df[(df['method'] == 'TRACED') & (df['nu'] == nu)]
        gaussian = df[(df['method'] == 'Gaussian HMM') & (df['nu'] == nu)]

        if len(traced) == 0 or len(gaussian) == 0:
            continue

        traced_nmi = traced['NMI'].mean()
        gaussian_nmi = gaussian['NMI'].mean()

        print(f"  nu={nu:2d}: TRACED NMI={traced_nmi:.3f}, Gaussian NMI={gaussian_nmi:.3f}")

    return summary


def create_latex_table(df):
    """Create LaTeX table for paper."""

    print("\n" + "=" * 70)
    print("LATEX TABLE")
    print("=" * 70)

    # Pivot for clean table
    pivot = df.groupby(['nu', 'method'])['F1'].agg(['mean', 'std']).round(3)

    print("""
\\begin{table}[t]
\\centering
\\caption{DAG Recovery (F1 Score) across tail heaviness}
\\label{tab:synthetic}
\\begin{tabular}{lcccccc}
\\toprule
Method & $\\nu=3$ & $\\nu=5$ & $\\nu=7$ & $\\nu=10$ & $\\nu=15$ & $\\nu=30$ \\\\
\\midrule""")

    for method in ['Single DAG', 'Gaussian HMM', 'TRACED']:
        row = method
        for nu in [3, 5, 7, 10, 15, 30]:
            try:
                mean = pivot.loc[(nu, method), 'mean']
                std = pivot.loc[(nu, method), 'std']
                row += f" & {mean:.3f}±{std:.3f}"
            except:
                row += " & -"
        row += " \\\\"
        print(row)

    print("""\\bottomrule
\\end{tabular}
\\end{table}""")


if __name__ == "__main__":
    # Run experiments
    results_df = run_comprehensive_experiment()

    # Save raw results
    results_df.to_csv('/Users/i767700/Github/ai-in-finance/chorok/v11_causal_factor_crowding/traced_results.csv', index=False)
    print("\nResults saved to traced_results.csv")

    # Analyze
    summary = analyze_results(results_df)

    # LaTeX
    create_latex_table(results_df)

    # Key takeaway
    print("\n" + "=" * 70)
    print("KEY TAKEAWAY")
    print("=" * 70)

    # Check if TRACED wins at low nu
    low_nu = results_df[results_df['nu'] <= 5]
    high_nu = results_df[results_df['nu'] >= 15]

    traced_low = low_nu[low_nu['method'] == 'TRACED']['F1'].mean()
    gaussian_low = low_nu[low_nu['method'] == 'Gaussian HMM']['F1'].mean()

    traced_high = high_nu[high_nu['method'] == 'TRACED']['F1'].mean()
    gaussian_high = high_nu[high_nu['method'] == 'Gaussian HMM']['F1'].mean()

    print(f"""
Heavy tails (nu <= 5):
  TRACED F1: {traced_low:.3f}
  Gaussian F1: {gaussian_low:.3f}
  → {'TRACED wins' if traced_low > gaussian_low else 'Gaussian wins'} by {abs(traced_low - gaussian_low):.3f}

Light tails (nu >= 15):
  TRACED F1: {traced_high:.3f}
  Gaussian F1: {gaussian_high:.3f}
  → {'TRACED wins' if traced_high > gaussian_high else 'Gaussian wins'} by {abs(traced_high - gaussian_high):.3f}

Hypothesis: TRACED should win at heavy tails (low nu) and be competitive at light tails.
""")
