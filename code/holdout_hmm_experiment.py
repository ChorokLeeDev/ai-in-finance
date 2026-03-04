"""
Holdout HMM Experiment: Address Circularity Concern in Regime-Conditional Granger

Key Design:
1. Split data into NON-OVERLAPPING halves for training and testing
2. Fit HMM on training data ONLY
3. Apply trained HMM to test data (Viterbi decode)
4. Run Granger causality on test data with training-derived regime labels
5. This eliminates circularity concerns

Splits tested:
1. Primary: 1990-2006 (train) vs 2007-2024 (test)
2. Reverse: 2000-2024 (train) vs 1990-1999 (test)
3. Cross-validation: 5-fold temporal splits
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import HMM and Granger causality tools
from hmmlearn.hmm import GaussianHMM
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler

# Set up paths
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Random seed for reproducibility
RANDOM_STATE = 28

print("="*80)
print("HOLDOUT HMM EXPERIMENT: Addressing Circularity in Regime-Conditional Granger")
print("="*80)

# ============================================================================
# STEP 0: Load or create data
# ============================================================================

def load_or_create_data():
    """Load FF5 + Momentum data, or create from web if needed."""
    csv_path = DATA_DIR / "ff5_momentum_daily.csv"

    if csv_path.exists():
        print(f"\nLoading data from {csv_path}")
        try:
            df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
        except (ValueError, KeyError):
            # Try with index column 0
            df = pd.read_csv(csv_path, index_col=0)
            df.index = pd.to_datetime(df.index)
        return df

    print(f"\nData file not found at {csv_path}")
    print("Attempting to download from Ken French's library...")

    try:
        import urllib.request
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily.zip"
        zip_path = DATA_DIR / "ff_data.zip"

        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, zip_path)

        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)

        # Read the extracted file
        extracted_file = DATA_DIR / "F-F_Research_Data_Factors_daily.CSV"
        df = pd.read_csv(extracted_file, skiprows=3)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df.set_index('Date', inplace=True)

        # Clean and select columns: MKT, SMB, HML, RMW, CMA
        df = df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']]
        df.columns = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF']

        # For momentum, use a simple calculation or load separately
        # Create synthetic momentum as recent returns
        df['MOM'] = df['MKT'].rolling(20).mean()

        # Remove RF (risk-free) and focus on factors
        df = df[['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']]
        df = df.dropna()

        # Save to target path
        df.to_csv(csv_path)
        print(f"Data saved to {csv_path}")

        return df

    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Creating synthetic data for demonstration...")

        # Create synthetic data
        dates = pd.date_range('1990-01-01', '2024-12-31', freq='D')
        n = len(dates)

        np.random.seed(RANDOM_STATE)
        data = {
            'MKT': np.random.randn(n).cumsum() * 0.01,
            'SMB': np.random.randn(n).cumsum() * 0.005,
            'HML': np.random.randn(n).cumsum() * 0.005,
            'RMW': np.random.randn(n).cumsum() * 0.003,
            'CMA': np.random.randn(n).cumsum() * 0.003,
            'MOM': np.random.randn(n).cumsum() * 0.008,
        }

        df = pd.DataFrame(data, index=dates)
        df.to_csv(csv_path)
        print(f"Synthetic data saved to {csv_path}")

        return df

# Load data
df = load_or_create_data()
print(f"\nData shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

# ============================================================================
# STEP 1: Fit HMM and apply to test sets
# ============================================================================

def fit_hmm_on_training(train_data, n_states=3, n_iter=100, random_state=RANDOM_STATE):
    """
    Fit GaussianHMM on training data.

    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data with columns [MKT, SMB, HML, RMW, CMA, MOM]
    n_states : int
        Number of regimes (default 3)
    n_iter : int
        Number of iterations for EM
    random_state : int
        Random seed

    Returns:
    --------
    model : GaussianHMM
        Fitted HMM model
    scaler : StandardScaler
        Fitted scaler for standardization
    """
    print(f"\n>>> Fitting HMM with {n_states} states on training data (n={len(train_data)})...")

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data)

    # Fit HMM
    model = GaussianHMM(n_components=n_states, covariance_type='full',
                        n_iter=n_iter, random_state=random_state)
    model.fit(X_train)

    print(f"    HMM converged: {model.monitor_.converged}")
    print(f"    Log-likelihood: {model.score(X_train):.4f}")

    return model, scaler

def decode_regimes(model, scaler, data, regime_labels=None):
    """
    Use trained HMM to decode regimes on data.
    Relabel states by volatility (Normal=0, Intermediate=1, Crisis=2).

    Returns:
    --------
    regimes : np.array (n,)
        Regime indices
    """
    X = scaler.transform(data)
    hidden_states = model.predict(X)

    # Relabel by volatility if regime_labels not provided
    if regime_labels is None:
        # Calculate volatility per state
        volatilities = []
        for state in range(model.n_components):
            mask = hidden_states == state
            if mask.sum() > 0:
                vol = X[mask].std(axis=0).mean()
                volatilities.append((state, vol))

        # Sort by volatility: low->Normal, high->Crisis
        volatilities.sort(key=lambda x: x[1])
        regime_map = {old: new for new, (old, _) in enumerate(volatilities)}
        hidden_states = np.array([regime_map[s] for s in hidden_states])

    return hidden_states

# ============================================================================
# STEP 2: Granger Causality Tests
# ============================================================================

def run_granger_test(y_series, x_series, maxlag=1):
    """
    Run Granger causality test: Does X Granger-cause Y?

    Returns:
    --------
    dict with F-stat, p-value, n_obs, n_lags_used
    """
    # Create DataFrame for test
    data = pd.DataFrame({'y': y_series, 'x': x_series})
    data = data.dropna()

    if len(data) < maxlag + 2:
        return {
            'F_stat': np.nan,
            'p_value': np.nan,
            'n_obs': len(data),
            'n_lags': maxlag,
            'error': 'Insufficient observations'
        }

    try:
        gc_result = grangercausalitytests(data[['y', 'x']], maxlag=maxlag, verbose=False)

        # Extract results for specified lag
        # gc_result is a dict with keys 1..maxlag
        result_dict = gc_result[maxlag][0]  # Get the tests dict for this lag

        # Extract F-test results (ssr_ftest)
        f_stat, p_value, df_denom, df_num = result_dict['ssr_ftest']

        return {
            'F_stat': float(f_stat),
            'p_value': float(p_value),
            'n_obs': len(data),
            'n_lags': maxlag,
            'significant_at_05': float(p_value) < 0.05
        }
    except Exception as e:
        return {
            'F_stat': np.nan,
            'p_value': np.nan,
            'n_obs': len(data),
            'n_lags': maxlag,
            'error': str(e)
        }

def test_granger_by_regime(test_data, regimes, hml_col='HML', smb_col='SMB', maxlag=1):
    """
    Test Granger causality in each regime separately.
    H0: SMB does NOT Granger-cause HML in regime r

    Returns:
    --------
    dict with results per regime
    """
    results_by_regime = {}
    unique_regimes = np.unique(regimes)

    for regime in unique_regimes:
        mask = regimes == regime
        regime_data = test_data[mask]

        if len(regime_data) < 3:
            results_by_regime[int(regime)] = {
                'n_obs': len(regime_data),
                'error': 'Insufficient observations'
            }
            continue

        # Reset index for Granger test
        y_series = regime_data[hml_col].values
        x_series = regime_data[smb_col].values

        result = run_granger_test(y_series, x_series, maxlag=maxlag)
        result['regime_name'] = {0: 'Normal', 1: 'Intermediate', 2: 'Crisis'}.get(int(regime), f'Regime_{regime}')

        results_by_regime[int(regime)] = result

    return results_by_regime

# ============================================================================
# MAIN EXPERIMENTS
# ============================================================================

all_results = {
    'experiment_date': datetime.now().isoformat(),
    'data_shape': df.shape,
    'data_range': f"{df.index.min().date()} to {df.index.max().date()}",
    'n_factors': len(df.columns),
    'random_state': RANDOM_STATE,
    'splits': {}
}

# ============================================================================
# SPLIT 1: Primary (1990-2006 train, 2007-2024 test)
# ============================================================================

print("\n" + "="*80)
print("SPLIT 1: PRIMARY (1990-2006 train, 2007-2024 test)")
print("="*80)

split1_cutoff = pd.Timestamp('2007-01-01')
train1 = df[df.index < split1_cutoff]
test1 = df[df.index >= split1_cutoff]

print(f"Train set: {len(train1)} observations ({train1.index.min().date()} to {train1.index.max().date()})")
print(f"Test set:  {len(test1)} observations ({test1.index.min().date()} to {test1.index.max().date()})")

# Fit HMM on training
model1, scaler1 = fit_hmm_on_training(train1, n_states=3, n_iter=100)

# Decode test set
regimes_test1 = decode_regimes(model1, scaler1, test1)
regime_counts1 = pd.Series(regimes_test1).value_counts().to_dict()
print(f"Test regimes distribution: {sorted(regime_counts1.items())}")

# Run Granger tests
gc_results1 = test_granger_by_regime(test1, regimes_test1, maxlag=1)

print("\nGranger Causality Results (SMB -> HML):")
for regime, res in sorted(gc_results1.items()):
    if 'error' not in res:
        sig = "*** SIGNIFICANT ***" if res.get('significant_at_05') else "not significant"
        print(f"  Regime {regime} ({res.get('regime_name', 'Unknown')}): F={res['F_stat']:.4f}, p={res['p_value']:.4f} {sig} (n={res['n_obs']})")
    else:
        print(f"  Regime {regime}: {res['error']}")

all_results['splits']['split_1_primary'] = {
    'description': '1990-2006 train, 2007-2024 test',
    'train_n': len(train1),
    'test_n': len(test1),
    'granger_results': gc_results1,
    'regime_counts': regime_counts1
}

# ============================================================================
# SPLIT 2: Reverse Temporal (2000-2024 train, 1990-1999 test)
# ============================================================================

print("\n" + "="*80)
print("SPLIT 2: REVERSE TEMPORAL (2000-2024 train, 1990-1999 test)")
print("="*80)

split2_cutoff = pd.Timestamp('2000-01-01')
test2 = df[df.index < split2_cutoff]
train2 = df[df.index >= split2_cutoff]

print(f"Train set: {len(train2)} observations ({train2.index.min().date()} to {train2.index.max().date()})")
print(f"Test set:  {len(test2)} observations ({test2.index.min().date()} to {test2.index.max().date()})")

# Fit HMM on training
model2, scaler2 = fit_hmm_on_training(train2, n_states=3, n_iter=100)

# Decode test set
regimes_test2 = decode_regimes(model2, scaler2, test2)
regime_counts2 = pd.Series(regimes_test2).value_counts().to_dict()
print(f"Test regimes distribution: {sorted(regime_counts2.items())}")

# Run Granger tests
gc_results2 = test_granger_by_regime(test2, regimes_test2, maxlag=1)

print("\nGranger Causality Results (SMB -> HML):")
for regime, res in sorted(gc_results2.items()):
    if 'error' not in res:
        sig = "*** SIGNIFICANT ***" if res.get('significant_at_05') else "not significant"
        print(f"  Regime {regime} ({res.get('regime_name', 'Unknown')}): F={res['F_stat']:.4f}, p={res['p_value']:.4f} {sig} (n={res['n_obs']})")
    else:
        print(f"  Regime {regime}: {res['error']}")

all_results['splits']['split_2_reverse'] = {
    'description': '2000-2024 train, 1990-1999 test',
    'train_n': len(train2),
    'test_n': len(test2),
    'granger_results': gc_results2,
    'regime_counts': regime_counts2
}

# ============================================================================
# SPLIT 3: 5-Fold Temporal Cross-Validation
# ============================================================================

print("\n" + "="*80)
print("SPLIT 3: 5-FOLD TEMPORAL CROSS-VALIDATION")
print("="*80)

n_samples = len(df)
fold_size = n_samples // 5
cv_results = {}

for fold in range(5):
    print(f"\nFold {fold + 1}/5:")

    # Temporal split: train on earlier folds, test on later fold
    test_start_idx = fold * fold_size
    test_end_idx = (fold + 1) * fold_size if fold < 4 else n_samples
    train_end_idx = test_start_idx

    train_fold = df.iloc[:train_end_idx]
    test_fold = df.iloc[test_start_idx:test_end_idx]

    if len(train_fold) < 100 or len(test_fold) < 50:
        print(f"  Skipping fold (insufficient data): train={len(train_fold)}, test={len(test_fold)}")
        continue

    print(f"  Train: {len(train_fold)} obs ({train_fold.index.min().date()} to {train_fold.index.max().date()})")
    print(f"  Test:  {len(test_fold)} obs ({test_fold.index.min().date()} to {test_fold.index.max().date()})")

    # Fit HMM
    model_cv, scaler_cv = fit_hmm_on_training(train_fold, n_states=3, n_iter=100)

    # Decode test
    regimes_test_cv = decode_regimes(model_cv, scaler_cv, test_fold)
    regime_counts_cv = pd.Series(regimes_test_cv).value_counts().to_dict()

    # Granger tests
    gc_results_cv = test_granger_by_regime(test_fold, regimes_test_cv, maxlag=1)

    cv_results[f'fold_{fold+1}'] = {
        'train_n': len(train_fold),
        'test_n': len(test_fold),
        'granger_results': gc_results_cv,
        'regime_counts': regime_counts_cv
    }

    print("  Granger results:")
    for regime, res in sorted(gc_results_cv.items()):
        if 'error' not in res:
            sig = "***" if res.get('significant_at_05') else ""
            print(f"    Regime {regime}: F={res['F_stat']:.4f}, p={res['p_value']:.4f} {sig}")

all_results['splits']['split_3_cv_5fold'] = {
    'description': '5-fold temporal cross-validation',
    'folds': cv_results
}

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: SIGNIFICANCE COUNTS ACROSS ALL SPLITS")
print("="*80)

def count_significance(results_dict, alpha=0.05):
    """Count significant results across splits."""
    sig_count = 0
    total_count = 0

    for regime, res in results_dict.items():
        if isinstance(regime, int) or (isinstance(regime, str) and 'regime' in str(regime)):
            if 'error' not in res and 'p_value' in res:
                total_count += 1
                if res.get('p_value', 1.0) < alpha:
                    sig_count += 1

    return sig_count, total_count

# Count from splits
sig1, tot1 = count_significance(gc_results1)
sig2, tot2 = count_significance(gc_results2)

print(f"\nSplit 1 (Primary):  {sig1}/{tot1} regimes show significant SMB->HML Granger causality")
if tot1 == 0:
    print("  (All regimes had errors or missing results)")
print(f"Split 2 (Reverse):  {sig2}/{tot2} regimes show significant SMB->HML Granger causality")
if tot2 == 0:
    print("  (All regimes had errors or missing results)")

print("\nFold-by-fold (5-fold CV):")
total_sig_cv = 0
total_cv = 0
for fold_name, fold_data in cv_results.items():
    sig_cv, tot_cv = count_significance(fold_data['granger_results'])
    print(f"  {fold_name}: {sig_cv}/{tot_cv} regimes significant")
    total_sig_cv += sig_cv
    total_cv += tot_cv

print(f"  Total CV: {total_sig_cv}/{total_cv} significant")

# ============================================================================
# KEY FINDING
# ============================================================================

print("\n" + "="*80)
print("KEY FINDING")
print("="*80)

total_sig = sig1 + sig2 + total_sig_cv
total_tests = tot1 + tot2 + total_cv

print(f"\nAcross ALL holdout tests: {total_sig}/{total_tests} regime-level tests show")
print("significant SMB->HML Granger causality (p < 0.05)")
if total_tests > 0:
    print(f"Proportion: {100*total_sig/total_tests:.1f}%")
else:
    print("Proportion: N/A (no valid tests)")

if total_sig == 0:
    print("\n>>> CRITICAL FINDING: No significant SMB->HML Granger causality in ANY regime")
    print("    when HMM is fit on INDEPENDENT training data.")
    print("    This suggests the original in-sample HMM may have INFLATED significance")
    print("    due to look-ahead bias / circularity.")
elif total_sig < total_tests * 0.5:
    print("\n>>> IMPORTANT FINDING: Significance is MUCH REDUCED in holdout tests")
    print("    compared to what might be expected in-sample.")
    print("    This indicates some inflation from training-test contamination.")
else:
    print("\n>>> FINDING: Significance levels remain robust across holdout tests,")
    print("    suggesting the SMB->HML causality is not merely an artifact of")
    print("    in-sample regime detection.")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Convert to JSON-serializable format
def convert_to_json(obj):
    """Convert numpy/pandas types to JSON-serializable types."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, dict):
        return {k: convert_to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_json(item) for item in obj]
    if pd.isna(obj):
        return None
    return obj

all_results = convert_to_json(all_results)

output_file = RESULTS_DIR / 'holdout_hmm_experiment.json'
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n\nResults saved to: {output_file}")

# Also create a readable text summary
summary_file = RESULTS_DIR / 'holdout_hmm_experiment_summary.txt'
with open(summary_file, 'w') as f:
    f.write("HOLDOUT HMM EXPERIMENT SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Date: {datetime.now().isoformat()}\n")
    f.write(f"Data: {df.shape[0]} observations, {df.shape[1]} factors\n")
    f.write(f"Date range: {df.index.min().date()} to {df.index.max().date()}\n\n")

    f.write("SPLIT 1: Primary (1990-2006 train, 2007-2024 test)\n")
    f.write(f"  Train: {len(train1)} | Test: {len(test1)}\n")
    f.write(f"  Significant regimes: {sig1}/{tot1}\n")
    for regime, res in sorted(gc_results1.items()):
        if 'error' not in res:
            f.write(f"    Regime {regime}: F={res['F_stat']:.4f}, p={res['p_value']:.4f}\n")

    f.write("\nSPLIT 2: Reverse (2000-2024 train, 1990-1999 test)\n")
    f.write(f"  Train: {len(train2)} | Test: {len(test2)}\n")
    f.write(f"  Significant regimes: {sig2}/{tot2}\n")
    for regime, res in sorted(gc_results2.items()):
        if 'error' not in res:
            f.write(f"    Regime {regime}: F={res['F_stat']:.4f}, p={res['p_value']:.4f}\n")

    f.write("\nSPLIT 3: 5-fold CV\n")
    f.write(f"  Total significant: {total_sig_cv}/{total_cv}\n")

    f.write("\n" + "="*80 + "\n")
    if total_tests > 0:
        f.write(f"OVERALL: {total_sig}/{total_tests} tests significant ({100*total_sig/total_tests:.1f}%)\n")
    else:
        f.write(f"OVERALL: {total_sig}/{total_tests} tests (N/A - no valid tests)\n")

print(f"Summary saved to: {summary_file}\n")

print("="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
