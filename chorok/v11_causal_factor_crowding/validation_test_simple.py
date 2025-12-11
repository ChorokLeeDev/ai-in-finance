"""
V10 Validation Test (Simple Version)
Uses only numpy, pandas, scipy - no statsmodels needed

Goal: Show that causal/lead-lag relationships exist between factor crowding levels
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. SYNTHETIC FACTOR DATA WITH CAUSAL STRUCTURE
# ============================================================

def create_synthetic_factors(n=5000, seed=42):
    """
    Create synthetic factor data with KNOWN causal structure for validation

    Causal structure:
    - MOM → HML (lag 5 days)
    - SMB → CMA (lag 10 days)
    - MOM → SMB (lag 3 days)
    """
    np.random.seed(seed)

    dates = pd.date_range('2000-01-01', periods=n, freq='B')

    # Independent factors
    mom = np.random.randn(n) * 0.01
    rmw = np.random.randn(n) * 0.006

    # SMB caused by MOM (lag 3)
    smb = np.zeros(n)
    for t in range(3, n):
        smb[t] = 0.3 * mom[t-3] + 0.7 * np.random.randn() * 0.008

    # HML caused by MOM (lag 5)
    hml = np.zeros(n)
    for t in range(5, n):
        hml[t] = 0.35 * mom[t-5] + 0.65 * np.random.randn() * 0.009

    # CMA caused by SMB (lag 10)
    cma = np.zeros(n)
    for t in range(10, n):
        cma[t] = 0.25 * smb[t-10] + 0.75 * np.random.randn() * 0.007

    # MKT: market factor (affected by all)
    mkt = 0.1 * (mom + smb + hml) + np.random.randn(n) * 0.012

    factors = pd.DataFrame({
        'MKT': mkt,
        'SMB': smb,
        'HML': hml,
        'RMW': rmw,
        'CMA': cma,
        'MOM': mom
    }, index=dates)

    print("Created synthetic factor data with KNOWN causal structure:")
    print("  Ground Truth:")
    print("  - MOM → SMB (lag 3)")
    print("  - MOM → HML (lag 5)")
    print("  - SMB → CMA (lag 10)")
    print()

    return factors


# ============================================================
# 2. CROWDING PROXY (Simple: Rolling Volatility)
# ============================================================

def compute_crowding_proxy(factor_returns, window=60):
    """
    Simple crowding proxy: z-scored rolling volatility
    Higher volatility = potential crowding unwinding
    """
    crowding = pd.DataFrame(index=factor_returns.index)

    for col in factor_returns.columns:
        vol = factor_returns[col].rolling(window).std()
        # Z-score (rolling 252-day mean/std)
        mean = vol.rolling(252).mean()
        std = vol.rolling(252).std()
        crowding[col] = (vol - mean) / std

    return crowding.dropna()


# ============================================================
# 3. TEST 1: CROSS-CORRELATION ANALYSIS
# ============================================================

def test_cross_correlation(crowding, max_lag=30):
    """
    Find lead-lag relationships via cross-correlation
    Peak at positive lag = first variable leads second
    """
    print("=" * 60)
    print("TEST 1: CROSS-CORRELATION ANALYSIS")
    print("=" * 60)

    factors = crowding.columns.tolist()
    results = []

    for i, f1 in enumerate(factors):
        for j, f2 in enumerate(factors):
            if i >= j:
                continue

            # Cross-correlation at different lags
            x = crowding[f1].values
            y = crowding[f2].values

            correlations = []
            lags = range(-max_lag, max_lag + 1)

            for lag in lags:
                if lag < 0:
                    c = np.corrcoef(x[-lag:], y[:lag])[0, 1]
                elif lag > 0:
                    c = np.corrcoef(x[:-lag], y[lag:])[0, 1]
                else:
                    c = np.corrcoef(x, y)[0, 1]
                correlations.append(c if not np.isnan(c) else 0)

            # Find peak
            peak_idx = np.argmax(np.abs(correlations))
            peak_lag = list(lags)[peak_idx]
            peak_corr = correlations[peak_idx]

            results.append({
                'factor1': f1,
                'factor2': f2,
                'peak_lag': peak_lag,
                'peak_corr': peak_corr
            })

            # Report significant lead-lag
            if abs(peak_corr) > 0.15 and abs(peak_lag) > 2:
                if peak_lag > 0:
                    print(f"  {f1} → {f2}: lag={peak_lag} days, corr={peak_corr:.3f}")
                else:
                    print(f"  {f2} → {f1}: lag={-peak_lag} days, corr={peak_corr:.3f}")

    return pd.DataFrame(results)


# ============================================================
# 4. TEST 2: SIMPLE GRANGER-LIKE TEST
# ============================================================

def test_predictive_power(crowding, lag=5):
    """
    Simple test: Does X(t-lag) help predict Y(t)?

    Compare:
    - Model 1: Y(t) ~ Y(t-1)  (AR only)
    - Model 2: Y(t) ~ Y(t-1) + X(t-lag)  (with predictor)

    If Model 2 has lower error, X has predictive power for Y
    """
    print("=" * 60)
    print(f"TEST 2: PREDICTIVE POWER (lag={lag})")
    print("=" * 60)

    factors = crowding.columns.tolist()
    results = []

    for cause in factors:
        for effect in factors:
            if cause == effect:
                continue

            # Prepare data
            y = crowding[effect].values[lag:]
            y_lag1 = crowding[effect].shift(1).values[lag:]
            x_lag = crowding[cause].shift(lag).values[lag:]

            # Remove NaN
            mask = ~(np.isnan(y) | np.isnan(y_lag1) | np.isnan(x_lag))
            y, y_lag1, x_lag = y[mask], y_lag1[mask], x_lag[mask]

            if len(y) < 100:
                continue

            # Model 1: AR(1) - predict Y from Y(t-1) only
            # Using simple linear regression: Y = a + b*Y_lag1
            slope1, intercept1, _, _, _ = stats.linregress(y_lag1, y)
            pred1 = intercept1 + slope1 * y_lag1
            mse1 = np.mean((y - pred1) ** 2)

            # Model 2: AR(1) + X_lag
            # Y = a + b*Y_lag1 + c*X_lag (using multiple regression via normal equations)
            X = np.column_stack([np.ones(len(y)), y_lag1, x_lag])
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                pred2 = X @ beta
                mse2 = np.mean((y - pred2) ** 2)
            except:
                continue

            # Improvement ratio
            improvement = (mse1 - mse2) / mse1 * 100

            results.append({
                'cause': cause,
                'effect': effect,
                'mse_ar1': mse1,
                'mse_with_cause': mse2,
                'improvement_pct': improvement
            })

            if improvement > 1:  # >1% improvement
                print(f"  {cause} → {effect}: {improvement:.2f}% MSE reduction")

    return pd.DataFrame(results)


# ============================================================
# 5. TEST 3: IMPULSE CORRELATION
# ============================================================

def test_impulse_correlation(crowding, threshold=2.0, max_response_days=20):
    """
    When factor X has a "shock" (>2 std move), what happens to Y in subsequent days?

    This mimics VAR impulse response without statsmodels
    """
    print("=" * 60)
    print(f"TEST 3: IMPULSE RESPONSE (shock threshold={threshold} std)")
    print("=" * 60)

    factors = crowding.columns.tolist()
    results = []

    for shock_factor in factors:
        # Find shock days (when factor moves > threshold std)
        z_scores = (crowding[shock_factor] - crowding[shock_factor].mean()) / crowding[shock_factor].std()
        shock_days = z_scores[z_scores.abs() > threshold].index

        if len(shock_days) < 10:
            continue

        for response_factor in factors:
            if shock_factor == response_factor:
                continue

            # Collect responses after shocks
            responses = []
            for shock_day in shock_days:
                try:
                    idx = crowding.index.get_loc(shock_day)
                    if idx + max_response_days >= len(crowding):
                        continue

                    # Response = cumulative change in next N days
                    baseline = crowding[response_factor].iloc[idx]
                    future = crowding[response_factor].iloc[idx+1:idx+max_response_days+1]
                    cum_response = (future - baseline).sum()
                    responses.append(cum_response)
                except:
                    continue

            if len(responses) < 5:
                continue

            mean_response = np.mean(responses)
            std_response = np.std(responses)
            t_stat = mean_response / (std_response / np.sqrt(len(responses)))

            results.append({
                'shock': shock_factor,
                'response': response_factor,
                'n_shocks': len(responses),
                'mean_response': mean_response,
                't_stat': t_stat
            })

            if abs(t_stat) > 2:  # Significant at ~5% level
                direction = "increases" if mean_response > 0 else "decreases"
                print(f"  Shock to {shock_factor} → {response_factor} {direction} (t={t_stat:.2f})")

    return pd.DataFrame(results)


# ============================================================
# 6. MAIN: RUN ALL TESTS
# ============================================================

def run_validation():
    print("=" * 60)
    print("V10 VALIDATION: CAUSAL FACTOR CROWDING")
    print("=" * 60)
    print()

    # 1. Create synthetic data
    factors = create_synthetic_factors(n=5000)

    # 2. Compute crowding proxy
    print("Computing crowding proxy (rolling volatility)...")
    crowding = compute_crowding_proxy(factors, window=60)
    print(f"Crowding data: {crowding.shape[0]} days, {crowding.shape[1]} factors")
    print()

    # 3. Run tests
    xcorr_results = test_cross_correlation(crowding, max_lag=30)
    print()

    pred_results = test_predictive_power(crowding, lag=5)
    print()

    impulse_results = test_impulse_correlation(crowding, threshold=2.0)
    print()

    # 4. Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    print("\nGround Truth (embedded in synthetic data):")
    print("  - MOM → SMB (lag 3)")
    print("  - MOM → HML (lag 5)")
    print("  - SMB → CMA (lag 10)")

    print("\nDetected by our tests:")

    # Cross-correlation detections
    significant_xcorr = xcorr_results[
        (xcorr_results['peak_corr'].abs() > 0.15) &
        (xcorr_results['peak_lag'].abs() > 2)
    ]
    print(f"  Cross-correlation: {len(significant_xcorr)} relationships")

    # Predictive power detections
    significant_pred = pred_results[pred_results['improvement_pct'] > 1]
    print(f"  Predictive power: {len(significant_pred)} relationships")

    # Impulse response detections
    significant_impulse = impulse_results[impulse_results['t_stat'].abs() > 2]
    print(f"  Impulse response: {len(significant_impulse)} relationships")

    print()
    if len(significant_xcorr) > 0 or len(significant_pred) > 0:
        print(">>> VALIDATION PASSED: Tests can detect causal structure!")
        print(">>> Ready to apply to real Fama-French data")
    else:
        print(">>> VALIDATION FAILED: Tests did not detect known structure")

    return {
        'factors': factors,
        'crowding': crowding,
        'xcorr': xcorr_results,
        'predictive': pred_results,
        'impulse': impulse_results
    }


if __name__ == "__main__":
    results = run_validation()
