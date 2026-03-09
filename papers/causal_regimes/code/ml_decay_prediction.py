"""
ML Decay Prediction Model - Survival Analysis for Factor Pair Decay
====================================================================

Predict WHICH factor pairs will show Granger causality decay and WHEN.

Models:
1. Cox Proportional Hazards (interpretable baseline)
2. Random Survival Forest (flexible nonparametric)
3. Logistic Regression (binary: will decay within 5 years?)

Features:
- Initial Granger F-statistic strength
- Regime volatility ratio (Crisis/Normal)
- Factor correlation stability
- Rolling significance persistence
- F-stat momentum and change rate

Validation: Leave-group-out (train 20 pairs, test 10 pairs)
Metrics: C-index (concordance), AUC, calibration
"""

import numpy as np
import pandas as pd
import urllib.request
import zipfile
import io
import json
from datetime import datetime
from scipy import stats
from scipy.cluster.vq import kmeans2
from scipy.optimize import curve_fit
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    brier_score_loss
)
import warnings
warnings.filterwarnings('ignore')

# Survival analysis imports
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored
    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False


# =============================================================================
# DATA LOADING
# =============================================================================

def download_ff_factors():
    """Download FF 5 factors + Momentum."""
    url_5f = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'
    with urllib.request.urlopen(url_5f, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df_5f = pd.read_csv(f, skiprows=3)

    df_5f.columns = df_5f.columns.str.strip()
    df_5f = df_5f.rename(columns={df_5f.columns[0]: 'Date'})
    df_5f = df_5f[df_5f['Date'].astype(str).str.match(r'^\d{8}$')]
    df_5f['Date'] = pd.to_datetime(df_5f['Date'], format='%Y%m%d')
    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
        df_5f[col] = pd.to_numeric(df_5f[col], errors='coerce')
    df_5f = df_5f.set_index('Date')

    url_mom = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip'
    with urllib.request.urlopen(url_mom, timeout=60) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df_mom = pd.read_csv(f, skiprows=13)

    df_mom.columns = df_mom.columns.str.strip()
    df_mom = df_mom.rename(columns={df_mom.columns[0]: 'Date', df_mom.columns[1]: 'MOM'})
    df_mom = df_mom[df_mom['Date'].astype(str).str.match(r'^\d{8}$')]
    df_mom['Date'] = pd.to_datetime(df_mom['Date'], format='%Y%m%d')
    df_mom['MOM'] = pd.to_numeric(df_mom['MOM'], errors='coerce')
    df_mom = df_mom.set_index('Date')[['MOM']]

    df = df_5f.join(df_mom, how='inner')
    return df.dropna()


# =============================================================================
# REGIME DETECTION
# =============================================================================

class StudentTHMM:
    """Minimal Student-t HMM for regime detection via K-means."""

    def __init__(self, n_regimes=3, random_state=28):
        self.n_regimes = n_regimes
        self.random_state = random_state

    def fit_predict(self, X):
        np.random.seed(self.random_state)
        X = np.asarray(X)
        K = self.n_regimes

        centroids, labels = kmeans2(X, K, minit='++')
        norms = np.linalg.norm(centroids, axis=1)
        order = np.argsort(norms)

        # Reorder: 0=normal, 1=elevated, 2=crisis
        new_labels = np.zeros_like(labels)
        for new_k, old_k in enumerate(order):
            new_labels[labels == old_k] = new_k

        return new_labels


# =============================================================================
# GRANGER CAUSALITY
# =============================================================================

def granger_test_hac(Y, X, max_lag=5):
    """Granger causality test with HAC standard errors."""
    T = len(Y)
    if T < max_lag + 20:
        return None

    Y_lags = np.column_stack([Y[max_lag-i-1:T-i-1] for i in range(max_lag)])
    X_lags = np.column_stack([X[max_lag-i-1:T-i-1] for i in range(max_lag)])
    Y_target = Y[max_lag:]

    X_restricted = sm.add_constant(Y_lags)
    X_unrestricted = sm.add_constant(np.column_stack([Y_lags, X_lags]))

    try:
        model_r = sm.OLS(Y_target, X_restricted).fit(cov_type='HAC', cov_kwds={'maxlags': max_lag})
        model_u = sm.OLS(Y_target, X_unrestricted).fit(cov_type='HAC', cov_kwds={'maxlags': max_lag})

        rss_r = (model_r.resid ** 2).sum()
        rss_u = (model_u.resid ** 2).sum()

        n = len(Y_target)
        k_r = X_restricted.shape[1]
        k_u = X_unrestricted.shape[1]

        F = ((rss_r - rss_u) / (k_u - k_r)) / (rss_u / (n - k_u))
        p_value = 1 - stats.f.cdf(F, k_u - k_r, n - k_u)

        return {'F': F, 'p_value': p_value}
    except Exception:
        return None


def compute_rolling_granger(df, source, target, regimes, regime_idx=0,
                            window_years=5, step_months=6):
    """Compute rolling 5-year Granger F-statistics."""
    results = []
    dates = df.index

    for year in range(1995, 2024):
        for month in [1, 7]:
            window_end = pd.Timestamp(f'{year}-{month:02d}-01')
            window_start = window_end - pd.DateOffset(years=window_years)

            if window_end > dates[-1] or window_start < dates[0]:
                continue

            mask = (dates >= window_start) & (dates < window_end) & (regimes == regime_idx)

            if mask.sum() < 50:
                continue

            Y = df.loc[mask, target].values
            X = df.loc[mask, source].values

            result = granger_test_hac(Y, X)
            if result:
                results.append({
                    'year': year + (month - 1) / 12,
                    'window_end': window_end,
                    'F': result['F'],
                    'p_value': result['p_value'],
                    'n_obs': mask.sum()
                })

    return pd.DataFrame(results)


# =============================================================================
# FEATURE EXTRACTION FOR SURVIVAL ANALYSIS
# =============================================================================

def extract_pair_features(df, source, target, regimes, rolling_df):
    """
    Extract time-invariant features for each factor pair for survival analysis.

    These features characterize the PAIR at baseline (first observation window).
    """
    if len(rolling_df) < 5:
        return None

    # Get baseline window data
    baseline_F = rolling_df.iloc[0]['F']
    early_F = rolling_df.iloc[:5]['F'].values

    # Feature 1: Initial Granger F-statistic
    initial_F = baseline_F

    # Feature 2: Initial F-stat volatility (first 5 windows)
    initial_F_vol = np.std(early_F)

    # Feature 3: Initial F-stat mean
    initial_F_mean = np.mean(early_F)

    # Get baseline period data for factor characteristics
    baseline_end = rolling_df.iloc[0]['window_end']
    baseline_start = baseline_end - pd.DateOffset(years=5)
    baseline_mask = (df.index >= baseline_start) & (df.index < baseline_end)

    if baseline_mask.sum() < 50:
        return None

    baseline_df = df.loc[baseline_mask]
    baseline_regimes = regimes[baseline_mask]

    # Feature 4: Regime volatility ratio (Crisis vol / Normal vol)
    crisis_vol = baseline_df.loc[baseline_regimes == 2, source].std() if (baseline_regimes == 2).sum() > 10 else 0
    normal_vol = baseline_df.loc[baseline_regimes == 0, source].std() if (baseline_regimes == 0).sum() > 10 else 1
    vol_ratio_source = crisis_vol / normal_vol if normal_vol > 0 else 1

    crisis_vol_t = baseline_df.loc[baseline_regimes == 2, target].std() if (baseline_regimes == 2).sum() > 10 else 0
    normal_vol_t = baseline_df.loc[baseline_regimes == 0, target].std() if (baseline_regimes == 0).sum() > 10 else 1
    vol_ratio_target = crisis_vol_t / normal_vol_t if normal_vol_t > 0 else 1

    # Feature 5: Factor correlation stability
    first_half = baseline_df.iloc[:len(baseline_df)//2]
    second_half = baseline_df.iloc[len(baseline_df)//2:]
    corr_first = first_half[source].corr(first_half[target])
    corr_second = second_half[source].corr(second_half[target])
    corr_stability = 1 - abs(corr_first - corr_second)  # Higher = more stable

    # Feature 6: Initial factor correlation
    initial_corr = baseline_df[source].corr(baseline_df[target])

    # Feature 7: Rolling significance persistence (% windows p < 0.05)
    sig_rate = (rolling_df['p_value'] < 0.05).mean()

    # Feature 8: Normal regime proportion
    normal_pct = (baseline_regimes == 0).mean()

    # Feature 9: Crisis regime proportion
    crisis_pct = (baseline_regimes == 2).mean()

    # Feature 10: Factor volatility ratio (source/target)
    source_vol = baseline_df[source].std()
    target_vol = baseline_df[target].std()
    factor_vol_ratio = source_vol / target_vol if target_vol > 0 else 1

    # Feature 11: Initial F-stat trend (slope over first 5 windows)
    if len(early_F) >= 3:
        t = np.arange(len(early_F))
        slope, _ = np.polyfit(t, early_F, 1)
        initial_trend = slope
    else:
        initial_trend = 0

    return {
        'initial_F': initial_F,
        'initial_F_vol': initial_F_vol,
        'initial_F_mean': initial_F_mean,
        'vol_ratio_source': vol_ratio_source,
        'vol_ratio_target': vol_ratio_target,
        'corr_stability': corr_stability,
        'initial_corr': initial_corr,
        'sig_rate': sig_rate,
        'normal_pct': normal_pct,
        'crisis_pct': crisis_pct,
        'factor_vol_ratio': factor_vol_ratio,
        'initial_trend': initial_trend
    }


def compute_decay_event(rolling_df, decay_threshold=0.5, min_initial_F=2.5):
    """
    Compute survival data: time-to-decay event.

    Decay event = F-stat drops significantly from peak level

    Definition:
    - Pairs need at least some initial significance (F > 2.5)
    - Event = F drops to < 50% of early peak

    Returns: (duration, event_occurred)
    """
    if len(rolling_df) < 5:
        return None, None

    # Use first 5 windows to establish "initial" significance
    early_F = rolling_df.iloc[:5]['F'].values
    peak_F = np.max(early_F)
    mean_early_F = np.mean(early_F)

    # If pair was never significant, censor immediately
    if peak_F < min_initial_F:
        duration = rolling_df.iloc[-1]['year'] - rolling_df.iloc[0]['year']
        return duration, False

    # Decay threshold: F drops below 50% of peak
    threshold = peak_F * (1 - decay_threshold)

    start_year = rolling_df.iloc[0]['year']

    # Find first time F drops below threshold (after window 3 to avoid noise)
    for idx in range(3, len(rolling_df)):
        row = rolling_df.iloc[idx]
        if row['F'] < threshold:
            duration = row['year'] - start_year
            return max(duration, 0.5), True

    # Censored: no decay observed
    duration = rolling_df.iloc[-1]['year'] - start_year
    return duration, False


def extract_time_varying_features(df, source, target, regimes, rolling_df):
    """
    Extract time-varying features for logistic regression at each time point.
    """
    if len(rolling_df) < 5:
        return None

    features_list = []

    for i in range(2, len(rolling_df)):
        row = rolling_df.iloc[i]
        window_end = row['window_end']

        hist_F = rolling_df.iloc[:i+1]['F'].values

        current_F = row['F']
        F_volatility = np.std(hist_F[-3:]) if len(hist_F) >= 3 else 0
        F_mean = np.mean(hist_F[-3:]) if len(hist_F) >= 3 else current_F

        lookback = pd.DateOffset(years=1)
        recent_mask = (df.index >= window_end - lookback) & (df.index < window_end)

        if recent_mask.sum() < 50:
            continue

        recent_df = df.loc[recent_mask]
        recent_regimes = regimes[recent_mask]

        normal_pct = (recent_regimes == 0).mean()
        crisis_pct = (recent_regimes == 2).mean()
        factor_corr = recent_df[source].corr(recent_df[target])

        source_vol = recent_df[source].std()
        target_vol = recent_df[target].std()
        vol_ratio = source_vol / target_vol if target_vol > 0 else 1

        if len(hist_F) >= 3:
            t = np.arange(len(hist_F[-3:]))
            slope, _ = np.polyfit(t, hist_F[-3:], 1)
            F_momentum = slope
        else:
            F_momentum = 0

        F_relative = current_F / np.max(hist_F) if np.max(hist_F) > 0 else 1
        time_idx = i / len(rolling_df)

        if i > 0:
            prev_F = rolling_df.iloc[i-1]['F']
            F_change = (current_F - prev_F) / prev_F if prev_F > 0 else 0
        else:
            F_change = 0

        features_list.append({
            'year': row['year'],
            'window_end': window_end,
            'current_F': current_F,
            'F_volatility': F_volatility,
            'F_mean': F_mean,
            'normal_pct': normal_pct,
            'crisis_pct': crisis_pct,
            'factor_corr': factor_corr,
            'vol_ratio': vol_ratio,
            'F_momentum': F_momentum,
            'F_relative': F_relative,
            'time_idx': time_idx,
            'F_change': F_change
        })

    return pd.DataFrame(features_list)


def label_future_decay(rolling_df, horizon_years=5, decay_pct=0.5):
    """Label whether significant decay occurs within horizon."""
    labels = []

    for i in range(len(rolling_df)):
        current_F = rolling_df.iloc[i]['F']
        current_year = rolling_df.iloc[i]['year']

        future_mask = (rolling_df['year'] > current_year) & \
                      (rolling_df['year'] <= current_year + horizon_years)

        if future_mask.sum() == 0:
            labels.append(np.nan)
            continue

        future_F = rolling_df.loc[future_mask, 'F'].values
        min_future_F = np.min(future_F)
        threshold = current_F * (1 - decay_pct)

        decayed = int(min_future_F < threshold)
        labels.append(decayed)

    return labels


# =============================================================================
# BUILD SURVIVAL DATASET
# =============================================================================

def build_survival_dataset(df, factors, regimes):
    """Build dataset for survival analysis (one row per pair)."""
    survival_data = []

    for source in factors:
        for target in factors:
            if source == target:
                continue

            pair_name = f"{source}->{target}"
            rolling_df = compute_rolling_granger(df, source, target, regimes)

            if len(rolling_df) < 10:
                continue

            # Extract pair-level features
            features = extract_pair_features(df, source, target, regimes, rolling_df)
            if features is None:
                continue

            # Compute survival outcome
            duration, event = compute_decay_event(rolling_df)
            if duration is None:
                continue

            features['pair'] = pair_name
            features['source'] = source
            features['target'] = target
            features['duration'] = duration
            features['event'] = int(event)

            survival_data.append(features)

    return pd.DataFrame(survival_data)


def build_classification_dataset(df, factors, regimes):
    """Build dataset for classification models (multiple rows per pair)."""
    all_pairs_data = []
    pair_info = []

    for source in factors:
        for target in factors:
            if source == target:
                continue

            pair_name = f"{source}->{target}"
            rolling_df = compute_rolling_granger(df, source, target, regimes)

            if len(rolling_df) < 10:
                continue

            features_df = extract_time_varying_features(df, source, target, regimes, rolling_df)
            if features_df is None or len(features_df) < 5:
                continue

            labels = label_future_decay(rolling_df)
            features_df['decay_label'] = labels[2:len(features_df)+2]
            features_df['pair'] = pair_name
            features_df['source'] = source
            features_df['target'] = target

            features_df = features_df.dropna(subset=['decay_label'])

            if len(features_df) > 0:
                all_pairs_data.append(features_df)
                pair_info.append({
                    'pair': pair_name,
                    'n_samples': len(features_df),
                    'decay_rate': features_df['decay_label'].mean()
                })

    if len(all_pairs_data) == 0:
        return None, None

    dataset = pd.concat(all_pairs_data, ignore_index=True)
    return dataset, pair_info


# =============================================================================
# SURVIVAL ANALYSIS MODELS
# =============================================================================

def train_cox_ph(train_data, test_data, feature_cols):
    """Train Cox Proportional Hazards model."""
    if not LIFELINES_AVAILABLE:
        return None

    # Prepare data for lifelines
    train_survival = train_data[feature_cols + ['duration', 'event']].copy()
    test_survival = test_data[feature_cols + ['duration', 'event']].copy()

    # Fit Cox PH
    cph = CoxPHFitter(penalizer=0.1)
    try:
        cph.fit(train_survival, duration_col='duration', event_col='event')
    except Exception as e:
        print(f"Cox PH fitting failed: {e}")
        return None

    # Evaluate on test set
    test_risk = cph.predict_partial_hazard(test_survival[feature_cols])

    # Concordance index
    c_index = concordance_index(
        test_survival['duration'],
        -test_risk.values.flatten(),  # Negative because higher risk = earlier event
        test_survival['event']
    )

    # Feature importance (hazard ratios)
    hazard_ratios = cph.summary[['exp(coef)', 'p']].copy()
    hazard_ratios.columns = ['hazard_ratio', 'p_value']

    return {
        'model': cph,
        'c_index': c_index,
        'hazard_ratios': hazard_ratios,
        'test_risk': test_risk
    }


def train_random_survival_forest(train_data, test_data, feature_cols, random_state=42):
    """Train Random Survival Forest."""
    if not SKSURV_AVAILABLE:
        return None

    X_train = train_data[feature_cols].values
    X_test = test_data[feature_cols].values

    # Convert to structured array for sksurv
    y_train = np.array(
        [(bool(e), d) for e, d in zip(train_data['event'], train_data['duration'])],
        dtype=[('event', bool), ('duration', float)]
    )
    y_test = np.array(
        [(bool(e), d) for e, d in zip(test_data['event'], test_data['duration'])],
        dtype=[('event', bool), ('duration', float)]
    )

    # Train RSF
    rsf = RandomSurvivalForest(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1
    )

    try:
        rsf.fit(X_train, y_train)
    except Exception as e:
        print(f"RSF fitting failed: {e}")
        return None

    # Predict risk scores
    risk_scores = rsf.predict(X_test)

    # Concordance index
    c_index = concordance_index_censored(
        y_test['event'],
        y_test['duration'],
        risk_scores
    )[0]

    # Feature importance (use permutation importance as fallback)
    try:
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rsf.feature_importances_
        }).sort_values('importance', ascending=False)
    except NotImplementedError:
        # Use a simple approximation: mean absolute prediction change
        from sklearn.inspection import permutation_importance
        try:
            perm_importance = permutation_importance(
                rsf, X_test, y_test,
                n_repeats=10,
                random_state=random_state,
                scoring=lambda est, X, y: concordance_index_censored(y['event'], y['duration'], est.predict(X))[0]
            )
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': perm_importance.importances_mean
            }).sort_values('importance', ascending=False)
        except Exception:
            # Fallback: equal importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': np.ones(len(feature_cols)) / len(feature_cols)
            })

    return {
        'model': rsf,
        'c_index': c_index,
        'feature_importance': feature_importance,
        'risk_scores': risk_scores
    }


def train_logistic_classifier(train_data, test_data, feature_cols, random_state=42):
    """Train Logistic Regression classifier for binary decay prediction."""
    X_train = train_data[feature_cols].values
    y_train = train_data['event'].values.astype(int)
    X_test = test_data[feature_cols].values
    y_test = test_data['event'].values.astype(int)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.5

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    brier = brier_score_loss(y_test, y_prob)

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(model.coef_[0])
    }).sort_values('importance', ascending=False)

    return {
        'model': model,
        'scaler': scaler,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'brier': brier,
        'feature_importance': importance,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def train_random_forest_classifier(train_data, test_data, feature_cols, random_state=42):
    """Train Random Forest classifier."""
    X_train = train_data[feature_cols].values
    y_train = train_data['event'].values.astype(int)
    X_test = test_data[feature_cols].values
    y_test = test_data['event'].values.astype(int)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        random_state=random_state,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.5

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    brier = brier_score_loss(y_test, y_prob)

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'model': model,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'brier': brier,
        'feature_importance': importance,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def cross_validate_models(survival_df, feature_cols, n_folds=5, random_state=42):
    """
    Perform leave-group-out cross-validation across multiple folds.
    Returns averaged performance metrics.
    """
    np.random.seed(random_state)
    all_pairs = survival_df['pair'].unique()
    np.random.shuffle(all_pairs)

    fold_size = len(all_pairs) // n_folds

    cv_results = {
        'cox_ph': {'c_index': []},
        'rsf': {'c_index': []},
        'logistic': {'auc': [], 'f1': []},
        'random_forest': {'auc': [], 'f1': []}
    }

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else len(all_pairs)

        test_pairs = all_pairs[test_start:test_end]
        train_pairs = np.concatenate([all_pairs[:test_start], all_pairs[test_end:]])

        train_data = survival_df[survival_df['pair'].isin(train_pairs)].copy()
        test_data = survival_df[survival_df['pair'].isin(test_pairs)].copy()

        if test_data['event'].sum() == 0 or train_data['event'].sum() == 0:
            continue

        # Cox PH
        if LIFELINES_AVAILABLE:
            try:
                cox_result = train_cox_ph(train_data, test_data, feature_cols)
                if cox_result:
                    cv_results['cox_ph']['c_index'].append(cox_result['c_index'])
            except Exception:
                pass

        # RSF
        if SKSURV_AVAILABLE:
            try:
                rsf_result = train_random_survival_forest(train_data, test_data, feature_cols, random_state=random_state+fold)
                if rsf_result:
                    cv_results['rsf']['c_index'].append(rsf_result['c_index'])
            except Exception:
                pass

        # Logistic
        try:
            lr_result = train_logistic_classifier(train_data, test_data, feature_cols, random_state=random_state+fold)
            cv_results['logistic']['auc'].append(lr_result['auc'])
            cv_results['logistic']['f1'].append(lr_result['f1'])
        except Exception:
            pass

        # Random Forest
        try:
            rf_result = train_random_forest_classifier(train_data, test_data, feature_cols, random_state=random_state+fold)
            cv_results['random_forest']['auc'].append(rf_result['auc'])
            cv_results['random_forest']['f1'].append(rf_result['f1'])
        except Exception:
            pass

    # Compute mean and std
    cv_summary = {}
    for model_name, metrics in cv_results.items():
        cv_summary[model_name] = {}
        for metric_name, values in metrics.items():
            if len(values) > 0:
                cv_summary[model_name][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'n_folds': len(values)
                }

    return cv_summary


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("ML DECAY PREDICTION - SURVIVAL ANALYSIS")
    print("=" * 70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Lifelines available: {LIFELINES_AVAILABLE}")
    print(f"scikit-survival available: {SKSURV_AVAILABLE}")

    # Load data
    print("\n[1] Loading Fama-French 5 Factors + Momentum...")
    df = download_ff_factors()
    df = df[(df.index >= '1990-01-01') & (df.index <= '2024-12-31')]
    df = df.drop('RF', axis=1)
    print(f"Data range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Observations: {len(df)}")

    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']

    # Fit HMM for regime detection
    print("\n[2] Detecting market regimes...")
    X_hmm = df[['Mkt-RF', 'SMB', 'HML']].values
    hmm = StudentTHMM(n_regimes=3)
    regimes = hmm.fit_predict(X_hmm)

    regime_counts = pd.Series(regimes).value_counts().sort_index()
    print(f"Regime distribution: Normal={regime_counts.get(0,0)}, "
          f"Elevated={regime_counts.get(1,0)}, Crisis={regime_counts.get(2,0)}")

    # Build survival dataset
    print("\n[3] Building survival dataset...")
    survival_df = build_survival_dataset(df, factors, regimes)
    print(f"Total factor pairs: {len(survival_df)}")
    print(f"Pairs with decay event: {survival_df['event'].sum()}")
    print(f"Event rate: {survival_df['event'].mean():.1%}")

    # Feature columns for survival
    survival_features = [
        'initial_F', 'initial_F_vol', 'initial_F_mean',
        'vol_ratio_source', 'vol_ratio_target', 'corr_stability',
        'initial_corr', 'sig_rate', 'normal_pct', 'crisis_pct',
        'factor_vol_ratio', 'initial_trend'
    ]

    # Train/test split (leave-group-out)
    np.random.seed(42)
    all_pairs = survival_df['pair'].values.copy()
    np.random.shuffle(all_pairs)

    n_test = 10
    test_pairs = all_pairs[:n_test]
    train_pairs = all_pairs[n_test:]

    train_survival = survival_df[survival_df['pair'].isin(train_pairs)].copy()
    test_survival = survival_df[survival_df['pair'].isin(test_pairs)].copy()

    print(f"\nTrain pairs: {len(train_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")
    print(f"Train decay rate: {train_survival['event'].mean():.1%}")
    print(f"Test decay rate: {test_survival['event'].mean():.1%}")

    results = {}

    # ==========================================================================
    # Cox Proportional Hazards
    # ==========================================================================
    print("\n" + "=" * 70)
    print("COX PROPORTIONAL HAZARDS")
    print("=" * 70)

    if LIFELINES_AVAILABLE:
        cox_result = train_cox_ph(train_survival, test_survival, survival_features)
        if cox_result:
            print(f"\nC-index (OOS): {cox_result['c_index']:.3f}")
            print("\nHazard Ratios (top features):")
            print("-" * 50)
            hr = cox_result['hazard_ratios'].sort_values('hazard_ratio', ascending=False)
            for idx in hr.head(5).index:
                row = hr.loc[idx]
                sig = '*' if row['p_value'] < 0.05 else ''
                print(f"  {idx:<25} HR={row['hazard_ratio']:.3f}  p={row['p_value']:.3f}{sig}")

            results['cox_ph'] = {
                'c_index': float(cox_result['c_index']),
                'hazard_ratios': cox_result['hazard_ratios'].to_dict()
            }
    else:
        print("lifelines not available - skipping Cox PH")

    # ==========================================================================
    # Random Survival Forest
    # ==========================================================================
    print("\n" + "=" * 70)
    print("RANDOM SURVIVAL FOREST")
    print("=" * 70)

    if SKSURV_AVAILABLE:
        rsf_result = train_random_survival_forest(train_survival, test_survival, survival_features)
        if rsf_result:
            print(f"\nC-index (OOS): {rsf_result['c_index']:.3f}")
            print("\nFeature Importance (top 5):")
            print("-" * 50)
            for _, row in rsf_result['feature_importance'].head(5).iterrows():
                print(f"  {row['feature']:<25} {row['importance']:.4f}")

            results['rsf'] = {
                'c_index': float(rsf_result['c_index']),
                'feature_importance': rsf_result['feature_importance'].to_dict('records')
            }
    else:
        print("scikit-survival not available - skipping RSF")

    # ==========================================================================
    # Logistic Regression Baseline
    # ==========================================================================
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION (Binary: Will Decay?)")
    print("=" * 70)

    lr_result = train_logistic_classifier(train_survival, test_survival, survival_features)
    print(f"\nAUC (OOS): {lr_result['auc']:.3f}")
    print(f"Precision: {lr_result['precision']:.3f}")
    print(f"Recall: {lr_result['recall']:.3f}")
    print(f"F1 Score: {lr_result['f1']:.3f}")
    print(f"Brier Score: {lr_result['brier']:.3f}")
    print("\nFeature Importance (top 5):")
    print("-" * 50)
    for _, row in lr_result['feature_importance'].head(5).iterrows():
        print(f"  {row['feature']:<25} {row['importance']:.4f}")

    results['logistic'] = {
        'auc': float(lr_result['auc']),
        'precision': float(lr_result['precision']),
        'recall': float(lr_result['recall']),
        'f1': float(lr_result['f1']),
        'brier': float(lr_result['brier']),
        'feature_importance': lr_result['feature_importance'].to_dict('records')
    }

    # ==========================================================================
    # Random Forest Classifier
    # ==========================================================================
    print("\n" + "=" * 70)
    print("RANDOM FOREST CLASSIFIER")
    print("=" * 70)

    rf_result = train_random_forest_classifier(train_survival, test_survival, survival_features)
    print(f"\nAUC (OOS): {rf_result['auc']:.3f}")
    print(f"Precision: {rf_result['precision']:.3f}")
    print(f"Recall: {rf_result['recall']:.3f}")
    print(f"F1 Score: {rf_result['f1']:.3f}")
    print(f"Brier Score: {rf_result['brier']:.3f}")
    print("\nFeature Importance (top 5):")
    print("-" * 50)
    for _, row in rf_result['feature_importance'].head(5).iterrows():
        print(f"  {row['feature']:<25} {row['importance']:.4f}")

    results['random_forest'] = {
        'auc': float(rf_result['auc']),
        'precision': float(rf_result['precision']),
        'recall': float(rf_result['recall']),
        'f1': float(rf_result['f1']),
        'brier': float(rf_result['brier']),
        'feature_importance': rf_result['feature_importance'].to_dict('records')
    }

    # ==========================================================================
    # Cross-Validation Analysis (5-fold)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    cv_results = cross_validate_models(survival_df, survival_features, n_folds=5)

    print("\nCross-validated Performance (Mean +/- Std):")
    print("-" * 60)
    for model_name, metrics in cv_results.items():
        if metrics:
            metrics_str = ', '.join([f"{k}={v['mean']:.3f}+/-{v['std']:.3f}"
                                      for k, v in metrics.items() if v])
            if metrics_str:
                print(f"  {model_name:<25} {metrics_str}")

    # ==========================================================================
    # Per-Pair Predictions
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PER-PAIR PREDICTIONS (Held-out Test Set)")
    print("=" * 70)

    print(f"\n{'Pair':<25} {'Actual':>8} {'LogisticP':>10} {'RF_P':>8} {'Status':>10}")
    print("-" * 70)

    pair_results = []
    for i, pair in enumerate(test_pairs):
        mask = test_survival['pair'] == pair
        if mask.sum() == 0:
            continue

        actual = int(test_survival.loc[mask, 'event'].values[0])
        lr_prob = lr_result['y_prob'][i]
        rf_prob = rf_result['y_prob'][i]

        # Ensemble prediction
        avg_prob = (lr_prob + rf_prob) / 2
        predicted = int(avg_prob > 0.5)
        correct = predicted == actual
        status = "CORRECT" if correct else "WRONG"

        pair_results.append({
            'pair': pair,
            'actual': actual,
            'lr_prob': float(lr_prob),
            'rf_prob': float(rf_prob),
            'ensemble_prob': float(avg_prob),
            'predicted': predicted,
            'correct': correct
        })

        print(f"{pair:<25} {actual:>8} {lr_prob:>10.2%} {rf_prob:>8.2%} {status:>10}")

    pair_accuracy = sum(1 for p in pair_results if p['correct']) / len(pair_results)
    print(f"\nPair-level accuracy: {pair_accuracy:.1%} ({sum(1 for p in pair_results if p['correct'])}/{len(pair_results)})")

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\n1. Model Performance (OOS):")
    print("-" * 50)
    print(f"   {'Model':<25} {'C-index/AUC':>15}")
    print("-" * 50)

    if 'cox_ph' in results:
        print(f"   {'Cox PH':<25} {results['cox_ph']['c_index']:>15.3f} (C-index)")
    if 'rsf' in results:
        print(f"   {'Random Survival Forest':<25} {results['rsf']['c_index']:>15.3f} (C-index)")
    print(f"   {'Logistic Regression':<25} {results['logistic']['auc']:>15.3f} (AUC)")
    print(f"   {'Random Forest':<25} {results['random_forest']['auc']:>15.3f} (AUC)")

    # Best model
    best_c_index = max(
        results.get('cox_ph', {}).get('c_index', 0),
        results.get('rsf', {}).get('c_index', 0)
    )
    best_auc = max(
        results['logistic']['auc'],
        results['random_forest']['auc']
    )

    print(f"\n2. Best Survival Model C-index: {best_c_index:.3f}")
    print(f"   Best Classification AUC: {best_auc:.3f}")
    print(f"   Pair-level Accuracy: {pair_accuracy:.1%}")

    print("\n3. Top Predictive Features:")
    print("-" * 50)
    # Aggregate feature importance across models
    all_importances = {}
    for model_name in ['logistic', 'random_forest']:
        for feat in results[model_name]['feature_importance']:
            name = feat['feature']
            if name not in all_importances:
                all_importances[name] = []
            all_importances[name].append(feat['importance'])

    avg_importance = {k: np.mean(v) for k, v in all_importances.items()}
    sorted_features = sorted(avg_importance.items(), key=lambda x: -x[1])

    for i, (feat, imp) in enumerate(sorted_features[:5], 1):
        print(f"   {i}. {feat}: {imp:.4f}")

    print("\n4. Interpretation:")
    print("-" * 50)
    top_feat = sorted_features[0][0]
    print(f"   - Most predictive feature: {top_feat}")
    print(f"   - Decay is predictable with {best_auc:.1%} AUC")
    print(f"   - {sum(1 for p in pair_results if p['correct'])}/{len(pair_results)} pairs correctly classified")

    # CV-averaged metrics (more reliable)
    cv_auc_mean = cv_results.get('logistic', {}).get('auc', {}).get('mean', 0)
    cv_auc_std = cv_results.get('logistic', {}).get('auc', {}).get('std', 0)
    cv_c_index_mean = cv_results.get('cox_ph', {}).get('c_index', {}).get('mean', 0)
    cv_c_index_std = cv_results.get('cox_ph', {}).get('c_index', {}).get('std', 0)

    print(f"\n5. Cross-Validated Performance (more reliable):")
    print("-" * 50)
    if cv_c_index_mean > 0:
        print(f"   - Cox PH C-index: {cv_c_index_mean:.3f} +/- {cv_c_index_std:.3f}")
    if cv_auc_mean > 0:
        print(f"   - Logistic AUC: {cv_auc_mean:.3f} +/- {cv_auc_std:.3f}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'data_range': f"{df.index[0].date()} to {df.index[-1].date()}",
        'n_pairs': len(survival_df),
        'n_train_pairs': len(train_pairs),
        'n_test_pairs': len(test_pairs),
        'train_pairs': list(train_pairs),
        'test_pairs': list(test_pairs),
        'decay_rate': float(survival_df['event'].mean()),
        'model_results': results,
        'cv_results': {
            k: {m: {'mean': v['mean'], 'std': v['std']}
                for m, v in metrics.items() if v}
            for k, metrics in cv_results.items() if metrics
        },
        'pair_predictions': pair_results,
        'pair_accuracy': float(pair_accuracy),
        'feature_columns': survival_features
    }

    output_path = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results/ml_decay_prediction.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    output = main()
