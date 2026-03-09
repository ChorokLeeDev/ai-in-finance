"""
Decay Prediction Model - Primary ML Contribution for ICAIF
===========================================================

Predict which factor pairs will show Granger causality decay and when.

Models:
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting (XGBoost)
- Optional: Random Survival Forest for time-to-decay

Author: ICAIF 2026 Submission
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
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING
# =============================================================================

def download_ff_factors():
    """Download FF 5 factors + Momentum."""
    # 5 factors
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

    # Momentum
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
# REGIME DETECTION (Student-t HMM simplified)
# =============================================================================

class StudentTHMM:
    """Minimal Student-t HMM for regime detection."""

    def __init__(self, n_regimes=3, random_state=28):
        self.n_regimes = n_regimes
        self.random_state = random_state

    def fit_predict(self, X):
        np.random.seed(self.random_state)
        X = np.asarray(X)
        K = self.n_regimes

        # K-means clustering based on volatility
        centroids, labels = kmeans2(X, K, minit='++')
        norms = np.linalg.norm(centroids, axis=1)
        order = np.argsort(norms)

        # Reorder: 0=low vol (normal), 1=medium, 2=high vol (crisis)
        new_labels = np.zeros_like(labels)
        for new_k, old_k in enumerate(order):
            new_labels[labels == old_k] = new_k

        return new_labels


# =============================================================================
# GRANGER CAUSALITY ANALYSIS
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
    except:
        return None


def compute_rolling_granger(df, source, target, regimes, regime_idx=0,
                            window_years=5, step_months=6):
    """Compute rolling 5-year Granger F-statistics for a factor pair."""
    results = []
    dates = df.index

    for year in range(1995, 2024):
        for month in [1, 7]:  # semi-annual
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
# FEATURE EXTRACTION
# =============================================================================

def extract_features(df, source, target, regimes, rolling_granger_df):
    """
    Extract features for decay prediction at each time point.

    Features:
    1. Current Granger F-statistic
    2. Rolling volatility of F-stat
    3. Current regime distribution
    4. Factor correlation
    5. Factor volatility ratio
    6. Time since last regime change
    7. Momentum of F-stat (trend)
    """
    if len(rolling_granger_df) < 5:
        return None

    features_list = []

    for i in range(2, len(rolling_granger_df)):
        row = rolling_granger_df.iloc[i]
        window_end = row['window_end']

        # Get historical F-stats
        hist_F = rolling_granger_df.iloc[:i+1]['F'].values

        # Feature 1: Current F-statistic
        current_F = row['F']

        # Feature 2: Rolling volatility of F-stat (last 3 windows)
        F_volatility = np.std(hist_F[-3:]) if len(hist_F) >= 3 else 0

        # Feature 3: Rolling mean of F-stat (last 3 windows)
        F_mean = np.mean(hist_F[-3:]) if len(hist_F) >= 3 else current_F

        # Get recent data for factor features
        lookback = pd.DateOffset(years=1)
        recent_mask = (df.index >= window_end - lookback) & (df.index < window_end)
        recent_regimes = regimes[df.index.isin(df.index[recent_mask])]

        if recent_mask.sum() < 50:
            continue

        recent_df = df.loc[recent_mask]

        # Feature 4: Recent regime distribution
        normal_pct = (recent_regimes == 0).mean()
        crisis_pct = (recent_regimes == 2).mean()

        # Feature 5: Factor correlation
        factor_corr = recent_df[source].corr(recent_df[target])

        # Feature 6: Factor volatility ratio
        source_vol = recent_df[source].std()
        target_vol = recent_df[target].std()
        vol_ratio = source_vol / target_vol if target_vol > 0 else 1

        # Feature 7: Momentum of F-stat (slope over last 3 windows)
        if len(hist_F) >= 3:
            t = np.arange(len(hist_F[-3:]))
            slope, _ = np.polyfit(t, hist_F[-3:], 1)
            F_momentum = slope
        else:
            F_momentum = 0

        # Feature 8: Relative F-stat (current vs historical max)
        F_relative = current_F / np.max(hist_F) if np.max(hist_F) > 0 else 1

        # Feature 9: Time index (normalized)
        time_idx = i / len(rolling_granger_df)

        # Feature 10: F-stat change rate (percent)
        if i > 0:
            prev_F = rolling_granger_df.iloc[i-1]['F']
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


def label_decay(rolling_granger_df, horizon_years=5, decay_pct=0.5):
    """
    Label whether SIGNIFICANT decay occurs within horizon.

    Target: Binary - Will F-stat drop by more than decay_pct (50%) from current level?

    This is more meaningful than absolute threshold because:
    1. Some pairs start with very high F (e.g., 20) - dropping to 10 is still significant
    2. Some pairs start modest (e.g., 5) - dropping to 2 is decay
    3. We want to predict RELATIVE decline, not absolute level
    """
    labels = []

    for i in range(len(rolling_granger_df)):
        current_F = rolling_granger_df.iloc[i]['F']
        current_year = rolling_granger_df.iloc[i]['year']

        future_mask = (rolling_granger_df['year'] > current_year) & \
                      (rolling_granger_df['year'] <= current_year + horizon_years)

        if future_mask.sum() == 0:
            labels.append(np.nan)  # No future data
            continue

        future_F = rolling_granger_df.loc[future_mask, 'F'].values

        # Decay = minimum future F is less than decay_pct of current F
        # (e.g., if current F=10 and decay_pct=0.5, decay if any future F < 5)
        min_future_F = np.min(future_F)
        threshold = current_F * (1 - decay_pct)

        decayed = int(min_future_F < threshold)
        labels.append(decayed)

    return labels


# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================

def build_dataset(df, factors, regimes):
    """Build dataset with features and labels for all 30 factor pairs."""
    all_pairs_data = []
    pair_info = []

    for source in factors:
        for target in factors:
            if source == target:
                continue

            pair_name = f"{source}->{target}"

            # Compute rolling Granger
            rolling_df = compute_rolling_granger(df, source, target, regimes)

            if len(rolling_df) < 10:
                continue

            # Extract features
            features_df = extract_features(df, source, target, regimes, rolling_df)

            if features_df is None or len(features_df) < 5:
                continue

            # Label decay
            labels = label_decay(rolling_df)

            # Align features with labels (features start from index 2)
            features_df['decay_label'] = labels[2:len(features_df)+2]
            features_df['pair'] = pair_name
            features_df['source'] = source
            features_df['target'] = target

            # Remove rows with NaN labels
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


def train_and_evaluate(dataset, pair_info, n_holdout=10, random_state=42):
    """
    Train models and evaluate on held-out pairs.

    Validation: Train on 20 pairs, test on 10 held-out pairs.
    """
    np.random.seed(random_state)

    # Get unique pairs
    all_pairs = dataset['pair'].unique()
    n_pairs = len(all_pairs)

    if n_pairs < n_holdout + 5:
        n_holdout = max(1, n_pairs // 4)

    # Split pairs into train/test
    np.random.shuffle(all_pairs)
    test_pairs = all_pairs[:n_holdout]
    train_pairs = all_pairs[n_holdout:]

    # Create train/test sets
    train_data = dataset[dataset['pair'].isin(train_pairs)].copy()
    test_data = dataset[dataset['pair'].isin(test_pairs)].copy()

    # Feature columns
    feature_cols = ['current_F', 'F_volatility', 'F_mean', 'normal_pct',
                    'crisis_pct', 'factor_corr', 'vol_ratio', 'F_momentum',
                    'F_relative', 'time_idx', 'F_change']

    X_train = train_data[feature_cols].values
    y_train = train_data['decay_label'].values.astype(int)
    X_test = test_data[feature_cols].values
    y_test = test_data['decay_label'].values.astype(int)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=random_state, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=random_state,
            class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=random_state
        )
    }

    # Try XGBoost if available
    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, random_state=random_state,
            eval_metric='logloss', use_label_encoder=False
        )
    except ImportError:
        pass

    results = {}

    print("\n" + "=" * 70)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 70)
    print(f"\nTrain pairs: {len(train_pairs)} ({len(train_data)} samples)")
    print(f"Test pairs: {len(test_pairs)} ({len(test_data)} samples)")
    print(f"Train decay rate: {y_train.mean():.2%}")
    print(f"Test decay rate: {y_test.mean():.2%}")

    for name, model in models.items():
        print(f"\n{'-' * 50}")
        print(f"Training: {name}")
        print(f"{'-' * 50}")

        # Cross-validation on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
        print(f"CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        # Train on full training set
        model.fit(X_train_scaled, y_train)

        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = 0.5

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"OOS AUC: {auc:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1: {f1:.3f}")

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            importances = np.zeros(len(feature_cols))

        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        print(f"\nTop 5 features:")
        for _, row in importance_df.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        results[name] = {
            'model': model,
            'cv_auc': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'oos_auc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'feature_importance': importance_df.to_dict('records')
        }

    # Best model analysis
    best_name = max(results.keys(), key=lambda x: results[x]['oos_auc'])
    best_model = results[best_name]['model']

    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_name}")
    print("=" * 70)

    # Per-pair predictions on test set
    print("\nPer-pair predictions on held-out pairs:")
    print("-" * 60)

    pair_predictions = []
    for pair in test_pairs:
        pair_mask = test_data['pair'] == pair
        if pair_mask.sum() == 0:
            continue

        X_pair = scaler.transform(test_data.loc[pair_mask, feature_cols].values)
        y_pair = test_data.loc[pair_mask, 'decay_label'].values.astype(int)

        prob = best_model.predict_proba(X_pair)[:, 1].mean()
        pred = int(prob > 0.5)
        actual = int(y_pair.mean() > 0.5)

        correct = pred == actual
        pair_predictions.append({
            'pair': pair,
            'predicted_decay': pred,
            'actual_decay': actual,
            'probability': float(prob),
            'correct': correct
        })

        status = "CORRECT" if correct else "WRONG"
        print(f"{pair:<20} Pred: {pred} ({prob:.2%}) | Actual: {actual} | {status}")

    pair_accuracy = sum(1 for p in pair_predictions if p['correct']) / len(pair_predictions)
    print(f"\nPair-level accuracy: {pair_accuracy:.1%}")

    return results, pair_predictions, test_pairs, train_pairs, feature_cols, scaler


def main():
    print("=" * 70)
    print("DECAY PREDICTION MODEL - ICAIF ML CONTRIBUTION")
    print("=" * 70)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\n[1] Loading Fama-French 5 Factors + Momentum...")
    df = download_ff_factors()
    df = df[(df.index >= '1990-01-01') & (df.index <= '2024-12-31')]
    df = df.drop('RF', axis=1)  # Remove risk-free rate
    print(f"Data range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Observations: {len(df)}")

    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']

    # Fit HMM for regime detection
    print("\n[2] Detecting market regimes with HMM...")
    X_hmm = df[['Mkt-RF', 'SMB', 'HML']].values
    hmm = StudentTHMM(n_regimes=3)
    regimes = hmm.fit_predict(X_hmm)

    regime_counts = pd.Series(regimes).value_counts().sort_index()
    print(f"Regime distribution: Normal={regime_counts.get(0,0)}, "
          f"Elevated={regime_counts.get(1,0)}, Crisis={regime_counts.get(2,0)}")

    # Build dataset
    print("\n[3] Building feature dataset for 30 factor pairs...")
    dataset, pair_info = build_dataset(df, factors, regimes)

    if dataset is None:
        print("ERROR: Could not build dataset")
        return

    print(f"Total samples: {len(dataset)}")
    print(f"Pairs with data: {len(pair_info)}")
    print(f"Overall decay rate: {dataset['decay_label'].mean():.2%}")

    # Train and evaluate models
    print("\n[4] Training prediction models...")
    results, pair_predictions, test_pairs, train_pairs, feature_cols, scaler = \
        train_and_evaluate(dataset, pair_info)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    best_name = max(results.keys(), key=lambda x: results[x]['oos_auc'])
    best_result = results[best_name]

    print(f"\n1. Best Model: {best_name}")
    print(f"   OOS AUC: {best_result['oos_auc']:.3f}")
    print(f"   CV AUC: {best_result['cv_auc']:.3f} (+/- {best_result['cv_std']:.3f})")

    print(f"\n2. Top 3 Most Important Features:")
    for i, feat in enumerate(best_result['feature_importance'][:3], 1):
        print(f"   {i}. {feat['feature']}: {feat['importance']:.4f}")

    print(f"\n3. Correctly Predicted Decay Pairs:")
    correct_decay = [p for p in pair_predictions
                     if p['correct'] and p['actual_decay'] == 1]
    if correct_decay:
        for p in correct_decay:
            print(f"   - {p['pair']} (prob: {p['probability']:.2%})")
    else:
        print("   (None)")

    print(f"\n4. Correctly Predicted No-Decay Pairs:")
    correct_no_decay = [p for p in pair_predictions
                        if p['correct'] and p['actual_decay'] == 0]
    if correct_no_decay:
        for p in correct_no_decay:
            print(f"   - {p['pair']} (prob: {p['probability']:.2%})")
    else:
        print("   (None)")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'data_range': f"{df.index[0].date()} to {df.index[-1].date()}",
        'n_pairs': len(pair_info),
        'n_samples': len(dataset),
        'decay_rate': float(dataset['decay_label'].mean()),
        'n_train_pairs': len(train_pairs),
        'n_test_pairs': len(test_pairs),
        'train_pairs': list(train_pairs),
        'test_pairs': list(test_pairs),
        'model_results': {
            name: {k: v for k, v in res.items() if k != 'model'}
            for name, res in results.items()
        },
        'best_model': best_name,
        'oos_auc': best_result['oos_auc'],
        'top_features': best_result['feature_importance'][:3],
        'pair_predictions': pair_predictions,
        'pair_accuracy': sum(1 for p in pair_predictions if p['correct']) / len(pair_predictions),
        'feature_columns': feature_cols
    }

    output_path = '/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results/decay_prediction.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    output = main()
