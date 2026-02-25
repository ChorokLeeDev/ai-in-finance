"""
Practitioner baselines for HML->SMB lead-lag detection.

Three approaches:
1. Rolling lagged cross-correlation (60-day)
2. Rolling Granger causality F-test (250-day)
3. Diebold-Yilmaz-style FEVD spillover (250-day VAR(1))

All split by regime from selected_fit_regimes.csv.
"""

import json
import io
import re
import warnings
import zipfile
from collections import Counter
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("/Users/i767700/Github/ai-in-finance/papers/causal_regimes/results")
OUTPUT_PATH = RESULTS_DIR / "practitioner_baseline.json"
REGIME_PATH = RESULTS_DIR / "selected_fit_regimes.csv"


def load_ff6_daily():
    """Load Fama-French 6 factors (5 + Momentum) daily, 1990-2024."""

    def download_french_daily(dataset):
        url = (
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            f"{dataset}_CSV.zip"
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csv_name = next(name for name in zf.namelist() if name.lower().endswith(".csv"))
            raw_text = zf.read(csv_name).decode("utf-8", errors="replace")
        lines = raw_text.splitlines()
        header_idx = next((i for i, line in enumerate(lines) if line.strip().startswith(",")), None)
        if header_idx is None:
            raise RuntimeError(f"Could not find header row in {dataset}")
        header = lines[header_idx].strip()
        date_rows = []
        for line in lines[header_idx + 1 :]:
            stripped = line.strip()
            if re.match(r"^\d{8},", stripped):
                date_rows.append(stripped)
        if not date_rows:
            raise RuntimeError(f"Could not find daily date rows in {dataset}")
        data = pd.read_csv(io.StringIO(header + "\n" + "\n".join(date_rows)))
        data.columns = [str(c).strip() for c in data.columns]
        data = data.rename(columns={data.columns[0]: "Date"})
        data["Date"] = data["Date"].astype(str).str.strip()
        data = data[data["Date"].str.fullmatch(r"\d{8}")]
        data["Date"] = pd.to_datetime(data["Date"], format="%Y%m%d")
        for col in data.columns:
            if col != "Date":
                data[col] = pd.to_numeric(data[col], errors="coerce")
        return data.set_index("Date").dropna(how="all")

    print("Downloading FF 5-factor daily data...")
    df5 = download_french_daily("F-F_Research_Data_5_Factors_2x3_daily")

    print("Downloading Momentum factor daily data...")
    dfm = download_french_daily("F-F_Momentum_Factor_daily")
    mom_col = next(
        (c for c in dfm.columns if "mom" in c.lower() or "wml" in c.lower()),
        None,
    )
    if mom_col is None:
        raise RuntimeError(f"Could not identify momentum column in {list(dfm.columns)}")
    dfm = dfm[[mom_col]].rename(columns={mom_col: "MOM"})

    # Merge on date index
    df = df5.join(dfm, how="inner")

    # Rename columns - strip whitespace first
    df.columns = [c.strip() for c in df.columns]
    rename_map = {"Mkt-RF": "MKT", "Mom": "MOM"}
    df = df.rename(columns=rename_map)

    # Keep only the 6 factors
    keep_cols = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Filter to 1990-2024
    df = df[(df.index >= "1990-01-01") & (df.index <= "2024-12-31")]

    print(f"Loaded {len(df)} daily observations, columns: {list(df.columns)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    return df


def load_regimes():
    """Load regime labels."""
    regimes = pd.read_csv(REGIME_PATH, parse_dates=["date"])
    regimes = regimes.set_index("date")
    print(f"Loaded {len(regimes)} regime labels: {regimes['regime_label'].value_counts().to_dict()}")
    return regimes


def assign_window_regime(dates, regimes_series):
    """Assign a regime to a window based on majority of days."""
    labels = regimes_series.reindex(dates).dropna()
    if len(labels) == 0:
        return None
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def rolling_correlation_baseline(df, regimes, window=60):
    """Baseline 1: Rolling lagged cross-correlation."""
    print("\n=== Baseline 1: Rolling Lagged Cross-Correlation (window=60) ===")

    hml = df["HML"]
    smb = df["SMB"]

    # HML(t-1) vs SMB(t): lag HML by 1
    hml_lag1 = hml.shift(1)
    smb_lag1 = smb.shift(1)

    # Rolling correlation: HML(t-1) vs SMB(t)
    corr_hml_to_smb = hml_lag1.rolling(window).corr(smb)
    # Rolling correlation: SMB(t-1) vs HML(t)
    corr_smb_to_hml = smb_lag1.rolling(window).corr(hml)

    # Drop NaNs
    valid = corr_hml_to_smb.dropna().index.intersection(corr_smb_to_hml.dropna().index)
    c1 = corr_hml_to_smb.loc[valid]
    c2 = corr_smb_to_hml.loc[valid]

    overall_mean_hml = float(c1.mean())
    overall_mean_smb = float(c2.mean())
    pct_hml_leads = float((c1.abs() > c2.abs()).mean() * 100)

    print(f"  Overall mean HML(t-1)->SMB(t) corr: {overall_mean_hml:.4f}")
    print(f"  Overall mean SMB(t-1)->HML(t) corr: {overall_mean_smb:.4f}")
    print(f"  % windows where |HML->SMB| > |SMB->HML|: {pct_hml_leads:.1f}%")

    result = {
        "overall": {
            "hml_to_smb_mean": round(overall_mean_hml, 6),
            "smb_to_hml_mean": round(overall_mean_smb, 6),
            "pct_hml_leads": round(pct_hml_leads, 2),
            "n_windows": len(valid),
        },
        "by_regime": {},
    }

    # By regime
    regime_labels = regimes["regime_label"]
    date_regimes = regime_labels.reindex(valid)
    for regime in ["Normal", "Elevated", "Crisis"]:
        mask = date_regimes == regime
        if mask.sum() == 0:
            continue
        c1r = c1[mask]
        c2r = c2[mask]
        mean_hml = float(c1r.mean())
        mean_smb = float(c2r.mean())
        pct = float((c1r.abs() > c2r.abs()).mean() * 100)
        result["by_regime"][regime] = {
            "hml_to_smb_mean": round(mean_hml, 6),
            "smb_to_hml_mean": round(mean_smb, 6),
            "pct_hml_leads": round(pct, 2),
            "n_windows": int(mask.sum()),
        }
        print(f"  {regime}: HML->SMB={mean_hml:.4f}, SMB->HML={mean_smb:.4f}, %HML leads={pct:.1f}%")

    return result


def rolling_granger_baseline(df, regimes, window=250, max_lag=1):
    """Baseline 2: Rolling Granger causality F-test."""
    print("\n=== Baseline 2: Rolling Granger Causality (window=250, lag=1) ===")

    dates = df.index
    regime_labels = regimes["regime_label"]

    hml_pvals = []
    smb_pvals = []
    window_dates = []
    window_regimes = []

    n = len(df)
    total_windows = n - window + 1
    report_every = max(1, total_windows // 10)

    for i in range(total_windows):
        if i % report_every == 0:
            print(f"  Granger window {i}/{total_windows}...")

        start, end = i, i + window
        sub = df.iloc[start:end][["HML", "SMB"]].copy()

        # Test HML -> SMB: does lagged HML help predict SMB?
        try:
            res_hml = grangercausalitytests(sub[["SMB", "HML"]], maxlag=max_lag, verbose=False)
            p_hml = res_hml[1][0]["ssr_ftest"][1]
        except Exception:
            p_hml = np.nan

        # Test SMB -> HML: does lagged SMB help predict HML?
        try:
            res_smb = grangercausalitytests(sub[["HML", "SMB"]], maxlag=max_lag, verbose=False)
            p_smb = res_smb[1][0]["ssr_ftest"][1]
        except Exception:
            p_smb = np.nan

        hml_pvals.append(p_hml)
        smb_pvals.append(p_smb)

        win_date = dates[end - 1]
        window_dates.append(win_date)

        win_dates_range = dates[start:end]
        reg = assign_window_regime(win_dates_range, regime_labels)
        window_regimes.append(reg)

    hml_pvals = np.array(hml_pvals)
    smb_pvals = np.array(smb_pvals)
    window_regimes = np.array(window_regimes)

    valid_hml = ~np.isnan(hml_pvals)
    valid_smb = ~np.isnan(smb_pvals)

    pct_hml = float(np.mean(hml_pvals[valid_hml] < 0.05) * 100)
    pct_smb = float(np.mean(smb_pvals[valid_smb] < 0.05) * 100)

    print(f"  Overall: HML->SMB significant in {pct_hml:.1f}% of windows")
    print(f"  Overall: SMB->HML significant in {pct_smb:.1f}% of windows")

    result = {
        "overall": {
            "pct_hml_sig": round(pct_hml, 2),
            "pct_smb_sig": round(pct_smb, 2),
            "median_p_hml": round(float(np.nanmedian(hml_pvals)), 4),
            "median_p_smb": round(float(np.nanmedian(smb_pvals)), 4),
            "n_windows": int(valid_hml.sum()),
        },
        "by_regime": {},
    }

    for regime in ["Normal", "Elevated", "Crisis"]:
        mask = window_regimes == regime
        if mask.sum() == 0:
            continue
        vm_hml = mask & valid_hml
        vm_smb = mask & valid_smb
        p_h = float(np.mean(hml_pvals[vm_hml] < 0.05) * 100) if vm_hml.sum() > 0 else 0
        p_s = float(np.mean(smb_pvals[vm_smb] < 0.05) * 100) if vm_smb.sum() > 0 else 0
        result["by_regime"][regime] = {
            "pct_hml_sig": round(p_h, 2),
            "pct_smb_sig": round(p_s, 2),
            "median_p_hml": round(float(np.nanmedian(hml_pvals[vm_hml])), 4) if vm_hml.sum() > 0 else None,
            "median_p_smb": round(float(np.nanmedian(smb_pvals[vm_smb])), 4) if vm_smb.sum() > 0 else None,
            "n_windows": int(mask.sum()),
        }
        print(f"  {regime}: HML->SMB sig={p_h:.1f}%, SMB->HML sig={p_s:.1f}%")

    return result


def spillover_baseline(df, regimes, window=250, horizon=5):
    """Baseline 3: Diebold-Yilmaz-style FEVD spillover."""
    print("\n=== Baseline 3: FEVD Spillover (window=250, VAR(1), horizon=5) ===")

    dates = df.index
    regime_labels = regimes["regime_label"]

    hml_to_smb_spill = []
    smb_to_hml_spill = []
    window_dates = []
    window_regimes = []
    skipped = 0

    n = len(df)
    total_windows = n - window + 1
    report_every = max(1, total_windows // 10)

    for i in range(total_windows):
        if i % report_every == 0:
            print(f"  Spillover window {i}/{total_windows}...")

        start, end = i, i + window
        sub = df.iloc[start:end][["HML", "SMB"]].copy()

        try:
            model = VAR(sub)
            fitted = model.fit(maxlags=1, ic=None, trend="c", verbose=False)
            fevd = fitted.fevd(horizon)

            # fevd.decomp shape: (horizon, n_vars, n_vars)
            # Variables order: [HML, SMB] (same as input)
            # fevd.decomp[h, i, j] = fraction of var i's FEV at horizon h due to var j
            decomp = fevd.decomp

            # HML->SMB: fraction of SMB's FEV explained by HML shocks
            # SMB is index 1, HML is index 0
            hml_to_smb = float(decomp[-1, 1, 0])
            smb_to_hml = float(decomp[-1, 0, 1])

            hml_to_smb_spill.append(hml_to_smb)
            smb_to_hml_spill.append(smb_to_hml)

            win_date = dates[end - 1]
            window_dates.append(win_date)

            win_dates_range = dates[start:end]
            reg = assign_window_regime(win_dates_range, regime_labels)
            window_regimes.append(reg)

        except Exception:
            skipped += 1
            continue

    print(f"  Completed {len(hml_to_smb_spill)} windows, skipped {skipped}")

    hml_arr = np.array(hml_to_smb_spill)
    smb_arr = np.array(smb_to_hml_spill)
    window_regimes = np.array(window_regimes)

    mean_hml = float(hml_arr.mean())
    mean_smb = float(smb_arr.mean())

    print(f"  Overall: HML->SMB spillover = {mean_hml:.4f}")
    print(f"  Overall: SMB->HML spillover = {mean_smb:.4f}")

    result = {
        "overall": {
            "hml_to_smb_mean": round(mean_hml, 6),
            "smb_to_hml_mean": round(mean_smb, 6),
            "hml_to_smb_median": round(float(np.median(hml_arr)), 6),
            "smb_to_hml_median": round(float(np.median(smb_arr)), 6),
            "n_windows": len(hml_arr),
            "n_skipped": skipped,
        },
        "by_regime": {},
    }

    for regime in ["Normal", "Elevated", "Crisis"]:
        mask = window_regimes == regime
        if mask.sum() == 0:
            continue
        mh = float(hml_arr[mask].mean())
        ms = float(smb_arr[mask].mean())
        result["by_regime"][regime] = {
            "hml_to_smb_mean": round(mh, 6),
            "smb_to_hml_mean": round(ms, 6),
            "hml_to_smb_median": round(float(np.median(hml_arr[mask])), 6),
            "smb_to_hml_median": round(float(np.median(smb_arr[mask])), 6),
            "n_windows": int(mask.sum()),
        }
        print(f"  {regime}: HML->SMB={mh:.4f}, SMB->HML={ms:.4f}")

    return result


def main():
    print("=" * 60)
    print("Practitioner Baselines for HML->SMB Lead-Lag Detection")
    print("=" * 60)

    df = load_ff6_daily()
    regimes = load_regimes()

    # Align dates
    common = df.index.intersection(regimes.index)
    print(f"Common dates between factors and regimes: {len(common)}")
    df = df.loc[common]
    regimes = regimes.loc[common]

    # Run baselines
    corr_result = rolling_correlation_baseline(df, regimes)
    granger_result = rolling_granger_baseline(df, regimes)
    spill_result = spillover_baseline(df, regimes)

    # Combine
    output = {
        "rolling_correlation": corr_result,
        "rolling_granger": granger_result,
        "spillover": spill_result,
        "metadata": {
            "n_obs": len(df),
            "date_range": [str(df.index[0].date()), str(df.index[-1].date())],
            "regime_counts": regimes["regime_label"].value_counts().to_dict(),
            "corr_window": 60,
            "granger_window": 250,
            "spillover_window": 250,
            "fevd_horizon": 5,
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nRolling Correlation (60-day):")
    print(f"  HML->SMB mean corr: {corr_result['overall']['hml_to_smb_mean']:.4f}")
    print(f"  SMB->HML mean corr: {corr_result['overall']['smb_to_hml_mean']:.4f}")
    print(f"  % HML leads: {corr_result['overall']['pct_hml_leads']:.1f}%")

    print(f"\nRolling Granger (250-day):")
    print(f"  HML->SMB significant: {granger_result['overall']['pct_hml_sig']:.1f}%")
    print(f"  SMB->HML significant: {granger_result['overall']['pct_smb_sig']:.1f}%")

    print(f"\nFEVD Spillover (250-day VAR(1)):")
    print(f"  HML->SMB mean spillover: {spill_result['overall']['hml_to_smb_mean']:.4f}")
    print(f"  SMB->HML mean spillover: {spill_result['overall']['smb_to_hml_mean']:.4f}")


if __name__ == "__main__":
    main()
