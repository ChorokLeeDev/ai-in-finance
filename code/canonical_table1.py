#!/usr/bin/env python3
"""
canonical_table1.py — Build Table 1 for the causal regimes paper.

For each canonical regime (Normal, Elevated, Crisis):
  - n_days, proportion, mean factor-vector norm, nu, self-transition prob
  - Granger causality (lag-9) for HML→SMB and SMB→HML
    with boundary handling (all 9 lags must lie within the same regime)

Saves results to canonical_table1.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from statsmodels.tsa.stattools import grangercausalitytests

# ── paths ──────────────────────────────────────────────────────────────
BASE = Path("/Users/i767700/Github/ai-in-finance/papers/causal_regimes")
REGIMES_PATH = BASE / "results" / "canonical_regimes.json"
OUTPUT_PATH  = BASE / "results" / "canonical_table1.json"

LAG = 9
FACTORS = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]

# ── 1. Load canonical regime assignments ───────────────────────────────
print("Loading canonical regimes …")
with open(REGIMES_PATH) as f:
    canon = json.load(f)

meta = canon["metadata"]
hmm  = meta["hmm_params"]

# Build a date→regime Series
assign_df = pd.DataFrame(canon["assignments"])
assign_df["date"] = pd.to_datetime(assign_df["date"])
assign_df = assign_df.set_index("date").sort_index()
regime_series = assign_df["regime_name"]

# Map regime_id to parameters (nu, diag_A)
id_to_name = {0: "Normal", 1: "Elevated", 2: "Crisis"}
nu_map    = {id_to_name[i]: hmm["nu"][i]     for i in range(3)}
diagA_map = {id_to_name[i]: hmm["diag_A"][i] for i in range(3)}

print(f"  {len(regime_series)} days, range {regime_series.index[0].date()} – {regime_series.index[-1].date()}")

# ── 2. Load Fama-French 5 + Momentum daily data ───────────────────────
print("Downloading Fama-French 5-factor + momentum daily data …")
ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily",
                     "famafrench", start="1990-01-01")[0]
mom = web.DataReader("F-F_Momentum_Factor_daily",
                     "famafrench", start="1990-01-01")[0]

df = ff5.join(mom)
df.columns = ["MKT", "SMB", "HML", "RMW", "CMA", "RF", "MOM"]
df = df.drop("RF", axis=1)

# pandas-datareader returns PeriodIndex; convert to DatetimeIndex
if isinstance(df.index, pd.PeriodIndex):
    df.index = df.index.to_timestamp()

# Filter to 1990-01-02 .. 2024-12-31
df = df.loc["1990-01-02":"2024-12-31"]
print(f"  Factor data: {len(df)} days, {df.index[0].date()} – {df.index[-1].date()}")

# ── 3. Align factors with regime assignments ──────────────────────────
common_dates = df.index.intersection(regime_series.index)
df = df.loc[common_dates]
regime_series = regime_series.loc[common_dates]
print(f"  Aligned: {len(df)} common trading days")

# Factor values are in percent; divide by 100 for returns (norms in %)
# Keep in percent for norm computation (matches paper convention)
factor_matrix = df[FACTORS].values  # (T, 6)


# ── 4. Per-regime statistics + Granger tests ──────────────────────────
def granger_within_regime(y_col, x_col, regime_name, lag):
    """
    Run Granger causality test x → y at given lag, using only
    observations whose ENTIRE lag window lies in the same regime.

    Returns (F-stat, p-value, n_clean).
    """
    # Boolean mask for this regime
    mask = (regime_series == regime_name).values  # shape (T,)

    # For each t, check that t and t-1 … t-lag are all in-regime
    clean = np.ones(len(mask), dtype=bool)
    for k in range(lag + 1):
        clean[k:] &= mask[k: len(mask)]
    # First `lag` observations can't have full history
    clean[:lag] = False

    n_clean = int(clean.sum())
    if n_clean < lag + 5:
        return np.nan, np.nan, n_clean

    # Build the two columns: [y, x]  (statsmodels convention: col 0 = y, col 1 = x)
    y_vals = df[y_col].values
    x_vals = df[x_col].values

    # Extract the clean indices
    idx = np.where(clean)[0]

    # We need contiguous-ish data for the VAR inside grangercausalitytests.
    # Instead, build the matrix manually and use grangercausalitytests on it.
    # grangercausalitytests expects a 2-col array of shape (n, 2).
    # It builds lags internally, so we need contiguous rows.
    # Strategy: collect *runs* of consecutive clean indices, keep runs > lag.
    runs = []
    start = idx[0]
    for i in range(1, len(idx)):
        if idx[i] != idx[i - 1] + 1:
            if idx[i - 1] - start + 1 > lag:
                runs.append((start, idx[i - 1]))
            start = idx[i]
    if idx[-1] - start + 1 > lag:
        runs.append((start, idx[-1]))

    # Concatenate runs (each run is internally contiguous and all-in-regime)
    pieces = []
    for s, e in runs:
        pieces.append(np.column_stack([y_vals[s:e + 1], x_vals[s:e + 1]]))

    if not pieces:
        return np.nan, np.nan, n_clean

    data = np.vstack(pieces)
    if len(data) <= lag + 2:
        return np.nan, np.nan, n_clean

    # Run Granger test (suppress printed output)
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = grangercausalitytests(data, maxlag=[lag], verbose=True)

    # Extract F-test result at the requested lag
    f_stat = result[lag][0]["ssr_ftest"][0]
    p_val  = result[lag][0]["ssr_ftest"][1]

    return f_stat, p_val, n_clean


print("\nComputing per-regime statistics and Granger tests …\n")

rows = []
regime_order = ["Normal", "Elevated", "Crisis"]
total_days = len(df)

for rname in regime_order:
    mask = (regime_series == rname)
    n_days = int(mask.sum())
    proportion = n_days / total_days

    # Mean factor vector norm (L2 across 6 factors, per day, then mean)
    norms = np.linalg.norm(factor_matrix[mask.values], axis=1)
    mean_norm = float(norms.mean())

    nu = nu_map[rname]
    self_trans = diagA_map[rname]

    # Granger: HML → SMB  (x=HML causes y=SMB)
    f_hs, p_hs, n_clean_hs = granger_within_regime("SMB", "HML", rname, LAG)
    # Granger: SMB → HML  (x=SMB causes y=HML)
    f_sh, p_sh, n_clean_sh = granger_within_regime("HML", "SMB", rname, LAG)

    # n_clean should be the same for both directions (same regime mask)
    n_clean = n_clean_hs

    row = {
        "regime":          rname,
        "n_days":          n_days,
        "proportion":      round(proportion, 4),
        "mean_norm":       round(mean_norm, 4),
        "nu":              round(nu, 2),
        "self_transition": round(self_trans, 4),
        "hml_to_smb_F":    round(f_hs, 3) if not np.isnan(f_hs) else None,
        "hml_to_smb_p":    round(p_hs, 6) if not np.isnan(p_hs) else None,
        "smb_to_hml_F":    round(f_sh, 3) if not np.isnan(f_sh) else None,
        "smb_to_hml_p":    round(p_sh, 6) if not np.isnan(p_sh) else None,
        "n_clean":         n_clean,
    }
    rows.append(row)

    # Pretty-print
    print(f"── {rname} ──")
    print(f"  n_days       = {n_days}  ({proportion:.1%})")
    print(f"  mean ||x||   = {mean_norm:.4f}")
    print(f"  ν            = {nu:.2f}")
    print(f"  P(stay)      = {self_trans:.4f}")
    print(f"  n_clean      = {n_clean}")
    p_hs_str = f"{p_hs:.2e}" if p_hs is not None and not np.isnan(p_hs) else "N/A"
    p_sh_str = f"{p_sh:.2e}" if p_sh is not None and not np.isnan(p_sh) else "N/A"
    print(f"  HML→SMB      F={f_hs:.3f}  p={p_hs_str}" if not np.isnan(f_hs) else "  HML→SMB      N/A")
    print(f"  SMB→HML      F={f_sh:.3f}  p={p_sh_str}" if not np.isnan(f_sh) else "  SMB→HML      N/A")
    print()

# ── 5. Save ────────────────────────────────────────────────────────────
output = {
    "description": "Table 1: Canonical regime summary with lag-9 Granger causality (boundary-clean)",
    "lag": LAG,
    "factors": FACTORS,
    "total_days": total_days,
    "regimes": rows,
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved → {OUTPUT_PATH}")
