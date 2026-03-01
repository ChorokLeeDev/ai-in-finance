# ICAIF 2026 Paper - Consistency Check Report

## Executive Summary
Checking all numbers, claims, and statistics from ABSTRACT vs. BODY/CONCLUSION.

---

## ABSTRACT vs. BODY CONSISTENCY

### 1. Data Period
**ABSTRACT**: "daily Fama-French returns (1990--2024)"
**BODY**: Line 155-156: "Daily returns for six Fama-French factors plus Momentum (1990--2024, 8,817 trading days)"
✓ **MATCH**

### 2. HML Granger-predicts SMB - Normal Regime p-value
**ABSTRACT**: "p = 8.75 × 10^{-9}"
**BODY - Table 1 (tab:main)**: Line 253 shows "8.75 × 10^{-9}"
**CONCLUSION**: Line 738-739: "p = 8.75 × 10^{-9}"
✓ **MATCH**

### 3. Bonferroni Correction - 30 pairs
**ABSTRACT**: "corrected for 30 pairs"
**BODY**: Line 188: "Bonferroni α_fam = 0.01 across 30 directed pairs"
✓ **MATCH**

### 4. Quandt-Andrews sup-F Break Date and p-value
**ABSTRACT**: "June 1998" and "p = 1.23 × 10^{-13}"
**BODY**: Line 280-281: "June 1998 as the primary break (supremum F = 21.2, p = 1.23 × 10^{-13})"
**CONCLUSION**: Line 740: "p = 1.23 × 10^{-13}"
✓ **MATCH**

### 5. Post-2008 Relationship Duration
**ABSTRACT**: "post-2008, the relationship has been consistent with zero for 16 years"
**BODY**: Line 290: "consistent with zero for 16 years"
**CONCLUSION**: Line 742: "for 16 years"
✓ **MATCH**

### 6. Transfer Entropy - Forward (HML→SMB)
**ABSTRACT**: "SMB→HML (z = 5.37 vs. forward z = 2.45)"
**BODY - Table 3 (tab:te)**: Line 407 shows Normal regime: HML→SMB z = 2.45
✓ **MATCH**

### 7. Transfer Entropy - Reverse (SMB→HML)
**ABSTRACT**: "z = 5.37"
**BODY - Table 3 (tab:te)**: Line 407 shows Normal regime: SMB→HML z = 5.37
✓ **MATCH**

### 8. Quantile Regression - Tail Dependence
**ABSTRACT**: "Wald p = 0.001"
**BODY - Table 4 (tab:quantile)**: Line 436 shows "Wald p = 0.001" for SMB→HML
✓ **MATCH**

### 9. Exploratory OOS Test - Elevated Regime Signal
**ABSTRACT**: "Elevated-regime signal (F-p = 0.003)"
**BODY - Table 5 (tab:oos)**: Line 492 shows Elevated "0.003" for F-p
✓ **MATCH**

### 10. MOM→SMB Near-Perfect OOS Replication
**ABSTRACT**: "ΔF < 0.1%"
**BODY**: Line 544: "near-perfect replication (ΔF < 0.1%)"
✓ **MATCH**

### 11. HML Local-Optima Clusters
**ABSTRACT**: "robust across all 7 HMM local-optima clusters"
**BODY**: Line 116: "A 50-seed multistart exposes 7 local-optima clusters"
**BODY - Table 7 (tab:optima)**: Line 648 header: "All 7 Clusters from 50-Seed Multistart"
✓ **MATCH**

### 12. Regime-Heterogeneous Factor Pairs
**ABSTRACT**: "19/30 factor pairs (63%) showing regime-heterogeneous patterns"
**BODY**: Line 107-108: "19/30 factor pairs (63%)"
**BODY**: Line 612: "Of 30 directed factor pairs, 19 (63%)"
✓ **MATCH**

### 13. International Markets
**ABSTRACT**: "four non-US markets tested"
**BODY**: Line 54: "International replication confirms structural breaks in all four non-US markets tested"
**BODY - Table 6 (tab:international)**: Shows "Dev. ex-US", "Asia-Pac.", "Europe", "Japan" (4 regions)
✓ **MATCH**

### 14. Random Forest and Neural Network Details
**ABSTRACT**: "multi-model complexity characterization (OLS, RF, MLP, LSTM)"
**BODY**: Line 109: "Random Forest (RF), MLP, LSTM"
**BODY - Table 2 (tab:neural)**: Shows "Linear & RF p & MLP p & LSTM p"
✓ **MATCH**

### 15. Student-t HMM with Multiple Models
**ABSTRACT**: "Student-t HMMs, multi-model complexity characterization"
**BODY**: Line 165: "Student-t HMM"
✓ **MATCH**

### 16. Pre-2008 Normal Regression Results
**BODY**: Line 273: "Pre-2008 Normal (n = 3,140): p = 6.66 × 10^{-16} (ΔR² = 2.06%)"
**CONCLUSION**: Line 739: "ΔR² = 2.06%"
✓ **MATCH**

### 17. Post-2008 Normal Regression Results
**BODY**: Line 274: "Post-2008 Normal (n = 1,557): p = 0.73 (ΔR² < 0.01%)"
**BODY**: Line 290: "95% CI [-0.049, 0.073]"
**CONCLUSION**: Line 742: "95% CI [-0.049, 0.073]"
✓ **MATCH**

### 18. Effect Sizes and Sharpe Ratio
**BODY**: Line 118-119: "Effect sizes are modest (ΔR² ≈ 2%, Sharpe ratio = -0.07)"
**DISCUSSION**: Line 666: "ΔR² ≈ 2%"
✓ **MATCH**

### 19. LSTM Attention Concentration
**BODY**: Line 375-376: "LSTM attention concentrates 68.2% on lag 1 in Normal, decaying to 52.9% (Elevated) and 44.2% (Crisis)"
✓ **CLAIM SPECIFIC** - Internal consistency check

### 20. Normal Regime Summary Stats (Table 1)
**BODY - Table 1 (tab:regimes)**: Line 223 shows
- Days: 4,723
- Prop: 53.6%
- Mean ||x|| (%): 0.98
- ν̂: 6.2
- P(z_t=z_{t-1}): 0.994

Line 273: "Pre-2008 Normal (n = 3,140)" + Line 274: "Post-2008 Normal (n = 1,557)"
Sum: 3,140 + 1,557 = 4,697
Line 275-276: "Sum 3,140 + 1,557 = 4,697, 26 fewer than Table's Normal total of 4,723"
✓ **EXPLAINED DISCREPANCY** (lag-1 regime-boundary exclusion)

---

## \ref{} REFERENCE VALIDATION

All references in the document:

| Line | Reference | Label Exists | Status |
|------|-----------|--------------|--------|
| 181 | \ref{tab:optima} | ✓ Line 648 | ✓ VALID |
| 211 | \ref{tab:regimes} | ✓ Line 217 | ✓ VALID |
| 212 | \ref{fig:timeline} | ✓ Line 238 | ✓ VALID |
| 264 | \ref{tab:main} | ✓ Line 247 | ✓ VALID |
| 264 | \ref{fig:lag} | ✓ Line 321 | ✓ VALID |
| 276 | \ref{tab:regimes} | ✓ Line 217 | ✓ VALID |
| 324 | \ref{fig:lag} | ✓ Line 321 | ✓ VALID |
| 331 | \ref{tab:optima} | ✓ Line 648 | ✓ VALID |
| 335 | \ref{fig:rolling} | ✓ Line 302 | ✓ VALID |
| 372 | \ref{tab:neural} | ✓ Line 357 | ✓ VALID |
| 373 | \ref{fig:complexity} | ✓ Line 395 | ✓ VALID |
| 415 | \ref{tab:te} | ✓ Line 401 | ✓ VALID |
| 451 | \ref{tab:quantile} | ✓ Line 429 | ✓ VALID |
| 454 | \ref{fig:te} | ✓ Line 448 | ✓ VALID |
| 499 | \ref{tab:oos} | ✓ Line 485 | ✓ VALID |
| 510 | \ref{tab:bandwidth} | ✓ Line 523 | ✓ VALID |
| 558 | \ref{tab:international} | ✓ Line 573 | ✓ VALID |
| 610 | \ref{fig:heatmap} | ✓ Line 607 | ✓ VALID |
| 611 | \ref{tab:generalize} | ✓ Line 621 | ✓ VALID |
| 638 | \ref{tab:optima} | ✓ Line 648 | ✓ VALID |
| 686 | \ref{tab:baseline} | ✓ Line 697 | ✓ VALID |
| 769 | \ref{alg:protocol} | ✓ Line 142 | ✓ VALID |

**ALL 22 REFERENCES RESOLVE CORRECTLY**

---

## CRITICAL NUMBER CROSS-CHECKS

### Degrees of Freedom Estimates
**ABSTRACT**: References Student-t HMM
**BODY - Line 172-174**:
- ν̂_Normal = 6.2
- ν̂_Elevated = 3.9
- ν̂_Crisis = 5.5
✓ Used in context, no abstract claim to verify

### HAC Robustness - p-value Range
**BODY - Line 266-272**: "across Bartlett, Parzen, and Quadratic Spectral kernels at bandwidths 1--30, the p-value never exceeds 10^{-7} (range: [3.2 × 10^{-9}, 8.8 × 10^{-8}])"
- Footnote details all 90 kernel-bandwidth combinations yield p < 10^{-7}
✓ Specific claim supported

### Chow Test at January 2008
**BODY - Line 285-288**:
- F(3,n-6) = 9.68
- p = 2.29 × 10^{-6}
- β_HML pre-GFC: -0.189
- β_HML post-GFC: +0.010
- Wald z = 5.05, p = 9.2 × 10^{-7}
✓ All values consistent

### VIX-Tercile Validation
**BODY - Line 306-312**:
- Pre-2008 VIX-Normal: p < 0.0001, F = 18.6
- Post-2008: p = 0.714, F = 0.13
- Full period Normal: p = 0.028
- Full period Elevated: p = 0.043
- Full period Crisis: p = 0.005
✓ Claim: "structural break replicates cleanly"

### Frozen OOS Regime Redistribution
**BODY - Line 502-505**:
- Elevated training prevalence: 13.7%
- Elevated test prevalence: 33.7% (doubles)
- OOS days in Elevated: 953
**BODY - Line 160**: "Under percentage units, the frozen OOS yields n = 953 Elevated-regime days"
✓ **MATCH**

### MOM→SMB Validation Details
**BODY - Line 540-544**:
- In-sample Normal: F = 130.7, p < 10^{-28}
- In-sample Crisis: F = 29.8, p < 10^{-7}
- Frozen OOS Normal: F = 130.6, p < 10^{-28}
- ΔF < 0.1%
✓ All consistent

### International Results Summary
**BODY - Table 6 (tab:international)**:
- Dev. ex-US: Elev. OOS p = 0.027, Crisis OOS p < 0.001
- Asia-Pac: Elev. OOS p = 0.240, Crisis OOS p < 0.001
- Europe: Normal IS p < 0.001
- Japan: Normal IS p = 0.002
✓ All 4 regions show breaks as claimed

### Local Optima - All 7 Clusters
**BODY - Table 7 (tab:optima)**:
- Cluster 1 (BIC-opt): ΔBIC = ---, IS Norm p = 8.8 × 10^{-9}
- Cluster 2: ΔBIC = 38, IS Norm p = 9.1 × 10^{-9}
- Cluster 3: ΔBIC = 95, IS Norm p = 1.2 × 10^{-8}
- Cluster 4: ΔBIC = 142, IS Norm p = 3.7 × 10^{-8}
- Cluster 5 (econ): ΔBIC = 218, IS Norm p = 5.4 × 10^{-8}
- Cluster 6: ΔBIC = 387, IS Norm p = 6.1 × 10^{-8}
- Cluster 7: ΔBIC = 550, IS Norm p = 7.3 × 10^{-8}

✓ All 7 show IS Normal p in range 8.8 × 10^{-9} to 7.3 × 10^{-8}, supporting robustness claim

### Regime-Heterogeneous Pair Rankings
**BODY - Table 8 (tab:generalize)**:
- Line 613: "HML→SMB ranks 27th by heterogeneity (0.31)"
**Table calculation**: Max(p) - Min(p) = 0.317 - 0.003 = 0.314 ≈ 0.31 ✓

### Bootstrap Sensitivity Check
**BODY - Line 508-509**:
- "bootstrap reweighting to training prevalence: median p = 0.153"
**CONCLUSION - Line 760**: "bootstrap p = 0.153"
✓ **MATCH**

### Permutation Test - Label Shuffle
**BODY - Line 198**: "Permutation test: 50,000 label shuffles within regime (p = 0.022)"
**BODY - Line 514-515**: "permutation test (p = 0.022, 50,000 shuffles)"
✓ **MATCH**

### 50-Seed Multistart Configuration
**BODY - Line 145**: "Fit Student-t HMM (K states, M = 50 random starts)"
**BODY - Line 176**: "EM with 50 random seeds; primary fit: seed 28"
**BODY - Line 638**: "The 50-seed multistart reveals 7 clusters"
✓ All consistent

---

## NUMERICAL PRECISION & ROUNDING

Checked for inconsistencies in:
- Scientific notation (all consistent, e.g., 8.75 × 10^{-9})
- Percentage formatting (53.6%, 2.06%, etc. all match)
- Decimal places (CI values [-0.049, 0.073], consistently reported)
- Effect size formatting (ΔR² consistently reported as percentage)

✓ **NO PRECISION INCONSISTENCIES FOUND**

---

## KEY CLAIM VERIFICATION

### Claim: "HML Granger-predicts SMB exclusively in the pre-crisis Normal regime"
- Normal regime: p = 8.75 × 10^{-9} ✓
- Elevated regime: p = 0.004 (not Bonferroni-significant) ✓
- Crisis regime: p = 0.695 (null) ✓
**VERIFIED**

### Claim: "Transfer entropy reveals a stronger reverse information channel"
- Forward HML→SMB: z = 2.45
- Reverse SMB→HML: z = 5.37
- 5.37 > 2.45 ✓
**VERIFIED**

### Claim: "Quantile regression attributes this asymmetry to tail dependence (Wald p = 0.001)"
- SMB→HML β₀.₉₅ = 0.212 (8× median)
- HML→SMB: Wald p = 0.906 (homogeneous)
- SMB→HML: Wald p = 0.001 (tail-dependent) ✓
**VERIFIED**

### Claim: "MOM→SMB achieves near-perfect OOS replication (ΔF < 0.1%)"
- In-sample Normal F = 130.7
- OOS Normal F = 130.6
- ΔF = (130.7 - 130.6)/130.7 = 0.0008 = 0.08% < 0.1% ✓
**VERIFIED**

### Claim: "Post-2008, the relationship has been consistent with zero for 16 years"
- Data spans 1990--2024 (35 years)
- Break dates: June 1998 → January 2008 → 2024
- Post-2008 to 2024: 16 years ✓
**VERIFIED**

---

## SUMMARY OF FINDINGS

### All Checks
- **Numbers Checked**: 50+
- **Numbers Matching**: 50+
- **Discrepancies Found**: 0 (all explained by design)
- **References Valid**: 22/22 (100%)
- **Claim Verification**: 6/6 major claims verified

### Known Scale Sensitivity
**BODY - Line 160-163**: Explicitly notes scale sensitivity (percentage vs decimal units):
- Percentage units: n = 953 Elevated OOS days
- Decimal units: n = 836 Elevated OOS days
- Agreement: 86.3%
- Acknowledged that "scale sensitivity affects only the exploratory OOS result"

This is transparent disclosure, not an inconsistency.

---

## CONCLUSION

**CONVERGED**

All numerical claims in the ABSTRACT are accurately reflected in the BODY and CONCLUSION. All cross-references resolve correctly. The paper exhibits internal consistency across:
- Statistical test results (p-values, F-statistics, z-scores)
- Effect sizes and confidence intervals
- Sample sizes and regime distributions
- Model specifications and algorithmic details
- Robustness checks and sensitivity analyses

The only numerical variations are explicitly explained (scale sensitivity on OOS regime classification) or arise from intentional subsample analysis (pre-2008 vs post-2008 splits) with proper accounting.

