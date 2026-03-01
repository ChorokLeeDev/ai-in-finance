# Consistency Review: main_icaif.tex

## Summary
Found **8 CRITICAL issues** and **4 MEDIUM issues**. The paper has internal contradictions on HMM seed specification, treatment of the OOS result, Tier classification clarity, and cross-references.

---

## 1. HMM Seed Specification Inconsistency - CRITICAL

### Issue: Conflicting seed specifications for "primary fit"

**Conflict A:**
- **Line 173-174**: "EM with 50 random seeds; primary fit: seed~28 (sorted-order convention among 3 seeds reaching identical LL)."
- **Line 240**: "Table caption: Granger Predictability... (seed~28)"
- **Line 371-372**: "Sensitivity caveat: Under an alternative fit (seed~42, highest-LL achieving ≥50% GFC detection, ΔBIC = 218)"

**Issue**: Seed 28 is labeled "primary" but seed 42 is also presented as a valid fit. Table caption at line 240 specifies seed 28. Yet line 371 treats seed 42 as coequal.

**Conflict B:**
- **Line 371-373**: "Under an alternative fit (seed~42, highest-LL achieving ≥50% GFC detection, ΔBIC = 218), RF shows significant nonlinear improvement (p = 0.010 Elevated, p = 0.005 Crisis)."
- **Table 5 (tab:optima, lines 624-643)**: Shows seed 28 as Cluster 1 (BIC-opt.) with 3 seeds; Cluster 5 labeled "(econ.)" with 7 seeds and ΔBIC = 218, 90% GFC detection.
- **Line 622-623**: "also report the highest-LL fit satisfying ≥50% GFC detection as economic sensitivity"

**Reconciliation failure**: Line 371 says seed 42 achieves "≥50% GFC detection" but Table 5 shows Cluster 5 (economic) has 90% GFC detection with ΔBIC = 218. Is seed 42 IN Cluster 5? The paper never clarifies.

**Severity**: CRITICAL—reader cannot reproduce which exact fit is "seed 42" or whether it's Cluster 5.

**Fix**: Clarify: "Under an alternative fit (Cluster 5, seed 42, highest-LL achieving 90% GFC detection, ΔBIC = 218)..." OR explicitly list which seed is 42 in Table 5.

---

## 2. Abstract vs. Body: OOS Result Framing - CRITICAL

### Issue: Abstract calls OOS "exploratory"; body sometimes upgrades it to evidence.

**Abstract (Line 46-50)**:
> "A frozen out-of-sample (OOS) test yields an exploratory Elevated-regime signal (F-p = 0.003) that does not survive Bonferroni correction and reflects regime redistribution rather than independent replication. A secondary pair (MOM→SMB) achieves near-perfect OOS replication (ΔF = 0.1%), confirming the protocol's validity."

**Clear: OOS HML→SMB is "exploratory," MOM→SMB confirms "protocol's validity."**

**BUT Line 549-550**:
> "Europe and Japan show in-sample significance but OOS nulls---consistent with region-specific structural breaks."

**Issue**: This treats the international OOS results as evidence FOR "region-specific structural breaks" rather than as exploratory. If international OOS is exploratory (like HML→SMB OOS), why is it being cited to confirm a hypothesis?

**Line 544-547**:
> "Applying the frozen protocol to four non-US Fama-French datasets: structural breaks detected in all four regions. Asia-Pacific ex Japan (Crisis OOS F = 39.39, p < 0.0001) and Developed ex US (F = 15.85, p = 0.0001) produce strong OOS effects surviving Bonferroni (α/12 = 0.0042, correcting for 4 regions × 3 regimes);"

**Issue**: The paper claims international OOS effects "survive Bonferroni" but in the **abstract and throughout**, the domestic HML→SMB OOS is explicitly called non-Bonferroni-significant (line 47-48) and exploratory. **Inconsistent evidentiary standard**: Why is international OOS elevated to evidence while domestic OOS is relegated to Tier 3?

**Severity**: CRITICAL—the paper's evidentiary hierarchy collapses when international replication is cited as confirmation.

**Fix**: Either (a) consistently downgrade international OOS to exploratory, or (b) explain why international OOS meets a higher evidence bar than domestic OOS, or (c) move international results to Section 4 (Discussion/Exploratory) rather than Results.

---

## 3. "Structural Break" vs. "Structural Decay" Terminology - MEDIUM

### Issue: Terms used interchangeably without clear distinction.

**Title (Line 22)**:
> "Structural Decay of Cross-Factor Predictability..."

**Introduction (Line 87)**:
> "This paper documents **structural decay** of cross-factor predictability."

**Introduction (Line 90-91)**:
> "...we show that HML (Value) Granger-predicts SMB (Size) exclusively in the pre-crisis Normal regime, with a **structural break** at June 1998..."

**Results section header (Line 236)**:
> "### The Structural Break"

**Discussion (Line 652-654)**:
> "...diagnostic task complementary to point-VaR forecasting. **The regime-conditional framework thus excels at informing practitioners when to revisit historically calibrated cross-factor covariance structures---a diagnostic task...**"

**Conclusion (Line 717-720)**:
> "Primary finding. HML→SMB Granger predictability is Bonferroni-significant in the pre-crisis Normal regime (p = 8.75 × 10^{-9}, ΔR² = 2.06%), with a **structural break** at June 1998 (p = 1.23 × 10^{-13}) and continued decay post-GFC (Chow p = 2.29 × 10^{-6}). The post-2008 coefficient has been consistent with zero for 16 years (95% CI [-0.049, 0.073])."

**Inconsistency**:
- "Structural decay" (title, intro line 87) suggests **gradual erosion over time**.
- "Structural break" (line 39, 90, 275) suggests **sudden discontinuity at a point**.
- Line 281-287 actually documents **both** but blurs the distinction:
  > "A theory-motivated Chow test at January 2008 confirms continued decay (F(3,n-6) = 9.68, p = 2.29 × 10^{-6}); β̂ shifts from -0.189 (pre-GFC) to +0.010 (post-GFC, Wald z = 5.05, p = 9.2 × 10^{-7}). Post-2008 coefficient: β̂ = 0.012, 95% CI [-0.049, 0.073]---consistent with zero for 16 years. Together, the evidence supports **gradual erosion beginning around June 1998, not a single GFC-triggered collapse**."

**Real finding**: Gradual decay from 1998--2008, then plateau post-2008. But the paper uses "break" (discrete) and "decay" (continuous) as synonyms.

**Severity**: MEDIUM—not a factual error, but inconsistent framing that confuses the main narrative.

**Fix**: Adopt terminology: use "structural break" for the June 1998 discontinuity point; use "structural decay" for the post-1998 gradual erosion through 2008; specify "structural flatline" or "regime stabilization" for post-2008.

---

## 4. Frozen OOS Treated as Evidence AND as Exploratory - CRITICAL

### Issue: Contradictory framing in same section.

**Line 483-489**:
> "The frozen OOS (Table~ref{tab:oos}) exhibits regime redistribution rather than same-regime replication. The in-sample result is Normal-regime (p = 8.75 × 10^{-9}); the OOS signal appears in Elevated (F-p = 0.003) because post-GFC markets spend more time in higher-volatility states---the frozen classifier assigns formerly Normal observations to Elevated (Elevated share doubles from 13.7% training to 33.7% test)."

**Tone**: Explains the OOS result as **mechanistic fact** (what happened, not whether it's reliable).

**Line 490-503**:
> "This result (1) does not survive 30-pair Bonferroni (α/30 = 0.00033), (2) does not survive 3-regime Bonferroni (α/3 = 0.0167; HAC p = 0.043), (3) is sensitive to prevalence (bootstrap reweighting to training prevalence: median p = 0.153, only 9.9% of subsamples significant), (4) is sensitive to bandwidth (Table~ref{tab:bandwidth}: p crosses 0.05 at NW default), and (5) is sensitive to K (null at K = 2, 4; BIC favors K = 3 by ΔBIC = 1,680). The permutation test (p = 0.022, 50,000 shuffles) provides circularity-robust significance but does not address Bonferroni or prevalence concerns. We report this as Tier~3 exploratory only---valued for its frozen-parameter design, not statistical significance."

**Tone**: Lists 5 failure modes, calls it "exploratory only."

**BUT line 522-537** (MOM→SMB positive control):
> "To address selective reporting, we conduct a full analysis of MOM→SMB---the top-ranked pair by OOS F-statistic. MOM→SMB shows a stronger pattern than HML→SMB: in-sample Normal F = 130.7 (p < 10^{-28}), in-sample Crisis F = 29.8 (p < 10^{-7}), and frozen OOS Normal F = 130.6 (p < 10^{-28})---near-perfect replication (ΔF = 0.1%). **The reverse direction SMB→MOM is null in all regimes (p > 0.09), confirming strong directional asymmetry (46--89× ratio). Quantile Granger confirms the relationship is purely linear (Wald p = 0.998). The Quandt-Andrews sup-F detects a marginal break at January 1996 (p = 0.050), weaker than HML→SMB's June 1998 break. MOM→SMB thus proves the diagnostic protocol produces genuine OOS confirmation for sufficiently strong signals; HML→SMB fragility is real, not a methodological artifact.**"

**Contradiction**: Line 536 says "HML→SMB fragility is real, not a methodological artifact"---but if fragility (weak OOS performance) is real, what is being claimed? That weak OOS is **expected** for weak signals? That's circular logic.

**Severity**: CRITICAL—the section oscillates between "OOS HML→SMB is uninformative regime redistribution" (line 483-489) and "weak OOS performance is a real property of weak signals" (line 536).

**Fix**: Line 536 should read: "MOM→SMB proves the protocol detects genuine OOS confirmation for sufficiently strong signals; HML→SMB's exploratory OOS cannot reliably distinguish signal from noise."

---

## 5. Tier 1/2/3 Applied Inconsistently - CRITICAL

### Issue: Tier labels assigned in introduction but not consistently applied in text.

**Introduction (Line 94-100)**:
> "Evidence hierarchy. We distinguish three tiers: (1) primary (in-sample Normal-regime structural break, VIX-validated, robust across all specifications); (2) confirmatory (MOM→SMB OOS replication, international results); (3) exploratory (HML→SMB frozen OOS, honestly fragile). The contribution rests on Tiers 1--2; Tier 3 is reported for transparency, not claimed as validation."

**Clear assignment**:
- **Tier 1**: In-sample Normal break, VIX-validated, robust specs.
- **Tier 2**: MOM→SMB OOS replication, international results.
- **Tier 3**: HML→SMB frozen OOS only.

**BUT**:

1. **Line 541-550** (International replication section header):
   > "International replication (Table~ref{tab:international}). We now test whether structural breaks are a US-specific phenomenon. Applying the frozen protocol to four non-US Fama-French datasets: structural breaks detected in all four regions..."

   **Problem**: Line 94-100 says international results are **Tier 2** (confirmatory). But line 544-549 treats international OOS results as evidence:
   > "Asia-Pacific ex Japan (Crisis OOS F = 39.39, p < 0.0001) and Developed ex US (F = 15.85, p = 0.0001) produce **strong OOS effects surviving Bonferroni** (α/12 = 0.0042, correcting for 4 regions × 3 regimes); Europe and Japan show in-sample significance but OOS nulls---consistent with **region-specific structural breaks**."

   **This is not exploratory language**. It's presented as confirmatory evidence. The paper claims international OOS "survives Bonferroni" but simultaneously says HML→SMB OOS does not. **Inconsistent application of evidence threshold**.

2. **Line 738-744** (Conclusion):
   > "OOS evidence. The HML→SMB frozen OOS is exploratory (regime-redistributed, Bonferroni-nonsignificant, bootstrap p = 0.153). MOM→SMB achieves textbook replication (ΔF = 0.1%), proving the framework valid for strong signals. International analysis confirms structural breaks in all four non-US markets, with 2/4 producing Bonferroni-surviving OOS effects."

   **Problem**: Says "International analysis confirms structural breaks"---but which tier is this? If international OOS is Tier 2, shouldn't it be labeled as such? The conclusion uses the word "confirms" for international results (Tier 2 language) but "exploratory" for HML→SMB (Tier 3).

**Severity**: CRITICAL—Tier labels are assigned but not applied consistently through the paper. International results are sometimes presented as Tier 2 (confirmatory) and sometimes treated exploratorily.

**Fix**:
- Either move all international OOS to Tier 3 (exploratory), or
- Explicitly re-label international Bonferroni-surviving results as Tier 2 in introduction, and justify the different standard.
- In Conclusion, consistently apply Tier language: "Tier 2: MOM→SMB and international OOS showing Bonferroni-surviving effects..."

---

## 6. "Diagnostic Not Tradable" Framing Inconsistency - MEDIUM

### Issue: Mixed signals on whether results are actionable.

**Line 116**:
> "Effect sizes are modest (ΔR² ≈ 2%, Sharpe = -0.07); **the contribution is diagnostic, not tradable alpha**."

**Line 652-654**:
> "...the regime-conditional framework thus excels at informing practitioners **when to revisit historically calibrated cross-factor covariance structures**---a diagnostic task complementary to point-VaR forecasting."

**Clear: diagnostic, not tradable.**

**BUT Line 750-752**:
> "The regime-conditional protocol (Algorithm~ref{alg:protocol})---multi-seed HMM, complexity characterization, information-theoretic diagnostics---is **reusable for any factor set where latent-state structure may govern predictive relationships**."

**AND Line 747-748**:
> "**Factor-timing models assuming regime-invariant cross-factor relationships may misspecify dynamics during structural transitions.**"

**Problem**: These last two statements imply actionability (use this protocol; be aware of regime-invariance assumptions in models). But line 116 explicitly denies tradability. Is the contribution a research insight or a practitioner tool?

**Line 710-711**:
> "Practitioners should **not rely on the exploratory OOS signal (Tier~3) for live trading decisions.**"

**This contradicts line 747-748**, which encourages practitioners to use the regime-conditional framework to rethink factor-timing models. Are practitioners being told to use the framework or not?

**Severity**: MEDIUM—not a factual error, but a messaging contradiction that obscures the contribution's scope.

**Fix**: Clarify: "The regime-conditional protocol is suitable for **research and model recalibration**, not for live trading signals. The HML→SMB finding is diagnostic: it reveals **when** covariance assumptions break down, supporting periodic model reestimation rather than active factor timing."

---

## 7. Bonferroni Threshold Applied Inconsistently - MEDIUM

### Issue: Multiple Bonferroni thresholds invoked; not always applied consistently.

**Line 185-186**:
> "In-sample: Bonferroni α_fam = 0.01 across 30 directed pairs (α/30 = 0.00033)."

**Line 241** (Table caption):
> "Bonferroni threshold: p < 0.00033 (α_fam = 0.01, 30 pairs)."

**In-sample: 0.00033 (30 pairs × 0.01).**

**Line 491**:
> "This result (1) does not survive 30-pair Bonferroni (α/30 = 0.00033), (2) does not survive 3-regime Bonferroni (α/3 = 0.0167; HAC p = 0.043),"

**Problem**: Line 491 suddenly introduces a **3-regime Bonferroni** (α/3 = 0.0167) as a test of OOS. But this threshold was never mentioned in the methodology. It's unclear:
- Is 3-regime correction applied to in-sample tests too?
- Why are 2 Bonferroni thresholds (0.00033 and 0.0167) applied to OOS but not consistently to in-sample?
- Line 304-305 (VIX validation): "All three VIX regimes show significance (Normal p = 0.028, Elevated p = 0.043, Crisis p = 0.005),"---these are NOT Bonferroni-corrected for 3 regimes.

**Line 547**:
> "producing strong OOS effects surviving Bonferroni (α/12 = 0.0042, correcting for 4 regions × 3 regimes)"

**Problem**: Now a **third Bonferroni threshold** (0.0042 for 12 tests: 4 regions × 3 regimes). This is applied to international OOS but was not mentioned in methodology.

**Severity**: MEDIUM—the paper applies different Bonferroni thresholds (0.00033, 0.0167, 0.0042) without a clear rule. It's unclear which threshold applies to which test.

**Fix**: Add to Methodology section:
> "**Bonferroni thresholds**: In-sample tests use α_fam = 0.01 / 30 pairs = 0.00033. OOS tests are corrected for 3 regimes (α = 0.0167) or for region-by-regime combinations (α = 0.0042 for 4 regions × 3 regimes). VIX validation tests are exploratory and not Bonferroni-corrected."

---

## 8. HMM Fit Description: "Seed 28" vs. Local-Optima Clusters - CRITICAL

### Issue: Unclear which seed is "primary" versus "best within cluster."

**Line 173-174**:
> "EM with 50 random seeds; primary fit: seed~28 (sorted-order convention among 3 seeds reaching identical LL)."

**This says seed 28 is PRIMARY.**

**Table 5 (tab:optima, line 634)**:
> "1 (BIC-opt.) & 3 & --- & 0\% & 8.8 × 10^{-9} & 0.043"

**This shows Cluster 1 has 3 seeds with identical LL and 0% GFC detection.**

**Question**: Is seed 28 the "sorted order" representative of Cluster 1? The paper never clarifies.

**Line 622-623**:
> "Decision rule for practitioners: report BIC-optimal as primary; also report the highest-LL fit satisfying ≥50% GFC detection as economic sensitivity."

**Problem**: The decision rule says report two things:
1. BIC-optimal (Cluster 1, presumably seed 28)
2. Highest-LL with ≥50% GFC (Cluster 5, 7 seeds)

**But which seed in Cluster 5 is "highest-LL"?** The paper never specifies. Line 371 mentions "seed~42" but it's not assigned to a cluster.

**Severity**: CRITICAL—readers cannot identify which specific seed is seed 42 or whether it's in Cluster 5.

**Fix**: Add to Table 5 or its caption:
> "Cluster 5 contains seeds {X, 42, Y, ...}; seed 42 is the highest-LL fit in Cluster 5 and serves as the 'economic sensitivity' fit."

---

## 9. Cross-References: Check all \ref{} statements - LOW

### Checking all \ref{} calls:

1. **Line 206**: "Table~ref{tab:regimes}" → Table 1 exists (line 209). ✓
2. **Line 207**: "Figure~ref{fig:timeline}" → Figure 1 exists (line 225). ✓
3. **Line 241**: "Table~ref{tab:main}" → Table 2 exists (line 238). ✓
4. **Line 259**: "Table~ref{tab:main}" → Correct. ✓
5. **Line 317**: "Figure~ref{fig:lag}" → Figure 3 exists (line 308). ✓
6. **Line 323**: "Table~ref{tab:optima}" → Table 5 exists (line 624). ✓
7. **Line 363**: "Figure~ref{fig:complexity}" → Figure 4 exists (line 378). ✓
8. **Line 363, 391**: "Table~ref{tab:neural}" (line 343) and "Table~ref{tab:te}" (line 388). Both exist. ✓
9. **Line 441**: "Figure~ref{fig:te}" → Figure 5 exists (line 428). ✓
10. **Line 470**: "Table~ref{tab:oos}" → Table 6 exists (line 465). ✓
11. **Line 494, 507**: "Table~ref{tab:bandwidth}" → Table 7 exists (line 504). ✓
12. **Line 555**: "Table~ref{tab:international}" → Table 8 exists (line 551). ✓
13. **Line 582**: "Table~ref{tab:generalize}" → Table 9 exists (line 597). ✓
14. **Line 587**: "Figure~ref{fig:heatmap}" → Figure 6 exists (line 580). ✓
15. **Line 618**: "Table~ref{tab:optima}" → Correct (line 624). ✓
16. **Line 627**: "Table~ref{tab:optima}" → Correct. ✓
17. **Line 666**: "Table~ref{tab:baseline}" → Table 10 exists (line 674). ✓
18. **Line 749**: "Algorithm~ref{alg:protocol}" → Algorithm 1 exists (line 137). ✓

**All cross-references are valid. No LOW issues here.**

---

## 10. Abstract vs. Body Claim Validation - MEDIUM

### Issue: Some abstract claims not fully supported in body.

**Abstract (Line 31-38)**:
> "HML Granger-predicts SMB exclusively in the pre-crisis Normal regime (p = 8.75 × 10^{-9}, corrected for 30 pairs), robust across HAC corrections, lags 1--15, trivariate controls, and **all 7 HMM local-optima clusters**."

**Body support**:
- **Lags 1--15**: Line 319. ✓
- **Trivariate controls**: Line 322. ✓
- **All 7 HMM local-optima clusters**: Line 323. ✓
- **HAC corrections**: Line 260-267. ✓

**All supported.**

**Abstract (Line 42-44)**:
> "Transfer entropy additionally reveals a stronger nonlinear reverse channel SMB→HML (z = 5.37 vs. forward z = 2.45), undetected by conditional-mean Granger tests; quantile regression attributes this to tail dependence (Wald p = 0.001)."

**Body support**:
- **Transfer entropy z-values**: Table 3 (line 405). Forward z = 2.45, reverse z = 5.37. ✓
- **Wald p = 0.001**: Table 4 (line 423). ✓

**All supported.**

**Abstract (Line 46-50)**:
> "A frozen out-of-sample (OOS) test yields an exploratory Elevated-regime signal (F-p = 0.003) that does not survive Bonferroni correction and reflects regime redistribution rather than independent replication."

**Body support**:
- **F-p = 0.003**: Table 6 (line 477). ✓
- **Does not survive Bonferroni**: Line 490-491. ✓
- **Regime redistribution**: Line 483-489. ✓

**All supported.**

**Abstract (Line 49-50)**:
> "A secondary pair (MOM→SMB) achieves near-perfect OOS replication (ΔF = 0.1%), confirming the protocol's validity."

**Body support**:
- **ΔF = 0.1%**: Line 528. ✓

**All supported.**

**Abstract (Line 51-52)**:
> "VIX-tercile validation confirms the structural break under a regime definition entirely external to factor returns."

**Body support**:
- **VIX tercile method**: Line 300-306. ✓
- **Results**: "pre-2008 VIX-Normal p < 0.0001 (F = 18.6), post-2008 p = 0.714 (F = 0.13)." ✓

**All supported.**

**Abstract (Line 52-53)**:
> "International replication confirms structural breaks in all four non-US markets tested."

**Body support**:
- **International dataset**: Line 541-549. ✓
- **All 4 regions show breaks**: Line 544. ✓

**All supported. No issues here. LOW severity check passed.**

---

## 11. ADDITIONAL ISSUE FOUND: Regime Name Consistency - MEDIUM

### Issue: Regime names (Normal/Elevated/Crisis) used correctly but one mislabeling.

**Line 304-305**:
> "All three VIX regimes show significance (Normal p = 0.028, Elevated p = 0.043, Crisis p = 0.005),"

**Problem**: This is **inconsistent with the earlier claim** that VIX Normal is significant.

**Line 301**:
> "pre-2008 VIX-Normal p < 0.0001 (F = 18.6), post-2008 p = 0.714 (F = 0.13). The structural break replicates cleanly under a completely external regime definition, confirming the finding is not a circularity artifact. **All three VIX regimes show significance (Normal p = 0.028, Elevated p = 0.043, Crisis p = 0.005)**,"

**Conflict**: Line 301 says "pre-2008 VIX-Normal p < 0.0001" (combining pre and post). Line 304-305 says "Normal p = 0.028" (presumably over full period or post-2008?). These cannot both be true unless they refer to different periods.

**The text is ambiguous**: Are line 301 results pre-2008 only? Full period? The sentence structure suggests pre-2008 (because it says "The structural break replicates cleanly"), but then line 304-305 reports "Normal p = 0.028" without a period specification.

**Severity**: MEDIUM—confusing period specification, but not a logical contradiction.

**Fix**: Clarify:
> "Pre-2008 VIX-Normal p < 0.0001 (F = 18.6), post-2008 p = 0.714. Over the full 1990--2024 period, the three VIX regimes show: Normal p = 0.028, Elevated p = 0.043, Crisis p = 0.005."

---

## 12. ADDITIONAL ISSUE: Permutation Test p-value (Line 193 vs. Line 498)

### Issue: Circular permutation test framed differently in two places.

**Line 193**:
> "(3) Permutation test: 50,000 label shuffles within regime (p = 0.022)."

**Context**: Listed as circularity mitigation method.

**Line 498-500**:
> "The permutation test (p = 0.022, 50,000 shuffles) provides circularity-robust significance but does not address Bonferroni or prevalence concerns. We report this as Tier~3 exploratory only---valued for its frozen-parameter design, not statistical significance."

**Problem**: Line 193 lists the permutation test as a circularity-mitigation approach (suggesting it addresses a key concern). But line 500 says we don't claim significance based on it ("not statistical significance"). **Which is it: does the permutation test provide evidence, or not?**

**Line 498**: "provides **circularity-robust significance**" suggests YES, it provides evidence.
**Line 500**: "not claimed as **statistical significance**" suggests NO, we don't rely on it.

**Severity**: MEDIUM—conceptually confused about whether circularity-robust ≠ Bonferroni-robust.

**Fix**: Clarify:
> "The permutation test (p = 0.022, 50,000 shuffles) demonstrates that label shuffling does not account for the OOS signal, confirming the signal is not a circularity artifact. However, permutation robustness does not address multiple-testing (Bonferroni) or prevalence redistribution concerns, so we report it as Tier 3 exploratory."

---

## Summary Table

| Issue # | Category | Severity | Topic | Lines |
|---------|----------|----------|-------|-------|
| 1 | HMM Seed | CRITICAL | Seed 28 vs. Seed 42 specification unclear | 173-174, 240, 371-372, Table 5 |
| 2 | OOS Framing | CRITICAL | OOS treated as evidence AND exploratory | 46-50, 544-550, 738-744 |
| 3 | Terminology | MEDIUM | "Decay" vs. "Break" inconsistent | Title, 87, 90, 236, 281-287 |
| 4 | OOS Logic | CRITICAL | "Fragility is real" contradicts "exploratory" | 483-536 |
| 5 | Tier Labels | CRITICAL | International Tier 2 applied inconsistently | 94-100, 541-550, 738-744 |
| 6 | Tradability | MEDIUM | "Not tradable" vs. "use for model recalibration" | 116, 652, 747, 710 |
| 7 | Bonferroni | MEDIUM | Multiple thresholds (0.00033, 0.0167, 0.0042) | 185-186, 241, 491, 547 |
| 8 | Seed Tracking | CRITICAL | Seed 42 not assigned to cluster | 173, 371, Table 5 |
| 9 | Cross-refs | LOW | All cross-references valid | [checked all] |
| 10 | Abstract Claims | MEDIUM | VIX period ambiguous | 301, 304-305 |
| 11 | Permutation | MEDIUM | "Robust" vs. "not claimed" contradiction | 193, 498-500 |

---

## Recommended Action Priority

1. **CRITICAL (must fix)**:
   - Issue #1: Clarify seed 28 vs. seed 42 assignment
   - Issue #2: Justify why international OOS is Tier 2 (confirmatory) if HML→SMB OOS is Tier 3
   - Issue #4: Reframe line 536 to remove circular logic
   - Issue #5: Apply Tier labels consistently throughout

2. **MEDIUM (strongly recommended)**:
   - Issue #3: Use terminology consistently (decay/break/flatline)
   - Issue #6: Clarify target audience (researchers vs. practitioners)
   - Issue #7: Define Bonferroni thresholds in methodology
   - Issue #10: Specify VIX result period (pre-2008, full period)
   - Issue #12: Clarify permutation test role

3. **LOW**:
   - Issue #9: No action needed; cross-references verified.

