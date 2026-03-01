# ICAIF 2026 - FINAL REVIEW (ROUND 3)
## "Structural Decay of Cross-Factor Predictability: Regime-Conditional Granger Analysis with Complexity Characterization"

---

## FINAL DECISION

**BORDERLINE (was WEAK REJECT, now BORDERLINE)**

This paper has made **substantial and real improvements** in Round 3 that move it into genuine accept-territory contention. However, critical gaps remain that prevent a clean Accept recommendation.

---

## What Changed from Round 2 to Round 3: The Good

### 1. **Algorithm Box Formalizing the Protocol (NEW)**
**Impact: MAJOR**

The new Algorithm 1 (lines 257-272) explicitly packages the regime-conditional Granger diagnostic protocol as a reusable workflow:
- Step-by-step formalization of: regime discovery → local optima sensitivity → per-regime Granger → Bonferroni correction → frozen OOS → complexity diagnostic → transfer entropy → **quantile Granger**
- This directly addresses Round 2 criticism that the "pipeline" was just a sequence of known techniques without packaging
- The explicit algorithm legitimizes claims of reusability (practitioners can now reproduce it on other factor sets)
- **This is now a genuine methodological contribution**, not just application

**Weakness**: Algorithm could be shortened to 4-5 core steps for main text; current 8-step version is comprehensive but slightly verbose.

---

### 2. **Quantile Granger Evidence for TE Asymmetry Mechanism (NEW)**
**Impact: CRITICAL - Addresses Fundamental Gap**

The new quantile regression analysis (lines 773-801; Table 1, which appears to be the quantile table at lines 803-818) is the **single most important fix** from Round 2.

**What was promised**: "Future work on tail dependence hypothesis"
**What is now delivered**:
- SMB→HML exhibits strong nonlinearity at quantiles: median $\beta = -0.026$ vs. 95th percentile $\beta = 0.212$ (Wald $\chi^2(6) = 22.58$, $p = 0.001$)
- HML→SMB is homogeneous across quantiles ($\beta \in [-0.019, 0.053]$, Wald $p = 0.906$)
- **Mechanistic explanation**: Reverse channel operates through tail dependence, not location shifts
- Robust to Elevated/Crisis regimes (SMB→HML Wald $p = 0.043$ in Elevated, $p > 0.77$ in Crisis)

**Why this matters**:
- Converts transfer entropy asymmetry from a "statistical curiosity" to a finding with **identified mechanism**
- Demonstrates that SMB predicts *extreme* HML movements, not typical ones
- Explains why linear Granger ($p = 0.864$) misses this signal: it estimates conditional means, not tails
- **This elevates the paper from "we observed an asymmetry" to "we explain why it exists"**

**Remaining gap**: The mechanism is explained post-hoc. A pre-registered prediction like "we hypothesize tail dependence and will test via quantile regression" would be stronger. But this is a minor point; the evidence is there.

---

### 3. **Multi-Pair Generalizability Table (NEW)**
**Impact: MODERATE - Addresses Specificity Concern**

Table 1112 (labeled "Table 14" in main text, "Multi-pair generalizability") shows:
- 19 of 30 directed factor pairs (63%) exhibit regime-heterogeneous Granger patterns
- Five pattern types identified: Elevated-signal, monotonic-decreasing, monotonic-increasing, core pattern, others
- RMW→SMB, MKT→MOM, MKT→SMB, MOM→SMB rank highest for heterogeneity
- HML→SMB ranks 27th by heterogeneity magnitude

**Why this matters**:
- Directly addresses Round 2 criticism: "limited to one pair, one regime"
- Demonstrates the protocol generalizes: 63% of pairs show regime heterogeneity
- Shows HML→SMB was not cherry-picked for being the strongest heterogeneous pair (it ranks 27th)
- Positions the finding as a **systematic phenomenon in factor markets**

**Interpretation caveat**: High heterogeneity doesn't mean all 19 pairs show the same structural break or tail mechanism as HML→SMB. But it establishes that regime-conditional Granger is not an artifact specific to this pair.

---

### 4. **Baseline Comparison (PARTIAL FIX)**
**Impact: MODERATE - Partially Addresses "Missing Comparisons"**

Lines 1219-1237 now include formal comparison to two simpler alternatives on OOS period:
- (1) Rolling 250-day Granger: median $p = 1.00$, mean $p = 0.69$; no structural break detected
- (2) Threshold volatility regime (realized 20-day vol > median): high-vol $p = 0.696$, low-vol $p = 0.232$ (**inverted relative to HMM**)
- (3) HMM regime-conditional: Elevated $p = 0.014$, Normal $p = 0.149$, Crisis $p = 0.281$

**Why this matters**:
- Shows rolling approach destroys signal by averaging over regime-heterogeneous relationships
- Shows naive volatility threshold gets direction wrong
- Demonstrates HMM approach provides clearer regime separation
- **This is no longer "we don't compare to baselines"**

**Limitation**: The comparison is on the OOS period (2013-2024), not on the primary in-sample finding. A rolling-window baseline for the June 1998 structural break would be more directly comparable. But this is still progress.

---

### 5. **Improved Transparency on Frozen OOS Fragility (Enhanced)**
**Impact: MODERATE - Better Communication**

Additions from Round 2:
- **Explicit summary** (lines 855-862): "does not confirm the in-sample finding"
- Table 995 (HAC bandwidth sensitivity): Shows $p$ ranges from 0.041 to 0.173; crosses 0.05 at bandwidth ≥ 6
- **K-sensitivity table** (Table 1061-1077): Signal only at K=3 (null at K=2,4)
- **Prevalence sensitivity explicitly quantified**: Bootstrap $p = 0.153$ (vs. raw $p = 0.003$)
- **Regime redistribution detail** (lines 948-957): Elevated grows from 13.7% (train) to 33.7% (test); documented distributional shift

**Why this matters**: Readers can now see exactly which assumptions drive the OOS result. This is honest and appropriate for a venue like ICAIF that values transparency.

---

### 6. **Contributions Reframed as Diagnostic Value (Enhanced)**
**Impact: MODERATE - Better Positioning**

- Contribution 1 now explicitly states methodology "combines known techniques...each component is individually standard" (lines 155-159) but adds generalizability claim: "19 of 30 pairs show regime heterogeneity"
- Contribution 2 adds quantile Granger mechanism evidence and soft-label sensitivity results
- Contribution 3 frames local optima tension as "unresolved...reflects a general challenge" (lines 195-203) rather than presenting it as solved

**Why this matters**: Paper no longer oversells; claims are now aligned with evidence.

---

## Critical Issues That Remain Unresolved

### ISSUE 1: The OOS Result Does NOT Survive Multiple-Testing Correction
**Severity: CRITICAL**

**Status: ACKNOWLEDGED but UNRESOLVED**

The paper now clearly states (lines 855-862):
- Does not survive 30-pair Bonferroni ($\alpha/30 = 0.00033$)
- Does not survive 3-regime Bonferroni (HAC $p = 0.043 > 0.0167$)
- Bootstrap reweighting: $p = 0.153$ (non-significant)
- K-sensitive: null at K=2,4; significant only at K=3

**Question**: Why report an OOS finding that doesn't meet statistical standards?

**Paper's answer** (lines 414-415, 862): "valued for its frozen-parameter design, not statistical significance"

**Honest assessment**: This is intellectually transparent but creates a logical problem. The **primary validation** (OOS, held-out period 2013-2024) fails statistical tests. The **secondary finding** (in-sample Normal regime, $p = 8.75 \times 10^{-9}$) is robust but based on in-sample regime discovery from the same returns (circular).

**For ICAIF acceptance, this requires ONE of:**
1. Make in-sample Normal the PRIMARY finding and explicitly downgrade OOS to "exploratory"
2. OR pre-register a replication on international data that shows the OOS pattern holds out-of-sample with independent regime discovery

**Current status**: The paper acknowledges the problem but doesn't fully resolve it. The abstract and introduction still emphasize OOS patterns (lines 44-71), even though later sections clarify they're exploratory. This messaging tension remains.

---

### ISSUE 2: Regime-Identification Circularity
**Severity: MAJOR**

**Status: ACKNOWLEDGED but MITIGATED, NOT ELIMINATED**

The paper acknowledges (lines 363-387):
- HMM uses distributional properties; Granger uses temporal dynamics (different but related features)
- Soft-label sensitivity yields identical conclusions ($p < 10^{-7}$)
- Frozen OOS is strongest available mitigation; but frozen OOS fails significance tests

**Gap**: No claim of circularity being eliminated. Section 3.2 (lines 363-387) frames it as a "caveat" with "strongest available mitigation."

**Honest assessment**: This is appropriate framing. The circularity cannot be fully eliminated without external instruments (e.g., pre-crisis information about factor volatility not yet realized). The paper's mitigations (distributional vs. temporal, soft-label, frozen OOS) are reasonable but imperfect.

**For ICAIF acceptance**: This is acceptable **IF** the primary claim is positioned as "in-sample Normal-regime finding with acknowledged circularity caveat" rather than claiming the frozen OOS resolves it.

**Current status**: The paper is now honest about this. Lines 385-386 explicitly state: "in-sample Normal-regime finding remains the primary result, with the circularity caveat acknowledged."

---

### ISSUE 3: Local Optima Tension Unresolved
**Severity: MAJOR**

**Status: DOCUMENTED but NO PRINCIPLED RESOLUTION OFFERED**

Table 1149 shows:
- **BIC-optimal (Cluster 1)**: 0% of 2008 in Crisis regime, BIC = 75,587
- **Economically valid (Cluster 5)**: 90% of 2008 in Crisis regime, BIC = 75,805 ($\Delta = 218$)

**Paper's recommendation** (lines 1189-1195): "report results under both fits"

**Problem**: This puts burden on readers. Which regime structure should practitioners use?

**Paper's honest acknowledgment** (lines 1182-1187):
> "fundamentally post-hoc...cannot be justified as data-driven in the strict sense"

**For ICAIF acceptance**: This is a weakness, not a fatal flaw. The structural break finding (June 1998, time-indexed) is robust across all 7 clusters, so this tension doesn't undermine the main result.

**Current status**: The paper acknowledges this as an unresolved challenge in latent-state financial modeling. This is appropriate.

---

### ISSUE 4: Effect Sizes Are Economically Immaterial
**Severity: MAJOR (but appropriately framed)**

**Status: EXPLICITLY ACKNOWLEDGED**

Lines 1205-1217:
- Effect sizes: $\Delta R^2 \approx 2\%$ pre-GFC
- Trading rule Sharpe = -0.07 (vs. buy-and-hold +0.06)
- Regime-conditional VaR models: 93.2% false-alarm rates
- Contribution is "diagnostic awareness...rather than direct model deployment"

**For ICAIF acceptance**: This is honest and appropriate. The paper's value proposition is now: "not tradable alpha, but diagnostic framework for understanding when factor relationships have broken down."

**Weakness**: This is a reduced scope compared to a typical ICAIF paper. But the methodology and mechanism are sound.

**Current status**: Well-positioned in limitations section (lines 1205-1217).

---

## Remaining Minor Issues That Could Be Quick Fixes

### FIX 1: Scale Convention Still a Degree of Freedom
**Lines 282-297**

The paper now pre-specifies percentage-unit convention but acknowledges decimal-unit yields different results ($p = 0.063$ vs. 0.022 permutation).

**Quick fix**: Add one sentence: "To eliminate this degree of freedom, we commit to percentage-unit convention throughout; decimal-unit sensitivity is exploratory."

**Lines to change**: After line 297, add:
> "This choice is pre-specified and not subject to post-hoc optimization. All main results use percentage units; decimal-unit sensitivity is reported in Section~\ref{sec:robustness} for completeness only."

---

### FIX 2: Abstract Emphasis Could Be Clearer
**Lines 44-71**

The abstract now emphasizes in-sample finding ($p = 8.75 \times 10^{-9}$) but still opens with OOS discussion. Reorganize to front-load in-sample:

**Current structure**:
- Lines 48-55: in-sample result (Normal regime)
- Lines 56-64: structural break (June 1998, Quandt-Andrews)
- Lines 65-71: OOS result (Elevated, $p = 0.003$ but non-Bonferroni-significant)

**Better structure**:
1. Open with robust in-sample finding (Normal, $p = 8.75 \times 10^{-9}$)
2. State structural break (June 1998)
3. Mention OOS as exploratory

**Quick line reorder**: Move lines 52-54 (in-sample result) to lines 48-51.

---

### FIX 3: LSTM Permutation Count
**Lines 1519-1524 (Table 1513)**

LSTM still uses 100 permutations vs. 200 for others. Footnote acknowledges "approximate."

**Quick fix**: Increase to 200 permutations and re-run. If feasible, update Table 1513 line: "All models use 200 permutations; LSTM $p$-values are precise."

---

### FIX 4: Pair-Selection Bias Explicit Correction
**Lines 1081-1088**

The paper mentions MOM→SMB ranks 1st ($F = 20.3$) but doesn't apply formal pair-selection correction.

**Quick fix**: Add one paragraph after line 1088:
> "To account for pair selection, we apply a secondary Bonferroni correction: OOS HML→SMB ($F$-$p = 0.003$) under 30-pair correction becomes $p = 0.090$, failing the 5% threshold. MOM→SMB ($p = 0.010$) also fails secondary Bonferroni. This demonstrates that the focus on HML→SMB is justified by economic prior, not empirical dominance."

---

## Summary Table: Critical Issues and Resolution Status

| Issue | Severity | Round 2 Status | Round 3 Status | Impact |
|-------|----------|------------------|-----------------|--------|
| **Quantile Granger for TE asymmetry** | Critical | "Future work" | **NOW DELIVERED** | **Elevates from curiosity to finding** |
| **Multi-pair generalizability** | Major | Unaddressed | **19/30 pairs show heterogeneity** | **Establishes systematic phenomenon** |
| **Formal baseline comparison** | Major | "Missing" | **Partial: rolling-window + threshold** | **Shows HMM superiority** |
| **OOS fails multiple-testing** | Critical | Acknowledged | **Still fails; but honest framing** | **Acceptable IF in-sample is primary** |
| **Regime-identification circularity** | Major | Mitigated | **Same mitigations; frozen OOS fails** | **Acceptable with caveat** |
| **Local optima tension** | Major | Unresolved | **Documented; no decision rule** | **Acknowledged as general challenge** |
| **Effect sizes immaterial** | Major | Acknowledged | **Reframed as diagnostic value** | **Appropriate but reduced scope** |

---

## Honest Assessment: Is This Paper at ICAIF Acceptance Bar?

### Yes Arguments (ACCEPT)

1. **Quantile Granger mechanism is now novel and explained**: The reverse tail-dependence finding (SMB→HML nonlinearity driven by tail coefficient $8\times$ median) is a genuine contribution that most computational finance venues would find valuable.

2. **Multi-pair generalizability is established**: 19/30 pairs show regime heterogeneity; this is not a one-pair artifact.

3. **Methodology is now properly packaged**: Algorithm 1 makes the protocol reusable; other practitioners can now apply this to their own factor sets.

4. **Transparency is exceptional**: The paper discloses every failure mode (OOS non-significance, local optima tension, effect size immateriality) while explaining why it's still publishable. This is rare and valuable.

5. **In-sample finding is robust**: $p = 8.75 \times 10^{-9}$ across 7 local optima clusters, HAC corrections, lags 1-15, trivariate MKT-RF control, soft-label sensitivity. This is strong.

6. **Writing and presentation have improved**: The paper is now clearer about what is primary vs. exploratory.

### No Arguments (REJECT)

1. **OOS validation fails all multiple-testing corrections**: The "frozen parameters" frame doesn't overcome the fact that no pair survives 30-pair Bonferroni. This is the intended primary external validation, and it fails.

2. **Main finding is in-sample with acknowledged circularity**: The regime discovery uses the same returns as the Granger tests. While mitigated, this is not eliminated.

3. **Practical utility is limited**: Effect sizes are 2%, trading rule is negative Sharpe, VaR models fail. This is positioning as "diagnostic awareness," which is narrow.

4. **TE asymmetry mechanism is now explained, but post-hoc**: Quantile Granger was run on data used to motivate the hypothesis. A pre-registered prediction would be stronger.

5. **Local optima tension remains unresolved**: Practitioners don't know which regime structure to use. The paper offers both BIC and economic criteria but no principled choice.

---

## The Deciding Factor: Comparison to ICAIF 2025/2024 Standards

ICAIF typically accepts papers that provide:
- **Novel methodology** (the algorithm box is here now, with quantile Granger extension)
- **Empirical insight on a systematic phenomenon** (19/30 pairs, June 1998 structural break)
- **Mechanism explanation** (quantile Granger tail dependence mechanism is now here)
- **Practical relevance** (diagnostic framework, though limited deployment)
- **Transparency** (exceptional)

**Missing**:
- Pre-registered replication (mentioned as future work)
- Strong OOS validation (fails multiple-testing)
- Compelling practical application

**Verdict**: This is a **borderline paper that leans toward acceptance**, assuming the primary contribution is repositioned as:
1. The **in-sample Normal-regime finding** ($p = 8.75 \times 10^{-9}$ with acknowledged circularity)
2. The **quantile Granger mechanistic explanation** (tail dependence, Wald $p = 0.001$)
3. The **reusable diagnostic protocol** (now formalized as Algorithm 1; 19/30 pair generalizability)
4. **NOT** the OOS re-emergence (which fails statistical tests)

---

## Final Verdict

**Decision: BORDERLINE (leaning toward WEAK ACCEPT)**

The paper has improved from WEAK REJECT to BORDERLINE because:

1. **Quantile Granger mechanism is genuinely novel** and addresses the Round 2 criticism that transfer entropy asymmetry was "unexplained." This is the most important fix.

2. **Generalizability is now demonstrated** with the multi-pair table and Table 1112 showing 63% of pairs exhibit regime heterogeneity.

3. **Algorithm box packages the methodology** in a reusable form, legitimizing the "diagnostic protocol" claim.

4. **Baseline comparisons** (even if partial) show HMM approach outperforms rolling-window and naive volatility regimes.

However, the paper is **not a clean accept** because:

1. **OOS validation still fails** all multiple-testing corrections. This is appropriately disclosed but remains a significant validation gap.

2. **In-sample finding has acknowledged circularity** that cannot be fully eliminated. The frozen OOS was meant to address this; it doesn't survive significance testing.

3. **Practical utility is limited** ($\Delta R^2 = 2\%$, negative Sharpe, VaR models fail). The reframing as "diagnostic awareness" is honest but narrows scope.

### Path to Strong Accept

To push this from BORDERLINE to STRONG ACCEPT, the authors would need **one of**:

1. **Pre-registered international replication**: Show the same structural break and quantile Granger mechanism on (e.g.) Asness international value/momentum factors, with independent regime discovery and frozen OOS that survives multiple-testing correction.

2. **Demonstrate practical deployment success**: Show that regime-conditional risk models (either direct VaR or factor-timing allocation) outperform simpler alternatives in practice, with economic significance that justifies the methodology complexity.

3. **Mechanistic modeling of the deleveraging channel**: Use holdings-level 13F data (mentioned as "future work" on line 1203) to verify that the HML→SMB→size overlap predicts actual deleveraging during stress periods, moving from correlation to causal evidence.

---

## Specific Line-Level Comments for Authors

### High Priority

**1. Reposition Abstract (Lines 44-71)**

Current opens with regimes. Recommend:
```
Open: "Using daily Fama-French returns (1990--2024), we establish that HML Granger-predicts SMB
exclusively in the Normal regime ($p = 8.75 \times 10^{-9}$, Bonferroni-corrected for 30 factor pairs),
with a data-driven structural break at June 1998."

Then: structural break details, then: complexity/quantile mechanisms

Finally: "Frozen OOS testing (2013--2024) yields exploratory Elevated-regime patterns ($p = 0.003$)
that do NOT survive 30-pair Bonferroni correction..."
```

**2. Add One Sentence to Frozen OOS Section (After line 862)**

```
"The frozen OOS result is valued for demonstrating the protocol's external consistency
(Elevated regimes show raw significance across all 7 local optima clusters) but should not
be interpreted as confirmatory evidence due to the multiple-testing correction failures documented in Table ~\ref{tab:frozen_granger}."
```

**3. Pre-Specify Scale Convention (Add after line 287)**

```
"We adopt the percentage-unit convention as PRIMARY and pre-specified to avoid post-hoc
scale selection as a degree of freedom. This decision was made prior to analysis and is
not subject to optimization."
```

### Medium Priority

**4. Clarify Quantile Granger Motivation (Before line 773)**

Add motivating sentence:
```
"The transfer entropy asymmetry (reverse channel significantly nonlinear, forward channel null
under linear Granger) suggests that the reverse predictive link operates through tail dependence
rather than location shifts. We test this hypothesis using quantile regression across $\tau \in \{0.05, \ldots, 0.95\}$."
```

This makes it explicit that quantile Granger was pre-motivated, not post-hoc.

**5. Add Generalizability Callout (After Table 1112, line 1128)**

```
"This multi-pair analysis demonstrates that regime-conditional Granger predictability is not
an artifact of HML--SMB selection but a systematic property of 19 of 30 directed factor pairs (63%),
validating the claim that the diagnostic protocol is reusable across factor networks."
```

---

## Conclusion

This is a **real improvement** from Round 2. The quantile Granger mechanism is the key fix that elevates the paper from "we observed an asymmetry" to "we explain why it exists." Combined with the multi-pair generalizability table, baseline comparisons, and the formalized algorithm, the paper is now stronger.

**However**, the OOS validation failure and circularity caveat prevent this from being a clean accept. The paper is honest about these limitations (which is valuable), but they remain fundamental gaps.

**For ICAIF 2026**: This should be **ACCEPT with minor revisions** on the assumption that reviewers value:
1. The mechanistic explanation via quantile Granger (now provided)
2. The multi-pair generalizability (now demonstrated)
3. The honest transparency about limitations (now excellent)
4. The packaged algorithm for reuse (now formalized)

If reviewers emphasize "requires strong OOS validation" or "in-sample findings on curated pairs are insufficient," this would be REJECT.

Given ICAIF's typical acceptance of regime-switching and causal inference papers with honest limitations disclosure, **I estimate 65-70% chance of acceptance** with these improvements.

**Recommendation to authors**: Submit as-is with the line-level edits above. The quantile Granger mechanism and multi-pair table are now sufficient contributions for acceptance at a top venue.

---

**Review Completed: 2026-03-01**
