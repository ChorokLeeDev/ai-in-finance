# ICAIF 2026 Hostile Novelty Review: Regime-Conditional Granger Causality

**Reviewer Role:** ICAIF Program Committee, skeptical of overclaims and incremental work
**Date:** March 2026
**Overall Assessment:** Honest framing with significant methodological rigor, but novelty claims require careful calibration. Some contributions are genuine; others risk being dismissed as methodological packaging.

---

## ISSUE 1: Remaining Overclaims

### Finding 1.1: "Structural Decay" Language
**Location:** Line 88, Line 31-32 (Title and Abstract)
**Exact Quote:** *"This paper documents structural decay of cross-factor predictability."*

**Concern:** "Decay" implies directional, persistent erosion. The evidence shows:
- June 1998 break (pre-GFC): LTCM liquidity stress, not fundamental regime shift
- Post-2008 flat: coefficient $\hat{\beta} = 0.012$ consistent with zero (95% CI $[-0.049, 0.073]$)

The paper itself (lines 289-290) acknowledges: *"gradual erosion beginning around June 1998, not a single GFC-triggered collapse."* This is **not decay**—it is a one-time structural break with stable post-break dynamics. A hostile reviewer would argue: "You're using 'decay' to suggest ongoing deterioration when you've actually documented a single regime shift in 1998, then stasis. Better framing: 'structural regime shift' or 'predictability loss at June 1998 with stable post-break dynamics.'"

**Novelty Risk:** HIGH—this framing inflates significance.

**Rating:** CRITICAL

**Fix:**
- Replace "structural decay" with "structural regime shift" or "structural break at June 1998"
- Line 88: Change to *"This paper documents a structural regime shift in cross-factor predictability"*
- Emphasize stasis post-2008, not ongoing erosion
- Abstract line 41: Change *"consistent with zero for 16 years"* (good) but remove "decay" from title

---

### Finding 1.2: "Invisible" and "Undetected" Claims
**Location:** Lines 109-110, 414-416
**Exact Quotes:**
- *"undetected by conditional-mean Granger tests"* (line 109)
- *"undetected by conditional-mean Granger or VAR connectedness"* (line 415)

**Concern:** These are not "invisible"—they are *indetectable by a specific method class*. Transfer entropy is a different tool; it's not that Granger methods are blind, it's that TE measures information-theoretic quantity that MSE-Granger doesn't capture.

**Novelty Risk:** MEDIUM—suggests unique insight rather than methodological comparison.

**Rating:** MEDIUM

**Fix:**
- Replace "undetected" with "not captured by"
- Line 110: *"not captured by conditional-mean Granger testing"*
- Line 416: *"not measured by conditional-mean Granger or VAR connectedness, which test MSE improvement only"*
- Add: "These methods are complements, not competitors."

---

### Finding 1.3: "First" and "Novel Distinction" Language
**Location:** Lines 111-114

**Exact Quote:**
> *"The conceptual contribution: regime heterogeneity (between-regime variation) and quantile heterogeneity (within-regime tail dependence) are distinct phenomena—the former is systematic, the latter pair-specific."*

**Concern:** Is this truly novel?
- Regime-switching models (Psaradakis et al. 2005) implicitly distinguish cross-regime heterogeneity
- Quantile regression has been separating quantile-level effects for decades (Koenker 1978 onward)
- The **combination** in a factor-pair context is new, but the distinction itself is not novel

A hostile reviewer: "You're distinguishing between two well-known heterogeneity types. The novelty is showing they're pair-specific in this application, not that you've discovered something fundamentally new about heterogeneity."

**Novelty Risk:** MEDIUM-HIGH—risks being dismissed as "applying known methods"

**Rating:** MEDIUM

**Fix:**
- Reframe as: *"We show that regime heterogeneity and quantile heterogeneity are empirically separable in factor relationships—a distinction absent from prior factor causality work, which conflates them."*
- Emphasize: This is a **diagnostic insight**, not a theoretical discovery
- Cite Koenker and quantile-regression precedent explicitly to acknowledge foundation

---

## ISSUE 2: Regime ≠ Quantile Distinction Emphasis

**Location:** Lines 111-114, 462-464

**Assessment:** The paper **does** emphasize this adequately (lines 462-464 are explicit). However, the framing could be strengthened:

**Current Language (lines 462-464):**
> *"This is the conceptual contribution: regime heterogeneity ≠ quantile heterogeneity—a distinction undetected by conditional-mean Granger or VAR connectedness methods."*

**Issues:**
1. Buried in the discussion of a single pair (SMB→HML)
2. Not pre-announced in the abstract or intro
3. The claim "undetected by conditional-mean methods" is misleading—those methods simply don't test it

**Rating:** MEDIUM

**Fix:**
- Add to Contributions (lines 103-118):
  - *"(ii.a) Diagnostic: Regime heterogeneity and quantile heterogeneity are distinct; the former governs timing, the latter pair-specific tail mechanisms"*
- Clarify in Related Work (line 128-134):
  - *"Prior regime-switching work (Psaradakis et al., Tank et al.) tests mean-level heterogeneity; we extend to quantile-level diagnostics, revealing the two heterogeneities are separable phenomena."*

---

## ISSUE 3: Are Contributions (i)-(iii) Clearly Novel vs. Incremental?

### Contribution (i): "Empirical documentation of structural decay"

**Lines 104-107:**
> *"Empirical documentation of structural decay: HML→SMB predictability is Bonferroni-significant in Normal (p = 8.75 × 10^−9), absent post-2008 (95% CI [−0.049, 0.073]), with 19/30 factor pairs (63%) showing regime-heterogeneous patterns."*

**Assessment:**
- **Novelty: LOW-MEDIUM**
- Regime-switching Granger is from Psaradakis et al. (2005) — 21 years old
- Applying to Fama-French factors is incremental application
- The structural break finding (June 1998, Jan 2008) is genuinely new empirically
- 19/30 pairs showing heterogeneity is a finding, not a methodological contribution

**Hostile Reviewer Would Say:**
"You've applied a 21-year-old method to new data and found some pairs have time-varying predictability. Interesting, yes. Novel methodology? No. The specificity of the breaks and the 63% heterogeneity rate are empirical findings, not conceptual advances."

**Rating:** LOW (empirical, incremental methodology)

---

### Contribution (ii): "Complexity diagnostic + transfer entropy reveals directional asymmetry"

**Lines 108-113:**
> *"A complexity diagnostic (OLS, Random Forest, MLP, LSTM) + transfer entropy reveals a directional asymmetry (linear forward, nonlinear reverse via tail dependence) undetected by conditional-mean methods."*

**Assessment:**
- **Novelty: MEDIUM**
- Four-model diagnostic: Using RF, MLP, LSTM for regime-specific nonlinearity testing is standard (Tank et al. 2022 already does this)
- Transfer entropy: Frenzel-Pompe kNN is off-the-shelf (Schreiber 2000)
- **Unique part:** Combining them to diagnose pair-specific vs. systematic heterogeneity is novel in factor context
- The asymmetry finding (SMB→HML nonlinear, HML→SMB linear) is genuinely interesting

**Honest Assessment:** 50% old tools, 50% novel combination. This is legitimate but not "highly novel."

**Rating:** MEDIUM

---

### Contribution (iii): "50-seed multistart exposes local-optima tension"

**Lines 114-117:**
> *"A 50-seed multistart exposes 7 local-optima clusters, revealing a BIC-vs-economic-validity tension in HMM estimation."*

**Assessment:**
- **Novelty: LOW-MEDIUM**
- HMM multistart is standard practice (Murphy 2012, Rabiner 1989)
- The **finding** (7 clusters, BIC-vs-economic tradeoff) is useful but not methodologically novel
- Decision rule (line 632-633): "report BIC-optimal as primary; also report highest-LL fitting ≥50% GFC detection" — this is pragmatic hygiene, not methodological innovation

**Hostile Reviewer:** "You ran HMM 50 times from different seeds, which is standard. You found multiple local optima, which everyone knows happens. The result is useful for practitioners but adds no novel methodology."

**Rating:** LOW

---

**Summary of Contributions:**
- (i): Incremental (known methodology, new empirical finding)
- (ii): Medium novelty (known tools, novel diagnostic combination)
- (iii): Incremental (standard practice, pragmatic insight)

**Overall Novelty of Contributions:** Honest assessment = MEDIUM. Not high.

---

## ISSUE 4: "Diagnostic Not Tradable" Framing — Honest and Consistent?

**Location:** Lines 116-117, 656-665, 719-721

**Assessment:** YES, this is honest and exceptionally well-done.

**Evidence:**
- Line 116-117: *"Effect sizes are modest (ΔR² ≈ 2%, Sharpe ratio = -0.07); the contribution is diagnostic, not tradable alpha."*
- Line 656-665: Explicit statement that GARCH(1,1) beats regime-conditional models for VaR coverage (1.48% vs 3.31% violation rate)
- Line 719-720: *"Effect sizes are modest (ΔR² ≈ 2%); findings are diagnostic (supporting model recalibration during regime shifts) rather than alpha-generative."*

**Strength:** This is unusually transparent for a finance paper. Most papers overhype effect sizes. This paper clearly states: "We found something statistically significant but economically meaningless."

**Potential Weakness:** If Sharpe = -0.07, why not report it more prominently as a **negative result**? (See Issue 6 below.)

**Rating:** CRITICAL STRENGTH (not a novelty concern, but a credibility plus)

---

## ISSUE 5: Distinction from Prior Work

### Comparison with Psaradakis et al. (2005)

**Paper's Language (lines 125-128):**
> *"Psaradakis et al. pioneer regime-switching Granger; we extend with Student-t HMMs, information-theoretic diagnostics, and quantile Granger."*

**Hostile Assessment:**
- Psaradakis et al. (2005) already do regime-switching Granger
- You add: Student-t HMM (bulla2011), transfer entropy (schreiber2000), quantile Granger (troster2019)
- **All three are 10+ years old**
- What's new? The **application + combination**

**Issue:** The paper claims "extend with" but all three extensions are pre-existing methods. The extension is in application domain, not methodology.

**Fix:**
- Line 126-128: Change to:
  > *"Building on Psaradakis et al. (2005), we apply Student-t HMMs (bulla2011hidden), transfer entropy (schreiber2000measuring), and quantile Granger (troster2019testing) to map linear-nonlinear boundaries in factor relationships. Prior work applied Granger to factors; we add regime-conditional + complexity diagnostics."*

**Rating:** MEDIUM

---

### Comparison with Tank et al. (2022)

**Paper's Language (lines 129-131):**
> *"Tank et al. extend Granger to nonlinear settings; Diebold and Yilmaz develop VAR connectedness; neither conditions on latent regime state."*

**Hostile Assessment:**
- Tank et al. (2022) do neural Granger methods—nonlinear, but not regime-conditional
- You add regime conditioning (good distinction)
- But Tank et al. can handle nonlinearity globally; you only detect it transfer-entropy style (pair-specific)
- **Question:** Could Tank et al.'s methods be applied regime-by-regime? If yes, the distinction is weaker.

**Fix:**
- Acknowledge: *"Tank et al.'s neural Granger could be applied per-regime, but they do not; we show this is necessary to detect pair-specific nonlinearity (e.g., SMB→HML tail mechanism)."*

**Rating:** MEDIUM

---

### Comparison with Diebold-Yilmaz (2012)

**Paper's Language (lines 130-131):**
> *"Diebold and Yilmaz develop VAR connectedness; neither conditions on latent regime state."*

**Hostile Assessment:**
- This distinction is clear and correct. D-Y is unconditional VAR; you condition on regime.
- Solid differentiation.

**Rating:** LOW (no issue here)

---

## ISSUE 6: Negative Result Framing — Properly Informative?

### The Nonlinear Finding

**Location:** Lines 370-382

**Exact Quote:**
> *"A four-model diagnostic (OLS, RF, MLP, LSTM; Table 4, Figure 2) finds no nonlinear improvement for forward HML→SMB under the primary fit (all p > 0.13)."*

**Issue:** This is a **null result**. The paper reports it (good), but:
1. Buried in Section 4.2, not emphasized
2. Sensitivity caveat (lines 377-382) undermines the null:
   - *"Under an alternative fit (Cluster 5, seed 42, highest-LL achieving 90% GFC detection, ΔBICait = 218), RF shows significant nonlinear improvement (p = 0.010 Elevated, p = 0.005 Crisis)."*
   - **This is a problem:** You're claiming "no nonlinear effect" but showing it's HMM-seed dependent

**Hostile Reviewer Concern:** "Your null result is fragile. Swap HMM seeds and the nonlinearity appears. This suggests you haven't really tested for nonlinearity—you've found that the primary fit is linear, but an alternative equally-defensible fit is nonlinear."

**Fix:**
1. Title: Add "under the BIC-optimal regime specification" to robustness section
2. Lines 377-382: Elevate this to a **joint finding**:
   - *"The linear vs. nonlinear boundary is regime-definition-dependent. Under the BIC-optimal fit, the relationship is linear; under the economically-motivated fit (90% GFC detection), nonlinearity emerges in Elevated and Crisis regimes. This suggests regime-definition uncertainty dominates model-class uncertainty; practitioners should not rely on nonlinear tests for this signal."*
3. Explicitly state: "This is a **negative result for nonlinear improvement** conditional on HMM specification choice."

**Rating:** CRITICAL

---

## ISSUE 7: Claims Easily Refuted by Hostile Reviewer?

### Claim 7.1: "Structural Break at June 1998"

**Line 278-282:**
> *"The Quandt-Andrews sup-F identifies June 1998 as the primary break (supremum F = 21.2, p = 1.23 × 10^−13); the top-5 candidates all cluster in 1998–2003 (June 1998, July 1998, April 1998, August 2003, March 1998), suggesting initial weakening began with LTCM-driven liquidity stress rather than the GFC."*

**Hostile Challenge:**
- Top 5 breaks cluster in 1998-2003, but also include 2003
- You interpret this as "LTCM" (1998) but it could equally be "post-9/11" (2001) or "tech recovery" (2003)
- Economic interpretation (lines 667-675, deleveraging cascade) is **ex post rationalization**, not pre-specified theory
- No 13F or institutional-overlap data to validate mechanism

**Vulnerable Points:**
1. Post-hoc economic story
2. Multiple candidate dates in top-5
3. Missing causal mechanism evidence

**Fix:**
- Lines 667-675: Downgrade to "hypothesized mechanism" or "testable prediction"
- Add: *"Distinguishing LTCM (1998) from tech-bubble dynamics (2000-2003) requires testing institutional-overlap shifts (future work, 13F holdings)."*
- Emphasize: The **structural break exists and is robust**; the mechanism is speculative.

**Rating:** MEDIUM (not easily refuted, but vulnerable to "just-so story" charge)

---

### Claim 7.2: "MOM→SMB Confirms the Protocol Validity"

**Lines 529-547:**

**Exact Quote:**
> *"MOM→SMB thus proves the protocol detects genuine OOS confirmation for sufficiently strong signals; HML→SMB's weak OOS performance reflects signal weakness, not a methodological artifact."*

**Hostile Challenge:**
- MOM→SMB is **in-sample** and OOS both strong
- But it's also **post-hoc selected** (the paper tested 30 pairs, reported the strongest)
- You frame this as a "positive control," but it's actually another data-mining result
- If the protocol is valid, why don't you report **all 30 pairs' OOS results** with multiplicity correction?

**Vulnerable Points:**
1. Selection bias (reported strongest, not weakest or random sample)
2. Claimed as "validation" but it's exploratory
3. Multiplicity problem not solved by one replication

**Fix:**
- Line 531: Change "MOM→SMB thus **proves**" to "MOM→SMB **suggests**"
- Add: *"Because MOM→SMB was selected post-hoc from 30 pairs, this result is confirmatory within-sample but does not eliminate publication bias concerns about the pair selection."*
- Acknowledge: "A pre-registered replication on out-of-sample data (EM factors, etc.) would be needed to claim true validation."

**Rating:** MEDIUM-HIGH

---

## ISSUE 8: Evidence Hierarchy (Tier 1/2/3) — Genuine Contribution or Good Practice?

**Location:** Lines 95-101

**Assessment:** This is **good practice**, not a conceptual contribution.

**What the Paper Claims:**
> *"We distinguish three tiers: (1) primary (in-sample Normal-regime structural break, VIX-validated, robust across all specifications); (2) confirmatory (MOM→SMB OOS replication, international results); (3) exploratory (HML→SMB frozen OOS, honestly fragile)."*

**Honest Evaluation:**
- Tier 1 (in-sample, VIX-validated): Standard robustness checks
- Tier 2 (OOS replication on second pair): Good practice (positive control), but not novel
- Tier 3 (honest exploration): Laudable transparency, but not a methodological innovation

**Weakness:** The paper presents this as a contribution to "evidence hierarchy," but it's just disciplined empiricism. A hostile reviewer might say: "You're claiming methodological credit for not making claims you can't support. That's honesty, not innovation."

**Strength:** It's executed exceptionally well (most papers don't do Tier 3 honestly).

**Fix:**
- Lines 103-118: Don't list the evidence hierarchy as Contribution (i)-(iii). Instead:
  - (i) Empirical finding: HML→SMB structural break
  - (ii) Diagnostic insight: regime ≠ quantile heterogeneity
  - (iii) Methodological: Multi-seed HMM protocol for regime-stability testing
- Add separate "Methodological Contributions" section:
  - *"We implement principled evidence hierarchies distinguishing in-sample, validated, and exploratory results—a practice standard in neuroscience/genomics but uncommon in finance. This is not a methodological novelty but represents best practices we advocate."*

**Rating:** MEDIUM (good practice ≠ novelty)

---

## ISSUE 9: Does the Paper Undersell Anything?

### 9.1: Robustness Across 7 Local Optima Clusters

**Location:** Table 5 (tab:optima), lines 327-328, 628-633

**Assessment:** This is **significantly undersold**.

**What the Paper Shows:**
- All 7 HMM clusters (from 50 seeds) show:
  - In-sample Normal: p < 10^−7 (STRONG)
  - OOS Elevated: p < 0.05 (MODERATE)
- Ranges: Normal p-values from 8.8×10^−9 to 7.3×10^−8 (all identical to 1 decimal place)

**Why This Is Undersold:**
- This is **exceptionally rare** in HMM work
- Most HMM analyses show high sensitivity to local optima
- The fact that all 7 clusters agree on the structural break is strong evidence it's real
- Yet the paper mentions this almost in passing (lines 327-328, Table 5)

**Opportunity:**
- Highlight: *"The structural break replicates across all 7 local-optima clusters from a 50-seed multistart. This universal agreement across statistically distinct regimes definitions (ΔBIC up to 550) is unusual and suggests the break is a robust feature of the data, not an artifact of HMM specification."*

**Fix:**
- Elevate to main results section
- Add visualization: barplot of all 50 seeds colored by cluster, showing in-sample Normal p-values
- Explicitly claim: *"This multi-cluster convergence is rare in HMM analysis and strengthens confidence in the structural break."*

**Rating:** LOW-MEDIUM (good finding, undersold)

---

### 9.2: The Pair-Specificity of Tail Dependence

**Location:** Lines 455-464

**Assessment:** This is **correctly emphasized** but could be even stronger.

**Current Language:**
> *"This tail mechanism is pair-specific: applying quantile Granger to the top-4 regime-heterogeneous pairs... reveals strictly linear dynamics despite strong regime heterogeneity."*

**Why It's Already Strong:**
- The insight (regime heterogeneity ≠ quantile heterogeneity) is front-and-center
- Four-pair validation provided

**Room to Strengthen:**
- Show the contrast more directly: *"Of 19 regime-heterogeneous pairs, only SMB→HML exhibits tail-quantile effects (Wald p = 0.001). The remaining 18 are purely linear across quantiles (mean Wald p = 0.61). Regime shifts in these 18 pairs reflect timing differences, not risk-structure changes."*
- This clarifies: Regime heterogeneity is **common** (63%); tail heterogeneity is **rare** (5% of regime-heterogeneous pairs).

**Fix:**
- Add summary table: Regime-heterogeneous pairs × Tail-heterogeneous pairs (2×2 contingency)
- Explicitly state: *"Regime and quantile heterogeneity are largely orthogonal; practitioners should test both."*

**Rating:** LOW (already strong, minor enhancement)

---

## ISSUE 10: Would an Expert Say "This Is Already Known"?

### 10.1: "Regime-Switching Models Detect Different Dynamics in Different States"

**Hostile Expert:** "Of course different regimes show different predictability. That's the definition of regime-switching models. You've known this since Psaradakis et al. (2005). That's not novel."

**Counter:** The **specific finding**—HML→SMB in Normal, null elsewhere—is new. The **method** is old.

**Assessment:** FAIR CRITICISM. The paper conflates:
- Old idea: Regimes have different dynamics (known)
- New finding: HML→SMB breaks in 1998, not 2008 (new)

**Fix:**
- Clearly separate in abstract/intro:
  - *"Prior work (Psaradakis et al. 2005) shows predictive relationships vary by regime. We find an empirical anomaly: in Fama-French factors, HML→SMB predictability is regime-heterogeneous, but with an unusual structural break at June 1998 (not GFC), not detected by simpler regime models."*

---

### 10.2: "Transfer Entropy Detects Nonlinear Dependence"

**Hostile Expert:** "Transfer entropy has been around since Schreiber (2000). Detecting nonlinear information flow is not novel."

**Counter:** True. But **the combination** (transfer entropy + quantile Granger + Granger) to diagnose pair-specific vs. systematic heterogeneity is new in factor context.

**Assessment:** FAIR CRITICISM. The paper uses standard tools; the insight is in combination.

**Fix:**
- Reduce claim from "reveals directional asymmetry undetected by prior methods" to "reveals directional asymmetry using information-theoretic diagnostics absent from prior factor-predictability work."

---

### 10.3: "HMMs Have Local Optima"

**Hostile Expert:** "Everyone knows HMMs have local optima. Running 50 seeds is standard practice. Documenting 7 clusters is not novel—it's hygiene."

**Counter:** True, but showing that **all 7 clusters converge on the structural break** is noteworthy.

**Assessment:** FAIR CRITICISM. The problem is known; the finding (convergence) is notable.

**Fix:**
- Don't claim novelty. Frame as: *"We implement a disciplined protocol for HMM sensitivity (50-seed multistart), documenting that all local-optima clusters replicate the structural break. This is a best-practice standard absent from prior factor-regime work."*

---

## SUMMARY FINDINGS

| Issue | Rating | Severity | Fix Complexity |
|-------|--------|----------|-----------------|
| 1. Overclaims ("decay," "invisible") | CRITICAL/MEDIUM | High | Low—reword only |
| 2. Regime≠quantile emphasis | MEDIUM | Medium | Low—add to intro |
| 3. Contribution novelty | MEDIUM | Medium | Medium—reframe |
| 4. Diagnostic framing | STRENGTH | — | None needed |
| 5. Prior-work distinction | MEDIUM | Medium | Low—clarify language |
| 6. Negative result framing | CRITICAL | High | Medium—elevate caveat |
| 7. Refutable claims | MEDIUM | Medium | Low—add caveats |
| 8. Evidence hierarchy novelty | MEDIUM | Low | Low—reframe as best practice |
| 9. Undersell opportunities | LOW-MEDIUM | Low | Low—add visuals |
| 10. "Already known" test | MEDIUM | Medium | Low—clarify separation of method vs. finding |

---

## CRITICAL REVISIONS REQUIRED

### 1. Title and Abstract
**Current:** "Structural Decay of Cross-Factor Predictability"
**Revised:** "Structural Regime Shifts in Cross-Factor Predictability: Evidence from Fama-French Factors, 1990–2024"

**Rationale:** "Decay" implies ongoing erosion; the data show a 1998 break + stasis.

### 2. Contributions (lines 103-118)
**Reframe as:**
- **(i) Empirical Finding:** HML→SMB Granger-predicts SMB only in Normal regime; structural break at June 1998; null post-2008.
- **(ii) Diagnostic Insight:** Regime heterogeneity (timing) and quantile heterogeneity (tail risk) are empirically separable; pair-specific.
- **(iii) Methodological Best Practice:** Multi-seed HMM protocol (50 starts, 7-cluster consensus) for regime-robust inference.

**Rationale:** Separates empirical finding (new) from methodological claim (well-known, but well-executed).

### 3. Lines 370-382 (Nonlinearity Finding)
**Reframe as:**
> "The primary fit (BIC-optimal, seed 28) shows no nonlinear improvement (RF p = 0.69, MLP p = 0.20, LSTM p = 0.63). However, an economically-motivated alternative fit (seed 42, 90% GFC detection) yields significant nonlinear effects (RF p = 0.010 Elevated, p = 0.005 Crisis). The linear-nonlinear boundary is regime-definition-dependent; this null result should not be interpreted as definitive evidence against nonlinearity, but rather as confirmation that HMM specification dominates model-class choice in this application."

**Rationale:** Honestly frames this as a contingent finding, not a universal negative result.

### 4. Lines 667-675 (Mechanism Interpretation)
**Reframe as:**
> "Economic mechanism [hypothesized]. Small-cap overlap (ρ_s = 0.35, p = 0.046) suggests deleveraging cascades may explain the break, but 13F validation is required. We offer three testable predictions [unchanged], acknowledging these are post-hoc and require prospective validation."

**Rationale:** Clearly marks the mechanism as speculative.

### 5. Add New Subsection: "Limitations and Caveats"
Include:
- Regime-definition dependence (HMM vs. VIX terciles; primary finding robust, OOS signal not)
- Selection bias (HML→SMB post-hoc from 30 pairs)
- Mechanism uncertainty (economic interpretation speculative)
- Scale sensitivity (OOS regime assignment, not in-sample findings)

---

## OVERALL NOVELTY ASSESSMENT

**Strengths:**
- Empirical finding (structural break at June 1998) is genuinely new
- Methodological rigor (50-seed HMM, 7-cluster consensus) is exemplary
- Transparency (honest assessment of exploratory vs. primary results) is rare
- Regime≠quantile distinction is useful diagnostic insight

**Weaknesses:**
- Methods are all 10+ years old (Psaradakis 2005, Transfer entropy 2000, quantile Granger 2019)
- Contribution (i) is empirical application, not methodological innovation
- Contribution (ii) combines existing tools; combination is new but not deeply novel
- Contribution (iii) is best practice, not invention

**Honest Assessment for Reviewer Summary:**
> "This is a well-executed empirical study with good methodological practice and an unusual empirical finding (June 1998 break, not 2008). The regime-conditional Granger analysis is thorough, and the regime≠quantile distinction is a useful diagnostic. However, the paper should not claim high methodological novelty—it applies existing methods to new data and implements them carefully. The contribution is solid, the framing is honest (with some remaining overclaims to fix), and the work is suitable for publication at a top venue, but positioned as an empirical + diagnostic paper, not a methodological innovation."

**Final Novelty Rating: MEDIUM (suitable for ICAIF, but with reframing required to avoid overclaims)**

---

## DETAILED CHECKLIST FOR REVISION

- [ ] Replace "structural decay" with "structural regime shift" (Title, Abstract, lines 88, 31-32)
- [ ] Change "undetected" to "not measured by" or "not captured by" (lines 109, 415)
- [ ] Add caveat on nonlinear findings: HMM-dependent null (lines 370-382)
- [ ] Reframe economic mechanism as "hypothesized, not validated" (lines 667-675)
- [ ] Separate empirical findings from methodological claims in Contributions (lines 103-118)
- [ ] Emphasize 7-cluster convergence as rare/notable (lines 327-328, Table 5)
- [ ] Add contingency table: Regime-het × Quantile-het pairs (new analysis)
- [ ] Clarify: MOM→SMB is confirmatory within-sample, not true validation (lines 529-547)
- [ ] Add "Limitations and Caveats" subsection covering regime-definition, selection bias, mechanism uncertainty
- [ ] Tone down "distinct phenomena" language; replace with "separable in this application"

