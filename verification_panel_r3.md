# ADVERSARIAL PANEL REVIEW: NARRATIVE SHARPNESS & CONTRIBUTION CLARITY
**Paper:** "The Rise and Fall of Cross-Factor Predictability: Regime-Dependent HML→SMB Granger Causality"
**Date:** February 25, 2026
**Review Conducted:** Post-narrative-surgery round (Round 3)

---

## EXECUTIVE SUMMARY

After narrative surgery, the paper has dramatically improved in **directness** but still struggles with **scope ambiguity**. The structural break finding is now prominently featured (p=8.75e-9 by line 91), but reviewers differ sharply on whether the paper knows what it is: a **structural-break discovery paper** (R1, R4) or a **regime-conditional methodology paper with modest OOS confirmation** (R2, R3). The frozen OOS result does not credibly resolve this ambiguity—it is presented honestly as fragile but may overstate its evidential weight.

---

## REVIEWER 1: NARRATIVE ASSASSIN
**Mandate:** Does the paper tell ONE clear story? Is the hero visible by page 1? Any remaining narrative dilution?

### SCORES
| Dimension | Score | Comment |
|-----------|-------|---------|
| **Narrative clarity** | 7.5/10 | Hero revealed by line 91, but dual-act structure muddles resolution. |
| **Organization** | 7/10 | Introduction now leads with structural break; Discussion clarifies scope. |
| **Technical rigor** | 8/10 | Methodology sound; local optima handling transparent. |
| **Honesty/disclosure** | 8/10 | Frozen OOS fragility disclosed; some positive framing in abstract. |
| **Novelty** | 6.5/10 | Structural break dated to Jan 2008 is novel; regime methodology less so. |
| **Impact** | 5/10 | Academic (not trading). Limited to cross-factor domain. |

### NARRATIVE DIAGNOSIS

**What works:** The introduction now tells a single, compelling story:
- **Act I (p. 81--98):** August 2007 quantitative meltdown → HML→SMB Granger causality is extraordinarily significant pre-GFC (p=8.75e-9), but breaks sharply at January 2008 (Chow p=2.29e-6), then vanishes for 16 years.
- This is **clear, time-indexed, and falsifiable.**

**What dilutes the story:**
1. **Dual resolution problem** (Section 5, Discussion). The paper claims both:
   - "The structural break IS the contribution" (line 676)
   - "OOS evidence is modest" (line 683)

   These are not equivalent findings. The structural break is **definitive and time-indexed**. The OOS Elevated result is **exploratory, regime-dependent, and sensitive to K.** Readers finish confused about which is the main finding.

2. **Frozen OOS framing** (lines 95--98):
   ```
   "A frozen out-of-sample design detects re-emergence in intermediate-volatility
   regimes post-GFC, though with a smaller effect (permutation p ∈ [0.063, 0.022])
   that is modest and sensitive to model specification."
   ```
   This is positioned as a **co-finding**, but it:
   - Does not survive Bonferroni correction
   - Requires K=3 (null at K=2)
   - Ranked 2nd of 30 pairs (not 1st)
   - Permutation p straddles significance threshold

   **Should this be in the abstract?** Yes, to show how structure re-emerges. **Should it be weighed equally with the structural break?** No.

3. **Contributions list ambiguity** (lines 121--140):
   - Contribution 1: "Regime-dependent predictive structure with a dated structural break" ✓ This is the hero.
   - Contribution 2: "Modest OOS evidence with disclosed fragility" — This is presented as equal weight but is **not the main story.**

   **Fix:** Reorder: make Contribution 1 "Structural break in HML→SMB predictability" (dating to Jan 2008, Bonferroni-significant, survived 16 years post-absence). Demote Contribution 2 to "Secondary evidence of regime-conditional re-emergence post-GFC via frozen OOS design."

4. **Narrative creep in Related Work** (lines 142--177). The paper contextualizes regime-switching models and Granger causality in detail, but **who cares about the methodology if the structural break is the hero?** The machinery should be in Methods, not in Related Work where it competes for narrative space.

### REMAINING ISSUES BLOCKING "STRONG ACCEPT"

1. **Lack of single-sentence clarity on contribution.** After reading lines 88--99, can a reviewer write one sentence describing this paper?
   - ✓ Best attempt: "HML→SMB Granger causality was Bonferroni-significant pre-2008, broke sharply in January 2008, and has remained absent for 16 years, with modest evidence of regime-dependent re-emergence in 2013--2024."
   - This works, but the paper does not state it as prominently as it should.

2. **Scope creep into risk monitoring** (Appendix A, Algorithm 1). The paper demonstrates that the structural break exists and OOS signal is modest. Why spend 50+ lines on VaR applications? This is **scope expansion**, not support for the main finding. The Sharpe=-0.07 trading result (line 880) already proves no economic value. Further applications feel like defensive padding.

3. **Dual-mechanism problem** (Appendix, lines 745--750):
   - Normal regime: "inventory management" story
   - Elevated/Crisis: "deleveraging cascade" story
   - Paper acknowledges this is inconsistent but does not resolve it.

   **Effect on narrative:** Readers finish unsure whether there is one phenomenon or two. The structural break (hero) is time-indexed, not mechanism-indexed, so this is acceptable—but the Discussion should explicitly state: "We document a **phenomenon** (structural break), not a **mechanism.** The dual-story ambiguity reflects genuine heterogeneity across regimes."

---

## REVIEWER 2: ORGANIZATION SURGEON
**Mandate:** Page efficiency? Methodology still too long? Appendix balance?

### SCORES
| Dimension | Score | Comment |
|-----------|-------|---------|
| **Narrative clarity** | 7/10 | Clearer than draft; still some organization bloat. |
| **Organization** | 6.5/10 | Main paper tight; appendices bloated (1,174 lines total). |
| **Technical rigor** | 8.5/10 | Methods are sound and transparent. |
| **Honesty/disclosure** | 8.5/10 | Local optima, regime selection, exclusion rates all disclosed. |
| **Novelty** | 6.5/10 | Structural break is novel; regime switching is standard. |
| **Impact** | 5/10 | Limited applicability; academic only. |

### PAGE EFFICIENCY AUDIT

**Main paper structure (lines 1--727):**
- Introduction: ~50 lines (good, now leads with hero)
- Related Work: 35 lines (acceptable but could be trimmed by 50%)
- Methodology: ~100 lines (acceptable for clarity)
- Results: 150 lines (appropriate for core findings)
- Discussion: 60 lines (good, includes scope statement)
- Conclusion: ~25 lines (tight)
- **Total: ~420 lines of prose + figures + tables = ~18 pages**

**Appendices (lines 729--1,274):**
- A: Economic mechanism (30 lines)
- B: FF25 overlap evidence (50 lines)
- C: Event-based checks (25 lines)
- D: MS-VAR comparison (20 lines)
- E: Trading strategy (15 lines)
- F: Risk monitoring + VaR (200 lines, with 2 algorithms & 4 tables)
- G: Extended robustness (150 lines, 2 tables, 4 figures)
- H: Local optima taxonomy (40 lines)
- **Total: ~530 lines of appendix = ~22 pages**

**Verdict:** **Appendices exceed main paper in length (22 vs. 18 pages).** This is not inherently wrong, but review against contribution:
- **Appendix A-B-C:** Necessary (mechanism hypothesis, overlap test, event validation) → Keep
- **Appendix D:** MS-VAR comparison adds 20 lines to show HML info is statistically significant but economically small. Verdict: **TRIM.** The paper already knows effect sizes are small (ΔR² ~2%). Readers understand trade-off between fit and parsimony. Move to 2-3 sentences in Methods.
- **Appendix E:** Trading strategy Sharpe=-0.07 proves no practical value. Paper already makes this point at line 698 (5 lines). Appendix expansion to 15 lines is **redundant**. TRIM to 5 lines: "A simple lag-1 strategy yields Sharpe=-0.07 vs. buy-hold SMB Sharpe=+0.06, confirming statistical significance ≠ economic significance."
- **Appendix F:** VaR application is **scope creep.** The structural break finding does not require VaR validation. Algorithm 1 is novel, but in what paper? If the core contribution is the structural break, why are practitioners reading VaR backtests? Either (a) reframe the entire paper as "risk monitoring application" (but that's not the hero), or (b) move VaR to a brief note ("Future work: applying Granger-informed regime signals to tail forecasting yields modest Christoffersen improvement"). **TRIM from 200 to 20 lines.**
- **Appendix G:** Extended robustness is appropriate and well-executed. Keep, but trim MS-VAR from this section (move to Methods trim).
- **Appendix H:** Local optima transparency is essential given HMM sensitivity. Keep.

### REMAINING ISSUES BLOCKING "STRONG ACCEPT"

1. **Appendix tail-wagging the main-paper dog.** The frozen OOS result (Section 3.2, 50 lines in main paper) is modest and does not survive Bonferroni correction. Yet it spawns:
   - Appendix B: FF25 overlap analysis (50 lines, using sensitivity fit seed~42, not primary seed~28)
   - Appendix C: Event-level validation (25 lines, 2/6 events match pattern, binomial p≈0.11)
   - Appendix F: VaR application (200 lines, unclear connection to frozen OOS fragility)

   **Verdict:** These are secondary. If the frozen OOS Elevated result is exploratory (not confirmatory), supporting it with 275 lines of appendix looks like **over-engineering to rescue a weak secondary finding.**

2. **Scale convention inconsistency** (lines 187--196, repeated at lines 556--558):
   - In-sample: percentage units (e.g., 0.10 = 0.1%)
   - Frozen OOS: decimal units first, then converted back to percentage for permutation test
   - **Why?** The paper states this is scale-invariant for Granger F-statistics, but permutation p-values differ: 0.063 (decimal, n=836) vs. 0.022 (percentage, n=953).

   **Verdict:** This inconsistency is disclosed but undermines confidence. Readers wonder: which n is correct? Why are regime sizes different (836 vs. 953)? The paper says "both conventions yield F-p < 0.05," but they actually yield **p=0.014 vs. p=0.003**, a meaningful difference when claiming OOS significance. **Recommendation:** Pick one convention and stick to it. The percentage-unit convention (n=953, p=0.022) is more conservative; use it consistently and drop the decimal-unit result entirely.

3. **Frozen OOS design does not eliminate in-sample bias.** Lines 241--254 argue:
   - "In-sample Granger tests condition on regime labels identified from the same returns, creating potential circularity."
   - "We mitigate this... a frozen OOS design eliminates this concern entirely: fit HMM on 1990--2012, apply to 2013--2024 without refitting."

   **But:** The pair selection itself (HML--SMB) was "in-sample screening" (lines 298--303). Choosing the HML--SMB pair post-hoc from 30 directed pairs, even if "motivated by economic prior," introduces selection bias. The OOS frozen design removes **regime identification** circularity but not **pair selection** bias. The paper correctly applies Bonferroni to this (30 pairs, α/30 = 0.00033), so OOS Elevated (p=0.014 F, permutation p=0.063) does **not** survive correction. The "modest evidence" framing is honest, but the claim that frozen OOS "eliminates this concern" overstates the case. **Recommendation:** Line 251, revise to "A frozen OOS design eliminates regime-identification circularity but not pair-selection bias; we apply Bonferroni correction to account for the 30-pair screen."

4. **Boundary exclusion at regime transitions.** Lines 282--287 state:
   - "For Granger tests at lag L, we require all lags within the same regime (ẑ_{t-ℓ} = k for ℓ ∈ {1, ..., L})."
   - Exclusion rates: 0.67% in-sample, 7.4% OOS.

   **Problem:** Regime transition days are precisely when predictive structure may be most dynamic. Excluding them is conservative but may **systematically underestimate** the transition effects that the paper is trying to study. **Recommendation:** Run sensitivity check: include transition-day observations, assign them to the majority regime, and report whether results hold. If they flip, the 7.4% exclusion is material.

---

## REVIEWER 3: FINANCE SKEPTIC
**Mandate:** Is the OOS fragility honestly disclosed? Is the p-value ordering credible? Any remaining "prepared everything" feeling?

### SCORES
| Dimension | Score | Comment |
|-----------|-------|---------|
| **Narrative clarity** | 6.5/10 | Honest but scattered; hard to extract single claim. |
| **Organization** | 6/10 | Methods and results clear; Discussion pages reveal scope anxiety. |
| **Technical rigor** | 7.5/10 | Sound, but HMM local optima undermine generalizability claims. |
| **Honesty/disclosure** | 8/10 | Paper admits fragility; but does it admit **enough**? |
| **Novelty** | 6/10 | Structural break is novel; but is regime-switching version new? |
| **Impact** | 4/10 | No trading profit, modest OOS evidence, regime-dependent. |

### OOS FRAGILITY AUDIT

**Disclosed fragilities** (in order of severity):

1. **Permutation p-value straddles significance threshold:**
   - Conservative (decimal, n=836): p=0.063
   - Aggressive (percentage, n=953): p=0.022
   - **Standard threshold: α=0.05**
   - **Verdict:** p=0.063 is outside the gate. The paper presents both and claims "both conventions yield F-p < 0.05." This is **technically true** (F-p=0.014) but **misleading** because permutation p (the appropriate test for non-parametric regime labels) is what matters. F-p is biased toward rejection under permuted label resampling.
   - **Honest statement (paper could be clearer):** "Permutation p=0.063 is exploratory; this result would not be published in isolation."

2. **Does not survive Bonferroni correction:**
   - 30 directed factor pairs screened
   - OOS Elevated HML→SMB: rank 2 by F-statistic (F=9.06)
   - Bonferroni threshold (α/30): p=0.00033
   - OOS permutation p=0.063 >> 0.00033
   - **Verdict:** Paper acknowledges this (line 559), calling it "exploratory." Fair. But in abstract (lines 47--50), it says "permutation p ∈ [0.063, 0.022]" without emphasizing "does not survive Bonferroni." **Readers may overweight OOS as confirmation rather than exploratory re-emergence evidence.**

3. **Sensitive to regime count K:**
   - K=2: null (p=0.572)
   - K=3: significant (p=0.025)
   - K=4: marginal (p=0.056)
   - **Verdict:** The effect **concentrates in a specific intermediate-volatility state.** If that state is an artifact of K=3 specification, the finding is fragile. Paper says BIC favors K=3 (ΔBIC=1,680 over K=2), but BIC penalizes complexity. Why prefer K=3 over K=2 on grounds of parsimony and then report OOS results as confirmatory? **Recommendation:** Include a sentence: "The OOS Elevated signal is specific to K=3 specification; generalization to other data or time periods may not detect a similar intermediate-volatility regime."

4. **Sensitive to HMM local optima:**
   - 7 local optima clusters identified across 50-seed multistart
   - OOS Elevated significant in 2 of 3 local optima (from primary fit perspective): seeds 28, 20, 6 (BIC=75,587, 0% GFC detection) and seeds 21, 9, etc. (BIC=75,805, 90% GFC detection)
   - **Problem:** The two "significant" optima represent **radically different regime interpretations.** Seed~28 assigns 0% of 2008 to Crisis (p=0.88). Seeds 21, 9 assign 90% of 2008 to Crisis (p=0.018). These are **mutually exclusive regime ontologies.** Yet both yield OOS Elevated significance.
   - **Interpretation:** The OOS Elevated result is **regime-ontology-agnostic.** It's saying: "In intermediate-volatility periods post-GFC, HML→SMB emerges, regardless of whether you label 2008 as Crisis or not." This is actually **reassuring for generalization** (the signal is robust to different regime definitions), but it obscures what "Elevated regime" really means across fits.
   - **Verdict:** Paper's transparency here is good (Appendix H), but main Results should state: "The OOS Elevated signal is robust across two distinct regime ontologies (BIC ±218), suggesting the intermediate-volatility state is structurally stable even when crisis-regime definitions differ."

5. **Secondary pair validation (MOM→SMB) ranks #1 by F-statistic, not HML→SMB:**
   - Lines 333--338: "HML→SMB ranks 2nd of 30 by F-statistic (F=9.06, p=0.003); the rank-based max-statistic p=2/30=0.067. The top-ranked pair is MOM→SMB (F=20.3), which was not the economically motivated target."
   - **Verdict:** This is honest but reveals **selection bias.** The paper is motivated by HML--SMB institutional crowding, but the data favor MOM→SMB. The paper briefly mentions MOM→SMB in Section 4.2 (lines 636--640) as showing "the regime-conditional phenomenon extends beyond a single factor pair," but this is **understated.** If MOM→SMB is stronger and equally regime-dependent, why is it not the main finding?
   - **Answer:** HML--SMB has prior economic justification (crowding, deleveraging hypothesis); MOM→SMB does not. But this introduces **prior-shopping bias.** The paper is honest about it but perhaps too casual. **Recommendation:** Devote a paragraph to this tension: "MOM→SMB is empirically strongest, but HML→SMB has stronger economic motivation. This suggests either (a) the momentum-size relationship reflects a similar deleveraging mechanism, or (b) our economic prior over-constrains pair selection. Pre-registered replication on international data with independent regime definitions would resolve this."

6. **Event-level validation: 2 of 6 events match predicted pattern (binomial p≈0.11):**
   - Lines 836--850, Table 9: Expected HML→SMB in stress episodes.
   - Actual: 2/6 events show expected pattern (2011 EU Debt, 2020 COVID); 4/6 directionally correct (but 2 reversed: 2018 Vol Shock, 2022 Rate Hikes).
   - **Verdict:** Low statistical power and heterogeneous results. Paper correctly calls this "exploratory, low-power" (line 845). But why include it at all if binomial p≈0.11? **Answer:** To show the per-regime pooled OOS result is more reliable than individual events (line 849). This is fair, but the implication is: "The regime-conditional signal is real but event-level predictability is noisy."

### REMAINING ISSUES BLOCKING "STRONG ACCEPT"

1. **"Prepared everything" fragmentation.** The paper includes:
   - **Main finding:** Structural break in Normal regime (p=8.75e-9, Bonferroni-significant, Chow-confirmed, TOST-verified) ✓ Definitive.
   - **Secondary finding:** OOS Elevated re-emergence (permutation p=0.063, not Bonferroni-significant) ✓ Exploratory.
   - **Supporting evidence:** FF25 overlap (ρ_s=0.35, p=0.046, sensitivity fit), event-level validation (2/6), MS-VAR (LR=114.95, p=8.0e-13, BIC penalizes), trading strategy (Sharpe=-0.07), VaR application (CC p=0.336), transfer entropy (TE SMB→HML z=5.37, p<1e-6), complexity diagnostics (RF/MLP/LSTM non-significant), rolling Granger (episodic peaks), lag sensitivity (robust 1--15).

   **Verdict:** The main finding stands alone. Everything else is **exploratory infrastructure.** A reader finishes thinking: "So, HML→SMB predictability broke in 2008 and hasn't returned reliably. Got it. Why the 1,100-line appendix?" **Recommendation:** Explicitly position all secondary analyses as **robustness checks, not confirmatory.** The paper does this locally but not globally.

2. **The January 2008 break date is "narrative-motivated, not data-selected"** (line 451):
   - Quandt-Andrews sup-F: breakpoint date is June 1998 (LTCM/Russia), not January 2008.
   - Paper acknowledges this (lines 451--457) but adopts January 2008 "narrative-motivated" because August 2007 quantitative meltdown is the story hook.
   - **Verdict:** This is intellectually honest but undermines generalizability. If the true breakpoint is June 1998, the paper is **narrative-fitting** to the GFC rather than discovering regime structure objectively. The Quandt-Andrews result suggests **multiple breaks**, not a single clean structural change.
   - **Recommendation:** Acknowledge that the single-break Chow test at January 2008 is a specific hypothesis test (valid) but not the data-driven breakpoint (June 1998 is). Reframe: "We test whether the January 2008 GFC marks a structural break in the HML→SMB relationship. The data support this Chow test (p=2.29e-6), but a full Quandt-Andrews scan identifies an earlier breakpoint at June 1998, suggesting potential multi-break structure. This ambiguity highlights that the calendar-crisis interpretation is one lens; a regime-switching model (HMM) captures dynamics without assuming specific dates."

3. **The "Regime label note" (lines 381--391) is buried in Results and not reflected in discussion.** Under the "leakage-safe" primary fit:
   - Crisis regime captures high-kurtosis statistical state
   - 0% of 2008 assigned to Crisis (vs. 83% under Gaussian HMM)
   - This is called a "regime label note" but it's actually a **major interpretive caveat.**

   **Verdict:** The paper is being cautious (avoiding the appearance of picking regimes to fit 2008), but this creates confusion. Readers expect "Crisis regime" to align with actual crises. The note reveals it doesn't—by design. **Recommendation:** Move this to the Introduction's methodology section: "To avoid regime selection bias, we fit a Student-t HMM unsupervised, minimizing the degree of freedom available for crisis-period fitting. This 'leakage-safe' approach ensures that regime-conditional Granger results reflect distributional properties, not backward-looking crisis labeling. As a result, the Crisis regime (high-kurtosis) does not align perfectly with calendar crises; this is a feature, not a bug."

4. **Lack of pre-registration or pre-specified analysis plan.** Paper states (line 319): "The HML--SMB pair was not formally pre-registered; this analysis is therefore exploratory with respect to pair selection, even though the economic prior (liquidity-mediated contagion) preceded data analysis." This is honest but limits impact. A finance journal reading this might say: "So you picked the HML--SMB pair, found it's significant in Normal regime (p=8.75e-9), and then froze the HMM and tested OOS. The OOS p=0.063 is not significant. Where's the replication?"
   - **Verdict:** The structural break finding is definitive and does not rely on pre-registration. The OOS evidence does. Paper is clear about this, but downstream readers may not be.

---

## REVIEWER 4: SHARP MESSAGE TEST
**Mandate:** Would you cite this paper? For what exactly? Is the message now sharp?

### SCORES
| Dimension | Score | Comment |
|-----------|-------|---------|
| **Narrative clarity** | 8/10 | Clear, well-written, hero visible. |
| **Organization** | 7.5/10 | Good structure; appendix could be compressed. |
| **Technical rigor** | 8/10 | Sound methods, transparent assumptions. |
| **Honesty/disclosure** | 8.5/10 | Fragilities disclosed; appropriate caveats. |
| **Novelty** | 7/10 | Structural break is novel; regime methodology is not. |
| **Impact** | 6/10 | Limited to factor timing; no practical trading application. |

### CITATION SHARPNESS ASSESSMENT

**What would you cite this paper for?**

**SHARP USE CASE 1: "Cross-factor predictability undergoes structural breaks"**
- Cite line 88--94: HML→SMB Granger causality was Bonferroni-significant pre-GFC (p=8.75e-9) but broke sharply in January 2008 (Chow p=2.29e-6) and is statistically absent post-GFC (p=0.73, TOST-confirmed).
- **Why cite:** Shows that factor-timing models assuming regime-invariant cross-factor relationships may misspecify dynamics during structural transitions (line 701).
- **Impact:** Moderate. Relevant to factor-model researchers, but niche audience.

**FUZZY USE CASE 2: "Regime-conditional Granger causality reveals hidden predictive structure"**
- Cite lines 114--117: Student-t HMM conditioning on latent market state reveals regime-dependent relationships invisible to unconditional analysis.
- **Why cite:** If I'm building a risk model and want to condition on regimes.
- **Problem:** The methodology is not novel (Hamilton 1989 Markov-switching, Bulla 2011 Student-t HMM, Psaradakis et al. 2005 regime-switching Granger). Paper credits prior work (lines 160--176) but does not propose a **new method.** It applies standard tools to a new domain (cross-factor Granger in Fama-French).
- **Verdict:** I would cite this for **empirical application** (HML--SMB structural break), not **methodological innovation.**

**WEAK USE CASE 3: "Out-of-sample frozen HMM design provides robust cross-regime validation"**
- Cite lines 251--254, 510--521: Fit HMM on 1990--2012, freeze parameters, apply to 2013--2024 without refitting.
- **Why cite:** Elegant design avoiding in-sample bias.
- **Problem:** OOS result does not survive Bonferroni correction (p=0.063 permutation, α/30=0.00033).
- **Verdict:** The design is citable, but the result is too modest to anchor a paper.

**REJECTED USE CASE: "HML→SMB predictability improves risk monitoring"**
- Appendix F (200 lines) on VaR application: hybrid HMM+volatility model passes Christoffersen test (CC p=0.336).
- **Why reject:**
  - The Granger effect is non-significant in 2013--2024 (p=0.162, line 979). The VaR improvement might reflect volatility-regime conditioning, not HML info content.
  - False-alarm rate 93.2% (line 964). On 14.9% of test days, alerts activate with no VaR breach following.
  - The "Hybrid" detector requires adding a volatility override (not principled HMM alone). This is engineering, not science.
  - **Verdict:** I would not cite this for risk monitoring. A GARCH(1,1) model (3.91% violation rate, simpler, no regime selection) competes strongly.

### ONE-SENTENCE ELEVATOR PITCH

**Proposed sharp message:**
> "HML→SMB Granger causality was extraordinarily significant in the pre-GFC Normal regime (p=8.75×10⁻⁹) but underwent a sharp structural break in January 2008 (Chow p=2.29×10⁻⁶) and has been statistically absent for 16 years, with modest evidence of regime-dependent re-emergence post-GFC."

**Verdict:** This is **clear, falsifiable, and citable.** This is what I would cite the paper for.

### REMAINING ISSUES BLOCKING "STRONG ACCEPT"

1. **The paper tries to do too much.** It starts as a structural-break discovery paper (hero: p=8.75e-9 and Chow break), then adds:
   - Frozen OOS validation (exploratory)
   - FF25 portfolio overlap mechanism (supportive but secondary)
   - Event-level consistency checks (heterogeneous)
   - MS-VAR comparison (shows effect is small)
   - Trading strategy backtest (negative Sharpe)
   - VaR risk monitoring application (engineering)
   - Transfer entropy directional asymmetry (exploratory)
   - Complexity diagnostics (OLS sufficient, no nonlinearity)
   - Local optima taxonomy (necessary transparency, but not a finding)

   **Verdict:** Lines 1--750 are focused and citable. Lines 751--1,274 are **exhaustive robustness checking** that could be summarized in 3--4 tables. The paper reads like the author has tried to anticipate every objection and pre-emptively address it. This is honest but dilutes the message.

2. **The Discussion does not resolve scope clearly.** Lines 674--702:
   - "The structural break IS the contribution" (line 676) ✓
   - "OOS evidence is modest" (line 683) ✓
   - "Economic magnitude and practical scope" (line 697): Sharpe=-0.07, "value is academic"

   **Problem:** These are presented as three separate sub-findings, not as a hierarchy. **Recommended restructuring:**

   ```
   Section 5.1 Main Finding: The Structural Break
   -----------------------------------------------
   HML→SMB Granger causality is Bonferroni-significant in pre-GFC Normal
   regime (p=8.75e-9) and breaks sharply at January 2008 (Chow p=2.29e-6),
   remaining absent for 16 years (p=0.73, TOST-confirmed). This finding is
   robust to HAC correction, lag specification, and HMM seed selection.

   Section 5.2 Secondary Evidence: OOS Re-emergence Post-GFC
   -----------------------------------------------------------
   A frozen OOS design (HMM trained 1990--2012, applied 2013--2024 without
   refitting) finds modest evidence of Elevated-regime HML→SMB (permutation
   p=0.063), consistent with partial recovery of cross-factor structure in
   intermediate-volatility periods. This result is exploratory: it does not
   survive Bonferroni correction, requires K=3 specification, and is
   sensitive to HMM local optima.

   Section 5.3 Practical Implications and Limitations
   ---------------------------------------------------
   Effect sizes are economically small (ΔR² ≈ 2%), do not generate trading
   profits (Sharpe=-0.07), and reflect predictive precedence, not structural
   causality. The academic value lies in documenting that factor-timing models
   assuming regime-invariant cross-factor relationships may misspecify
   dynamics during structural transitions.
   ```

   **Current Discussion does this but buries the hierarchy.** Reordering would sharpen the message.

3. **Missing ablation: what if we don't use regimes?** The paper's hero is regime-conditional. But what if we simply fit unconditional Granger causality before and after January 2008? Lines 443--444 provide a preview:
   - Pre-GFC Normal: p=6.66e-16
   - Post-GFC Normal: p=0.73
   - Overall pre-2008: ~6.66e-16 (implied)
   - Overall post-2008: p=0.73 (implied)

   **Verdict:** The structural break is **not regime-specific**; it's **time-indexed.** The regime conditioning sharpens the signal (Bonferroni p=8.75e-9 vs. estimated p=6.66e-16 unconditional), but the break itself is robust to regime definition. **Recommendation:** Include a 1-paragraph ablation: "An unconditional Granger test before and after January 2008 yields qualitatively identical conclusions: strong pre-GFC (p≈1e-15) and null post-GFC (p≈0.7), confirming the structural break is not an artifact of regime conditioning."

4. **No confidence interval or posterior distribution for the breakpoint.** The paper tests January 2008 (Chow p=2.29e-6) and notes the Quandt-Andrews sup-F identifies June 1998. But are these two breakpoints separated enough to conclude 2008 is special? What if there's a credible set [1997, 2003] and [2006, 2009] overlapping? **Recommendation:** Include Bai-Perron sequential testing or a credible interval for breakpoint timing. Current Chow + Quandt-Andrews analysis is a start but leaves ambiguity about how sharp the break really is.

---

## CONSENSUS PANEL SUMMARY

| Finding | Consensus | Confidence |
|---------|-----------|-----------|
| **Structural break is real and time-indexed to January 2008** | STRONG | 9/10 |
| **Pre-GFC Normal regime finding (p=8.75e-9) is robust and Bonferroni-significant** | STRONG | 9/10 |
| **OOS Elevated re-emergence (p=0.063 permutation) is exploratory, not confirmatory** | STRONG | 9/10 |
| **Frozen OOS design eliminates regime-identification bias but not pair-selection bias** | STRONG | 8/10 |
| **Effect sizes are economically small (ΔR² ≈ 2%), no trading application** | STRONG | 9/10 |
| **Paper knows its main contribution (structural break) but undersells clarity of scope** | MODERATE | 7/10 |
| **Appendices exceed necessary robustness; contain scope creep (VaR, trading strategy)** | MODERATE | 7/10 |
| **HMM local optima and regime K-sensitivity are material but appropriately disclosed** | MODERATE | 7/10 |

---

## COLLECTIVE RECOMMENDATION: WEAK ACCEPT to ACCEPT

### What Has Improved in Round 3 (Post-Narrative Surgery)
✓ Introduction now leads with hero result by line 91
✓ Gaussian vs Student-t comparison section deleted (removed 30 lines of defense)
✓ Main result renamed to "The Structural Break in HML→SMB Predictability"
✓ Post-GFC attenuation hypotheses compressed (14 → 6 lines)
✓ Incremental R² table removed, converted to 5-line inline text
✓ Discussion rewritten: "The structural break IS the contribution" + "OOS evidence is modest"
✓ Trading strategy appendix compressed (50 → 10 lines)
✓ Event-based appendix narrative compressed

### Remaining Barriers to "Strong Accept"

1. **SCOPE AMBIGUITY:** Paper title and abstract position "structural break" (definitive) and "regime-dependent re-emergence" (exploratory) as co-equal findings. Readers are confused about which is the contribution.
   - **Fix:** Explicitly reorder Discussion to make structural break primary (Section 5.1), OOS Elevated secondary (Section 5.2), and practical limitations tertiary (Section 5.3). Add one-sentence hierarchy: "The structural break is the main finding; OOS re-emergence is exploratory supporting evidence."

2. **OOS OVERWEIGHT:** Permutation p=0.063 does not survive Bonferroni correction (α/30=0.00033). Yet it is featured in abstract and Contributions as co-finding. This is honest disclosure but risks overstating evidential weight.
   - **Fix:** Move OOS to "supporting evidence" rather than "contribution." Reorder Contributions: (1) Structural break with dated breakpoint, (2) Secondary evidence of regime-conditional re-emergence with disclosed fragility.

3. **APPENDIX SCOPE CREEP:** VaR application (200 lines), trading strategy (15 lines), and MS-VAR (20 lines) are not necessary to support the main finding. They are engineering applications or methodological comparisons.
   - **Fix:** Trim VaR from 200 to 20 lines (mention as future work). Delete trading strategy backtest (already proven to fail at line 698). Move MS-VAR to Methods note (5 lines) confirming effect is real but small.

4. **QUANDT-ANDREWS AMBIGUITY:** January 2008 Chow test is significant (p=2.29e-6) but Quandt-Andrews sup-F identifies June 1998 breakpoint. Paper acknowledges this but calls January 2008 "narrative-motivated." This invites skepticism about whether the break is objectively discovered or story-fitted.
   - **Fix:** Acknowledge multi-break structure explicitly: "Full Quandt-Andrews scan identifies June 1998 as the sup-F maximizing date (LTCM/Russia crisis), suggesting multiple breaks may be present. The January 2008 Chow test confirms the GFC marks a structural break, but does not identify it as the sole or primary breakpoint."

5. **PAIR SELECTION TRANSPARENCY:** MOM→SMB ranks #1 by F-statistic (F=20.3), but HML→SMB is studied because it has economic motivation (crowding, deleveraging). This introduces prior-shopping bias.
   - **Fix:** Acknowledge this explicitly: "Our primary pair selection is guided by an economic prior (HML--SMB institutional crowding) rather than pure empirical strength. MOM→SMB is empirically strongest but lacks ex-ante motivation. This prior-driven selection reduces degrees of freedom but introduces potential prior-shopping bias. Pre-registered replication on independent data would provide definitive confirmation."

---

## FINAL SCORES (PANEL AVERAGE)

| Dimension | R1 | R2 | R3 | R4 | AVERAGE | INTERPRETATION |
|-----------|----|----|----|----|---------|-----------------|
| **Narrative clarity** | 7.5 | 7.0 | 6.5 | 8.0 | **7.3/10** | Good but could be sharper |
| **Organization** | 7.0 | 6.5 | 6.0 | 7.5 | **6.75/10** | Bloated appendix pulls down score |
| **Technical rigor** | 8.0 | 8.5 | 7.5 | 8.0 | **8.0/10** | Sound methods, appropriate caveats |
| **Honesty/disclosure** | 8.0 | 8.5 | 8.0 | 8.5 | **8.25/10** | Strong transparency; some understatement of caveats |
| **Novelty** | 6.5 | 6.5 | 6.0 | 7.0 | **6.5/10** | Structural break is novel; methodology is not |
| **Impact** | 5.0 | 5.0 | 4.0 | 6.0 | **5.0/10** | Niche audience (factor timing); no practical application |

---

## PUBLICATION RECOMMENDATION

**Consensus: ACCEPT with MINOR REVISIONS**

The paper now **clearly knows what its contribution is:** a **structural break in HML→SMB Granger causality dated to January 2008, with robust in-sample evidence (p=8.75e-9, Bonferroni-significant) and modest out-of-sample supporting evidence (p=0.063 permutation, exploratory).**

After narrative surgery, this is **sharp, citable, and honest about limitations.** The appendices are overly comprehensive but not wrong. The frozen OOS design is elegant, and the transparency about local optima and regime sensitivity is exemplary.

**Recommendation for final round:**
1. Reorder Discussion to clarify hierarchy (structural break primary, OOS secondary)
2. Trim VaR appendix from 200 to 20 lines
3. Add one explicit sentence to Quandt-Andrews caveat acknowledging multi-break possibility
4. Strengthen pair-selection transparency: acknowledge MOM→SMB is empirically strongest but HML→SMB has prior motivation
5. (Optional) Include unconditional Granger before/after 2008 to show break is robust to regime definition

**With these minor revisions, this is a publishable contribution to the factor-investing literature.**

---

## END OF PANEL REVIEW

**Generated:** February 25, 2026
**Paper Status:** Post-narrative-surgery revision
**Reviewer Panel Composition:** 4 independent adversarial perspectives (Narrative, Organization, Finance Skeptic, Impact)
