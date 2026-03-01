# Logical Flow and Structure Analysis: main_icaif.tex

## EXECUTIVE SUMMARY

The paper has **strong foundational logic** with clear evidence hierarchy and primary findings, but suffers from **significant structural gaps** in the Discussion section, weak forward references, and an organizational problem where exploratory findings are presented without sufficient flagging. The abstract promises match delivery, but the arrangement often violates the principle of presenting evidence hierarchy front-to-back.

---

## 1. ABSTRACT vs. DELIVERY ALIGNMENT

**RATING: MEDIUM ISSUE**

**Location:** Lines 30-54 (Abstract) vs. entire paper

**Problem:**
The abstract makes four core claims:
1. HML→SMB breaks at June 1998 in Normal regime pre-crisis
2. Transfer entropy reveals nonlinear reverse channel (SMB→HML)
3. Frozen OOS yields exploratory signal not surviving Bonferroni
4. International replication confirms breaks in all 4 markets

The paper **delivers all four**, but the **sequencing is problematic**:
- Transfer entropy findings (Section 4.2) appear AFTER the structural break section, but the abstract leads with them as a major discovery
- The exploratory OOS signal is front-loaded in the abstract (line 46-48) before sufficient caveating
- International replication is mentioned in abstract but relegated to a single table (Table 3) with minimal discussion

**Impact:** Reader expectation mismatch. The abstract's ordering (regime significance → structural break → transfer entropy → OOS exploratory → international) does NOT match paper's logical build.

**Fix:** Reorder abstract to match paper's actual evidence hierarchy:
- Lead with primary finding (in-sample Normal regime, structural break, VIX validation)
- Secondary: Transfer entropy directional asymmetry
- Tertiary: International confirmation
- Exploratory: OOS results (shorten mention, emphasize fragility earlier)

---

## 2. INTRODUCTION: EVIDENCE HIERARCHY CLARITY

**RATING: LOW-MEDIUM ISSUE**

**Location:** Lines 94-100 (Evidence Hierarchy section)

**Problem:**
The evidence hierarchy is **introduced correctly and explicitly**, which is excellent. However:

1. **Tier definitions are backward-looking:** Introduced at line 94 AFTER the paper's main claim (line 87), which is fine, but Tier 1/2/3 should be **applied consistently throughout** the paper.

2. **Tier application inconsistency:**
   - "Primary finding" (line 717-724) repeats Tier 1 information
   - Tier 3 (exploratory OOS) gets substantial space in Results (lines 459-540) without constant reinforcement that it's exploratory
   - Tier 2 (international, MOM→SMB) is scattered across Results without labeled subsections

3. **Missing forward reference:** The Discussion (line 575-578) says "assess how general the findings are...how sensitive they are" but doesn't explicitly map this to Tier 2 and 3 evidence, forcing reader to infer the mapping.

**Impact:** MEDIUM. Reader must mentally track evidence tiers while reading. The Tier 1/2/3 framework exists but isn't integrated into section headings or result summaries.

**Fix:** Add Tier labels to table captions and subsection headers:
- "Results: Tier 1 (Primary In-Sample Findings)"
- "Results: Tier 2 (Confirmatory Evidence: MOM→SMB and International)"
- "Results: Tier 3 (Exploratory: Frozen OOS Analysis)"

---

## 3. METHODOLOGY → RESULTS TRANSITION

**RATING: MEDIUM ISSUE**

**Location:** Lines 135-201 (Methodology) → Line 202 (Results header)

**Problem:**
1. **Algorithm 1 (Protocol, lines 137-150)** describes 7 steps but the paper never explicitly states which are applied to which section of Results. Reader must infer:
   - Steps 1-4 → Table 1-2 (Regime characteristics, structural break)
   - Step 5 → Table 3-4 (Frozen OOS)
   - Step 6-7 → Tables 5-6 (Transfer entropy, quantile Granger)

2. **Missing transition sentence** between Methodology (line 201 ends) and Results (line 202 starts). No bridging statement like "We apply this protocol sequentially, first to regime characterization, then to structural break detection..."

3. **Data discussion (lines 152-160)** includes a scale-sensitivity caveat ("HMM emission probabilities are not scale-invariant") that is **critical context** for interpreting the frozen OOS results (Tier 3), but the caveat is buried and not revisited.

**Impact:** MEDIUM. Reader must trace methodology steps forward manually; no explicit map provided.

**Fix:** Add transition paragraph after line 201:
"We apply this protocol in three stages. First, we characterize regimes and test the primary hypothesis (HML→SMB in Normal regime) with Bonferroni correction and robustness checks (Steps 1-4). Second, we validate this signal using VIX terciles and frozen OOS (Step 5), with explicit caveating of regime redistribution effects. Third, we diagnose the mechanism and directional asymmetry (Steps 6-7). Throughout, we report effect sizes and note where results depend on methodological choices (HMM scale, local optima, lag selection)."

---

## 4. RESULTS SECTION: ORGANIZATIONAL COHERENCE

**RATING: MEDIUM-HIGH ISSUE**

**Location:** Lines 202-571 (entire Results section)

**Problem A: Structural break findings dominate without sufficient framing (lines 236-330)**

The "Structural Break" subsection (4.2) is large and detailed, but:
- **No opening sentence** explaining why we test structural breaks (motivation is in abstract/intro but not restated)
- **Quandt-Andrews test (lines 274-288)** is presented without preamble: reader must infer this tests "if/when" the HML→SMB relationship broke
- **Rolling Granger (lines 317-330)** is a robustness check but labeled generically as "Robustness" without specifying "robustness of structural break claim"
- Figure 3 (rolling Granger, lines 289-297) is introduced but never discussed in the text—it just floats as illustration

**Impact:** MEDIUM. The structural break finding is primary but feels scattered across subsections without unified narrative.

**Problem B: Complexity Characterization section (lines 332-457) has weak logical connector**

Lines 332-341 open with three questions:
1. Is the channel linear or nonlinear?
2. Does direction matter (SMB→HML)?
3. What is the mechanism?

Then **four model classes (Table 4)** answer Q1, **Transfer Entropy (Table 5)** answers Q2, **Quantile Granger (Table 6)** answers Q3. This structure is **sound in principle** but:
- The questions are rhetorical; no bridging between answer to Q1 and Q2
- Lines 371-376 note "fit-dependent" linearity caveat, which should come BEFORE the four-model diagnostic, not after
- "Sensitivity caveat" (line 371) undermines the claim at line 364 that nonlinear methods show "no improvement," but the reader has already accepted that conclusion

**Impact:** MEDIUM. Logical coherence is present but fragile; minor reordering would strengthen.

**Problem C: Frozen OOS section (lines 459-540) violates evidence hierarchy**

This is marked as Tier 3 (exploratory) at lines 501-502, but:
- Subsection title "Frozen OOS and Validation" (line 459) uses the word "Validation," which implies Tier 1/2 rigor
- **40 lines of detailed results** (Table 5-6, sensitivity analyses) before the Tier 3 flag at line 501
- Reader has invested cognitive effort in the exploratory signal before learning it's exploratory
- MOM→SMB positive control (lines 522-540) is excellent but appears **after** the main (weak) result rather than before, reversing the evidence-first principle

**Impact:** MEDIUM-HIGH. Tier 3 results are presented as if Tier 1 until explicitly flagged. Reordering would improve reader trust.

**Problem D: International replication (lines 541-571) is underdeveloped**

- **Only a table (Table 7), no text summary** of key findings beyond 3 sentences (lines 541-549)
- No discussion of why breaks occur at different dates (2003-2014 across regions)
- No integration with the deleveraging hypothesis from the Discussion
- Feels like an afterthought appended to Results rather than part of the evidence hierarchy

**Impact:** LOW. This is secondary evidence and correctly positioned as such, but the execution is rushed.

---

## 5. DISCUSSION SECTION: LOGICAL GAPS AND GRAB-BAG ORGANIZATION

**RATING: CRITICAL ISSUE**

**Location:** Lines 573-760 (Discussion)

**Problem A: Discussion lacks overarching thesis (lines 573-579)**

Opening paragraph claims to assess "how general" and "how sensitive" the findings are, but:
- No explicit statement of what the Discussion will conclude
- No roadmap of sections: Multi-pair generalizability → Local optima → Economic magnitude → Economic interpretation → Baseline comparison → Scope/limitations

The Discussion **reads as a grab-bag of subsections** without a unifying argument. Contrast with Introduction (line 87): "This paper documents structural decay of cross-factor predictability"—that is a thesis. The Discussion has no analogous summary.

**Impact:** CRITICAL. Reader cannot predict what comes next; each subsection feels independent.

**Fix:** Add opening thesis after line 578:
"The evidence for HML→SMB structural decay is robust within the US context (Tier 1), with secondary confirmation in other factor pairs and international markets (Tier 2), but the exploratory OOS findings are fragile and limited to regime redistribution (Tier 3). Below, we assess the generality of the phenomenon, the stability of our methodological choices, and the economic implications."

**Problem B: Multi-pair generalizability (lines 590-615) is underdeveloped**

- Figure 4 (heatmap, lines 580-588) shows all-pairs causality but is never discussed in the text
- "19 of 30 pairs (63%) show regime heterogeneity" is stated (lines 592-593) but not interpreted
- Why are these other pairs secondary? The paper says HML→SMB reflects "economic prior" (line 538) but doesn't explain why institutional crowding is more plausible than the other 18 pairs' mechanisms

**Impact:** MEDIUM. The paper claims the phenomenon is general but doesn't make the case convincingly.

**Fix:** Add interpretation of heatmap and other pairs. Why does MOM→SMB have a stronger signal? What does the 63% prevalence of regime heterogeneity suggest about factor structure?

**Problem C: Local optima section (lines 617-643) lacks integration**

- Table 8 shows 7 clusters, all with significant in-sample Normal $p < 10^{-8}$
- This is excellent robustness, but the section reads as a technical aside rather than evidence consolidation
- Decision rule for practitioners (lines 620-622) is helpful but isolated; should be integrated into Conclusion

**Impact:** LOW. The finding is solid; presentation could be tightened.

**Problem D: Economic magnitude (lines 645-654) contradicts main findings**

- States "Effect sizes are modest ($\Delta R^2 \approx 2\%)...do not generate trading profits (Sharpe = -0.07)"
- GARCH(1,1) outperforms regime-conditional models for VaR
- This **raises a question not answered in the paper:** If the effect is unprofitable and doesn't improve risk forecasts, why should practitioners care?

The paper answers this at lines 651-654: the finding is a "diagnostic task" (when to revisit covariance structures), not a trading strategy. **But this is the first mention of the practical value proposition.** The paper has spent 5000+ words on a predictive signal and waits until the Discussion to explain its utility.

**Impact:** MEDIUM. The paper needs an earlier statement of why practitioners should care about statistical predictability that doesn't generate alpha.

**Problem E: Economic interpretation (lines 656-664) is speculative**

- Proposes a "deleveraging cascade" hypothesis with three testable predictions
- This is interesting, but it's presented in the Discussion as speculation without evidence
- The paper tests **no element** of this hypothesis (no 13F analysis, no portfolio-level breakdown verification)

**Impact:** MEDIUM. Hypothesis is generative for future work but belongs in Future Work section, not Discussion proper.

**Problem F: Baseline comparison (lines 666-689) is weakly motivated**

- Why compare to rolling-window and threshold-based regimes only?
- Why not compare to regime-switching VARs (Psaradakis et al., cited at line 124) or neural Granger (Tank et al., line 128)?
- The paper cites these methods but doesn't compare against them

**Impact:** LOW-MEDIUM. The section is incomplete but not essential to the main claims.

**Problem G: Scope and limitations (lines 691-713) overlaps with introduction**

- Lines 691-693 repeat that Granger ≠ structural causality (already stated at line 22)
- "Trivariate controls...most prominent common driver" (lines 694-695) adds no new information relative to Results
- "Pair selection is post-hoc" (line 698) was already disclosed at lines 195-200

**Impact:** LOW. Repetition is appropriate for emphasis, but this section doesn't advance understanding.

---

## 6. FORWARD REFERENCES AND UNDEFINED CONCEPTS

**RATING: LOW-MEDIUM ISSUE**

**Location:** Throughout

**Problems:**
1. **Line 115:** "BIC-vs-economic-validity tension" is introduced but not explained until Section 3.2.2 (line 175-178)
2. **Line 157-160:** Scale sensitivity caveat is mentioned for HMM but not connected to the frozen OOS problem until line 702-703 (Discussion)
3. **Line 370:** "Random Forest importance = 0.043, 4× the mean" uses "the mean" without defining the mean (mean across features? lags?)
4. **Line 661:** "FF25 portfolio overlap analysis" is mentioned but never shown or defined in the paper

**Impact:** LOW. Most forward references are quickly resolved, but FF25 analysis is never presented.

**Fix:**
- Define all undefined quantities (e.g., "mean importance across all lags and regimes")
- Either present the FF25 analysis or remove the reference (lines 659-661)

---

## 7. CONCLUSION SECTION

**RATING: LOW ISSUE**

**Location:** Lines 715-760

**Problem:**
The Conclusion restates findings from Results without adding synthesis:
- Lines 717-724 repeat the primary finding (Table 2 + Quandt-Andrews)
- Lines 726-729 repeat VIX validation (line 299-306)
- Lines 731-736 repeat transfer entropy and quantile Granger findings (Section 4.2)

However, the Conclusion **does add value** in three ways:
1. **Implications for practice** (lines 746-752): Regime-conditional protocol is reusable
2. **Future work** (lines 754-759): Explicit roadmap
3. **Data availability** (lines 761-765): Transparency

**Impact:** LOW. Repetition is acceptable; the section meets minimum standards.

**Fix:** Strengthen implications (lines 746-752) with a concrete statement like:
"Factor-timing models should implement frozen HMM validation during backtesting to detect regime-dependent instability in cross-factor relationships. Our protocol (Algorithm 1) requires minimal additional computation and provides economic value through model recalibration triggers."

---

## 8. LOGICAL JUMPS AND MISSING CONNECTORS

### Jump 1: From normal-regime significance to structural break (lines 259-274)
**Location:** Lines 259-274
**Issue:** The paper shows Normal-regime HML→SMB is significant (Table 2, line 259), then immediately tests for a structural break in the **full sample** (Quandt-Andrews, lines 274-288). The logical connection is: "If HML→SMB is significant in pre-crisis times and null post-crisis, a structural break must exist." But this reasoning is implicit, not explicit.
**Fix:** Add transition: "Given the stark difference between Normal-regime significance and post-2008 nullity (line 269), we test whether a structural break can explain the decay."
**Rating:** LOW (easy inference)

### Jump 2: From linear Granger to nonlinear complexity (lines 332-365)
**Location:** Lines 332-365
**Issue:** The paper tests whether Granger is linear (four-model diagnostic) but doesn't explain why transfer entropy is needed if Granger already measures information flow. The logical connection (Granger tests conditional mean; TE tests mutual information) is only made at lines 443-446, **after** the results.
**Fix:** Move lines 443-446 to line 338, before the four-model diagnostic.
**Rating:** MEDIUM (confusing ordering)

### Jump 3: From in-sample significance to OOS fragility (lines 459-501)
**Location:** Lines 459-501
**Issue:** The frozen OOS subsection begins with "To test whether...regime-conditional structure generalizes" (lines 461-463) without acknowledging that the paper just spent 70 lines proving it **does** generalize (robustness checks, local optima, lags 1-15). The reader expects OOS results to **confirm** the in-sample finding, not undermine it.
**Fix:** Reframe OOS section: "While the in-sample finding is robust across specifications, whether this signal extrapolates to post-2012 data is uncertain. Below, we assess the frozen OOS performance and identify regime redistribution as a confound."
**Rating:** MEDIUM (reader expectation mismatch)

### Jump 4: From two-direction Granger to "directional asymmetry" claim (lines 407-447)
**Location:** Lines 407-447
**Issue:** Normal-regime HML→SMB Granger is significant ($p < 10^{-8}$) but SMB→HML Granger is null ($p = 0.864$, Table 2 line 249). This is presented as support for "directionality." But then transfer entropy shows SMB→HML is stronger ($z = 5.37$ vs. 2.45, Table 5). The paper reconciles this at lines 442-446, but the reader is surprised by the reversal. The directional asymmetry **between Granger and TE** is interesting but not clearly framed as the paper's novel finding.
**Fix:** Open Section 4.2 with: "We observe a surprising directional asymmetry: Granger causality (conditional-mean test) detects HML→SMB but not SMB→HML, while transfer entropy (mutual information test) shows the opposite. This asymmetry reveals the linear vs. nonlinear boundary of information flow."
**Rating:** MEDIUM (logical coherence present but delayed)

---

## 9. STRENGTH OF EVIDENCE PRESENTATION

**Front-loading:** Does the paper lead with strongest evidence?

**Yes, generally:**
- Lines 87-92: Main claim (HML→SMB Normal regime, structural break)
- Lines 94-100: Evidence hierarchy explicitly stated
- Table 2: Primary result with Bonferroni correction (line 259)
- VIX validation (lines 299-306): Independent regime definition

**But with caveats:**
- Tier 3 (exploratory OOS) receives equal visual weight as Tier 1 in Results section
- Transfer entropy asymmetry (interesting but secondary) is elevated in the abstract (line 42-45)
- International replication (confirmatory) is relegated to a brief mention at the end of Results

**Recommendation:** Reduce space devoted to Tier 3 exploratory results; move international replication to prominently labeled subsection in Discussion.

---

## 10. ABSTRACT-BODY COHERENCE: FINAL CHECK

| Abstract Claim | Paper Delivery | Alignment |
|---|---|---|
| HML→SMB Granger-predicts SMB "exclusively in Normal regime" | Yes, Table 2, $p = 8.75 \times 10^{-9}$ | ✓ |
| Structural break June 1998, $p = 1.23 \times 10^{-13}$ | Yes, lines 275-288 | ✓ |
| Post-2008 consistent with zero (16 years) | Yes, lines 284-285, CI $[-0.049, 0.073]$ | ✓ |
| Robust across HAC, lags, controls, 7 clusters | Yes, lines 260-330, Table 8 | ✓ |
| Transfer entropy SMB→HML reverse stronger ($z = 5.37$) | Yes, Table 5, lines 405-407 | ✓ |
| Quantile regression reveals tail dependence ($p = 0.001$) | Yes, Table 6, lines 438-446 | ✓ |
| Frozen OOS exploratory signal ($F$-$p = 0.003$), doesn't survive Bonferroni | Yes, Table 5, lines 490-491 | ✓ |
| MOM→SMB near-perfect replication ($\Delta F = 0.1\%$) | Yes, lines 527-528 | ✓ |
| International breaks in 4 non-US markets | Yes, Table 7, lines 541-549 | ✓ |

**Verdict:** Abstract promises match delivery. **No major mismatch.**

---

## CRITICAL ISSUES SUMMARY

| Issue | Location | Severity | Type |
|---|---|---|---|
| Discussion lacks overarching thesis | Lines 573-579 | **CRITICAL** | Organization |
| Tier 3 (exploratory) not flagged early enough | Lines 459-501 | **HIGH** | Evidence hierarchy |
| Complexity section has weak connectors between Q1/Q2/Q3 | Lines 332-376 | **MEDIUM** | Logical flow |
| OOS section titled "Validation" (implies Tier 1) when exploratory | Line 459 | **MEDIUM** | Framing |
| Missing transition between Methodology and Results | Line 201→202 | **MEDIUM** | Connectors |
| Transfer entropy motivation not stated until results presented | Lines 337-446 | **MEDIUM** | Logical ordering |
| Multi-pair generalizability underdeveloped | Lines 590-615 | **MEDIUM** | Comprehensiveness |
| FF25 portfolio analysis mentioned but never shown | Lines 659-661 | **MEDIUM** | Incomplete claims |
| Scale-sensitivity caveat separated from OOS implications | Lines 157-160 vs. 702-703 | **MEDIUM** | Connectors |
| Economic value proposition not stated until Discussion | Lines 651-654 | **MEDIUM** | Reader motivation |

---

## RECOMMENDATIONS FOR REVISION

### Priority 1 (Restructure for clarity):
1. **Add Discussion thesis paragraph** (after line 578) that maps subsections to evidence tiers
2. **Reorder frozen OOS section**: Present MOM→SMB positive control **before** HML→SMB exploratory result
3. **Add transition paragraph** between Methodology (after line 201) mapping algorithm steps to results sections

### Priority 2 (Strengthen logical connectors):
4. **Motivate transfer entropy earlier** (in Section 4.2 opening) by noting the Granger/TE directional asymmetry
5. **Move complexity caveat** (lines 371-376) to **before** the four-model diagnostic (lines 362-376), not after
6. **Strengthen "Economic interpretation" paragraph** (lines 656-664): Either present evidence for deleveraging hypothesis or move to Future Work

### Priority 3 (Polish):
7. **Define "mean importance"** at line 370
8. **Either present or remove** FF25 analysis reference (lines 659-661)
9. **Rename subsection** line 459 from "Frozen OOS and Validation" → "Frozen OOS: Exploratory Analysis"
10. **Add figure discussion** for Figure 3 (rolling Granger) in text (currently just a caption at lines 289-297)

---

## OVERALL ASSESSMENT

**Logical Coherence: 6/10**
- Primary evidence chain is sound (intro → hypothesis → test → robustness → VIX validation)
- Secondary evidence (transfer entropy, international) is well-motivated but secondary
- Exploratory findings (frozen OOS) undermine coherence by receiving equal weight

**Evidence Hierarchy Implementation: 6/10**
- Tier 1/2/3 framework is introduced and stated explicitly ✓
- But inconsistently applied to section headings and captions
- Tier 3 is not flagged early enough in Results
- Tier 2 (international, MOM→SMB) feels rushed

**Discussion Organization: 4/10**
- Lacks unifying thesis or roadmap
- Subsections feel disconnected (local optima, economic magnitude, baseline comparison)
- Speculation (deleveraging hypothesis) mixes with limitations

**Forward References: 7/10**
- Generally well-managed; minor undefined quantities
- One significant gap: FF25 analysis mentioned but never shown

**Abstract-Body Alignment: 9/10**
- All abstract claims are delivered
- Minor: Ordering in abstract differs from paper's logical flow

**Overall Recommendation:** The paper's **core finding is solid and logically supported**, but the **organizational structure** could be tightened to improve readability. The Discussion is the weakest section; restructuring around a clear thesis would dramatically improve coherence. The exploratory Tier 3 results should be either cut or much more heavily caveated upfront.

---

## CONCRETE FIX EXAMPLES

### FIX 1: Add Discussion Thesis (after line 578)

**Original (lines 573-579):**
```
\section{Discussion}

We now assess how general the findings are (multi-pair and
international), how sensitive they are to HMM estimation
(local optima, baselines), and what they imply for practitioners
(economic magnitude, mechanism, limitations).
```

**Revised:**
```
\section{Discussion}

\textbf{Thesis.} The structural decay of HML$\to$SMB predictability is
robust evidence at Tier~1 (US in-sample), with confirmatory support
from other factor pairs and international markets (Tier~2), but the
frozen OOS signal is exploratory and reflects regime redistribution
rather than independent replication (Tier~3). Below, we assess the
generality of the phenomenon across factor pairs and geographies,
the stability of the HMM methodology under local optima, and the
economic implications for practitioners.
```

### FIX 2: Reorder Frozen OOS Section

**Original (lines 459-540):**
```
\subsection{Frozen OOS and Validation}
All results so far are in-sample... [HML→SMB weak results: 40 lines]
\textbf{MOM$\to$SMB positive control.} [strong results: 18 lines]
```

**Revised:**
```
\subsection{Frozen OOS: Exploratory Analysis and Positive Controls}

\textbf{MOM$\to$SMB positive control.} [strong results: 18 lines]
To validate the diagnostic protocol, we conduct a full analysis of
MOM$\to$SMB---the top-ranked pair by OOS $F$-statistic... [lines 522-540]

\textbf{HML$\to$SMB frozen OOS (exploratory).} [weak results: 40 lines]
The primary HML$\to$SMB finding achieves strong in-sample confirmation
but does NOT generalize cleanly to frozen OOS... [lines 459-520]
```

**Rationale:** Leading with the strong result (MOM→SMB) establishes protocol validity before presenting the weak result, improving reader credibility assessment.

### FIX 3: Transfer Entropy Motivation (before line 337)

**Current (lines 332-341):**
```
The structural break is robust, but what is the \emph{mechanism}?
Is the Normal-regime channel linear, or does nonlinear structure
lurk beneath the Granger surface? And does direction matter---does
SMB also predict HML? We use a two-stage diagnostic...
```

**Revised:**
```
The structural break is robust, but we uncover a puzzle: Granger
causality detects HML$\to$SMB at $p < 10^{-8}$ in Normal regime
(Table~\ref{tab:main}, line 248) but finds no reverse channel
SMB$\to$HML ($p = 0.864$, line 249). However, a directional asymmetry
emerges when we measure information flow differently. Transfer entropy
(which tests mutual information including tail dependence, not
conditional-mean improvement) detects a much stronger reverse channel
(Table~\ref{tab:te}: SMB$\to$HML $z = 5.37$ vs. HML$\to$SMB $z = 2.45$).

This directional asymmetry suggests that Granger and TE measure
fundamentally different properties of predictive relationships. Below,
we diagnose whether this reflects nonlinearity or tail dependence via
(1)~four-model diagnostic testing nonlinear improvement, and
(2)~quantile Granger isolating the mechanism.
```

**Rationale:** Reader learns the core puzzle (direction reversal) and motivation for diagnostic before results, improving logical flow.

