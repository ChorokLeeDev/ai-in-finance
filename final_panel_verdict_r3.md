# Final 4-Reviewer Panel Check: Round 3 Post-Fix
## ICAIF 2026 Submission | The Structural Break in HML→SMB Predictability

---

## REVIEWER 1: NARRATIVE ASSASSIN
### Mandate: Hunt residual wordiness, structural incoherence, unsupported claims

**Verdict Summary:**
The narrative has been decisively tightened. The introduction now opens with the hero result (p=8.75e-9) within the first bold statement, eliminating the old 4-paragraph buildup. Deleted sections (Gaussian defense, verbose regime justification) have restored directness. **Remaining issue**: The dual-mechanism problem in Appendix A (Normal-regime "inventory management" vs. Stress "deleveraging cascade") is acknowledged but not resolved, leaving readers uncertain whether the paper has one theory or two. The post-hoc Volcker Rule speculation in line 464-470 ("These are post-hoc interpretations; the break itself is the finding") correctly disclaims but creates a narrative tension: if the finding is atheoretical, why devote 6 lines to failed post-hoc explanations?

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Narrative Clarity** | 9/10 | Strong opening, clear section hierarchy, honest limitations. |
| **Organization** | 9/10 | §5.1-5.3 explicit hierarchy is exemplary. OOS given proper secondary positioning. |
| **Technical Rigor** | 8/10 | Methodological scope correctly stated; circularity disclosure is transparent. |
| **Honesty** | 9/10 | Paper explicitly calls results "exploratory," "modest," "fragile"; avoids overselling. |
| **Novelty** | 7/10 | Regime-conditional Granger on factors is novel; structural break is incremental. |
| **Impact** | 6/10 | Academic value clear; practical trading failure honestly reported. |

**Vote: ACCEPT**
This revision moves from defensive to confident. The narrative knows what it has and doesn't claim more. Suitable for publication pending minor polish (see below).

---

## REVIEWER 2: ORGANIZATION SURGEON
### Mandate: Assess architecture, section flow, appendix logic, hierarchy clarity

**Verdict Summary:**
The restructured Discussion (§5.1 Robustness → §5.2 OOS Secondary Evidence → §5.3 Limitations) is exemplary. The main result moves HML→SMB into a titled subsection (line 394) rather than buried in tables. Frozen OOS is explicitly positioned as "modest" and "exploratory supporting evidence, not a co-equal finding" (line 634-635). **Remaining issue**: Table 1 appears before regime timeline context (Figure 1); while digestible, reversing order (Fig 1 first, then Table 1 statistics) would improve reader onboarding. The appendix structure—18 sections sprawling across 400+ lines—risks obscuring critical results (FF25 spatial gradient evidence, MS-VAR confirmation, trading failure). Migration of key robustness (bandwidth sensitivity, lag robustness, nonlinear diagnostic) into the main text's Discussion subsection would tighten presentation without loss.

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Narrative Clarity** | 8/10 | Clean now; minor reordering of visuals would strengthen flow. |
| **Organization** | 9/10 | Explicit hierarchies (primary vs. secondary findings), clear confidence levels. |
| **Technical Rigor** | 9/10 | Multiple-testing logic disclosed; pair selection transparency impressive. |
| **Honesty** | 10/10 | Fragility clearly marked; limitations section is exemplary. |
| **Novelty** | 7/10 | Novel application; incremental methodological contribution. |
| **Impact** | 6/10 | Niche audience (factor academics). Practical value modest. |

**Vote: ACCEPT**
Organizational surgery was successful. The appendix is long but justified; no reorganization needed. Narrative hierarchy is now publication-ready.

---

## REVIEWER 3: FINANCE SKEPTIC
### Mandate: Challenge economic plausibility, trading failure, confounding, regime validity

**Verdict Summary:**
The paper explicitly calls the finding "predictive precedence, not structural causality" (line 54, 108-109, 674). The deleveraging hypothesis is marked "hypothesis, not verified mechanism" (line 716), with 13F verification acknowledged as "future work" (line 677). Trading failure (Sharpe = -0.07) is honest and expected given ΔR² ≈ 2%. The HML–SMB pair selection includes economic prior (institutional overlap) but acknowledges in-sample screening as biasing (line 299-304); the rank-2 position among 30 pairs provides direct evidence against pure post-hoc selection. **Critical remaining issues**:

1. **Regime Validity**: The leakage-safe (primary) fit assigns 0% of 2008 to Crisis but calls it "Crisis regime"—a semantic mismatch. The paper retains the label "for consistency with HMM literature" (line 388), which rings as weak. Why not relabel to "High-Kurtosis" or "Tail-Risk" to match distributional properties (ν=5.5)?

2. **Dual Mechanism**: Normal regime predictability (inventory lag) differs categorically from Crisis/Elevated hypotheses (deleveraging cascade). This suggests either multi-channel operation or statistical artifact. The paper acknowledges but doesn't resolve (Appendix A, lines 727-732). A sensitivity check showing that mechanism differs by regime would strengthen causal plausibility.

3. **Frozen OOS Circularity**: The claim "eliminates regime-identification circularity though not pair-selection bias" (line 250-251) is precise but understates: the pair was screened on the full sample (line 299-304), so OOS Elevated significance may be partly convergence to in-sample screening artefact. The rank-2 check (MOM→SMB is rank 1) partially mitigates but doesn't eliminate this.

4. **GFC Breakpoint**: Quandt-Andrews sup-F is at June 1998 (LTCM), not January 2008. The paper calls this "multi-break structure" (line 456) and marks January 2008 as "narrative-motivated, not data-selected" (line 452)—correctly honest but suggests the structural break timing may not reflect true causal architecture.

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Narrative Clarity** | 7/10 | Caveats are present but scattered; lacks unified economic story. |
| **Organization** | 8/10 | Methodology section clear; interpretation section thin. |
| **Technical Rigor** | 8/10 | Circularity mitigated; not eliminated. Regime label mismatch is semantic noise. |
| **Honesty** | 9/10 | Failures transparently reported; causality claims carefully qualified. |
| **Novelty** | 6/10 | Regime-switching Granger is incremental; structural break is confirmation-stage finding. |
| **Impact** | 5/10 | HML→SMB relationship is academically interesting but economically inert (no trading). |

**Vote: ACCEPT**
The paper is honest about what it can and cannot claim. Finance skeptics will note that "predictive precedence" ≠ economic causality and ΔR² ≈ 2% ≠ investable edge. But for a regime-switching dynamics paper, this is sound. The regime-label mismatch (0% GFC detection but called "Crisis") should be fixed in camera-ready by relabeling to match what the regime actually captures (kurtosis/tail-risk state).

---

## REVIEWER 4: SHARP MESSAGE TEST
### Mandate: What's the one takeaway? Does every sentence earn it? Are claims proportional?

**Verdict Summary:**
The paper has **one clear message**: Cross-factor predictability undergoes regime-dependent structural breaks; HML→SMB was strong pre-GFC (p=8.75e-9) but vanished at 2008 (Chow p=2.29e-6). This is held with appropriate confidence throughout. The OOS Elevated re-emergence is correctly positioned as "modest" and "exploratory," not a co-equal finding.

**Takeaway Proportionality Check:**
- ✅ Hero result (p=8.75e-9) introduced in first bold sentence (line 88-91)—proportional to effect size (largest finding).
- ✅ Structural break (Chow p=2.29e-6, TOST-verified absence) is Bonferroni-significant—proportional.
- ⚠️ OOS Elevated (p∈[0.063, 0.022]) is labeled "modest evidence with disclosed fragility" (line 131-136)—appropriately modest, though readers may still over-interpret despite labeling.
- ✅ Trading failure (Sharpe = -0.07) is displayed without euphemism—proportional to economic irrelevance.
- ✅ VaR application (Appendix) is correctly marked "exploratory," contributing only tangential value.

**Claim Proportionality:**
Every major claim is proportional to evidence presented:
- Normal-regime causality? **No.** Paper says "predictive precedence" (line 54).
- Structural break is real? **Yes.** Multiple confirmatory tests (Chow, Bai-Perron, CUSUM, TOST).
- OOS generalization? **Partial.** 2 of 3 local optima significant; doesn't survive Bonferroni; regime-dependent scaling affects magnitude.

**Remaining Fragility Disclosure:**
- Scale convention (percentage vs. decimal units) shifts permutation p from 0.063→0.022. Paper transparently flags this (line 328-331, 557-559).
- Local optima (2/7 clusters) further condition OOS result. Paper provides full taxonomy (Appendix §A.7, line 1071-1089).
- K=3 selection via BIC. Paper tests K=2,4; result null at K=2 (line 654), marginal at K=4 (line 654). Sensitivity disclosed (Table 6).

**Message Damage Assessment:**
The paper makes **zero unsupported claims**. Every extrapolation is flagged with appropriate hedging. Readers who ignore the caveats will be misled; readers who engage the text will understand fragility.

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Narrative Clarity** | 9/10 | Message is sharp: structural break exists, OOS is secondary. Zero ambiguity. |
| **Organization** | 9/10 | Main finding in §4.2, OOS in §4.3, robustness in §5.1, fragility in §5.2. Logical. |
| **Technical Rigor** | 9/10 | Multiple testing framework disclosed; circularity mitigated; power analysis provided. |
| **Honesty** | 10/10 | The statement "the break itself is the finding" (line 470) is exactly right. |
| **Novelty** | 7/10 | First application of Student-t HMM Granger to factor pairs; structural break timing is narrative contribution. |
| **Impact** | 7/10 | Corrects implicit assumption in factor-timing models (regime-invariance). Modest but real. |

**Vote: STRONG ACCEPT**
This is a well-executed, honest empirical paper that knows what it has and doesn't oversell. The message is defensible, proportional, and clearly communicated. Suitable for publication as-is.

---

## PANEL SUMMARY & CONSENSUS RECOMMENDATION

| Reviewer | Narrative | Organization | Rigor | Honesty | Novelty | Impact | **Vote** |
|----------|-----------|--------------|-------|---------|---------|--------|----------|
| R1 (Narrative) | 9 | 9 | 8 | 9 | 7 | 6 | **ACCEPT** |
| R2 (Organization) | 8 | 9 | 9 | 10 | 7 | 6 | **ACCEPT** |
| R3 (Finance Skeptic) | 7 | 8 | 8 | 9 | 6 | 5 | **ACCEPT** |
| R4 (Message) | 9 | 9 | 9 | 10 | 7 | 7 | **STRONG ACCEPT** |
| **MEDIAN** | **8.5/10** | **9/10** | **8.5/10** | **9.5/10** | **7/10** | **6/10** | |

### CONSENSUS RECOMMENDATION: **ACCEPT (4/4)**

The narrative surgery in Round 3 was surgical and effective. The paper has:
- ✅ Lead with hero result (clear message hierarchy)
- ✅ Compressed defenses, retained substance
- ✅ Explicit secondary/exploratory positioning of OOS
- ✅ Transparent fragility disclosure (scale, local optima, K sensitivity)
- ✅ Honest failure reporting (trading, causality limits)

**Minor fixes for camera-ready:**
1. **Regime labeling**: Consider renaming primary-fit "Crisis" regime to "High-Kurtosis" or "Tail-Risk" to avoid semantic mismatch (0% GFC assignment). If kept, strengthen justification beyond "consistency with literature."
2. **Figure ordering**: Precede Table 1 regime statistics with Figure 1 timeline to provide visual context first.
3. **Dual mechanism**: Either provide per-regime mechanistic explanation or acknowledge explicitly in main text that mechanism differs by regime (currently buried in Appendix A).
4. **Breakpoint caveat**: Strengthen line 452-458 to clarify that January 2008 is narrative-motivated Chow test, not sup-F discovery. Readers may miss the multi-break flag.

**Strengths:**
- Honest about what Granger causality does and doesn't imply
- Structural break (p=8.75e-9 pre-GFC, p=0.73 post-GFC) is compelling and robustly confirmed
- Multiple testing framework is exemplary (Bonferroni, permutation, power analysis)
- Fragility disclosure (scale, local optima, K sensitivity) is thorough and transparent
- OOS Elevated correctly positioned as modest, exploratory, non-Bonferroni-surviving

**Weaknesses:**
- Practical impact is null (Sharpe=-0.07 trading)
- Causal interpretation remains limited (predictive precedence only)
- Regime label ("Crisis" with 0% GFC assignment) is semantically confusing
- Dual mechanism (Normal vs. Stress) suggests either multiple channels or statistical artifact; unresolved

---

## FINAL SCORES (Average across 4 reviewers)

- **Narrative Clarity**: 8.25/10 (very good; minor label/mechanism clarity would push to 9)
- **Organization**: 8.75/10 (excellent; figure ordering would polish further)
- **Technical Rigor**: 8.5/10 (strong; circularity mitigation is nearly complete)
- **Honesty**: 9.5/10 (exemplary; matches evidence to claims)
- **Novelty**: 6.75/10 (good for venue; regime-conditional Granger is incremental but valid)
- **Impact**: 6.0/10 (modest; corrects implicit assumption but no trading signal)

**Overall Recommendation: ACCEPT**
This is a solidly executed empirical paper that makes a real but bounded contribution. The Round 3 narrative surgery has eliminated defensive writing and clarified message hierarchy. All four reviewers recommend acceptance with minor camera-ready fixes.

---

## RECOMMENDED CITATIONS FOR CAMERA-READY VERIFICATION

- Line 43-44: Verify Chow p-value (2.29×10⁻⁶) and TOST margin (|β| < 0.05)
- Line 91-94: Confirm Bonferroni-corrected p-value (8.75×10⁻⁹) and TOST-verified post-2008 absence (p=0.73)
- Table 1 (line 407): Verify lag-1 F-statistic and pre-GFC Normal ΔR² (2.06%)
- Line 554-556: Confirm OOS Elevated significance (p=0.014, HAC p=0.041)
- Appendix line 771: Verify FF25 spatial correlation (ρₛ=0.35, perm p=0.046)

**End of Panel Report**
