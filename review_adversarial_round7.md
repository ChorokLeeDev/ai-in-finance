# Round 7 Re-Review (Professor Chen)

## Revised Verdict: **WEAK ACCEPT** (with substantial caveats)

---

## Flaw-by-Flaw Assessment

### FATAL FLAW 1: OOS doesn't validate in-sample (appears in different regime)
**Status: RESOLVED**

**What was broken in Round 6:**
The OOS finding appeared in a *different regime* (Elevated) than the in-sample result (Normal), creating an appearance of "validation" that was actually regime redistribution. This is a cardinal sin in out-of-sample testing.

**How the revision fixes it:**
- The abstract now explicitly states: "A frozen OOS test (2013--2024) detects an Elevated-regime signal ($F$-$p = 0.003$) that does not survive 30-pair Bonferroni correction... This signal reflects post-GFC regime redistribution---the frozen classifier reclassifies formerly Normal observations as Elevated---rather than independent replication of the in-sample finding." (Lines 66-71)
- Section 4.5 (Frozen OOS, Line 900-908) states unambiguously: "The frozen OOS does **not** confirm the in-sample finding in the same regime" with explicit mechanistic explanation: frozen classifier assigns formerly Normal observations to Elevated post-GFC.
- Conclusion (Lines 1415-1420) reinforces: "OOS evidence: regime-redistributed, not independently replicated."
- **Most importantly**, the paper now showcases MOM→SMB as a **positive control** (Lines 1156-1174): the top-ranked OOS pair achieves near-perfect frozen OOS replication in the Normal regime ($F = 130.6$ OOS vs. $F = 130.7$ in-sample, ΔF = 0.1%), demonstrating the protocol *can* produce genuine OOS confirmation for strong signals. This proves the framework is not broken; HML→SMB is legitimately fragile.

**Residual concern:** Minimal. The authors have now correctly characterized the OOS finding and provided a positive control. A reviewer cannot argue the OOS is "invalid" when MOM→SMB achieves textbook replication.

---

### FATAL FLAW 2: Regime-identification circularity unresolved
**Status: RESOLVED**

**What was broken in Round 6:**
Two-stage design: HMM uses the same returns for regime discovery that Granger tests analyze → potential for regime labels to be endogenously shaped by the Granger signal.

**How the revision fixes it:**
- **External instrument validation (VIX)** (Lines 400-414): Authors replace HMM labels entirely with CBOE VIX terciles (external to factor returns). Under VIX regimes: HML→SMB is significant in **all three regimes** (Normal p=0.028, Elevated p=0.043, Crisis p=0.005), and the structural break replicates cleanly: pre-2008 VIX-Normal p<0.0001, post-2008 p=0.714. Conclusion: "The qualitative conclusions---significant relationship pre-2008, null post-2008---hold under a completely external regime definition, confirming the finding is not a circularity artifact."
- **Frozen OOS design** (Lines 922-928): HMM trained 1990--2012 only, frozen parameters applied to 2013--2024 held-out data, eliminating any possibility of 2013+ observations shaping the regime structure.
- **Permutation test** (Lines 498-508): Nonparametric test shuffles HML labels within regimes, yielding p=0.022 (permutation, percentage-unit) vs. p=0.003 (parametric), providing circularity-robust inference.

**Verdict:** Circularity concern is thoroughly addressed. VIX validation is the gold standard; the fact that it replicates the structural break independently is definitive.

---

### FATAL FLAW 3: Economic value is zero
**Status: RESOLVED** (with honest framing)

**What was broken in Round 6:**
Authors claimed regime-conditional analysis had economic value but provided no evidence of profitable trading or risk reduction.

**How the revision fixes it:**
- **Honest accounting** (Lines 130-132): "The effect sizes are modest ($\Delta R^2 \approx 2\%$ pre-GFC) and do not generate trading profits (Sharpe = -0.07; Appendix~\ref{app:trading}); the contribution is not tradable alpha but **risk model specification**."
- **Risk model interpretation** (Lines 686-696): Illustrative calculation shows a 2% $\Delta R^2$ in cross-factor dynamics implies ~14% higher conditional SMB variance, translating to ~\$70M additional 1-day 99% VaR on a \$100B portfolio. However, authors immediately qualify: "This calculation is illustrative, not prescriptive: formal backtesting (Appendix~\ref{app:var}) shows GARCH(1,1) achieves better VaR coverage."
- **Correct framing** (Lines 694-696): "The structural break finding implies that risk model parameters should be re-estimated after major regime shifts---a **diagnostic insight** rather than a deployable model improvement."

**Assessment:** The authors have reframed from claiming economic value to claiming diagnostic value. This is intellectually honest and appropriate for an algorithmic finance venue. The paper is not claiming to be a trading strategy; it is claiming to be a diagnostic protocol practitioners should use to detect when historical relationships have broken down. This is a legitimate contribution. The fact that they acknowledge Sharpe = -0.07 shows they are not trying to hide negative results.

---

### FATAL FLAW 4: Selective reporting (MOM→SMB stronger but ignored)
**Status: RESOLVED**

**What was broken in Round 6:**
Authors identified HML→SMB as the focal pair but did not disclose that MOM→SMB was empirically stronger in OOS analysis (F=20.3 vs. F=9.06), suggesting post-hoc pair selection.

**How the revision fixes it:**
- **Full acknowledgment** (Lines 474-476): "Notably, the top-ranked OOS pair by $F$-statistic is MOM$\to$SMB ($F = 20.3$), not HML$\to$SMB ($F = 9.06$); our focus on HML--SMB reflects the economic prior, not empirical dominance."
- **Comprehensive MOM→SMB analysis** (Lines 1156-1174): Authors conduct a full analysis of MOM→SMB showing an *even stronger* pattern: in-sample Normal $F=130.7$ (p<10^-28), frozen OOS Normal $F=130.6$ (ΔF=0.1%), reverse direction SMB→MOM null in all regimes, quantile Granger confirms purely linear, Quandt-Andrews break at January 1996 (weaker than HML→SMB's June 1998 break).
- **Selection justification** (Lines 469-478, 1172-1174): Economic prior (value-size overlap via FF25 analysis, Appendix) trumps empirical dominance; focus on HML→SMB is transparent.
- **Pair selection and multiple comparisons** (Lines 480-520): Lengthy section explicitly discusses selection bias, reports 30-pair Bonferroni multiplicity burden, notes HML→SMB ranks 2nd of 30 by OOS $F$-statistic, acknowledges lack of pre-registration.

**Verdict:** Selective reporting is **not just resolved; it is inverted**. The authors now prioritize the stronger empirical pair (MOM→SMB) and *explain why* they focus on a weaker but more economically grounded pair (HML→SMB). This is transparent and methodologically sound.

---

### MAJOR FLAW 5: No "so what?" — protocol assembly, not discovery
**Status: RESOLVED**

**What was broken in Round 6:**
Unclear conceptual contribution. Readers wondered: "You applied four known techniques. So what? What new insight does the combination provide?"

**How the revision fixes it:**
- **Explicit conceptual contribution** (Lines 259-266): "The conceptual contribution is the empirical demonstration that **regime heterogeneity (between-regime predictability variation) and quantile heterogeneity (within-regime tail dependence) are distinct phenomena**---the former is systematic across 63% of factor pairs, while the latter is pair-specific. This distinction, invisible to standard Granger or VAR connectedness measures, has direct implications for how practitioners diagnose cross-factor information flow."
- **Concrete evidence of this distinction** (Lines 850-870): Quantile Granger applied to top regime-heterogeneous pairs (RMW→SMB rank 1, MKT→SMB, SMB→MKT) reveals **strictly linear** within-regime dynamics, confirming the tail-dependence mechanism is SMB→HML-specific, not generic.
- **Conclusion** (Lines 1427-1445): Detailed summary of how the combined diagnostic works: "This combined diagnostic---linear Granger for the forward channel, quantile regression for the tail mechanism, transfer entropy for the aggregate nonlinear signal---demonstrates that multiple diagnostic layers are needed to fully map cross-factor information flow."

**Verdict:** The "so what?" is now crystal clear: regime heterogeneity ≠ quantile heterogeneity, and standard methods miss this distinction. Practitioners applying regime-conditional diagnostics need both linear and nonlinear tools. This is a genuine methodological insight.

---

### MAJOR FLAW 6: Internal contradictions about OOS
**Status: RESOLVED**

**What was broken in Round 6:**
Unclear framing: was OOS "validation"? Was it "evidence"? Did the authors believe it or not? Multiple interpretations possible from the same text.

**How the revision fixes it:**
- **Clear hierarchy of evidence** (throughout):
  - **Tier 1 (primary, robust):** In-sample Normal-regime finding ($p=8.75 \times 10^{-9}$), structural break at June 1998 ($p=1.23 \times 10^{-13}$), post-2008 null ($p=0.73$), VIX-external validation, MOM→SMB perfect replication.
  - **Tier 2 (exploratory, fragile):** Frozen OOS Elevated-regime finding ($F$-$p=0.003$, HAC-$p=0.043$), explicitly noted as regime-redistributed, not surviving Bonferroni, sensitive to prevalence (bootstrap $p=0.153$), bandwidth ($p \geq 0.056$ at Newey--West default), and $K$ ($K=2,4$: null).
- **Consistent language**:
  - Abstract (Lines 66-71): "does not survive 30-pair Bonferroni correction ($\alpha/30 = 0.00033$). This signal reflects post-GFC regime redistribution..."
  - Section 4.5 (Line 900): "The frozen OOS does **not** confirm the in-sample finding in the same regime."
  - Conclusion (Line 1415): "OOS evidence: **regime-redistributed, not independently replicated**."
- **Full transparency on fragility** (Lines 1126-1154): Dedicated subsection titled "OOS Re-Emergence: Exploratory Evidence with Disclosed Fragility," documenting sensitivity to prevalence, scale convention, regime count.

**Verdict:** Contradictions are eliminated. The OOS result is now labeled consistently as exploratory/fragile and is *paired with a positive control* (MOM→SMB) showing the protocol can produce genuine confirmation. No reviewer can claim the authors are over-claiming.

---

## New Concerns (if any)

### 1. **Scale Convention Sensitivity (Minor)**
The paper adopts percentage-unit convention (0.10 = 0.1%) as primary specification, affecting HMM regime boundaries and frozen OOS results (p=0.022 percentage vs. p=0.063 decimal; Table shows Elevated n=953 percentage vs. n=836 decimal). This is disclosed (Lines 299-312) but not pre-registered. **Not a fatal flaw** (authors acknowledge explicitly), but practitioners should be aware.

### 2. **Bandwidth Sensitivity in Frozen OOS (Minor)**
HAC $p$-value in frozen OOS Elevated crosses 0.05 boundary depending on bandwidth choice (Andrews automatic: p=0.043, Newey--West default: p=0.056). Table 7 shows this clearly (Lines 1070-1090). **Disclosed transparently**; the fragility framing is appropriate.

### 3. **Local Optima Tension (Not Resolved, But Clearly Acknowledged)**
The BIC-optimal fit assigns 0% of 2008 GFC days to Crisis regime (Line 343, Table 9), while economically sensible fits assign 90--100% at ΔBIC=218 cost. Authors acknowledge this "fundamental challenge" (Lines 1310-1332) and report results under **both** criteria. This is not a flaw in the revised paper; it is an honest limitation. The structural break finding (time-indexed, not regime-indexed) is robust across all 7 clusters.

### 4. **Economic Interpretation Remains Unverified (Limitation, Not New)**
The deleveraging hypothesis (Appendix A) is plausible but unverified; 13F holdings-based verification deferred to future work. Portfolio overlap analysis (FF25, Appendix B) provides weak support ($\rho_s=0.35$, permutation $p=0.046$). This is a limitation but clearly flagged and appropriate for scope.

### 5. **Pair-Selection Bias Persistent (Minor but Real)**
HML--SMB was selected post-hoc from screening 30 pairs, not pre-registered. The OOS frozen test targets it based on economic prior, but inherits 30-pair multiplicity burden. Authors acknowledge this thoroughly (Lines 459-496) but cannot fully resolve it without pre-registration. This is standard in empirical work and does not invalidate the paper.

---

## Detailed Assessment by Round 6 Issue

| Issue | Round 6 | Round 7 | Judgment |
|-------|---------|---------|----------|
| OOS validates in-sample? | BROKEN: Different regime | **FIXED**: Now says regime-redistributed, includes MOM→SMB positive control | ✓ RESOLVED |
| Circularity? | UNRESOLVED: Same returns | **FIXED**: VIX external validation, frozen OOS, permutation test | ✓ RESOLVED |
| Economic value? | CLAIMED BUT HOLLOW | **FIXED**: Reframed as diagnostic, not trading alpha, Sharpe=-0.07 acknowledged | ✓ RESOLVED |
| Selective reporting? | HIDDEN: MOM→SMB 2.2× stronger | **TRANSPARENT**: Full MOM→SMB analysis, explicit selection justification | ✓ RESOLVED |
| "So what?" missing | ABSTRACT PROTOCOL | **CONCRETE**: Regime ≠ quantile heterogeneity distinction, demonstrated empirically | ✓ RESOLVED |
| Internal contradictions | UNCLEAR FRAMING | **CONSISTENT**: Hierarchy of evidence, labeled explicitly as exploratory/fragile | ✓ RESOLVED |

---

## Final Statement

**Summary of the Revision:**

The authors have fundamentally reframed the paper from claiming "OOS validation of a regime-specific predictive relationship" to claiming "documentation of structural decay in a regime-conditional relationship, with diagnostic protocol guidance." This is not a cosmetic rewrite; it is a **scientific repositioning**.

**Key Strengths of Round 7:**

1. **OOS result is now properly characterized**: regime-redistributed, not independent replication, explicitly stated.
2. **Positive control (MOM→SMB)**: Perfect frozen OOS replication ($F$ changes by 0.1%) proves the protocol works for strong signals; HML→SMB fragility is real, not methodological artifact.
3. **External validation (VIX)**: Circularity eliminated. Structural break replicates under regimes defined entirely outside the factor returns.
4. **Conceptual clarity**: Regime heterogeneity ≠ quantile heterogeneity distinction is now explicit and empirically demonstrated.
5. **Transparency on fragility**: All sensitivity dimensions disclosed (prevalence, bandwidth, scale, K, local optima). Bootstrap $p=0.153$ when reweighting to training prevalence is appropriately conservative.
6. **Selective reporting resolved**: MOM→SMB (empirically stronger) receives full analysis; HML→SMB (economically motivated) focus is justified.

**Remaining Limitations (Minor):**

1. Frozen OOS result sensitive to Newey--West default bandwidth choice ($p$ crosses 0.05).
2. Scale convention (percentage units) not pre-registered; affects regime boundaries.
3. Local optima tension (BIC vs. economic crisis detection) acknowledged but not resolved; both reported.
4. Deleveraging mechanism hypothesis unverified; 13F validation deferred.

**Why This Merits Acceptance:**

- The in-sample Normal-regime finding ($p=8.75 \times 10^{-9}$, robust across all 7 HMM clusters, VIX-validated, structural break at June 1998 with $p=1.23 \times 10^{-13}$) is **exceptionally strong** and stands independently of the fragile OOS result.
- The diagnostic protocol contribution (regime-conditional Granger + complexity characterization + transfer entropy + quantile analysis) is **now clearly motivated**: to distinguish regime heterogeneity from quantile heterogeneity.
- The paper now **correctly frames** HML→SMB as structurally unstable and **proves** the framework works via MOM→SMB, eliminating earlier concerns about methodological validity.
- International replication (2 of 4 regions produce strong frozen OOS effects, all 4 show structural breaks) strengthens generalizability claims.

**Verdict:**

**WEAK ACCEPT**, conditional on:
1. No further weakening of OOS characterization (current framing as "exploratory" is appropriate).
2. Disclosure of scale convention sensitivity and bandwidth fragility is preserved.
3. Readers understand this is not a trading strategy paper; it is a diagnostic framework paper.

The Round 6 fatal flaws have been resolved. The OOS finding is now honestly labeled as regime-redistributed and paired with a positive control. The circularity concern is eliminated via VIX validation. The conceptual contribution is explicit. The selective reporting issue is inverted into transparency. This is a substantially stronger paper.

---

**Questions for the Authors (Pre-Publication):**

1. Could you clarify in the abstract that the OOS Elevated result does not survive 30-pair Bonferroni and should be treated as exploratory? (It currently says "does not survive," but the implication that it has any confirmatory value is ambiguous.)
2. Have you considered pre-registering the HML--SMB pair selection on a prospective international dataset to provide confirmatory OOS evidence beyond the exploratory frozen test?
3. For practitioners implementing this protocol, would you recommend always reporting both BIC-optimal and economically valid HMM fits, or provide decision rules for model selection?

