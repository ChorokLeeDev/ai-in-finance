# Adversarial Review Synthesis — ICAIF Submission
## 4-Reviewer Panel, February 25 2026

---

## Mechanical Fix Status

| Fix | Status | Verification |
|-----|--------|-------------|
| Duplicate labels (fig:rolling, fig:lag_sensitivity) | ✅ Fixed | Labels unique; no "multiply defined" warnings |
| Missing \Description{} tags | ✅ Fixed | All 7 figures now have tags |
| 15 unused bib entries removed | ✅ Fixed | 72/72 entries match citations |
| shu2025dynamic page numbers | ✅ Fixed | pages={13--36} added |
| tab:regimes overfull hbox | ✅ Fixed | tabcolsep reduced to 4pt |
| Compilation | ✅ Clean | 17 pages (with refs), 0 undefined refs, 0 multiply-defined, 2 overfull hboxes remaining |

**Page budget after fixes:** Conclusion ends on page 10–11; appendices start page 11; references page 15–17. Main body is still ~10–11 pages — borderline. Need ~0.5 page of cuts to be safe.

---

## Cross-Reviewer Consensus: Issues Raised by 3+ Reviewers

These issues were independently flagged by multiple reviewers and represent the highest-priority targets:

### 1. THE p=0.063 / p=0.022 DUAL REPORTING (All 4 reviewers)

**The problem:** The abstract headlines p=0.022 (percentage-unit permutation); p=0.063 (decimal-unit) is in a footnote. Every reviewer flagged this as selective reporting or cherry-picking.

**Consensus fix:**
- Report both results with equal prominence in the abstract and §3.5
- Frame as a range: "permutation p ∈ [0.022, 0.063] depending on scale convention"
- Acknowledge this straddles the 5% threshold
- Consider designating the more conservative (decimal-unit) result as primary

### 2. VaR CONTRIBUTION DISCONNECT (3 reviewers: Finance, ML, Structure)

**The problem:** Contribution 4 claims VaR improvement, but (a) seed 42 used for VaR has null Granger OOS (p=0.466), (b) the improvement comes from the volatility override, not the Granger signal. This is the single most damaging credibility issue.

**Consensus fix:**
- Demote from "Contribution 4" to an appendix-only proof-of-concept
- Or rewrite to explicitly attribute improvement to the volatility-override architecture, not the Granger finding
- Remove VaR from the abstract or caveat it heavily

### 3. "CORROBORATION" FRAMING IS MISLEADING (3 reviewers: Finance, Econometrics, Structure)

**The problem:** In-sample finding is Normal-regime, pre-GFC. OOS finding is Elevated-regime, post-GFC. These are different regimes, different eras, different effect sizes (ΔR² 2.06% vs 0.73%). Calling this "corroboration" overstates.

**Consensus fix:**
- Reframe as "regime migration" or "conditional re-emergence," not "corroboration"
- Lead with the structural break as the central finding, not the pre-GFC magnitude
- Present OOS Elevated as suggestive/exploratory rather than confirmatory

### 4. CONTRIBUTION MISMATCH / OVERSELLING (3 reviewers: Finance, ML, Structure)

**The problem:** Four contributions are claimed but evidence is thin for each. C1 is pre-GFC only; C2 is borderline; C3 (Student-t HMM + multistart) is not novel; C4 (VaR) is disconnected.

**Consensus fix:**
- Restructure as 2–3 honest contributions: (1) Historical regime-dependent anomaly + structural break; (2) Suggestive OOS evidence with disclosed fragility; (3) Regime-taxonomy engineering for factor monitoring
- Drop or heavily caveat C4

---

## Prioritized Action List (by expected review impact)

### TIER 1: Fix Before Submission (blocks acceptance)

| # | Issue | Reviewers | Effort | Fix |
|---|-------|-----------|--------|-----|
| 1 | **Dual p-value selective reporting** | All 4 | 30 min | Promote p=0.063 to equal prominence; report range in abstract |
| 2 | **VaR contribution disconnect** | R2, R4, R1 | 1 hr | Demote to appendix proof-of-concept; remove from Contribution list |
| 3 | **"Corroboration" → "regime migration"** | R2, R3, R1 | 1 hr | Rewrite abstract, §1, §5.1 to frame OOS as conditional re-emergence |
| 4 | **Page limit (~11 pages main body)** | — | 1 hr | Compress §2 Related Work (1.8pp → 0.75pp); condense §5.1 convention note |
| 5 | **Contribution list restructuring** | R2, R4, R1 | 45 min | Reduce to 3 honest contributions; drop C4 or reframe |

### TIER 2: Strongly Recommended (reduces attack surface)

| # | Issue | Reviewers | Effort | Fix |
|---|-------|-----------|--------|-----|
| 6 | **Multiple testing: no pre-registration for OOS** | R3 | 30 min | Add explicit disclosure that OOS test was not formally pre-registered; apply 3-regime correction as sensitivity |
| 7 | **"41 of 50 seeds" → "2 of 3 optima"** | R3, R4 | 20 min | Reframe in abstract and §5.1; clarify seeds ≠ independent tests |
| 8 | **K=3 look-ahead bias** | R3 | 20 min | Rewrite §3.2 to state K=3 was verified (not selected) on training data; disclose full-sample selection came first |
| 9 | **Abstract readability** | R1 | 45 min | Restructure: economic question → primary finding → structural break → OOS → scope. Cut technical minutiae. |
| 10 | **Narrative oscillation** | R1 | 1 hr | Lead with the structural break as the story; position OOS as conditional. Remove the "strong → weak → strong → weaker" arc. |

### TIER 3: Recommended (strengthens paper)

| # | Issue | Reviewers | Effort | Fix |
|---|-------|-----------|--------|-----|
| 11 | **Appendices C+D report negative results** | R1, R2 | 30 min | Either cut (saves space) or consolidate into "Validation Limits" subsection in Discussion |
| 12 | **Transfer entropy inconsistency** | R4 | 30 min | Add paragraph reconciling TE reverse-direction finding (SMB→HML z=5.37) with linearity claim |
| 13 | **Four-model diagnostic: low permutations (200)** | R4 | 20 min | Acknowledge statistical power limitation; qualify "predominantly linear" as conditional on primary seed |
| 14 | **MS-VAR BIC tension** | R4 | 20 min | Frame BIC rejection as evidence that effect is real but too small to justify model complexity |
| 15 | **Reproducibility: seed 28 convention** | R4 | 20 min | Add fixed seed list to code repo; disclose that exact replication requires provided seeds |
| 16 | **Related Work: 1.8pp with 5 repeated "Gap:" markers** | R1 | 45 min | Consolidate to single 0.75-page narrative; remove subsection headers and "Gap:" rhetoric |
| 17 | **Deleveraging mechanism: speculative, weakly supported** | R2 | 20 min | Explicitly label as "speculative hypothesis" throughout; demote from "interpretation" |
| 18 | **HAC bandwidth=1 justification** | R3 | 15 min | Report sensitivity range [0.018, 0.041] in main text rather than just BW=1 result |
| 19 | **Trading strategy: remove or reframe** | R2, R1 | 20 min | Either cut entirely (saves space) or frame as deliberate negative result |
| 20 | **Regime boundary 7.4% exclusion** | R3 | 15 min | Add limitation note about non-standard residuals from boundary cleaning |

---

## Estimated Total Revision Effort

- **Tier 1 (blocking):** ~4.5 hours
- **Tier 2 (strongly recommended):** ~3 hours
- **Tier 3 (strengthening):** ~4.5 hours
- **Total for comprehensive revision:** ~12 hours

---

## Reviewer Verdict Summary

| Reviewer | Persona | Verdict | Key Demand |
|----------|---------|---------|------------|
| R1 (Structure) | Program committee senior | Conditional Accept after major revision | Fix contribution mismatch; rewrite abstract; compress Related Work |
| R2 (Finance) | Quant finance expert | Reject → Conditional Accept if revised | Remove VaR contribution; reframe economic magnitude honestly |
| R3 (Econometrics) | Time series statistician | Reject → Conditional Accept if revised | Fix dual p-value reporting; address multiple testing; disclose K=3 look-ahead |
| R4 (ML) | ML/causality researcher | Conditional Accept after revision | Qualify linearity claim; address reproducibility; reconcile TE inconsistency |

**Overall:** The paper has genuine methodological rigor and an interesting empirical finding. The core problem is **overselling** — framing modest, fragile results as robust confirmatory evidence. A revision that leads with the structural break, honestly ranges the OOS evidence, drops the VaR contribution, and compresses the paper to 10 pages would be competitive at ICAIF.
