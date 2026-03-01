# ICAIF Pre-Submission Audit Report

**Paper:** Regime-Dependent Predictive Structure Between Equity Factors
**Date:** February 25, 2026
**Compiled PDF:** 19 pages (sigconf, with bibliography)

---

## Step 1: LaTeX Compilation Sanity Check

### Compilation Status
The paper compiles successfully under pdflatex + bibtex with ACM sigconf format. All citations resolve. All `\ref` targets have matching `\label` definitions. No undefined references ("??") in the output.

### CRITICAL — Duplicate Labels (Must Fix)

Two figures are defined twice — once in the main body and once in Appendix H (Extended Robustness). LaTeX silently resolves `\ref{}` to the *last* definition, meaning main-body cross-references point to the appendix versions:

| Label | Main body (line) | Appendix (line) |
|---|---|---|
| `fig:rolling` | 903 | 1487 |
| `fig:lag_sensitivity` | 925 | 1507 |

**Fix:** Remove the duplicate figures from Appendix H (they appear to be copy-paste artifacts — the appendix text around them is also duplicated from the main body), OR rename to `fig:rolling_app` / `fig:lag_sensitivity_app`.

### Minor — Missing `\Description{}` Tags

ACM accessibility requires `\Description{}` for all figures. Five figures in the appendices lack them (lines 1119, 1432, 1490, 1510, 1574). The main-body figures are fine.

### Minor — Unused Labels (7)

`sec:limitations`, `sec:trading`, `tab:algo_params`, `tab:events`, `tab:frozen_events`, `tab:trading`, `tab:var` are defined but never `\ref`'d. Non-fatal but suggests incomplete refactoring.

### Minor — Unused Bibliography Entries (15)

15 of 87 bib entries are never cited (e.g., `christoffersen1998evaluating`, `bollerslev1986generalized`, `kupiec1995techniques`). Consider removing to reduce clutter, especially since several are VaR-related and might confuse reviewers about what's actually being cited.

### Minor — BibTeX Warning

`shu2025dynamic`: page numbers missing in both `pages` and `numpages` fields.

### Overfull Boxes

4 overfull hboxes (widths 14–26pt). The worst is Table `tab:regimes` at line 535 (26.4pt overfull). Consider reducing `\tabcolsep` or switching to `\scriptsize` for that table.

### TOST and Barnett Citations: ✅ Clean

- `schuirmann1987comparison` (TOST) — present and correctly cited at line 654
- `barnett2009granger` (transfer entropy equivalence) — present and correctly cited at line 1593

### `app:optima` Appendix: ✅ Clean

Label defined at line 1630, referenced at lines 397 and 477. No issues.

---

## Step 2: Page Budget Check

### Page Structure (compiled sigconf)

| Section | Pages | Start Page |
|---|---|---|
| Abstract + Title | 1 | 1 |
| Introduction (§1) | 2 | 2 |
| Related Work (§2) | 1.5 | 3 |
| Methodology (§3) | 2 | 4 |
| Results (§4) | 2 | 6 |
| Discussion (§5) | 3 | 8 |
| Conclusion (§6) | 1 | 11 |
| **Main body total** | **~11 pages** | **Pages 1–11** |
| Appendices (A–I) | 6 | 12–17 |
| References | 2 | 17–19 |
| **Total** | **19 pages** | |

### Verdict: ⚠️ OVER LIMIT

ICAIF sigconf typically allows **10 pages** for the main body (excluding references and appendices). The current main body runs to **page 11** — approximately 1 page over.

**Recommended cuts to reach 10 pages:**

1. **Remove duplicate figures from §5.5 Robustness** (fig:rolling and fig:lag_sensitivity already appear in the appendix — removing them from the main body saves ~0.5 page and fixes the duplicate label bug simultaneously)
2. **Trim §5.1 Frozen OOS** — the convention note (lines 778–790) explaining decimal vs. percentage units is ~15 lines of detail that could move to a footnote or appendix
3. **Condense §5.5 Robustness** — the trivariate MKT-RF discussion (lines 928–942) and K-sensitivity (lines 944–956) could each be reduced to 2–3 sentences with "see Appendix" pointers

These three changes together should save ~1–1.5 pages.

---

## Step 3: Adversarial Single-Reviewer Pass

### The Single Strongest Objection

**A hostile R2 (finance domain expert) would argue:** *"The paper's primary OOS result is fragile — it depends on a unit convention choice, and the more conservative convention fails the permutation test."*

Here's how that objection unfolds:

---

#### Objection 1: The p=0.063 Decimal-Unit Result Undermines the Primary Finding

**Where it lives:** Abstract footnote (line 51), §3.5 (line 511), §5.1 (lines 778–790)

**The reviewer's attack:** "The authors report two permutation p-values for the same frozen OOS Elevated test — p=0.022 (percentage units, n=953, 50K shuffles) and p=0.063 (decimal units, n=836, 10K shuffles). They designate the passing result as 'primary' because 'the frozen HMM was trained on percentage-unit data.' But this is a researcher degree of freedom: the choice of which pipeline to call 'primary' was made *after* seeing both results. Under the more conservative decimal-unit specification, the permutation test fails at 5%. A result that crosses and recrosses the significance boundary depending on a scale convention is not robust — it's borderline."

**Why this is dangerous:** The abstract and §1 prominently feature p=0.022, creating an impression of clear significance. A careful reader will find the footnote disclosing p=0.063, and the gap between headline and disclosure will read as spin. ICAIF reviewers who work in quantitative finance will be sensitive to this.

**Concrete fix:**
- In the abstract, change "permutation p = 0.022" to "permutation p = 0.022 (percentage-unit primary) to 0.063 (decimal-unit robustness bound)"
- In §3.5 (line 511), move the decimal-unit result from a footnote into the main text so both are equally visible
- Add one sentence: "We acknowledge this range straddles the conventional 5% threshold; readers should interpret the permutation evidence as suggestive rather than definitive."

---

#### Objection 2: The VaR Application Is Disconnected from the Granger Finding

**Where it lives:** §4.4 (line 175–176), §5.4 (line 882), Appendix G

**The reviewer's attack:** "The paper presents a 'hybrid VaR detector' as its practical contribution, but then acknowledges (line 882) that 'the improvement reflects primarily the volatility-override mechanism rather than the Granger link per se.' Even more damaging: the VaR application uses seed 42 (sensitivity fit), for which the frozen OOS Granger result is *null* (Elevated p=0.466, per Table in Appendix). So the paper's headline practical contribution is tested on a seed where the paper's headline statistical finding does not hold. This is a logical disconnect: the theory says Granger causality enables better risk monitoring, but the risk monitoring works via a mechanism (volatility override) that doesn't require Granger causality at all."

**Why this is dangerous:** ICAIF is a venue that values practical relevance. A VaR contribution that's disconnected from the paper's core finding reads as padding.

**Concrete fix:**
- In §1.2 Contribution 4 (line 200–204), strengthen the framing: "The VaR improvement is driven by the regime-adaptive architecture (volatility override) rather than the Granger predictor itself; this is a proof-of-concept for regime-conditional monitoring, not evidence that the Granger link improves tail forecasts."
- In the abstract (line 175–176), either remove the VaR result or explicitly caveat: "A hybrid VaR model achieves 5.60% violation rate; the improvement is driven by the volatility-override mechanism rather than the Granger signal."
- Consider demoting the VaR from "Contribution 4" to an appendix-only discussion.

---

#### Secondary Objections (lower priority, but expect them)

**3. The "2 of 3 local optima" framing oversells robustness.** The 41 seeds in Optima A+B converge to *identical* parameters within each optimum, so effective robustness is 2 independent HMM solutions, not 41 runs. The paper discloses this (line 806) but the abstract says "2 of 3 local optima" without this caveat.

**4. The in-sample result is exclusively pre-GFC (p=0.73 post-2008).** A reviewer could argue the paper's "primary finding" is a historical artifact of pre-Dodd-Frank market structure, and the OOS "corroboration" in a different regime (Elevated vs. Normal) with a different effect size (ΔR²=0.73% vs 2.06%) isn't actually corroborating the same phenomenon.

**5. 9 appendices is heavy for ICAIF.** Some appendix content (Appendix C: Early Warning, with 0/3 detections; Appendix D: Events, with 2/6 matches) arguably weakens rather than strengthens the paper. A hostile reviewer will cite these as evidence the method doesn't generalize.

---

## Step 4: Go / No-Go Assessment

### Verdict: **CONDITIONAL GO** — fixable issues, none blocking if addressed

**Blocking issues (must fix before submission):**

1. ☐ **Duplicate labels** (fig:rolling, fig:lag_sensitivity) — will cause cross-reference errors. Fix by removing duplicates from Appendix H. ~5 min.

2. ☐ **Page limit** — main body is ~11 pages, needs to be ≤10. Remove duplicate figures from §5.5, condense §5.1 convention note and §5.5 trivariate/K-sensitivity into tighter prose. ~1 hour.

3. ☐ **Missing \Description{} tags** on 5 appendix figures — ACM will reject without them. ~10 min.

**Strongly recommended (reduces reviewer attack surface):**

4. ☐ **Promote p=0.063 disclosure** — move from footnote to main text in abstract and §3.5. Present the range [0.022, 0.063] rather than cherry-picking the favorable endpoint.

5. ☐ **Clarify VaR disconnect** — either demote to appendix-only or add explicit caveat that VaR improvement is not driven by the Granger signal.

6. ☐ **Remove unused bib entries** (15 entries) — reduces noise and avoids reviewer questions.

7. ☐ **Fix BibTeX warning** for `shu2025dynamic` (missing page numbers).

**Optional improvements:**

8. ☐ Consider cutting Appendix C (Early Warning) and Appendix D (Events) — they report negative/weak results that a hostile reviewer will weaponize.

9. ☐ Clean up unused \label definitions.

10. ☐ Fix 4 overfull hboxes.
