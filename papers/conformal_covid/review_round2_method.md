# MethodCritic Round 2 Review

**Paper**: Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study
**File**: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/uai_2026/main.tex`
**Date**: 2026-02-20

## Executive Summary

The paper is substantially improved from Round 1. No remaining FATAL issues. One MAJOR internal consistency contradiction about Stack Overflow's inclusion/exclusion from n=16 must be resolved before submission. The other targeted items (SHAP footnote, retraining p-value, COVID-era row) range from acceptable to needing minor clarification.

---

## Issue 1: Stack Overflow Inclusion/Exclusion Contradiction [MAJOR]

**The problem.** Three statements in the paper directly contradict each other regarding whether Stack Overflow is included in the n=16 primary endpoint:

- **Line 284** (Table 2 footnote): "n=16 includes it [Stack Overflow] in the multiclass primary set"
- **Line 533** (Table 5 footnote): "Stack Overflow (3 classes) exhibits near-binary ceiling effect; **excluded from n=16** multiclass primary endpoint"
- **Line 89** (Methods): "Excluding binary tasks yields 8 external multiclass datasets, and together with the 8 multiclass SALT tasks this gives the primary endpoint of n=16"

If Stack Overflow is one of the 9 external datasets in Table 5, and n=16 = 8 SALT + 8 external, then Stack Overflow must be excluded to get 8 external. But line 284 says it is included.

Additionally, the figure caption (line 296) says "8 external domains" which is consistent with Stack Overflow being excluded. And line 599 says "Including Stack Overflow (3 classes, near-binary ceiling) to make n=17 weakens correlation to rho=0.654," which confirms n=16 EXCLUDES Stack Overflow.

**Verdict**: Line 284 is the error. The footnote "$^\ddagger$n=15 excludes Stack Overflow (near-binary ceiling effect, 3 classes); n=16 includes it in the multiclass primary set" is backwards -- n=16 should be the version WITHOUT Stack Overflow (the primary), and n=17 would be with it.

**Required fix**: Rewrite the footnote to: "$^\ddagger$n=15 excludes Stack Overflow (3 classes, near-binary ceiling); the primary n=16 endpoint also excludes Stack Overflow (binary ceiling effect). Including Stack Overflow yields n=17 with reduced correlation ($\rho = 0.654$; see Appendix)." Or simply remove the n=15/n=16 distinction if it no longer serves a purpose, and clarify that Stack Overflow is excluded from the primary endpoint throughout.

---

## Issue 2: COVID-era n=9 Row Conceptual Validity [MODERATE]

**Line 270/282**: The "COVID-era" grouping (n=9) is defined as "8 SALT tasks plus Stack Overflow (temporal shift from 2015-2018 split), the one external dataset sharing a COVID-adjacent temporal structure."

**Problems**:
1. Stack Overflow's temporal shift (2015-2018) is not COVID-era by any definition. The footnote calls it "COVID-adjacent temporal structure" which is a stretch -- a 2015-2018 split has nothing to do with COVID-19.
2. The paper itself says Stack Overflow has a "near-binary ceiling effect (3 classes)" which makes it diagnostically uninformative -- so adding it to the COVID-era group adds noise, not signal.
3. The rho improves from 0.833 to 0.883 by adding this one point, which looks like cherry-picking a favorable inclusion criterion.

**Recommendation**: Either (a) remove the COVID-era row entirely since it serves no clear purpose when the SALT-only (n=8) and full multiclass (n=16) rows already exist, or (b) rename it to something honest like "SALT + temporal-shift external" and add a note that this grouping is post-hoc. The current "COVID-era" label is misleading.

---

## Issue 3: SHAP Assumption Footnote in A1 [ACCEPTABLE with minor suggestion]

**Line 152**: The footnote on assumption (A1) reads:

> "Assumption (A1) posits additivity in probability space. In practice, SHAP values for tree ensembles are computed in log-odds space; the additive decomposition is therefore an approximation. We use (A1) as an idealised model to derive the monotonicity result, treating SHAP-derived C as an empirical proxy for the theoretical concentration parameter."

This is adequate for a UAI audience. It honestly discloses the gap (log-odds vs. probability space) and frames C as an empirical proxy rather than claiming exact correspondence. No further action needed, though one could optionally add a sentence noting that the approximation tends to be reasonable when probabilities are not extreme (i.e., away from 0/1 where log-odds diverge).

---

## Issue 4: Retraining p=0.04 in Abstract [ACCEPTABLE]

**Line 45**: The abstract says "+19 pp, p = 0.04 (unadjusted)."
**Line 349**: The body says "p=0.04, unadjusted; Holm-corrected over 3 tasks: p=0.12."

The abstract's "(unadjusted)" qualifier is honest and sufficient. A reader seeing "unadjusted" knows to look for the adjusted value in the body, where Holm p=0.12 is transparently reported. The abstract also frames the entire framework as "exploratory" and uses "suggest that" and "may partially recover" -- appropriately hedged language. No further action needed.

---

## Issue 5: Cross-Reference Consistency Check [MINOR issues found]

1. **Domain count in abstract vs. body**: The abstract (line 45) says "External validation across 9 held-out domains." But if total domains = 9 (line 89: "9 domains") and SALT is one of those 9, then there are only 8 held-out domains. The abstract should say "8 held-out domains" or "9 domains including SALT." This is a minor but sloppy inconsistency.

   However, Table 5 lists 9 external datasets (Covertype, Shuttle, Avila, PAMAP2, KDDCup99, Pendigits, Satimage, Gas Sensor, Stack Overflow). If 2 binary datasets were excluded from the 11 additional datasets to get 9 remaining, and Stack Overflow is then excluded from the n=16 primary for ceiling-effect reasons, that gives 8 external in the primary. But if there are 9 external *domains* total (some excluded from primary but still "validated"), then "9 held-out domains" could be correct for the broader validation claim. This needs clarification.

2. **n=11 row** (line 271): The table shows "Multiclass (4 dom.)" at n=11. That's 8 SALT (1 domain) + 3 external datasets from 3 domains = 4 domains total. This seems to correspond to an earlier version of the analysis. The footnote (line 283) explains it uses single-seed external values and gives the multi-seed-consistent rho=0.818. This is fine as historical context but adds complexity. Consider whether this row is still needed given the n=16 primary.

---

## Issue 6: Other Submission-Blocking Items [None Found]

- Abstract: No duplicate text. Length is long but within UAI limits.
- Theorem: Bounds verified, assumptions disclosed, footnote on SHAP approximation present.
- Table 1 footnotes: Categories (SEV/ROB) defined, high-variance asterisks explained.
- Figure references: All figures referenced in text.
- Contribution 6 (line 67) references "9 held-out domains" -- same domain-count issue as above.
- The paper correctly scopes claims to gradient-boosted models throughout.

---

## Summary of Required Actions

| # | Issue | Severity | Action |
|---|-------|----------|--------|
| 1 | Stack Overflow in/out of n=16 contradiction | MAJOR | Fix Table 2 footnote (line 284) to match rest of paper |
| 2 | COVID-era n=9 row is conceptually misleading | MODERATE | Remove row or rename honestly |
| 3 | SHAP footnote | OK | No action needed |
| 4 | Retraining p=0.04 abstract | OK | No action needed |
| 5a | "9 held-out domains" in abstract should be 8 | MINOR | Fix count |
| 5b | n=11 row in Table 2 adds complexity | MINOR | Consider removing |

## Verdict

**MINOR REVISION REQUIRED** -- The Stack Overflow contradiction (Issue 1) is the only item that must be fixed before submission. It is a clear factual inconsistency that any careful reviewer will catch. Issues 2 and 5a are recommended fixes. The rest is acceptable.
