# MethodCritic R10 Verification Review

**Paper**: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/uai_2026/main.tex`
**Focus**: Verify R9 fixes (Stack Overflow concentration correction, "datasets/benchmarks" terminology)
**Date**: 2026-02-20

---

## 1. Stack Overflow Concentration Fix: 7.4% -> 48.9%

### Locations checked

| Line | Context | Value | Consistent? |
|------|---------|-------|-------------|
| 520  | Table 5 (framework validation) row | `48.9` | YES |
| 531  | Table 5 footnote ($^\P$) | `C=48.9\%` | YES |
| 520  | Table 5 Step 2 classification | VULN | YES (48.9% > 40% threshold) |
| 520  | Table 5 Actual classification | ROB | YES (drop = -7.0 pp, not at-risk) |

**No residual 7.4% references found anywhere in the paper.** The old value has been fully purged.

### Stack Overflow narrative consistency across all sections

| Section | Line(s) | Description | Consistent? |
|---------|---------|-------------|-------------|
| Abstract | 45 | "KDDCup99/Stack Overflow are boundary cases" | YES -- both are correctly called boundary cases |
| Contribution 6 | 67 | "Stack Overflow (near-binary ceiling) are boundary cases" | YES |
| Section 3.1 | 89 | "Excluding Stack~Overflow (near-binary ceiling effect) yields 8 external multiclass datasets" | YES |
| Table 2 footnote | 282 | "Stack~Overflow (3 classes, near-binary ceiling effect) is excluded from all multiclass endpoints" | YES |
| Section 5.5 | 351 | "Binary tasks exhibit the expected ceiling effect and are excluded" | MINOR ISSUE (see below) |
| Section 6 | 378 | "Stack Overflow (3 classes) exhibits the near-binary ceiling effect" | YES |
| Table 5 footnote | 531 | Full explanation: K=3, C=48.9%, VULN predicted, ROB actual, -7.0 pp, excluded from n=16 | YES |
| App. F (ICC) | 597 | "Including Stack Overflow (3 classes, near-binary ceiling) to make n=17 weakens correlation to rho=0.654" | YES |

**Verdict on Stack Overflow fix**: CLEAN. The correction from 7.4% to 48.9% is fully propagated. The classification change from ROB (predicted) to VULN (predicted) / ROB (actual) is correct: 48.9% > 40% threshold means the diagnostic predicts VULN, but the actual outcome is ROB due to the near-binary ceiling effect. This is logically coherent and the footnote explains it well.

---

## 2. "9 non-supply-chain domains" -> "datasets/benchmarks" Fix

### Locations checked

| Line | Current text | Correct? |
|------|-------------|----------|
| 45 (Abstract) | "9 non-supply-chain datasets" | YES -- uses "datasets" |
| 67 (Contribution 6) | "9 external non-supply-chain datasets" | YES -- uses "datasets" |
| 89 (Section 3.1) | "9 non-supply-chain benchmarks" | YES -- uses "benchmarks" |
| 351 (Section 5.5) | "9 non-supply-chain benchmarks" | YES -- uses "benchmarks" |

**No remaining instances of "9 non-supply-chain domains" found.** The fix correctly distinguishes between "datasets" (concrete collections) and "domains" (conceptual categories, used when counting to 9 domains = 1 SALT + 8 external).

---

## 3. Domain/Dataset Counting Consistency

The paper uses two distinct counting systems:
- **Datasets**: 9 external datasets (Covertype, Shuttle, Avila, PAMAP2, KDDCup99, Pendigits, Satimage, Gas Sensor, Stack Overflow)
- **Domains**: 9 domains for the n=16 primary endpoint = 1 (SALT) + 8 (external multiclass, excluding Stack Overflow)

All instances verified:
- n=16 = 8 SALT multiclass + 8 external multiclass (Stack Overflow excluded) -- **CORRECT**
- 9 domains = 1 SALT + 8 external multiclass domains -- **CORRECT**
- n=17 = n=16 + Stack Overflow -- **CORRECT** (line 597)
- n=19 "Combined (11 dom.)" in Table 2 -- presumably includes binary tasks; not part of primary endpoint

---

## 4. Additional Numerical Consistency Checks

### Cross-referencing Stack Overflow treatment

- **Threshold sensitivity table (Table 4, line 470)**: "16 multiclass tasks (8 SALT + 8 external)" -- CORRECT, Stack Overflow excluded.
- **Figure 3 caption (line 294)**: "16 multiclass tasks in 9 domains (dark circles: 8 SALT supply-chain tasks; orange triangles: 8 external domains, 10-seed means)" -- CORRECT, Stack Overflow would not appear as an orange triangle.

### Deterministic/near-deterministic count

Contribution 6 claims "7/9 datasets (6 deterministic, 1 near-deterministic)":
- 6 deterministic (10/10): Covertype, Avila, PAMAP2, Pendigits, Satimage, Gas Sensor -- CORRECT
- 1 near-deterministic (9/10): Shuttle -- CORRECT
- 2 boundary: KDDCup99, Stack Overflow -- CORRECT

Section 6 (line 378) says "7/9 deterministic or near-deterministic outcomes (6 at 10/10, Shuttle at 9/10)" -- **CONSISTENT**.

---

## 5. Minor Issue Found

**Section 5.5 (line 351)**: "Binary tasks exhibit the expected ceiling effect and are excluded from the multiclass primary endpoint."

This sentence says "Binary tasks" but Stack Overflow is K=3 (ternary), classified as "near-binary ceiling." A strict reader could argue Stack Overflow is not binary. However, the paper has already established the "near-binary ceiling effect" concept in Sections 1, 3, and the appendix, so this is likely understood in context. The sentence could be more precise by saying "Binary and near-binary ceiling-effect tasks" but this is cosmetic.

**Severity**: MINOR (cosmetic precision, not a substantive inconsistency).

---

## 6. Verdict

Both R9 fixes are **correctly and consistently applied** throughout the paper:

1. **Stack Overflow concentration**: All references show C=48.9%, classification = VULN (predicted) / ROB (actual), with the near-binary ceiling explanation. No stale 7.4% values remain. The logical chain is sound: high concentration triggers VULN prediction, but K=3 creates a ceiling effect that prevents the predicted failure from materializing.

2. **"datasets/benchmarks" terminology**: All 4 instances of "9 non-supply-chain domains" have been replaced with either "datasets" or "benchmarks" as appropriate. The word "domains" is now reserved for the conceptual counting system (9 domains = 1 SALT + 8 external multiclass).

**Status**: CLEAN -- no action required beyond the optional cosmetic fix in Section 5.5.
