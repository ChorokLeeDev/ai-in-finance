
# TECHNICAL REVIEW SYNTHESIS
## "Diagnosing Conformal Prediction Failures Under Distribution Shift" | UAI 2026 | 2026-02-20

Four expert agents reviewed this paper: Method Critic, Literature Auditor, Statistician, Brutal Reviewer.

---

## EXECUTIVE SUMMARY

The paper proposes SHAP concentration as a pre-deployment diagnostic for conformal prediction vulnerability. The core empirical contribution (rho=0.853, n=16, 9 domains) is verified correct by independent recomputation. However, a copy-paste error duplicates the abstract, the primary result (n=16) lacks a corresponding figure, Theorem 1's numerical verification is mathematically wrong, and Table 2 contains unexplained rows. None of these invalidate the contribution, but several would trigger rejection if unfixed.

---

## UNANIMOUS FINDINGS (all 4 reviewers agree)

### 1. Abstract duplication — FATAL
All four reviewers independently flagged lines 45-46 of `main.tex` as containing a duplicated paragraph. This renders the compiled PDF with a double-length abstract. **Desk-rejection risk.** Fix: delete one copy (30-second edit).

### 2. COVID-era n=9 row in Table 2/3 is undefined
All four reviewers flagged that the "COVID-era n=9" row composition is never explained. No text, footnote, or appendix defines which 9 tasks comprise this group. Fix: add footnote or remove row (text edit).

### 3. Covertype drop inconsistency (82 pp vs 81.8 pp)
Three reviewers (Statistician, Brutal, Method Critic via class counts) flagged rounding inconsistencies between abstract and body. Fix: standardize to one representation (text edit).

---

## HIGH-PRIORITY FINDINGS (2-3 reviewers agree)

### 4. No figure for primary n=16 result — HIGH
**Brutal + Method Critic**: The only scatter plot shows n=11 (filename: `figure_n11_correlation.pdf`, caption reports rho=0.833). The primary endpoint (n=16, rho=0.853) has no visualization. A file `figure_n12_correlation.pdf` exists on disk but is not included. **Requires generating/including a new figure.**

### 5. Theorem 1 numerical verification is wrong — HIGH
**Method Critic**: Independent recomputation of Eq(5) with eps=0, h_bar=1/K yields bounds (0.518, 0.545, 0.474, 0.261, 0.296) that differ substantially from claimed values (0.785, 0.841, 0.990, 0.806, 0.825). The theorem proof logic is sound and the correct bounds are still valid, but the appendix numbers are wrong. **Requires recomputation.**

### 6. Table 2 n=11 row uses single-seed values (superseded) — HIGH
**Statistician + Brutal**: The n=11 row reports rho=0.909, but Appendix G discloses that multi-seed recalculation yields rho=0.818. The table presents the stale single-seed value alongside multi-seed results without disclosure. Fix: update to multi-seed or add footnote (text edit + possible recomputation).

### 7. Table 2 n=15 row never explained — MEDIUM-HIGH
**Statistician + Brutal**: The "Multiclass (8 dom.), n=15" row is never defined. Which task is excluded from n=16? Fix: add footnote (text edit).

### 8. Retraining p-value needs multiplicity correction — HIGH
**Method Critic**: The "+19pp, p=0.04" claim tested 3 tasks (family), making Holm-adjusted p=0.12. Fix: reframe as exploratory or report adjusted p (text edit).

### 9. Assumption A1 gap (SHAP output space vs probability space) — MEDIUM-HIGH
**Method Critic + Brutal**: SHAP decomposes in margin/log-odds space; A1 operates in probability space. The gap should be disclosed. Fix: add remark after A1 (text edit).

---

## UNIQUE FINDINGS BY REVIEWER

### Method Critic only:
- Partial correlation non-significance (rho_partial=0.629, p=0.131 at n=8) underplays that within-SALT evidence cannot disentangle concentration from cardinality (MEDIUM)
- Holm-Bonferroni for metric selection exactly at p=0.050 boundary (MEDIUM)
- Class count discrepancies between Table 1 and data files (MEDIUM)
- No prediction set sizes reported for standard APS in Table 1 (MEDIUM)

### Literature Auditor only:
- `angelopoulos2024conformal` lists venue as JASA; actual venue is ICLR 2024 (must fix, text edit)
- Missing Lei et al. (2018) split CP foundational citation (recommended, text edit)
- Missing UCI repository citation for external datasets (recommended, text edit)
- RelBench BibTeX missing arXiv ID (text edit)

### Statistician only:
- RAPS 10-seed values cited in main text without noting they differ from Table 1's 50-seed values (MODERATE)
- i-shippoint classified as "At-risk" in Table A.5 but "ROB" in Table 1 (mean vs median confusion) (MODERATE)
- "7/9 deterministic" counting ambiguous because Covertype is also 10/10 deterministic (MINOR)

### Brutal Reviewer only:
- "8 tasks with severe feature turnover" is factually wrong -- 2 of 8 have stable features (HIGH, text edit)
- Stack Overflow absent from framework validation table despite being discussed in text (MEDIUM-HIGH)
- "7 types of natural distribution shift" claimed without enumeration (MEDIUM, text edit)
- "Asymmetric evidence" undefined jargon in abstract (MEDIUM, text edit)
- No external dataset description table (MEDIUM, requires new table)
- Section 6 structure buries cross-domain validation as afterthought (LOW-MEDIUM)
- Why boosting works but RF does not belongs in Discussion, not appendix (LOW-MEDIUM)

---

## CROSS-AGENT INSIGHTS

All four reviewers independently verified that the core statistical results (rho=0.853, threshold precision/recall, KS statistics) are arithmetically correct and internally consistent. The paper's empirical contribution is solid. The problems are concentrated in (a) presentation errors (abstract duplication, missing figure, unexplained table rows), (b) one mathematical error in the appendix (theorem bounds), and (c) framing issues (retraining p-value, A1 assumption gap). The Method Critic and Brutal Reviewer converge on the theorem and retraining issues; the Statistician and Brutal Reviewer converge on Table 2 unexplained rows. The Literature Auditor found no mischaracterizations of prior work -- a strong signal of intellectual honesty.

---

## REQUIRED ACTIONS (blocking)

1. **Delete duplicated abstract paragraph** (line 46 of main.tex) -- text edit, 30 seconds
2. **Generate and include n=16 scatter plot figure** -- requires computation/figure generation
3. **Recompute Theorem 1 bounds in Appendix S7** -- requires running corrected code
4. **Define or remove COVID-era n=9 and n=15 rows in Table 2** -- text edit
5. **Update n=11 row to multi-seed value (rho=0.818) or add footnote** -- text edit
6. **Fix `angelopoulos2024conformal` venue from JASA to ICLR 2024** -- BibTeX edit
7. **Correct "8 tasks with severe feature turnover" to "8 SALT tasks"** -- text edit

## SUGGESTED IMPROVEMENTS (non-blocking)

- Add Holm-adjusted p-value for retraining claim or reframe as exploratory
- Add remark about A1 operating in probability space vs SHAP in output space
- Add prediction set sizes to Table 1 or companion table
- Add Lei et al. (2018) and UCI repository citations
- Define "asymmetric evidence" and "7 types of distribution shift" or remove claims
- Reconcile i-shippoint status (ROB vs At-risk) with a single clear statement
- Add one-paragraph external dataset description table
- Move cross-domain validation from Section 6 into Results (Section 5)
- Standardize task naming (s-shipcond vs sales-shipcond)
- Standardize Covertype drop to 81.8 pp throughout

---

## EFFORT CLASSIFICATION

**Simple text edits (no computation):** Items 1, 4, 5, 6, 7, and most suggested improvements. Roughly 15 items fixable in a single editing pass (~2-3 hours).

**Requires new computation/figures:** Items 2 (n=16 scatter plot) and 3 (theorem bound recomputation). The scatter plot data exists (`figure_n12_correlation.pdf` on disk); the theorem recomputation requires re-running the bound calculation with the correct formula. Estimate: 1-2 hours.

---

## VERDICT: CONDITIONAL RECOMMEND

**Reason:** The core scientific contribution is verified correct and novel. No reviewer questioned the validity of the primary correlation (rho=0.853) or the diagnostic's utility. However, the abstract duplication alone would cause desk-rejection, and the missing n=16 figure and incorrect theorem bounds would likely cause reviewer rejection. All blocking issues are fixable in one revision pass (estimated 4-6 hours total). After fixing the 7 required actions, this paper is a solid UAI contribution.
