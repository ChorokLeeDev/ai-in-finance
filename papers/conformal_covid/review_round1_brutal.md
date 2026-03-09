# Brutal Review: "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"

**Reviewer perspective**: Most demanding reviewer at UAI 2026.
**Date**: 2026-02-20
**File reviewed**: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/uai_2026/main.tex`

---

## CRITICAL BUG (FATAL — MUST FIX BEFORE SUBMISSION)

### 1. Abstract is duplicated — entire second half appears twice (lines 44–46)

The abstract ends with a **word-for-word repeat** of the paragraph beginning "seeds), we show that, for gradient-boosted models...". Line 45 and line 46 of the `.tex` file are essentially the same sentence block, printed twice. The only difference between the two copies is the final sentence of the decision-framework bullet:

- Line 45 (copy 1): `consider quarterly retraining if vulnerable (+19~pp recovery, $p = 0.04$); defer routine retraining for lower-risk tasks.`
- Line 46 (copy 2): `suggest that quarterly retraining may partially recover coverage when vulnerable (+19~pp, $p = 0.04$); and defer routine retraining for lower-risk tasks.`

The compiled PDF therefore prints the full second half of the abstract twice. This is a disqualifying submission error. The paragraph starting "seeds), we show..." appears twice in succession. One copy must be deleted. The surviving copy should be copy 2 (line 46), which uses hedged language consistent with the body text. The break between the two copies also produces a grammatically broken sentence: "...from 0.1\% to 77\% (all paired $p \leq 0.005$, 50\nseeds), we show..." — the line break mid-sentence between "50" and "seeds" will render as a space in PDF but reads oddly in source.

---

## HIGH-SEVERITY ISSUES

### 2. No figure exists for the primary (n=16, rho=0.853) result

The paper's claimed primary endpoint is the n=16 multiclass correlation (rho=0.853, p<0.001 across 9 domains). Yet **no figure in the paper shows these 16 points**. The only scatter plot (`fig:n11_correlation`, `results/figure_n11_correlation.pdf`) shows n=8 SALT tasks plus 3 additional points ("lighter points: 3 binary/cross-domain tasks"), for a total of 11 points, and its caption reports rho=0.833 — the SALT-only result.

A reviewer will immediately ask: where is the figure for your stated primary result? The figure label (`fig:n11_correlation`) and the file name (`figure_n11_correlation.pdf`) both embed "n11", which exposes that the primary result was upgraded from n=11 to n=16 without generating a corresponding figure. The file `results/figure_n12_correlation.pdf` exists on disk but is never included in the paper. A scatter plot showing all 16 multiclass tasks, labeled by domain, is essential for the primary result to be credible.

**Consequence**: reviewers will question whether rho=0.853 is even real given no visualization is provided.

### 3. Jackknife significance count is internally contradictory

- Section 5.3 body (line 252): "statistical significance is not maintained for **2 of 8** jackknife samples ($p = 0.052$) due to reduced power at $n=7$"
- Table 3 footnote (line 282): "LOO stability (SALT): $\rho \in [0.75, 0.96]$; **6/8 jackknife significant**"

These two statements are directly contradictory. "6/8 significant" means **2 of 8 are not significant**, which is consistent with the text. But the body text says "significance is not maintained for 2 of 8 jackknife samples" — which also means 6/8 are significant. So the numbers agree, but the framing is opposite: the body emphasizes the 2 failures while the table footnote emphasizes the 6 successes. More problematically, the body text gives one specific $p$-value ($p = 0.052$) as if it applies to all 2 non-significant samples, while the table uses a count. A skeptical reviewer will ask: are these two instances saying the same thing? Yes — but the framing discrepancy looks like an error and will prompt a query. The body text should be aligned with the table: "6/8 jackknife samples remain significant; 2 fall to $p = 0.052$ at $n=7$."

### 4. "Among 8 tasks with severe feature turnover" is factually wrong (line 252)

Section 5.3 opens with "Among 8 tasks with **severe feature turnover**, SHAP concentration correlates with coverage degradation." But not all 8 tasks have severe feature turnover. Table 2 explicitly shows:
- i-incoterms: Jaccard = 0.58 (stable features)
- s-office: SALESORG Jaccard = 0.61 (stable features)

The paper's own Section 5.2 says the key distinction is tasks with ID-based features (Jaccard ~0.02) vs entity-based features (Jaccard > 0.5). Two of the 8 SALT tasks have entity-based stable features, not severe feature turnover. The sentence should say "Among the 8 supply chain tasks" or "Across all 8 SALT tasks."

### 5. COVID-era group (n=9, rho=0.883) in Table 3 is never defined

Table 3 (Stratified Correlation Analysis) contains a row "COVID-era, n=9, rho=0.883, p=0.002" with no Kendall tau and no bootstrap CI. This group is never defined anywhere in the text — not in Section 3 (Methodology), not in the caption, not in any footnote. A reviewer will ask: what datasets constitute the "COVID-era" group? Is this SALT's 8 tasks plus 1 more? Which one? The missing Kendall tau for this row is also unexplained. Either define this group in the methodology or remove the row.

Similarly, "Multiclass (8 dom.) n=15, rho=0.882" is never explained. Where does n=15 come from? (It appears to be n=16 minus one dataset, but which one and why is never stated.) These phantom rows undermine the table's credibility.

### 6. Covertype drop is reported inconsistently (81.8 pp vs 82 pp)

- Abstract (both copies): "Covertype ($C=49.8\%$, **82~pp** drop)"
- Section 5.3 (line 260): "correctly flagging catastrophic failure (Covertype, **82~pp** drop)"
- Section 6.4 (line 351): "Covertype is the key external catastrophic case ($C=49.8\%$, **81.8~pp** drop, 10/10 seeds)"

The body of Section 6.4 gives 81.8 pp while the abstract and Section 5.3 say 82 pp. The abstract is likely the rounded value. But the inconsistency in the body itself (Section 6.4 vs Section 5.3) signals sloppiness to reviewers. Pick one representation throughout.

### 7. The "5 applicable tasks" for theorem verification mixes catastrophic and robust tasks without explanation

The theorem (Theorem 1) concerns **score inflation** causing coverage failure. The "conservative bound verification" in Appendix A.8 applies it to: s-shipcond, s-payterms, s-group (catastrophic), **i-plant** (ROB, 10.6% drop), and **s-incoterms** (ROB, 8.5% drop).

The paper never explains why i-plant and s-incoterms are "applicable" for this theorem when they experience only modest drops and are classified as robust. Table A.11 (KS tests) shows i-plant has a rightward score shift (mean_cal=0.73 → mean_test=0.86), consistent with the theorem — but i-incoterms and i-shippoint show leftward shifts (mean test < mean cal), which **contradicts** the theorem. The paper buries this in a footnote to Table A.11 ("Two robust tasks show test scores shifted leftward") without reconciling it with the "verified on all 5 applicable tasks" claim in the introduction (line 60). A reviewer will ask: why are only 5 of 8 tasks "applicable"? The criterion for applicability must be stated in the main text, not just the appendix.

### 8. Stack Overflow appears in abstract and introduction but is absent from Appendix Table A.5

The appendix decision-framework validation table (Table A.5, `tab:framework_validation`) lists 8 external multiclass datasets. Stack Overflow is discussed in the abstract, introduction (item 6), and Section 7 as a boundary case, but it **does not appear as a row in Table A.5**. The table caption says "all multiclass tasks" but Stack Overflow (3 classes, near-binary ceiling) is omitted. This either means Stack Overflow is binary (which contradicts "3 classes") or it was forgotten from the table. Either way, reviewers will notice the inconsistency between the textual claim of "9 external domains" and the 8-row external section of the table.

---

## MEDIUM-SEVERITY ISSUES

### 9. Abstract's "precision 0.71→1.00; recall 0.83" is misleading for "30–45% thresholds"

Introduction item 4 (line 64) states: "the framework shows a precision–recall trade-off across 30–45% thresholds (precision 0.71→1.00; recall 0.83)."

From Table A.2: at 30%, precision=0.71; at 45%, precision=1.00; at 40% and 35%, precision=0.83. The claim that recall=0.83 uniformly across 30–45% is correct (the single FN persists). But "precision 0.71→1.00 across 30–45%" obscures that at 35% and 40% the precision is 0.83 (not a smooth transition). More importantly, the table shows **25% threshold gives precision=0.63**, so starting the range at 30% is not natural. The framing should note that 35–45% all give precision ≥ 0.83, which is the actionable message.

### 10. The figure caption for fig:n11_correlation contradicts the primary result

The caption reads: "Spearman $\rho=0.833$, $p=0.010$ for 8 severe-shift tasks (dark). Lighter points: 3 binary/cross-domain tasks (ceiling effect)."

Problems:
(a) rho=0.833 is the SALT-only result (n=8), but the paper's headline result is rho=0.853 (n=16). A reader looking at the figure will see the wrong correlation coefficient.
(b) "3 binary/cross-domain tasks" is vague — which 3? Covertype is multiclass and catastrophic; are these 3 binary tasks from external datasets? This needs to be specified.
(c) The figure shows only 11 points while the primary claim covers 16 tasks. The disconnect between the figure and the primary result is the most serious presentation problem in the paper.

### 11. ACI section (Section 6.1) presents numbers without a table or citation to appendix

Section 6.1 presents specific ACI numbers ("+47.5 pp (s-shipcond), +71.8 pp (s-group), +60.0 pp (s-payterms); usable-set rates are 30%, 82%, and 40%") but provides no table and no appendix cross-reference. The numbers hang in narrative form with no data provenance. In a paper focused on statistical rigor, every empirical result should be traceable to a table.

### 12. "Partial transfer with asymmetric evidence" is unexplained jargon in the abstract

The abstract says "partial transfer with asymmetric evidence." The term "asymmetric evidence" is never defined. What does it mean? Apparently that evidence for high-concentration failure is stronger (1 clear case: Covertype) than evidence for low-concentration robustness (multiple cases, but most are low-concentration by construction). This is a meaningful claim but requires one sentence of definition on first use. As written it reads as buzzword hedging.

### 13. Section 3 conformal prediction setup is incomplete for reproduction

The 3-step setup in Section 3.2 is too minimal:
- Step 2 says "Calibrate conformal predictor on 50% of validation set" — but does not state what the other 50% is used for. (It's used as the evaluation set; this is stated later in Section 3.3 but should appear here.)
- The calibration quantile formula is in Appendix A.1 but not in the main text, making the methodology section incomplete without reading the appendix.
- There is no statement of the APS scoring rule in Section 3 — it appears only in Section 4 (Theory). A reader of Section 3 alone cannot understand what is being calibrated.

### 14. i-shippoint is called both "false positive" (ROB) and "at-risk" in different places

In Section 5.3 (line 256): "Two tasks with concentration >40% remain robust: sales-office...and **item-shippoint** (high model variance). These motivate the protective-factor check."

But in Table A.5 (framework validation): `i-shippoint: 48.8%, VULN, At-risk*` — with a dagger noting "Median drop 1.2%; most seeds maintain coverage."

And the Table 1 main results: `i-shippoint: ROB* (high model variance, classified by median)`.

So i-shippoint is simultaneously: a false positive in Section 5.3 (classified as ROB but predicted VULN), labeled "At-risk" in Table A.5 (implying it IS at-risk), and classified ROB by median in Table 1. A reviewer will spend time on this task trying to understand its true status. The paper needs a single clear statement: i-shippoint has C=48.8% (>40%), would be predicted VULN by the rule, but is classified ROB by median coverage, making it a false positive for the rule. The "At-risk*" label in Table A.5 is confusing because it implies the framework correctly identified it as at-risk, but then it's cited as a false positive in the text.

### 15. The "7 types of natural distribution shift" claim is unsupported

Section 5.3 (line 260): "8 external datasets spanning **7 types of natural distribution shift**." This claim is never substantiated. What are the 7 types? Domain shift? Temporal shift? Covariate shift? Label shift? Without enumeration, this is an empty claim that reviewers will flag.

### 16. Theorem 1 assumption A1 (additive decomposition) is not realistic for tree-based models

Assumption A1 states the predicted probability decomposes additively as $\hat{p}(y|x) = C \cdot g(y|x_1) + (1-C) \cdot h(y|x_{\setminus 1})$. This is the SHAP additive decomposition applied to probabilities — but SHAP is a post-hoc explanation, not how tree-based models actually compute probabilities. The softmax output of a tree ensemble is not additive in features. The paper acknowledges the theorem is "under explicit assumptions" but never discusses whether these assumptions hold approximately for LightGBM. A reviewer will note this gap. The theorem provides motivation for the metric but its assumptions are not empirically validated.

### 17. The retraining "+19 pp recovery" is misleadingly framed in the abstract

The abstract says "consider quarterly retraining if vulnerable (+19 pp recovery, $p=0.04$)." The body (Section 6.4) clarifies: this applies to sales-shipcond (+18.9 pp); sales-payterms gets partial recovery (+15.5 pp); and sales-group gets zero recovery (0.0%→2.2%). The abstract makes it sound like "+19 pp" is the expected recovery for any vulnerable task — but it's the result for ONE of THREE catastrophic tasks, and that task's RAPS analysis shows the underlying mechanism (concentrated single-feature dependence) is not solved by retraining either. The abstract should say "up to +19 pp for some vulnerable tasks."

### 18. External datasets are mentioned without a description table

Avila Bible, PAMAP2, Pendigits, Satimage, Gas Sensor Array Drift — these datasets appear in Section 5.3 and the appendix but are never described. What domain are they from? What kind of distribution shift do they have? What is the temporal structure? A reader has no idea why any of these datasets should experience distribution shift. A one-paragraph (or one-table) description of external datasets and their shift types is necessary.

### 19. Notation inconsistency: "validation data" used as both COVID-era and pre-COVID

Throughout the paper, "validation data" (February–July 2020) is used for:
(a) SHAP concentration computation (diagnostic)
(b) Calibration of the conformal predictor (50%)
(c) Evaluation of coverage (remaining 50%)

This triple use of "validation data" is confusing. The paper uses the calibration/evaluation split within validation, but a reader encountering "computed on validation data" for SHAP and "calibrated on validation data" for CP will struggle to track what data is used where. A diagram or clearer notation would help substantially.

---

## LOW-SEVERITY / STYLE ISSUES

### 20. "before deployment" tautology in opening sentence

Abstract line 40: "practitioners lack tools to anticipate which deployed models will fail **before deployment**." A model cannot be deployed and simultaneously be before deployment. The intended meaning is "before observing test-period outcomes." Rephrase.

### 21. Section 4 (Theory) appears before Section 5 (Results) but references Section 6 results

Section 4 ends (line 186) with an empirical claim: "catastrophic tasks show strong rightward conformity-score shifts (KS = 0.68–0.96)." This references results from Section 5 and Appendix A.8 — but the reader hasn't seen them yet. The theory section thus relies on results presented later. Either move this empirical validation sentence to Section 5, or add a forward reference ("as we demonstrate in Section 5.3 and Appendix A.8").

### 22. "sales-shipcond" vs "s-shipcond" naming inconsistency

The main table uses abbreviated names (s-shipcond, s-group, s-payterms, i-plant, etc.). The retraining section (line 347) switches to spelled-out forms: "sales-shipcond," "sales-payterms," "sales-group." These are used interchangeably throughout. Pick one convention.

### 23. The Holm-Bonferroni correction arithmetic is presented misleadingly

Appendix A.6 (line 593): "Under Holm–Bonferroni correction for this 5-test family, the adjusted $p$-value for top-1 concentration is $5 \times 0.010 = 0.050$, at the conventional significance boundary."

This is mathematically correct (for rank-1 in Holm), but presenting the result as "at the conventional significance boundary" when the uncorrected result is p=0.010 (well significant) and the corrected result is exactly p=0.050 is uncomfortable. It means that after correction, the result is only marginally significant. The paper should be more direct: "After Bonferroni correction, the primary result is at the p=0.050 boundary" rather than burying this in the appendix.

### 24. The "protective factor" definition changes between text and framework

Section 5.3 (line 256): "sales-office (secondary feature SALESORG has Jaccard = 0.61) and item-shippoint (high model variance)" are listed as false positives motivating the protective-factor check.

Section 7 (line 364): The protective factor is defined as "secondary features with train-validation Jaccard > 0.5 AND importance > 15%." This correctly addresses s-office (SALESORG Jaccard=0.61 and importance>15%). But the protective-factor definition does NOT address i-shippoint's high-variance issue. i-shippoint (C=48.8%) would fail the protective-factor check (no high-Jaccard secondary feature is documented) but was still classified ROB. The framework as written would still incorrectly flag i-shippoint as VULN even after applying the protective-factor check. This false positive is unaddressed by the framework.

### 25. ECE decimal inconsistency in Table A.4

Table A.4 (Appendix baselines) shows:
- i-plant: $\Delta$ECE = +0.**1314** (4 significant figures)
- i-shippoint: $\Delta$ECE = +0.**0727** (4 significant figures)
- All other ECE values use 3 significant figures (+0.408, +0.176, etc.)

Pick consistent precision throughout.

### 26. The conformal prediction step numbering creates a disconnect

Section 3.2 lists three steps: (1) Train, (2) Calibrate, (3) Evaluate. But "Evaluate" in step 3 covers both validation evaluation AND test evaluation, which are very different. The paper's design is: calibrate on 50% of validation, evaluate on other 50% of validation (not test), and separately evaluate on test. Step 3 as written implies evaluation is on held-out validation AND test, but this conflates the two.

### 27. Section 6 (Extended Experiments) interrupts the narrative arc

The paper structure is: Methodology → Theory → Results → **Extended Experiments** → Decision Framework → Discussion. Section 6 contains ACI, RAPS, shift detection, cross-domain validation, retraining, and placebo — essentially a grab-bag. The cross-domain validation (Section 6.4) is the core external validity evidence and should appear in Results (Section 5), not buried as experiment 5 of 6 in Section 6. The ACI and RAPS results are supporting material. The current structure makes the cross-domain result look like an afterthought.

### 28. Missing: why does boosting specifically show concentration-drop correlation when RF does not?

The Discussion (Section 8) is only one paragraph and does not address the most natural theoretical question: why do gradient-boosted models develop concentrated feature dependence while random forests do not? The paper acknowledges RF fails to replicate (rho=0.30) and gives a mechanistic account in Appendix A.9 (compressed concentration range, smoothed probabilities). But this explanation is critical to the paper's claims and should appear in Discussion, not the appendix. Reviewers will ask this question in their first read.

### 29. Missing: Figure 1 (SHAP dynamics) caption refers to panels C–D that may not exist in the PDF

Figure 1 caption (`fig:shap`, line 288): "(A) Catastrophic task... (B) Robust task... (C–D) Importance dynamics and ranking stability."

Panels C and D are mentioned in the caption. If the actual PDF figure (`figures/figure3_feature_importance.pdf`) does not have panels C and D, this caption is wrong. The file exists on disk but the caption was presumably written to match 4 panels; if the figure was redesigned to show only 2 panels, the caption is stale.

---

## FIGURE/FILE VERIFICATION

### Files confirmed present:
- `figures/figure3_feature_importance.pdf` — EXISTS
- `figures/conformity_score_cdfs.pdf` — EXISTS
- `results/figure_n11_correlation.pdf` — EXISTS

### Files NOT included but potentially needed:
- A figure for the n=16 primary result — `results/figure_n12_correlation.pdf` EXISTS ON DISK but is NOT included in the paper. This is the most important missing figure.

### No broken `\ref{}` or `\cite{}` detected. No `??` in compiled output (log file clean).

---

## SUMMARY SCORECARD

| Category | Issues Found | Severity |
|---|---|---|
| Abstract duplicate paragraph | 1 | FATAL |
| No figure for primary n=16 result | 1 | HIGH |
| Jackknife count framing contradiction | 1 | HIGH |
| "8 tasks with severe feature turnover" incorrect | 1 | HIGH |
| COVID-era group undefined in Table 3 | 1 | HIGH |
| Covertype drop 81.8 vs 82 pp inconsistency | 1 | MEDIUM-HIGH |
| "5 applicable tasks" mixes catastrophic/robust | 1 | MEDIUM-HIGH |
| Stack Overflow absent from framework table | 1 | MEDIUM-HIGH |
| i-shippoint false positive status contradictory | 1 | MEDIUM |
| "+19 pp recovery" misleadingly framed | 1 | MEDIUM |
| No external dataset description table | 1 | MEDIUM |
| ACI section lacks table/appendix reference | 1 | MEDIUM |
| "Asymmetric evidence" undefined | 1 | MEDIUM |
| "7 types of distribution shift" unsupported | 1 | MEDIUM |
| Theorem A1 not realistic for trees | 1 | MEDIUM |
| Figure caption rho=0.833 vs primary rho=0.853 | 1 | MEDIUM |
| i-shippoint protective factor gap | 1 | MEDIUM |
| Style/minor issues | ~12 | LOW |

**Total major issues requiring attention before submission: 15+**

The abstract duplication is a show-stopper. Fix it first. The missing n=16 figure is the second priority. Everything else is addressable in one revision pass.
