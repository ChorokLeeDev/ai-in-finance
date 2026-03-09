# Statistical Audit: "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"

**Auditor role**: Elite statistician reviewer
**Date**: 2026-02-20
**Document**: `papers/conformal_covid/uai_2026/main.tex`

---

## CRITICAL FINDING 1: Abstract Contains Verbatim Duplicate Paragraph

**Severity: CRITICAL — Submission-blocking**

Lines 45–46 of the LaTeX source contain two nearly identical paragraphs in the abstract. The first ends with:
> "...consider quarterly retraining if vulnerable (+19 pp recovery, p = 0.04); defer routine retraining for lower-risk tasks."

The second (duplicate) paragraph ends with:
> "...suggest that quarterly retraining may partially recover coverage when vulnerable (+19 pp, p = 0.04); and defer routine retraining for lower-risk tasks."

Both paragraphs begin identically: *"seeds), we show that, for gradient-boosted models, SHAP concentration is strongly associated with failure severity: ρ = 0.853..."*

This is a clear copy-paste error. The abstract contains approximately two full repetitions of the same content. This alone would likely cause desk rejection or immediate reviewer confusion. The duplicate must be removed; only one version should remain.

---

## FINDING 2: Abstract ρ = 0.853 vs. Table 2 Primary Result

**Severity: MAJOR — Potential inconsistency**

The abstract states: *"ρ = 0.853, p < 0.001 across 16 multiclass tasks in 9 domains (bootstrap 95% CI [0.50, 0.96]; Kendall τ = 0.667, p < 0.001)"*

Table 2 (Stratified Correlation Analysis) shows:
- Row "Multiclass (9 dom.), n=16": ρ = **0.853**, Kendall τ = **0.667**, p < 0.001, Boot. 95% CI = **[0.50, 0.96]**

**Verdict: CONSISTENT.** The abstract and table match exactly on ρ, τ, and CI for the primary n=16 result.

However, a subtlety: the abstract says "16 multiclass tasks in 9 domains." The table confirms n=16 for "Multiclass (9 dom.)." Section 3.1 explains: "8 SALT tasks + 8 external multiclass tasks = 16, with SALT counting as 1 domain + 8 external domains = 9 domains total." This arithmetic is internally consistent.

---

## FINDING 3: Table 2 — "COVID-era n=9" Row Composition

**Severity: MODERATE — Requires clarification**

Table 2 contains the row: "COVID-era, n=9, ρ=0.883, p=0.002, Boot. 95% CI=[0.39, 1.00]"

This row has **no Kendall τ reported** (shown as "---"). No explanation is provided in the table footnote or in Section 3.3 for what constitutes the "COVID-era n=9" group.

**Questions raised:**
1. Which 9 datasets compose this group? It cannot be a simple subset of the 8 SALT tasks (n=8 < 9), so presumably it is 8 SALT tasks + 1 external dataset. Which external dataset is included and why?
2. If COVID-era = SALT (8) + Covertype (1) = 9, why would Covertype qualify as "COVID-era"? Covertype uses a different temporal shift (geographic, not COVID).
3. The n=9 row yields ρ=0.883, which is *higher* than the full n=16 result (ρ=0.853). This is arithmetically plausible if the added external datasets are noisier, but no narrative explanation is provided.
4. Why is Kendall τ not reported for n=9 but reported for all other rows?

**Verdict: UNEXPLAINED.** The composition of the "COVID-era n=9" group is never defined in the main text or appendix. This is a gap that a reviewer would flag.

---

## FINDING 4: Table 2 — "Multiclass (4 dom.), n=11" Row

**Severity: MODERATE**

Table 2 row: "Multiclass (4 dom.), n=11, ρ=0.909, τ=0.782, p<0.001, Boot. 95% CI=[0.61, 1.00]"

The memory file notes the "early n=11" result as ρ=0.909. Appendix G (app:icc) states: "The early n=11 result (ρ=0.909) used single-seed external values; switching to multi-seed means at n=11 yields ρ=0.818."

**Problem**: The n=11 row in Table 2 still reports ρ=0.909. If the multi-seed reanalysis yields ρ=0.818 at n=11, then the n=11 row in Table 2 is the *single-seed* value, which contradicts the paper's methodology of using multi-seed means (Section 3.5, cross-domain extension uses "10-seed means for all external datasets"). The table does not note that n=11 uses single-seed values vs. the other rows using multi-seed means. This creates a methodological inconsistency within Table 2 itself.

Also: "4 domains" for n=11 means 8 SALT + 3 external tasks in 3 additional domains = 4 total domains (with SALT as 1). This implies 3 external multiclass datasets were added first. Which 3? This is not specified.

---

## FINDING 5: Bootstrap CI for SALT n=8 — [0.29, 1.00] vs. "10,000 resamples"

**Severity: MINOR — Verification needed**

Section 3.3 states: "Bootstrap 95% CI: [0.29, 1.00] (10,000 resamples)."
Table 2 shows: Boot. 95% CI = [0.29, 1.00] for n=8. **Consistent.**

However, App. G (icc) notes: "bootstrap resampling (1,000 resamples of 10K validation samples) yields coefficient of variation below 1%"—this refers to SHAP stability bootstrap (1,000 resamples), not the correlation CI bootstrap (10,000 resamples). These are two different bootstrap procedures. The distinction is potentially confusing but technically not contradictory.

**Concern**: With n=8, the upper bound of [0.29, 1.00] reaching exactly 1.00 is expected (Spearman ρ is bounded at 1.00). The width of this CI ([0.71 range]) is extremely wide for an n=8 sample — this is statistically appropriate but should be noted as a limitation, which the paper does acknowledge.

---

## FINDING 6: LOO Stability — "6/8 significant" vs. "2/8 non-significant"

**Severity: MINOR — Technically consistent but phrasing is misleading**

**Table 2 footnote**: "LOO stability (SALT): ρ ∈ [0.75, 0.96]; 6/8 jackknife significant."

**Section 3.3 (bullet 3)**: "statistical significance is not maintained for 2 of 8 jackknife samples (p = 0.052) due to reduced power at n=7."

Mathematically, 6/8 significant ↔ 2/8 non-significant. These are logically equivalent and thus **consistent**.

However, note the p-value cited: "p = 0.052" is given for the non-significant cases. The text implies both non-significant jackknife samples have p = 0.052 exactly, which would be a remarkable coincidence. More likely, 0.052 is representative of the least significant case, and the other non-significant case may have a different p-value. The phrasing "p = 0.052" as if it applies to both deserves clarification.

Also: the LOO range [0.75, 0.96] is stated to be for Spearman ρ, but with n=7 the minimum possible Spearman ρ when exactly one rank is changed is approximately 0.75 (removing the most influential point from a monotone pattern) — this range is plausible but narrow, suggesting very high LOO stability, which is reassuring.

---

## FINDING 7: Threshold Table (App D) — Precision/Recall/F1 Arithmetic Verification

**Severity: MAJOR — Arithmetic issue detected**

App D (Table 5, threshold sensitivity) at threshold=40%:
- TP=5, FP=1, FN=1, Precision=0.83, Recall=0.83, F1=0.83

**Arithmetic check:**
- Precision = TP/(TP+FP) = 5/(5+1) = 5/6 = **0.8333** → rounds to 0.83 ✓
- Recall = TP/(TP+FN) = 5/(5+1) = 5/6 = **0.8333** → rounds to 0.83 ✓
- F1 = 2×Precision×Recall/(Precision+Recall) = 2×(5/6)×(5/6)/((5/6)+(5/6)) = 2×(25/36)/(10/6) = (50/36)/(10/6) = (50/36)×(6/10) = 300/360 = **5/6 = 0.8333** → rounds to 0.83 ✓

**Implied TN**: Total tasks = n=16 (confirmed by table caption). TN = 16 - TP - FP - FN = 16 - 5 - 1 - 1 = **9**.

**At-risk tasks (TP+FN = 6)**: These are the tasks with coverage drop > 15pp.

Let's cross-check against Table 6 (framework validation). At-risk tasks (drop > 15pp):
- SALT: s-payterms (77.1pp), s-shipcond (71.6pp), s-group (71.2pp), i-shippoint (18.5pp) → 4 SALT at-risk
- External: Covertype (81.8pp drop) → 1 external at-risk
- KDDCup99: mean drop = 15.9pp → classified as At-risk (above 15pp cutoff)

That gives **6 at-risk tasks** (TP+FN=6). Footernote confirms: "the single FP is s-office...and the single FN is KDDCup99."

**So at 40% threshold:**
- VULN predictions: s-payterms (54.2%), s-shipcond (50.7%), i-shippoint (48.8%), s-group (47.3%), s-office (42.6%), Covertype (49.8%) → 6 VULN
- Of these 6 VULN: 5 TP (payterms, shipcond, i-shippoint, s-group, Covertype) + 1 FP (s-office)
- At-risk not predicted VULN: KDDCup99 (21.1% < 40%) → 1 FN
- TN: remaining 9 tasks correctly ROB

**TP+FP = 6 predictions of VULN; TP+FN = 6 truly at-risk; TN = 9 correctly ROB; total = 16. ✓**

**However**: There is a discrepancy. Table 6 lists i-shippoint (48.8%) as VULN and "At-risk*" — but Table 1 shows i-shippoint drop = 18.5pp (mean over 50 seeds), which is > 15pp, so it counts as at-risk. The footnote in Table 6 says "Median drop 1.2%; most seeds maintain coverage" — yet it's still classified as At-risk (> 15pp mean). This classification inconsistency (mean vs. median for threshold purposes) deserves explicit acknowledgment.

---

## FINDING 8: RAPS Numbers — Main Text vs. Table RAPS (App H)

**Severity: MODERATE — Inconsistency in what "APS Drop" means**

**Section 3.2 (RAPS)** states: "s-group: 73.5±33.9% → 10.4±1.3%; s-payterms: 79.9±29.4% → 35.1±28.6%"

**Table RAPS (App H)** confirms: s-group APS Drop = 73.5±33.9, RAPS Drop = 10.4±1.3. s-payterms APS Drop = 79.9±29.4, RAPS Drop = 35.1±28.6. **These are internally consistent.**

However, the **Table RAPS footnote** explicitly states: "APS drops differ from 50-seed Table 1 due to smaller seed range (10 vs. 50) and high variance."

**Cross-check against Table 1 (50-seed):**
- s-group: Table 1 drop = 71.2%; RAPS table APS drop = 73.5%. Difference = 2.3pp — plausible given seed range.
- s-payterms: Table 1 drop = 77.1%; RAPS table APS drop = 79.9%. Difference = 2.8pp — plausible.
- s-shipcond: Table 1 drop = 71.6%; RAPS table APS drop = 60.4±31.8%. Difference = 11.2pp — larger discrepancy.

The s-shipcond discrepancy (11.2pp between 10-seed RAPS table and 50-seed Table 1) is notable. With CV > 50% and only 10 seeds, this is within plausible sampling variation but is large. The footnote appropriately flags this. However, the main text Section 3.2 references the RAPS table values (73.5±33.9%, 79.9±29.4%) without clarifying these are 10-seed values, potentially creating confusion when a reader compares them against Table 1's 50-seed values (71.2%, 77.1%).

**Specifically**: The Section 3.2 claim "s-group: 73.5±33.9%" uses the 10-seed APS value, while Table 1 reports 71.2% (50-seed). A reader might interpret the main text as referring to the 50-seed Table 1 value. The paper should either clarify "(10-seed)" in Section 3.2 or consistently reference Table 1 values.

---

## FINDING 9: KS Statistics — Abstract/Intro vs. Table KS

**Severity: MINOR — Consistent**

**Section 2 (intro, contribution 2)**: "KS = 0.68--0.96, all p < 10^{-10}"

**Table KS (App I)**:
- s-shipcond: KS = 0.956
- s-payterms: KS = 0.748
- s-group: KS = 0.676

Range for catastrophic tasks: [0.676, 0.956]. The intro states [0.68, 0.96]. Rounding: 0.676 rounds to 0.68 ✓; 0.956 rounds to 0.96 ✓.

**Section 4 (theory)** also states "KS = 0.68--0.96, all p < 10^{-10}". **Consistent with Table KS.**

However: Table KS includes both catastrophic AND robust tasks, and robust tasks also show KS > 0.5 (e.g., s-office: KS=0.994, i-plant: KS=0.741). The intro/text implies KS=0.68--0.96 is specific to "catastrophic tasks." Section 2 says "catastrophic tasks (KS = 0.68--0.96, all p < 10^{-10}; Figure fig:cdfs)." The table label separates catastrophic (KS 0.676--0.956) from robust. So the range correctly refers to the catastrophic subset. ✓

**Subtle issue**: s-office (robust) has KS=0.994, which is *higher* than all catastrophic tasks' KS values. This means KS alone (without knowing score direction) does not distinguish catastrophic from robust. The table footnote mentions "two robust tasks (i-incoterms, i-shippoint) show test scores shifted leftward" — but s-office's KS=0.994 represents a near-degenerate distribution (both calibration and test scores = 1.00). This is a nuance not addressed in the main text's presentation of KS results.

---

## FINDING 10: "n=9 domains" vs. Dataset Count Arithmetic

**Severity: MODERATE — Accounting requires clarification**

Section 3.1: "External validation uses 11 additional datasets spanning 10 domains. Excluding binary tasks yields 8 external multiclass datasets, and together with the 8 multiclass SALT tasks this gives the primary endpoint of n=16 multiclass tasks across 9 domains (SALT counts as one domain)."

Cross-check: Table 6 (framework_validation) lists these 8 external datasets:
1. Covertype
2. Shuttle
3. Avila Bible
4. PAMAP2
5. KDDCup99
6. Pendigits
7. Satimage
8. Gas Sensor

That's 8 external multiclass tasks. The 11 additional datasets spanning 10 domains implies 3 binary tasks excluded. Which 3 binary datasets? These are not named. The abstract says "External validation across 9 held-out domains" — but Section 3.1 says "10 non-supply-chain domains" (for the 11 additional datasets). With 3 binary tasks excluded:
- If the 3 excluded binary tasks all come from 1 domain, then 10 - 1 = 9 external multiclass domains.
- But the text says "8 external multiclass datasets" — that could be from fewer than 8 domains if multiple datasets share a domain.

The abstract says "9 held-out domains" while Section 6 (external validation) says "11 additional datasets across 10 non-supply-chain domains." The arithmetic: 10 external domains + 1 SALT domain = 11 domains for n=19 combined (Table 2 row "Combined (11 dom.), n=19"). The Table 2 "Combined (11 dom.)" row uses n=19 (presumably including Stack Overflow and other binary tasks). So 11 total domains = 1 SALT + 10 external, consistent with Section 3.1.

For the primary n=16 endpoint: 8 SALT + 8 external = 16 tasks in 9 domains (1 SALT + 8 external domains). But Section 3.1 says 10 external domains; if 8 multiclass external datasets span 8 external domains (1 dataset per domain), then 1 SALT + 8 = 9 domains. This is consistent. But if any 2 of the 8 external datasets share a domain, the count would be < 9. The paper never explicitly maps datasets to domains for external data.

The conclusion says "Standard shift detectors...detect shift for all tasks uniformly but do not distinguish catastrophic from robust outcomes (ρ ≤ 0.19, all p > 0.6)." Section 3.5 lists MMD ρ = -0.048, C2ST ρ = 0.191, PSI ρ = 0.071 — all ≤ 0.19. The abstract says "ρ ≤ 0.19." **Consistent** (0.191 rounds to 0.19). ✓

---

## FINDING 11: Covertype Drop — 82pp vs. 81.8pp Inconsistency

**Severity: MINOR**

The abstract states: "Covertype (C=49.8%, 82 pp drop)"
Section 3.3 states: "Covertype is the key external catastrophic case (C=49.8%, 81.8 pp drop)"
Contribution 6 (intro): "(C=49.8%, drop=81.8±pp)"

The abstract rounds 81.8 to "82 pp" while the main text reports 81.8. This is a rounding difference. **Technically consistent** but the abstract should match the precision used in the body (81.8pp) or explicitly note it is approximate.

---

## FINDING 12: Retraining Numbers — "+19 pp" and "+18.9 pp"

**Severity: MINOR**

The abstract says "quarterly retraining (+19 pp recovery, p = 0.04)."
Section 3.4 says "Quarterly retraining improves sales-shipcond by +18.9 pp (p=0.04)."
Introduction contribution 4 says "+19 pp (p=0.04)."

18.9pp rounds to 19pp. **Consistent** across abstract and intro when rounded, but the body text uses 18.9pp. This is acceptable but the precision should be consistent.

---

## FINDING 13: Table 2 — Row Order and Primary Result Labeling

**Severity: MINOR**

Table 2 lists rows in order: n=8, n=9, n=11, n=15, n=16 (bolded as primary), n=19. The primary result (n=16) is properly bolded. However:

- The n=9 "COVID-era" row appears between n=8 SALT and n=11, which is logical by sample size.
- The n=11 row ("Multiclass (4 dom.)") shows ρ=0.909 — higher than the primary n=16 result of ρ=0.853. As mentioned in Finding 4, the n=11 value uses single-seed external data, not multi-seed. If the n=11 multi-seed value is ρ=0.818 (as stated in App G), then the n=11 row is presenting a *superseded* analysis. Including it in the same table as the primary n=16 result without marking it as "single-seed" creates a potentially misleading comparison.

---

## FINDING 14: External Validation Counting — "7/9" vs. "6 deterministic + 1 near-deterministic"

**Severity: MINOR — Consistent but confusing**

Section 6 (cross-domain): "Ten-seed replication yields deterministic/near-deterministic behavior for 7/9 domains (6 deterministic, 1 near-deterministic)."
Introduction contribution 6: "7/9 domains (6 deterministic, 1 near-deterministic): Forest Covertype is correctly flagged...KDDCup99...and Stack Overflow...are boundary cases."

Table 6 lists 8 external datasets. The "9 held-out domains" from the abstract suggests 9 external datasets, but Table 6 shows only 8. The resolution appears to be that Stack Overflow is included in the "9 held-out domains" but excluded from the primary n=16 multiclass endpoint (as it exhibits "near-binary ceiling effect"). So the 9 held-out domains = 8 in Table 6 + Stack Overflow (binary, excluded from primary analysis).

Of 9 held-out domains: 7/9 are deterministic/near-deterministic (6 robust + Covertype deterministic = 7?). But Covertype is *catastrophic*, not robust. Recounting:
- Deterministic robust: Shuttle (9/10 = near-deterministic), Avila (10/10), PAMAP2 (10/10), Pendigits (10/10), Satimage (10/10), Gas Sensor (10/10) = 6 deterministic robust + 1 near-deterministic robust = 7 non-catastrophic deterministic
- Covertype: 10/10 deterministic *catastrophic*
- KDDCup99: seed-dependent (boundary)
- Stack Overflow: near-binary ceiling

So the "7/9 deterministic/near-deterministic" likely counts: 6 fully deterministic robust + Shuttle (9/10) = 7, with KDDCup99 and Covertype excluded from this count? But Covertype is also 10/10 deterministic. The counting is ambiguous. The text should clarify: 7/9 deterministic in their outcome (whether robust or catastrophic) vs. 2/9 seed-dependent.

---

## FINDING 15: "Multiclass (8 dom.), n=15" Row in Table 2

**Severity: MODERATE — Unexplained**

Table 2 contains: "Multiclass (8 dom.), n=15, ρ=0.882, τ=0.714, p<0.001, Boot. CI=[0.60, 0.97]"

This intermediate row (n=15 between the n=11 and n=16 primary) is never explained in the text. What is the composition of the n=15 "Multiclass (8 dom.)" group? It appears to be n=16 minus 1 task — possibly excluding Stack Overflow or KDDCup99. The difference between n=15 (8 domains) and n=16 (9 domains) is exactly 1 task from 1 domain. Which domain/task is excluded from n=15 but included in n=16?

Without this explanation, the n=15 row appears as an unexplained intermediate result.

---

## FINDING 16: Holm-Bonferroni Correction in App G

**Severity: MINOR — Arithmetic error**

App G states: "Under Holm-Bonferroni correction for this 5-test family, the adjusted p-value for top-1 concentration is 5 × 0.010 = 0.050."

This is the Bonferroni correction (multiply by number of tests), not the Holm-Bonferroni procedure. The Holm procedure uses the rank-ordered p-values and adjusts the most significant by 5×, the second-most by 4×, etc. If p=0.010 is the smallest p-value (rank 1), then the Holm-adjusted p-value is indeed 5 × 0.010 = 0.050. **This is correct for the Holm procedure applied to the smallest p-value.** However, calling it "Holm-Bonferroni" while computing it as simple multiplication is only correct if top-1 is the most significant test (rank 1 in ascending p-value order). The paper assumes this but does not state the p-values for the other 4 tests (top-2, top-3, HHI, entropy) that would confirm top-1 has the smallest p-value. This should be verified.

---

## SUMMARY TABLE OF FINDINGS

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| 1 | **Abstract verbatim duplicate paragraph** | CRITICAL | Must fix |
| 2 | Abstract ρ=0.853 matches Table 2 n=16 row | OK | Consistent |
| 3 | "COVID-era n=9" composition never defined | MAJOR | Must clarify |
| 4 | n=11 row in Table 2 uses single-seed values (superseded by multi-seed ρ=0.818) | MAJOR | Must disclose |
| 5 | Bootstrap CI [0.29, 1.00] for n=8 — 10,000 resamples | OK | Consistent |
| 6 | LOO "6/8 significant" vs. "2/8 non-significant" | OK | Logically equivalent |
| 7 | Threshold table arithmetic (TP=5, FP=1, FN=1, total=16) | OK | Verified correct |
| 7b | i-shippoint at-risk classification uses mean (not median) | MINOR | Inconsistency in footnote |
| 8 | RAPS 10-seed values in text vs. 50-seed Table 1 without explicit note in main text | MODERATE | Must clarify in text |
| 9 | KS 0.68--0.96 matches Table KS (rounds correctly) | OK | Consistent |
| 9b | s-office KS=0.994 higher than all catastrophic tasks — not discussed | MINOR | Potential confusion |
| 10 | "9 held-out domains" vs. "10 non-supply-chain domains" arithmetic | MODERATE | Clarify mapping |
| 11 | Covertype: abstract "82 pp" vs. text "81.8 pp" | MINOR | Rounding inconsistency |
| 12 | "+19 pp" vs. "+18.9 pp" across sections | MINOR | Rounding inconsistency |
| 13 | n=11 row with single-seed ρ=0.909 presented alongside multi-seed results | MODERATE | Disclose or update |
| 14 | "7/9 deterministic" counting ambiguous (Covertype is also 10/10) | MINOR | Clarify |
| 15 | "Multiclass (8 dom.), n=15" row never explained | MODERATE | Must explain composition |
| 16 | Holm-Bonferroni: assumes top-1 is rank-1 without showing other p-values | MINOR | Should state |

---

## PRIORITY FIXES REQUIRED

1. **IMMEDIATE**: Remove one of the two duplicate abstract paragraphs (Finding 1). This is a submission-blocking error.

2. **HIGH**: Define the composition of the "COVID-era n=9" group in Table 2 (Finding 3). Add a footnote explaining which 9 tasks comprise this group.

3. **HIGH**: Disclose that the n=11 row in Table 2 uses single-seed external values, not multi-seed (Finding 4). Either update n=11 to multi-seed (ρ=0.818) or add a footnote: "†Single-seed external values; multi-seed yields ρ=0.818, p=0.002 (App. G)."

4. **HIGH**: Explain the n=15 "Multiclass (8 dom.)" row in Table 2 (Finding 15). Identify which dataset is excluded.

5. **MODERATE**: In Section 3.2 (RAPS), explicitly label the percentages as 10-seed values and cross-reference Table 1's 50-seed values (Finding 8). E.g., "s-group: 73.5±33.9% [10-seed; 71.2% at 50 seeds, Table 1]."

6. **MODERATE**: Clarify the "7/9 deterministic/near-deterministic" counting to include whether Covertype (10/10 catastrophic) is counted (Finding 14).
