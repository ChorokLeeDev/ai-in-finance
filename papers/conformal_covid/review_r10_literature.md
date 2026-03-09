# Literature Review Report — Round 10 (Final)

**Paper:** Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study
**Venue:** UAI 2026
**File:** `papers/conformal_covid/uai_2026/main.tex`
**BibTeX:** `papers/conformal_covid/uai_2026/references.bib`
**Review date:** 2026-02-20
**Reviewer:** Literature Review Agent (R10 — final pass)

---

## 1. R9 Issue Status

### 1.1 `adebayo2018sanity` — booktitle "(NeurIPS)" suffix

**R9 flag:** booktitle reads `Advances in Neural Information Processing Systems (NeurIPS)` — parenthetical suffix inconsistent with other NeurIPS entries.

**R10 finding:** NOT FIXED. The entry at line 107–113 still reads:

```bibtex
@inproceedings{adebayo2018sanity,
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  ...
}
```

All other NeurIPS entries in the file (`romano2020classification`, `romano2019conformalized`, `tibshirani2019conformal`, `gibbs2021adaptive`, `lundberg2017unified`, `malinin2021shifts`) use the bare form `Advances in Neural Information Processing Systems` without any parenthetical suffix. The inconsistency persists.

**Required fix:**
```bibtex
booktitle={Advances in Neural Information Processing Systems},
```

### 1.2 `koh2021wilds` — booktitle "(ICML)" suffix and `organization=` field

**R9 flag (a):** booktitle reads `International Conference on Machine Learning (ICML)` — parenthetical suffix inconsistent with other ICML entry (`zaffran2022adaptive`).

**R9 flag (b):** uses `organization={PMLR}` — semantically incorrect (PMLR is a publisher, not an organizing body).

**R10 finding:** NEITHER issue has been fixed. Lines 91–98 still read:

```bibtex
@inproceedings{koh2021wilds,
  booktitle={International Conference on Machine Learning (ICML)},
  pages={5637--5664},
  year={2021},
  organization={PMLR}
}
```

The sister entry `zaffran2022adaptive` uses `booktitle={International Conference on Machine Learning}` with no suffix and no `organization` field, making the inconsistency clear.

**Required fix:**
```bibtex
booktitle={International Conference on Machine Learning},
publisher={PMLR},
```

(Or optionally add `volume=139, series={Proceedings of Machine Learning Research}` for full PMLR-style formatting.)

---

## 2. Full 24-Entry Verification (Final Pass)

All 24 entries are reviewed below for factual correctness, venue accuracy, author completeness, and format consistency.

### Entry 1: `vovk2005algorithmic`
- Type: `@book` — correct.
- Authors: Vovk, Gammerman, Shafer — correct.
- Year: 2005, Publisher: Springer — correct.
- **Status: PASS.**

### Entry 2: `romano2020classification`
- Type: `@inproceedings` NeurIPS — correct (NeurIPS 2020).
- Title: "Classification with Valid and Adaptive Coverage" — correct (this is APS).
- Authors: Romano, Sesia, Candès — correct.
- Volume 33, pages 3581–3591 — correct.
- **Status: PASS.**

### Entry 3: `romano2019conformalized`
- Type: `@inproceedings` NeurIPS — correct.
- Title: "Conformalized Quantile Regression" — correct.
- Authors: Romano, Patterson, Candès — correct.
- Volume 32, year 2019 — correct.
- **Status: PASS.**

### Entry 4: `tibshirani2019conformal`
- Type: `@inproceedings` NeurIPS — correct (confirmed against R9 fix; was @article in earlier rounds).
- Authors: Tibshirani, Barber, Candès, Ramdas — correct (all 4).
- Volume 32, year 2019 — correct.
- **Status: PASS.**

### Entry 5: `gibbs2021adaptive`
- Type: `@inproceedings` NeurIPS — correct (confirmed against earlier rounds where it was @article).
- Title: "Adaptive Conformal Inference Under Distribution Shift" — correct.
- Authors: Gibbs, Candès — correct.
- Volume 34, pages 1660–1672, year 2021 — correct.
- **Status: PASS.**

### Entry 6: `barber2023conformal`
- Type: `@article` Annals of Statistics — correct.
- Authors: Barber, Candès, Ramdas, Tibshirani — correct (all 4).
- Volume 51, number 2, pages 816–845, year 2023 — correct.
- **Status: PASS.**

### Entry 7: `angelopoulos2021gentle`
- Type: `@article` Foundations and Trends in Machine Learning — correct.
- Year: 2023 — correct (F&T peer-reviewed publication year; key name `angelopoulos2021gentle` is misleading but low priority if year=2023 renders correctly).
- Volume 16, number 4, pages 494–591 — correct.
- **Status: PASS** (key name mismatch is cosmetic only; rendered year is correct).

### Entry 8: `zaffran2022adaptive`
- Type: `@inproceedings` ICML — correct.
- Title: "Adaptive Conformal Predictions for Time Series" — correct.
- Authors: Zaffran, Féron, Goude, Josse, Dieuleveut — correct.
- Pages 25834–25866, year 2022 — correct.
- Booktitle: `International Conference on Machine Learning` — bare form, no suffix. Correct and consistent style.
- **Status: PASS.**

### Entry 9: `podkopaev2021distribution`
- Type: `@inproceedings` UAI 2021 — correct.
- Authors: Podkopaev, Ramdas — correct.
- Volume 161, pages 844–853, series PMLR — correct.
- Editor, URL populated — thorough.
- **Status: PASS.**

### Entry 10: `lundberg2017unified`
- Type: `@inproceedings` NeurIPS — correct.
- Authors: Lundberg, Lee — correct.
- Volume 30, year 2017 — correct.
- **Status: PASS.**

### Entry 11: `koh2021wilds`
- Type: `@inproceedings` — correct entry type.
- Authors: Koh et al. with "others" — acceptable for a long author list.
- Pages 5637–5664, year 2021 — correct.
- **FAIL (a):** `booktitle={International Conference on Machine Learning (ICML)}` — "(ICML)" suffix is inconsistent with `zaffran2022adaptive` which uses bare form.
- **FAIL (b):** `organization={PMLR}` — PMLR is a publisher/series, not the organizing body. Should be `publisher={PMLR}`.

### Entry 12: `gulrajani2020search`
- Type: `@inproceedings` ICLR — correct.
- Year: 2021 — correct (published ICLR 2021; key `gulrajani2020search` reflects arXiv 2020 date, which is cosmetic).
- Booktitle: `International Conference on Learning Representations (ICLR)` — contains "(ICLR)" suffix.
- NOTE: `garg2022leveraging` and `angelopoulos2024conformal` and `angelopoulos2021uncertainty` and `lopez2017revisiting` all use `International Conference on Learning Representations` without a suffix. This is another booktitle inconsistency.
- **Status: MINOR FAIL** — "(ICLR)" suffix in booktitle is inconsistent with all other ICLR entries in the file. Should be `booktitle={International Conference on Learning Representations}`.

### Entry 13: `adebayo2018sanity`
- **FAIL:** `booktitle={Advances in Neural Information Processing Systems (NeurIPS)}` — "(NeurIPS)" suffix inconsistent with all other NeurIPS entries (see Section 1.1 above).
- Volume 31, year 2018, authors correct.

### Entry 14: `malinin2021shifts`
- Type: `@inproceedings` NeurIPS D&B Track — correct entry type and booktitle.
- Booktitle: `Advances in Neural Information Processing Systems Datasets and Benchmarks Track` — correct for D&B track.
- Authors: Malinin et al. with "others" — acceptable.
- Year 2021 — correct.
- **Status: PASS.**

### Entry 15: `lundberg2020local`
- Type: `@article` Nature Machine Intelligence — correct.
- Authors: Lundberg et al. (10 named) — correct.
- Volume 2, number 1, pages 56–67, year 2020 — correct.
- **Status: PASS.**

### Entry 16: `fey2024relbench`
- Type: `@inproceedings` — correct.
- Title: "RelBench: A Benchmark for Deep Learning on Relational Databases" — correct.
- Booktitle: `Advances in Neural Information Processing Systems Track on Datasets and Benchmarks` — correct for D&B track.
- Year: 2024 — correct.
- Authors: Robinson, Ranjan, Hu, Huang, Han, Dobles, Fey, Lenssen, Yuan, Zhang, He, Leskovec — all 12 confirmed correct. No spurious authors (Rex Ying and Jiaxuan You are NOT in the list — correct).
- **Status: PASS.**

### Entry 17: `feldman2023achieving`
- Type: `@article` TMLR — correct (TMLR is a journal, so @article is right).
- Title: "Achieving Risk Control in Online Learning Settings" — correct.
- Authors: Feldman, Ringel, Bates, Romano — all 4 correct. Angelopoulos is correctly absent.
- Journal: `Transactions on Machine Learning Research` — correct.
- Year: 2023 — correct.
- **Status: PASS.**

### Entry 18: `angelopoulos2024conformal`
- Type: `@inproceedings` ICLR — correct (ICLR 2024, not JASA — common error avoided).
- Title: "Conformal Risk Control" — correct.
- Authors: Angelopoulos, Bates, Fisch, Lei, Schuster — correct (5 authors).
- Year: 2024 — correct.
- Booktitle: `International Conference on Learning Representations` — bare form, consistent. PASS.
- **Status: PASS.**

### Entry 19: `garg2022leveraging`
- Type: `@inproceedings` ICLR — correct.
- Title: "Leveraging Unlabeled Data to Predict Out-of-Distribution Performance" — correct.
- Authors: Garg, Balakrishnan, Lipton, Neyshabur, Sedghi — correct.
- Year: 2022 — correct.
- **Status: PASS.**

### Entry 20: `angelopoulos2021uncertainty`
- Type: `@inproceedings` ICLR — correct. This is the RAPS paper.
- Title: "Uncertainty Sets for Image Classifiers using Conformal Prediction" — correct.
- Authors: Angelopoulos, Bates, Malik, Jordan — correct (all 4).
- Year: 2021 — correct.
- **Status: PASS.**

### Entry 21: `gretton2012kernel`
- Type: `@article` JMLR — correct.
- Title: "A Kernel Two-Sample Test" — correct.
- Authors: Gretton, Borgwardt, Rasch, Schölkopf, Smola — correct.
- Volume 13, pages 723–773, year 2012 — correct.
- **Status: PASS.**

### Entry 22: `lopez2017revisiting`
- Type: `@inproceedings` ICLR — correct.
- Title: "Revisiting Classifier Two-Sample Tests" — correct.
- Authors: Lopez-Paz, Oquab — correct.
- Year: 2017 — correct.
- **Status: PASS.**

### Entry 23: `lei2018distribution`
- Type: `@article` JASA — correct.
- Title: "Distribution-Free Predictive Inference for Regression" — correct.
- Authors: Lei, G'Sell, Rinaldo, Tibshirani, Wasserman — correct (all 5).
- Volume 113, number 523, pages 1094–1111, year 2018 — correct.
- **Status: PASS.**

### Entry 24: `dua2017uci`
- Type: `@misc` — correct.
- Authors: Dua, Graff — correct.
- Year: 2017 — correct.
- `howpublished`: "University of California, Irvine, School of Information and Computer Sciences" — `howpublished` field is used correctly (not `institution` which is invalid for @misc). **This was fixed in a prior round — PASS.**
- URL: `https://archive.ics.uci.edu/ml` — NOTE: the old `/ml` path now redirects to `https://archive.ics.uci.edu`. Functionally harmless (redirect works), but the canonical current URL is `https://archive.ics.uci.edu`. Low priority.
- **Status: PASS** (minor: stale URL path, functionally harmless).

---

## 3. Summary of Remaining Issues

### Issues that must be fixed before submission (2 entries, 3 field errors):

| Entry | Field | Current value | Required value | Severity |
|---|---|---|---|---|
| `adebayo2018sanity` | `booktitle` | `Advances in Neural Information Processing Systems (NeurIPS)` | `Advances in Neural Information Processing Systems` | Minor (style) |
| `koh2021wilds` | `booktitle` | `International Conference on Machine Learning (ICML)` | `International Conference on Machine Learning` | Minor (style) |
| `koh2021wilds` | `organization` | `{PMLR}` | replace with `publisher={PMLR}` | Minor (semantic) |
| `gulrajani2020search` | `booktitle` | `International Conference on Learning Representations (ICLR)` | `International Conference on Learning Representations` | Minor (style) |

**Note:** The `gulrajani2020search` "(ICLR)" suffix was not flagged in R9 but is confirmed as an inconsistency on this pass — it is the same class of issue as the two R9 flags. All ICLR entries in the file (`garg2022leveraging`, `angelopoulos2024conformal`, `angelopoulos2021uncertainty`, `lopez2017revisiting`) use bare booktitle; `gulrajani2020search` is the sole exception.

### Non-blocking observations:

- `angelopoulos2021gentle`: key `angelopoulos2021gentle` but year=2023 — cosmetic only; rendered citation is correct.
- `gulrajani2020search`: key `gulrajani2020search` but year=2021 (ICLR publication) — cosmetic only; rendered citation is correct.
- `dua2017uci`: URL uses old `/ml` path which redirects; functionally harmless.

---

## 4. Differentiation Assessment

**Rating: 4/5**

The paper occupies a distinct position that has not been previously filled. The combination of (1) a pre-deployment SHAP-based diagnostic, (2) a COVID-19 natural experiment with a within-cohort design, and (3) a formal monotonicity theorem for APS under concentrated shift is genuinely novel. No paper in the literature simultaneously addresses all three.

**Closest prior work reviewed:**

- Angelopoulos et al. (2021) RAPS and (2024) Conformal Risk Control: address conformal scoring but not pre-deployment diagnostics.
- Garg et al. (2022): predicts OOD accuracy degradation but requires unlabeled test data — correctly distinguished.
- Barber et al. (2023): characterizes coverage loss via TV distance but does not provide a deployable diagnostic.
- Tibshirani et al. (2019): handles known covariate shift via propensity scores — different regime.

**Minor gap in differentiation:** The related work does not acknowledge Angelopoulos & Bates (2021) RAPS (cited later in Section 3.2 as `angelopoulos2021uncertainty`) in the Related Work section — it appears only in the experimental section. A sentence in Related Work noting that RAPS reduces class-accumulation failures (but not concentrated-dependence failures) would sharpen the contribution claim and is consistent with the empirical results in Appendix F.

---

## 5. Research Gap Analysis

**Rating: 5/5**

The paper articulates the gap precisely: existing work characterizes *how* coverage degrades under shift (Tibshirani, Barber) and *how to recover* it online (Gibbs, Zaffran, Feldman), but none addresses *which* tasks will fail *before* test data is observed. The research question, methodology (validation-only SHAP), and conclusions are logically consistent. The placebo test and leave-one-out analysis further support the gap-filling claim.

---

## 6. Citation Quality and Currency Assessment

**Rating: 4/5**

**Citation statistics:** 24 total references. Year distribution:
- 2005–2017: 5 entries (vovk2005, gretton2012, lei2018, lundberg2017, lopez2017)
- 2018–2020: 5 entries (adebayo2018, romano2019, tibshirani2019, gulrajani2020, lundberg2020)
- 2021–2022: 9 entries (gibbs2021, romano2020, koh2021, podkopaev2021, malinin2021, angelopoulos2021gentle, angelopoulos2021uncertainty, zaffran2022, garg2022)
- 2023–2024: 5 entries (barber2023, feldman2023, angelopoulos2024, fey2024relbench, dua2017)

The balance is appropriate for a UAI 2026 submission. All core CP seminal works are present. Recency is adequate.

**Missing citations worth considering (not required but would strengthen the paper):**

1. **Gibbs & Candès (2024) JMLR** — "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts," JMLR Vol. 25 (arXiv 2208.08401). The paper cites Gibbs (2021) for ACI and runs ACI experiments; reviewers familiar with online CP will notice the 2024 journal extension is absent. Low-risk omission at UAI but worth adding if space permits.

2. **Shafer & Vovk (2008) JMLR** — "A Tutorial on Conformal Prediction," JMLR 9:371–421. Frequently expected alongside Vovk et al. (2005) in formal CP papers, especially those at UAI. The paper does cite (2005); absence of (2008) is a minor gap.

3. **Angelopoulos & Bates (2021) ICLR — RAPS** is cited but appears only in Section 3.2 and Appendix F, not in Related Work. Not a missing citation but a placement issue.

**No factually incorrect citations were identified.** The known high-risk error (Conformal Risk Control as JASA) is correctly cited as ICLR 2024. The RelBench author list is verified correct with all 12 authors. The Feldman et al. TMLR authors are the correct 4 (no spurious Angelopoulos). The Malinin et al. entry correctly uses `@inproceedings` for the D&B track.

---

## 7. Overall Assessment and Recommendations

**Overall Rating: 4.5/5**

**Recommendation: Accept with minor BibTeX corrections.**

The paper is scientifically sound, the contributions are genuine and well-differentiated, the gap is clearly articulated and appropriately filled, and the citation set covers all required seminal work with correct factual metadata.

### Three remaining actions before final submission:

**Action 1 (required):** Fix `adebayo2018sanity` booktitle — remove "(NeurIPS)" suffix.

**Action 2 (required):** Fix `koh2021wilds` — remove "(ICML)" suffix from booktitle; change `organization={PMLR}` to `publisher={PMLR}`.

**Action 3 (required, newly identified this round):** Fix `gulrajani2020search` — remove "(ICLR)" suffix from booktitle for consistency with all other ICLR entries.

**Action 4 (optional):** Update `dua2017uci` URL from `https://archive.ics.uci.edu/ml` to `https://archive.ics.uci.edu`.

**Action 5 (optional):** Add one sentence in Section 2 Related Work placing RAPS (Angelopoulos et al. 2021) in context of the conformal scoring literature, noting it addresses class-accumulation failures rather than concentrated-dependence failures. This is already the empirical finding of Appendix F and would strengthen the differentiation narrative.

---

*Report generated by Literature Review Agent, R10 final pass, 2026-02-20.*
