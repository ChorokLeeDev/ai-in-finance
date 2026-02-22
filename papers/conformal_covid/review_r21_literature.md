# Round 21 Literature Review — Conformal COVID (UAI 2026)

**Date**: 2026-02-22
**Reviewer**: Literature Review Agent (Round 21)
**Paper**: "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"

---

## Summary

This is a focused bibliographic audit. Prior rounds (R1–R20) fixed many errors. This round reads both files from scratch and independently verifies every entry category, with special attention to the four entries flagged in the task brief.

---

## 1. Targeted Entry Verification

### 1.1 `gibbs2025conditional` — CONFIRMED CORRECT

```bibtex
@article{gibbs2025conditional,
  title={Conformal Prediction with Conditional Guarantees},
  author={Gibbs, Isaac and Cherian, John J and Cand{\`e}s, Emmanuel J},
  journal={Journal of the Royal Statistical Society: Series B (Statistical Methodology)},
  volume={87},
  number={4},
  pages={1100--1126},
  year={2025}
}
```

Verified against Oxford Academic (doi: 10.1093/jrsssb/qkaf008):
- Volume 87, Issue 4, September 2025 — CORRECT
- Pages 1100-1126 — CORRECT
- Author names (Gibbs, Cherian, Candès) — CORRECT
- Journal name — CORRECT (OUP moved the journal from Wiley in 2023; both naming conventions are acceptable in literature)

**Status: No errors.**

---

### 1.2 `kasa2025adapting` — CONFIRMED CORRECT

```bibtex
@inproceedings{kasa2025adapting,
  title={Adapting Prediction Sets to Distribution Shifts Without Labels},
  author={Kasa, Kevin and Zhang, Zhiyu and Yang, Heng and Taylor, Graham W.},
  booktitle={Proceedings of the Forty-First Conference on Uncertainty in Artificial Intelligence},
  pages={1990--2010},
  volume={286},
  series={Proceedings of Machine Learning Research},
  publisher={PMLR},
  year={2025}
}
```

Verified against https://proceedings.mlr.press/v286/kasa25a.html:
- Pages 1990-2010 — CORRECT
- Volume 286 — CORRECT
- Author list (Kasa, Zhang, Yang, Taylor) — CORRECT
- Conference name "Forty-First Conference on Uncertainty in Artificial Intelligence" — CORRECT

**Status: No errors.**

---

### 1.3 `kasa2023empirically` — CONFIRMED CORRECT (appropriate format)

```bibtex
@misc{kasa2023empirically,
  title={Empirically Validating Conformal Prediction on Modern Vision Architectures...},
  author={Kasa, Kevin and Taylor, Graham W.},
  howpublished={arXiv:2307.01088. Presented at ICML 2023 Workshop on Structured
                Probabilistic Inference \& Generative Modeling},
  year={2023}
}
```

Verified against arXiv:2307.01088 and OpenReview:
- Authors (Kasa, Taylor — 2 authors only) — CORRECT
- Workshop: ICML 2023 SPIGM workshop — CORRECT
- arXiv ID 2307.01088 — CORRECT
- @misc with howpublished is the appropriate entry type for a workshop paper without PMLR proceedings

**Status: No errors.**

---

### 1.4 `lundberg2018consistent` — APPROPRIATE AS CITED (with one note)

```bibtex
@article{lundberg2018consistent,
  title={Consistent Individualized Feature Attribution for Tree Ensembles},
  author={Lundberg, Scott M and Erion, Gabriel G and Lee, Su-In},
  journal={arXiv preprint arXiv:1802.03888},
  year={2018}
}
```

This paper (arXiv:1802.03888) introduced TreeSHAP's consistency theorem. It was never published as a standalone peer-reviewed article; the implementation and extensions were incorporated into `lundberg2020local` (Nature Machine Intelligence 2020). Citing the 2018 arXiv preprint as the original TreeSHAP consistency reference is academically correct. The text in Section 3.3 appropriately cites both: `\citep{lundberg2018consistent,lundberg2020local}`.

**Status: No errors. The dual citation is correct and expected.**

---

## 2. Full BibTeX Audit — All Entries

### CONFIRMED ERRORS

#### ERROR 1: `malinin2021shifts` — WRONG AUTHOR ORDER and MISSING AUTHOR

**Current bib entry (reconstructed from the file)**:
```
author={Malinin, Andrey and Band, Neil and Chesnokov, German and Gal, Yarin
        and Gales, Mark JF and Noskov, Alexey and Ploskonosov, Andrey
        and Prokhorenkova, Liudmila and Provilkov, Ivan and Raina, Vatsal and others}
```

**Actual author order** (verified against the official NeurIPS Datasets and Benchmarks proceedings page at datasets-benchmarks-proceedings.neurips.cc):

1. Andrey Malinin
2. Neil Band
3. Yarin Gal  ← **appears 3rd, not 4th**
4. Mark J. F. Gales  ← **appears 4th, not 5th**
5. Alexander Ganshin  ← **MISSING ENTIRELY**
6. German Chesnokov  ← **appears 6th, not 3rd**
7. Alexey Noskov
8. Andrey Ploskonosov
9. Liudmila Prokhorenkova
10. Ivan Provilkov
11. Vatsal Raina
12. Vyas Raina  ← second Raina missing before `others`
13. Denis Roginskiy
14. Mariya Shmatova
15. Panagiotis (Panos) Tigas
16. Boris Yangel

**Problems**:
1. Chesnokov and Gal are transposed: bib has Band → Chesnokov → Gal, actual is Band → Gal → Gales → Ganshin → Chesnokov.
2. Alexander Ganshin (5th author) is absent — absorbed into `others`, causing Chesnokov and Gal to appear in wrong positions.
3. Vyas Raina (second Raina, 12th author) is absorbed into `others` before being explicitly named.

**Fix**:
```bibtex
author={Malinin, Andrey and Band, Neil and Gal, Yarin and Gales, Mark J. F.
        and Ganshin, Alexander and Chesnokov, German and Noskov, Alexey
        and Ploskonosov, Andrey and Prokhorenkova, Liudmila and Provilkov, Ivan
        and Raina, Vatsal and Raina, Vyas and others}
```

**Severity**: Medium. The first five named authors are wrong in order, and a named author (Ganshin) is silently dropped. In plainnat style, only the first author and "et al." appear in running text, so this does not affect rendering of inline citations — but it is a factual metadata error that would be caught in camera-ready checks.

---

### CONFIRMED CORRECT ENTRIES

The following were individually audited and contain no errors:

| Key | Venue | Vol/Pages | Status |
|-----|-------|-----------|--------|
| `vovk2005algorithmic` | Springer book | — | Correct |
| `romano2020classification` | NeurIPS 2020, vol 33, pp 3581-3591 | Confirmed | Correct |
| `romano2019conformalized` | NeurIPS 2019, vol 32 | No pages given; minor omission only | Correct |
| `tibshirani2019conformal` | NeurIPS 2019, vol 32 | No pages given; consistent with romano2019 style | Correct |
| `gibbs2021adaptive` | NeurIPS 2021, vol 34, pp 1660-1672 | Confirmed | Correct |
| `barber2023conformal` | Annals of Statistics, vol 51, no 2, pp 816-845, 2023 | Confirmed | Correct |
| `angelopoulos2023gentle` | F&T in ML, vol 16, no 4, pp 494-591, 2023 | Confirmed correct (peer-reviewed 2023 version) | Correct |
| `zaffran2022adaptive` | ICML 2022, PMLR v162, pp 25834-25866 | Confirmed | Correct |
| `podkopaev2021distribution` | UAI 2021, PMLR v161, pp 844-853 | Confirmed | Correct |
| `lundberg2017unified` | NeurIPS 2017, vol 30 | Correct |
| `koh2021wilds` | ICML 2021, PMLR v139, pp 5637-5664 | Confirmed | Correct |
| `gulrajani2021search` | ICLR 2021 | Correct; no pages field (ICLR norm) |
| `lundberg2020local` | Nature Machine Intelligence, vol 2, no 1, pp 56-67, 2020 | Confirmed | Correct |
| `fey2024relbench` | NeurIPS 2024 D&B Track | 12 authors confirmed correct in R6 | Correct |
| `feldman2023achieving` | TMLR 2023 | 4 authors confirmed correct in prior rounds | Correct |
| `angelopoulos2024conformal` | ICLR 2024 | Correct; venue correct (NOT JASA) |
| `garg2022leveraging` | ICLR 2022 | Correct |
| `angelopoulos2021uncertainty` | ICLR 2021 | Correct |
| `gretton2012kernel` | JMLR vol 13, pp 723-773, 2012 | Confirmed | Correct |
| `lopez2017revisiting` | ICLR 2017 | Correct |
| `lei2018distribution` | JASA vol 113, no 523, pp 1094-1111, 2018 | Confirmed | Correct |
| `dua2017uci` | @misc with howpublished | Fixed in prior rounds; correct URL used | Correct |
| `siddiqi2006credit` | Wiley book | Correct |
| `ding2023class` | NeurIPS 2023, vol 36 | Correct |
| `cauchois2021knowing` | JMLR vol 22, no 81, pp 1-42, 2021 | Confirmed | Correct |
| `shafer2008tutorial` | JMLR vol 9, pp 371-421, 2008 | Confirmed | Correct |
| `gibbs2024conformal` | JMLR vol 25, no 162, pp 1-36, 2024 | Confirmed | Correct |
| `bhatnagar2023improved` | ICML 2023, PMLR, pp 2337-2363 | Confirmed | Correct |
| `gardner2023tableshift` | NeurIPS 2023 D&B Track | Confirmed | Correct |
| `ke2017lightgbm` | NeurIPS 2017, vol 30 | Pages 3146-3154 missing (minor omission) | Acceptable |
| `miller2021accuracy` | ICML 2021, PMLR v139, pp 7721-7735 | Confirmed | Correct |

---

### MINOR OMISSIONS (Non-Errors)

**`ke2017lightgbm`**: Missing `pages={3146--3154}`. Actual pages confirmed as 3146-3154 (NeurIPS proceedings). This is a minor omission that does not affect correctness but is mildly sloppy. Consistent with how some other NeurIPS entries in this file also omit page numbers (`romano2019conformalized`, `tibshirani2019conformal`, `lundberg2017unified`). The omission is stylistically consistent, so this is cosmetic only.

**`malinin2021shifts` booktitle**: Currently "Advances in Neural Information Processing Systems Datasets and Benchmarks Track". The official proceedings URL is datasets-benchmarks-proceedings.neurips.cc; a more precise booktitle would be "Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)". However, the current abbreviated form is widely accepted in the field and is not wrong.

---

## 3. Citation Accuracy in Text

### Claims verified against cited papers:

**Claim**: "citet{vovk2005algorithmic} introduced conformal prediction with exchangeability guarantees"
**Assessment**: Correct. The 2005 Springer book is the canonical source.

**Claim**: "citet{shafer2008tutorial} provide an accessible tutorial"
**Assessment**: Correct. The 2008 JMLR tutorial is the standard pedagogical reference.

**Claim**: "citet{ding2023class} show that class-conditional coverage guarantees become harder to achieve as class count grows"
**Assessment**: Correct characterization of the paper's contribution.

**Claim**: "citet{gibbs2025conditional} provide finite-sample conditional guarantees via a covariate-shift reweighting framework interpolating between marginal and conditional validity"
**Assessment**: Correct and accurate. The paper develops methods interpolating marginal and conditional CP, confirmed.

**Claim**: "citet{tibshirani2019conformal} study covariate shift with known propensity scores"
**Assessment**: Correct.

**Claim**: "citet{barber2023conformal} provide a comprehensive treatment beyond exchangeability, bounding coverage loss by the total variation distance between test and calibration score distributions"
**Assessment**: Correct characterization of Theorem 1 in Barber et al. 2023.

**Claim**: "citet{kasa2023empirically} empirically characterize how CP degrades across many vision architectures and shift types"
**Assessment**: Correct. The paper covers multiple architectures (ViT, ResNet families) and distribution shift conditions.

**Claim**: "citet{kasa2025adapting} propose entropy-based methods (ECP/EACP) that adapt prediction sets using unlabeled test data at deployment time"
**Assessment**: Correct. The paper proposes ECP (Entropy-based Conformal Prediction) and EACP.

**Claim**: "citet{gulrajani2021search} show domain generalization methods often fail to improve over ERM under realistic distribution shifts"
**Assessment**: Correct. This is the DomainBed finding. The paper does not restrict to temporal shift (a common mischaracterization fixed in earlier rounds); the text here says "realistic distribution shifts" — accurate.

**Claim**: "citet{gardner2023tableshift} benchmark tabular distribution shift across 15 tasks"
**Assessment**: Correct. TableShift contains 15 binary classification tasks.

**Claim**: "citet{angelopoulos2021uncertainty} [RAPS]"
**Assessment**: The cited paper is "Uncertainty Sets for Image Classifiers using Conformal Prediction" which introduces RAPS. Correct.

**Claim on PSI footnote**: "Population Stability Index (PSI): a monitoring metric standard in credit scoring practice [siddiqi2006credit]. PSI > 0.002 indicates detectable nonzero shift; conventional significance threshold is > 0.1"
**Assessment**: The Siddiqi book is the appropriate citation for PSI as a credit scoring monitoring metric. Threshold values (0.002, 0.1) are practitioner conventions not explicitly cited — the footnote format (not citing a specific page for the thresholds) is appropriate since these are field standards.

**Claim**: "citet{lundberg2018consistent,lundberg2020local} [TreeSHAP TreeExplainer]"
**Assessment**: Correct dual citation. The 2018 preprint introduced the consistency theorem and fast TreeSHAP algorithm; the 2020 Nature MI paper introduced TreeExplainer with clinical applications.

---

## 4. Missing References Assessment

Checking whether any critical references expected at UAI 2026 are absent:

**`lei2018distribution`** (JASA 2018, split CP canonical reference): Present. Good.

**`angelopoulos2023gentle`** (F&T 2023 survey): Present. Good.

**`romano2019conformalized`** (CQR, regression CP): Present (cited in Related Work for regression extension). Good.

**`gibbs2024conformal`** (ACI extension, JMLR 2024): Present. Good.

**Barber, Candès, Ramdas, Tibshirani (2023) Annals of Statistics**: Present as `barber2023conformal`. Good.

**Cauchois, Gupta, Duchi (2021) JMLR**: Present as `cauchois2021knowing`. Good.

**Potential gap**: The paper does not cite Shafer & Vovk (2008) specifically as a tutorial recommendation in the methods section — but it is cited in Related Work. Sufficient.

**Potential gap**: Conformal Risk Control (Angelopoulos et al., ICLR 2024) is cited as `angelopoulos2024conformal`. Good.

**Assessment**: No critical missing references identified for a UAI 2026 paper in this specific scope (tabular CP + shift diagnostics). The bibliography is well-curated for the paper's claims.

---

## 5. Overall Assessment

### Confirmed Errors Requiring Fix: 1

**Priority 1 (Fix before submission):**

- **`malinin2021shifts` author order and missing author**: Gal and Chesnokov are transposed; Alexander Ganshin (5th author in official proceedings) is absent. This is a factual metadata error. Fix by correcting the author field as specified in Section 2 above.

### Confirmed Correct (No Fix Needed): All other entries

The four entries flagged for special attention (`kasa2023empirically`, `kasa2025adapting`, `gibbs2025conditional`, `lundberg2018consistent`) are all verified correct.

### Cosmetic / Optional Improvements:

- Add `pages={3146--3154}` to `ke2017lightgbm` (confirmed from NeurIPS proceedings)
- Optionally expand `malinin2021shifts` booktitle to the full official name

---

## 6. Recommended Fix

In `references.bib`, update `malinin2021shifts` author field:

**Current (incorrect)**:
```
author={Malinin, Andrey and Band, Neil and Chesnokov, German and Gal, Yarin and Gales, Mark JF and Noskov, Alexey and Ploskonosov, Andrey and Prokhorenkova, Liudmila and Provilkov, Ivan and Raina, Vatsal and others},
```

**Corrected**:
```
author={Malinin, Andrey and Band, Neil and Gal, Yarin and Gales, Mark J. F. and Ganshin, Alexander and Chesnokov, German and Noskov, Alexey and Ploskonosov, Andrey and Prokhorenkova, Liudmila and Provilkov, Ivan and Raina, Vatsal and Raina, Vyas and others},
```

---

## 7. Verdict

**Round 21 finding**: 1 confirmed error (`malinin2021shifts` author order/missing author). All other entries are clean. The bibliography is in excellent shape after 20 prior rounds of correction. Fix the Malinin author field and the paper is ready for submission.

---

*Sources verified against:*
- [Oxford Academic JRSS-B, Vol. 87 Issue 4](https://academic.oup.com/jrsssb/article-abstract/87/4/1100/8058684)
- [PMLR v286 kasa25a](https://proceedings.mlr.press/v286/kasa25a.html)
- [NeurIPS 2021 D&B Shifts proceedings page](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/ad61ab143223efbc24c7d2583be69251-Abstract-round2.html)
- [arXiv 2307.01088 (kasa2023empirically)](https://arxiv.org/abs/2307.01088)
- [JMLR v25/22-1218 (gibbs2024conformal)](https://jmlr.org/papers/v25/22-1218.html)
- [PMLR v162/zaffran22a (zaffran2022adaptive)](https://proceedings.mlr.press/v162/zaffran22a.html)
