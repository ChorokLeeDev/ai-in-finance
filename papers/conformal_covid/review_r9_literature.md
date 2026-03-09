# Literature Review Report — Final BibTeX Audit (R9)

**Paper:** Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study
**Target venue:** UAI 2026
**BibTeX file:** `uai_2026/references.bib` (24 entries)
**Review date:** 2026-02-20
**Reviewer note:** R8 already corrected `fey2024relbench` author list and `feldman2023achieving` author list. This round audits all 24 entries end-to-end.

---

## Entry-by-Entry Audit

### 1. `vovk2005algorithmic`
```
@book{vovk2005algorithmic, year=2005, publisher=Springer}
```
**Status: PASS.** Vovk, Gammerman, Shafer (2005) Springer. Correct.

---

### 2. `romano2020classification`
```
@inproceedings{romano2020classification, volume=33, pages=3581--3591, year=2020}
```
**Status: PASS.** NeurIPS 2020, vol. 33, pp. 3581–3591. Verified correct.

---

### 3. `romano2019conformalized`
```
@inproceedings{romano2019conformalized, volume=32, year=2019}
```
**Status: MINOR.** NeurIPS 2019, vol. 32. Pages not present; standard practice omits pages for NeurIPS 2019 (pre-paginated proceedings). Acceptable. Entry type `@inproceedings` is correct.

**Booktitle style note:** This entry uses bare `"Advances in Neural Information Processing Systems"` while `koh2021wilds` and `adebayo2018sanity` append `(ICML)` / `(NeurIPS)` parenthetical abbreviations in their booktitles. See issue 24 below (style inconsistency).

---

### 4. `tibshirani2019conformal`
```
@inproceedings{tibshirani2019conformal, volume=32, year=2019}
```
**Status: PASS.** NeurIPS 2019, vol. 32. Entry type `@inproceedings` consistent with `romano2019conformalized`. No pages, consistent.

---

### 5. `gibbs2021adaptive`
```
@inproceedings{gibbs2021adaptive, volume=34, pages=1660--1672, year=2021}
```
**Status: PASS.** Confirmed: NeurIPS 2021, vol. 34, pp. 1660–1672. Entry type `@inproceedings` is correct. Note: the extended JMLR 2024 version (Gibbs & Candes, "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts", JMLR vol. 25, arXiv 2208.08401) is not cited, but the paper does not directly use ACI in a way that requires the JMLR extension. Acceptable for UAI.

---

### 6. `barber2023conformal`
```
@article{barber2023conformal, journal={The Annals of Statistics}, volume=51, number=2, pages=816--845, year=2023}
```
**Status: PASS.** Annals of Statistics 51(2):816–845, 2023. Confirmed correct.

---

### 7. `angelopoulos2021gentle`
```
@article{angelopoulos2021gentle, journal={Foundations and Trends in Machine Learning}, volume=16, number=4, pages=494--591, year=2023}
```
**Status: PASS with NOTE.** The body correctly reflects the F&T in ML 2023 publication (vol. 16, no. 4, pp. 494–591). The key `angelopoulos2021gentle` is misleading (arXiv year 2021) but `year=2023` is correct in the entry, so rendered output is accurate. For cleanliness, key could be renamed `angelopoulos2023gentle`, but this is a low-priority cosmetic issue. No factual error.

---

### 8. `zaffran2022adaptive`
```
@inproceedings{zaffran2022adaptive, booktitle={International Conference on Machine Learning}, pages=25834--25866, year=2022}
```
**Status: PASS.** ICML 2022, PMLR vol. 162, pp. 25834–25866. Pages verified correct. Missing `volume=162` and `series={Proceedings of Machine Learning Research}` and `publisher={PMLR}`, but these are optional fields and their absence is standard. Acceptable.

---

### 9. `podkopaev2021distribution`
```
@inproceedings{podkopaev2021distribution,
  booktitle={Proceedings of the 37th Conference on Uncertainty in Artificial Intelligence},
  editor={de Campos, Cassio and Maathuis, Marloes},
  pages=844--853, volume=161,
  series={Proceedings of Machine Learning Research}, publisher={PMLR}, year=2021,
  url={https://proceedings.mlr.press/v161/podkopaev21a.html}
}
```
**Status: PASS.** UAI 2021, PMLR vol. 161, pp. 844–853. Verified correct. This is the most complete PMLR entry in the file. No issues.

---

### 10. `lundberg2017unified`
```
@inproceedings{lundberg2017unified, booktitle={Advances in Neural Information Processing Systems}, volume=30, year=2017}
```
**Status: PASS with MINOR.** NeurIPS 2017, vol. 30, pp. 4766–4777. Pages are absent but the entry type and volume are correct. Adding `pages=4766--4777` would improve completeness but is not required.

---

### 11. `koh2021wilds`
```
@inproceedings{koh2021wilds,
  booktitle={International Conference on Machine Learning (ICML)},
  pages=5637--5664, year=2021, organization={PMLR}
}
```
**Status: MINOR — style inconsistency.**

Two issues:
1. **Booktitle style:** Uses `"International Conference on Machine Learning (ICML)"` with parenthetical abbreviation. Most entries in this file use bare names without abbreviations. This is inconsistent but not factually wrong.
2. **`organization` vs `publisher`:** Standard BibTeX convention for PMLR proceedings uses `publisher={PMLR}` not `organization={PMLR}`. `organization` is a valid field for `@inproceedings` but semantically misapplied here — PMLR is the publisher, not an organizing body. Low-priority.
3. **Missing `volume=139`:** PMLR vol. 139. Not required, but consistent with how `podkopaev2021distribution` is formatted (which includes `volume`).

---

### 12. `gulrajani2020search`
```
@inproceedings{gulrajani2020search, booktitle={International Conference on Learning Representations (ICLR)}, year=2021}
```
**Status: MINOR — key/year mismatch.**

Key is `gulrajani2020search` (arXiv year 2020) but `year=2021` (ICLR publication year). The `year=2021` is factually correct. The mismatch is only in the cite key itself, which does not affect rendered output. Low-priority cosmetic issue. No factual error.

**Content description check:** The paper's description in the text — "show domain generalization methods often fail to improve over ERM under realistic distribution shifts" — is correct per our memory notes. No paraphrase error detected in main.tex.

---

### 13. `adebayo2018sanity`
```
@inproceedings{adebayo2018sanity,
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume=31, year=2018}
```
**Status: MINOR — style inconsistency.**

Booktitle appends `(NeurIPS)` parenthetical, inconsistent with `romano2020classification`, `romano2019conformalized`, `tibshirani2019conformal`, `gibbs2021adaptive` which all use bare `"Advances in Neural Information Processing Systems"`. Pages (9525–9536) absent but acceptable.

---

### 14. `malinin2021shifts`
```
@inproceedings{malinin2021shifts,
  booktitle={Advances in Neural Information Processing Systems Datasets and Benchmarks Track},
  year=2021}
```
**Status: PASS.** Entry type `@inproceedings` is correct (confirmed: published in NeurIPS 2021 D&B Track proceedings at datasets-benchmarks-proceedings.neurips.cc). Booktitle is accurate. No pages or volume available for D&B Track 2021 in the same format as main proceedings; absence is acceptable. No issues.

---

### 15. `lundberg2020local`
```
@article{lundberg2020local,
  journal={Nature Machine Intelligence}, volume=2, number=1, pages=56--67, year=2020,
  publisher={Nature Publishing Group}}
```
**Status: PASS.** Nature Machine Intelligence 2(1):56–67, 2020. Correct.

---

### 16. `fey2024relbench`
```
@inproceedings{fey2024relbench,
  author={Robinson, Joshua and Ranjan, Rishabh and Hu, Weihua and Huang, Kexin
          and Han, Jiaqi and Dobles, Alejandro and Fey, Matthias and Lenssen, Jan Eric
          and Yuan, Yiwen and Zhang, Zecheng and He, Xinwei and Leskovec, Jure},
  booktitle={Advances in Neural Information Processing Systems Track on Datasets and Benchmarks},
  year=2024}
```
**Status: PASS.** Author list corrected in R8. All 12 authors verified correct. Booktitle for NeurIPS 2024 D&B Track is accurate. No `volume` field — NeurIPS 2024 main conference is vol. 37, but D&B Track does not share this volume; omitting `volume` is correct. No issues.

---

### 17. `feldman2023achieving`
```
@article{feldman2023achieving,
  author={Feldman, Shai and Ringel, Liran and Bates, Stephen and Romano, Yaniv},
  journal={Transactions on Machine Learning Research}, year=2023}
```
**Status: PASS.** Author list corrected in R8 (Ringel and Romano present; Angelopoulos removed). Entry type `@article` is correct (TMLR is a journal). No volume/number/pages — TMLR does not use traditional pagination; this is standard practice for TMLR citations. No issues.

---

### 18. `angelopoulos2024conformal`
```
@inproceedings{angelopoulos2024conformal,
  title={Conformal Risk Control},
  author={Angelopoulos, Anastasios N and Bates, Stephen and Fisch, Adam and Lei, Lihua and Schuster, Tal},
  booktitle={International Conference on Learning Representations}, year=2024}
```
**Status: PASS.** Confirmed ICLR 2024. Authors: Angelopoulos, Bates, Fisch, Lei, Schuster — all 5 verified correct. Entry type `@inproceedings` correct. No issues.

---

### 19. `garg2022leveraging`
```
@inproceedings{garg2022leveraging,
  author={Garg, Saurabh and Balakrishnan, Sivaraman and Lipton, Zachary C and Neyshabur, Behnam and Sedghi, Hanie},
  booktitle={International Conference on Learning Representations}, year=2022}
```
**Status: PASS.** Confirmed ICLR 2022. Authors: Garg, Balakrishnan, Lipton, Neyshabur, Sedghi — all 5 verified correct. No issues.

---

### 20. `angelopoulos2021uncertainty`
```
@inproceedings{angelopoulos2021uncertainty,
  title={Uncertainty Sets for Image Classifiers using Conformal Prediction},
  author={Angelopoulos, Anastasios N and Bates, Stephen and Malik, Jitendra and Jordan, Michael I},
  booktitle={International Conference on Learning Representations}, year=2021}
```
**Status: PASS.** ICLR 2021 (RAPS paper). Authors Angelopoulos, Bates, Malik, Jordan confirmed correct. No issues.

---

### 21. `gretton2012kernel`
```
@article{gretton2012kernel,
  journal={Journal of Machine Learning Research}, volume=13, pages=723--773, year=2012}
```
**Status: PASS.** JMLR 13:723–773, 2012. Confirmed correct.

---

### 22. `lopez2017revisiting`
```
@inproceedings{lopez2017revisiting,
  booktitle={International Conference on Learning Representations}, year=2017}
```
**Status: PASS.** ICLR 2017. Confirmed correct.

---

### 23. `lei2018distribution`
```
@article{lei2018distribution,
  journal={Journal of the American Statistical Association},
  volume=113, number=523, pages=1094--1111, year=2018}
```
**Status: PASS.** JASA 113(523):1094–1111, 2018. Confirmed correct. Authors Lei, G'Sell, Rinaldo, Tibshirani, Wasserman — correct.

---

### 24. `dua2017uci`
```
@misc{dua2017uci,
  author={Dua, Dheeru and Graff, Casey},
  title={{UCI} Machine Learning Repository},
  year=2017,
  howpublished={University of California, Irvine, School of Information and Computer Sciences},
  url={https://archive.ics.uci.edu/ml}
}
```
**Status: MINOR.**

Two issues:
1. **`howpublished` misuse:** The field currently contains an institution description (`"University of California, Irvine, School of Information and Computer Sciences"`), which reads awkwardly as a `howpublished` value. Standard usage is `howpublished={\url{https://archive.ics.uci.edu/ml}}` or `howpublished={University of California, Irvine. \url{https://archive.ics.uci.edu/ml}}` combining both. The current entry separates the institution into `howpublished` and the URL into a standalone `url` field; both will render but `howpublished` should contain the primary access information. Low-priority.
2. **URL is HTTP/outdated path:** `https://archive.ics.uci.edu/ml` redirects to `https://archive.ics.uci.edu`. The new UCI ML Repository URL is `https://archive.ics.uci.edu`. Minor but worth updating.

---

## Summary Table

| # | Key | Entry Type | Status | Issue Severity |
|---|-----|-----------|--------|---------------|
| 1 | `vovk2005algorithmic` | @book | PASS | — |
| 2 | `romano2020classification` | @inproceedings | PASS | — |
| 3 | `romano2019conformalized` | @inproceedings | PASS | — |
| 4 | `tibshirani2019conformal` | @inproceedings | PASS | — |
| 5 | `gibbs2021adaptive` | @inproceedings | PASS | — |
| 6 | `barber2023conformal` | @article | PASS | — |
| 7 | `angelopoulos2021gentle` | @article | NOTE | Key name misleading (2021 vs year=2023); no factual error |
| 8 | `zaffran2022adaptive` | @inproceedings | PASS | — |
| 9 | `podkopaev2021distribution` | @inproceedings | PASS | — |
| 10 | `lundberg2017unified` | @inproceedings | MINOR | Pages absent (4766–4777); not required |
| 11 | `koh2021wilds` | @inproceedings | MINOR | Booktitle style inconsistency; `organization` vs `publisher` |
| 12 | `gulrajani2020search` | @inproceedings | NOTE | Key year (2020) vs year=2021; cosmetic only |
| 13 | `adebayo2018sanity` | @inproceedings | MINOR | Booktitle style inconsistency with `(NeurIPS)` suffix |
| 14 | `malinin2021shifts` | @inproceedings | PASS | — |
| 15 | `lundberg2020local` | @article | PASS | — |
| 16 | `fey2024relbench` | @inproceedings | PASS | R8 fix confirmed correct |
| 17 | `feldman2023achieving` | @article | PASS | R8 fix confirmed correct |
| 18 | `angelopoulos2024conformal` | @inproceedings | PASS | — |
| 19 | `garg2022leveraging` | @inproceedings | PASS | — |
| 20 | `angelopoulos2021uncertainty` | @inproceedings | PASS | — |
| 21 | `gretton2012kernel` | @article | PASS | — |
| 22 | `lopez2017revisiting` | @inproceedings | PASS | — |
| 23 | `lei2018distribution` | @article | PASS | — |
| 24 | `dua2017uci` | @misc | MINOR | `howpublished` structure; outdated URL path |

**PASS: 18 entries. NOTE (cosmetic): 2 entries. MINOR (no factual error): 4 entries. FAIL: 0 entries.**

---

## Consolidated Issues

### Booktitle Style Inconsistency (entries 11, 13)

Three booktitle styles coexist in the file:
- Bare name: `"Advances in Neural Information Processing Systems"` (entries 2–5, 10)
- With parenthetical abbreviation: `"Advances in Neural Information Processing Systems (NeurIPS)"` (entry 13), `"International Conference on Machine Learning (ICML)"` (entry 11)
- Spelled out without abbreviation: `"International Conference on Learning Representations"` (entries 18–20, 22)

**Recommendation:** Standardize all NeurIPS entries to bare `"Advances in Neural Information Processing Systems"` (drop `(NeurIPS)` suffix from `adebayo2018sanity`). For ICML, either add `volume=139` to `koh2021wilds` or drop the `(ICML)` suffix. UAI's `plainnat` style renders whatever is in `booktitle`, so inconsistency is visible to readers.

**Specific fix for `adebayo2018sanity`:**
```bibtex
booktitle={Advances in Neural Information Processing Systems},
```

**Specific fix for `koh2021wilds`:**
```bibtex
booktitle={International Conference on Machine Learning},
organization={PMLR},
```
or preferably:
```bibtex
booktitle={Proceedings of the 38th International Conference on Machine Learning},
volume={139},
series={Proceedings of Machine Learning Research},
publisher={PMLR},
```

### `dua2017uci` Cleanup (entry 24)

**Recommended fix:**
```bibtex
@misc{dua2017uci,
  author = {Dua, Dheeru and Graff, Casey},
  title = {{UCI} Machine Learning Repository},
  year = {2017},
  howpublished = {University of California, Irvine, School of Information and Computer Sciences. \url{https://archive.ics.uci.edu}},
}
```

### Key Name Cosmetic Issues (entries 7, 12)

- `angelopoulos2021gentle`: key year is 2021 (arXiv), but `year=2023` (F&T publication). Rendered citations will say 2023. No action required unless house style requires key-year consistency.
- `gulrajani2020search`: key year is 2020 (arXiv), but `year=2021` (ICLR). Same pattern. No action required.

---

## Missing Citation Check

The paper uses PSI (Population Stability Index) as a shift detector in Section 5.3. Per our memory notes, PSI has no canonical academic citation; it is a practitioner heuristic from credit scoring. The paper does not attempt to cite PSI, which is appropriate. No action needed.

The paper uses ACI from Gibbs & Candes (2021). The extended JMLR 2024 version exists but is not required given the paper's use of ACI is for comparison rather than as a primary method. Acceptable.

The paper cites `shafer` / `vovk` foundational work but does not cite Shafer & Vovk (2008) JMLR tutorial separately. Only the 2005 book is cited. For UAI this is borderline acceptable; the book covers the same material. No blocking issue.

---

## Overall Assessment

**No factual errors remain in any of the 24 entries.** All author lists, venues, entry types, and years are correct. The R8 fixes to `fey2024relbench` and `feldman2023achieving` are confirmed accurate.

The four MINOR items (booktitle style inconsistency, `dua2017uci` URL) are polish-level issues that are unlikely to trigger reviewer concern but are worth fixing before camera-ready submission.

**Recommended priority order for final cleanup:**
1. Standardize NeurIPS booktitles (drop `(NeurIPS)` suffix from `adebayo2018sanity`): 1 line change.
2. Standardize ICML booktitle for `koh2021wilds` (drop `(ICML)` suffix): 1 line change.
3. Fix `dua2017uci` `howpublished` to combine institution and URL, update URL path.
4. (Optional) Add pages to `lundberg2017unified`.

**Verdict: Ready for submission. No blocking issues.**
