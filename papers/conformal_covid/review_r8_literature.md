# Literature Review Report — UAI 2026 Submission
## "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"

**Review Round:** R8 (Final BibTeX audit, 2026-02-20)
**Scope:** Final check of all 24 BibTeX entries and in-text citation accuracy after previous rounds of fixes.

---

## Paper Overview

- **Title:** Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study
- **Domain:** Conformal prediction, distribution shift, explainability, supply chain ML
- **Core claims:** SHAP concentration is a pre-deployment diagnostic for conformal vulnerability; formal theorem (Theorem 1) proves monotone score inflation; empirical validation across 16 multiclass tasks in 9 domains; rho=0.853, p<0.001.
- **Bibliography size:** 24 entries
- **Venue:** UAI 2026 (anonymous submission)

---

## 1. Differentiation from Existing Research

**Rating: 4/5**

The paper positions itself clearly against Tibshirani et al. (2019), Barber et al. (2023), and Gibbs et al. (2021). The pre-deployment framing (vs. adaptive/post-hoc methods) is a genuine distinguishing angle. The distinction between shift detection and severity prediction is well-stated and supported empirically.

No significant overlooked competitors were identified that would undermine the novelty claim. The diagnostic angle using SHAP on CP is not replicated in any 2023-2025 publication found.

---

## 2. Research Gap Analysis

**Rating: 4/5**

The gap is crisply articulated: existing work characterizes *how* CP degrades under shift but does not identify *which* models will fail before deployment. The paper fills exactly this gap. Logical consistency between gap identification, methodology (SHAP concentration on validation data only), and conclusion (pre-deployment diagnostic) is strong throughout.

---

## 3. Citation Quality Assessment — Full 24-Entry Audit

**Total entries:** 24
**Overall quality:** Good. Previous rounds resolved the most serious errors. This round identifies five remaining issues of varying severity.

---

### 3.1 Entry-by-Entry Findings

#### `vovk2005algorithmic` — CLEAN
```bibtex
@book{vovk2005algorithmic, publisher={Springer}, year={2005}}
```
Correct. Standard canonical citation for CP.

---

#### `romano2020classification` — CLEAN
```bibtex
@inproceedings{...booktitle={Advances in Neural Information Processing Systems}, volume={33}, pages={3581--3591}, year={2020}}
```
Correct for APS (NeurIPS 2020). Authors Romano, Sesia, Candes confirmed correct.

---

#### `romano2019conformalized` — CLEAN
```bibtex
@inproceedings{...booktitle={Advances in Neural Information Processing Systems}, volume={32}, year={2019}}
```
Correct for CQR (NeurIPS 2019).

---

#### `tibshirani2019conformal` — CLEAN (fixed in prior round)
```bibtex
@inproceedings{...booktitle={Advances in Neural Information Processing Systems}, volume={32}, year={2019}}
```
Correct. Previously was @article; now @inproceedings. Confirmed.

---

#### `gibbs2021adaptive` — CLEAN (fixed in prior round)
```bibtex
@inproceedings{...booktitle={Advances in Neural Information Processing Systems}, volume={34}, pages={1660--1672}, year={2021}}
```
Correct. Previously was @article; now @inproceedings. Confirmed.

---

#### `barber2023conformal` — CLEAN
```bibtex
@article{...journal={The Annals of Statistics}, volume={51}, number={2}, pages={816--845}, year={2023}}
```
Correct. Annals of Statistics 2023 is the right venue.

---

#### `angelopoulos2021gentle` — ISSUE: KEY YEAR MISMATCH (minor)
```bibtex
@article{angelopoulos2021gentle,
  ...
  journal={Foundations and Trends in Machine Learning},
  volume={16}, number={4}, pages={494--591},
  year={2023}
}
```
**Issue:** The BibTeX key is `angelopoulos2021gentle` (implying 2021) but `year={2023}` is correct for the peer-reviewed F&T publication (published 27 March 2023, Vol. 16, No. 4, pp. 494-591). The 2021 in the key refers to the arXiv preprint date. This is a common inconsistency — the key misleads readers into thinking the 2021 arXiv is being cited. The `year=` field is correct; only the key label is misleading.

**Severity:** Low — plainnat renders the year from the `year=` field (2023), so output is correct. However, reviewers who cross-check the key notice the discrepancy.

**Recommendation:** Change key to `angelopoulos2023gentle` and update the `\cite{}` call in main.tex. The year=2023 and volume/page data are already correct.

---

#### `zaffran2022adaptive` — CLEAN
```bibtex
@inproceedings{...booktitle={International Conference on Machine Learning}, pages={25834--25866}, year={2022}}
```
Correct. ICML 2022 confirmed.

---

#### `podkopaev2021distribution` — CLEAN (well-formed)
```bibtex
@inproceedings{...
  booktitle={Proceedings of the 37th Conference on Uncertainty in Artificial Intelligence},
  editor={...}, pages={844--853}, volume={161},
  series={Proceedings of Machine Learning Research}, publisher={PMLR},
  year={2021}, url={https://proceedings.mlr.press/v161/podkopaev21a.html}
}
```
Correct. UAI 2021 PMLR v161, pages 844-853 confirmed.

---

#### `lundberg2017unified` — CLEAN
```bibtex
@inproceedings{...booktitle={Advances in Neural Information Processing Systems}, volume={30}, year={2017}}
```
Correct. NeurIPS 2017, TreeSHAP / SHAP original paper.

---

#### `koh2021wilds` — MINOR STYLE NOTE
```bibtex
@inproceedings{...booktitle={International Conference on Machine Learning (ICML)}, pages={5637--5664}, year={2021}, organization={PMLR}}
```
Technically correct but inconsistent with other ICML entries (e.g., zaffran2022adaptive uses "International Conference on Machine Learning" without the "(ICML)" parenthetical). Not an error; cosmetic.

---

#### `gulrajani2020search` — ISSUE: KEY/YEAR INCONSISTENCY (minor)
```bibtex
@inproceedings{gulrajani2020search,
  ...
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```
**Issue:** Key is `gulrajani2020search` (arXiv year 2020) but `year={2021}` is the correct ICLR publication year. The in-text description is now correct (DomainBed/ERM framing — confirmed fixed in prior round). However, the key/year mismatch is a known inconsistency flagged in memory. The rendered citation will show 2021 correctly; the key is only misleading internally.

**Severity:** Low — cosmetic. Rendered output is correct. Changing the key to `gulrajani2021search` would require updating all \cite{} calls.

---

#### `adebayo2018sanity` — MINOR STYLE NOTE
```bibtex
@inproceedings{...booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, volume={31}, year={2018}}
```
Content correct. The "(NeurIPS)" parenthetical in booktitle is inconsistent with other NeurIPS entries that use bare "Advances in Neural Information Processing Systems". Cosmetic only.

---

#### `malinin2021shifts` — CLEAN (fixed in prior round)
```bibtex
@inproceedings{...booktitle={Advances in Neural Information Processing Systems Datasets and Benchmarks Track}, year={2021}}
```
**Status:** Previously was @article with arXiv journal. Now @inproceedings. The booktitle matches the canonical form for NeurIPS 2021 D&B Track papers. Confirmed correct.

**Note on author list:** The paper has a large author list (16 authors). The bib entry uses "others" via `\and` truncation. Verified: Band, Chesnokov, Gal, Gales, Noskov, Ploskonosov, Prokhorenkova, Provilkov, Raina (Vatsal), and others are all present in the actual author list. The truncation is acceptable for a 16-author paper.

---

#### `lundberg2020local` — CLEAN
```bibtex
@article{...journal={Nature Machine Intelligence}, volume={2}, number={1}, pages={56--67}, year={2020}}
```
Correct. Nature Machine Intelligence 2020, TreeExplainer paper. Verified.

---

#### `fey2024relbench` — ISSUE: AUTHOR LIST INCORRECT (moderate)
```bibtex
@inproceedings{fey2024relbench,
  title={RelBench: A Benchmark for Deep Learning on Relational Databases},
  author={Fey, Matthias and Hu, Weihua and Huang, Kexin and Lenssen, Jan Eric and Ranjan, Rishabh and Robinson, Joshua and Ying, Rex and You, Jiaxuan and Leskovec, Jure},
  booktitle={Advances in Neural Information Processing Systems Track on Datasets and Benchmarks},
  year={2024}
}
```
**Issue:** The actual NeurIPS 2024 D&B Track author list is: **Joshua Robinson, Rishabh Ranjan, Weihua Hu, Kexin Huang, Jiaqi Han, Alejandro Dobles, Matthias Fey, Jan Eric Lenssen, Yiwen Yuan, Zecheng Zhang, Xinwei He, Jure Leskovec** (12 authors). The bib entry omits: **Jiaqi Han, Alejandro Dobles, Yiwen Yuan, Zecheng Zhang, Xinwei He** and incorrectly includes **Rex Ying** and **Jiaxuan You** who are not authors of this paper (they are authors of earlier RelBench/PyG work).

**Severity:** Moderate. Including wrong authors is a factual error. Rex Ying and Jiaxuan You are prominent researchers who could notice their names wrongly attributed. The first author (Robinson) is also not listed first.

**Recommendation:** Replace the author field with the correct 12-author list or use "Robinson, Joshua and others" (acceptable for 12 authors). Correct author order: Robinson, Ranjan, Hu, Huang, Han, Dobles, Fey, Lenssen, Yuan, Zhang, He, Leskovec.

---

#### `feldman2023achieving` — ISSUE: AUTHOR LIST INCOMPLETE (minor)
```bibtex
@article{feldman2023achieving,
  title={Achieving Risk Control in Online Learning Settings},
  author={Feldman, Shai and Bates, Stephen and Angelopoulos, Anastasios N},
  journal={Transactions on Machine Learning Research},
  year={2023}
}
```
**Issue 1:** The actual author list is **Feldman, Shai; Ringel, Liran; Bates, Stephen; Romano, Yaniv** — four authors, not three. Liran Ringel (co-first author at Technion) and Yaniv Romano are missing; Angelopoulos is not an author of this paper.

**Severity:** Moderate. Wrong authors cited — Angelopoulos is attributed work he did not do on this paper; Romano and Ringel are omitted.

**Issue 2:** The TMLR entry should include `volume` and `issn` fields for completeness, though TMLR assigns article numbers rather than volume/page. The `@article` type is correct for TMLR.

**Recommendation:** Update author field to: `Feldman, Shai and Ringel, Liran and Bates, Stephen and Romano, Yaniv`.

---

#### `angelopoulos2024conformal` — CLEAN
```bibtex
@inproceedings{...booktitle={International Conference on Learning Representations}, year={2024}}
```
Correct. ICLR 2024 confirmed. (Note: This is a known common error in the field where it is sometimes misattributed to JASA — this entry correctly uses ICLR.)

---

#### `garg2022leveraging` — CLEAN
```bibtex
@inproceedings{...booktitle={International Conference on Learning Representations}, year={2022}}
```
Correct. ICLR 2022 confirmed.

---

#### `angelopoulos2021uncertainty` — CLEAN
```bibtex
@inproceedings{...booktitle={International Conference on Learning Representations}, year={2021}}
```
Correct. ICLR 2021 confirmed for RAPS paper.

---

#### `gretton2012kernel` — CLEAN
```bibtex
@article{...journal={Journal of Machine Learning Research}, volume={13}, pages={723--773}, year={2012}}
```
Correct. JMLR 2012, MMD paper. Verified.

---

#### `lopez2017revisiting` — CLEAN
```bibtex
@inproceedings{...booktitle={International Conference on Learning Representations}, year={2017}}
```
Correct. ICLR 2017, C2ST paper. Verified.

---

#### `lei2018distribution` — CLEAN
```bibtex
@article{...journal={Journal of the American Statistical Association}, volume={113}, number={523}, pages={1094--1111}, year={2018}}
```
Correct. JASA 2018, split conformal / regression paper. Verified. This is the canonical split CP citation; its presence is important.

---

#### `dua2017uci` — CLEAN (fixed in prior round)
```bibtex
@misc{dua2017uci,
  howpublished={University of California, Irvine, School of Information and Computer Sciences},
  url={https://archive.ics.uci.edu/ml}
}
```
`institution=` → `howpublished=` fixed. URL is HTTPS. Correct.

---

### 3.2 Summary of Issues

| Entry | Issue | Severity | Action Required |
|-------|-------|----------|-----------------|
| `angelopoulos2021gentle` | Key says 2021, year=2023 (correct); misleading key | Low | Rename key to `angelopoulos2023gentle` + update \cite{} |
| `gulrajani2020search` | Key says 2020, year=2021 (correct); misleading key | Low | Rename key to `gulrajani2021search` + update \cite{} (optional) |
| `fey2024relbench` | Author list has 2 wrong authors (Ying, You) and omits 5 actual authors | Moderate | Fix author field |
| `feldman2023achieving` | Wrong authors: Angelopoulos listed instead of Ringel + Romano missing | Moderate | Fix author field |
| `adebayo2018sanity`, `koh2021wilds` | Parenthetical "(NeurIPS)" / "(ICML)" in booktitle; inconsistent style | Cosmetic | Normalise booktitle style |

---

## 4. In-Text Citation Accuracy Checks

### 4.1 `\citet{angelopoulos2024conformal}` (line 78)
Text: "Angelopoulos et al. (2024) generalize conformal prediction to broader risk measures."

The cited paper is "Conformal Risk Control" (ICLR 2024) — correct framing. No issue.

### 4.2 `\citep{feldman2023achieving}` (line 78)
Text: "extensions by... Feldman et al. (2023) for online risk control."

The paper is correctly described. However, as noted above, the author list in the bib entry is wrong. The rendered citation will show "Feldman et al." which is correct (Feldman is the first author), but internally the bib has wrong authors.

### 4.3 `\citep{gulrajani2020search}` (line 80)
Text: "Gulrajani and Lopez-Paz (2021) show domain generalization methods often fail to improve over ERM under realistic distribution shifts."

The rendered year will be 2021 (correct — year={2021} in bib). Description is accurate per prior round fix. The in-text reference to "(2021)" may display as "(2020)" if natbib uses the key year rather than year= field — **verify the compiled PDF rendering**. In plainnat, the `year=` field governs the rendered year, so this should render as 2021. Low risk.

### 4.4 `\citet{garg2022leveraging}` (line 80)
Text: "Garg et al. (2022) use unlabeled test data to predict accuracy degradation."

Correct. ATC paper, ICLR 2022. Framing accurate.

### 4.5 PSI citation
In Section 3.5 and Section 5.4, PSI (Population Stability Index) is used as a shift detector. There is no citation for PSI in the bibliography. As noted in agent memory, PSI has no single canonical academic citation (it is a practitioner heuristic from credit scoring). The paper uses it as a baseline without citation, which is acceptable, but a footnote acknowledging this would be defensive against reviewer questions.

### 4.6 `\citep{vovk2005algorithmic}` — Missing Shafer & Vovk (2008)
The paper cites only Vovk et al. (2005) as the CP foundations reference. The companion tutorial Shafer & Vovk (2008) JMLR is not cited. At UAI this is not strictly required given the paper is empirical, but UAI reviewers with a theoretical CP background occasionally note its absence. Low priority for a paper that already cites the 2005 book.

---

## 5. Missing Citations Assessment

### 5.1 Gibbs & Candes (2024) JMLR
The paper cites `gibbs2021adaptive` (NeurIPS 2021 ACI). The extended version "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts" was published in JMLR Vol. 25, 2024. The ACI experiments in Section 5.1 would benefit from also citing the JMLR extension, especially since the paper uses ACI across all 8 tasks with non-stationary shifts. This is a low-priority addition but reviewers working on online CP will notice.

### 5.2 Romano, Sesia, Candes (2020) APS vs. Angelopoulos et al. (2021) RAPS
Both are properly cited. No gap.

### 5.3 Conformal prediction survey coverage
The paper cites `angelopoulos2021gentle` (F&T 2023) as a tutorial. This is sufficient.

---

## 6. Overall Assessment

### Ratings
- Differentiation: 4/5
- Research Gap: 4/5
- Citation Quality: 3.5/5 (two moderate author-list errors; otherwise clean)

### Priority Action List (ordered)

1. **[MODERATE — fix before submission]** `fey2024relbench`: Replace author field. Correct list: Robinson, Joshua and Ranjan, Rishabh and Hu, Weihua and Huang, Kexin and Han, Jiaqi and Dobles, Alejandro and Fey, Matthias and Lenssen, Jan Eric and Yuan, Yiwen and Zhang, Zecheng and He, Xinwei and Leskovec, Jure. (Remove Ying and You; add Han, Dobles, Yuan, Zhang, He.)

2. **[MODERATE — fix before submission]** `feldman2023achieving`: Replace author field with: Feldman, Shai and Ringel, Liran and Bates, Stephen and Romano, Yaniv. (Remove Angelopoulos; add Ringel and Romano.)

3. **[LOW — cosmetic, optional]** `angelopoulos2021gentle`: Rename key to `angelopoulos2023gentle` and update all `\cite{angelopoulos2021gentle}` calls in main.tex (appears on lines 74 and 78). The rendered output is correct already (year=2023).

4. **[LOW — cosmetic, optional]** `gulrajani2020search`: Rename key to `gulrajani2021search` and update \cite{} call (line 80). The rendered output is already correct (year=2021).

5. **[COSMETIC]** Normalise booktitle style: remove parenthetical "(ICML)" from `koh2021wilds` and "(NeurIPS)" from `adebayo2018sanity` to match the style of all other NeurIPS/ICML entries.

6. **[OPTIONAL]** Add a footnote when PSI is first introduced noting it is a practitioner heuristic without a single canonical academic citation (credit scoring origin).

7. **[OPTIONAL]** Consider adding Gibbs & Candes JMLR 2024 citation alongside the existing NeurIPS 2021 ACI citation for completeness in the ACI experiments section.

### Publication Recommendation
**Accept with Minor Revision** — the two moderate author-list errors (fey2024relbench and feldman2023achieving) are the only substantive issues remaining. All previously flagged structural errors have been corrected. The bibliography is otherwise clean and the in-text citation framing is accurate.

---

*Review generated by literature-reviewer agent on 2026-02-20. Search tools used to verify: Angelopoulos & Bates F&T 2023 (volume/pages confirmed); Feldman et al. TMLR (full author list confirmed); Conformal Risk Control ICLR 2024 (venue confirmed); RelBench NeurIPS 2024 D&B Track (author list confirmed via NeurIPS proceedings); Podkopaev & Ramdas UAI 2021 (PMLR v161 confirmed); Gulrajani & Lopez-Paz ICLR 2021 (year confirmed).*
