# Literature Review Report — UAI 2026 Submission (Round 7)

**Paper:** "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"
**File:** `papers/conformal_covid/uai_2026/main.tex`
**BibTeX:** `papers/conformal_covid/uai_2026/references.bib`
**Review date:** 2026-02-20
**Previous round (R6) fixes verified:** tibshirani2019conformal @article→@inproceedings, malinin2021shifts arXiv→@inproceedings D&B Track, Gulrajani description corrected.

---

## 1. Paper Overview

The paper proposes SHAP concentration as a pre-deployment diagnostic for conformal prediction failure under distribution shift, using COVID-19 as a natural experiment across 8 supply chain tasks. It provides a formal theorem (score inflation), empirical correlation results (rho=0.853, n=16), and an operational decision framework.

---

## 2. Differentiation from Existing Research

**Rating: 4/5**

The paper clearly positions itself at the intersection of conformal prediction theory and practical shift diagnostics. Its pre-deployment framing is genuinely novel — prior work characterizes failure post-hoc or requires test-time data. Related work section correctly distinguishes from Garg et al. (2022) (requires unlabeled test data), Tibshirani et al. (shift-aware but assumes known propensity scores), and Barber et al. (theoretical bounds, not diagnostic).

One potentially relevant gap: Mondrian/class-conditional conformal methods (Venn predictors) are not mentioned despite being a standard technique when class-label shift is the mechanism. This is a minor omission since the paper's claim is not about remediation but diagnosis.

---

## 3. Research Gap Analysis

**Rating: 4/5**

The gap ("can we predict which models fail before seeing test data?") is clearly stated and the paper addresses it directly. The formal theorem, cross-domain validation, and placebo test strengthen the gap-filling claim. The decision framework provides operational closure.

The main residual gap (acknowledged in Discussion) is prospective validation — all evidence is associative from a single supply chain benchmark plus 9 external datasets. The paper handles this honestly.

---

## 4. Citation Quality Assessment

**Rating: 3.5/5**

### 4.1 Entries with Confirmed Issues

#### ISSUE 1 (REMAINING from R6): `gibbs2021adaptive` — Wrong entry type

```bibtex
@article{gibbs2021adaptive,
  ...
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={1660--1672},
  year={2021}
}
```

NeurIPS is a conference proceedings, not a journal. The entry should be `@inproceedings` with `booktitle`, not `@article` with `journal`. This is the same inconsistency pattern noted in MEMORY.md (tibshirani was fixed in R6; gibbs was not). The NeurIPS proceedings URL is `proceedings.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html`.

**Fix:**
```bibtex
@inproceedings{gibbs2021adaptive,
  title={Adaptive Conformal Inference Under Distribution Shift},
  author={Gibbs, Isaac and Cand{\`e}s, Emmanuel},
  booktitle={Advances in Neural Information Processing Systems},
  volume={34},
  pages={1660--1672},
  year={2021}
}
```

#### ISSUE 2: `fey2024relbench` — Booktitle omits D&B Track

```bibtex
@inproceedings{fey2024relbench,
  ...
  booktitle={Advances in Neural Information Processing Systems},
  volume={37},
  year={2024}
}
```

RelBench was published in the NeurIPS 2024 **Datasets and Benchmarks Track**, not the main NeurIPS conference track. The main conference volume is 37, but D&B is a separate track with its own proceedings (confirmed at `neurips.cc/Conferences/2024/CallForDatasetsBenchmarks` and the paper's official NeurIPS page ending in `-Paper-Datasets_and_Benchmarks_Track.pdf`). Using the main conference booktitle with volume=37 misattributes the track, which is the same error pattern documented in MEMORY.md for RelBench.

**Fix:**
```bibtex
@inproceedings{fey2024relbench,
  title={RelBench: A Benchmark for Deep Learning on Relational Databases},
  author={Fey, Matthias and Hu, Weihua and Huang, Kexin and Lenssen, Jan Eric and Ranjan, Rishabh and Robinson, Joshua and Ying, Rex and You, Jiaxuan and Leskovec, Jure},
  booktitle={Advances in Neural Information Processing Systems Track on Datasets and Benchmarks},
  year={2024}
}
```

#### ISSUE 3: `gulrajani2020search` — Key year mismatch (minor, cosmetic)

```bibtex
@inproceedings{gulrajani2020search,
  ...
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

The key says `2020` (arXiv submission date) but `year=2021` (ICLR publication). The entry type and booktitle are now correct (fixed in R6). The key mismatch is cosmetic — `plainnat` renders the year from the `year` field, not the key — but stylistically inconsistent. This was flagged in MEMORY.md as a known pattern. No in-text citation will render incorrectly since plainnat uses the year field. No immediate action required unless the venue has style-guide requirements for key consistency.

#### ISSUE 4: `malinin2021shifts` — Booktitle wording check

```bibtex
@inproceedings{malinin2021shifts,
  ...
  booktitle={Advances in Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2021}
}
```

The actual proceedings are published at `datasets-benchmarks-proceedings.neurips.cc` under the title "Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks". The current booktitle is an acceptable informal rendering that reviewers will recognize; it is not wrong, but for full consistency with the fey2024relbench fix above, both should use the same wording pattern. See Fix for Issue 2.

#### ISSUE 5: `dua2017uci` — Entry type is `@misc` with no `howpublished`

```bibtex
@misc{dua2017uci,
  author = {Dua, Dheeru and Graff, Casey},
  title = {{UCI} Machine Learning Repository},
  year = {2017},
  url = {http://archive.ics.uci.edu/ml},
  institution = {University of California, Irvine, School of Information and Computer Sciences}
}
```

Minor issues:
- `institution` is not a standard BibTeX field for `@misc`; it is silently ignored by most processors. The canonical form uses `howpublished = {University of California, Irvine, School of Information and Computer Sciences}`.
- The URL uses HTTP; the current UCI repository is at `https://archive.ics.uci.edu/ml`.
- The `year=2017` is the original repository citation year; newer datasets used in this paper (Gas Sensor, PAMAP2, etc.) postdate 2017. This is acceptable practice for repository-level citations.

**Fix:**
```bibtex
@misc{dua2017uci,
  author = {Dua, Dheeru and Graff, Casey},
  title = {{UCI} Machine Learning Repository},
  howpublished = {University of California, Irvine, School of Information and Computer Sciences},
  year = {2017},
  url = {https://archive.ics.uci.edu/ml}
}
```

### 4.2 Entries Confirmed Correct

| Key | Finding |
|-----|---------|
| `vovk2005algorithmic` | Correct @book, Springer 2005 |
| `romano2020classification` | Correct @inproceedings, NeurIPS vol 33, pages 3581-3591 |
| `romano2019conformalized` | Correct @inproceedings, NeurIPS vol 32 |
| `tibshirani2019conformal` | Correct @inproceedings (fixed R6), NeurIPS vol 32 |
| `barber2023conformal` | Correct @article, Annals of Statistics 51(2), 816-845 |
| `angelopoulos2021gentle` | Correct @article, F&T in ML vol 16(4), 494-591, year=2023 |
| `zaffran2022adaptive` | Correct @inproceedings, ICML 2022, pages 25834-25866 |
| `podkopaev2021distribution` | Correct @inproceedings, UAI 2021, PMLR vol 161, pages 844-853 |
| `lundberg2017unified` | Correct @inproceedings, NeurIPS vol 30 |
| `koh2021wilds` | Correct @inproceedings, ICML 2021, PMLR |
| `adebayo2018sanity` | Correct @inproceedings, NeurIPS vol 31 |
| `lundberg2020local` | Correct @article, Nature Machine Intelligence 2(1), 56-67 |
| `feldman2023achieving` | Correct @article, TMLR 2023 (TMLR is a journal; @article is correct) |
| `angelopoulos2024conformal` | Correct @inproceedings, ICLR 2024 |
| `garg2022leveraging` | Correct @inproceedings, ICLR 2022 |
| `angelopoulos2021uncertainty` | Correct @inproceedings, ICLR 2021 (this is RAPS) |
| `gretton2012kernel` | Correct @article, JMLR vol 13, 723-773 |
| `lopez2017revisiting` | Correct @inproceedings, ICLR 2017 |
| `lei2018distribution` | Correct @article, JASA 113(523), 1094-1111 |
| `gulrajani2020search` | Correct type/venue (R6 fix confirmed); only key year cosmetic mismatch |

### 4.3 Missing Citations

The following papers are cited in the main text but have no BibTeX entry — all appear to be correctly accounted for. No orphan citations were found.

The following are potentially relevant missing citations that a UAI reviewer may notice:

1. **Shafer & Vovk (2008) JMLR tutorial** — "A Tutorial on Conformal Prediction," JMLR 9:371-421. The paper cites Vovk et al. 2005 book as the foundational reference but not the 2008 tutorial, which is the most-cited introductory reference alongside Angelopoulos & Bates. Absence is acceptable given that Angelopoulos & Bates (2021/2023) is cited and covers similar ground.

2. **Gibbs & Candès (2024) JMLR** — "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts," JMLR Vol. 25 (arXiv 2208.08401, published 2024). The paper cites Gibbs & Candès (2021) for ACI but not the extended 2024 JMLR paper. Reviewers working on online CP may flag this. Low priority.

3. **Bhatnagar et al. (2023) ICML** — "Improved Online Conformal Prediction via Strongly Adaptive Online Learning." An extension of ACI used in the paper; can be cited alongside `zaffran2022adaptive` in the Adaptive Methods paragraph.

---

## 5. Summary of Issues by Priority

| Priority | Issue | Key | Action |
|----------|-------|-----|--------|
| HIGH | Entry type: @article should be @inproceedings | `gibbs2021adaptive` | Change @article to @inproceedings; change `journal=` to `booktitle=` |
| HIGH | Booktitle omits D&B Track | `fey2024relbench` | Add "Track on Datasets and Benchmarks" to booktitle |
| MEDIUM | `institution` not a valid @misc field; HTTP URL | `dua2017uci` | Replace `institution` with `howpublished`; update URL to HTTPS |
| LOW | Key year (2020) vs. publication year (2021) cosmetic mismatch | `gulrajani2020search` | Optional rename to `gulrajani2021search` if style guide requires it |
| LOW | Booktitle wording inconsistency between malinin2021shifts and fey2024relbench | `malinin2021shifts` | Align wording with fey fix |

---

## 6. Overall Assessment

The bibliography is in good shape after R6 fixes. Two substantive errors remain: `gibbs2021adaptive` using `@article` for a conference paper (same NeurIPS-type issue as the tibshirani fix in R6), and `fey2024relbench` mis-attributing the main NeurIPS track instead of the D&B Track. Both are straightforward one-line fixes. The `dua2017uci` `institution` field is cosmetic but technically malformed BibTeX. All other 18 entries check out as correct.

**Recommendation:** Fix Issues 1 and 2 before final submission. Fix Issue 3 (dua) as a clean-up pass. Issues 4 and 5 are optional polish.

---

## 7. Differentiation, Gap, and Coverage Summary

**Differentiation: 4/5** — Genuinely novel pre-deployment framing. No closely competing concurrent work missed.

**Research Gap: 4/5** — Gap well-articulated and addressed. Observational/associative caveat appropriately disclosed.

**Citation Quality: 3.5/5** — Core foundational works all present. Two entry-type/track errors need fixing. No critical missing citations.

**Overall: 4/5** — Clear accept-level work. Minor bibliography corrections required.
