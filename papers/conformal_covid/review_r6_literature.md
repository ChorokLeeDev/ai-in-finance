# Literature Review Report — Round 6
**Paper:** "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"
**Venue target:** UAI 2026
**Review date:** 2026-02-20
**Reviewer:** Literature Review Agent (R6)

---

## Paper Overview

The paper proposes SHAP concentration (fraction of total SHAP importance in the top feature) as a pre-deployment diagnostic for conformal prediction vulnerability under distribution shift. Using 8 supply chain classification tasks from RelBench/SALT with a COVID-19 temporal split, it demonstrates that concentration correlates with coverage degradation (Spearman rho=0.853, p<0.001 across n=16 multiclass tasks in 9 domains). The paper includes a formal theorem (score inflation under concentrated feature shift), external validation across 9 non-supply-chain domains, and comparisons with standard shift detectors (MMD, C2ST, PSI).

---

## Status of R5 Issues

### Issue 1: NeurIPS entry-type inconsistency (tibshirani2019conformal vs romano2019conformalized)

**Status: NOT FIXED.**

The inconsistency persists. In `references.bib`:
- `tibshirani2019conformal` is `@article` (journal = "Advances in Neural Information Processing Systems")
- `romano2019conformalized` is `@inproceedings` (booktitle = "Advances in Neural Information Processing Systems")
- `gibbs2021adaptive` is `@article` (journal = "Advances in Neural Information Processing Systems")
- `romano2020classification` is `@inproceedings`
- `lundberg2017unified` is `@inproceedings`
- `adebayo2018sanity` is `@inproceedings`

Within the same .bib file, NeurIPS papers are split between `@article` and `@inproceedings`. The `plainnat` bibliography style tolerates either, but the inconsistency is visible and is a flag for reviewers. **Recommended fix:** standardize all NeurIPS papers to `@inproceedings` (the preferred form for NeurIPS proceedings). Affected entries: `tibshirani2019conformal`, `gibbs2021adaptive`.

### Issue 2: Gulrajani citation description ("robust methods often fail under temporal shift")

**Status: NOT FIXED.**

The paper still says (Section 2, Related Work, Shift Detection paragraph):
> "\citet{gulrajani2020search} show robust methods often fail under temporal shift."

This is an inaccurate characterization. The paper's actual main finding is that carefully tuned ERM outperforms domain generalization algorithms on average across DomainBed benchmarks — it is not specifically about temporal shift and does not make a general claim that "robust methods fail under temporal shift." The paper's BibTeX entry (`gulrajani2020search`) also has a year mismatch: the key uses `2020` (arXiv preprint date) while the `year=2021` field correctly gives the ICLR publication year.

**Recommended fix:** Change the sentence to accurately reflect the paper's contribution, e.g.:
> "\citet{gulrajani2020search} show that domain generalization methods often fail to improve over ERM under realistic distribution shifts."

The claim about temporal shift specifically should either be dropped or supported by a different citation.

### Issue 3: malinin2021shifts is arXiv only

**Status: FIXED.**

The current entry remains `@article{malinin2021shifts, journal={arXiv preprint arXiv:2107.07455}}`. However, the paper was formally published in the NeurIPS 2021 Datasets and Benchmarks Track proceedings (datasets-benchmarks-proceedings.neurips.cc). This is a peer-reviewed venue and should be cited as such.

**Recommended fix:**
```bibtex
@inproceedings{malinin2021shifts,
  title={Shifts: A Dataset of Real Distributional Shift Across Multiple Large-Scale Tasks},
  author={Malinin, Andrey and Band, Neil and ...},
  booktitle={Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
  volume={1},
  year={2021}
}
```

### Issue 4: Gibbs & Candes (2024) JMLR optional addition for ACI

**Status: NOT ADDRESSED.**

The paper cites only `gibbs2021adaptive` (NeurIPS 2021) for ACI. The extended JMLR 2024 paper ("Conformal Inference for Online Prediction with Arbitrary Distribution Shifts," JMLR Vol. 25, paper 22-1218, arXiv:2208.08401) is the published journal version with stronger guarantees. For a paper that runs ACI experiments across all 8 tasks and discusses its limitations in detail, citing only the workshop/conference version while the full journal paper exists is a gap. A UAI reviewer familiar with online CP will notice the omission.

**Recommended fix:** Add citation alongside `gibbs2021adaptive`:
```bibtex
@article{gibbs2024conformal,
  title={Conformal Inference for Online Prediction with Arbitrary Distribution Shifts},
  author={Gibbs, Isaac and Cand{\`e}s, Emmanuel},
  journal={Journal of Machine Learning Research},
  volume={25},
  number={162},
  pages={1--36},
  year={2024}
}
```

---

## New Issues Found in R6

### Issue 5: fey2024relbench — year in BibTeX key vs. NeurIPS D&B Track

**Status: New finding.**

The current entry:
```bibtex
@inproceedings{fey2024relbench,
  booktitle={Advances in Neural Information Processing Systems},
  volume={37},
  year={2024}
}
```

Confirmed: RelBench was accepted to the NeurIPS 2024 Datasets and Benchmarks Track (not the main conference track). The `booktitle` should reflect the D&B proceedings, not the main NeurIPS "Advances" proceedings. The volume=37 is the main NeurIPS conference volume; the D&B track uses its own proceedings identifier. This is a minor but technically incorrect citation form.

**Recommended fix:**
```bibtex
@inproceedings{fey2024relbench,
  title={{RelBench}: A Benchmark for Deep Learning on Relational Databases},
  author={Fey, Matthias and Hu, Weihua and ...},
  booktitle={Advances in Neural Information Processing Systems Track on Datasets and Benchmarks},
  year={2024}
}
```

### Issue 6: Shafer & Vovk (2008) JMLR tutorial not cited

**Status: New finding.**

The paper cites `vovk2005algorithmic` (book) in the introduction as the origin of conformal prediction, and `angelopoulos2021gentle` as a tutorial. However, the Shafer & Vovk (2008) JMLR tutorial ("A Tutorial on Conformal Prediction," JMLR vol. 9, pp. 371-421) is the canonical accessible exposition of the core exchangeability guarantee and is expected alongside the 2005 book in any conformal prediction paper at a top ML venue. UAI reviewers frequently check for this citation.

**Severity:** Low — the paper already cites the Angelopoulos & Bates gentle introduction, which serves a similar function. However, for completeness, especially given the formal guarantee framing in the introduction, the Shafer & Vovk 2008 reference could be added.

**Recommended fix:** Add in the Related Work opening sentence alongside `vovk2005algorithmic`:
```bibtex
@article{shafer2008tutorial,
  title={A Tutorial on Conformal Prediction},
  author={Shafer, Glenn and Vovk, Vladimir},
  journal={Journal of Machine Learning Research},
  volume={9},
  pages={371--421},
  year={2008}
}
```

### Issue 7: Missing citation for Bhatnagar et al. (2023) ICML on strongly adaptive online CP

**Status: New finding.**

Section 2 (Adaptive Methods) and Section 5.1 (ACI experiments) discuss ACI in depth but cite only `gibbs2021adaptive` and `zaffran2022adaptive`. Bhatnagar et al. (2023, ICML) "Improved Online Conformal Prediction via Strongly Adaptive Online Learning" is the most prominent 2023 ICML result on online conformal prediction and directly extends ACI with better adaptive regret bounds. A paper running ACI across 8 tasks and discussing its limitations should engage with this work. Omission is noticeable to reviewers active in online CP.

**Severity:** Moderate — the paper's ACI section is a secondary experiment, not a main contribution, but given the depth of the ACI discussion (Section 5.1, including usable-set rate analysis), this citation is expected.

**Recommended fix:** Add after `\citet{zaffran2022adaptive}` in Section 2:
> "and \citet{bhatnagar2023improved} for strongly adaptive regret bounds."

```bibtex
@inproceedings{bhatnagar2023improved,
  title={Improved Online Conformal Prediction via Strongly Adaptive Online Learning},
  author={Bhatnagar, Aadyot and Wang, Huan and Xiong, Caiming and Bai, Yu},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  pages={1--30},
  year={2023},
  organization={PMLR}
}
```

### Issue 8: PSI has no academic canonical citation

**Status: New finding (consistent with agent memory).**

Section 5.3 uses PSI (Population Stability Index) as a shift detector and cites no source for it. PSI is a practitioner heuristic from credit scoring with no canonical academic paper. The paper uses it without definition or citation. This is fine if PSI is treated as a standard tool, but a footnote acknowledging this (no single academic source) would pre-empt reviewer questions.

**Recommended fix:** Add a footnote in Section 5.3 on first mention of PSI:
> "PSI is a practitioner heuristic originating in credit scoring with no single canonical academic reference; we use the standard formulation PSI $= \sum_i (p_i - q_i)\ln(p_i/q_i)$."

---

## 1. Differentiation from Existing Research

**Rating: 4/5**

### Strengths
- The pre-deployment framing is genuinely novel: existing CP-under-shift work (Tibshirani 2019, Barber 2023, Gibbs 2021) characterizes how coverage degrades or how to adapt, but no prior work identifies which specific models will fail before test data is available.
- The COVID-19 natural experiment design — identical shift, heterogeneous outcomes — is a clean and credible identification strategy.
- The SHAP concentration metric is simple, model-specific, and operationally cheap, distinguishing it from post-hoc metrics.

### Areas for Improvement
- The Gulrajani citation (Issue 2 above) slightly misrepresents a well-known paper; a reviewer familiar with DomainBed will notice.
- The paper could explicitly acknowledge the "Conditional Coverage Diagnostics for Conformal Prediction" line of work (Arxiv 2512.11779, 2025), which also diagnoses CP reliability issues, though from a different angle (conditional vs. marginal coverage estimation).

### Overlooked Related Research (from search)
- "Not all distributional shifts are equal: Fine-grained robust conformal inference" (arXiv 2402.13042, 2024) addresses heterogeneous severity of CP degradation under different shift types — directly relevant to the paper's central observation.
- "Adapting Prediction Sets to Distribution Shifts Without Labels" (arXiv 2406.01416, 2024) — related to the pre-deployment framing (no labeled test data).

---

## 2. Research Gap Analysis

**Rating: 4/5**

The paper clearly articulates the gap (can we predict which models will fail before deployment?) and maintains logical consistency from introduction through conclusion. The placebo test and model-sensitivity appendix strengthen the gap-filling argument. The main remaining gap concerns generalizability to non-boosting models, which the paper acknowledges but does not fully resolve.

---

## 3. Citation Quality Assessment

**Rating: 3.5/5**

### Citation Statistics
- Total references: 22 entries
- Year distribution: 2005(1), 2008(0 — missing Shafer/Vovk), 2012(1), 2017(2), 2018(2), 2019(2), 2020(2), 2021(6), 2022(2), 2023(2), 2024(3)
- Most recent: fey2024relbench (2024)

### Issues Summary
| Priority | Issue | Type |
|----------|-------|------|
| HIGH | Gulrajani citation description inaccurate | Misrepresentation |
| HIGH | malinin2021shifts should be @inproceedings NeurIPS D&B | Wrong entry type/venue |
| MEDIUM | tibshirani2019conformal/@article vs romano2019conformalized/@inproceedings | Inconsistency |
| MEDIUM | Gibbs & Candes 2024 JMLR not cited (ACI extended version) | Missing |
| MEDIUM | Bhatnagar et al. 2023 ICML not cited | Missing |
| LOW | fey2024relbench booktitle should reflect D&B track | Minor inaccuracy |
| LOW | Shafer & Vovk 2008 JMLR tutorial not cited | Missing (optional) |
| LOW | PSI has no academic citation | Missing footnote |

### Positives
- Core CP foundations well covered (Vovk 2005, Romano 2019/2020, Tibshirani 2019, Barber 2023, Lei 2018)
- SHAP lineage correct (Lundberg 2017, Lundberg 2020 TreeExplainer)
- angelopoulos2021gentle correctly cited with F&T 2023 publication details
- angelopoulos2024conformal correctly cited as ICLR 2024
- feldman2023achieving correctly cited as TMLR @article
- dua2017uci cited for external validation datasets

---

## 4. Overall Assessment and Recommendations

**Overall Rating: 4/5**

### Top 3 Strengths
1. Novel pre-deployment framing with a clean natural experiment; no prior CP work addresses the "which model will fail" question before test data arrival.
2. Formal theorem with verified bounds + rich empirical evidence (50 seeds, 8 tasks, 9 domains, multiple model classes).
3. Honest reporting of limitations (exploratory threshold, model specificity, sparse external catastrophic cases).

### Priority Recommendations (ordered)

1. **Fix Gulrajani description** (HIGH): Change "show robust methods often fail under temporal shift" to accurately reflect the DomainBed/ERM finding. This is the most likely reviewer complaint.

2. **Fix malinin2021shifts entry type** (HIGH): Change from `@article{arXiv}` to `@inproceedings` citing the NeurIPS 2021 Datasets and Benchmarks Track proceedings.

3. **Standardize NeurIPS entry types** (MEDIUM): Convert `tibshirani2019conformal` and `gibbs2021adaptive` from `@article` to `@inproceedings` to match the other NeurIPS entries in the file.

4. **Add Gibbs & Candes 2024 JMLR** (MEDIUM): For the ACI section, cite the full published journal version alongside the 2021 NeurIPS paper.

5. **Add Bhatnagar et al. 2023 ICML** (MEDIUM): One sentence in the Adaptive Methods paragraph is sufficient.

6. **Fix fey2024relbench booktitle** (LOW): Reflect NeurIPS D&B track in booktitle.

7. **Add PSI footnote** (LOW): Note that PSI has no canonical academic citation.

8. **Consider adding Shafer & Vovk 2008** (LOW/OPTIONAL): Useful for completeness at a theory-oriented venue like UAI.

### Publication Recommendation

**Minor Revision** — The science and novelty are strong and the paper is close to acceptance-ready. The citation issues are correctable and none involve substantive scientific claims (except item 1, which is a mischaracterization of a cited paper). Fixing the Gulrajani description is the single most important action before resubmission.

---

*Sources consulted:*
- [Gibbs & Candes JMLR 2024](https://jmlr.org/papers/v25/22-1218.html)
- [Malinin et al. NeurIPS D&B 2021](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/ad61ab143223efbc24c7d2583be69251-Abstract-round2.html)
- [Shafer & Vovk JMLR 2008](https://jmlr.org/papers/v9/shafer08a.html)
- [Bhatnagar et al. ICML 2023](https://proceedings.mlr.press/v202/bhatnagar23a.html)
- [Gulrajani & Lopez-Paz ICLR 2021](https://openreview.net/forum?id=lQdXeXDoWtI)
- [RelBench NeurIPS 2024 D&B](https://neurips.cc/virtual/2024/poster/97659)
- [arXiv 2402.13042 fine-grained robust CP](https://arxiv.org/abs/2402.13042)
- [arXiv 2406.01416 adapting prediction sets without labels](https://arxiv.org/abs/2406.01416)
