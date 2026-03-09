# 문헌 리뷰 보고서 (Round 3) — Literature Review Report

**검토 대상**: `papers/conformal_covid/uai_2026/main.tex`
**검토 기준일**: 2026-02-20
**라운드**: Round 3 (Round 1 수정 반영 후 재검토)

---

## 논문 개요 (Paper Overview)

- **제목**: "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"
- **제출처**: UAI 2026
- **핵심 기여**: SHAP concentration을 사전 배포(pre-deployment) 진단 지표로 활용하여 conformal prediction의 distribution shift 하에서의 실패 심각도를 예측. COVID-19를 자연 실험으로 활용하여 8개 공급망 task에서 커버리지 저하(0.1%~77.1%)를 분석. 이론적으로 APS score inflation이 concentration과 단조적 관계임을 증명.
- **현재 인용문헌**: 총 19개 (references.bib 기준)

---

## 1. 기존 연구와의 차별성 평가 (Differentiation Assessment)

**평점: 4/5**

### 강점

- 기존 CP 이론 연구(Tibshirani 2019, Barber 2023)가 "shift 하에서 어떻게 성능이 떨어지는가"를 이론적으로 분석한 것과 달리, 본 논문은 "어떤 모델이 사전 배포 시점에 실패할지"를 예측하는 진단 도구를 제안하는 명확한 포지셔닝을 갖는다.
- MMD, C2ST, PSI가 모든 task에서 동일하게 shift를 감지하지만 심각도를 구분하지 못한다는 empirical 증거를 통해 차별성이 잘 확립되어 있다.
- Garg et al. (2022)의 "unlabeled test data 활용" 방법과의 비교를 통해 pre-deployment vs. test-time 관측 구분이 명확하다.

### 간과된 관련 연구

아래 논문들은 본 논문과 주제적으로 관련되며 Related Work에서 언급하거나 차별화해야 할 필요성이 있다:

1. **Gibbs & Candès (2024) — "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts"**
   JMLR Vol. 25 (2024). ACI(2021)의 확장으로, step-size를 시간에 따라 적응시켜 임의적 분포 변화에 대응. 현재 논문은 `gibbs2021adaptive`만 인용하고 있으나, 동일 저자의 JMLR 2024 후속 논문이 존재한다. 특히 ACI 관련 실험(Section 6.1)에서 이 논문을 언급하는 것이 적절하다.
   URL: https://jmlr.org/papers/v25/22-1218.html

2. **Bhatnagar et al. (2023) — "Improved Online Conformal Prediction via Strongly Adaptive Online Learning"**
   ICML 2023. ACI 계열 개선 방법으로, strongly adaptive regret 최소화를 통해 모든 구간에서 일관된 커버리지를 달성. zaffran2022adaptive와 함께 ACI 확장 맥락에서 언급할 수 있다.
   URL: https://proceedings.mlr.press/v202/bhatnagar23a.html

3. **ACM Computing Surveys — "Conformal Prediction: A Data Perspective" (2025)**
   CP 전반에 대한 survey. 향후 검토자가 survey 인용을 요구할 가능성이 있다.
   URL: https://dl.acm.org/doi/10.1145/3736575

4. **Quiñonero-Candela et al. (2009) — "Dataset Shift in Machine Learning"**
   MIT Press. covariate shift, label shift, dataset shift 분류의 표준 참고서. Related Work의 Shift Detection 절에서 shift 유형 분류의 이론적 배경으로 인용을 고려할 수 있다 (필수는 아님).

5. **RelBench NeurIPS 2024 버전**: `fey2023relbench`의 BibTeX에 `year={2023}`으로 되어 있으나, 실제 NeurIPS 2024 Datasets and Benchmarks Track에 게재됨. 연도 및 venue 수정 필요 (아래 Section 3 참조).

### 개선 필요 사항

- ACI 실험을 다루는 Section 6.1에서 Gibbs & Candès 2024 (JMLR) 버전을 보조 인용으로 추가하는 것을 권장. 이 논문은 ACI를 단순 적용이 아닌 개선된 형태로 발전시켰기 때문에, 현재 논문이 "ACI가 커버리지를 회복하지만 informativeness를 희생한다"는 결론을 내리는 맥락에서 최신 방법론과의 비교 맥락이 약해질 수 있다.

---

## 2. 연구 갭 충족도 평가 (Research Gap Analysis)

**평점: 4.5/5**

### 식별된 갭의 명확성

Introduction 첫 단락에서 갭을 명확하게 제시: "While prior work characterizes how conformal prediction degrades under shift, a critical gap remains: Can we identify which deployed models will fail before observing test data?" — 이는 명료하고 설득력 있다.

### 갭 충족 여부 및 정도

- **SALT 내부 갭 충족**: 8개 task에서 ρ=0.833, p=0.010으로 통계적으로 유의미한 연관성 확인. 충분히 갭을 충족한다.
- **이론적 갭 충족**: Theorem 1이 APS score inflation의 단조성을 형식화하여 empirical 관찰에 이론적 근거 제공.
- **Cross-domain 갭**: 16개 task, 9개 도메인으로 확장하여 ρ=0.853, p<0.001 달성. 부분적 전이성(external catastrophic 증거가 Covertype에 편중)을 솔직하게 인정한 것은 강점.

### 논리적 일관성 평가

- SHAP concentration이 pre-deployment에서 계산 가능하다는 점이 일관되게 강조됨.
- Binary ceiling effect 설명이 진단의 적용 범위를 명확히 한정하는 좋은 논리적 보완이다.
- Model-specificity (boosting 모델에서만 강함) 인정이 솔직하고 논리적 일관성을 유지한다.

### 추가로 다뤄야 할 갭

- **실시간 모니터링과 pre-deployment 진단의 결합**: 논문은 Section 7에서 coverage 모니터링을 권장하지만, empirical coverage 모니터링 자체가 online conformal prediction 문헌(Gibbs 2021, 2024; Bhatnagar 2023)과 어떻게 연결되는지에 대한 논의가 부족하다.
- **Calibration set size sensitivity**: 대규모 캘리브레이션 셋(sales-level n_cal=35,737; item-level n_cal=146,948) 맥락에서 concentration이 안정적이라는 주장은 잘 지지되나, 소규모 데이터셋에서의 일반화 갭이 언급되지 않는다.

---

## 3. 인용문헌 품질 평가 (Citation Quality Assessment)

**평점: 3.5/5**

### 인용문헌 통계

| 구분 | 수치 |
|------|------|
| 총 인용 수 | 19개 |
| 2005 이전 | 1개 (Vovk 2005) |
| 2012-2018 | 4개 |
| 2019-2021 | 10개 |
| 2022-2024 | 4개 |
| 2025+ | 0개 |

### BibTeX 오류 및 수정 필요 사항

#### [오류 1] `fey2023relbench` — 연도 및 venue 오류 (중요)

```bibtex
% 현재 (오류):
@article{fey2023relbench,
  title={RelBench: A Benchmark for Deep Learning on Relational Databases},
  author={Fey, Matthias and others},
  journal={arXiv preprint},
  note={arXiv:2407.20060},
  year={2023}
}
```

RelBench는 실제로 **NeurIPS 2024 Datasets and Benchmarks Track**에 정식 게재되었다.
출처: https://proceedings.neurips.cc/paper_files/paper/2024/file/25cd345233c65fac1fec0ce61d0f7836-Paper-Datasets_and_Benchmarks_Track.pdf

```bibtex
% 수정안:
@inproceedings{fey2024relbench,
  title={{RelBench}: A Benchmark for Deep Learning on Relational Databases},
  author={Fey, Matthias and Hu, Weihua and Huang, Kexin and Lenssen, Jan Eric
          and Rishi, Rishabh and Robinson, Joshua and Sriram, Anirudh
          and Olivares, Alejandro and others},
  booktitle={Advances in Neural Information Processing Systems (Datasets and Benchmarks Track)},
  year={2024}
}
```

**논문 본문에서 `fey2023relbench` -> `fey2024relbench`로 key 변경 필요** (Section 3.1, Appendix A.2).
arXiv 제출 연도(2024년 7월)와 실제 NeurIPS 2024 게재를 구분하여 올바른 peer-reviewed 버전을 인용해야 한다.

#### [오류 2] `angelopoulos2021gentle` — 버전 불일치

```bibtex
% 현재 (불완전):
@article{angelopoulos2021gentle,
  title={A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification},
  author={Angelopoulos, Anastasios N and Bates, Stephen},
  journal={arXiv preprint arXiv:2107.07511},
  year={2021}
}
```

이 논문은 **Foundations and Trends in Machine Learning, Vol. 16, No. 4, pp. 494-591 (2023)**으로 정식 출판되었다.
출처: https://dl.acm.org/doi/10.1561/2200000101

UAI는 peer-reviewed 저널 버전 인용을 선호한다. 수정안:

```bibtex
@article{angelopoulos2023gentle,
  title={Conformal Prediction: A Gentle Introduction},
  author={Angelopoulos, Anastasios N and Bates, Stephen},
  journal={Foundations and Trends in Machine Learning},
  volume={16},
  number={4},
  pages={494--591},
  year={2023},
  publisher={Now Publishers}
}
```

**note**: 에이전트 메모리에 이미 기록된 것처럼, arXiv 버전과 F&T 버전은 별도로 인용하는 것이 정확하다. 논문 key를 `angelopoulos2023gentle`로 변경하면 본문 인용 key도 갱신 필요.

#### [오류 3] `feldman2023achieving` — venue 오류

```bibtex
% 현재 (오류):
@inproceedings{feldman2023achieving,
  title={Achieving Risk Control in Online Learning Settings},
  author={Feldman, Shai and Bates, Stephen and Romano, Yaniv},
  booktitle={Transactions on Machine Learning Research},  % TMLR는 journal
  year={2023}
}
```

TMLR는 conference proceeding이 아닌 저널이다. `@article`로 변경 필요:

```bibtex
@article{feldman2023achieving,
  title={Achieving Risk Control in Online Learning Settings},
  author={Feldman, Shai and Bates, Stephen and Romano, Yaniv},
  journal={Transactions on Machine Learning Research},
  year={2023}
}
```

#### [확인 필요] `gulrajani2020search` — 연도 불일치

```bibtex
@inproceedings{gulrajani2020search,
  title={In search of lost domain generalization},
  author={Gulrajani, Ishaan and Lopez-Paz, David},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

key가 `gulrajani2020`이지만 `year=2021`이다 (arXiv는 2020년, ICLR 발표는 2021년). key를 `gulrajani2021search`로 통일하거나 연도를 일관되게 유지할 것을 권장.

#### [형식 이슈] `fey2023relbench` — arXiv ID 없음

현재 `note={arXiv:2407.20060}` 형식으로 처리하고 있으나, NeurIPS 2024 버전으로 갱신하면 이 이슈는 해결된다.

### 누락된 핵심 문헌

#### 필수 추가 권장 (Priority High)

**1. Gibbs & Candès (2024) — JMLR Vol. 25**
"Conformal Inference for Online Prediction with Arbitrary Distribution Shifts"
ACI의 직접적 후속 연구. 현재 `gibbs2021adaptive`를 인용하지만 2024년 JMLR 확장판이 존재한다. Section 6.1 ACI 실험 논의 시 각주 또는 보조 인용으로 추가 권장.

```bibtex
@article{gibbs2024conformal,
  title={Conformal Inference for Online Prediction with Arbitrary Distribution Shifts},
  author={Gibbs, Isaac and Cand{\`e}s, Emmanuel},
  journal={Journal of Machine Learning Research},
  volume={25},
  pages={1--36},
  year={2024}
}
```

**2. Shafer & Vovk (2008) — JMLR Vol. 9, pp. 371–421**
"A Tutorial on Conformal Prediction"
CP 분야에서 Vovk 2005 book과 함께 핵심 기초 문헌으로 인용되는 논문. 현재 `vovk2005algorithmic`만 인용하고 있다. UAI 리뷰어 중 교환가능성 기초에 관심 있는 이론 지향 리뷰어는 이 논문의 누락을 지적할 가능성이 있다.
*단, 이미 `angelopoulos2021gentle`가 tutorial 역할을 일부 대체하므로 필수는 아니며, 추가하면 차별성이 높아진다.*

#### 선택적 추가 권장 (Priority Medium)

**3. Bhatnagar et al. (2023) — ICML**
"Improved Online Conformal Prediction via Strongly Adaptive Online Learning"
ACI 관련 실험을 다루는 Related Work 또는 Section 6.1에서 언급하면 관련 문헌 커버리지가 향상된다. 필수는 아니나, ACI 계열 방법론 섹션에서 `zaffran2022adaptive` 옆에 추가하면 좋다.

**4. Quiñonero-Candela et al. (2009) — MIT Press**
"Dataset Shift in Machine Learning"
covariate shift, label shift, temporal shift 분류의 고전적 참고서. Related Work의 Shift Detection 절에서 shift 유형을 언급할 때 보조 인용으로 적절하다. 필수는 아님.

### 적절성 이슈 (Correctness Issues)

#### [이슈 1] Related Work에서 `romano2019conformalized` 인용 맥락 확인 필요

Related Work에서:
> "Recent work extends to classification~\citep{romano2020classification} and regression~\citep{romano2019conformalized,lei2018distribution}."

`romano2019conformalized`(CQR)는 quantile regression을 conformalize하는 방법이며, `lei2018distribution`은 split CP의 기초 논문이다. 두 논문 모두 regression에 적절하다. 문제 없음.

#### [이슈 2] `angelopoulos2024conformal` 인용 맥락 적절성 확인

Related Work에서:
> "\citet{angelopoulos2024conformal} generalize conformal prediction to broader risk measures."

Conformal Risk Control (ICLR 2024)의 내용과 일치한다. 에이전트 메모리에 기록된 것처럼 venue(ICLR 2024)가 이미 수정되어 정확하다.

#### [이슈 3] `adebayo2018sanity` 인용 맥락 검토

Related Work에서:
> "Our work connects SHAP~\citep{lundberg2017unified} to model reliability assessment, extending the use of feature attribution from debugging~\citep{adebayo2018sanity} to prospective failure diagnosis."

Adebayo et al. (2018) "Sanity Checks for Saliency Maps"는 gradient saliency map의 건전성을 검증하는 논문으로, SHAP (Lundberg 2017)와는 다른 계열의 feature attribution 방법이다. "feature attribution from debugging"이라는 맥락에서 인용은 적절하나, SHAP와의 관계를 더 직접적으로 연결하는 논문을 함께 인용하는 것을 고려할 수 있다.

예: **Ribeiro et al. (2016) LIME** 또는 **Molnar (2022) "Interpretable Machine Learning" book**. 단, 이는 선택적 개선사항이며 필수 수정이 아니다.

#### [이슈 4] `lundberg2020local` — 인용 정확성

TreeExplainer(Lundberg 2020, Nature Machine Intelligence) 인용은 정확하다. `lundberg2017unified`(NeurIPS 2017)와 `lundberg2020local`(NMI 2020)을 함께 인용하는 현재 구조는 적절하다.

### UCI 데이터셋 개별 원본 인용 필요성

**결론: `dua2017uci` 저장소 단일 인용으로 충분하다.**

외부 validation에 사용된 Covertype, KDDCup99, Gas Sensor, Pendigits, PAMAP2, Avila, Satimage, Shuttle 등은 모두 UCI ML Repository에서 획득한 것으로, 현재 `dua2017uci` 인용이 이 역할을 담당한다. UAI 투고 관행상 UCI 전체 저장소 인용으로 충분하며, 개별 데이터셋 원본 논문 인용은 필수가 아니다.

단, **Gas Sensor Array Drift Dataset**은 UCI #224로 Vergara et al. (2012) 또는 Fonollosa et al. (2015)가 원본 논문이므로, Section 6.4 또는 Appendix에서 이 데이터셋을 별도로 설명할 때 각주로 원본을 인용하면 좋다. 현재 본문에는 이 데이터셋이 `dua2017uci`로만 처리되어 있어 문제가 없다.

### 최신성 평가

| 연도 구간 | 인용 수 | 비율 |
|-----------|---------|------|
| 2022-2024 | 4 | 21% |
| 2019-2021 | 10 | 53% |
| ~2018 | 5 | 26% |

UAI 2026 제출 논문으로서 2022~2024 인용이 21%는 다소 낮은 편이다. Gibbs & Candès 2024 JMLR 버전 추가를 통해 최신성을 소폭 개선할 수 있다. 다만, 핵심 CP 이론 문헌이 2019-2021에 집중되어 있다는 것은 이 분야의 특성을 반영하므로 치명적 약점은 아니다.

---

## 4. 종합 평가 및 권고사항 (Overall Assessment & Recommendations)

**종합 평점: 4/5**

### 주요 강점 (3가지)

1. **명확한 갭과 실용적 기여**: "어떤 모델이 실패할지 사전에 예측"이라는 갭이 명확하며, SHAP concentration을 활용한 pre-deployment 진단 프레임워크가 실용적이다.

2. **엄밀한 통계적 처리**: 50 seed 앙상블, paired Wilcoxon 검정, bootstrap CI, LOO 분석, ICC 분석 등 다층적 통계 검증이 충실히 이루어졌다.

3. **솔직한 한계 인정**: model-specificity(boosting에 한정), n=8 탐색적 임계값, KDDCup99 false negative 등을 회피하지 않고 명시적으로 다룬다.

### 주요 개선사항 (우선순위 순)

**[Priority 1 — 필수 수정]**
`fey2023relbench` BibTeX 오류 수정: `year=2023` → `year=2024`, `journal={arXiv preprint}` → NeurIPS 2024 Datasets and Benchmarks Track 정식 인용으로 변경. 논문 key도 `fey2024relbench`로 갱신하고 본문의 모든 인용 key를 업데이트.

**[Priority 2 — 필수 수정]**
`angelopoulos2021gentle` BibTeX 오류 수정: arXiv 2021 → Foundations and Trends in Machine Learning 2023 (Vol. 16, No. 4, pp. 494–591) 정식 버전으로 갱신. Key를 `angelopoulos2023gentle`로 변경하거나 `year=2023` 및 journal/volume/pages 추가.

**[Priority 3 — 필수 수정]**
`feldman2023achieving` venue 오류 수정: `@inproceedings` → `@article` (TMLR는 저널).

**[Priority 4 — 강력 권장]**
Gibbs & Candès (2024) JMLR 버전 추가: Section 6.1 ACI 실험 논의에서 `gibbs2021adaptive`의 후속 JMLR 2024 논문을 보조 인용으로 추가. 이는 최신 ACI 문헌을 포괄함을 보여주는 동시에 "ACI가 개선되었음에도 informativeness 문제가 남아있다"는 논점 강화에 기여한다.

**[Priority 5 — 선택적 개선]**
`gulrajani2020search` key와 연도 일관성 수정: key를 `gulrajani2021search`로 변경하거나 note에 arXiv 2020, ICLR 2021 정보를 추가.

**[Priority 6 — 선택적 개선]**
Shafer & Vovk (2008) "A Tutorial on Conformal Prediction" (JMLR) 추가: 이론적 근거를 강화하려는 리뷰어를 위해 `vovk2005algorithmic` 옆에 함께 인용하는 것을 고려. 필수는 아니나 UAI 이론 지향 리뷰어 대응에 유용.

### 게재 권고 여부

**Accept (Minor Revision)**

BibTeX 오류 3건(Priority 1~3)은 technical fix이며, 내용적 변경 없이 수정 가능하다. 현재 논문의 실험 설계, 이론적 기여, 통계 검증은 충분한 수준이다. R10 시뮬레이션 리뷰어 점수(8.0/8.0/7.5)와 일관된 평가로, BibTeX 수정 후 제출 준비가 완료된 상태로 판단된다.

---

## 요약: 즉시 수정이 필요한 BibTeX 변경사항

| 우선순위 | 항목 | 현재 | 수정 |
|---------|------|------|------|
| P1 | `fey2023relbench` | year=2023, arXiv | year=2024, NeurIPS Datasets & Benchmarks |
| P2 | `angelopoulos2021gentle` | year=2021, arXiv | year=2023, F&T in Machine Learning Vol.16 No.4 |
| P3 | `feldman2023achieving` | `@inproceedings` | `@article` (TMLR는 저널) |
| P4 | (신규 추가) | 없음 | `gibbs2024conformal` JMLR 2024 추가 |
| P5 | `gulrajani2020search` | key와 year 불일치 | key를 `gulrajani2021search`로 수정 |
