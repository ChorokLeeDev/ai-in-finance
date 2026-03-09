# Round 3 Review Synthesis Report

**논문**: "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"
**제출처**: UAI 2026
**검토일**: 2026-02-20
**소스 에이전트**: Literature (문헌), Method (방법론), Insight (핵심 기여)

---

## EXECUTIVE SUMMARY

세 에이전트 분석 결과, 논문의 핵심 기여(SHAP concentration 기반 pre-deployment 진단)는 명확하고 실용적이며 통계적 검증이 충실하다. 그러나 **1개의 논리적 모순**(COVID-era n=9에 binary task 포함)과 **3건의 BibTeX 오류**, **dataset/domain counting 모호성**이 제출 전 수정이 필요하다. 이 문제들은 모두 30분 이내에 수정 가능한 범위이다.

---

## 1. 세 에이전트 공통 지적 사항 (최우선)

없음. 세 에이전트는 서로 다른 축을 분석하여 직접적으로 동일한 문제를 지적한 경우는 없다. 다만, **sample size 제약(n=16)**은 Insight 에이전트와 Literature 에이전트가 모두 핵심 한계로 인식하고 있으며, **model-specificity(boosting 전용)**도 양 에이전트가 공통으로 지적했다. 이 두 항목은 이미 논문에 명시적으로 acknowledge되어 있으므로 추가 수정보다는 현행 disclosure의 적절성을 확인하는 수준이다.

---

## 2. 에이전트별 고유 발견 사항

### Method 에이전트 고유
- **COVID-era n=9 binary task 포함 모순** [MAJOR]: Table 2의 COVID-era 행이 binary task(study-outcome)을 포함하면서 논문의 "binary ceiling effect -> multiclass만 분석" 주장과 모순
- **"11 datasets, 10 domains" counting 모호** [MODERATE]: Section 3.1의 dataset/domain 수가 검증 시 불명확
- **Section 3.1 domain count 자체 불일치** [MODERATE]: "10 domains" vs "9 domains" 전환 로직이 불투명
- **Figure label `fig:n11_correlation`** [MINOR]: 실제 n=16 figure를 참조하지만 label이 n11
- **Covertype drop rounding 혼용** [MINOR]: 82 pp vs 81.8 pp 혼재

### Literature 에이전트 고유
- **BibTeX 오류 3건** [필수 수정]:
  - `fey2023relbench`: year=2023, arXiv -> NeurIPS 2024 정식 게재로 수정
  - `angelopoulos2021gentle`: arXiv 2021 -> F&T in ML 2023 정식 출판으로 수정
  - `feldman2023achieving`: `@inproceedings` -> `@article` (TMLR는 journal)
- **Gibbs & Candes (2024) JMLR 누락** [강력 권장]: ACI 실험(Section 6.1) 맥락에서 후속 논문 미인용
- **`gulrajani2020search` key-year 불일치** [선택적]

### Insight 에이전트 고유
- **Theorem 1 bound의 실용적 tightness 부족**: gap이 큼 (0.518 vs observed 0.98). 증거 강도 6/10
- **40% threshold의 이론적 근거 부재**: n=8에서 도출된 exploratory 값
- **Class cardinality confound 미해결**: partial correlation에서 concentration이 non-significant (p=0.131)
- **External catastrophic evidence sparsity**: Covertype 단일 사례에 의존 (n=1)
- **Retraining 효과 p=0.04 -> Holm correction 후 p=0.12**: non-significant

---

## 3. 수정 항목 우선순위 정렬

### Submission-Blocking (제출 전 반드시 수정)

| # | 항목 | 소스 | 수정 난이도 |
|---|------|------|------------|
| 1 | COVID-era n=9 binary task 모순 해결 | Method | 5분 (Table 행 제거 또는 footnote 추가) |
| 2 | `fey2023relbench` BibTeX: year=2024, NeurIPS venue | Literature | 5분 |
| 3 | `angelopoulos2021gentle` BibTeX: F&T ML 2023 | Literature | 5분 |
| 4 | `feldman2023achieving` BibTeX: @article로 변경 | Literature | 2분 |

### Major (제출 가능하나 reviewer 지적 확률 높음)

| # | 항목 | 소스 | 수정 난이도 |
|---|------|------|------------|
| 5 | Dataset/domain counting 명확화 ("11 datasets, 10 domains") | Method | 15분 (Appendix table 추가) |
| 6 | Gibbs & Candes (2024) JMLR 인용 추가 | Literature | 5분 |

### Minor (cosmetic, reviewer 지적 가능성 낮음)

| # | 항목 | 소스 | 수정 난이도 |
|---|------|------|------------|
| 7 | Figure label `fig:n11_correlation` -> `fig:correlation` | Method | 2분 |
| 8 | Covertype drop 값 통일 (82 pp or 81.8 pp) | Method | 2분 |
| 9 | `gulrajani2020search` key-year 통일 | Literature | 2분 |

### 논문 한계 (수정 불가, Acknowledge 유지)

- n=16 sample size, bootstrap CI [0.50, 0.96] 넓음 (Insight/Literature 공통)
- Model-specificity: RF rho=0.30, MLP rho=0.43 (Insight/Literature 공통)
- Theorem 1 bound looseness (Insight)
- Class cardinality confound at n=8 (Insight)
- Threshold 40%의 exploratory 성격 (Insight)

---

## 4. Cross-Agent Insights

Method 에이전트가 발견한 "COVID-era n=9 binary task 모순"은 Insight 에이전트의 "binary ceiling effect 논의"와 직접 연결된다. Insight 에이전트는 binary ceiling effect가 논문의 diagnostic scope를 multiclass로 한정하는 핵심 논거라고 분석했고, Method 에이전트는 바로 그 논거가 Table 2에서 위반되고 있음을 포착했다. 이는 compound risk이다: reviewer가 ceiling effect 논리와 Table 2를 cross-check하면 cherry-picking 의심으로 이어질 수 있다.

Literature 에이전트와 Insight 에이전트 간 갈등은 없다. Literature는 citation 정확성과 completeness에 집중했고, Insight는 증거 강도와 기여의 본질적 한계를 분석했다. 두 분석이 수렴하는 결론: **논문의 실험 설계와 통계 처리는 UAI 수준이며, 주요 한계는 structural(sample size, model scope)이지 methodological이 아니다.**

---

## 5. UAI 2026 Accept 가능성 평가

**높음 (75-85%)**

근거:
- 시뮬레이션 R10 점수 8.0/8.0/7.5 = 평균 7.83 (accept range)
- Insight 에이전트의 기여별 증거 강도: 7/10, 6/10, 5/10 -- 주력 기여(Contribution 1)가 가장 강함
- Literature 에이전트: 4/5 종합 (Accept with Minor Revision)
- Method 에이전트: Minor Revision Required
- 핵심 약점(n=16, model-specificity)은 이미 논문에 명시적 acknowledge 존재
- UAI의 실용적 기여 가치 인정 경향에 부합

감점 요인:
- 이론 지향 reviewer가 Theorem 1의 loose bound에 실망할 가능성
- n=16 correlation 연구의 statistical power에 대한 근본적 의문
- External catastrophic evidence가 Covertype 1건에 의존

---

## VERDICT

### SUBMIT NOW (Blocking 수정 4건 완료 후)

Submission-blocking 항목 4건(COVID-era binary 모순, BibTeX 3건)은 모두 합계 15분 이내에 수정 가능하다. Major 항목 2건(counting 명확화, Gibbs 2024 인용)도 가능하면 함께 수정을 권장하나, 이것이 없어도 제출은 가능하다. 논문의 핵심 기여, 실험 설계, 통계 검증은 UAI accept 기준을 충족한다.

**권장 수정 순서**: #1 -> #2 -> #3 -> #4 -> #5 -> #6 -> #7/#8/#9 (총 소요시간 ~40분)

---

## 관련 파일

- 논문: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/uai_2026/main.tex`
- BibTeX: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/uai_2026/references.bib`
- COVID-era 데이터: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/results/cross_domain_statistics.json`
- External validation: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/results/external_multiseed_validation.json`
- Literature 리뷰: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/review_r3_literature.md`
- Method 비평: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/review_r3_method.md`
- Insight 분석: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/review_r3_insights.md`

---

*Generated: 2026-02-20 | Synthesis of 3 agent reports (Literature, Method, Insight)*
