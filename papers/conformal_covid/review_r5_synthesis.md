# Round 5 최종 종합 판정 보고서

**논문**: "Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study"
**대상**: UAI 2026 제출본 (`uai_2026/main.tex`)
**종합일**: 2026-02-20
**입력**: Literature Agent, Method Agent, Insights Agent (Round 5)

---

## 1. Submission-Blocking 이슈

**없음.**

세 에이전트 모두 fatal/major 이슈를 발견하지 못했다. 이전 라운드에서 지적된 모든 치명적 문제(Table 2 n=9 행, abstract 중복, theorem 수치 오류, bib venue 오류 등)가 수정 완료되었음을 Method Agent와 Literature Agent가 독립적으로 확인했다.

---

## 2. 남은 이슈 목록

### MODERATE (1건)

| # | 이슈 | 에이전트 | 설명 |
|---|------|----------|------|
| 1 | Appendix A.3 "(pre-COVID)" 표기 오류 | Method | 라인 425. Validation = Feb-Jul 2020 (COVID 기간)인데 "pre-COVID"로 표기. 리뷰어에게 데이터 누출 의심을 유발할 수 있음 |

### MINOR (10건)

| # | 이슈 | 에이전트 | 설명 |
|---|------|----------|------|
| 2 | Abstract "77%" vs 본문 "77.1%" | Method | 라인 44. 비대칭 반올림 (0.1%는 정확, 77.1%만 반올림) |
| 3 | Placebo "6--140x" vs 실제 "6--143x" | Method | 라인 355. 자체 테이블(s-shipcond=143x)과 불일치 |
| 4 | "identical temporal shift" 모호성 | Method | 라인 52. "identical temporal split"이 더 정확 |
| 5 | 외부 도메인 카운팅 혼동 가능성 | Method | "9 domains"가 두 가지 다른 의미로 사용됨 |
| 6 | Binary 데이터셋 2개 미명시 | Method | 11개 중 Table 5에 9개만 나오고, 2개 binary 이름 없음 |
| 7 | Figure 파일 경로 (제출 패키지) | Method | 상위 results/에 위치 -- 제출 시 패키지 구조 확인 필요 |
| 8 | NeurIPS entry type 비일관성 | Literature | tibshirani2019=@article, romano2019=@inproceedings 혼용 |
| 9 | gulrajani2020search 본문 설명 부정확 | Literature | "temporal shift"를 "realistic distribution shifts"로 수정 권고 |
| 10 | Gibbs & Candes (2024) JMLR 미인용 | Literature | ACI 후속작. 필수는 아니나 CP 심사자가 인지할 가능성 |
| 11 | Code availability 미명시 | Method | UAI 필수는 아니나 재현성 기대치 상승 추세 |

---

## 3. 에이전트 간 교차 분석

세 에이전트의 결론이 수렴한다: **submission-blocking 이슈 없음, 핵심 수치 일관성 확보, 문헌 커버리지 충족**. Method Agent가 발견한 "(pre-COVID)" 표기 오류(#1)는 Insights Agent의 "투명한 한계 인정"이라는 강점 평가와 대비되는 유일한 긴장점이나, 이는 오타 수준의 문제로 의도적 왜곡이 아님이 명확하다. Literature Agent의 gulrajani 설명 부정확(#9)과 Method Agent의 "identical temporal shift" 모호성(#4)은 모두 정밀한 표현 문제로, 동일한 패턴(원문을 약간 과도하게 좁게/넓게 요약)을 보인다.

---

## 4. 최종 판정

### SUBMIT NOW

Fatal/Major 이슈 0건. MODERATE 1건은 1분 이내 수정 가능한 표기 오류. MINOR 10건은 best practice 수준으로, 제출을 차단할 근거가 없다. Insights Agent의 분석대로 이론-실증-실무 삼각 구조가 견고하고, 시뮬레이션 리뷰 8.0/8.0/7.5 수준이 유지된다.

**권장 수정 (제출 전 5분 작업)**:
1. 라인 425: "(pre-COVID)" 삭제 -> "computed on validation data only"
2. 라인 44: "77%" -> "77.1%"
3. 라인 355: "6--140" -> "6--143"
4. 제출 패키지에 `results/figure_n16_correlation.pdf` 포함 확인

---

## 5. 리뷰어 예상 질문 TOP 3

### Q1: "외부 catastrophic failure 증거가 Covertype 1건뿐인데, SHAP concentration의 일반적 진단력을 주장할 수 있는가?"

**대비 전략**: (a) Covertype은 n=10 seed 전체에서 81.8pp drop으로 완벽히 재현되며, 이는 cherry-picking이 아닌 체계적 결과임을 강조. (b) 반대 방향 증거도 동등하게 강력: low-concentration 6개 외부 도메인이 모두 robust (10/10). (c) Mixed-effects 분석(3 boosting model, n=24)에서 beta1=1.64, p=0.0006으로 SALT 내부에 국한되지 않는 패턴. (d) n=1 한계는 구조적(catastrophic shift 자체가 드문 현상)이며, 이를 해결하려면 의도적으로 모델을 실패시키는 intervention study가 필요함을 향후 연구로 제시.

### Q2: "RF (rho=0.30)와 MLP (rho=0.43)에서 진단력이 비유의한데, 이 방법론이 gradient boosting에만 유효한 것 아닌가?"

**대비 전략**: (a) 이를 숨기지 않고 논문에 명시적으로 보고한 점이 신뢰도를 높임. (b) Tree SHAP은 exact Shapley value 계산이 가능한 유일한 model class이며, RF/MLP의 approximate SHAP은 concentration 측정 자체에 noise를 도입. (c) MLP의 failure mode는 "global sensitivity"로 qualitatively 다르며 (entropy 패턴이 다름), 이는 별도의 neural-network-specific diagnostic이 필요함을 시사. (d) Gradient boosting은 tabular data의 production 표준이므로 실무적 범위는 충분히 넓음.

### Q3: "Theorem 1의 가정 (A1)이 log-odds space에서만 성립하는데, 이론적 결과의 실질적 유용성은 무엇인가?"

**대비 전략**: (a) (A1)의 근사적 성격은 footnote에서 명시적으로 인정. (b) Theorem의 역할은 exact bound가 아니라 "concentration이 왜 failure를 예측하는가"에 대한 방향성 제시 (directional insight). (c) 5개 적용 가능 task 전체에서 conservative bound가 충족되므로 이론이 경험적으로 vacuous하지 않음. (d) Catastrophic task의 prediction entropy 감소라는 반직관적 현상이 이론의 메커니즘("모델이 confidently wrong")과 정합하는 점이 이론의 설명력을 지지. (e) RAPS의 mechanism-dependent 결과(class-accumulation vs concentrated-dependence failure 분리)가 이론의 세부 예측을 확인.

---

*세 에이전트 분석 종합 완료. 제출 준비 상태.*
