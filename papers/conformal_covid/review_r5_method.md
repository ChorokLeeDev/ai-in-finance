# MethodCritic R5 -- 최종본 방법론 검토

## 검토 대상
`/Users/i767700/Github/ai-in-finance/papers/conformal_covid/uai_2026/main.tex`
검토일: 2026-02-20

---

## Executive Summary

전체적으로 수치 일관성이 잘 유지되어 있음. 이전 라운드에서 지적된 주요 문제들은 모두 수정됨. 남은 이슈는 submission-blocking 수준이 아니며, 대부분 minor~moderate 수준. 가장 중요한 발견은 Appendix A.3의 "(pre-COVID)" 표기 오류와 abstract의 77% vs 77.1% 반올림 불일치.

---

## 1. Submission-Blocking 이슈: 없음

이전 라운드에서 발견된 fatal 이슈(Table 2 n=9 행, abstract 중복, theorem 수치 오류 등)는 모두 수정 완료.

---

## 2. 숫자 일관성 검증

### 2.1 전체 통과 항목
| 숫자 | Abstract | Intro | Results | Table | Appendix | 판정 |
|------|----------|-------|---------|-------|----------|------|
| rho=0.853 (n=16) | O | O | O | O (Tab3) | O (App F) | PASS |
| rho=0.833 (n=8) | -- | O | O | O (Tab3) | -- | PASS |
| tau=0.667 (n=16) | O | O | -- | O (Tab3) | -- | PASS |
| Boot CI [0.50, 0.96] | O | O | -- | O (Tab3) | -- | PASS |
| p<=0.005 (Wilcoxon) | O | O | O | O (Tab1) | -- | PASS |
| Drop range 0.1%-77.1% | -- | O | O | O (Tab1) | O (Placebo) | PASS |
| +19pp retraining | O | O | -- | -- | -- | PASS (18.9 -> 19 rounding) |
| Covertype 81.8pp | O | -- | O | O (Tab5) | -- | PASS |
| KDDCup99 15.9+-21.4pp | -- | -- | O | O (Tab5) | O (App F) | PASS |
| KS 0.68-0.96 | O | O | -- | -- | O (Tab KS) | PASS (0.676-0.956 반올림) |

### 2.2 Minor 불일치 발견

#### (a) Abstract: "77%" vs 본문/테이블 "77.1%"
- **위치**: Abstract 라인 44 "coverage drops ranging from 0.1\% to 77%"
- **본문/테이블**: 모두 "77.1%" (라인 61, 191, 205, 444, 446, 690)
- **심각도**: MINOR -- abstract에서의 반올림이지만, "0.1%"는 정확히 보고하면서 "77.1%"만 반올림하는 것은 비대칭적
- **수정안**: `77%` -> `77.1%`로 통일 (2글자 추가, 공간 문제 없음)

#### (b) Placebo 비율 범위: "6--140x" vs 실제 "6--143x"
- **위치**: 라인 355 "6--140$\times$ lower"
- **Placebo 테이블**: s-shipcond = 143x (라인 444)
- **심각도**: MINOR -- 하지만 본인 테이블과 불일치
- **수정안**: `6--140$\times$` -> `6--143$\times$`

---

## 3. 방법론적 이슈

### 3.1 MODERATE: Appendix A.3 "(pre-COVID)" 표기 오류
- **위치**: 라인 425 "computed on validation (pre-COVID) data only"
- **문제**: Methodology (라인 91-94)에서 Validation = Feb-Jul 2020 (COVID onset)으로 정의. 즉 validation data는 **pre-COVID가 아니라 COVID 기간** 데이터
- **영향**: SHAP concentration이 모델의 학습된 의존 구조를 측정한다는 점에서 실질적 영향은 제한적이나, 리뷰어에게 데이터 누출 의심을 줄 수 있음
- **수정안**: "(pre-COVID)" 삭제 -> "computed on validation data only"로 변경. 또는 SHAP이 training data에서 계산되었다면 "computed on training data only"로 수정

### 3.2 MINOR: "identical temporal shift" 표현의 모호성
- **위치**: 라인 52 "8 classification tasks experience identical temporal shift"
- **문제**: 8개 task가 동일한 시간 경계(Feb 2020, Jul 2020)를 공유하지만, 각 task의 feature/label distribution shift 정도는 매우 다름 (Jaccard 0.02 ~ 0.61). "Identical temporal shift"는 "동일한 시간 분할"을 의미하지 "동일한 분포 변화"를 의미하지 않음
- **현재 상태**: 본문에서 Jaccard 등으로 차이를 설명하므로 혼동 가능성은 낮음
- **수정안**: "identical temporal split" 또는 "the same temporal boundary"로 변경하면 더 정확

### 3.3 MINOR: 외부 도메인 카운팅의 잠재적 혼동
- **위치**:
  - 라인 89: "11 additional datasets spanning 10 domains"
  - 라인 67 (Contribution #6): "9 external non-supply-chain domains"
  - 라인 272: primary endpoint = "16 multiclass tasks in 9 domains"
- **문제**: "9 domains"가 두 가지 다른 의미로 사용됨:
  1. Contribution #6: 9개 외부 도메인 (Stack Overflow 포함)
  2. Primary endpoint: 8개 외부 multiclass 도메인 + SALT = 9 도메인
- **심각도**: 우연의 일치로 같은 숫자가 나와서 혼동은 낮으나, 정밀한 리뷰어가 의문을 제기할 수 있음
- **수정안**: Contribution #6을 "across up to 9 external domains" 또는 "across 9 external domains (8 multiclass + 1 near-binary)" 등으로 명확화

### 3.4 MINOR: 이름 없는 binary 데이터셋 2개
- **위치**: 라인 89 "11 additional datasets" -- Table 5에는 9개만 나옴
- **문제**: 2개 binary 데이터셋이 본문/부록 어디에도 이름이 나오지 않음. 재현성에 소폭 영향
- **수정안**: Appendix에 binary 데이터셋 이름과 기본 결과 추가 (예: "Binary datasets: X and Y; excluded from multiclass analysis due to ceiling effect")

---

## 4. 논리/해석 이슈

### 4.1 MINOR: Figure 2 (fig:n16_correlation) 파일 경로
- **위치**: 라인 293 `\includegraphics{results/figure_n16_correlation.pdf}`
- **상태**: 파일이 `papers/conformal_covid/results/figure_n16_correlation.pdf`에 존재하며 컴파일 시 정상 포함됨 (main.log 확인). 그러나 uai_2026 디렉토리의 `results/` 하위가 아닌 상위 `results/`에 위치 -- 제출 시 패키지 구조에 주의 필요

### 4.2 확인 완료: Theorem 수치 일관성
- Bound verification (Appendix H): 5개 task 모두 bound < observed, 일관성 확인
- T(C) 단조증가 조건 epsilon < (1-q_alpha)/(K-1): epsilon=0에서 자명히 성립
- (A1) footnote (라인 152): log-odds vs probability space 간극 명시됨

### 4.3 확인 완료: Table 간 숫자 교차 검증
- Table 1 (main results) ↔ Table 2 (overlap) ↔ Table 5 (framework) ↔ Table 7 (placebo): 모든 drop 값 일치
- Table 1 ↔ Table 8 (RF comparison): LGB drop 값 일치
- Table 3 (correlation) ↔ 본문 ↔ abstract: 모든 rho/tau/p/CI 일치
- Table 4 (diagnostic comparison) ↔ 본문: 모든 값 일치
- Table 6 (threshold) ↔ Table 5 (framework): TP/FP/FN 사례 식별 일치

---

## 5. Reproducibility 체크

- Seeds 42-91 (50개): 명시됨 (라인 109, 415)
- Software versions: Python 3.9, LightGBM 3.3, SHAP 0.41 (라인 433)
- Hardware: 8 cores, 8GB RAM (라인 432)
- Hyperparameters: 전체 명시됨 (라인 415)
- SHAP subsample size: 10,000 (라인 424)
- Conformal calibration: 50% split of validation (라인 103, 419)
- 외부 데이터셋 source: UCI (라인 351) -- 구체적 URL/version 없음 (MINOR)
- **Pre-registration**: 없음 (exploratory study로 명시)
- **Code availability**: 미명시 -- 코드 공개 계획 언급 없음

---

## 6. 최종 판정

### 발견된 이슈 요약

| # | 이슈 | 심각도 | Submission-blocking? |
|---|------|--------|---------------------|
| 1 | Abstract "77%" vs "77.1%" | MINOR | No |
| 2 | Placebo "6-140x" vs "6-143x" | MINOR | No |
| 3 | Appendix "(pre-COVID)" 표기 오류 | MODERATE | No, but reviewer-attracting |
| 4 | "identical temporal shift" 모호성 | MINOR | No |
| 5 | 외부 도메인 카운팅 혼동 가능성 | MINOR | No |
| 6 | Binary 데이터셋 2개 미명시 | MINOR | No |
| 7 | Figure 파일 경로 (제출 패키지) | MINOR | 제출 시 확인 필요 |
| 8 | Code availability 미명시 | MINOR | No (UAI 필수 아님) |

### Verdict: CONDITIONALLY ACCEPTABLE

Fatal/Major 이슈 없음. MODERATE 이슈 1건 (#3)은 수정 권장하나 submission을 차단하지는 않음. 나머지 MINOR 이슈들은 best practice 수준.

### 권장 수정 (우선순위)
1. **라인 425**: "(pre-COVID)" 삭제 또는 정확한 표현으로 교체
2. **라인 44 (Abstract)**: "77%" -> "77.1%"
3. **라인 355**: "6--140" -> "6--143"
4. **제출 전**: `results/figure_n16_correlation.pdf`가 제출 패키지에 올바른 상대 경로로 포함되는지 확인
5. (선택) 라인 52: "identical temporal shift" -> "identical temporal split"
6. (선택) Binary 데이터셋 이름을 Appendix에 추가
