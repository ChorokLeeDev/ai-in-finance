# MethodCritic Analysis Report -- Round 3 (2026-02-20)

## 검토 범위

이전 라운드에서 수정 완료된 항목(Abstract 중복, Theorem 수치, p=0.04 unadjusted, A1 footnote, Table 2 footnote)은 제외하고, 수정 후 새롭게 생긴 일관성 문제와 논리적 문제에 집중.

---

## Executive Summary

수정 후 논문은 대부분 일관적이나, 5개의 구체적 문제가 남아있다. 가장 중요한 것은 COVID-era n=9에 binary task(study-outcome)을 포함시킨 것이 논문의 핵심 주장("binary ceiling effect로 인해 multiclass만 분석")과 모순되는 점이다. n=15/n=16 footnote은 사실관계가 정확하다. Figure label은 cosmetic 문제이나 reviewer가 소스를 볼 가능성이 있어 수정 권장.

---

## 발견사항

### 1. COVID-era n=9에 binary task 포함 -- 논리적 모순 [MAJOR]

**문제**: Table 2의 COVID-era 행(n=9, rho=0.883)은 "8 SALT + study-outcome(rel-trial)"로 구성된다. 그런데 study-outcome은 **binary task**이다 (`cross_domain_statistics.json`에서 `"type": "binary"` 확인).

**모순**: 논문의 핵심 주장 중 하나가 "Binary APS exhibits a structural ceiling effect; concentration is diagnostic specifically for multiclass settings" (Contribution 5, line 65)이다. Binary task를 multiclass 분석 체계에서 체계적으로 제외하면서도, COVID-era 그룹에는 binary task를 포함시키는 것은 논리적으로 일관되지 않는다.

**추가 문제**: Table 2에서 COVID-era 행은 "Multiclass (SALT)" 행과 "Multiclass (4 dom.)" 행 사이에 위치한다. "Group" 열에 "COVID-era"라고만 표기되어 있어, 독자는 이것도 multiclass 그룹이라고 오해할 수 있다. Footnote에 "study-outcome"이 포함된다고 명시되어 있으나, study-outcome이 binary라는 사실은 footnote에 기술되지 않았다.

**근거**: `cross_domain_statistics.json`의 `covid_era.tasks` 배열에서 study-outcome은 `"type": "binary"`, `"concentration": 20.8`, `"coverage_drop": -1.3`으로 기록되어 있다.

**영향**:
- Study-outcome은 low concentration(20.8%) + negative drop(-1.3%)이므로 correlation을 **부풀리는** 방향으로 작용한다 (low C = low drop 패턴에 부합)
- Binary ceiling effect로 인해 drop이 거의 0인 것이지, concentration이 낮아서 robust한 것이 아닐 수 있다
- n=9에서 이 task를 제거하면 n=8 SALT-only (rho=0.833)과 동일해진다

**수정 방안**:
- (A) COVID-era 행을 제거하고 n=8 SALT-only를 첫 행으로 유지 (가장 깔끔)
- (B) COVID-era 행을 유지하되, footnote에 "study-outcome is binary; included here for temporal-alignment completeness rather than multiclass diagnostic analysis"를 추가
- (C) study-outcome이 binary임에도 COVID-era에 포함시키는 이유를 본문에서 명시적으로 정당화 (권장하지 않음 -- 논리적 모순을 해결하지 못함)

**심각도**: MAJOR. Reviewer가 이 모순을 발견하면 cherry-picking 의심을 받을 수 있다.

---

### 2. n=15 footnote 정확성 검증 -- 사실 확인 [OK, 정확함]

**확인 결과**: n=15 footnote(line 284)의 내용은 사실과 일치한다.

- "$n=15$ is an intermediate analysis (7 external multiclass domains)": 8 SALT + 7 external = 15. 여기서 7 external = {Covertype, Shuttle, Avila, PAMAP2, Pendigits, Satimage, Gas Sensor}
- "$n=16$ adds KDDCup99 as the 8th external domain": 맞음. KDDCup99를 추가하면 8 external multiclass가 되어 n=16
- "Stack Overflow (3 classes, near-binary ceiling effect) is excluded from all multiclass endpoints": 맞음. Stack Overflow는 `external_multiseed_validation.json`에서 `predicted: "vulnerable"`, `actual: "robust"`, `correct: false`로 기록됨. 3 classes로 near-binary ceiling effect 해당.

**결론**: n=15/n=16 counting은 정확하다. 추가 수정 불필요.

---

### 3. Figure label 불일치 [MINOR]

**문제**: Figure의 LaTeX label이 `\label{fig:n11_correlation}`이나, 실제로는 `results/figure_n16_correlation.pdf`를 포함하고 있으며, caption도 n=16을 올바르게 기술한다.

- Line 295: `\includegraphics{results/figure_n16_correlation.pdf}` -- n=16 figure 사용 (올바름)
- Line 296: caption에서 "across all 16 multiclass tasks in 9 domains" (올바름)
- Line 296: `\label{fig:n11_correlation}` -- label 이름이 n11 (부정확)
- Line 251: `Figure~\ref{fig:n11_correlation}` -- 본문에서는 n=8 SALT 결과를 설명하면서 이 figure를 참조

**영향**: LaTeX label은 compiled PDF에 나타나지 않으므로 reviewer는 볼 수 없다. 그러나 소스 코드 공개 시 혼란을 줄 수 있으며, 학술적 위생(hygiene) 관점에서 수정 권장.

**수정 방안**: `fig:n11_correlation` -> `fig:correlation` 또는 `fig:n16_correlation`으로 변경. 모든 `\ref{fig:n11_correlation}` 참조도 함께 변경.

**심각도**: MINOR. 기능적 문제 없음.

---

### 4. "11 additional datasets spanning 10 non-supply-chain domains" 계산 검증 [MODERATE]

**문제**: Section 5.5 (line 353)에서 "External validation extends to 11 additional datasets across 10 non-supply-chain domains"라고 기술한다.

**검증**:
- Table `framework_validation`에 나열된 external datasets: 9개 (Covertype, Shuttle, Avila, PAMAP2, KDDCup99, Pendigits, Satimage, Gas Sensor, Stack Overflow)
- `cross_domain_statistics.json`의 non-SALT tasks: study-outcome(1), driver-dnf(1), driver-top3(1) = 3개
- 총합: 9 + 3 = 12 tasks. 단, rel-f1을 1 dataset(2 tasks)으로 세면 9 + 1 + 1 = 11 datasets.

**문제점**: "11 additional datasets"에서 "dataset"과 "task"의 구분이 모호하다. rel-f1은 1개 dataset에서 2개 task를 추출한 것이므로 11로 세는 것이 맞을 수 있으나, 논문 어디에서도 이 counting 방식을 명시하지 않는다.

**"10 domains" 검증**:
- 9 UCI dataset domains + clinical-trials + motorsport = 11 domains (만약 모든 UCI가 별개 domain이면)
- Pendigits(digit recognition)와 Avila(handwriting style) 또는 다른 pair가 같은 domain으로 묶여야 10이 됨
- 어떤 두 domain이 합쳐지는지 명시되지 않음

**수정 방안**: Appendix에 전체 dataset list와 domain 매핑을 명시적으로 추가하거나, Section 3.1의 기존 설명을 확장. 특히 어떤 datasets가 같은 domain으로 분류되는지 명시.

**심각도**: MODERATE. 현재 상태에서 reviewer가 직접 세면 숫자가 맞지 않을 수 있다.

---

### 5. Covertype drop 값 rounding 불일치 [MINOR]

**문제**: Covertype의 coverage drop이 논문 내에서 두 가지 값으로 기술된다.

- Abstract (line 45): "82~pp drop"
- Contribution 6 (line 67): "82~pp drop"
- Section 5.3 (line 259): "82~pp drop"
- Section 5.5 (line 353): "81.8~pp drop"

**데이터 확인**: `external_multiseed_validation.json`에서 10-seed mean drop = 81.8 pp (std 0.09). `verified_n11_multiclass.json`에서 single-seed drop = 83.28 pp.

**영향**: "82 pp"은 81.8의 반올림이므로 factually 틀리지 않으나, 같은 논문 내에서 "82"와 "81.8"을 혼용하면 정밀도에 대한 의문을 유발한다.

**수정 방안**: 전체를 "82 pp"(반올림)로 통일하거나 "81.8 pp"(정확값)로 통일. 1-decimal로 통일 권장.

**심각도**: MINOR.

---

### 6. Section 3.1의 domain count와 나머지 논문의 불일치 [MODERATE]

**문제**: Section 3.1 (line 89)에서:
> "External validation uses 11 additional datasets spanning 10 domains."

그런데 바로 다음 문장에서:
> "Excluding binary tasks yields 8 external multiclass datasets, and together with the 8 multiclass SALT tasks this gives the primary endpoint of n=16 multiclass tasks across 9 domains"

**불일치**: "10 domains" (all external) vs "9 domains" (multiclass only, SALT 포함). 이것은 사실 일관적일 수 있다: 10 external non-supply-chain domains 중 binary-only domains (clinical-trials, motorsport)을 제거하면 8 external domains이 되고, SALT를 1 domain으로 추가하면 9 domains. 그런데:

- 8 external multiclass datasets가 8 domains인지 확인 필요
- Covertype, Shuttle, Avila, PAMAP2, KDDCup99, Pendigits, Satimage, Gas Sensor = 8 datasets
- 이것이 8 separate domains이면 8 + SALT = 9 domains (맞음)
- 하지만 위에서 "10 domains"이라 했으므로 11 datasets에서 일부가 같은 domain

이 counting이 자체적으로 일관되려면:
- 11 external datasets = 10 domains (2 datasets가 1 domain 공유)
- 8 external multiclass datasets = 8 domains (domain 공유 pair에서 한쪽이 binary)
- 그 pair가 뭔지? clinical-trials(study-outcome, binary)와 motorsport(driver-dnf+driver-top3, binary) 제거 -> 나머지 9 datasets = 8 domains? 아니, 9 datasets면 9 domains.

**결론**: Counting이 자체적으로 일관되려면 추가 설명이 필요하다. 현재 상태에서는 reviewer가 혼란스러울 수 있다.

**수정 방안**: Section 3.1 또는 Appendix에 complete enumeration table 추가. "11 datasets, 10 domains" 대신 구체적으로 나열.

**심각도**: MODERATE. Reviewer가 counting을 검증하려 할 때 혼란 유발 가능.

---

## 심각도 요약

| # | 문제 | 심각도 |
|---|------|--------|
| 1 | COVID-era n=9에 binary task 포함 (논리적 모순) | MAJOR |
| 2 | n=15/n=16 footnote 정확성 | OK (정확) |
| 3 | Figure label `fig:n11_correlation` | MINOR |
| 4 | "11 datasets, 10 domains" counting 모호 | MODERATE |
| 5 | Covertype drop rounding 불일치 | MINOR |
| 6 | Section 3.1 domain count 자체 일관성 | MODERATE |

---

## 권장 조치 (우선순위순)

1. **[MAJOR] COVID-era n=9 행 처리**: Table 2에서 COVID-era 행을 제거하거나, study-outcome이 binary임을 footnote에 명시하고 포함 이유를 정당화. 가장 깔끔한 해법은 행 제거 (n=8 SALT-only가 이미 존재하므로 정보 손실 없음).

2. **[MODERATE] Dataset/domain enumeration**: Section 3.1 또는 Appendix에 전체 external dataset list를 table로 추가. 각 dataset의 domain 분류, class 수, binary/multiclass 구분, shift 유형을 명시. "11 datasets, 10 domains"의 근거를 명확히.

3. **[MINOR] Figure label 수정**: `fig:n11_correlation` -> `fig:correlation` 또는 `fig:n16_correlation`.

4. **[MINOR] Covertype drop 통일**: 전체를 "82 pp" 또는 "81.8 pp"로 통일.

---

## Verdict

**MINOR REVISION REQUIRED**

MAJOR 이슈(#1)는 논리적 모순이나 수정이 간단하다 (Table 행 제거 또는 footnote 추가). 나머지는 counting 명확화와 cosmetic 수정. Submission을 blocking하는 수준은 아니나, #1은 반드시 수정해야 한다.

---

## 관련 파일

- 논문: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/uai_2026/main.tex`
- COVID-era 데이터: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/results/cross_domain_statistics.json` (covid_era section)
- External multiseed: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/results/external_multiseed_validation.json`
- Verified n=11: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/results/verified_n11_multiclass.json`
- Figure: `/Users/i767700/Github/ai-in-finance/papers/conformal_covid/uai_2026/results/figure_n16_correlation.pdf`
