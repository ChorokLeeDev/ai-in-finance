# 문헌 리뷰 보고서 (Literature Review Report) — Round 5

**논문**: Diagnosing Conformal Prediction Failures Under Distribution Shift: A COVID-19 Case Study
**검토일**: 2026-02-20
**검토 대상 파일**: `uai_2026/main.tex`, `uai_2026/references.bib`
**검토 범위**: references.bib 전체 오류 최종 확인 / 2024-2026년 누락 최신 문헌 / 본문 citation 사용 정확성 / 전반적 문헌 커버리지

---

## 1. references.bib 전체 오류 최종 확인

이전 라운드에서 수정된 5개 항목(angelopoulos2024conformal, fey2024relbench, angelopoulos2021gentle, feldman2023achieving, lei2018distribution, dua2017uci)은 모두 정확하다. 아래는 잔존 이슈 목록이다.

### 1.1 문제 없음 (확인 완료)

| key | 확인 사항 | 상태 |
|-----|-----------|------|
| `vovk2005algorithmic` | Springer 2005, Algorithmic Learning in a Random World | OK |
| `romano2020classification` | NeurIPS 2020, vol.33, pp.3581-3591 | OK |
| `romano2019conformalized` | NeurIPS 2019, vol.32 (CQR) | OK |
| `tibshirani2019conformal` | NeurIPS 2019, @article, vol.32 | OK |
| `gibbs2021adaptive` | NeurIPS 2021, vol.34, pp.1660-1672 | OK |
| `barber2023conformal` | Annals of Statistics 2023, vol.51, no.2, pp.816-845 | OK |
| `angelopoulos2021gentle` | F&T in ML 2023, vol.16, no.4, pp.494-591 | OK (이전 수정) |
| `zaffran2022adaptive` | ICML 2022, pp.25834-25866 | OK |
| `podkopaev2021distribution` | UAI 2021, vol.161, PMLR | OK |
| `lundberg2017unified` | NeurIPS 2017 | OK |
| `koh2021wilds` | ICML 2021, pp.5637-5664, PMLR | OK |
| `gulrajani2020search` | ICLR 2021 (key명 `gulrajani2020search`인데 venue year=2021) | 경미한 이슈 (아래 설명) |
| `adebayo2018sanity` | NeurIPS 2018 | OK |
| `malinin2021shifts` | @article, arXiv:2107.07455 — **저널 미확정 arXiv 상태** | 경미한 이슈 (아래) |
| `lundberg2020local` | Nature Machine Intelligence 2020, vol.2, no.1 | OK |
| `fey2024relbench` | NeurIPS 2024, vol.37 | OK (이전 수정) |
| `feldman2023achieving` | TMLR 2023, @article | OK (이전 수정) |
| `angelopoulos2024conformal` | ICLR 2024 | OK (이전 수정) |
| `garg2022leveraging` | ICLR 2022 | OK |
| `angelopoulos2021uncertainty` | ICLR 2021 (RAPS) | OK |
| `gretton2012kernel` | JMLR 2012, vol.13, pp.723-773 | OK |
| `lopez2017revisiting` | ICLR 2017 | OK |
| `lei2018distribution` | JASA 2018, vol.113, no.523, pp.1094-1111 | OK (이전 수정) |
| `dua2017uci` | @misc, 2017 | OK (이전 수정) |

### 1.2 잔존 경미한 이슈 (수정 권고)

**이슈 1: `gulrajani2020search` — key와 year 불일치**

```bibtex
@inproceedings{gulrajani2020search,
  ...
  year={2021}   % key는 2020이지만 year는 2021
}
```

- arXiv 최초공개는 2020년이나 실제 게재는 ICLR 2021이다.
- 현재 `year=2021`은 정확하다. 다만 key명 `gulrajani2020search`가 연도 불일치를 암시한다.
- **권고**: key를 `gulrajani2021search`로 변경하거나, 현재 상태(year=2021 정확, key는 내부 식별자)로 그대로 두되 본문 내 \citet{} 호출이 올바른지 확인. UAI 심사자 입장에서 key명은 보이지 않으므로 **실질적 문제 없음**. 변경 선택사항.

**이슈 2: `malinin2021shifts` — arXiv 상태 항목**

```bibtex
@article{malinin2021shifts,
  journal={arXiv preprint arXiv:2107.07455},
  year={2021}
}
```

- 2021년 arXiv 투고 이후 NeurIPS 2021 Datasets & Benchmarks Workshop에 발표된 것으로 알려져 있다.
- 검색 결과 동 논문의 확정 게재 여부가 불분명하여 arXiv 인용은 허용 가능하다.
- **권고**: 주석에 "(NeurIPS 2021 Datasets & Benchmarks Track, workshop paper)" 정도를 note 필드로 추가하거나 현행 유지. 어느 쪽이든 치명적 오류는 아니다.

**이슈 3: `tibshirani2019conformal` — @article vs @inproceedings**

```bibtex
@article{tibshirani2019conformal,
  journal={Advances in Neural Information Processing Systems},
  ...
}
```

- NeurIPS 논문을 @article + journal=NeurIPS로 표기하는 것은 bib style에 따라 허용된다. `romano2019conformalized`는 @inproceedings로 처리되어 있어 **동일 venue에 대한 entry type 비일관성**이 존재한다.
- **권고**: 두 항목의 entry type을 @inproceedings로 통일하거나, 논문 내 모든 NeurIPS 항목을 @article로 통일. 현 혼용 상태는 UAI 심사자의 눈에 띌 수 있다.

### 1.3 인용 누락 확인 (이전 라운드 검토 대비)

- Shafer & Vovk (2008) JMLR "A Tutorial on Conformal Prediction" — 현재 **미인용**. Related Work "Conformal Prediction" 단락에서 Vovk 2005 book만 인용되어 있다. 이 tutorial은 CP 분야의 표준 입문 레퍼런스로 UAI 심사자가 기대하는 참조문헌이나, vovk2005algorithmic 이미 인용 중이므로 **선택적 추가**.
- Gibbs & Candes (2024) JMLR "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts" (arXiv:2208.08401, JMLR Vol.25) — 현재 gibbs2021adaptive(NeurIPS 2021 ACI)만 인용되어 있음. 2024 JMLR 후속 작은 "adaptive step-size tuning"을 다루며 본 논문의 ACI 실험과 직접 관련된다. **추가 고려 필요**.

---

## 2. 2024-2026년 누락 최신 문헌 검토

### 2.1 반드시 인용해야 할 논문 (강력 권고)

없음. 아래 항목들은 "있으면 좋은" 수준이다.

### 2.2 있으면 좋은 논문 (선택적 추가)

**[A] Gibbs & Candes (2024) JMLR — ACI 후속**

- 제목: "Conformal Inference for Online Prediction with Arbitrary Distribution Shifts"
- 게재: JMLR Vol.25, 2024 (arXiv:2208.08401)
- 관련성: 본 논문 Section 5.1에서 ACI 실험을 보고함. gibbs2021adaptive(NeurIPS 2021)만 인용 중인데, 이 JMLR 2024 확장판이 step-size 적응 문제(본 논문의 ACI 실험과 직접 연관)를 다룬다. 분야 심사자가 인지할 가능성이 높다.
- **권고**: "Adaptive Methods" 단락 \citet{gibbs2021adaptive} 인용 뒤 "and extended by \citet{gibbs2024online}" 형태로 추가.

**[B] Bhatnagar et al. (2023) ICML — strongly adaptive online conformal**

- 제목: "Improved Online Conformal Prediction via Strongly Adaptive Online Learning"
- 게재: ICML 2023, pp.2337-2363
- 관련성: zaffran2022adaptive, feldman2023achieving와 동일 "adaptive methods" 클러스터. ACI 계열 확장 중 가장 인용률이 높은 ICML 논문이다.
- **권고**: "Adaptive Methods" 단락에 추가 가능하나, 이미 해당 클러스터가 3개 문헌 인용 중이므로 선택사항.

**[C] Xu et al. (2025) ICLR — Wasserstein-Regularized CP**

- 제목: "Wasserstein-Regularized Conformal Prediction under General Distribution Shift"
- 게재: ICLR 2025 (arXiv:2501.13430)
- 관련성: barber2023conformal의 TV-distance bound를 Wasserstein distance로 일반화. 본 논문의 Barber 2023 인용 맥락(Section 2 Related Work)에서 언급할 수 있다. 그러나 본 논문은 이론적 bound를 개선하는 것이 아니라 경험적 진단을 제안하므로 필수성은 낮다.
- **권고**: 선택사항. 인용 시 "Recent work further tightens coverage bounds via Wasserstein distance \citep{xu2025wasserstein}" 형태로 1문장 추가.

**[D] "Conformal Prediction: A Data Perspective" (ACM Computing Surveys, 2025)**

- 관련성: CP 분야 최신 종합 서베이. Related Work 서두에서 분야 맥락을 제시할 때 인용 가능.
- **권고**: angelopoulos2021gentle로 이미 커버되므로 필수 아님.

### 2.3 인용 불필요 확인

- "Conditional Coverage Diagnostics for Conformal Prediction" (arXiv:2512.11779, 2025) — 본 논문 제출 시점(UAI 2026 마감 2026년 초 예상) 기준 arXiv 2025-12 논문으로, 인용이 기대되지 않는다.
- "Wasserstein CP" (ICLR 2025) — 위 [C] 참조. 선택사항.

---

## 3. 본문 Citation 정확성 검토

### 3.1 정확한 인용 사례

- **\citet{vovk2005algorithmic}**: "introduced conformal prediction with exchangeability guarantees" — 정확.
- **\citet{barber2023conformal}**: "bounding coverage loss by the total variation distance between test and calibration score distributions" — Annals of Statistics 2023 논문의 핵심 내용과 정확히 일치.
- **\citet{romano2020classification}**: APS에 대한 인용 — 정확.
- **\citet{gibbs2021adaptive}**: "Adaptive Conformal Inference (ACI) for non-stationary settings" — 정확.
- **\citet{tibshirani2019conformal}**: "covariate shift with known propensity scores" — 정확.
- **\citet{lundberg2020local}**: TreeExplainer 인용 — 정확 (Nature Machine Intelligence 2020).
- **\citet{angelopoulos2021uncertainty}**: RAPS 인용 — 논문 제목은 "Uncertainty Sets for Image Classifiers using Conformal Prediction"이고 RAPS를 제안. 정확.
- **\citet{garg2022leveraging}**: "unlabeled test data to predict accuracy degradation" — 정확.
- **\citet{koh2021wilds}**: WILDS 벤치마크 인용 — 정확.

### 3.2 잠재적 부정확 또는 과장 표현

**이슈 1: \citet{angelopoulos2024conformal}의 설명**

본문 (Related Work, Adaptive Methods 단락):
> "\citet{angelopoulos2024conformal} generalize conformal prediction to broader risk measures."

- Conformal Risk Control은 coverage (0/1 loss) 대신 일반적인 bounded loss function으로 conformal quantile을 정의한다. "broader risk measures"라는 표현은 옳지만 다소 모호하다.
- "generalize conformal prediction to control arbitrary bounded loss functions beyond binary coverage" 정도로 표현하면 더 정확하나, 현행 표현이 틀린 것은 아니다. **경미한 개선 사항**.

**이슈 2: \citet{gulrajani2020search}의 설명**

본문 (Shift Detection 단락):
> "\citet{gulrajani2020search} show robust methods often fail under temporal shift."

- 실제 Gulrajani & Lopez-Paz (2021)는 domain generalization 벤치마크 DomainBed를 통해 ERM이 최신 domain generalization 방법을 능가함을 보인다. "robust methods often fail under temporal shift"라는 표현은 이 논문의 결론을 temporal shift에 국한하여 과도하게 좁게 해석한 것이다. 논문 자체는 temporal shift 전용이 아니며, ERM vs. DG 방법 일반의 비교가 주제다.
- **권고**: "show that domain generalization methods often fail to improve over empirical risk minimization under realistic distribution shifts \citep{gulrajani2020search}"로 수정하거나, 현행 문구를 "\citet{gulrajani2020search} demonstrate that standard domain generalization methods often fail to outperform ERM under natural distribution shifts"로 조정.

**이슈 3: \citet{malinin2021shifts}의 사용**

본문: "Shifts~\citep{malinin2021shifts} provide benchmarks"
- Shifts Dataset은 실제로 weather prediction, machine translation 등을 포함한 복수 태스크 실배포 분포 변화 데이터셋이다. "provide benchmarks"라는 간결한 표현은 문맥상 적절하다.

**이슈 4: \citet{zaffran2022adaptive} 설명**

본문: "extensions by \citet{zaffran2022adaptive} for time series"
- Zaffran et al. (2022)는 ICML 2022 논문으로 EnbPI와 ACI의 관계를 다루며 시계열 예측에 대한 conformal 방법을 제안한다. "extensions ... for time series"는 정확한 요약이다.

### 3.3 종합 평가

총 인용 24개 중 심각한 오류: **0건**. 경미한 표현 부정확: **1건** (gulrajani2020search 설명). Entry-type 비일관성: **1건** (NeurIPS 논문의 @article/@inproceedings 혼용).

---

## 4. 전반적 문헌 커버리지 평가 (UAI 2026 Accept 기준)

### 4.1 강점

1. **핵심 CP 이론 문헌 완비**: Vovk 2005, Barber 2023, Romano 2019/2020, Tibshirani 2019, Lei 2018, Angelopoulos 2021/2024 — UAI 심사자가 기대하는 모든 기초 문헌이 포함되어 있다.

2. **Adaptive methods 클러스터 양호**: Gibbs 2021 (ACI), Zaffran 2022 (time series), Feldman 2023 (TMLR) 세 편이 인용되어 온라인/적응형 CP 맥락을 충분히 커버한다.

3. **Distribution shift 벤치마크 커버**: WILDS (Koh 2021), Shifts (Malinin 2021), Gulrajani 2021 — 세 분야 표준 벤치마크/탐구 논문이 모두 포함되어 있다.

4. **방법론 인용 정확**: TreeExplainer (Lundberg 2020), APS (Romano 2020), RAPS (Angelopoulos 2021), SHAP (Lundberg 2017) 모두 정확하게 인용됨.

5. **이전 라운드 수정 완료**: ICLR 2024 venue, NeurIPS 2024 RelBench, F&T 2023, TMLR journal, JASA 2018, UCI citation 모두 올바르게 수정됨.

### 4.2 개선 가능 사항 (우선순위 순)

| 순위 | 항목 | 심각도 | 권고 |
|------|------|--------|------|
| 1 | NeurIPS 항목 @article/@inproceedings 비일관성 | 낮음 | tibshirani2019conformal를 @inproceedings로 통일하거나 모두 @article로 통일 |
| 2 | gulrajani2020search 본문 설명 부정확 | 낮음 | "temporal shift"를 "realistic distribution shifts" 또는 "ERM over domain generalization"으로 수정 |
| 3 | Gibbs & Candes (2024) JMLR 미인용 | 낮음 | ACI 후속작 JMLR 2024 추가 고려 |
| 4 | malinin2021shifts arXiv 상태 | 낮음 | note 필드 추가 또는 현행 유지 |

### 4.3 종합 판정

**문헌 커버리지: UAI 2026 Accept 기준 충족**

총 참고문헌 24편. 분량 대비 커버리지가 적절하며, CP 이론/방법론/적용 분야의 핵심 문헌이 모두 포함되어 있다. 이전 라운드에서 지적된 주요 오류(ICLR venue, NeurIPS 2024, F&T, TMLR, JASA, UCI)가 완전히 수정되었으므로, 남은 이슈는 모두 경미한 수준이다.

UAI 2026 심사자 관점에서 "인용 품질로 인한 감점" 요소는 현재 실질적으로 없다. 선택적으로 Gibbs 2024 JMLR을 추가하면 ACI 실험 섹션의 완성도가 소폭 향상된다.

---

## 5. 수정 체크리스트 (이번 라운드)

- [ ] **필수**: `tibshirani2019conformal` entry type을 @inproceedings로 변경하거나, `romano2019conformalized`를 @article로 변경하여 NeurIPS 항목 type 통일
- [ ] **권고**: Section 2 Related Work, "robust methods often fail under temporal shift" → "domain generalization methods often fail to improve over ERM under realistic distribution shifts"로 수정
- [ ] **선택**: Gibbs & Candes (2024) JMLR (arXiv:2208.08401) 추가: `gibbs2024online` key로 ACI 섹션에 병기
- [ ] **선택**: `malinin2021shifts`에 `note = {NeurIPS 2021 Datasets \& Benchmarks Workshop}` 추가

---

## 참고: 검색 기반 확인 문헌

- Gibbs & Candes (2024) JMLR: https://jmlr.org/papers/v25/22-1218.html
- Bhatnagar et al. (2023) ICML: https://proceedings.mlr.press/v202/bhatnagar23a.html
- Wasserstein-Regularized CP (ICLR 2025): https://openreview.net/forum?id=aJ3tiX1Tu4
- Shafer & Vovk (2008) JMLR tutorial: https://jmlr.org/papers/v9/shafer08a.html
- Conditional Coverage Diagnostics (arXiv 2512.11779): https://arxiv.org/abs/2512.11779
- CP: A Data Perspective, ACM Computing Surveys (2025): https://dl.acm.org/doi/abs/10.1145/3736575
