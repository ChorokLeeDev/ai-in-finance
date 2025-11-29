# NeurIPS 2025 Submission Plan

**Target**: NeurIPS 2025 Main Conference
**Deadline**: May 2025 (예상: 5월 중순)
**남은 시간**: ~4개월

---

## 현재 상태 (2025-11-30 기준)

### 핵심 발견: Error Propagation Hypothesis
FK Attribution은 **error propagation** 구조가 있는 도메인에서만 유효:
- **SALT (ERP)**: ρ = 0.900 ✅
- **Trial (Clinical)**: ρ = 0.943 ✅
- **Stack (Q&A)**: ρ = -0.500 ❌

이 발견이 논문의 핵심 contribution이 됨.

### 완료된 것
- [x] RelUQ Framework 정의 (Algorithm, I/O, Theory claims)
- [x] **Attribution-Error Validation** (THE KEY RESULT)
- [x] **Error Propagation Hypothesis** 발견 및 검증
- [x] Multi-domain validation (rel-salt, rel-trial, rel-amazon, rel-stack)
- [x] Stability 검증 (error propagation 도메인에서 ρ ≥ 0.90)
- [x] LaTeX draft 업데이트 (Abstract, Introduction, Experiments, Conclusion)
- [x] Clinical Trials (rel-trial) 데이터 로더 추가
- [x] Ablation study (K, P, n, subsample rate) - 이미 paper에 포함
- [x] Basic baselines (Feature-level, Correlation clustering, Random) - 이미 paper에 포함

### 부족한 것
- [ ] InfoSHAP baseline (NeurIPS 2023 - 최신 uncertainty attribution)
- [ ] SHAP variance baseline
- [ ] Figure 생성 (현재 placeholder만 있음)
- [ ] Theoretical proof (Error Propagation 조건 formal화)
- [ ] 논문 완성

---

## 보강 사항 상세 계획

### 1. Baseline 추가 (필수)

NeurIPS 리뷰어가 물을 것: "기존 방법과 비교했나?"

#### 1.1 InfoSHAP (NeurIPS 2023)
```
논문: "InfoSHAP: Shapley Values for Information Theoretic Feature Attribution"
왜 비교해야 하나: 가장 최신 uncertainty attribution 방법

구현 계획:
1. InfoSHAP 논문 읽고 핵심 알고리즘 파악
2. 공식 코드 있으면 사용, 없으면 재구현
3. 동일 데이터셋 (rel-f1, rel-stack, rel-amazon)에 적용
4. Stability 비교 (우리 vs InfoSHAP)

예상 결과:
- InfoSHAP = feature-level → stability 낮을 것
- RelUQ > InfoSHAP on stability (가설)

실험 코드 위치: experiments/baselines/infoshap_baseline.py
```

#### 1.2 Permutation Importance (Variance)
```
방법: 기존 permutation importance를 variance에 적용
      (prediction 대신 uncertainty 변화 측정)

구현:
- 이미 RelUQ에서 하는 것과 유사
- Feature-level로 수행 후 집계 방식만 다르게

비교:
- Feature-level permutation (24 features) vs FK-level (5 groups)
- 같은 mechanism, 다른 granularity

실험 코드 위치: experiments/baselines/permutation_baseline.py
```

#### 1.3 SHAP Variance Attribution
```
방법: SHAP 값의 분산을 uncertainty attribution으로 사용
      (InfoSHAP과 다른 접근)

구현:
- TreeSHAP으로 각 예측의 SHAP 계산
- SHAP 값들의 분산 = feature importance for variance

비교:
- SHAP-based vs Permutation-based
- 둘 다 feature-level → 불안정 예상

실험 코드 위치: experiments/baselines/shap_variance_baseline.py
```

#### 1.4 Data-Driven Grouping (Correlation Clustering)
```
이미 완료: experiment_fk_vs_datadriven_f1.py

추가 필요:
- rel-stack, rel-amazon에도 동일 실험
- 논문용 테이블 정리
```

### 2. Ablation Study (필수)

NeurIPS 리뷰어가 물을 것: "하이퍼파라미터에 민감하지 않나?"

#### 2.1 Ensemble Size (K) 민감도
```
실험:
- K = {3, 5, 7, 10, 15, 20}
- 각 K에서 attribution 계산
- Stability 측정

예상:
- K=3: 높은 variance (불안정)
- K=5: sweet spot (현재 사용)
- K=10+: diminishing returns

도표: Line plot (K vs Stability)
```

#### 2.2 Permutation Runs (P) 민감도
```
실험:
- P = {1, 3, 5, 10, 20}
- 각 P에서 attribution 계산
- Stability와 계산 시간 측정

예상:
- P=1: 높은 variance
- P=5: 충분히 안정 (현재 사용)
- P=10+: 거의 차이 없음

도표: Line plot (P vs Stability) + computation time
```

#### 2.3 Sample Size Scaling
```
실험:
- n = {500, 1000, 2000, 3000, 5000, 10000}
- 각 n에서 attribution 계산
- Top FK 일관성 및 stability 측정

예상:
- n < 1000: 불안정
- n >= 2000: 안정적 수렴

도표:
- Line plot (n vs Stability)
- Heatmap (FK attribution % across n)
```

#### 2.4 Subsampling Rate 민감도
```
실험:
- subsample = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
- colsample = {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}

예상:
- rate=1.0: ensemble 다양성 부족 → variance ≈ 0
- rate=0.8: 적절한 다양성 (현재 사용)
- rate=0.5: 너무 적은 데이터 → 불안정

도표: Heatmap (subsample × colsample → Stability)
```

### 3. Theoretical Contribution (중요)

NeurIPS 리뷰어가 물을 것: "이론적 기여가 뭐냐?"

#### 3.1 Variance Reduction Bound
```
목표: FK grouping이 feature-level 대비 variance를 줄인다는 것을 증명

Theorem (Draft):
Let X = [X_1, ..., X_n] be features with FK-induced correlation structure.
Let G = {g_1, ..., g_k} be FK groups where features within g_i are correlated.
Then for permutation-based attribution:

Var[α_FK(g_i)] ≤ (1/|g_i|) * Σ_{f ∈ g_i} Var[α_feature(f)]

즉, FK-level variance ≤ feature-level variance의 평균

증명 아이디어:
- Within-group correlation → attribution 분산이 그룹 내에서 공유됨
- 그룹으로 집계 → 분산의 합이 아닌 평균

필요 작업:
1. Correlation structure를 formal하게 정의
2. Permutation attribution의 variance 분석
3. Grouping의 효과를 bound로 표현
```

#### 3.2 Stability Guarantee
```
목표: 특정 조건 하에서 stability ≥ threshold 보장

Proposition:
If within-group correlation ρ_within > ρ_between (between-group correlation),
then FK grouping achieves stability ρ ≥ f(ρ_within, ρ_between, k)
where k is number of groups.

증명 아이디어:
- High within-group correlation → 그룹 내 feature들이 함께 움직임
- Low between-group correlation → 그룹 간 독립적
- 따라서 attribution이 안정적
```

#### 3.3 Actionability Formalization (Optional)
```
Actionability를 formal하게 정의하기 어려움
대안: Case study로 qualitative하게 보여주기
```

### 4. 대규모 실험 (권장)

NeurIPS 리뷰어가 물을 것: "더 큰 데이터에서도 되나?"

#### 4.1 Larger Sample Sizes
```
현재: n=3000
목표: n=10000, n=50000 (가능하면)

주의:
- rel-amazon: 20M reviews → sampling 필요
- 계산 시간 증가 → caching 필수

예상 결과:
- Stability 유지 또는 향상
- Top FK 일관성 유지
```

#### 4.2 More Datasets (Optional)
```
RelBench 추가 데이터셋:
- rel-avito (Classifieds)
- rel-hm (Fashion retail)
- rel-event (Event recommendation)
- rel-trial (Clinical trials)

우선순위:
1. rel-hm: 리테일 도메인, e-commerce와 비교
2. rel-trial: 헬스케어 도메인, 다양성 확보
```

#### 4.3 Failure Case Analysis
```
언제 RelUQ가 실패하는가?

가설:
- FK groups가 너무 적을 때 (k < 3)
- FK groups 크기가 극도로 불균형할 때
- Within-group correlation이 낮을 때

실험:
- 의도적으로 failure 조건 만들기
- Synthetic data로 boundary 테스트
```

### 5. 논문 작성 (필수)

#### 5.1 Figure 제작
```
필수 Figures:
1. Overview diagram: RelUQ pipeline (Input → FK mapping → Ensemble → Attribution)
2. Stability comparison: Bar chart (RelUQ vs baselines)
3. Ablation plots: K, P, n sensitivity
4. Multi-domain results: Table or grouped bar chart

Optional:
5. Case study drill-down: DRIVER → specific features → entities
6. Correlation heatmap: FK groups vs random groups
```

#### 5.2 Writing
```
Section별 분량 (NeurIPS 8 pages):
- Abstract: 150 words
- Introduction: 1 page
- Related Work: 0.5 page
- Method: 1.5 pages (Algorithm + Theory)
- Experiments: 3 pages (Tables + Figures + Analysis)
- Conclusion: 0.5 page
- References: 별도

Appendix (supplementary):
- Full proofs
- Additional experiments
- Implementation details
```

---

## 타임라인 (4개월)

### Month 1 (Dec 2025)
```
Week 1-2: Baselines 구현 및 실험
  - [ ] InfoSHAP 구현/적용
  - [ ] Permutation baseline
  - [ ] SHAP variance baseline
  - [ ] Correlation clustering (3 datasets)

Week 3-4: Ablation study
  - [ ] K sensitivity
  - [ ] P sensitivity
  - [ ] n scaling
  - [ ] Subsampling rate
```

### Month 2 (Jan 2026)
```
Week 1-2: Theory 강화
  - [ ] Variance reduction bound 증명
  - [ ] Stability guarantee 증명
  - [ ] 지도교수 검토

Week 3-4: 대규모 실험
  - [ ] n=10000 실험
  - [ ] 추가 데이터셋 (rel-hm)
  - [ ] Failure case 분석
```

### Month 3 (Feb 2026)
```
Week 1-2: 논문 작성
  - [ ] Method section 완성
  - [ ] Experiments section 완성
  - [ ] Figures 제작

Week 3-4: 논문 작성 계속
  - [ ] Introduction 다듬기
  - [ ] Related work 완성
  - [ ] Abstract 최종화
```

### Month 4 (Mar 2026)
```
Week 1-2: 리뷰 및 수정
  - [ ] 지도교수 리뷰
  - [ ] 동료 리뷰
  - [ ] 피드백 반영

Week 3-4: 최종 준비
  - [ ] Camera-ready 품질 체크
  - [ ] Supplementary 준비
  - [ ] 제출
```

---

## 파일 구조 계획

```
ai-in-finance/
├── paper/
│   ├── main.tex                    # 메인 논문
│   ├── supplementary.tex           # 부록
│   ├── figures/
│   │   ├── overview.pdf
│   │   ├── stability_comparison.pdf
│   │   └── ablation_*.pdf
│   └── NEURIPS_2025_PLAN.md        # 이 파일
│
├── experiments/
│   ├── baselines/
│   │   ├── infoshap_baseline.py
│   │   ├── permutation_baseline.py
│   │   └── shap_variance_baseline.py
│   ├── ablation/
│   │   ├── ablation_ensemble_size.py
│   │   ├── ablation_permutation_runs.py
│   │   ├── ablation_sample_size.py
│   │   └── ablation_subsampling.py
│   └── large_scale/
│       ├── experiment_10k.py
│       └── experiment_additional_datasets.py
│
└── chorok/v3_fk_risk_attribution/  # 기존 코드
```

---

## 리스크 및 대응

| 리스크 | 확률 | 대응 |
|--------|------|------|
| InfoSHAP이 우리보다 나음 | 낮음 | Actionability로 차별화 |
| Theory 증명 실패 | 중간 | Empirical focus로 전환 |
| 대규모 실험 시간 부족 | 중간 | n=10000까지만, 나머지 appendix |
| Deadline miss | 낮음 | KDD 2026으로 전환 |

---

## 다음 액션

1. **지금 당장**: Baselines 구현 시작 (InfoSHAP 먼저)
2. **이번 주**: Ablation 실험 설계 및 코드 작성
3. **지도교수 미팅**: 이 계획 공유, Theory 방향 조언 요청

---

*생성일: 2025-11-29*
*마지막 수정: 2025-11-29*
