# Hierarchical Bayesian Intervention - 방향 정리 (Plain Korean)

## 핵심: Hierarchical Error Propagation (계층적 오류 전파)

### 우리가 하는 것이 아닌 것 (NOT the point)

> "어떤 FK 테이블이 가장 중요한가?"

이건 **당연한 거다**. 누구나 알 수 있다:
- F1 레이싱 → DRIVER (당연히)
- 패션 리테일 → PRODUCT (당연히)
- 제조업 → ITEM (당연히)

**5살짜리도 대답할 수 있는 질문. ML이 필요 없다.**

### 우리가 진짜 하는 것 (THE point)

> "불확실성이 계층 구조를 통해 어떻게 전파되는가? 어떤 개입이 리스크를 얼마나 줄이는가?"

```
FK 테이블 (당연함 - 이게 기여가 아님)
  │
  ├── 어떤 컬럼이 문제인가?
  │     │
  │     ├── 어떤 값 범위가 문제인가?
  │     │     │
  │     │     └── 어떤 개입(intervention)이 필요한가?
  │     │           └── 얼마나 개선되는가? [95% CI로]
```

### 예시: F1 레이싱

```
DRIVER 테이블 (5살도 알아 - NOT the contribution)
    │
    ├── experience_years 컬럼 → DRIVER 불확실성의 45%
    │       │
    │       ├── experience ∈ [0, 2] (루키) → 컬럼 불확실성의 68%
    │       │       │
    │       │       └── 개입: 루키 데이터 50개 더 수집
    │       │           └── 예상 효과: -8% 불확실성 [95% CI: -12%, -4%]
    │       │
    │       └── experience ∈ [10+] (베테랑) → 12% (우선순위 낮음)
```

**핵심 기여**:
1. **오류 전파 추적**: FK → Column → Value 계층을 통한 불확실성 흐름
2. **개입 효과 정량화**: 뭘 바꾸면 얼마나 좋아지는지 (CI와 함께)
3. **리스크 기반 의사결정**: 어떤 데이터에 투자할지 우선순위

---

## Bayesian ML이 필요한 이유

### 핵심: 데이터는 항상 부족하다

```
"데이터가 충분하다" = 착각

현실:
├── 새 고객 가입 → 데이터 없음
├── 새 상품 출시 → 판매 데이터 없음
├── 시장 변화 (코로나 등) → 과거 데이터 무의미
├── Edge cases → 항상 부족
└── Long-tail 카테고리 → 샘플 수 적음
```

**이게 바로 우리가 불확실성을 측정하는 이유.**

데이터가 충분하면 point estimate만 믿으면 됨.
하지만 데이터는 항상 부족하니까:

```
1. 예측에 불확실성이 있다 (epistemic uncertainty)
        ↓
2. 그 불확실성이 어디서 오는지 알아야 한다 (FK attribution)
        ↓
3. "어디서 오는지"에 대한 추정도 불확실하다!
        ↓
4. 그래서 Bayesian이 필요하다 (uncertainty over uncertainty)
```

### Bootstrap vs Bayesian - 진짜 차이

| 상황 | Bootstrap | Bayesian |
|------|-----------|----------|
| 데이터 많음 | OK | OK |
| **데이터 적음** | CI 불안정, 너무 넓음 | **Prior가 regularization** |
| **새 FK 추가** | 처음부터 다시 | **계층 prior로 정보 공유** |
| **극단값** | shrinkage 없음 | **평균 쪽으로 shrinkage** |

**핵심 장점**: ITEM importance 추정할 때, CUSTOMER나 ORDER의 정보를 빌려올 수 있음!

### Hierarchical Information Sharing

```
# 모든 FK가 공유하는 hyperprior
σ_FK ~ HalfNormal(1.0)  # "FK들의 importance가 보통 얼마나 다른가?"

# 개별 FK의 importance
α_ITEM ~ Normal(0, σ_FK)
α_CUSTOMER ~ Normal(0, σ_FK)
```

**마법**: α_ITEM 추정할 때, ITEM 데이터만 쓰는 게 아님.
모든 FK의 정보로 σ_FK를 추정하고, 그게 α_ITEM에 영향을 줌.

### 이게 왜 중요한가?

데이터가 적은 FK (예: 새로 추가된 테이블)도 안정적인 추정 가능.
다른 FK들의 "importance가 보통 이 정도다"라는 정보를 빌려오기 때문.

---

## Bayesian DL 방법들 (쉽게 설명)

### 1. Deep Ensemble (딥 앙상블)
- **원리**: 같은 모델을 5개 만드는데, 시작점(seed)을 다르게 함
- **불확실성 측정**: 5개 모델 예측이 다르면 → 불확실함
- **장점**: 간단하고, 성능 좋고, 바로 쓸 수 있음
- **단점**: 모델 5개 돌려야 해서 비용 5배
- **평가**: ⭐⭐⭐⭐⭐ 가장 실용적

### 2. MC Dropout (몬테카를로 드롭아웃)
- **원리**: 학습할 때 쓰는 dropout을 예측할 때도 켜놓음
- **불확실성 측정**: 같은 입력에 여러 번 예측 → 결과가 다르면 불확실
- **장점**: 모델 1개만 있으면 됨, 구현 쉬움
- **단점**: 진짜 Bayesian은 아님 (근사치)
- **평가**: ⭐⭐⭐⭐ 실용적이지만 이론적 한계 있음

### 3. SWAG (SGD 궤적 기반)
- **원리**: 학습하면서 지나간 파라미터들을 기억해서 분포 만듦
- **장점**: Ensemble보다 메모리 효율적
- **단점**: 구현 복잡, 튜닝 필요
- **평가**: ⭐⭐⭐ 논문에서 자주 인용되지만 실전에서는 Ensemble이 이김

### 4. 진짜 BNN (Variational Inference)
- **원리**: 모든 weight에 확률분포 부여, ELBO 최적화
- **장점**: 이론적으로 가장 "올바른" Bayesian
- **단점**: 학습이 매우 어려움, tabular에서 잘 안됨
- **우리 실험 결과**: R² ≈ 0 (완전 실패)
- **평가**: ⭐⭐ 연구용으로는 의미있지만 실전에서는 비추

---

## 우리 실험 결과 요약

| 방법 | FK Attribution 성공? | 실용성 |
|------|---------------------|--------|
| LightGBM Ensemble | ✅ ρ=1.0 | 매우 높음 |
| MC Dropout | ✅ ρ=0.9 | 높음 |
| 진짜 BNN (Pyro) | ❌ ρ=0.3 | 낮음 (학습 실패) |

**결론**: FK Attribution은 **실용적인 Bayesian 방법들**에서 잘 작동함.

---

## 논문 방향: Hierarchical Bayesian Intervention

### 핵심 기여 (Contributions)

1. **Structured Decomposition**: 불확실성을 FK 계층 구조로 분해
2. **Intervention Effects**: "뭘 바꾸면 얼마나 좋아지는지" 정량화
3. **Credible Intervals**: 효과 추정치에 대한 신뢰구간 제공
4. **Practical Algorithm**: 어떤 UQ 방법(ensemble, MC Dropout)과도 호환

### 왜 이게 NeurIPS Bayesian ML에 맞는가?

1. **Hierarchical Bayesian Model**: FK→Column→Value 구조를 prior로 인코딩
2. **Posterior over Interventions**: 개입 효과의 불확실성까지 정량화
3. **Causal Connection**: do-operator와의 연결로 이론적 기반

---

## 핵심 메시지 (한 줄 요약)

> "어떤 테이블이 중요한가?" → 당연한 거. 5살도 안다.
>
> "그 테이블 안에서 어떤 컬럼의 어떤 값 범위가 문제고,
> 뭘 얼마나 바꾸면 리스크가 얼마나 줄어드는가?" → 이게 우리가 하는 것.
>
> **Hierarchical Error Propagation + Bayesian Risk Quantification**

---

## 예상 Output

```
═══════════════════════════════════════════════════════════════
HIERARCHICAL BAYESIAN INTERVENTION ANALYSIS
═══════════════════════════════════════════════════════════════

LEVEL 1: FK TABLE
───────────────────────────────────────────────────────────────
FK Table   │ Intervention          │ Expected Effect [95% CI]
───────────────────────────────────────────────────────────────
PLANT      │ Data quality audit    │ -12.3% [-15.1%, -9.5%]
ITEM       │ Data quality audit    │  -8.7% [-11.2%, -6.2%]
───────────────────────────────────────────────────────────────

LEVEL 2: COLUMN (within PLANT)
───────────────────────────────────────────────────────────────
Column     │ Intervention          │ Expected Effect [95% CI]
───────────────────────────────────────────────────────────────
capacity   │ Impute missing        │ -5.2% [-6.8%, -3.6%]
efficiency │ Fix measurement error │ -3.8% [-5.1%, -2.5%]
───────────────────────────────────────────────────────────────

RECOMMENDED ACTION:
  1. PLANT 데이터 품질 개선 → 예상 -12.3% 불확실성 감소
  2. 최악의 경우에도 -9.5%는 보장됨 (95% 신뢰)
```

---

## 다음 할 일

1. **Intervention Framework 구현**: 개입 시뮬레이션 + 효과 추정
2. **Bootstrap CI 구현**: 빠른 검증용 (full Bayesian 전 단계)
3. **Hierarchical Pyro Model**: 진짜 Bayesian posterior inference
4. **Multi-domain 검증**: SALT, F1, H&M, Stack에서 테스트
5. **논문 초안** 작성

---

*Created: 2025-12-09*
*Updated: 2025-12-09 - Hierarchical Bayesian Intervention 방향 반영*
*Author: ChorokLeeDev*
