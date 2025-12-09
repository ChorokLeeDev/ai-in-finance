# Hierarchical Bayesian Intervention - 방향 정리 (Plain Korean)

## 우리가 하고 싶은 것

**Hierarchical Bayesian Intervention Analysis**:
테이블 데이터에서 "뭘 얼마나 바꾸면 불확실성이 얼마나 줄어드는지"를 계층적으로 찾아내는 방법.

### v3 (이전): 어디가 문제인지만 알려줌
```
PLANT 테이블이 37% 기여
```

### v4 (목표): 뭘 얼마나 바꾸면 되는지까지 알려줌
```
PLANT 테이블 → 데이터 품질 개선 시 불확실성 -12% [95% CI: -15%, -9%]
  └─ capacity 컬럼 → 결측치 보정 시 -5% [95% CI: -7%, -3%]
      └─ capacity ∈ [0, 100] → 추가 데이터 수집 시 -3% [95% CI: -4.5%, -1.7%]
```

**핵심 차이**: 단순히 "어디"가 아니라 "얼마나 바꾸면 얼마나 좋아지는지" + 신뢰구간

---

## Bayesian ML이 필요한 이유

### 이유 1: Credible Intervals (신뢰구간)

일반 방법: "데이터 개선하면 12% 좋아집니다"
Bayesian: "데이터 개선하면 12% 좋아집니다 **[95% 확률로 9%~15% 사이]**"

이 신뢰구간이 중요한 이유:
- **의사결정**: "최악의 경우에도 9%는 좋아지니까 투자할 가치 있음"
- **우선순위**: 신뢰구간이 겹치면 더 확실한 것부터 하기
- **리스크 관리**: 넓은 신뢰구간 = 데이터 더 필요함

### 이유 2: Hierarchical Structure (계층 구조)

관계형 데이터베이스 = 자연스러운 계층 구조
```
Database
├── 테이블 A (FK: customer)
│   ├── 컬럼 A1 (age)
│   │   ├── 값 범위 [20-30]
│   │   └── 값 범위 [30-40]
│   └── 컬럼 A2 (income)
├── 테이블 B (FK: product)
```

Bayesian Hierarchical Model은 이 구조를 prior로 자연스럽게 인코딩할 수 있음.

### 이유 3: Intervention = do-calculus

Causal inference의 do-operator와 연결:
- P(Y | X) = 관찰 (현재 상태)
- P(Y | do(X=x)) = 개입 (바꾸면 어떻게 되는지)

"PLANT 데이터를 개선하면" = do(PLANT_quality = improved)

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

> "관계형 데이터의 불확실성을 계층적으로 분해하고,
> 각 레벨에서 개입 효과를 **신뢰구간과 함께** 제공한다.
> 이를 통해 '뭘 얼마나 바꾸면 얼마나 좋아지는지' 알 수 있다."

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
