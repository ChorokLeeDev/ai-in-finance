# Bayesian Deep Learning - 방향 정리 (Plain Korean)

## 우리가 하고 싶은 것

**FK Attribution**: 테이블 데이터에서 "어느 테이블(FK)이 예측 불확실성에 가장 큰 영향을 주는가?"를 찾아내는 방법.

예를 들어 자동차 경주 결과 예측에서:
- 드라이버 정보가 문제인지?
- 팀 정보가 문제인지?
- 날씨 정보가 문제인지?

이걸 알면 **어디를 개선해야 예측이 좋아지는지** 알 수 있음.

---

## Bayesian ML이 필요한 이유

일반 ML은 "정답은 42입니다"라고만 말함.
Bayesian ML은 "정답은 42인데, 확신도는 80%입니다"라고 말함.

이 **불확실성(uncertainty)**을 FK별로 분해해서 "어느 테이블 때문에 불확실한지" 찾는 게 우리 연구.

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

## 논문 방향 제안

### 옵션 A: "Practical Bayesian" 접근
- Deep Ensemble + MC Dropout 중심
- "FK Attribution이 실용적인 UQ 방법들에서 일관되게 작동한다"
- 장점: 실험 쉬움, 결과 확실함
- 단점: "진짜 Bayesian 아니잖아" 비판 가능

### 옵션 B: "Bayesian 관점에서 분석" 접근
- FK Attribution을 Bayesian 이론으로 정당화
- 예: "uncertainty decomposition은 posterior predictive variance의 구조적 분해"
- 장점: 이론적 기여 강조 가능
- 단점: 수학적으로 엄밀해야 함

### 옵션 C: 하이브리드
- 이론: Bayesian 관점에서 FK Attribution 정의
- 실험: 실용적 방법들(Deep Ensemble, MC Dropout)로 검증
- "이론적으로 올바르고, 실용적으로도 작동함"

**추천: 옵션 C** - NeurIPS Bayesian ML 트랙에 가장 적합

---

## 핵심 메시지 (한 줄 요약)

> "테이블 데이터의 불확실성을 FK 단위로 분해하면,
> 어디를 고쳐야 예측이 좋아지는지 알 수 있다.
> 이건 Deep Ensemble이든 MC Dropout이든 일관되게 작동한다."

---

## 다음 할 일

1. **Deep Ensemble 구현** (현재 LightGBM만 있음 → Neural Net 버전 추가)
2. **이론적 정당화** 작성 (posterior variance decomposition)
3. **추가 데이터셋**에서 검증 (rel-f1 외 다른 데이터셋)
4. **논문 초안** 작성 시작

---

*Created: 2025-12-09*
*Author: ChorokLeeDev*
