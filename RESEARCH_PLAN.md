# Selective Prediction Under Distribution Shift

## 핵심 아이디어
Distribution shift 상황에서 uncertainty가 높은 샘플은 예측을 거부(abstain)하면 정확도가 올라간다.
COVID-19를 natural experiment로 사용하여 이를 검증.

## Phase 1: Tabular Selective Prediction (2-3주)

### 질문
SALT 데이터에서 COVID 전후로:
1. Uncertainty가 높은 샘플을 거부하면 정확도가 올라가는가?
2. Shift 상황에서 adaptive threshold가 fixed보다 나은가?

### 방법
1. LightGBM ensemble (5 seeds) → uncertainty = prediction entropy
2. Selective prediction: uncertainty > threshold인 샘플 거부
3. Risk-coverage curve 그리기
4. Train vs Val 비교 (COVID shift 효과)

### 성공 기준
- Coverage 80%에서 Val accuracy 향상 → Phase 2 진행
- 효과 없음 → 분석 후 pivot

## Phase 2: Temporal Adaptive (추가 2-3주, Phase 1 성공 시)
- Shift 감지 시 자동으로 threshold 조절
- "Self-adaptive selective prediction under distribution shift"

## 완료된 작업
- [x] SALT 8개 태스크 분석
- [x] COVID shift ranking (results/covid_shift_analysis/)
  - sales-group: JS=0.33 (highest)
  - item-shippoint: JS=0.13
- [ ] LightGBM ensemble 학습
- [ ] Selective prediction 구현
- [ ] Risk-coverage curve
