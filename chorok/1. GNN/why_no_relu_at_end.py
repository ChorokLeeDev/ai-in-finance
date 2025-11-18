"""
왜 마지막 레이어에는 ReLU를 안 넣을까?
실험으로 확인해보기
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 80)
print("🔍 왜 마지막 레이어에는 ReLU를 안 넣을까?")
print("=" * 80)

# ============================================================================
# 시나리오: 논문 1개를 분류하는 상황
# ============================================================================

print("\n📌 상황: 모델이 논문 1개의 카테고리를 예측")
print("-" * 80)

# 7개 카테고리: AI, 생물, 물리, 화학, 수학, 공학, 의학
categories = ["AI", "생물", "물리", "화학", "수학", "공학", "의학"]
true_label = 0  # 실제로는 AI 논문

# 모델의 원시 출력 (logits) - 마지막 레이어 출력
logits = torch.tensor([5.2, -1.3, 0.8, -0.5, 2.1, -2.0, 0.3])

print(f"\n실제 정답: {categories[true_label]} (인덱스 {true_label})")
print(f"\n모델의 원시 출력 (logits):")
for i, (cat, score) in enumerate(zip(categories, logits)):
    print(f"  {cat:6s}: {score:6.2f}")

# ============================================================================
# Case 1: ReLU 없음 (정상적인 경우)
# ============================================================================

print("\n" + "=" * 80)
print("✅ Case 1: 마지막에 ReLU 안 씀 (정상)")
print("=" * 80)

# Softmax로 확률 변환
probs_normal = F.softmax(logits, dim=0)

print(f"\nSoftmax 확률 변환:")
for i, (cat, prob) in enumerate(zip(categories, probs_normal)):
    bar = "█" * int(prob * 50)
    print(f"  {cat:6s}: {prob:6.4f} {bar}")

# CrossEntropyLoss 계산
loss_normal = -torch.log(probs_normal[true_label])
print(f"\nCrossEntropyLoss:")
print(f"  Loss = -log(P(AI)) = -log({probs_normal[true_label]:.4f}) = {loss_normal:.4f}")

# 예측
pred_normal = torch.argmax(logits)
print(f"\n예측 결과:")
print(f"  예측: {categories[pred_normal]} (정답!)")
print(f"  신뢰도: {probs_normal[pred_normal]:.1%}")

# ============================================================================
# Case 2: ReLU 있음 (잘못된 경우)
# ============================================================================

print("\n" + "=" * 80)
print("❌ Case 2: 마지막에 ReLU 씀 (문제 발생!)")
print("=" * 80)

# ReLU 적용 - 음수를 0으로
logits_with_relu = F.relu(logits)

print(f"\nReLU 적용 후 (음수 → 0):")
print(f"  원래:  {logits.tolist()}")
print(f"  ReLU:  {logits_with_relu.tolist()}")

print(f"\n음수 점수의 변화:")
for i, (cat, before, after) in enumerate(zip(categories, logits, logits_with_relu)):
    if before < 0:
        print(f"  {cat:6s}: {before:6.2f} → {after:6.2f} (정보 손실!)")

# Softmax로 확률 변환
probs_with_relu = F.softmax(logits_with_relu, dim=0)

print(f"\nSoftmax 확률 변환:")
for i, (cat, prob) in enumerate(zip(categories, probs_with_relu)):
    bar = "█" * int(prob * 50)
    print(f"  {cat:6s}: {prob:6.4f} {bar}")

# CrossEntropyLoss 계산
loss_with_relu = -torch.log(probs_with_relu[true_label])
print(f"\nCrossEntropyLoss:")
print(f"  Loss = -log(P(AI)) = -log({probs_with_relu[true_label]:.4f}) = {loss_with_relu:.4f}")

# 예측
pred_with_relu = torch.argmax(logits_with_relu)
print(f"\n예측 결과:")
print(f"  예측: {categories[pred_with_relu]} (여전히 정답)")
print(f"  신뢰도: {probs_with_relu[pred_with_relu]:.1%}")

# ============================================================================
# 비교 분석
# ============================================================================

print("\n" + "=" * 80)
print("📊 비교 분석")
print("=" * 80)

print(f"\n1. 확률 분포 변화:")
print(f"   ReLU 없음: AI={probs_normal[0]:.4f}, 생물={probs_normal[1]:.4f}, 물리={probs_normal[2]:.4f}")
print(f"   ReLU 있음: AI={probs_with_relu[0]:.4f}, 생물={probs_with_relu[1]:.4f}, 물리={probs_with_relu[2]:.4f}")

print(f"\n2. 손실(Loss) 변화:")
print(f"   ReLU 없음: {loss_normal:.4f}")
print(f"   ReLU 있음: {loss_with_relu:.4f}")
print(f"   차이: {abs(loss_normal - loss_with_relu):.4f}")

print(f"\n3. 문제점:")
print(f"   - 음수 점수가 모두 0이 되어 정보 손실")
print(f"   - 음수는 '이 카테고리가 아니다'라는 중요한 정보")
print(f"   - 예: 생물=-1.3 → '생물학 논문이 아니다' (강한 신호)")
print(f"   - ReLU 후: 생물=0 → '정보 없음' (신호 손실)")

# ============================================================================
# 실제 문제 상황
# ============================================================================

print("\n" + "=" * 80)
print("⚠️  실제 문제 상황: 예측이 틀리는 경우")
print("=" * 80)

# 헷갈리는 경우: AI와 물리가 비슷한 점수
logits_confusing = torch.tensor([2.5, -3.0, 2.3, -1.0, -2.0, -2.5, -1.5])

print(f"\n모델 출력 (AI와 물리가 비슷):")
for i, (cat, score) in enumerate(zip(categories, logits_confusing)):
    print(f"  {cat:6s}: {score:6.2f}")

# ReLU 없음
probs_conf_normal = F.softmax(logits_confusing, dim=0)
pred_conf_normal = torch.argmax(logits_confusing)

print(f"\n✅ ReLU 없음:")
print(f"  AI 확률: {probs_conf_normal[0]:.4f}")
print(f"  물리 확률: {probs_conf_normal[2]:.4f}")
print(f"  예측: {categories[pred_conf_normal]}")

# ReLU 있음
logits_conf_relu = F.relu(logits_confusing)
probs_conf_relu = F.softmax(logits_conf_relu, dim=0)
pred_conf_relu = torch.argmax(logits_conf_relu)

print(f"\n❌ ReLU 있음:")
print(f"  AI 확률: {probs_conf_relu[0]:.4f}")
print(f"  물리 확률: {probs_conf_relu[2]:.4f}")
print(f"  예측: {categories[pred_conf_relu]}")

print(f"\n문제:")
print(f"  - 음수 점수들이 0이 되어 비교 정보 손실")
print(f"  - '생물=-3.0'은 '절대 생물 아님'을 의미했는데 사라짐")
print(f"  - 모델이 '왜 이 카테고리가 아닌지'를 표현할 수 없음")

# ============================================================================
# 결론
# ============================================================================

print("\n" + "=" * 80)
print("💡 결론")
print("=" * 80)

print("""
왜 마지막 레이어에 ReLU를 안 쓰는가?

1. 음수는 중요한 정보다!
   - 양수: "이 카테고리일 가능성 높음"
   - 음수: "이 카테고리가 아님" (중요!)
   - ReLU는 음수를 0으로 만들어 정보 손실

2. CrossEntropyLoss는 음수를 사용한다
   - Softmax: exp(음수) = 작은 확률 (유효한 입력!)
   - Loss = -log(확률) 계산에 음수가 필수

3. 중간 레이어 vs 마지막 레이어
   - 중간: ReLU 필요 (비선형성 추가)
   - 마지막: ReLU 불필요 (원시 점수 유지)

4. 원시 점수(logits)가 더 좋다
   - 전체 범위 표현 가능: -∞ ~ +∞
   - 모델이 확신도를 자유롭게 표현
   - "매우 확실히 아님" (-5.0) vs "확실함" (+5.0)

실전 패턴:
    ✅ 중간 레이어: conv → ReLU → conv
    ✅ 마지막 레이어: conv (ReLU 없음!)
""")

print("=" * 80)
print("✅ 이해가 되셨나요? 마지막은 항상 ReLU 없이!")
print("=" * 80)
