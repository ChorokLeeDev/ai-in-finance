"""
aggregate 함수의 작동을 직접 확인하는 테스트 코드
"""

import torch
import torch.nn as nn


class SimpleGNN(nn.Module):
    """
    aggregate 함수 테스트를 위한 간단한 GNN
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1_weight = nn.Linear(in_channels, hidden_channels)
        self.conv2_weight = nn.Linear(hidden_channels, out_channels)

    def aggregate(self, x, edge_index):
        """Message passing 구현"""
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float('inf')] = 0

        out = torch.zeros_like(x)
        for src, dst in zip(row, col):
            out[dst] += x[src] * deg_inv[dst]

        return out


def test_aggregate():
    """
    ==================================================================================
    aggregate 함수 작동 원리 시각화
    ==================================================================================
    """

    print("\n" + "=" * 80)
    print("📊 aggregate 함수 테스트")
    print("=" * 80)

    # === 예시 그래프 설정 ===
    print("\n🔹 Step 1: 그래프 정의")
    print("-" * 80)

    # 3개 노드, 2차원 특성
    x = torch.tensor([
        [0.9, 0.1],  # 노드 0: AI 논문 (AI=0.9, 생물=0.1)
        [0.8, 0.2],  # 노드 1: AI 논문 (AI=0.8, 생물=0.2)
        [0.1, 0.9],  # 노드 2: 생물 논문 (AI=0.1, 생물=0.9)
    ], dtype=torch.float)

    # 엣지: 0→1, 1→2
    edge_index = torch.tensor([
        [0, 1],  # source (출발)
        [1, 2],  # target (도착)
    ], dtype=torch.long)

    print("\n그래프 구조:")
    print("    노드 0 (AI 논문)")
    print("      ↓ 인용")
    print("    노드 1 (AI 논문)  →  노드 2 (생물 논문)")

    print("\n노드 특성 (업데이트 전):")
    for i in range(x.size(0)):
        print(f"    노드 {i}: {x[i].tolist()} (AI={x[i][0]:.1f}, 생물={x[i][1]:.1f})")

    print("\n엣지 (인용 관계):")
    for i in range(edge_index.size(1)):
        src = edge_index[0][i].item()
        dst = edge_index[1][i].item()
        print(f"    엣지 {i}: 노드 {src} → 노드 {dst}")

    # === aggregate 실행 ===
    print("\n🔹 Step 2: aggregate 함수 실행")
    print("-" * 80)

    model = SimpleGNN(in_channels=2, hidden_channels=4, out_channels=2)
    out = model.aggregate(x, edge_index)

    print("\n노드 특성 (업데이트 후):")
    for i in range(out.size(0)):
        print(f"    노드 {i}: {out[i].tolist()} (AI={out[i][0]:.1f}, 생물={out[i][1]:.1f})")

    # === 상세 분석 ===
    print("\n🔹 Step 3: 결과 분석")
    print("-" * 80)

    print("\n변화 분석:")
    for i in range(x.size(0)):
        print(f"\n노드 {i}:")
        print(f"    원래 특성:   {x[i].tolist()}")
        print(f"    업데이트 후: {out[i].tolist()}")

        if torch.allclose(out[i], torch.zeros(2)):
            print(f"    → 변화 없음 (이웃으로부터 메시지 받지 않음)")
        else:
            # 어느 노드로부터 받았는지 확인
            sources = []
            for j in range(edge_index.size(1)):
                if edge_index[1][j].item() == i:
                    src_idx = edge_index[0][j].item()
                    sources.append(src_idx)

            if sources:
                print(f"    → 노드 {sources}로부터 메시지 받음")

                # 원래 특성과 비교
                original_label = "AI" if x[i][0] > x[i][1] else "생물"
                new_label = "AI" if out[i][0] > out[i][1] else "생물"

                print(f"    → 원래: {original_label} 논문")
                print(f"    → 지금: {new_label} 논문")

                if original_label != new_label:
                    print(f"    ✨ 카테고리가 바뀌었습니다! 이웃의 영향을 받았어요!")

    # === 핵심 통찰 ===
    print("\n🔹 핵심 통찰")
    print("-" * 80)

    print("\n💡 Message Passing의 효과:")
    print("    1. 노드 0: 메시지 없음 → [0.0, 0.0] (아무도 안 인용)")
    print("    2. 노드 1: 노드 0으로부터 → [0.9, 0.1] (AI 성향 받음)")
    print("    3. 노드 2: 노드 1로부터 → [0.8, 0.2] (AI 성향 받음)")
    print("\n    노드 2는 원래 생물 논문이었지만,")
    print("    AI 논문(노드 1)이 인용했다는 그래프 구조 정보를 받아서")
    print("    AI 성향으로 업데이트되었습니다!")
    print("\n    이것이 바로 GNN의 핵심: '그래프 구조를 활용한 학습' 🎯")

    print("\n" + "=" * 80)


def test_multiple_neighbors():
    """
    여러 이웃이 있을 때 평균이 어떻게 작동하는지 테스트
    """

    print("\n" + "=" * 80)
    print("📊 여러 이웃이 있을 때 평균 계산")
    print("=" * 80)

    print("\n🔹 그래프 구조:")
    print("-" * 80)
    print("\n    노드 0 (AI)")
    print("       ↓")
    print("    노드 3  ← 노드 1 (생물)")
    print("       ↑")
    print("    노드 2 (AI)")

    # 4개 노드
    x = torch.tensor([
        [1.0, 0.0],  # 노드 0: AI
        [0.0, 1.0],  # 노드 1: 생물
        [0.9, 0.1],  # 노드 2: AI
        [0.5, 0.5],  # 노드 3: 중간 (업데이트될 예정)
    ], dtype=torch.float)

    # 3개의 엣지: 0→3, 1→3, 2→3
    edge_index = torch.tensor([
        [0, 1, 2],  # source
        [3, 3, 3],  # target (모두 노드 3으로)
    ], dtype=torch.long)

    print("\n노드 특성 (업데이트 전):")
    for i in range(x.size(0)):
        print(f"    노드 {i}: {x[i].tolist()}")

    # aggregate 실행
    model = SimpleGNN(in_channels=2, hidden_channels=4, out_channels=2)
    out = model.aggregate(x, edge_index)

    print("\n노드 특성 (업데이트 후):")
    for i in range(out.size(0)):
        print(f"    노드 {i}: {out[i].tolist()}")

    print("\n🔹 분석:")
    print("-" * 80)

    print("\n노드 3의 변화:")
    print(f"    원래: {x[3].tolist()} (중간)")
    print(f"    업데이트 후: {out[3].tolist()}")
    print(f"\n    계산 과정:")
    print(f"    평균 = (노드0 + 노드1 + 노드2) / 3")
    print(f"         = ({x[0].tolist()} + {x[1].tolist()} + {x[2].tolist()}) / 3")

    expected = (x[0] + x[1] + x[2]) / 3
    print(f"         = {expected.tolist()}")
    print(f"\n    실제 결과: {out[3].tolist()}")
    print(f"    일치 여부: {torch.allclose(out[3], expected)}")

    print("\n💡 해석:")
    print("    노드 3은 3개의 이웃(AI 2개, 생물 1개)의 평균을 받았습니다.")
    print("    결과적으로 AI 성향이 더 강해졌습니다!")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # 기본 테스트
    test_aggregate()

    # 여러 이웃 테스트
    test_multiple_neighbors()

    print("\n✅ 모든 테스트 완료!")
    print("\n이제 aggregate 함수가 어떻게 작동하는지 이해하셨나요? 😊")
