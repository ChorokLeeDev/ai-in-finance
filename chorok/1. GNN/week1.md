# Week 1: GNN 기초 마스터

## 사전 준비
**⚠️ 중요: Week 1을 시작하기 전에 환경 설정을 완료하세요!**

환경 설정 가이드는 [0. Setup 폴더](../0.%20Setup/)를 참조하세요:
- **자동 설정**: [create_gnn_env.ps1](../0.%20Setup/create_gnn_env.ps1) 스크립트 실행
- **수동 설정**: [SETUP_CONDA.md](../0.%20Setup/SETUP_CONDA.md) 가이드 참조

환경 설정이 완료되면 `conda activate gnn_env` 명령으로 환경을 활성화하세요.

---

## 목표
- Graph Neural Network의 핵심 원리를 코드로 이해하기
- Message passing 직접 구현
- PyTorch Geometric 익숙해지기
- Node classification 완성

## 필수 자료
- PyG Tutorial: https://pytorch-geometric.readthedocs.io/
- Stanford CS224W: http://web.stanford.edu/class/cs224w/
- RelBench: https://github.com/snap-stanford/relbench

---

### 1. GCN (2017) - Semi-Supervised Classification with GCN
**논문**: https://arxiv.org/abs/1609.02907

#### 핵심 개념
- **Equation 1 (message passing rule)**: 각 노드는 자신과 이웃 노드의 특징 벡터를 평균낸 뒤(linear transform + normalization), 활성화 함수를 적용해 새로운 특징을 만듭니다. 즉, feature smoothing + linear transformation을 동시에 수행.

- **Renormalization trick (A + I)**: A만 사용하면 self-loop가 없기 때문에 한 레이어를 거칠 때 자기 정보가 사라짐. 또한 degree로 나누는 정규화가 불안정해질 수 있음. 자기 자신을 neighbor로 포함시켜서 해결.

- **Computational complexity: O(|E|F)**: 희소 행렬 곱셈에서는 연산 전에 어떤 항목이 0이 될지 알 수 있습니다. 인접 행렬에서 0인 부분은 연결되지 않은 노드 쌍을 의미하기 때문입니다. 따라서 그 부분의 곱셈은 굳이 계산할 필요가 없어 연산에서 제외할 수 있습니다. 즉, **연결(엣지)**이 실제 연산의 단위가 되어 O(|E|)번 연산이 일어나고, 각 연산마다 F차원 특징 벡터를 처리하므로 전체 복잡도는 **O(|E|F)**가 됩니다.

#### 요약
- **학습형태**: Transductive
- **핵심아이디어**: 인접행렬과 노드 특징 행렬 곱으로 이웃 노드 정보 집계
- **복잡도**: O(|E|F), 엣지 수 × 특징 차원에 비례
- **한계**: inductiveness 부재, 고정된 그래프에 한정

---

### 2. GraphSAGE (2017) - Inductive Representation Learning
**논문**: https://arxiv.org/abs/1706.02216

#### 핵심 개념
- **Neighborhood sampling이 왜 필요한가?**: GCN처럼 전체 그래프의 모든 neighbor를 사용하면, 대규모 그래프에서는 계산량이 폭발적. 각 노드마다 고정된 개수의 이웃만 샘플링 (e.g., 25개). mini-batch 학습이 가능해짐. 즉, scalability를 위해 도입된 핵심 기법.

- **3가지 aggregator (mean, LSTM, pool) 차이**:
  - **Mean**: 단순히 이웃 노드 특징의 평균을 계산. 안정적이고 GCN과 유사한 방식
  - **LSTM**: 이웃 노드 특징을 순서대로 처리하여 더 풍부한 표현력 확보. 노드 순서에 따라 결과가 달라질 수 있음. 연산 속도가 비교적 느림.
  - **Pool**: MLP와 max pooling을 통해 비선형성 도입. 노드 순서에 무관하게 주요 패턴 포착에 효과적. 연산 속도가 빠름.

- **Inductive vs Transductive**:

| 구분 | 설명 | 예시 |
|------|------|------|
| **Transductive** | 학습 시 전체 그래프(노드 포함)를 알고 있음 | GCN, node classification |
| **Inductive** | 학습 시 보지 못한 새 노드/새 그래프에도 적용 가능 | GraphSAGE, link prediction across graphs |

#### 요약
- **학습형태**: Inductive
- **핵심아이디어**: 노드마다 일정 수의 이웃 노드 sampling해 집계
- **복잡도**: O(NF), 노드 수에 비례 (노드마다 고정된 수 k의 이웃을 sampling)
- **한계**: Sampling으로 정보 손실 가능성

---

### 3. GAT (2018) - Graph Attention Networks
**논문**: https://arxiv.org/abs/1710.10903

#### 핵심 개념
- **Attention coefficient 계산 방법**:
  1. 각 엣지 $(i, j)$에 대해 attention 계수 $e_{ij}$를 계산:
     $$e_{ij} = \text{LeakyReLU}(a^T[W h_i \, || \, W h_j])$$
     - $h_i, h_j$: 노드 $i$와 $j$의 특징 벡터
     - $W$: 학습 가능한 가중치 행렬
     - $a$: 학습 가능한 attention 벡터
     - $||$: concatenation 연산

  2. Softmax 함수로 attention 계수 정규화:
     $$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$
     - $\mathcal{N}(i)$: 노드 $i$의 이웃 노드 집합

  최종적으로 $\alpha_{ij}$는 노드 $j$가 노드 $i$에 미치는 중요도를 나타내며, 메시지 전달 시 가중치로 사용됩니다.

- **Multi-head attention이 GNN에서 왜 효과적?**: 여러 attention head를 두면, 다양한 관계 패턴을 병렬적으로 학습 가능. 평균하거나 concat하면 representation의 안정성과 표현력이 향상됨. 특히 over-smoothing 방지, gradient 안정성 증가 효과 있음.

#### 요약
- **학습형태**: Transductive
- **핵심아이디어**: Attention 메커니즘으로 이웃 노드 중요도 학습
- **복잡도**: O(|E|F), 엣지 수에 비례
- **한계**: Inductiveness 부재, 큰 그래프에서 비용 큼

---

# Message Passing Equations 정리

## GCN (Graph Convolutional Networks)

### Layer-wise Propagation Rule
$$H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}\right)$$

**Notation:**
- $H^{(l)} \in \mathbb{R}^{N \times F}$: 레이어 $l$의 노드 특징 행렬 (N개 노드, F차원 특징)
- $\tilde{A} = A + I \in \mathbb{R}^{N \times N}$: self-loop가 추가된 인접 행렬
- $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$: $\tilde{A}$의 degree 행렬 (대각 행렬)
- $W^{(l)} \in \mathbb{R}^{F \times F'}$: 학습 가능한 가중치 행렬
- $\sigma$: 활성화 함수 (예: ReLU)

**의미**: 각 노드의 새로운 표현은 자신과 이웃 노드들의 특징을 degree에 따라 정규화한 후 가중합하여 생성. Symmetric normalization ($\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$)을 통해 노드 degree 차이를 보정.

**Self-loop ($A + I$)가 레이어에만 있는 이유**: 원본 그래프는 수정하지 않고, GNN 계산 시에만 임시로 추가. 각 노드가 이웃 정보뿐만 아니라 자기 자신의 정보도 유지하기 위함.

**Degree로 정규화하는 이유**: 정규화 없이 이웃 특징을 단순히 더하면, degree가 큰 노드(허브 노드)는 값이 매우 커지고 degree가 작은 노드는 작아짐. $\frac{1}{\sqrt{d_i d_j}}$로 나누면 평균 효과가 생겨 값 범위가 안정적이고, 모든 노드를 공정하게 비교 가능.

### Node-wise 형태
$$h_i^{(l+1)} = \sigma\left(\sum_{j \in \mathcal{N}_i \cup \{i\}} \frac{1}{\sqrt{d_i d_j}} W^{(l)} h_j^{(l)}\right)$$

**Notation:**
- $h_i^{(l)} \in \mathbb{R}^{F}$: 노드 $i$의 레이어 $l$에서의 특징 벡터
- $\mathcal{N}_i$: 노드 $i$의 이웃 노드 집합
- $d_i$: 노드 $i$의 degree (= $|\mathcal{N}_i|$)

**의미**: 각 노드는 자신과 이웃들의 특징을 $\frac{1}{\sqrt{d_i d_j}}$로 정규화하여 평균. 이웃이 많은 노드(high degree)의 영향력을 줄이고, 이웃이 적은 노드의 영향력을 높임.

**직관**: "이웃들의 feature를 degree로 정규화해서 평균내기"

**예시** (Cora 데이터셋): N=2708 노드, F=1433 특징, 평균 degree ≈ 3.9

---

## GraphSAGE (Sample and Aggregate)

### Message Passing Rule
$$h_i^{(k)} = \sigma\left(W^{(k)} \cdot \text{CONCAT}\left(h_i^{(k-1)}, \text{AGG}\left(\{h_j^{(k-1)}, \forall j \in \mathcal{N}_i\}\right)\right)\right)$$

**Notation:**
- $h_i^{(k)} \in \mathbb{R}^{F_k}$: 노드 $i$의 레이어 $k$에서의 임베딩
- $\text{AGG}$: Aggregation 함수 (mean, LSTM, pool 등)
- $\text{CONCAT}$: 벡터 연결 (concatenation)
- $W^{(k)} \in \mathbb{R}^{F_k \times 2F_{k-1}}$: 학습 가능한 가중치 행렬

**의미**: 자신의 특징 $h_i^{(k-1)}$과 이웃들의 집계된 특징 $\text{AGG}(\cdot)$를 연결(concat)한 후, 가중치 행렬로 변환. 이를 통해 자신의 정보와 이웃 정보를 명시적으로 구분.

### Aggregator 종류

**1) Mean Aggregator**
$$\text{AGG}_{\text{mean}} = \frac{1}{|\mathcal{N}_i|}\sum_{j \in \mathcal{N}_i} h_j$$

**의미**: 이웃들의 특징 벡터를 단순 평균. GCN과 유사하지만 self-loop 없이 이웃만 고려.

**2) LSTM Aggregator**
$$\text{AGG}_{\text{LSTM}} = \text{LSTM}\left(\{h_j, \forall j \in \pi(\mathcal{N}_i)\}\right)$$

**Notation:**
- $\pi(\mathcal{N}_i)$: 이웃 노드의 랜덤 순열 (permutation)
- **LSTM**: 순차 데이터를 순서대로 처리하며 이전 정보를 기억하는 신경망 (예: 문장 "나는 학교에 간다"를 단어별로 읽으며 의미 누적)

**의미**: LSTM을 사용해 이웃 노드를 순차적으로 처리. 더 풍부한 표현력을 제공하지만, 그래프는 원래 순서가 없으므로 랜덤 순열 $\pi$로 섞어서 사용. 계산 비용이 높음.

**3) Pooling Aggregator**
$$\text{AGG}_{\text{pool}} = \max\left(\{\sigma(W_{\text{pool}} h_j + b), \forall j \in \mathcal{N}_i\}\right)$$

**의미**: 각 이웃 특징에 MLP를 적용한 후 element-wise max pooling. 비선형성을 도입하면서도 순서 무관.

**직관**: "이웃 중 일부만 샘플링(고정 개수 K)해서 aggregation 함수로 모으기"

**예시**: 노드당 25개 이웃 샘플링 → 복잡도 O(25 × F) = O(F)

---

## GAT (Graph Attention Networks)

GAT는 3단계로 구성됩니다: **(1) Attention Coefficient 계산** → **(2) Attention Mechanism으로 가중합** → **(3) Multi-head로 병렬 실행**

### Step 1: Attention Coefficient 계산
**목적**: 각 이웃 노드가 얼마나 중요한지 점수를 매기기

$$e_{ij} = \text{LeakyReLU}\left(a^T [W h_i \, || \, W h_j]\right)$$

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$

**Notation:**
- $e_{ij} \in \mathbb{R}$: 노드 $i$와 $j$ 사이의 raw attention score (정규화 전)
- $\alpha_{ij} \in [0,1]$: 노드 $j$가 노드 $i$에 미치는 attention 가중치 ($\sum_{j \in \mathcal{N}_i} \alpha_{ij} = 1$)
- $a \in \mathbb{R}^{2F'}$: 학습 가능한 attention 메커니즘 벡터
- $W \in \mathbb{R}^{F' \times F}$: 학습 가능한 선형 변환 행렬
- $||$: 벡터 연결 (concatenation)
- $\text{LeakyReLU}$: 음수에서도 작은 기울기를 가지는 활성화 함수

**계산 과정**:
1. 두 노드의 변환된 특징 $W h_i$와 $W h_j$를 concat
2. 학습 가능한 벡터 $a$와 내적하여 attention score $e_{ij}$ 계산
3. Softmax로 정규화하여 확률 분포 $\alpha_{ij}$ 생성 (합이 1이 되도록)

#### 파라미터 상세 설명

**예시 설정**: 입력 차원 $F=4$, 출력 차원 $F'=2$

**(1) 선형 변환 행렬 (Linear Transformation Matrix) $W \in \mathbb{R}^{F' \times F}$**

- **역할**: 입력 노드 특징을 새로운 표현 공간으로 변환 (일반 신경망의 weight와 동일)
- **크기**: $(F' \times F) = (2 \times 4)$
- **초기화**: 랜덤 초기화 (예: Xavier, Kaiming initialization)
- **학습**: 역전파(backpropagation)로 업데이트

예시:
$$W = \begin{bmatrix} 0.5 & -0.3 & 0.2 & 0.1 \\ 0.4 & 0.6 & -0.1 & 0.3 \end{bmatrix}, \quad h_i = \begin{bmatrix} 1.0 \\ 0.5 \\ -0.2 \\ 0.8 \end{bmatrix}$$

$$W h_i = \begin{bmatrix} 0.5 \times 1.0 + (-0.3) \times 0.5 + 0.2 \times (-0.2) + 0.1 \times 0.8 \\ 0.4 \times 1.0 + 0.6 \times 0.5 + (-0.1) \times (-0.2) + 0.3 \times 0.8 \end{bmatrix} = \begin{bmatrix} 0.39 \\ 0.96 \end{bmatrix}$$

**(2) Attention 메커니즘 벡터 $a \in \mathbb{R}^{2F'}$**

- **역할**: 두 노드의 concat된 특징 $[W h_i \, || \, W h_j]$를 스칼라 점수로 압축
- **크기**: $(2F') = (4)$ — $W h_i$ 차원 $F'=2$ + $W h_j$ 차원 $F'=2$
- **왜 두 노드를 concat?**: 그래프에서는 노드 간 **연결(edge)의 중요도**를 평가해야 하므로, 두 노드의 특징을 함께 봐야 관계를 파악 가능. $h_i$만 보거나 $h_j$만 보면 둘 사이의 호환성을 알 수 없음.
- **초기화**: 랜덤 초기화
- **학습**: 역전파로 업데이트

예시:
$$a = \begin{bmatrix} 0.8 \\ -0.5 \\ 0.3 \\ 0.6 \end{bmatrix}, \quad W h_i = \begin{bmatrix} 0.39 \\ 0.96 \end{bmatrix}, \quad W h_j = \begin{bmatrix} 0.52 \\ 0.71 \end{bmatrix}$$

$$[W h_i \, || \, W h_j] = \begin{bmatrix} 0.39 \\ 0.96 \\ 0.52 \\ 0.71 \end{bmatrix}$$

$$a^T [W h_i \, || \, W h_j] = 0.8 \times 0.39 + (-0.5) \times 0.96 + 0.3 \times 0.52 + 0.6 \times 0.71 = 0.414$$

**(3) Raw Attention Score $e_{ij}$**

- **역할**: 노드 $i$와 $j$의 관련성을 나타내는 정규화 전 점수
- **계산**: $e_{ij} = \text{LeakyReLU}(a^T [W h_i \, || \, W h_j])$
- **LeakyReLU**: $\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ 0.01 \times x & \text{if } x \leq 0 \end{cases}$
- **활성화 함수가 필요한 이유**: 선형 변환 $W$와 내적 $a^T$만으로는 선형 결합에 불과. 비선형성을 추가해야 복잡한 관계 패턴 학습 가능.
- **일반 ReLU vs LeakyReLU**: 일반 ReLU는 $x \leq 0$일 때 출력이 0이 되어 gradient가 죽는 문제 발생(dying ReLU). LeakyReLU는 음수에서도 작은 기울기(0.01)를 유지하여 학습 안정성 향상.

예시:
$$e_{ij} = \text{LeakyReLU}(0.414) = 0.414$$

모든 이웃에 대해 계산 후 Softmax로 정규화:
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$

**핵심**: $W$와 $a$는 **학습되는 파라미터**, $e_{ij}$와 $\alpha_{ij}$는 **계산되는 값**

### Step 2: Attention Mechanism (Single-head)
**목적**: 계산된 attention coefficient로 이웃들의 특징을 가중합

$$h_i' = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} W h_j\right)$$

**Notation:**
- $h_i' \in \mathbb{R}^{F'}$: 노드 $i$의 업데이트된 특징 벡터
- $\sigma$: 활성화 함수 (주로 ReLU 또는 ELU). 비선형성을 추가하여 복잡한 패턴 학습 가능. 없으면 여러 레이어를 쌓아도 하나의 선형 변환과 동일.

**의미**: Step 1에서 계산된 attention 가중치 $\alpha_{ij}$를 사용하여, 각 이웃의 변환된 특징 $W h_j$를 가중합. 중요한 이웃일수록 더 큰 영향을 미침.

### Step 3: Multi-head Attention
**목적**: 여러 attention head를 병렬로 실행하여 다양한 관계 패턴 학습

$$h_i' = \Big|\Big|_{k=1}^K \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k W^k h_j\right)$$

**Notation:**
- $K$: attention head의 개수
- $\alpha_{ij}^k$: $k$번째 head의 attention 가중치 (각 head마다 독립적으로 Step 1 수행)
- $W^k \in \mathbb{R}^{F' \times F}$: $k$번째 head의 가중치 행렬
- $||_{k=1}^K$: K개 벡터를 연결 (concatenation)

**의미**:
- 각 head는 독립적인 파라미터 $(W^k, a^k)$를 가지고 Step 1-2를 수행
- K개의 서로 다른 attention 결과를 concat하여 최종 특징 생성
- 각 head는 다른 관계 패턴(예: 인용 관계, 공동 저자 관계 등)을 학습 가능
- 마지막 레이어에서는 averaging 사용: $h_i' = \frac{1}{K}\sum_{k=1}^K \sigma(\cdots)$

**계층 구조**:
```
Multi-head Attention (K개 병렬 실행)
  ├─ Head 1: Attention Mechanism
  │   ├─ Attention Coefficient 계산 (a^1, W^1 사용)
  │   └─ 가중합 계산
  ├─ Head 2: Attention Mechanism
  │   ├─ Attention Coefficient 계산 (a^2, W^2 사용)
  │   └─ 가중합 계산
  └─ ...

최종 출력 = [Head1 출력 || Head2 출력 || ... || HeadK 출력]
```

**직관**: "이웃마다 다른 중요도(attention)를 학습해서 weighted sum"

**예시**: K=8 heads, F'=8 → 출력 차원 = 64. Over-smoothing 방지에 효과적.

---

## 세 모델 비교

| 특징 | GCN | GraphSAGE | GAT |
|------|-----|-----------|-----|
| **Aggregation** | Mean (고정) | 선택 가능 (mean/LSTM/pool) | Attention (학습) |
| **이웃 중요도** | 균등 ($1/\sqrt{d_i d_j}$) | 균등 | 차등 ($\alpha_{ij}$) |
| **학습 방식** | Transductive | Inductive | Inductive |
| **샘플링** | 전체 이웃 | 고정 개수 (S개) | 전체 이웃 |
| **복잡도** | $O(\|E\| \cdot F)$ | $O(N \cdot S \cdot F)$ | $O(\|E\| \cdot F')$ |
| **장점** | 간단, 빠름 | 확장성, 새 노드 적용 | 해석성, 성능 |
| **단점** | 새 그래프 적용 어려움 | 샘플링으로 정보 손실 | 계산 비용 높음 |

**복잡도 표기**: $N$ = 노드 수, $|E|$ = 엣지 수, $F$ (또는 $F'$) = 특징 차원, $S$ = 샘플링 개수 (상수)
- **특징 차원 $F$**: 각 노드를 나타내는 벡터의 차원. 예: Cora 논문은 1433차원(단어 등장 여부), 소셜 네트워크 사용자는 128차원(나이, 성별, 관심사 등을 임베딩).
- **Hidden channels**: GNN 중간 레이어의 특징 차원. 예: 입력 1433차원 → hidden 16차원 → 출력 7차원(클래스). 하이퍼파라미터로 조정 가능.
- **Dropout**: 과적합 방지를 위해 학습 중 특징 벡터의 일부 차원을 랜덤하게 0으로 만드는 정규화 기법. GraphSAGE의 샘플링(이웃 노드 제한)과는 다름.
- **임베딩**: 범주형 데이터를 연속 벡터로 변환.
  - **초기화**: 직업 "학생"을 랜덤 벡터 $[0.01, -0.02, 0.03, ...]$ (32차원)로 시작 (처음엔 의미 없음)
  - **학습**: 태스크(예: 친구 추천, 논문 분류)를 수행하며 역전파로 테이블 값 업데이트 → 비슷한 범주는 비슷한 벡터로, 다른 범주는 먼 벡터로 자동 학습
  - **사용**: 직업(32차원), 성별(8차원), 나이(1차원) 등을 concat → MLP로 압축 → 고정 차원(F=128) 생성
  - **핵심**: 사람이 규칙을 정하지 않고, 신경망이 데이터에서 패턴을 발견하여 자동으로 의미 있는 숫자 학습
- **GCN**: 모든 엣지 처리 → 엣지 개수에 비례
- **GraphSAGE**: 노드마다 S개만 샘플링 → 노드 개수에 비례 (S는 상수이므로 $O(N)$)
- **GAT**: 모든 엣지에 attention 계산 → 엣지 개수에 비례

### 공통점
- Message passing 프레임워크
- 이웃의 feature를 aggregate
- 여러 layer 쌓기 가능

### 차이점의 핵심
```
GCN:       "평균" (fixed normalization)
GraphSAGE: "샘플링 + aggregation 함수"
GAT:       "attention으로 가중치 학습"
```

---

# 구현 코드

## SimpleGNN from scratch

```python
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

class SimpleGNN(nn.Module):
    """
    가장 기본적인 Graph Convolution을 직접 구현
    목적: Message passing의 원리를 완전히 이해
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1_weight = nn.Linear(in_channels, hidden_channels)
        self.conv2_weight = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        # x: (num_nodes, in_channels)
        # edge_index: (2, num_edges)
        
        # Layer 1: Transform + Aggregate + Activate
        x = self.conv1_weight(x)  # (N, hidden)
        x = self.aggregate(x, edge_index)  # Message passing
        x = torch.relu(x)
        
        # Layer 2
        x = self.conv2_weight(x)
        x = self.aggregate(x, edge_index)
        
        return x
    
    def aggregate(self, x, edge_index):
        """
        핵심: Message passing 구현
        각 노드는 이웃들의 feature를 평균낸다
        """
        # edge_index[0]: source nodes
        # edge_index[1]: target nodes
        
        # Step 1: Compute node degrees for normalization
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float('inf')] = 0
        
        # Step 2: Aggregate messages from neighbors
        out = torch.zeros_like(x)
        for src, dst in zip(row, col):
            # 이웃의 feature를 더함
            out[dst] += x[src] * deg_inv[dst]
        
        return out
```

## PyTorch Geometric 구현

```python
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    """
    PyTorch Geometric 사용
    SimpleGNN과 동일한 결과를 내야 함
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
```

## Cora 데이터셋 학습

```python
def train_gnn():
    """
    Cora 데이터셋: 논문 인용 그래프
    - 2708개 논문 (nodes)
    - 5429개 인용 관계 (edges)
    - 7개 카테고리 분류
    
    목표: 85% 이상 test accuracy
    """
    
    # 데이터 로드
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0]
    
    print("Dataset Statistics:")
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges}")
    print(f"Features: {data.num_node_features}")
    print(f"Classes: {dataset.num_classes}")
    print(f"Train mask: {data.train_mask.sum().item()}")
    print(f"Val mask: {data.val_mask.sum().item()}")
    print(f"Test mask: {data.test_mask.sum().item()}")
    
    # 모델 초기화
    model = GCN(
        in_channels=dataset.num_node_features,
        hidden_channels=16,
        out_channels=dataset.num_classes
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 학습
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            # Validation
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1)
                train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
            model.train()
            
            print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    # 최종 테스트
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
    
    print(f'\nFinal Test Accuracy: {test_acc:.4f}')
    return model, data, test_acc
```

---

# Week 1 완료 조건

## 논문 이해
- [ ] GCN, GraphSAGE, GAT 논문 읽고 핵심 개념 이해
- [ ] Message passing equation을 수식으로 쓸 수 있음
- [ ] 세 모델의 차이점 설명 가능

## 코드 구현
- [ ] SimpleGNN을 from scratch로 구현 완료
- [ ] Cora에서 85% 이상 test accuracy 달성
- [ ] PyTorch Geometric 기본 함수들 사용 가능

## 실험
- [ ] Hyperparameter tuning 실험 3가지 이상 (hidden_channels, layers, dropout 등)
- [ ] 결과 분석 및 시각화

## 이해도 점검
- [ ] "GNN이 왜 작동하는가?" 설명 가능
- [ ] "언제 GNN을 쓰면 안 되는가?" 답변 가능
- [ ] SimpleGNN과 GCN이 같은 결과를 내는지 검증
- [ ] Self-loop 추가/제거의 영향 이해

### GNN이 왜 작동하는가?

**핵심**: Message passing을 통해 그래프의 **구조(topology)**와 **특징(feature)**을 동시에 학습

1. **구조 정보 활용**: 연결된 노드끼리 비슷한 특징을 공유하는 경향(homophily)을 활용. 예: 같은 분야 논문끼리 인용, 같은 관심사 친구끼리 연결
2. **이웃 집계**: 여러 레이어를 거치며 점점 더 먼 이웃의 정보를 모음 (1-hop → 2-hop → 3-hop). 각 노드는 주변 그래프 구조의 요약(summary)을 학습
3. **비선형 변환**: 활성화 함수($\sigma$)와 학습 가능한 가중치($W$)로 복잡한 패턴 학습
4. **귀납적 편향(inductive bias)**: 그래프 구조를 명시적으로 활용하므로, 완전 연결 신경망보다 적은 데이터로도 효과적 학습

**직관**: "친구의 친구는 나와 비슷할 것이다" - 이웃 정보를 모으면 노드 자체를 더 잘 이해할 수 있음

### 언제 GNN을 쓰면 안 되는가?

1. **그래프 구조가 의미 없을 때**
   - 예: 노드가 랜덤으로 연결된 경우, 엣지가 노이즈인 경우
   - 해결: 일반 MLP 사용

2. **Over-smoothing 문제**
   - 레이어를 많이 쌓으면(>10) 모든 노드의 표현이 비슷해짐 (구별 불가)
   - 이유: 이웃 정보를 계속 평균내면 결국 전체 그래프의 평균으로 수렴
   - 해결: Residual connection, Jumping Knowledge Network, 적은 레이어 사용

3. **Heterophily (이질성)가 강할 때**
   - 연결된 노드끼리 **다른** 특징을 가지는 경우 (예: 사기꾼-정상 유저 연결)
   - GNN은 homophily(유사성) 가정에 기반
   - 해결: Heterophily-aware GNN (예: H2GCN, GPR-GNN)

4. **노드 간 연결이 희박하거나 없을 때**
   - Isolated 노드가 많으면 이웃 정보 활용 불가
   - 해결: 가상 엣지 추가, feature-only 방법 병행

5. **Long-range dependency가 중요할 때**
   - 매우 먼 거리의 노드 간 관계가 중요한 경우
   - GNN은 local 정보 위주 (레이어 수 제한)
   - 해결: Graph Transformer, 전역 정보 추가

**핵심**: GNN은 "이웃이 유사하다"는 가정에 의존. 이 가정이 깨지면 성능 하락.
