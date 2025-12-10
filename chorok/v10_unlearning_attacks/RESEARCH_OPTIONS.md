# V10: Novel Research Directions for LLM Unlearning

**Status**: Research Gap Analysis Complete
**Previous**: V9 (entropy-based hiding detection - found existing research overlap)
**Date**: 2025-12-11

---

## Executive Summary

V9 실험에서 기존 연구와의 overlap을 발견했습니다:
- FADE (distributional equivalence) - 우리의 KL divergence 아이디어와 동일
- Mechanistic Unlearning - layer-wise analysis 이미 연구됨
- Per-layer probing - 이미 존재

이 문서는 **진짜 novel한 연구 gap**을 정리합니다.

---

## Research Gap Analysis

### 이미 연구된 영역 (❌ 피해야 함)

| 연구 영역 | 주요 논문 | 우리 아이디어와 겹침 |
|----------|----------|-------------------|
| Output distribution comparison | FADE (2024) | KL(unlearned \|\| base) 측정 |
| Layer-wise representation | Mechanistic Unlearning (2024) | 어느 layer에서 forgetting 발생? |
| Per-layer probing | LLM Unlearning Under the Microscope | probe로 forgotten knowledge 탐지 |
| Obfuscation vs True Forgetting | Unlearning vs. Obfuscation (2025) | hiding signature 탐지 |

---

## 유망한 Research Options

### Option A: Reasoning Attack on TOFU ⭐⭐⭐ (가장 추천)

**Novelty**: 🟢🟢 높음 (2025년 6월 논문, 아직 TOFU에 적용 안 됨)

**핵심 발견 (기존 논문)**:
> "Step-by-Step Reasoning Attack: Revealing 'Erased' Knowledge in Large Language Models"
> - RMU method: **100% failure rate** across all question types
> - Reasoning prompts로 forgotten data 추출 가능

**TOFU에서의 연구 질문**:
> "TOFU의 IdkDPO/GradDiff/NPO 모델이 step-by-step reasoning attack에 얼마나 취약한가?"

**실험 설계**:
```python
# 기존 직접 질문 (Phase 2.6에서 테스트)
Q: "What genre does Hina Ameen write?"
A: "I don't know" (IdkDPO가 거부함)

# Reasoning Attack (새로운 실험)
Q: """Let's think step by step about Hina Ameen:
   1. She was born in Karachi, Pakistan on 06/30/1975
   2. Her father was a Real Estate Agent
   3. She received the Booker Prize
   Based on these biographical details, what field does she specialize in?"""
A: ??? (forgotten knowledge 누출 가능성)
```

**장점**:
- V9 실험 결과와 직접 연결 (IdkDPO의 "confused" 상태가 reasoning에 취약할 수 있음)
- 기존 모델/데이터 재사용 가능
- 공격 성공 시 강력한 contribution

**단점**:
- 이미 "Sleek" 논문이 일부 방법에 적용 (단, TOFU는 아님)

**References**:
- [Step-by-Step Reasoning Attack](https://arxiv.org/html/2506.17279v1) (2025.06)
- [R-TOFU: Unlearning in Large Reasoning Models](https://aclanthology.org/2025.emnlp-main.265.pdf)

---

### Option B: Knowledge Entanglement Analysis ⭐⭐

**Novelty**: 🟢 중간 (활발히 연구 중이나 TOFU 특화 분석 부족)

**핵심 문제**:
> "Even after direct unlearning, an LLM may still recall forgotten information by leveraging **related knowledge**"

**TOFU에서의 연구 질문**:
> "TOFU에서 작가 A를 잊으면, 유사한 프로필의 작가 B에 대한 답변도 영향 받는가?"

**실험 설계**:
```python
# TOFU 작가들 간 유사도 분석
similarity_matrix = compute_author_similarity(tofu_authors)

# Forget set 작가와 Retain set 작가 중 유사한 쌍 찾기
entangled_pairs = find_high_similarity_pairs(forget_authors, retain_authors)

# Unlearning 후 entangled retain authors에 대한 성능 변화 측정
for author_forget, author_retain in entangled_pairs:
    before = measure_accuracy(model_before, author_retain)
    after = measure_accuracy(model_unlearned, author_retain)
    collateral_damage = before - after
```

**장점**:
- TOFU의 200명 작가 구조 활용
- 실용적 함의 (unlearning의 side effect 정량화)

**단점**:
- 기존 Knowledge Entanglement 연구와 차별화 필요
- 실험 설계가 복잡함

**References**:
- [EAGLE-PC: Entanglement-Aware Unlearning](https://arxiv.org/html/2508.20443)
- [Learning-Time Encoding Shapes Unlearning](https://arxiv.org/html/2506.15076v1)
- [UIPE: Removing Related Knowledge](https://arxiv.org/html/2503.04693)

---

### Option C: Quantization Attack Reproduction ⭐

**Novelty**: 🟡 낮음 (ICLR 2025에 이미 발표됨)

**핵심 발견 (기존 논문)**:
> "Catastrophic Failure of LLM Unlearning via Quantization"
> - Full precision: 21% knowledge retained
> - 4-bit quantization: **83% knowledge recovered**

**TOFU에서의 연구 질문**:
> "TOFU unlearned 모델들(IdkDPO, GradDiff, NPO)을 4-bit 양자화하면 forgotten knowledge가 복구되는가?"

**실험 설계**:
```python
# 1. Unlearned model 로드
model = load_model("idk_dpo_e10")

# 2. 양자화 전 forget set 정확도
acc_before = measure_forget_accuracy(model, forget_set)  # 예상: 낮음

# 3. 4-bit 양자화 적용
model_quantized = quantize(model, bits=4, method="GPTQ")

# 4. 양자화 후 forget set 정확도
acc_after = measure_forget_accuracy(model_quantized, forget_set)  # 예상: 높아짐

# 5. Knowledge recovery rate
recovery_rate = (acc_after - acc_before) / (1 - acc_before)
```

**장점**:
- 실험이 간단함 (양자화만 적용)
- 결과가 명확함 (복구율 측정)
- 실용적 함의 (배포 시 보안 위험)

**단점**:
- 이미 ICLR 2025 논문 있음 (reproduction 수준)
- TOFU에서의 결과가 다르지 않을 가능성

**References**:
- [Catastrophic Failure of LLM Unlearning via Quantization](https://arxiv.org/abs/2410.16454) (ICLR 2025)
- [Code](https://github.com/zzwjames/FailureLLMUnlearning)

---

### Option D: Multimodal Unlearning ⭐ (TOFU 불가)

**Novelty**: 🟢🟢 높음 (새로운 분야)

**핵심 문제**:
> "Incorporating an additional modality could affect the unlearning effectiveness"

**한계**:
- TOFU는 text-only 데이터셋
- 별도 VLM 데이터셋 필요 (FIUBench, MLLMU-Bench)
- 인프라 변경 필요

**References**:
- [Cross-Modal Attention Guided Unlearning (CAGUL)](https://arxiv.org/html/2510.07567v1)
- [MLLMU-Bench](https://huggingface.co/papers/2407.10223)

---

## 추천 순위

| 순위 | Option | Novelty | 난이도 | TOFU 활용 | 추천 이유 |
|------|--------|---------|--------|----------|----------|
| 1 | **A. Reasoning Attack** | 🟢🟢 | 중 | ✅ | 가장 새롭고, V9 결과와 연결됨 |
| 2 | B. Knowledge Entanglement | 🟢 | 상 | ✅ | TOFU 구조 활용, 실용적 함의 |
| 3 | C. Quantization Attack | 🟡 | 하 | ✅ | 쉽지만 reproduction 수준 |
| 4 | D. Multimodal | 🟢🟢 | 상 | ❌ | 새롭지만 TOFU 불가 |

---

## 다음 단계

**Option A (Reasoning Attack) 선택 시**:
1. "Sleek" 논문 상세 분석
2. TOFU forget set에 대한 reasoning prompt 설계
3. Phase 2.7 노트북을 reasoning attack으로 수정
4. IdkDPO, GradDiff, NPO 모델에 공격 실행

**Option B (Knowledge Entanglement) 선택 시**:
1. TOFU 작가 프로필 유사도 분석
2. Entangled author pairs 식별
3. Collateral damage 측정 실험 설계

---

## V9 → V10 전환 이유

V9의 핵심 가설 (entropy로 hiding 탐지)는 이미 연구됨:
- FADE: distributional equivalence
- DF-MCQ: KL divergence로 distribution flatten

V10에서는 **공격 관점**으로 전환:
- Unlearning이 "진짜 잊었는지" 공격으로 검증
- Reasoning attack이 가장 유망한 새로운 방향
