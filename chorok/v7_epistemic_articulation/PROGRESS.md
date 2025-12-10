# V7 Epistemic Articulation - Final Research Summary

## Research Question
Can we train LLMs to express internal uncertainty through hedging language, using token entropy as a self-supervised signal?

---

## Part 1: Hypothesis Validation - CONFIRMED

### Core Finding
**Token entropy DOES distinguish certain vs uncertain prompts.**

Test results (`quick_test_v3.py` on GPT-2):
```
Factual questions avg entropy:    3.37
Subjective questions avg entropy: 4.26
Difference:                       0.89 (significant)
```

### Mistral-7B Experiment (`entropy_vs_consistency.ipynb`)
Compared token entropy vs sample consistency on 26 test questions:

| Metric | Entropy | Consistency |
|--------|---------|-------------|
| Gap (halluc - factual) | 0.119 | 0.125 |
| p-value | 0.025* | 0.105 |
| Effect size (Cohen's d) | **1.00** | 0.48 |
| Correlation between metrics | -0.424 (weak, different signals) |

**Conclusion:** Entropy has stronger statistical significance and larger effect size. The two metrics capture different aspects of uncertainty.

---

## Part 2: Literature Review - CRITICAL FINDINGS

### Closely Related Work

1. **R-Tuning** (Zhang et al., Nov 2023)
   - Teaches models to refuse uncertain questions
   - Uses sample consistency for uncertainty estimation

2. **FUT: Faithful Uncertainty Tuning** (Mar 2024)
   - **Very similar to our approach**
   - Uses sample consistency (not token entropy)
   - DPO training for hedging alignment
   - Already shown to work on TruthfulQA

3. **Semantic Entropy** (Nature, 2024)
   - Clusters responses by meaning, then computes entropy
   - Explicitly claims token entropy is "naive" and "inferior"
   - State-of-the-art for hallucination detection

### Critical Gap
Our original approach (token entropy → hedging via DPO) is **not novel enough**:
- FUT already does consistency → hedging via DPO
- Semantic entropy is considered superior for detection
- Simply replacing consistency with token entropy is incremental

---

## Part 3: Potential Novel Angles

### The "Forking Tokens" Opportunity

Recent work (2024-2025) shows token entropy used in **new ways**:

1. **High-Entropy Minority Tokens for RL** (2024)
   - Top 20% high-entropy tokens drive reasoning gains
   - Called "forking points" - where model makes key decisions

2. **SENT: Semantic + Token Entropy** (2024)
   - Combines both approaches
   - Token entropy provides real-time signal

3. **Research Opportunity:**
   - Identify "forking tokens" in QA responses
   - Entropy trajectory patterns differ by question type
   - First-token entropy as early predictor

### Experiment Created: `entropy_patterns.ipynb`
Designed to explore:
- Entropy trajectories for factual vs hallucinated responses
- First-token entropy as predictor
- Entropy spikes on specific token types (names, numbers)
- **NOT RUN YET** - requires GPU

---

## Part 4: Assessment for NeurIPS 2026

### Original V7 Approach: NOT RECOMMENDED
- Novelty: **Low** (FUT exists, semantic entropy is "superior")
- Contribution: Incremental at best
- Verdict: Would likely be rejected

### Pivoted Approach (Forking Tokens): UNCERTAIN
- Novelty: **Medium-High** (recent, unexplored for hedging training)
- Risk: May not find significant patterns
- Requires: Running `entropy_patterns.ipynb` to validate

### Honest Recommendation
The original V7 hypothesis was validated technically, but the **research novelty is insufficient** for a top venue. The "forking tokens" angle is promising but requires more exploration.

---

## Files Created

| File | Purpose | Status |
|------|---------|--------|
| `quick_test_v3.py` | Response-level entropy | SUCCESS |
| `entropy_vs_consistency.ipynb` | Compare entropy vs consistency | Results: entropy wins on effect size |
| `entropy_patterns.ipynb` | Forking tokens exploration | Created, not run |
| `colab_training.ipynb` | GPU training setup | Working |
| `generate_dataset.py` | DPO pair generation | Working |
| `train_dpo.py` | DPO training | Ready |

---

## Final Literature Check (Dec 10, 2025)

### First-Token Entropy - NOT NOVEL

| Paper | Date | Finding |
|-------|------|---------|
| [First Hallucination Tokens Are Different](https://arxiv.org/abs/2507.20836) | Jul 2025 | First token more detectable - **same as our finding** |
| [HaMI](https://arxiv.org/html/2504.07863v2) | Apr 2025 | First token **6% worse** than adaptive selection |
| [FactCheckmate](https://arxiv.org/html/2410.02899) | Oct 2024 | Pre-generation detection (hidden states) - more powerful |

### Conclusion
Our first-token entropy finding (gap=0.357) was:
1. Already published (Jul 2025)
2. Known to be suboptimal (HaMI)
3. Superseded by pre-generation methods (FactCheckmate)

---

## V7 Status: CLOSED

**Reason:** Research space is saturated
- Detection: Semantic entropy, FactCheckmate, HaMI, INSIDE
- Training: FUT, R-Tuning
- Token analysis: First Hallucination Tokens paper

**Time invested:** ~2 weeks
**Value captured:**
- Deep understanding of UQ/hallucination detection landscape
- Know what NOT to pursue for NeurIPS 2026
- Working entropy measurement codebase

---

## Transition to V8

**Next direction:** Machine Unlearning Verification via Epistemic Uncertainty

The UQ expertise developed in V7 transfers directly to V8, which focuses on:
- Using epistemic uncertainty to distinguish "hiding" vs "true unlearning"
- Iterative unlearning with UQ as feedback/stopping criterion
- Addressing the acknowledged stopping criterion problem in unlearning

See: `../v8_unlearning_uq/`

---

*Closed: 2025-12-10*
*Transitioned to V8: 2025-12-10*
