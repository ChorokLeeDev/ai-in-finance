# Literature Review: Epistemic Articulation in LLMs

**Date:** 2025-12-10
**Focus:** Teaching LLMs to express what they know and don't know

---

## 1. Abstention & "I Don't Know" Training

### R-Tuning (NAACL 2024 Outstanding Paper)
- **Paper:** [R-Tuning: Instructing Large Language Models to Say 'I Don't Know'](https://arxiv.org/abs/2311.09677)
- **Key idea:** Refusal-aware instruction tuning that identifies knowledge gaps between pre-trained parameters and instruction data
- **Method:** Construct refusal-aware data from knowledge intersection, fine-tune to refuse OOD questions
- **Result:** Improves ability to answer known questions AND refuse unknown ones; generalizes to other tasks
- **Gap:** Binary refuse/answer, doesn't handle partial uncertainty or provenance

### US-Tuning (ACL 2025)
- **Paper:** [Know the Unknown: An Uncertainty-Sensitive Method for LLM Instruction Tuning](https://arxiv.org/abs/2406.10099)
- **Key idea:** Two-stage approach - knowledge boundary recognition + instruction adherence
- **Method:** Train on known vs unknown question datasets with "I am unsure" responses
- **Result:** Llama2-7B achieves 93% accuracy on recognizing unknown questions, beats GPT-4 by 4.2%
- **Gap:** Still binary classification, doesn't articulate *what* is uncertain within an answer

### Know Your Limits Survey (TACL 2025)
- **Paper:** [Know Your Limits: A Survey of Abstention in LLMs](https://arxiv.org/html/2407.18418v2)
- **Key insight:** Abstention is a spectrum - from full refusal to hedging to conflicting conclusions
- **Taxonomy:** Query-based, model-based, value-based abstention
- **Gap:** Survey focuses on *when* to abstain, not *how* to articulate partial knowledge

---

## 2. Uncertainty Quantification Methods

### MC Dropout for LLMs
- **Challenge:** Computationally expensive at LLM scale (multiple forward passes)
- **Tree of Uncertain Thoughts (TouT):** Uses MC Dropout for uncertainty at decision points in reasoning
- **BLoB/BLoRA:** Bayesian modeling in LoRA adapters - cheaper than full ensembles
- **Status:** Active area but mostly for classification/scoring, not generation articulation

### Semantic Entropy (Nature 2024)
- **Paper:** [Detecting hallucinations in LLMs using semantic entropy](https://www.nature.com/articles/s41586-024-07421-0)
- **Method:** Cluster semantically equivalent outputs, compute entropy over clusters
- **Use:** Hallucination detection
- **Gap:** Post-hoc detection, doesn't train model to express uncertainty

### Token-Level Uncertainty
- **TokUR:** [Token-Level Uncertainty for LLM Reasoning](https://arxiv.org/html/2505.11737)
- **Method:** Aggregate token-level uncertainties via low-rank weight perturbation
- **Decomposes:** Aleatoric (data inherent) vs epistemic (model knowledge gap)
- **Gap:** Computes uncertainty, doesn't articulate it in natural language

### Kernel Language Entropy (NeurIPS 2024)
- **Paper:** [Fine-grained Uncertainty Quantification from Semantic Similarities](https://arxiv.org/html/2405.20003v1)
- **Method:** Use semantic similarity kernels for fine-grained UQ
- **Gap:** Still outputs scores, not linguistic hedging

---

## 3. Confidence Elicitation & Verbalization

### Can LLMs Express Their Uncertainty? (ICLR 2024)
- **Paper:** [Confidence Elicitation in LLMs](https://arxiv.org/abs/2306.13063)
- **Finding:** LLMs are systematically overconfident when verbalizing confidence
- **Pattern:** Verbalized confidence clusters in 80-100% range, multiples of 5 (mimics humans)
- **Mitigation:** Human-inspired prompts, consistency sampling, aggregation strategies
- **Gap:** Focuses on numeric confidence, not natural hedging language

### Linguistic Calibration
- **Paper:** [Linguistic Calibration of Language Models](https://arxiv.org/html/2404.00474v1)
- **Goal:** Produce calibrated text-based confidence statements
- **Challenge:** Trading off factuality and specificity
- **Gap:** Calibration post-hoc, not integrated into generation training

### MetaFaith (EMNLP 2025)
- **Paper:** [Faithful Natural Language Uncertainty Expression in LLMs](https://arxiv.org/html/2505.24858)
- **Method:** Metacognitive prompting for faithful uncertainty expression
- **Finding:** Linguistic uncertainty encourages cautious user behavior, improves human-AI teaming
- **Gap:** Prompt-based, not trained behavior

---

## 4. RAG-Specific Uncertainty & Hallucination

### RAG-HAT (EMNLP 2024)
- **Paper:** [Hallucination-Aware Tuning Pipeline for LLM in RAG](https://aclanthology.org/2024.emnlp-industry.113/)
- **Method:** Train hallucination detectors, use GPT-4 to correct, DPO on preference pairs
- **Gap:** Focuses on correctness, not articulating source/uncertainty

### RAGTruth Corpus
- **Paper:** [A Hallucination Corpus for RAG](https://arxiv.org/abs/2401.00396)
- **Content:** 18K annotated responses with word-level hallucination labels
- **Types:** Contradictions (against context) vs Unsupported claims (not grounded)
- **Gap:** Benchmark for detection, not for training articulation

### ReDeEP (Mechanistic Interpretability)
- **Paper:** [Detecting Hallucinations via Mechanistic Interpretability](https://openreview.net/forum?id=ztzZDzgfrh)
- **Finding:** Hallucinations occur when Knowledge FFNs overemphasize parametric knowledge while Copying Heads fail to integrate external context
- **Method:** Decouple external context vs parametric knowledge utilization
- **Gap:** Detection mechanism, not generation guidance

### Do RALMs Know When They Don't Know?
- **Paper:** [arxiv 2509.01476](https://arxiv.org/html/2509.01476)
- **Question:** Can retrieval-augmented LMs recognize their knowledge limits?
- **Relevance:** Directly addresses our question for RAG setting

---

## 5. Linguistic Hedging in Generation

### Human-Like Uncertainty Expression
- **Paper:** [Can LLMs Express Uncertainty Like Humans?](https://arxiv.org/html/2509.24202)
- **Finding:** Models understand semantic gradation ("almost certain" > "probably not")
- **Problem:** Still systematically overconfident, don't hedge when they should

### LACIE Training
- **Finding:** LACIE training improves confidence separation, makes models hedge more on uncertain content
- **Method:** Train to implicitly signal certainty through tone and detail level
- **Gap:** Not combined with explicit uncertainty signals like MC dropout

### Psychological Perspectives
- **Paper:** [Do Language Models Mirror Human Confidence?](https://arxiv.org/html/2506.00582)
- **Finding:** Verbalized confidence influenced by persona bias, not actual uncertainty
- **Implication:** Need grounding in actual uncertainty, not just prompted behavior

---

## 6. Gap Analysis: What's Missing?

### Existing Work Does:
| Capability | Papers |
|------------|--------|
| Binary abstention (refuse/answer) | R-Tuning, US-Tuning |
| Uncertainty scores (not language) | Semantic Entropy, TokUR, KLE |
| Hallucination detection (post-hoc) | RAGTruth, ReDeEP |
| Prompt-based hedging | MetaFaith, confidence elicitation |
| RAG source detection | RAG-HAT, ReDeEP |

### Nobody Does:
1. **Train** model to express fine-grained uncertainty in natural language (not prompting)
2. **Combine** token-level UQ (MC dropout) with linguistic articulation training
3. **Provenance-aware** hedging: "The documents say X, but I'm uncertain about Y"
4. **Decomposed** articulation: "I know A, I'm unsure about B, I don't know C"

---

## 7. Research Opportunity

### Your Proposed Contribution

**"Epistemic Articulation Training"** combines:
1. **MC Dropout** → Token/span-level uncertainty signals
2. **Masking** → Controlled "unknown" situations for training
3. **Reward** → Alignment between uncertainty and linguistic hedging
4. **RAG extension** → Provenance tracking (retrieved vs parametric vs inferred)

### Why This Is Novel

| Component | Prior Work | Your Addition |
|-----------|------------|---------------|
| MC Dropout for LLMs | Used for scoring | Training signal for articulation |
| Abstention training | Binary refuse/answer | Gradient of hedging intensity |
| Linguistic hedging | Prompt-based | Trained behavior |
| RAG uncertainty | Detection | Generation with provenance |

### Differentiators from Closest Work

**vs R-Tuning/US-Tuning:**
- They: Binary (answer or refuse)
- You: Spectrum of hedging, partial answers, provenance tracking

**vs MetaFaith:**
- They: Prompt engineering for hedging
- You: Train the model to intrinsically hedge when uncertain

**vs Semantic Entropy:**
- They: Detect uncertainty post-hoc
- You: Express uncertainty during generation

**vs ReDeEP:**
- They: Interpret where hallucinations come from
- You: Train model to articulate source provenance

---

## 8. Recommended Reading (Priority Order)

### Must Read (Core Problem)
1. R-Tuning (NAACL 2024) - Closest prior work
2. US-Tuning (ACL 2025) - Alternative approach
3. Know Your Limits Survey - Comprehensive overview

### Important (Methods)
4. TokUR - Token-level uncertainty
5. Semantic Entropy (Nature 2024) - UQ method
6. MetaFaith - Linguistic uncertainty

### Context (RAG-Specific)
7. RAGTruth - Hallucination benchmark
8. ReDeEP - Mechanistic view
9. Do RALMs Know When They Don't Know?

### Background (Calibration)
10. Can LLMs Express Uncertainty? (ICLR 2024)
11. Linguistic Calibration

---

## 9. Key Questions to Address

1. **Training data:** How to generate (input, uncertainty, hedged_output) triplets?
   - Masking-based: Mask known facts, reward hedging on masked
   - Corruption-based: Introduce errors, reward skepticism
   - RAG-based: Vary retrieval quality, reward appropriate attribution

2. **Evaluation:** How to measure epistemic articulation?
   - Calibration: Does hedging align with actual uncertainty?
   - Faithfulness: Does "I don't know" correlate with errors?
   - Human eval: Do humans find it more trustworthy?

3. **Architecture:** Where does uncertainty come from?
   - MC Dropout (requires multiple passes)
   - Single-pass uncertainty heads
   - Attention pattern analysis

4. **Generalization:** Does training generalize?
   - Across domains?
   - Across uncertainty types?
   - To real OOD (not just masked)?

---

## 10. Sources

- [R-Tuning GitHub](https://github.com/shizhediao/R-Tuning)
- [Awesome-LLM-Uncertainty-Reliability-Robustness](https://github.com/jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness)
- [RAGTruth Corpus](https://arxiv.org/abs/2401.00396)
- [Abstention Survey](https://arxiv.org/html/2407.18418v2)
- [Confidence Elicitation (ICLR 2024)](https://arxiv.org/abs/2306.13063)
- [MetaFaith](https://arxiv.org/html/2505.24858)
- [TokUR](https://arxiv.org/html/2505.11737)
- [Semantic Entropy (Nature 2024)](https://www.nature.com/articles/s41586-024-07421-0)

---

*Literature review compiled on 2025-12-10*
