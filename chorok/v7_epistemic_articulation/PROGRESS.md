# V7 Epistemic Articulation - Progress

## Validated (CPU, no GPU needed)

### Core Hypothesis: CONFIRMED
**Internal model uncertainty IS detectable through token entropy.**

Test results (`quick_test_v3.py`):
```
Factual questions avg entropy:    3.37
Subjective questions avg entropy: 4.26
Difference:                       0.89 (significant)
```

Key examples:
| Question | Type | Entropy | Response |
|----------|------|---------|----------|
| Capital of France? | factual | 2.52 | "The capital of France is Paris" |
| Best color? | subjective | 4.38 | "I think it's the best color" |

Note: Model **already** uses hedging ("I think") on uncertain questions naturally!

## Files Created

| File | Purpose | Status |
|------|---------|--------|
| `quick_test.py` | Basic entropy test | Works |
| `quick_test_v2.py` | Multi-metric analysis | Works |
| `quick_test_v3.py` | Response-level entropy | **SUCCESS** |
| `generate_dataset.py` | Generate DPO pairs | Works (40 pairs) |
| `train_dpo.py` | DPO training script | Ready |
| `evaluate_truthfulqa.py` | Evaluation | Baseline: 20% |
| `kaggle_notebook.py` | Kaggle notebook | Has issues |

## What Requires GPU

1. **Training** - DPO fine-tuning (1-3 hours on A100)
2. **Large model inference** - Llama 7B+ for stronger signals
3. **Dataset generation** - More preference pairs with large models

## Next Steps

### Option 1: Colab Pro ($10/mo)
- Get A100 GPU access
- Run `train_dpo.py` with Mistral-7B

### Option 2: KAIST Computing Resources
- Check if lab has GPU cluster access
- Run training there

### Option 3: Prove concept with CPU-trainable model
- Use distilgpt2 (smaller, faster)
- Won't match NeurIPS quality but proves the method

## Research Contributions (for NeurIPS 2026)

1. **Self-supervised uncertainty signal** - No human labels needed
2. **Entropy → hedging alignment** - Novel training objective
3. **Measurable improvement** - Can evaluate on TruthfulQA
