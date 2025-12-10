# Running Epistemic Articulation on GPU

## Prerequisites

```bash
# Install dependencies
pip install torch transformers accelerate
pip install trl peft bitsandbytes
pip install flash-attn --no-build-isolation  # For Flash Attention 2

# Login to HuggingFace (needed for Llama)
huggingface-cli login
```

## Quick Start

### 1. Generate Dataset (with Llama 3.1-8B)

```bash
python generate_dataset_llama.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --samples 10 \
    --output dpo_dataset_llama.json
```

Expected: ~30 min on A100, generates 25 preference pairs

### 2. Train with DPO

```bash
python train_dpo.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset dpo_dataset_llama.json \
    --output ./epistemic_llama \
    --epochs 3
```

Expected: ~1-2 hours on A100

### 3. Evaluate

```bash
python evaluate_truthfulqa.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --trained_model ./epistemic_llama
```

## Alternative Models

If Llama access is an issue:

```bash
# Mistral (no license needed)
python generate_dataset_llama.py --model mistralai/Mistral-7B-Instruct-v0.3

# Qwen (open)
python generate_dataset_llama.py --model Qwen/Qwen2.5-7B-Instruct
```

## Expected Results

| Metric | Base Model | After Training |
|--------|------------|----------------|
| Alignment Rate | ~30-40% | ~70-80% (target) |
| Factual Score | +0.5 | +0.5 (maintain) |
| Subjective Score | -0.5 to 0 | +0.5 to +1.0 |
| Impossible Score | -1.0 | +0.5 to +1.0 |

## Troubleshooting

**OOM errors:**
```python
# In train_dpo.py, reduce batch size:
batch_size: int = 1
gradient_accumulation: int = 16
```

**Flash attention not available:**
```python
# In generate_dataset_llama.py, change:
attn_implementation="eager"  # instead of "flash_attention_2"
```

**Llama access denied:**
1. Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Accept license agreement
3. Run `huggingface-cli login` with your token
