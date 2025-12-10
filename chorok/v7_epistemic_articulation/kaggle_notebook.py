# %% [markdown]
# # Epistemic Articulation: Teaching LLMs to Express Uncertainty
#
# This notebook trains a model to hedge when uncertain and be confident when certain.
#
# **GPU:** P100 (16GB) - Use 4-bit quantization for 7B models
#
# **Runtime:** ~2-3 hours total

# %% [markdown]
# ## 1. Setup

# %%
!pip install -q transformers accelerate bitsandbytes
!pip install -q trl peft datasets
!pip install -q scipy

# %%
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json
from tqdm import tqdm

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## 2. Hedging Detection

# %%
HEDGING_PHRASES = [
    "i think", "i believe", "i'm not sure", "i'm uncertain",
    "possibly", "probably", "maybe", "perhaps",
    "might be", "could be", "may be",
    "it seems", "it appears", "it looks like",
    "generally", "typically", "usually", "often",
    "as far as i know", "i don't know", "i'm not certain",
]

CONFIDENT_PHRASES = [
    "definitely", "certainly", "absolutely", "clearly",
    "obviously", "undoubtedly", "without doubt",
    "the answer is",
]

def detect_hedging(text):
    text_lower = text.lower()
    hedging = [p for p in HEDGING_PHRASES if p in text_lower]
    confident = [p for p in CONFIDENT_PHRASES if p in text_lower]
    return {
        "has_hedging": len(hedging) > 0,
        "hedging_score": len(hedging) - len(confident),
        "hedging_phrases": hedging,
    }

# Test
print(detect_hedging("I think Paris is the capital"))
print(detect_hedging("Paris is definitely the capital"))

# %% [markdown]
# ## 3. Load Model (4-bit Quantized)

# %%
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # No license needed

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Model loaded!")

# %% [markdown]
# ## 4. Entropy Computation

# %%
def compute_entropy(model, tokenizer, prompt, response):
    """Compute mean entropy of response tokens."""
    full_text = prompt + response
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_len = len(tokenizer(prompt)["input_ids"])

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]

    entropies = []
    for i in range(prompt_len - 1, len(logits) - 1):
        probs = F.softmax(logits[i].float(), dim=-1)
        top_probs, _ = torch.topk(probs, k=min(1000, probs.shape[0]))
        entropy = -torch.sum(top_probs * torch.log(top_probs + 1e-10)).item()
        entropies.append(entropy)

    return np.mean(entropies) if entropies else 0

def categorize_entropy(entropy):
    if entropy < 3.0:
        return "confident"
    elif entropy < 5.0:
        return "moderate"
    else:
        return "uncertain"

# %% [markdown]
# ## 5. Alignment Reward

# %%
def compute_reward(entropy_category, hedging_info):
    """
    Reward alignment between entropy and hedging:
    - High entropy + hedging = good (+1)
    - High entropy + confident = bad (-1)
    - Low entropy + confident = good (+0.5)
    - Low entropy + hedging = slightly bad (-0.5)
    """
    has_hedging = hedging_info["has_hedging"]

    if entropy_category == "uncertain":
        return 1.0 if has_hedging else -1.0
    elif entropy_category == "confident":
        return -0.5 if has_hedging else 0.5
    else:
        return 0.1 * hedging_info["hedging_score"]

# %% [markdown]
# ## 6. Generate Dataset

# %%
PROMPTS = {
    "factual": [
        "What is the capital of France?",
        "What is 2 + 2?",
        "Who wrote Romeo and Juliet?",
    ],
    "uncertain": [
        "What will the stock market do tomorrow?",
        "What is the exact population of Tokyo right now?",
        "Who will win the next World Cup?",
    ],
    "subjective": [
        "What is the best programming language?",
        "What is the meaning of life?",
        "What is the best movie ever made?",
    ],
    "impossible": [
        "What is my favorite color?",
        "What did I have for breakfast?",
        "What am I thinking right now?",
    ],
}

def format_prompt(question):
    return f"<s>[INST] {question} [/INST]"

def generate_samples(model, tokenizer, question, n_samples=5):
    """Generate multiple responses and score them."""
    prompt = format_prompt(question)
    samples = []

    for _ in range(n_samples):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        entropy = compute_entropy(model, tokenizer, prompt, response)
        hedging = detect_hedging(response)
        reward = compute_reward(categorize_entropy(entropy), hedging)

        samples.append({
            "response": response,
            "entropy": entropy,
            "hedging": hedging,
            "reward": reward,
        })

    return samples

# %%
# Generate dataset
print("Generating preference dataset...")
dataset = []

for category, questions in PROMPTS.items():
    print(f"\n{category}:")
    for question in tqdm(questions):
        samples = generate_samples(model, tokenizer, question, n_samples=5)
        samples.sort(key=lambda x: x["reward"], reverse=True)

        if samples[0]["reward"] != samples[-1]["reward"]:
            dataset.append({
                "prompt": format_prompt(question),
                "chosen": samples[0]["response"],
                "rejected": samples[-1]["response"],
                "category": category,
            })
            print(f"  ✓ {question[:30]}... margin={samples[0]['reward'] - samples[-1]['reward']:.2f}")

print(f"\nGenerated {len(dataset)} preference pairs")

# %% [markdown]
# ## 7. Prepare for DPO Training

# %%
# Convert to HuggingFace Dataset
hf_dataset = Dataset.from_list([{
    "prompt": d["prompt"],
    "chosen": d["chosen"],
    "rejected": d["rejected"],
} for d in dataset])

print(hf_dataset)

# %%
# Setup LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# %% [markdown]
# ## 8. DPO Training

# %%
# Training config
training_args = DPOConfig(
    output_dir="./epistemic_model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=2,
    max_length=512,
    beta=0.1,
    logging_steps=5,
    save_steps=50,
    bf16=True,
    remove_unused_columns=False,
    optim="paged_adamw_8bit",  # Memory efficient optimizer
)

# Need reference model for DPO
ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# %%
# Create trainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=hf_dataset,
    tokenizer=tokenizer,
)

# %%
# Train!
print("Starting DPO training...")
trainer.train()

# %%
# Save
trainer.save_model("./epistemic_model_final")
print("Model saved!")

# %% [markdown]
# ## 9. Evaluation

# %%
# Test the trained model
test_questions = [
    ("What is the capital of France?", "factual"),
    ("What will happen tomorrow?", "uncertain"),
    ("What is the best programming language?", "subjective"),
    ("What is my name?", "impossible"),
]

print("\n" + "="*60)
print("EVALUATION: Trained Model Responses")
print("="*60)

for question, category in test_questions:
    prompt = format_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    hedging = detect_hedging(response)

    print(f"\n[{category.upper()}] {question}")
    print(f"Response: {response[:150]}...")
    print(f"Hedging detected: {hedging['has_hedging']} - {hedging['hedging_phrases']}")

# %% [markdown]
# ## 10. Summary
#
# **What we built:**
# - Self-supervised training signal from entropy
# - No human labels needed
# - Model learns to hedge when uncertain
#
# **Next steps:**
# - Larger dataset (more prompts)
# - Longer training
# - Evaluate on TruthfulQA benchmark
# - Human evaluation study
