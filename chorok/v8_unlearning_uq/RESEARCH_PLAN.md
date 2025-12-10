# V8 Research Plan: Iterative Unlearning with Epistemic Uncertainty Feedback

## Research Question

> **Can epistemic uncertainty serve as a diagnostic and stopping criterion for iterative unlearning, distinguishing knowledge suppression (hiding) from genuine knowledge removal?**

---

## Core Hypothesis

### The Insight

Current unlearning methods "hide rather than erase" - they modify outputs but knowledge remains recoverable. The key observation:

| State | Model Output | Internal Knowledge | Epistemic UQ | Adversarial Recovery |
|-------|--------------|-------------------|--------------|---------------------|
| Before unlearning | Answers correctly | Present | Low | N/A |
| After HIDING | Refuses/wrong | Still present | Low | Possible (~55%) |
| After TRUE unlearning | Refuses/wrong | Removed | High | Not possible |

### The Mechanism

**Why UQ should distinguish hiding vs true unlearning:**

1. **Hiding preserves weights** → Model still has confident internal representations → Low uncertainty
2. **True removal changes weights** → Model must rely on priors → High uncertainty (like never learned)
3. **Base model comparison** → Provides "never learned" baseline for uncertainty level

---

## Proposed Contribution

### Novel Aspect: UQ as Iterative Stopping Criterion

**Current approach (broken):**
```
unlearn(model, forget_data, epochs=N)  # N is arbitrary
return model
```

**Our approach:**
```
while True:
    model = unlearn_step(model, forget_data)
    uq_forget = measure_epistemic_uq(model, forget_queries)
    uq_base = measure_epistemic_uq(base_model, forget_queries)

    if uq_forget >= uq_base * threshold:
        break  # True unlearning achieved

    if safety_check_failed(model):
        break  # Model collapse prevention

return model
```

### Why This Is Novel

| Existing Work | What They Do | What's Missing |
|---------------|--------------|----------------|
| Becker & Liebig 2022 | UQ for ML unlearning | Not LLMs |
| SAE methods (CRISP, etc.) | Localize knowledge | Don't verify removal |
| TOFU/WMDP benchmarks | Behavioral testing | Don't detect hiding |
| Gradient ascent | Iterative unlearning | No principled stopping |

**Our contribution:** Combine LLM uncertainty quantification with iterative unlearning as feedback loop.

---

## Experimental Design

### Phase 1: Quick Validation (1-2 weeks)

**Goal:** Does epistemic UQ differ between hiding and true unlearning?

**Setup:**
1. Base model: Llama-2-7B or Mistral-7B
2. Fine-tune on TOFU forget set (creates "knowledge")
3. Apply gradient ascent unlearning (creates "hiding")
4. Measure UQ before/after

**Key measurements:**
```python
# Step 1: Establish baseline
uq_base = measure_uq(base_model, forget_queries)  # Never learned
uq_finetuned = measure_uq(finetuned_model, forget_queries)  # Knows it

# Step 2: After unlearning
uq_unlearned = measure_uq(unlearned_model, forget_queries)

# Step 3: Test recovery
recovered_model = adversarial_finetune(unlearned_model, trigger_data)
uq_recovered = measure_uq(recovered_model, forget_queries)

# Key hypothesis tests:
assert uq_finetuned < uq_base  # Knowing = confident
assert uq_unlearned > uq_finetuned  # Unlearning increases UQ

# Critical test:
if uq_unlearned < uq_base:
    print("HIDING detected - UQ not high enough")
elif uq_unlearned >= uq_base:
    print("Possible TRUE unlearning - verify with adversarial")
```

**Success criteria for Phase 1:**
- [ ] UQ increases after unlearning (basic requirement)
- [ ] UQ_unlearned < UQ_base suggests hiding (key insight)
- [ ] Adversarial recovery correlates with UQ gap

### Phase 2: Iterative Approach (2-3 weeks)

**Goal:** Implement UQ-guided iterative unlearning

**Approach:**
1. Unlearn in small steps (not one big gradient ascent)
2. Measure UQ after each step
3. Stop when UQ reaches base model level
4. Compare to fixed-epoch baseline

**Algorithm:**
```python
def uq_guided_unlearn(model, forget_data, base_model, max_steps=100):
    uq_target = measure_uq(base_model, forget_data)

    for step in range(max_steps):
        # Small unlearning step
        model = gradient_ascent_step(model, forget_data, lr=small)

        # Measure current UQ
        uq_current = measure_uq(model, forget_data)

        # Check stopping criterion
        if uq_current >= uq_target * 0.95:
            print(f"Stopped at step {step}: UQ reached target")
            return model, step

        # Safety: prevent collapse
        if perplexity(model, retain_data) > threshold:
            print("Warning: Model degrading")
            break

    return model, max_steps
```

**Evaluation:**
- Compare adversarial recovery rates: UQ-guided vs fixed-epoch
- Compare utility preservation (retain set accuracy)
- Analyze UQ trajectory during unlearning

### Phase 3: Robustness Testing (2 weeks)

**Goal:** Validate UQ-based stopping is robust

**Tests:**
1. **Different UQ methods:** Semantic entropy, token entropy, ensemble
2. **Different unlearning methods:** GA, NPO, LoRA
3. **Different attack types:** Jailbreaking, in-context learning, fine-tuning
4. **Different knowledge types:** Facts, skills, personas

**Expected results:**
| Method | Fixed-Epoch | UQ-Guided |
|--------|-------------|-----------|
| Adversarial recovery | ~55% | <20% (target) |
| Utility retained | 85% | 85%+ |
| Training time | Fixed | Variable (often shorter) |

---

## Metrics

### Primary: Uncertainty Ratio (UR)

```
UR = UQ_unlearned / UQ_base

Where:
- UQ_base = Epistemic uncertainty of base model (never learned)
- UQ_unlearned = Epistemic uncertainty after unlearning

Interpretation:
- UR < 1: HIDING (model still knows, just hiding)
- UR ≈ 1: TRUE UNLEARNING candidate (verify with adversarial)
- UR > 1: Over-unlearned or collapsed
```

### Secondary Metrics

1. **Adversarial Recovery Rate (ARR)**
   - % of "unlearned" facts recoverable via attacks
   - Lower is better

2. **Utility Preservation (UP)**
   - Accuracy on retain set / Original accuracy
   - Higher is better

3. **UQ-ARR Correlation**
   - Does low UR predict high ARR?
   - Key validation of hypothesis

---

## UQ Methods to Compare

| Method | Cost | Quality | Use Case |
|--------|------|---------|----------|
| **Semantic Entropy** | High (N samples) | Best | Gold standard |
| **Token Entropy** | Low (1 pass) | Medium | Fast screening |
| **MC Dropout** | Medium (N passes) | Medium | Cheap ensemble |
| **Self-Verbalized** | Low | Low | Baseline |

**Recommended:** Start with token entropy for fast iteration, validate with semantic entropy.

---

## Resources Required

| Resource | Requirement | Notes |
|----------|-------------|-------|
| GPU | 1x A100 (40GB) | For 7B model fine-tuning |
| Time | ~6 weeks | Phases 1-3 |
| Data | TOFU dataset | Publicly available |
| Models | Llama-2-7B, Mistral-7B | Open weights |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| UQ doesn't distinguish hiding | Medium | High | Try multiple UQ methods |
| Computational cost too high | Low | Medium | Use efficient UQ (token entropy) |
| Model collapses during iteration | Medium | Medium | Safety checks, small steps |
| Correlation not strong enough | Medium | High | Report negative results, pivot |

---

## Success Criteria

### Minimum Viable Result
- Show UQ differs between hiding and true unlearning
- Demonstrate correlation between UQ and adversarial recovery

### Strong Result
- UQ-guided stopping reduces adversarial recovery rate significantly
- Method works across multiple UQ methods and unlearning algorithms

### Publication-Worthy
- Comprehensive study of UQ as unlearning diagnostic
- New iterative unlearning protocol with UQ feedback
- Practical recommendations for practitioners

---

## Timeline

### Week 1-2: Phase 1 (Quick Validation)
- [ ] Set up TOFU + base model
- [ ] Implement UQ measurement pipeline
- [ ] Run baseline experiments
- [ ] **Decision point:** Continue if UQ shows signal

### Week 3-4: Phase 2 (Iterative Approach)
- [ ] Implement UQ-guided unlearning loop
- [ ] Compare to fixed-epoch baseline
- [ ] Analyze UQ trajectories

### Week 5-6: Phase 3 (Robustness + Writing)
- [ ] Test across UQ methods
- [ ] Adversarial testing
- [ ] Draft findings

---

## Potential Venues

| Venue | Deadline | Fit |
|-------|----------|-----|
| NeurIPS 2026 | May 2026 | Primary target |
| ICLR 2026 | Sep 2025 | Backup (tight) |
| ACL 2026 | Feb 2026 | If more NLP-focused |

---

## Next Steps

1. **Immediate:** Download TOFU, set up evaluation pipeline
2. **This week:** Implement token entropy measurement
3. **Decision point:** After Phase 1, assess if signal is strong enough

---

*Plan created: 2025-12-10*
*Research direction: Epistemic UQ as iterative stopping criterion for LLM unlearning*
