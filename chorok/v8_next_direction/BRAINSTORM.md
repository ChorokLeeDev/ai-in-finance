# V8 Research Direction Brainstorm

**Goal:** Find a novel, NeurIPS 2026-worthy research direction

**Constraint:** Must avoid saturated areas (hallucination detection, epistemic expression)

---

## Lessons from V7

What to avoid:
- Token entropy for hallucination detection (saturated)
- Training LLMs to hedge (FUT, R-Tuning exist)
- First-token analysis (already published)
- Anything that semantic entropy already does better

---

## Potential Directions

### 1. Distribution Shift Detection via Uncertainty

**Idea:** Use uncertainty to detect when deployed models encounter distribution shift in real-time.

**Why novel?**
- Most UQ work focuses on detection, not temporal monitoring
- You have rel-salt dataset with COVID-19 as natural experiment
- Real-world application: early warning system

**Research question:** Can model uncertainty spike BEFORE performance degrades, enabling proactive retraining?

**Advantages:**
- Concrete benchmark (COVID-19 Feb 2020)
- Practical application (finance, healthcare)
- Different framing than hallucination detection

---

### 2. Uncertainty-Aware Tool Use

**Idea:** Teach LLMs to use external tools when internal uncertainty is high.

**Why novel?**
- Tool use is hot topic, uncertainty integration is underexplored
- Self-supervised signal: no labels needed
- Practical: reduces hallucination by routing to tools

**Research question:** Can we train LLMs to recognize "I should look this up" vs "I know this"?

---

### 3. Calibration Without Ground Truth

**Idea:** Self-supervised calibration using consistency/entropy signals.

**Why novel?**
- Most calibration requires labeled validation sets
- Could enable calibration in new domains without labels
- Builds on V7 entropy work but different application

---

### 4. Multi-Agent Uncertainty Propagation

**Idea:** How does uncertainty propagate in multi-agent LLM systems?

**Why novel?**
- Multi-agent systems are emerging (AutoGen, CrewAI)
- Uncertainty from one agent affects downstream agents
- No existing work on UQ in multi-agent LLM systems

---

### 5. Efficient Single-Pass Uncertainty

**Idea:** Match ensemble/sampling quality with single forward pass.

**Why novel?**
- Semantic entropy requires N samples (slow)
- FactCheckmate uses probes but still needs training
- Could we distill ensemble uncertainty into single pass?

---

## Quick Assessment

| Direction | Novelty | Feasibility | Impact | NeurIPS Fit |
|-----------|---------|-------------|--------|-------------|
| Distribution Shift | High | High (have data) | High | Good |
| Tool Use | Medium-High | Medium | High | Good |
| Calibration | Medium | High | Medium | OK |
| Multi-Agent | High | Low (complex) | Medium | Risky |
| Single-Pass | Medium | Medium | High | Good |

---

## Recommendation: Direction 1 (Distribution Shift)

**Why:**
1. You already have rel-salt dataset with COVID-19 shift
2. Clear, measurable hypothesis
3. Practical application story
4. Builds on V7 uncertainty measurement skills
5. Less crowded than hallucination detection

**Concrete framing:**
"Early Warning Systems for ML Distribution Shift: A COVID-19 Case Study"

**Key experiments:**
1. Train model on pre-COVID data
2. Monitor uncertainty during COVID period
3. Show uncertainty spikes BEFORE accuracy drops
4. Demonstrate lead time for retraining trigger

---

*Created: 2025-12-10*
