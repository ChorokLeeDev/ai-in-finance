# Risk Manager's Guide to Hierarchical Uncertainty Analysis

**For**: Risk managers, data scientists, ML practitioners
**Purpose**: Understand and act on model uncertainty in relational data

---

## The Core Problem

Your ML model says "I'm 95% confident in this prediction" but you don't know:
- Is that confidence trustworthy?
- If not, WHERE is the data problem?
- WHAT should I fix, and HOW MUCH will it help?

**This framework answers all three questions.**

---

## The Unknown Unknowns Problem

### Three Types of Model States

| State | Model Says | Reality | Risk Level |
|-------|------------|---------|------------|
| **Unknown Unknowns** | "Very confident" | Has never seen this region | **HIGHEST RISK** |
| **Known Unknowns** | "Very uncertain" | Knows it's a hard case | **MEDIUM RISK** |
| **Known Knowns** | "Reasonably confident" | Actually learned the pattern | **LOW RISK** |

### Why Unknown Unknowns Are Dangerous

```
Model trained on shipping points 0-6 (common routes)
New data comes in for shipping points 7-79 (rare routes)

Model says: "I'm 99.9% confident the delivery will be on time"
Reality: Model has NEVER seen these routes, it's guessing

This is the HIGHEST risk scenario:
- Model is confident
- Model is wrong
- You don't know it's wrong
```

### The Uncertainty Journey

When you start collecting data for a problematic region:

```
Stage 1: OVERCONFIDENT (0% target data)
         Uncertainty: 0.001 (very low)
         Model has never seen this region
         Confidently predicting based on unrelated patterns
         → DANGEROUS: Unknown unknowns

Stage 2: CORRECTLY UNCERTAIN (25% target data)
         Uncertainty: 0.101 (high)
         Model has now seen some hard cases
         Realizes "this is different from what I know"
         → GOOD: Converted to known unknowns

Stage 3: LEARNED (100% target data)
         Uncertainty: 0.066 (moderate)
         Model has learned the patterns
         Can make informed predictions
         → BEST: Known knowns
```

**Key insight**: Uncertainty INCREASING is often GOOD - it means the model discovered it was overconfident.

---

## What This Framework Tells You

### Level 1: Which Table? (Obvious)

```
"ITEM table contributes most to uncertainty"
```

You already knew this. A domain expert could tell you ITEM matters for sales prediction.

**Not the value-add.**

### Level 2: Which Column? (Useful)

```
"Within ITEM, SHIPPINGPOINT column contributes 227% of the uncertainty"
"ITEMINCOTERMSCLASS contributes only 24%"
```

Now you know WHERE in the ITEM data to focus. SHIPPINGPOINT is 9x more important than ITEMINCOTERMSCLASS.

**This is actionable.**

### Level 3: Which Values? (Actionable)

```
"Within SHIPPINGPOINT, values [7-79] contribute 92% of uncertainty (n=201 samples)"
"Values [0-6] contribute 49% of uncertainty (n=599 samples)"
```

Now you know EXACTLY which shipping points need more data.

**This is your investment decision.**

---

## The Investment Decision

### Quantified ROI

From our experiments:

| Domain | Target Region | Samples Needed | Uncertainty Reduction |
|--------|---------------|----------------|----------------------|
| F1 Racing | surname [271-413] | 551 | **84.5%** |
| SALT Manufacturing | SHIPPINGPOINT [7-79] | 568 | **34.8%** |

### Business Question

> "Is collecting 568 samples for shipping points 7-79 worth a 35% reduction in prediction uncertainty?"

Now you can answer this:
- Cost of collecting 568 samples: $X
- Value of 35% less prediction uncertainty: $Y
- Decision: If Y > X, collect the data

---

## Action Checklist for Risk Managers

### Step 1: Run Hierarchical Decomposition

```
Input: Your trained model + data
Output:
  L1: Which FK tables contribute to uncertainty
  L2: Which columns within the top FK
  L3: Which value ranges within the top column
```

### Step 2: Identify Unknown Unknowns

Look for regions where:
- Model has LOW uncertainty (confident)
- But LOW sample count (hasn't seen much data)

These are your **unknown unknowns** - highest risk.

### Step 3: Quantify Intervention Effect

Run the data quantity experiment:
- How much does uncertainty decrease with 10%, 25%, 50%, 100% more data?
- Calculate the ROI of data collection

### Step 4: Prioritize Investments

Rank interventions by:
1. **Risk reduction**: How much uncertainty decreases
2. **Cost**: How hard is it to collect this data
3. **Business impact**: How important are predictions in this region

### Step 5: Monitor

After collecting data:
- Re-run decomposition
- Verify uncertainty decreased as expected
- Identify next highest-risk region

---

## Key Takeaways

1. **Low uncertainty ≠ Safe**: A model that has never seen a region will be confidently wrong

2. **Unknown unknowns are the real risk**: Regions where model is confident but has no data

3. **The framework converts unknown unknowns to known unknowns**: By identifying where the model SHOULD be uncertain but isn't

4. **Uncertainty is reducible**: Collecting targeted data reduces uncertainty by 35-85%

5. **The hierarchy matters**: Don't just ask "which table" (obvious), ask "which column, which values" (actionable)

---

## Example Output

```
═══════════════════════════════════════════════════════════════════
HIERARCHICAL ERROR PROPAGATION: SALT Manufacturing
═══════════════════════════════════════════════════════════════════

LEVEL 1: FK TABLE (everyone knows this)
  ITEM: 249% importance

LEVEL 2: COLUMN (which data to fix?)
  └─ SHIPPINGPOINT: 227% (this is 9x more important than other columns)

LEVEL 3: VALUE RANGE (where exactly?)
  └─ [7-79]: 92% importance, only 201 samples
     → ACTION: Collect more data for shipping points 7-79
     → EXPECTED: 35% uncertainty reduction with 568 samples

RISK ALERT:
  Shipping points 7-79 may be "unknown unknowns"
  - Low sample count (201)
  - Check if model is overconfident in this region
  - Priority: HIGH
═══════════════════════════════════════════════════════════════════
```

---

## Summary

> **"Which table matters?"** → Everyone knows this. Not useful.
>
> **"Which column and which values need more data, and how much will it help?"** → This is what risk managers need.
>
> **Unknown unknowns (confident but wrong) are the highest risk.** This framework finds them.

---

*Document created: 2025-12-09*
*Part of: Hierarchical Bayesian Intervention Analysis for Relational Data*
