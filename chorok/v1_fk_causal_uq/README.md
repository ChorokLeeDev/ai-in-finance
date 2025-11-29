# V1: FK-Causal-UQ (DEPRECATED)

**Status**: DEPRECATED (2025-11-29)
**Outcome**: Negative result - no advantage over Causal SHAP

---

## Hypothesis

FK structure can be used as causal prior for interventional uncertainty attribution.

## Method

1. FK relationships → DAG structure
2. Backdoor adjustment using FK DAG
3. Interventional uncertainty attribution (do-calculus)

## Result

| Method | ρ with Ground Truth |
|--------|---------------------|
| Causal SHAP | 0.900 |
| FK-Causal-UQ (Ours) | 0.900 |

**Verdict**: Identical performance. FK structure provides NO advantage.

## Why It Failed

1. Both methods are permutation-based
2. DAG knowledge doesn't change permutation results significantly
3. Causal SHAP (Heskes 2020) already solves this problem

## Files

| File | Description |
|------|-------------|
| `fk_causal_uq.py` | Main implementation |
| `compare_causal_shap.py` | Comparison with Causal SHAP |
| `synthetic_causal_data.py` | Synthetic validation |
| `semi_synthetic_validation.py` | Semi-synthetic validation |
| `THEORY.md` | Formal theory (correct but not useful) |
| `DEPRECATED.md` | Deprecation notes |

## Lessons Learned

1. "Causal" label doesn't automatically mean better
2. Need to compare with existing causal methods before claiming novelty
3. Permutation ≈ Permutation regardless of DAG knowledge

---

*Archived: 2025-11-29*
