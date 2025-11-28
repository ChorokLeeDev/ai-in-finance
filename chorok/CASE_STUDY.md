# Case Study: FK Uncertainty Attribution in Supply Chain (SALT)

**Date**: 2025-11-29
**Research Question**: Which FK relationships caused uncertainty during COVID-19?

---

## Executive Summary

Using Leave-One-Out (LOO) FK Uncertainty Attribution on the SALT dataset,
we identified which foreign key relationships contributed most to model
uncertainty during the COVID-19 distribution shift (Feb 2020 onset).

### Key Findings

| Task | Top FK by Change | Interpretation |
|------|-----------------|----------------|
| sales-group | HEADERINCOTERMS (+0.064) | Trade terms became more uncertain |
| sales-office | HEADERINCOTERMS (-0.040) | Trade terms stabilized office predictions |
| sales-payterms | SALESGROUP (+0.447) | Sales organization drove uncertainty |
| sales-shipcond | CUSTOMERPAYMENTTERMS (-0.011) | Payment terms became less informative |
| sales-incoterms | SALESGROUP (-0.379) | Sales group became more stable |
| item-plant | SOLDTOPARTY (-0.000) | Customer identity unchanged |
| item-shippoint | BILLTOPARTY (+0.214) | Billing party became more uncertain |
| item-incoterms | PAYERPARTY (-0.236) | Payer relationships stabilized |

---

## Business Context: SALT Dataset

### What is SALT?

SALT (Simulated Anonymized Logistics Transactions) is a supply chain dataset
from SAP that simulates B2B sales transactions. It contains:

- **Sales Documents**: Customer orders with pricing, quantities, dates
- **Foreign Keys**: Links to customers, products, organizational units

### COVID-19 Impact on Supply Chains

COVID-19 (Feb 2020 onset) disrupted global supply chains:
- Demand volatility
- Shipping delays and route changes
- Payment term renegotiations
- Trade term (Incoterms) modifications

---

## Interpreting FK Attribution Changes

### 1. HEADERINCOTERMS (Trade Terms)

**What it is**: International Commercial Terms defining delivery responsibilities

**Pre-COVID**: Standard trade terms (e.g., FOB, CIF) were predictive
**Post-COVID**: Trade terms became more variable as shipping routes changed

**Attribution Change**: +0.064 (became MORE uncertain for sales-group)

**Business Interpretation**:
> During COVID, companies renegotiated Incoterms to adapt to shipping
> disruptions. The historical relationship between Incoterms and sales
> outcomes broke down, increasing model uncertainty.

### 2. SALESGROUP (Sales Organization)

**What it is**: Internal sales team or regional organization

**Pre-COVID**: Sales group was a stable predictor of outcomes
**Post-COVID**: Regional lockdowns created divergent patterns

**Attribution Change**: +0.447 (sales-payterms), -0.379 (sales-incoterms)

**Business Interpretation**:
> Different sales regions experienced COVID impacts at different times.
> For payment terms, this increased uncertainty (hard to predict which
> region's patterns would apply). For trade terms, regional consistency
> actually helped predictions.

### 3. CUSTOMERPAYMENTTERMS (Payment Terms)

**What it is**: Agreed payment schedule (e.g., Net 30, Net 60)

**Pre-COVID**: Payment terms were stable and predictive
**Post-COVID**: Many customers renegotiated payment terms

**Attribution Change**: -0.011 (became slightly LESS uncertain)

**Business Interpretation**:
> Surprisingly, payment terms became slightly more stable. This could be
> because companies standardized on extended payment terms during COVID,
> reducing variability.

### 4. BILLTOPARTY / PAYERPARTY (Customer Entities)

**What it is**: The customer receiving the invoice / making payment

**Pre-COVID**: Customer identity was moderately predictive
**Post-COVID**: Some customers' behavior changed dramatically

**Attribution Changes**: BILLTOPARTY +0.214, PAYERPARTY -0.236

**Business Interpretation**:
> Mixed effects. For shipping point predictions, billing relationships
> became more uncertain (companies changed billing arrangements). For
> trade term predictions, payer relationships became more stable
> (existing payers maintained consistency).

---

## Why This Matters for ML in Production

### 1. Early Warning System

FK Attribution identifies WHICH data sources are driving uncertainty:

- If HEADERINCOTERMS attribution spikes → Check trade term data quality
- If SALESGROUP attribution spikes → Consider regional model stratification
- If CUSTOMERPAYMENTTERMS attribution changes → Validate payment term encoding

### 2. Feature Engineering Guidance

High-attribution FKs during distribution shifts suggest:

- These features need more robust representations
- Consider adding temporal context (recent vs. historical patterns)
- May need domain-specific normalization

### 3. Model Monitoring

Track FK attribution over time to detect:

- Gradual drift in specific relationships
- Sudden shifts indicating external events
- Recovery patterns after disruptions

---

## Comparison: FK Attribution vs. SHAP

| Scenario | SHAP Says | FK Attribution Says |
|----------|-----------|---------------------|
| HEADERINCOTERMS | Moderate importance | High uncertainty contribution |
| TRANSACTIONCURRENCY | High importance | Low uncertainty contribution |
| SALESGROUP | Variable importance | High shift sensitivity |

**Key Insight**: A feature can be IMPORTANT for predictions (high SHAP)
but NOT cause uncertainty (stable relationship). Conversely, a feature
with moderate SHAP importance might cause HIGH uncertainty if its
relationship to the target is unstable.

---

## Actionable Recommendations

### For Supply Chain ML Models

1. **Monitor Trade Terms**: HEADERINCOTERMS is a leading indicator of
   uncertainty during disruptions. Add monitoring dashboards.

2. **Regional Stratification**: SALESGROUP sensitivity suggests regional
   models may outperform global models during heterogeneous shocks.

3. **Customer Segmentation**: BILLTOPARTY/PAYERPARTY changes indicate
   customer-level model updates may be needed after major events.

4. **Feature Stability Analysis**: Before deployment, analyze FK attribution
   on historical distribution shifts to identify vulnerable features.

### For Research

1. **Temporal Attribution**: Track monthly FK attribution to create
   early warning systems for model degradation.

2. **Causal Analysis**: Combine FK attribution with domain knowledge
   to understand WHY certain relationships destabilize.

3. **Multi-Dataset Validation**: Confirm patterns on other supply chain
   datasets (requires proprietary data).

---

## Limitations

1. **Simulated Data**: SALT is a simulated dataset; real-world patterns
   may differ.

2. **Single Event**: COVID-19 is one distribution shift; other events
   (e.g., trade wars, natural disasters) may show different patterns.

3. **Correlation Not Causation**: FK attribution identifies statistical
   associations, not causal mechanisms.

4. **Sample Size**: Monthly analysis has limited samples per period,
   increasing variance in attribution estimates.

---

## Conclusion

FK Uncertainty Attribution reveals that **HEADERINCOTERMS** and **SALESGROUP**
were the primary FK relationships driving model uncertainty during COVID-19
in the SALT dataset. This suggests that trade logistics and regional sales
organization are the most sensitive aspects of supply chain ML models
to external disruptions.

**Practical Takeaway**: Monitor trade term and regional organization
features most closely for distribution shift detection.

---

**Last Updated**: 2025-11-29
