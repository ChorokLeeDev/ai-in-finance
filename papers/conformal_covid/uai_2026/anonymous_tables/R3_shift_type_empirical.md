# Table R3: Empirical Shift Type Characterization

**Question**: What type of distribution shift occurs in SALT tasks?

## Methodology

- **Covariate shift**: KS test per feature, Bonferroni-corrected (α=0.05)
- **Label shift**: Jensen-Shannon divergence between train/test label distributions
- **Concept shift**: Accuracy drop from holdout to post-shift test set

## Results Summary

|      Task       | Cov KS | %Sig | Label JS | Acc Drop |  Classification |
|:---------------:|:------:|:----:|:--------:|:--------:|:---------------:|
| sales-shipcond  |  0.50  | 50%  |  0.083   |  +11.9%  |  Cov + Concept  |
| sales-group     |  0.50  | 50%  |  0.330   |   +1.0%  | Cov (dominant)  |
| sales-payterms  |  0.50  | 50%  |  0.078   |   +4.0%  | Cov (dominant)  |
| item-plant      |  0.86  | 86%  |  0.144   |   -8.7%  | Cov (dominant)  |
| item-shippoint  |  0.86  | 86%  |  0.152   |  +13.1%  |  Cov + Concept  |
| sales-incoterms |  0.50  | 50%  |  0.085   |   +7.6%  |  Cov + Concept  |
| item-incoterms  |  0.86  | 86%  |  0.140   |  +63.4%  |  Cov + Concept  |
| sales-office    |  0.50  | 50%  |  0.026   |   -0.2%  | Cov (dominant)  |

## Key Findings

1. **Covariate shift confirmed in ALL tasks**: 50-86% of features show significant KS test

2. **Concept shift varies**:
   - High (>7% acc drop): sales-shipcond, item-shippoint, sales-incoterms, item-incoterms
   - Low/none: sales-group, sales-payterms, item-plant, sales-office

3. **Robust task (sales-office)**: Covariate shift BUT no concept shift → maintained coverage

4. **Catastrophic failures require BOTH**: Covariate AND concept shift

## Per-Feature Covariate Shift Details

### Sales-level tasks (6 features)

|        Feature        | Shifted |  KS  |
|:---------------------:|:-------:|:----:|
| SALESDOCUMENTTYPE     |   No    | 0.0  |
| SALESORGANIZATION     | **Yes** | 1.0  |
| DISTRIBUTIONCHANNEL   | **Yes** | 1.0  |
| ORGANIZATIONDIVISION  | **Yes** | 1.0  |
| BILLINGCOMPANYCODE    |   No    | 0.0  |
| TRANSACTIONCURRENCY   |   No    | 0.0  |

### Item-level tasks (7 features)

|          Feature          | Shifted |  KS  |
|:-------------------------:|:-------:|:----:|
| SALESDOCUMENTITEM         | **Yes** | 1.0  |
| SALESDOCUMENTITEMCATEGORY |   No    | 0.0  |
| PRODUCT                   | **Yes** | 1.0  |
| SOLDTOPARTY               | **Yes** | 1.0  |
| SHIPTOPARTY               | **Yes** | 1.0  |
| BILLTOPARTY               | **Yes** | 1.0  |
| PAYERPARTY                | **Yes** | 1.0  |
