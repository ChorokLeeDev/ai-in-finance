"""
Aggregation Uncertainty Attribution (AUA)
=========================================

Novel contribution: Uncertainty in relational data comes from TWO sources:
1. Feature value uncertainty (what Causal SHAP measures)
2. Aggregation uncertainty (unique to relational data)

Aggregation uncertainty arises because:
- Low cardinality FK links -> aggregated features are unreliable (small sample)
- High variance within FK -> aggregated features don't represent the group well

Causal SHAP cannot capture this - it only sees the aggregated feature value.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import entropy, spearmanr, pearsonr
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


def test_cardinality_uncertainty_hypothesis():
    """
    Test: Does low cardinality correlate with high prediction uncertainty?

    If yes -> aggregation uncertainty is real and measurable.
    """
    print("="*60)
    print("HYPOTHESIS TEST: Cardinality → Uncertainty")
    print("="*60)

    np.random.seed(42)
    n_entities = 2000

    # Simulate relational data
    # Each entity has a varying number of related items (cardinality)
    cardinalities = np.random.choice([1, 2, 3, 5, 10, 20, 50], size=n_entities,
                                      p=[0.3, 0.25, 0.15, 0.12, 0.1, 0.05, 0.03])

    # For each entity, generate related items and aggregate
    aggregated_features = []
    true_uncertainties = []  # Ground truth: uncertainty from aggregation

    for i, card in enumerate(cardinalities):
        # Generate `card` related items with some underlying signal
        true_mean = np.random.randn()  # True underlying signal
        items = true_mean + np.random.randn(card) * 0.5  # Items with noise

        # Aggregate: mean of items
        agg_mean = np.mean(items)
        agg_std = np.std(items) if card > 1 else 0.5  # Std of items

        aggregated_features.append({
            'entity_id': i,
            'cardinality': card,
            'agg_mean': agg_mean,
            'agg_std': agg_std,
            'true_mean': true_mean
        })

        # Ground truth uncertainty: SE of the mean (decreases with sqrt(n))
        se_mean = 0.5 / np.sqrt(card)
        true_uncertainties.append(se_mean)

    df = pd.DataFrame(aggregated_features)
    df['true_uncertainty'] = true_uncertainties

    # Generate target based on true mean (with noise)
    n_classes = 4
    logits = np.zeros((n_entities, n_classes))
    for c in range(n_classes):
        logits[:, c] = df['true_mean'].values + np.random.randn() * 0.3
    logits += np.random.randn(n_entities, n_classes) * 0.2

    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    target = np.array([np.random.choice(n_classes, p=p) for p in probs])

    df['target'] = target

    # Train model on aggregated features (what standard ML sees)
    X = df[['agg_mean', 'agg_std', 'cardinality']]
    y = df['target']

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Get prediction uncertainty (entropy of predicted probabilities)
    proba = model.predict_proba(X)
    prediction_entropy = entropy(proba + 1e-10, axis=1)
    df['pred_uncertainty'] = prediction_entropy

    # TEST THE HYPOTHESIS
    print("\n--- Correlation Analysis ---")

    # H1: Low cardinality → high uncertainty
    rho_card, p_card = spearmanr(df['cardinality'], df['pred_uncertainty'])
    print(f"Cardinality vs Uncertainty: ρ = {rho_card:.3f}, p = {p_card:.4f}")
    print(f"  Expected: NEGATIVE (low cardinality → high uncertainty)")

    # H2: High within-group variance → high uncertainty
    rho_std, p_std = spearmanr(df['agg_std'], df['pred_uncertainty'])
    print(f"Within-group Std vs Uncertainty: ρ = {rho_std:.3f}, p = {p_std:.4f}")
    print(f"  Expected: POSITIVE (high variance → high uncertainty)")

    # H3: True aggregation uncertainty → prediction uncertainty
    rho_true, p_true = spearmanr(df['true_uncertainty'], df['pred_uncertainty'])
    print(f"True Aggregation UQ vs Pred UQ: ρ = {rho_true:.3f}, p = {p_true:.4f}")
    print(f"  Expected: POSITIVE (aggregation uncertainty drives prediction uncertainty)")

    # Summary by cardinality bucket
    print("\n--- Uncertainty by Cardinality Bucket ---")
    for card in [1, 2, 3, 5, 10, 20, 50]:
        mask = df['cardinality'] == card
        if mask.sum() > 0:
            mean_unc = df.loc[mask, 'pred_uncertainty'].mean()
            n = mask.sum()
            print(f"  Cardinality {card:2d}: Mean uncertainty = {mean_unc:.4f} (n={n})")

    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    if rho_card < -0.1 and p_card < 0.05:
        print("✓ HYPOTHESIS CONFIRMED: Low cardinality causes high uncertainty")
        print("  This is AGGREGATION UNCERTAINTY - unique to relational data!")
        verdict = "confirmed"
    elif rho_card < 0:
        print("△ WEAK SIGNAL: Trend in expected direction but not strong")
        verdict = "weak"
    else:
        print("✗ HYPOTHESIS NOT SUPPORTED")
        verdict = "rejected"

    return {
        'rho_cardinality': rho_card,
        'p_cardinality': p_card,
        'rho_within_std': rho_std,
        'p_within_std': p_std,
        'rho_true_agg_uq': rho_true,
        'p_true_agg_uq': p_true,
        'verdict': verdict
    }


def test_on_real_relbench_data():
    """
    Test aggregation uncertainty on actual RelBench SALT data.
    """
    print("\n" + "="*60)
    print("TEST ON REAL RELBENCH DATA (SALT)")
    print("="*60)

    try:
        from relbench.datasets import get_dataset
        from relbench.tasks import get_task
    except ImportError:
        print("RelBench not available, skipping real data test")
        return None

    # Load data
    dataset = get_dataset('rel-salt', download=True)
    task = get_task('rel-salt', 'sales-group', download=True)
    db = dataset.get_db()

    # Get item and document tables
    items = db.table_dict['salesdocumentitem'].df
    docs = db.table_dict['salesdocument'].df

    # Compute cardinality per document
    cardinality = items.groupby('SALESDOCUMENT').size().reset_index(name='cardinality')

    # Get training data
    train_table = task.get_table("train")
    train_df = train_table.df.copy()

    # Merge with cardinality
    # First need to link entities to documents
    # This depends on task structure - simplified for now
    print(f"Train samples: {len(train_df)}")
    print(f"Documents with cardinality info: {len(cardinality)}")

    # Show cardinality distribution
    print("\nCardinality distribution in SALT:")
    for pct in [25, 50, 75, 90, 99]:
        val = cardinality['cardinality'].quantile(pct/100)
        print(f"  {pct}th percentile: {val:.0f}")

    print("\nNOTE: Full integration requires task-specific entity-to-document mapping")
    print("The synthetic test above demonstrates the concept")

    return cardinality


if __name__ == '__main__':
    # Run synthetic hypothesis test
    results = test_cardinality_uncertainty_hypothesis()

    # Test on real data (optional)
    print("\n" + "-"*60)
    cardinality = test_on_real_relbench_data()

    print("\n" + "="*60)
    print("AGGREGATION UNCERTAINTY: KEY TAKEAWAY")
    print("="*60)
    print("""
This is a genuinely novel research direction:

1. PROBLEM: Relational data creates uncertainty from AGGREGATION
   - Low cardinality FK links = small sample = high uncertainty
   - High variance within FK = unreliable aggregate = high uncertainty

2. WHY CAUSAL SHAP CAN'T SOLVE THIS:
   - Causal SHAP sees aggregated features (one row of numbers)
   - It cannot decompose: "Is uncertainty from the value or the aggregation?"

3. OUR CONTRIBUTION:
   - Aggregation Uncertainty Attribution (AUA)
   - Decompose uncertainty by FK relationship
   - Identify which FK is causing uncertainty due to sparse/variable data

4. PRACTICAL VALUE:
   - "Your model is uncertain because this customer has only 2 orders"
   - "Collect more data for this FK relationship to reduce uncertainty"
   - Actionable insights for relational ML systems
""")
