"""
SIMULATION vs OPTIMIZATION: A Framework for Actionable UQ
==========================================================

Two fundamentally different question types:

SIMULATION (What IF?):
- "What if I switch from Customer A to Customer B?"
- "What if lead time increases by 2 days?"
- "What if we use shipping method X instead of Y?"
→ Fixed scenario, evaluate uncertainty

OPTIMIZATION (What SHOULD?):
- "Which customer should I prioritize to minimize uncertainty?"
- "What is the optimal allocation across shipping points?"
- "What feature values minimize my prediction risk?"
→ Search for best decision

This experiment demonstrates both approaches on SAP SALT data.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import lightgbm as lgb
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from data_loader_salt import load_salt_data


def train_ensemble(X, y, n_models=10, base_seed=42):
    models = []
    for i in range(n_models):
        model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=base_seed + i, verbose=-1
        )
        model.fit(X, y)
        models.append(model)
    return models


def get_uncertainty(models, X):
    if len(X) == 0:
        return np.array([])
    preds = np.array([m.predict(X) for m in models])
    return preds.var(axis=0)


def get_entity_stats(models, X, y, entity_col):
    """Get uncertainty stats per entity."""
    uncertainties = get_uncertainty(models, X)
    data = pd.DataFrame({
        'entity': X[entity_col].values,
        'uncertainty': uncertainties
    })
    return data.groupby('entity').agg({
        'uncertainty': ['mean', 'count']
    }).reset_index()


# ============================================================================
# SIMULATION QUESTIONS
# ============================================================================

def simulation_q1_switch_entity(models, X, y, entity_col, from_entity, to_entity):
    """
    SIMULATION Q1: What if I switch all orders from entity A to entity B?

    Example: "What if I switch from Customer 86919 to Customer 29806?"
    """
    from_mask = X[entity_col] == from_entity
    to_mask = X[entity_col] == to_entity

    if from_mask.sum() == 0 or to_mask.sum() == 0:
        return None

    # Baseline: current portfolio
    baseline_unc = get_uncertainty(models, X).mean()

    # Simulation: Replace from_entity samples with to_entity characteristics
    X_sim = X.copy()
    to_samples = X[to_mask]

    for idx in X[from_mask].index:
        replacement_idx = np.random.choice(to_samples.index)
        X_sim.loc[idx] = X.loc[replacement_idx]

    simulated_unc = get_uncertainty(models, X_sim).mean()

    return {
        'question': f"What if I switch from entity {from_entity} to {to_entity}?",
        'baseline_uncertainty': baseline_unc,
        'simulated_uncertainty': simulated_unc,
        'change': (simulated_unc - baseline_unc) / baseline_unc * 100,
        'samples_affected': from_mask.sum()
    }


def simulation_q2_remove_entity(models, X, y, entity_col, entity):
    """
    SIMULATION Q2: What if I completely remove this entity from portfolio?

    Example: "What if I stop ordering from Customer 86919?"
    """
    entity_mask = X[entity_col] == entity

    if entity_mask.sum() == 0:
        return None

    # Baseline
    baseline_unc = get_uncertainty(models, X).mean()

    # Simulation: Remove entity entirely
    X_sim = X[~entity_mask]
    simulated_unc = get_uncertainty(models, X_sim).mean()

    return {
        'question': f"What if I remove entity {entity}?",
        'baseline_uncertainty': baseline_unc,
        'simulated_uncertainty': simulated_unc,
        'change': (simulated_unc - baseline_unc) / baseline_unc * 100,
        'samples_removed': entity_mask.sum()
    }


def simulation_q3_add_noise(models, X, col_to_fk, fk_name, noise_level=0.1):
    """
    SIMULATION Q3: What if data quality degrades for this FK?

    Example: "What if shipping point data becomes 10% noisier?"
    """
    baseline_unc = get_uncertainty(models, X).mean()

    fk_cols = [c for c, f in col_to_fk.items() if f == fk_name]
    X_sim = X.copy()

    for col in fk_cols:
        if col in X_sim.columns:
            std = X[col].std()
            noise = np.random.normal(0, std * noise_level, len(X))
            X_sim[col] = X[col] + noise

    simulated_unc = get_uncertainty(models, X_sim).mean()

    return {
        'question': f"What if {fk_name} data quality degrades by {noise_level*100:.0f}%?",
        'baseline_uncertainty': baseline_unc,
        'simulated_uncertainty': simulated_unc,
        'change': (simulated_unc - baseline_unc) / baseline_unc * 100,
        'features_affected': fk_cols
    }


def simulation_q4_mix_entities(models, X, y, entity_col, weights):
    """
    SIMULATION Q4: What if I rebalance my entity portfolio?

    Example: "What if I shift 30% from high-unc to low-unc entities?"
    weights: dict of {entity: proportion}
    """
    baseline_unc = get_uncertainty(models, X).mean()

    # Build simulated portfolio according to weights
    X_sim_list = []
    total_weight = sum(weights.values())

    for entity, weight in weights.items():
        entity_samples = X[X[entity_col] == entity]
        if len(entity_samples) > 0:
            n_samples = int(len(X) * (weight / total_weight))
            if n_samples > 0:
                sampled = entity_samples.sample(n=min(n_samples, len(entity_samples)),
                                                 replace=True, random_state=42)
                X_sim_list.append(sampled)

    if len(X_sim_list) == 0:
        return None

    X_sim = pd.concat(X_sim_list).reset_index(drop=True)
    simulated_unc = get_uncertainty(models, X_sim).mean()

    return {
        'question': f"What if I rebalance portfolio to {weights}?",
        'baseline_uncertainty': baseline_unc,
        'simulated_uncertainty': simulated_unc,
        'change': (simulated_unc - baseline_unc) / baseline_unc * 100,
        'new_portfolio_size': len(X_sim)
    }


# ============================================================================
# OPTIMIZATION QUESTIONS
# ============================================================================

def optimization_q1_best_single_entity(models, X, y, entity_col, min_samples=5):
    """
    OPTIMIZATION Q1: Which single entity minimizes uncertainty?

    Example: "Which customer should I prioritize?"
    """
    uncertainties = get_uncertainty(models, X)

    data = pd.DataFrame({
        'entity': X[entity_col].values,
        'uncertainty': uncertainties
    })

    entity_stats = data.groupby('entity').agg({
        'uncertainty': ['mean', 'std', 'count']
    }).reset_index()
    entity_stats.columns = ['entity', 'mean_unc', 'std_unc', 'count']
    entity_stats = entity_stats[entity_stats['count'] >= min_samples]

    best = entity_stats.nsmallest(1, 'mean_unc').iloc[0]
    worst = entity_stats.nlargest(1, 'mean_unc').iloc[0]

    return {
        'question': "Which entity minimizes uncertainty?",
        'best_entity': int(best['entity']),
        'best_uncertainty': best['mean_unc'],
        'best_count': int(best['count']),
        'worst_entity': int(worst['entity']),
        'worst_uncertainty': worst['mean_unc'],
        'improvement_potential': (worst['mean_unc'] - best['mean_unc']) / worst['mean_unc'] * 100
    }


def optimization_q2_optimal_allocation(models, X, y, entity_col, top_k=5, min_samples=10):
    """
    OPTIMIZATION Q2: What is the optimal allocation across top entities?

    Example: "How should I split orders across top 5 customers?"
    """
    uncertainties = get_uncertainty(models, X)

    data = pd.DataFrame({
        'entity': X[entity_col].values,
        'uncertainty': uncertainties
    })

    entity_stats = data.groupby('entity').agg({
        'uncertainty': ['mean', 'count']
    }).reset_index()
    entity_stats.columns = ['entity', 'mean_unc', 'count']
    entity_stats = entity_stats[entity_stats['count'] >= min_samples]

    # Get top_k lowest uncertainty entities
    best_entities = entity_stats.nsmallest(top_k, 'mean_unc')

    # Optimal allocation: inversely proportional to uncertainty
    best_entities['inv_unc'] = 1 / (best_entities['mean_unc'] + 1e-10)
    best_entities['optimal_weight'] = best_entities['inv_unc'] / best_entities['inv_unc'].sum()

    allocation = []
    for _, row in best_entities.iterrows():
        allocation.append({
            'entity': int(row['entity']),
            'uncertainty': row['mean_unc'],
            'optimal_weight': row['optimal_weight'],
            'current_count': int(row['count'])
        })

    return {
        'question': f"What is the optimal allocation across top {top_k} entities?",
        'allocation': allocation,
        'expected_portfolio_uncertainty': sum(a['uncertainty'] * a['optimal_weight'] for a in allocation)
    }


def optimization_q3_minimize_given_constraint(models, X, y, entity_col, must_include=None, min_volume=0.1):
    """
    OPTIMIZATION Q3: Minimize uncertainty given constraints

    Example: "Minimize uncertainty, but must include Customer A with at least 10%"
    """
    uncertainties = get_uncertainty(models, X)

    data = pd.DataFrame({
        'entity': X[entity_col].values,
        'uncertainty': uncertainties
    })

    entity_stats = data.groupby('entity').agg({
        'uncertainty': 'mean'
    }).reset_index()
    entity_stats.columns = ['entity', 'mean_unc']

    # Simple constraint: must_include gets at least min_volume
    if must_include is None:
        must_include = []

    # Remaining weight goes to lowest uncertainty entities
    remaining_weight = 1.0 - len(must_include) * min_volume

    # Sort by uncertainty
    entity_stats = entity_stats.sort_values('mean_unc')

    allocation = {}

    # First, allocate to must_include entities
    for entity in must_include:
        if entity in entity_stats['entity'].values:
            allocation[int(entity)] = min_volume

    # Then allocate remaining to best entities
    for _, row in entity_stats.iterrows():
        if int(row['entity']) not in allocation:
            if remaining_weight > 0:
                alloc = min(remaining_weight, 0.3)  # cap at 30% per entity
                allocation[int(row['entity'])] = alloc
                remaining_weight -= alloc
            if remaining_weight <= 0:
                break

    # Normalize
    total = sum(allocation.values())
    allocation = {k: v/total for k, v in allocation.items()}

    return {
        'question': f"Minimize uncertainty given constraint: must include {must_include}",
        'allocation': allocation,
        'constrained_entities': must_include,
        'n_entities_used': len(allocation)
    }


def optimization_q4_threshold_selection(models, X, y, entity_col, uncertainty_threshold):
    """
    OPTIMIZATION Q4: Which entities should I use if I have an uncertainty threshold?

    Example: "Which customers keep uncertainty below 0.01?"
    """
    uncertainties = get_uncertainty(models, X)

    data = pd.DataFrame({
        'entity': X[entity_col].values,
        'uncertainty': uncertainties
    })

    entity_stats = data.groupby('entity').agg({
        'uncertainty': ['mean', 'count']
    }).reset_index()
    entity_stats.columns = ['entity', 'mean_unc', 'count']

    # Filter by threshold
    qualifying = entity_stats[entity_stats['mean_unc'] <= uncertainty_threshold]

    return {
        'question': f"Which entities have uncertainty <= {uncertainty_threshold}?",
        'qualifying_entities': qualifying['entity'].tolist(),
        'n_qualifying': len(qualifying),
        'n_total': len(entity_stats),
        'fraction_qualifying': len(qualifying) / len(entity_stats),
        'total_samples_from_qualifying': qualifying['count'].sum()
    }


def run_full_demo():
    print("=" * 80)
    print("SIMULATION vs OPTIMIZATION: Business Decision Framework")
    print("=" * 80)

    # Load data
    print("\nLoading SAP SALT data...")
    X, y, feature_cols, col_to_fk = load_salt_data(sample_size=20000)

    entity_col = 'SOLDTOPARTY'
    print(f"Entity column: {entity_col}")
    print(f"Unique entities: {X[entity_col].nunique()}")

    # Train ensemble
    print("\nTraining ensemble...")
    models = train_ensemble(X, y, n_models=10)

    # Get entity stats for reference
    entity_stats = get_entity_stats(models, X, y, entity_col)
    entity_stats.columns = ['entity', 'mean_unc', 'count']
    entity_stats = entity_stats[entity_stats['count'] >= 5]

    high_unc = entity_stats.nlargest(3, 'mean_unc')
    low_unc = entity_stats.nsmallest(3, 'mean_unc')

    print("\nReference - High uncertainty entities:")
    for _, row in high_unc.iterrows():
        print(f"  Entity {int(row['entity'])}: unc={row['mean_unc']:.6f}")

    print("\nReference - Low uncertainty entities:")
    for _, row in low_unc.iterrows():
        print(f"  Entity {int(row['entity'])}: unc={row['mean_unc']:.6f}")

    # =========================================================================
    print("\n" + "=" * 80)
    print("SIMULATION QUESTIONS (What IF?)")
    print("=" * 80)

    print("\n" + "-" * 60)
    print("SIMULATION Q1: Entity Switch")
    result = simulation_q1_switch_entity(
        models, X, y, entity_col,
        from_entity=high_unc.iloc[0]['entity'],
        to_entity=low_unc.iloc[0]['entity']
    )
    if result:
        print(f"\nQuestion: {result['question']}")
        print(f"Baseline uncertainty: {result['baseline_uncertainty']:.6f}")
        print(f"Simulated uncertainty: {result['simulated_uncertainty']:.6f}")
        print(f"Change: {result['change']:+.2f}% ({'reduced' if result['change'] < 0 else 'increased'})")
        print(f"Samples affected: {result['samples_affected']}")

    print("\n" + "-" * 60)
    print("SIMULATION Q2: Remove High-Uncertainty Entity")
    result = simulation_q2_remove_entity(
        models, X, y, entity_col,
        entity=high_unc.iloc[0]['entity']
    )
    if result:
        print(f"\nQuestion: {result['question']}")
        print(f"Baseline uncertainty: {result['baseline_uncertainty']:.6f}")
        print(f"Simulated uncertainty: {result['simulated_uncertainty']:.6f}")
        print(f"Change: {result['change']:+.2f}%")

    print("\n" + "-" * 60)
    print("SIMULATION Q3: Data Quality Degradation")
    result = simulation_q3_add_noise(models, X, col_to_fk, 'ITEM', noise_level=0.1)
    print(f"\nQuestion: {result['question']}")
    print(f"Baseline uncertainty: {result['baseline_uncertainty']:.6f}")
    print(f"Simulated uncertainty: {result['simulated_uncertainty']:.6f}")
    print(f"Change: {result['change']:+.2f}%")
    print(f"Features affected: {result['features_affected']}")

    print("\n" + "-" * 60)
    print("SIMULATION Q4: Portfolio Rebalancing")
    weights = {
        low_unc.iloc[0]['entity']: 0.5,
        low_unc.iloc[1]['entity']: 0.3,
        low_unc.iloc[2]['entity']: 0.2
    }
    result = simulation_q4_mix_entities(models, X, y, entity_col, weights)
    if result:
        print(f"\nQuestion: {result['question']}")
        print(f"Baseline uncertainty: {result['baseline_uncertainty']:.6f}")
        print(f"Simulated uncertainty: {result['simulated_uncertainty']:.6f}")
        print(f"Change: {result['change']:+.2f}%")

    # =========================================================================
    print("\n" + "=" * 80)
    print("OPTIMIZATION QUESTIONS (What SHOULD?)")
    print("=" * 80)

    print("\n" + "-" * 60)
    print("OPTIMIZATION Q1: Best Single Entity")
    result = optimization_q1_best_single_entity(models, X, y, entity_col)
    print(f"\nQuestion: {result['question']}")
    print(f"Best entity: {result['best_entity']} (unc={result['best_uncertainty']:.6f}, n={result['best_count']})")
    print(f"Worst entity: {result['worst_entity']} (unc={result['worst_uncertainty']:.6f})")
    print(f"Improvement potential: {result['improvement_potential']:.1f}%")

    print("\n" + "-" * 60)
    print("OPTIMIZATION Q2: Optimal Allocation")
    result = optimization_q2_optimal_allocation(models, X, y, entity_col, top_k=5)
    print(f"\nQuestion: {result['question']}")
    print("\nOptimal allocation:")
    for alloc in result['allocation']:
        print(f"  Entity {alloc['entity']}: {alloc['optimal_weight']:.1%} "
              f"(unc={alloc['uncertainty']:.6f})")
    print(f"\nExpected portfolio uncertainty: {result['expected_portfolio_uncertainty']:.6f}")

    print("\n" + "-" * 60)
    print("OPTIMIZATION Q3: Constrained Optimization")
    must_include = [high_unc.iloc[0]['entity']]
    result = optimization_q3_minimize_given_constraint(
        models, X, y, entity_col,
        must_include=must_include,
        min_volume=0.15
    )
    print(f"\nQuestion: {result['question']}")
    print(f"Number of entities in solution: {result['n_entities_used']}")
    print("\nAllocation:")
    for entity, weight in sorted(result['allocation'].items(), key=lambda x: -x[1])[:5]:
        print(f"  Entity {entity}: {weight:.1%}")

    print("\n" + "-" * 60)
    print("OPTIMIZATION Q4: Threshold Selection")
    threshold = entity_stats['mean_unc'].median()
    result = optimization_q4_threshold_selection(models, X, y, entity_col, threshold)
    print(f"\nQuestion: {result['question']}")
    print(f"Qualifying entities: {result['n_qualifying']}/{result['n_total']} ({result['fraction_qualifying']:.1%})")
    print(f"Total samples from qualifying: {result['total_samples_from_qualifying']}")

    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: SIMULATION vs OPTIMIZATION")
    print("=" * 80)

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        SIMULATION (What IF?)                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Q1: What if I switch from Entity A to Entity B?                              ║
║     → Evaluate uncertainty of a specific scenario                            ║
║                                                                              ║
║ Q2: What if I remove this entity entirely?                                   ║
║     → Impact assessment of entity removal                                    ║
║                                                                              ║
║ Q3: What if data quality degrades?                                           ║
║     → Sensitivity to noise/errors                                            ║
║                                                                              ║
║ Q4: What if I rebalance my portfolio?                                        ║
║     → Evaluate a proposed allocation                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║                       OPTIMIZATION (What SHOULD?)                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Q1: Which entity minimizes uncertainty?                                      ║
║     → Find the single best entity                                            ║
║                                                                              ║
║ Q2: What is the optimal allocation?                                          ║
║     → Find weights that minimize portfolio uncertainty                       ║
║                                                                              ║
║ Q3: How do I minimize given constraints?                                     ║
║     → Optimize with business constraints                                     ║
║                                                                              ║
║ Q4: Which entities meet my threshold?                                        ║
║     → Filter entities by uncertainty requirement                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

KEY DIFFERENCE:
- SIMULATION: User provides scenario → System evaluates uncertainty
- OPTIMIZATION: User provides objective → System finds best scenario
""")

    print("=" * 80)


if __name__ == "__main__":
    run_full_demo()
