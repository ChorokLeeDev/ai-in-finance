"""
Data loader for rel-trial (Clinical Trials) dataset.
FK structure represents error propagation in clinical trial outcomes.

Study → Sponsor (funding source)
Study → Facility (where trial runs)
Study → Condition (disease being studied)
Study → Intervention (drug/treatment)

Task: study-adverse (predict number of adverse events)
"""

import pandas as pd
import numpy as np
from relbench.datasets import get_dataset
from relbench.tasks import get_task

def load_trial_data(sample_size=3000, task_name='study-adverse'):
    """
    Load rel-trial data with FK structure for attribution analysis.

    Returns:
        X: Feature DataFrame
        y: Target array
        entity_col: Main entity column name
        col_to_fk: Mapping from column to FK group
    """
    print(f"Loading rel-trial dataset (task: {task_name})...")

    ds = get_dataset("rel-trial", download=True)
    db = ds.get_db()
    task = get_task("rel-trial", task_name, download=True)

    train_table = task.get_table("train")
    train_df = train_table.df.copy()

    # Main entity is study (nct_id)
    entity_col = 'nct_id'
    target_col = task.target_col

    # Get study info
    studies = db.table_dict['studies'].df

    # Get sponsor info
    sponsors_studies = db.table_dict['sponsors_studies'].df
    sponsors = db.table_dict['sponsors'].df

    # Get facility info
    facilities_studies = db.table_dict['facilities_studies'].df
    facilities = db.table_dict['facilities'].df

    # Get condition info
    conditions_studies = db.table_dict['conditions_studies'].df
    conditions = db.table_dict['conditions'].df

    # Get intervention info
    interventions_studies = db.table_dict['interventions_studies'].df
    interventions = db.table_dict['interventions'].df

    # Get design info
    designs = db.table_dict['designs'].df

    # Get eligibility info
    eligibilities = db.table_dict['eligibilities'].df

    # Merge train with studies
    df = train_df[[entity_col, target_col]].merge(
        studies[[
            'nct_id', 'study_type', 'phase', 'enrollment',
            'number_of_arms', 'number_of_groups', 'has_dmc',
            'is_fda_regulated_drug', 'is_fda_regulated_device'
        ]],
        on='nct_id',
        how='left'
    )

    # Add lead sponsor info (first sponsor per study)
    lead_sponsors = sponsors_studies[sponsors_studies['lead_or_collaborator'] == 'lead'].drop_duplicates('nct_id')
    lead_sponsors = lead_sponsors.merge(sponsors, on='sponsor_id', how='left')
    df = df.merge(
        lead_sponsors[['nct_id', 'sponsor_id', 'agency_class']].rename(
            columns={'sponsor_id': 'lead_sponsor_id', 'agency_class': 'sponsor_agency_class'}
        ),
        on='nct_id',
        how='left'
    )

    # Add facility count and country info
    facility_counts = facilities_studies.groupby('nct_id').size().reset_index(name='num_facilities')
    df = df.merge(facility_counts, on='nct_id', how='left')
    df['num_facilities'] = df['num_facilities'].fillna(0)

    # Get primary facility country
    primary_facility = facilities_studies.drop_duplicates('nct_id').merge(
        facilities[['facility_id', 'country']], on='facility_id', how='left'
    )
    df = df.merge(
        primary_facility[['nct_id', 'country']].rename(columns={'country': 'primary_country'}),
        on='nct_id',
        how='left'
    )

    # Add condition count
    condition_counts = conditions_studies.groupby('nct_id').size().reset_index(name='num_conditions')
    df = df.merge(condition_counts, on='nct_id', how='left')
    df['num_conditions'] = df['num_conditions'].fillna(0)

    # Get primary condition
    primary_condition = conditions_studies.drop_duplicates('nct_id').merge(
        conditions, on='condition_id', how='left'
    )
    df = df.merge(
        primary_condition[['nct_id', 'condition_id']].rename(columns={'condition_id': 'primary_condition_id'}),
        on='nct_id',
        how='left'
    )

    # Add intervention count
    intervention_counts = interventions_studies.groupby('nct_id').size().reset_index(name='num_interventions')
    df = df.merge(intervention_counts, on='nct_id', how='left')
    df['num_interventions'] = df['num_interventions'].fillna(0)

    # Get primary intervention
    primary_intervention = interventions_studies.drop_duplicates('nct_id').merge(
        interventions, on='intervention_id', how='left'
    )
    df = df.merge(
        primary_intervention[['nct_id', 'intervention_id']].rename(columns={'intervention_id': 'primary_intervention_id'}),
        on='nct_id',
        how='left'
    )

    # Add design info
    df = df.merge(
        designs[[
            'nct_id', 'allocation', 'intervention_model',
            'primary_purpose', 'masking'
        ]].drop_duplicates('nct_id'),
        on='nct_id',
        how='left'
    )

    # Add eligibility info
    df = df.merge(
        eligibilities[[
            'nct_id', 'gender', 'minimum_age', 'maximum_age',
            'healthy_volunteers', 'adult', 'child', 'older_adult'
        ]].drop_duplicates('nct_id'),
        on='nct_id',
        how='left'
    )

    # Sample
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    print(f"  Samples: {len(df)}")

    # Prepare features
    y = df[target_col].values

    # Feature columns (excluding entity_col and target)
    feature_cols = [c for c in df.columns if c not in [entity_col, target_col]]

    # Encode categoricals
    X = df[feature_cols].copy()

    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
            X[col] = X[col].astype('category').cat.codes
        elif X[col].dtype == 'bool':
            X[col] = X[col].astype(int)

    X = X.fillna(-1)

    # Define FK grouping
    # Error propagation structure:
    # STUDY (design) → SPONSOR (funding) → FACILITY (location) → CONDITION (disease) → INTERVENTION (drug)

    col_to_fk = {
        # STUDY group (study-level attributes)
        'study_type': 'STUDY',
        'phase': 'STUDY',
        'enrollment': 'STUDY',
        'number_of_arms': 'STUDY',
        'number_of_groups': 'STUDY',
        'has_dmc': 'STUDY',
        'is_fda_regulated_drug': 'STUDY',
        'is_fda_regulated_device': 'STUDY',
        'allocation': 'STUDY',
        'intervention_model': 'STUDY',
        'primary_purpose': 'STUDY',
        'masking': 'STUDY',

        # SPONSOR group (who funds)
        'lead_sponsor_id': 'SPONSOR',
        'sponsor_agency_class': 'SPONSOR',

        # FACILITY group (where it runs)
        'num_facilities': 'FACILITY',
        'primary_country': 'FACILITY',

        # CONDITION group (what disease)
        'num_conditions': 'CONDITION',
        'primary_condition_id': 'CONDITION',

        # INTERVENTION group (what drug/treatment)
        'num_interventions': 'INTERVENTION',
        'primary_intervention_id': 'INTERVENTION',

        # ELIGIBILITY group (who can participate)
        'gender': 'ELIGIBILITY',
        'minimum_age': 'ELIGIBILITY',
        'maximum_age': 'ELIGIBILITY',
        'healthy_volunteers': 'ELIGIBILITY',
        'adult': 'ELIGIBILITY',
        'child': 'ELIGIBILITY',
        'older_adult': 'ELIGIBILITY',
    }

    # Filter to only include columns that exist
    col_to_fk = {k: v for k, v in col_to_fk.items() if k in X.columns}

    print(f"  Features: {len(X.columns)}")
    print(f"  FK groups: {len(set(col_to_fk.values()))}")
    print(f"  FK mapping: {set(col_to_fk.values())}")

    return X, y, entity_col, col_to_fk


if __name__ == "__main__":
    X, y, entity_col, col_to_fk = load_trial_data(sample_size=3000)
    print(f"\nLoaded: X={X.shape}, y={len(y)}")
    print(f"Columns: {list(X.columns)}")
    print(f"FK groups: {set(col_to_fk.values())}")
