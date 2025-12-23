"""
FK Classification Configuration for Causal FK Hypothesis Validation

This file contains domain-expert classifications of FKs as causal vs correlational
for each dataset and task in the RelBench benchmark.

Classification Criteria:
- Causal FK: FK points to data that directly determines the target
- Correlational FK: FK points to data that is correlated but not causally linked
"""

# FK Classifications by Dataset
# Format: {table_name: {fk_column: {"type": "causal"|"correlational", "reason": str}}}

F1_FK_CLASSIFICATION = {
    "results": {
        "raceId": {
            "type": "causal",
            "reason": "Race context directly determines outcome"
        },
        "driverId": {
            "type": "causal",
            "reason": "Driver identity directly affects performance"
        },
        "constructorId": {
            "type": "causal",
            "reason": "Team/car directly affects performance"
        }
    },
    "qualifying": {
        "raceId": {
            "type": "correlational",
            "reason": "Qualifying race context correlated but doesn't cause race result"
        },
        "driverId": {
            "type": "correlational",
            "reason": "Qualifying driver info is correlational (same driver, different context)"
        },
        "constructorId": {
            "type": "correlational",
            "reason": "Qualifying team info is correlational"
        }
    },
    "standings": {
        "raceId": {
            "type": "correlational",
            "reason": "Historical standings don't cause current race result"
        },
        "driverId": {
            "type": "correlational",
            "reason": "Historical driver standings are correlational"
        }
    }
}

SALT_FK_CLASSIFICATION = {
    "salesdocumentitem": {
        "SALESDOCUMENT": {
            "type": "causal",
            "reason": "Order header directly determines item processing"
        },
        "SOLDTOPARTY": {
            "type": "causal",
            "reason": "Customer identity directly affects plant/shipping decisions"
        },
        "SHIPTOPARTY": {
            "type": "correlational",
            "reason": "Ship-to party is determined after plant decision"
        },
        "BILLTOPARTY": {
            "type": "correlational",
            "reason": "Billing party doesn't affect plant assignment"
        },
        "PAYERPARTY": {
            "type": "correlational",
            "reason": "Payer doesn't affect plant assignment"
        }
    },
    "customer": {
        "ADDRESSID": {
            "type": "causal",
            "reason": "Address/location directly affects plant routing"
        }
    }
}

TRIAL_FK_CLASSIFICATION = {
    "outcomes": {
        "nct_id": {
            "type": "causal",
            "reason": "Study characteristics directly determine outcomes"
        }
    },
    "interventions_studies": {
        "nct_id": {
            "type": "causal",
            "reason": "Study context for intervention"
        },
        "intervention_id": {
            "type": "causal",
            "reason": "Drug/treatment directly causes outcome"
        }
    },
    "conditions_studies": {
        "nct_id": {
            "type": "causal",
            "reason": "Study context for condition"
        },
        "condition_id": {
            "type": "causal",
            "reason": "Disease type directly affects success rate"
        }
    },
    "facilities_studies": {
        "nct_id": {
            "type": "correlational",
            "reason": "Study context (correlational for facility)"
        },
        "facility_id": {
            "type": "correlational",
            "reason": "Hospital location doesn't cause drug efficacy"
        }
    },
    "sponsors_studies": {
        "nct_id": {
            "type": "correlational",
            "reason": "Study context (correlational for sponsor)"
        },
        "sponsor_id": {
            "type": "correlational",
            "reason": "Funding source doesn't cause drug efficacy"
        }
    }
}

# Task-specific FK relevance
# Which FKs are actually used for each prediction task

TASK_FK_MAPPING = {
    "rel-f1": {
        "driver-position": {
            "primary_table": "results",
            "relevant_fks": ["results", "qualifying", "standings"],
            "target": "position"
        },
        "driver-dnf": {
            "primary_table": "results",
            "relevant_fks": ["results", "qualifying"],
            "target": "dnf"
        },
        "driver-top3": {
            "primary_table": "results",
            "relevant_fks": ["results", "qualifying", "standings"],
            "target": "top3"
        },
        "results-position": {
            "primary_table": "results",
            "relevant_fks": ["results", "qualifying"],
            "target": "position"
        },
        "qualifying-position": {
            "primary_table": "qualifying",
            "relevant_fks": ["qualifying"],
            "target": "position"
        }
    },
    "rel-salt": {
        "item-plant": {
            "primary_table": "salesdocumentitem",
            "relevant_fks": ["salesdocumentitem", "customer"],
            "target": "PLANT"
        },
        "item-shippoint": {
            "primary_table": "salesdocumentitem",
            "relevant_fks": ["salesdocumentitem", "customer"],
            "target": "SHIPPINGPOINT"
        },
        "item-incoterms": {
            "primary_table": "salesdocumentitem",
            "relevant_fks": ["salesdocumentitem", "customer"],
            "target": "ITEMINCOTERMSCLASSIFICATION"
        },
        "sales-office": {
            "primary_table": "salesdocument",
            "relevant_fks": ["salesdocumentitem"],
            "target": "SALESOFFICE"
        },
        "sales-group": {
            "primary_table": "salesdocument",
            "relevant_fks": ["salesdocumentitem"],
            "target": "SALESGROUP"
        },
        "sales-payterms": {
            "primary_table": "salesdocument",
            "relevant_fks": ["salesdocumentitem"],
            "target": "CUSTOMERPAYMENTTERMS"
        },
        "sales-shipcond": {
            "primary_table": "salesdocument",
            "relevant_fks": ["salesdocumentitem"],
            "target": "SHIPPINGCONDITION"
        },
        "sales-incoterms": {
            "primary_table": "salesdocument",
            "relevant_fks": ["salesdocumentitem"],
            "target": "HEADERINCOTERMSCLASSIFICATION"
        }
    },
    "rel-trial": {
        "study-outcome": {
            "primary_table": "studies",
            "relevant_fks": ["interventions_studies", "conditions_studies", "facilities_studies", "sponsors_studies"],
            "target": "outcome"
        },
        "study-adverse": {
            "primary_table": "studies",
            "relevant_fks": ["interventions_studies", "conditions_studies", "facilities_studies"],
            "target": "adverse"
        },
        "site-success": {
            "primary_table": "facilities",
            "relevant_fks": ["facilities_studies", "sponsors_studies"],
            "target": "success"
        },
        "condition-sponsor-run": {
            "primary_table": "conditions",
            "relevant_fks": ["conditions_studies", "sponsors_studies"],
            "target": "sponsor_run"
        },
        "site-sponsor-run": {
            "primary_table": "facilities",
            "relevant_fks": ["facilities_studies", "sponsors_studies"],
            "target": "sponsor_run"
        }
    }
}


def get_fk_classification(dataset: str, table: str, fk_col: str) -> dict:
    """Get the classification for a specific FK."""
    if dataset == "rel-f1":
        config = F1_FK_CLASSIFICATION
    elif dataset == "rel-salt":
        config = SALT_FK_CLASSIFICATION
    elif dataset == "rel-trial":
        config = TRIAL_FK_CLASSIFICATION
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if table in config and fk_col in config[table]:
        return config[table][fk_col]
    else:
        return {"type": "unknown", "reason": "Not classified"}


def get_task_fks(dataset: str, task: str) -> dict:
    """Get FK configuration for a specific task."""
    if dataset in TASK_FK_MAPPING and task in TASK_FK_MAPPING[dataset]:
        return TASK_FK_MAPPING[dataset][task]
    else:
        return None


def summarize_classifications():
    """Print a summary of all FK classifications."""
    print("=" * 60)
    print("FK CLASSIFICATION SUMMARY")
    print("=" * 60)

    for dataset, config in [
        ("rel-f1", F1_FK_CLASSIFICATION),
        ("rel-salt", SALT_FK_CLASSIFICATION),
        ("rel-trial", TRIAL_FK_CLASSIFICATION)
    ]:
        print(f"\n{dataset}:")
        print("-" * 40)

        causal_count = 0
        correlational_count = 0

        for table, fks in config.items():
            for fk, info in fks.items():
                fk_type = info["type"]
                if fk_type == "causal":
                    causal_count += 1
                else:
                    correlational_count += 1
                print(f"  {table}.{fk}: {fk_type}")

        print(f"  Total: {causal_count} causal, {correlational_count} correlational")

    print("=" * 60)


if __name__ == "__main__":
    summarize_classifications()
