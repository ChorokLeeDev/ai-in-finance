"""
Ensemble Status Checker

**Purpose**: Monitor progress of ensemble training for UQ research

**What this does**:
1. Scans results/ensemble/ directory for existing predictions
2. Shows which tasks have complete ensembles (5+ seeds)
3. Lists missing seed/task combinations
4. Calculates overall Phase 2 completion percentage

**Usage**:
    # Show status for all SALT tasks
    python chorok/check_ensemble_status.py

    # Show detailed breakdown
    python chorok/check_ensemble_status.py --verbose

    # Export status to JSON
    python chorok/check_ensemble_status.py --export status.json
"""

import argparse
from pathlib import Path
import json
from collections import defaultdict

# All SALT tasks that need ensembles
ALL_SALT_TASKS = [
    'item-plant',
    'item-shippoint',
    'item-incoterms',
    'sales-office',
    'sales-group',
    'sales-payterms',
    'sales-shipcond',
    'sales-incoterms'
]

# Standard configuration
REQUIRED_SEEDS = 5
STANDARD_SEEDS = [42, 43, 44, 45, 46]
SAMPLE_SIZE = 50000
DATASET = 'rel-salt'


def scan_ensemble_predictions(dataset='rel-salt'):
    """Scan results/ensemble directory and return status dict"""

    results_dir = Path(f'results/ensemble/{dataset}')

    if not results_dir.exists():
        return {}

    # Track predictions by task
    task_predictions = defaultdict(list)

    for task_dir in results_dir.iterdir():
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name

        # Find all prediction files
        for pred_file in task_dir.glob('seed_*_sample_*.pkl'):
            # Extract seed from filename: seed_{seed}_sample_{size}.pkl
            parts = pred_file.stem.split('_')
            try:
                seed_idx = parts.index('seed') + 1
                seed = int(parts[seed_idx])

                sample_idx = parts.index('sample') + 1
                sample_size = int(parts[sample_idx])

                task_predictions[task_name].append({
                    'seed': seed,
                    'sample_size': sample_size,
                    'path': str(pred_file)
                })
            except (ValueError, IndexError):
                continue

    return dict(task_predictions)


def analyze_status(verbose=False):
    """Analyze ensemble completion status"""

    predictions = scan_ensemble_predictions(DATASET)

    status = {
        'complete': [],
        'incomplete': [],
        'missing': []
    }

    print("=" * 70)
    print("ENSEMBLE TRAINING STATUS - SALT Dataset")
    print("=" * 70)
    print(f"Required seeds per task: {REQUIRED_SEEDS}")
    print(f"Standard seeds: {STANDARD_SEEDS}")
    print()

    for task in ALL_SALT_TASKS:
        if task in predictions:
            preds = predictions[task]
            num_seeds = len(preds)
            seeds_found = sorted([p['seed'] for p in preds])

            # Check if complete (5+ seeds)
            if num_seeds >= REQUIRED_SEEDS:
                status['complete'].append(task)
                symbol = "[OK]"
                label = "COMPLETE"
            else:
                status['incomplete'].append(task)
                symbol = "[!!]"
                label = "INCOMPLETE"

            print(f"{symbol} {task:20s} [{num_seeds}/{REQUIRED_SEEDS} seeds] - {label}")

            if verbose:
                print(f"    Seeds found: {seeds_found}")
                missing_seeds = [s for s in STANDARD_SEEDS if s not in seeds_found]
                if missing_seeds:
                    print(f"    Missing seeds: {missing_seeds}")
                print()

        else:
            status['missing'].append(task)
            print(f"[  ] {task:20s} [0/{REQUIRED_SEEDS} seeds] - MISSING")

            if verbose:
                print(f"    Missing seeds: {STANDARD_SEEDS}")
                print()

    # Summary statistics
    total_tasks = len(ALL_SALT_TASKS)
    complete_tasks = len(status['complete'])
    completion_pct = (complete_tasks / total_tasks) * 100

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total tasks: {total_tasks}")
    print(f"Complete: {complete_tasks} ({completion_pct:.1f}%)")
    print(f"Incomplete: {len(status['incomplete'])}")
    print(f"Missing: {len(status['missing'])}")
    print()

    # Phase 2 status
    if complete_tasks == total_tasks:
        print("[OK] Phase 2 (Ensemble Training): COMPLETE")
        print("  -> Ready to proceed with Phase 3 (Uncertainty Quantification)")
    else:
        remaining = total_tasks - complete_tasks
        total_models_needed = remaining * REQUIRED_SEEDS
        print(f"[!!] Phase 2 (Ensemble Training): {completion_pct:.1f}% COMPLETE")
        print(f"  -> {remaining} tasks remaining")
        print(f"  -> Estimated {total_models_needed} models to train")
        print(f"  -> Run: python chorok/train_missing_ensembles.py")

    print("=" * 70)

    return status, predictions


def export_status(status, predictions, output_file):
    """Export status to JSON file"""

    export_data = {
        'dataset': DATASET,
        'required_seeds': REQUIRED_SEEDS,
        'total_tasks': len(ALL_SALT_TASKS),
        'complete_tasks': len(status['complete']),
        'incomplete_tasks': len(status['incomplete']),
        'missing_tasks': len(status['missing']),
        'completion_percentage': (len(status['complete']) / len(ALL_SALT_TASKS)) * 100,
        'status': status,
        'predictions': predictions
    }

    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"\nStatus exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Check ensemble training status')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed seed information')
    parser.add_argument('--export', type=str, help='Export status to JSON file')
    args = parser.parse_args()

    status, predictions = analyze_status(verbose=args.verbose)

    if args.export:
        export_status(status, predictions, args.export)


if __name__ == '__main__':
    main()
