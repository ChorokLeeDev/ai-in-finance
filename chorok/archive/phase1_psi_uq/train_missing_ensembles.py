"""
Batch Training Script for Missing Ensemble Models

**Purpose**: Automate training of remaining ensemble models for Phase 2 of UQ research

**What this does**:
1. Identifies missing seed/task combinations
2. Trains LightGBM models for each missing combination
3. Saves predictions for ensemble UQ analysis
4. Tracks progress and provides time estimates

**Usage**:
    # Train all missing ensembles
    python chorok/train_missing_ensembles.py

    # Train specific task only
    python chorok/train_missing_ensembles.py --task item-shippoint

    # Dry run (show what would be trained)
    python chorok/train_missing_ensembles.py --dry-run

**Output**: Ensemble predictions saved to results/ensemble/rel-salt/{task}/
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

# Tasks that need complete ensembles (5 seeds each)
MISSING_TASKS = [
    'item-shippoint',
    'item-incoterms',
    'sales-payterms',
    'sales-shipcond',
    'sales-incoterms'
]

# sales-group needs additional seeds (already has seed 456)
SALES_GROUP_EXISTING_SEEDS = [456]
SALES_GROUP_NEEDED_SEEDS = [42, 43, 44, 45]

# Standard configuration
STANDARD_SEEDS = [42, 43, 44, 45, 46]
SAMPLE_SIZE = 50000
NUM_TRIALS = 10
DATASET = 'rel-salt'


def check_existing_predictions(task, seed, sample_size):
    """Check if predictions already exist for a given task/seed combination"""
    pred_path = Path(f'results/ensemble/{DATASET}/{task}/seed_{seed}_sample_{sample_size}.pkl')
    return pred_path.exists()


def get_missing_combinations():
    """Get list of (task, seed) combinations that need training"""
    missing = []

    # Check missing tasks
    for task in MISSING_TASKS:
        for seed in STANDARD_SEEDS:
            if not check_existing_predictions(task, seed, SAMPLE_SIZE):
                missing.append((task, seed))

    # Check sales-group additional seeds
    for seed in SALES_GROUP_NEEDED_SEEDS:
        if not check_existing_predictions('sales-group', seed, SAMPLE_SIZE):
            missing.append(('sales-group', seed))

    return missing


def train_model(task, seed, sample_size, num_trials, dry_run=False):
    """Train a single model with given parameters"""

    cmd = [
        sys.executable,  # Use same Python interpreter
        'examples/lightgbm_autocomplete.py',
        '--dataset', DATASET,
        '--task', task,
        '--seed', str(seed),
        '--sample_size', str(sample_size),
        '--num_trials', str(num_trials)
    ]

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Training: {task} (seed={seed})")
    print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        return True

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ Completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed with error:")
        print(f"  {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch train missing ensemble models')
    parser.add_argument('--task', type=str, help='Train only this specific task')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be trained without actually training')
    parser.add_argument('--num-trials', type=int, default=NUM_TRIALS, help='Number of Optuna trials')
    parser.add_argument('--sample-size', type=int, default=SAMPLE_SIZE, help='Training sample size')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()

    # Get missing combinations
    missing = get_missing_combinations()

    # Filter by task if specified
    if args.task:
        missing = [(t, s) for t, s in missing if t == args.task]
        if not missing:
            print(f"No missing seeds for task '{args.task}'")
            return

    # Summary
    print("=" * 70)
    print("ENSEMBLE TRAINING BATCH JOB")
    print("=" * 70)
    print(f"Dataset: {DATASET}")
    print(f"Sample size: {args.sample_size:,}")
    print(f"Num trials: {args.num_trials}")
    print(f"Total models to train: {len(missing)}")
    print(f"Estimated time: ~{len(missing) * 15} minutes ({len(missing) * 15 / 60:.1f} hours)")
    print()

    # Group by task for display
    task_counts = {}
    for task, seed in missing:
        task_counts[task] = task_counts.get(task, 0) + 1

    print("Missing combinations by task:")
    for task, count in sorted(task_counts.items()):
        seeds = [s for t, s in missing if t == task]
        print(f"  {task}: {count} seeds {seeds}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN MODE] - No actual training will occur\n")

    # Confirm before starting
    if not args.dry_run and not args.yes:
        response = input("\nProceed with training? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Track results
    results = {
        'started': datetime.now().isoformat(),
        'successful': [],
        'failed': [],
        'total': len(missing)
    }

    # Train each model
    for i, (task, seed) in enumerate(missing, 1):
        print(f"\n[{i}/{len(missing)}] ", end='')

        success = train_model(task, seed, args.sample_size, args.num_trials, args.dry_run)

        if success:
            results['successful'].append({'task': task, 'seed': seed})
        else:
            results['failed'].append({'task': task, 'seed': seed})

    # Final summary
    results['completed'] = datetime.now().isoformat()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total: {results['total']}")
    print(f"Successful: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")

    if results['failed']:
        print("\nFailed combinations:")
        for item in results['failed']:
            print(f"  {item['task']} (seed={item['seed']})")

    # Save results log
    if not args.dry_run:
        log_path = Path('chorok/results/ensemble_training_log.json')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nLog saved to: {log_path}")


if __name__ == '__main__':
    main()
