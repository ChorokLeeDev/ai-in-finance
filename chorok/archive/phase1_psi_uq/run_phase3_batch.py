"""
Batch Phase 3: Uncertainty Quantification for All SALT Tasks

Run temporal uncertainty analysis for all 8 SALT tasks automatically.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

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

SAMPLE_SIZE = 10000


def run_uncertainty_analysis(task: str) -> dict:
    """Run uncertainty analysis for a single task."""
    print(f"\n{'='*70}")
    print(f"Processing: {task}")
    print(f"{'='*70}")

    cmd = [
        sys.executable,
        'chorok/temporal_uncertainty_analysis.py',
        '--task', task,
        '--sample_size', str(SAMPLE_SIZE)
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return {'task': task, 'status': 'success'}
    except subprocess.CalledProcessError as e:
        print(f"ERROR for {task}:")
        print(e.stderr)
        return {'task': task, 'status': 'failed', 'error': str(e.stderr[:500])}


def main():
    print("="*70)
    print("PHASE 3: UNCERTAINTY QUANTIFICATION ANALYSIS")
    print("="*70)
    print(f"Total tasks: {len(ALL_SALT_TASKS)}")
    print(f"Sample size: {SAMPLE_SIZE:,}")
    print()

    results = {
        'started': datetime.now().isoformat(),
        'tasks': []
    }

    # Process each task
    for i, task in enumerate(ALL_SALT_TASKS, 1):
        print(f"\n[{i}/{len(ALL_SALT_TASKS)}] {task}")
        task_result = run_uncertainty_analysis(task)
        results['tasks'].append(task_result)

    results['completed'] = datetime.now().isoformat()

    # Summary
    successful = [r for r in results['tasks'] if r['status'] == 'success']
    failed = [r for r in results['tasks'] if r['status'] == 'failed']

    print("\n" + "="*70)
    print("PHASE 3 COMPLETE")
    print("="*70)
    print(f"Successful: {len(successful)}/{len(ALL_SALT_TASKS)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed tasks:")
        for r in failed:
            print(f"  - {r['task']}")

    # Save log
    log_path = Path('chorok/results/phase3_batch_log.json')
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nLog saved: {log_path}")

    print("\n" + "="*70)
    print("NEXT STEP: Phase 4 (Correlation Analysis)")
    print("="*70)
    print("Run: python chorok/compare_shift_uncertainty.py")
    print("="*70)


if __name__ == '__main__':
    main()
