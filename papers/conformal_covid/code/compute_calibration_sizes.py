import json
import math
import sys

sys.path.insert(0, "/Users/i767700/Github/ai-in-finance")

from relbench.datasets import get_dataset
from relbench.tasks import get_task

TASKS = [
    "sales-shipcond", "sales-payterms", "sales-group", "item-plant",
    "item-shippoint", "sales-incoterms", "item-incoterms", "sales-office",
]

print("Loading SALT dataset...")
dataset = get_dataset("rel-salt", download=False)

results = {}
for task_name in TASKS:
    print(f"\n--- {task_name} ---")
    task = get_task("rel-salt", task_name, download=False)
    val_table = task.get_table("val")
    test_table = task.get_table("test")
    n_val = len(val_table.df)
    n_test = len(test_table.df)
    n_cal = math.floor(n_val / 2)
    print(f"  n_val  = {n_val}")
    print(f"  n_cal  = {n_cal} (floor(n_val/2))")
    print(f"  n_test = {n_test}")
    results[task_name] = {
        "n_val": n_val,
        "n_cal": n_cal,
        "n_test": n_test,
    }

out_path = "/Users/i767700/Github/ai-in-finance/papers/conformal_covid/results/calibration_sizes.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
