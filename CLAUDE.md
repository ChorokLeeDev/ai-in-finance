# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of RelBench (Relational Deep Learning Benchmark) from Stanford SNAP, extended with uncertainty quantification (UQ) research focused on studying COVID-19 as a natural distribution shift experiment. The core research question: Does ML uncertainty spike during documented distribution shifts like COVID-19?

## Common Commands

### Installation
```bash
pip install -e .[full,example,dev,test]  # Install with all dependencies
pip install -e .                          # Core only (no PyTorch/PyG)
```

### Running Tests
```bash
pytest test/                              # Run all tests
pytest test/base test/datasets            # Run core tests (CI subset)
pytest test/modeling/                     # Run modeling tests (requires PyTorch)
pytest test/base/test_tasks.py -v         # Run single test file
```

### Training Models
```bash
# GNN examples (from examples/ directory)
python examples/gnn_entity.py --dataset rel-f1 --task driver-position
python examples/gnn_autocomplete.py --dataset rel-f1 --task results-position

# LightGBM baselines
python examples/lightgbm_entity.py --dataset rel-f1 --task driver-position
python examples/lightgbm_autocomplete.py --dataset rel-f1 --task results-position

# Ensemble training for UQ research
python examples/run_regression_ensemble.py --dataset rel-f1 --task driver-position --num_trials 5 --seeds 42 43 44 45 46

# Conformal prediction analysis
python examples/analyze_regression_uq_conformal.py --dataset rel-f1 --task results-position --num_seeds 3 --alpha 0.1
```

### Linting/Formatting
```bash
pre-commit run --all-files               # Run all pre-commit hooks
black .                                   # Format Python code
isort --profile black .                   # Sort imports
```

## Architecture

### Core Package (`relbench/`)

- **`base/`**: Foundation classes
  - `table.py`: `Table` - pandas DataFrame wrapper with metadata
  - `database.py`: `Database` - collection of Tables with pkey/fkey relationships
  - `dataset.py`: `Dataset` - Database + temporal split timestamps (val/test)
  - `task_base.py`, `task_entity.py`, `task_autocomplete.py`, `task_recommendation.py`: Task definitions with train/val/test tables and evaluation

- **`datasets/`**: Dataset implementations (amazon, avito, event, f1, hm, salt, stack, trial)
  - Each file defines a `Dataset` subclass with schema and data loading
  - Datasets cached to `~/.cache/relbench`

- **`tasks/`**: Task definitions per dataset
  - `__init__.py` exports `get_task(dataset_name, task_name)` and `get_task_names(dataset_name)`

- **`modeling/`**: PyTorch Geometric integration (requires `[full]` install)
  - `graph.py`: Convert Database to PyG HeteroData via pkey/fkey links
  - `loader.py`: Temporal neighbor sampling for mini-batch training
  - `nn.py`: `HeteroGraphSAGE` model with PyTorch Frame for tabular encoding

- **`metrics.py`**: Evaluation metrics (MAE, RMSE, accuracy, ROC-AUC, MAP, etc.)

### Examples (`examples/`)

- `gnn_*.py`: GNN training scripts for entity/autocomplete/recommendation tasks
- `lightgbm_*.py`: LightGBM baseline scripts
- `baseline_*.py`: Simple baselines (most frequent, etc.)
- `*_ensemble.py`, `*_uq*.py`: Uncertainty quantification research scripts

### Research Extensions (`chorok/`)

Custom research code for UQ and COVID-19 distribution shift analysis:
- `analyze_uq_ensemble.py`: Ensemble uncertainty analysis
- `run_salt_experiments.py`: SALT dataset experiments for COVID shift study
- `week*.md`: Research notes and progress

## Key APIs

```python
from relbench.datasets import get_dataset
from relbench.tasks import get_task

# Load dataset and task
dataset = get_dataset("rel-amazon", download=True)
task = get_task("rel-amazon", "user-churn", download=True)

# Get database and task tables
db = dataset.get_db()
train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")

# Evaluate predictions
metrics = task.evaluate(predictions, val_table)
```

## Dataset/Task Reference

Available datasets: rel-amazon, rel-avito, rel-event, rel-f1, rel-hm, rel-salt, rel-stack, rel-trial

Key dataset for UQ research: **rel-salt** (has COVID-19 temporal shift with splits at Feb 2020 / Jul 2020)

## Research Context

The `chorok/` directory and `RESEARCH_PLAN.md` contain ongoing UQ research. Key hypothesis: epistemic uncertainty should spike during COVID-19 (Feb-Jul 2020) enabling early detection of distribution shifts. Focus tasks are regression autocomplete tasks on rel-salt dataset.
