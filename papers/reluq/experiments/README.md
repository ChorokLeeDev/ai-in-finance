# RelUQ Experiments

This directory contains the experimental framework for the RelUQ paper.

## Quick Start

### Run a single experiment
```bash
python run_full_experiments.py --dataset rel-f1 --task driver-position --n_models 10 --sample_size 3000
```

### Run all experiments (takes ~2-4 hours)
```bash
python run_full_experiments.py --run_all --n_models 10 --sample_size 5000 --seeds 42 43 44
```

### Output
Results are saved to `results/full_experiments/results_final.json` with:
- Per-task Spearman correlations (uncertainty attribution vs error impact)
- Bootstrap confidence intervals
- Baseline comparisons (correlation clustering)
- Intervention experiment results

## Pre-registered Classifications

The following CEP (Causal Error Propagation) classifications were made **before** running experiments:

| Dataset | CEP? | Rationale |
|---------|------|-----------|
| rel-salt | Yes | ERP system: sales → item → shipment |
| rel-trial | Yes | Clinical: study → site → outcome |
| rel-f1 | Yes | Racing: driver → race → result |
| rel-avito | Yes | Classifieds: user → ad → interaction |
| rel-amazon | No | E-commerce: user-item associative |
| rel-hm | No | Fashion: user-item recommendations |
| rel-stack | No | Q&A: user-post bidirectional |
| rel-event | No | Events: user-event symmetric |

## Expected Results

Based on theory, CEP domains should show:
- High attribution-error correlation (ρ ≥ 0.80)
- Significant intervention effect (≥10% error reduction)

Non-CEP domains should show:
- Low/negative correlation (ρ ≈ 0)
- Minimal intervention effect (<5% error reduction)

## Files

- `run_full_experiments.py`: Main experiment runner
- `results/`: Output directory for experiment results
