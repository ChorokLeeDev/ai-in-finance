#!/bin/bash

# Parallel Ensemble Training Runner for tmux
# Usage: bash chorok/run_training.sh

echo "========================================================================"
echo "Starting Parallel Ensemble Training in tmux"
echo "========================================================================"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gnn_env

# Change to project directory
cd "$(dirname "$0")/.."

# Run parallel training
python chorok/train_parallel.py --yes

echo ""
echo "========================================================================"
echo "Training Complete!"
echo "========================================================================"
echo ""
echo "Check results with:"
echo "  python chorok/check_ensemble_status.py"
