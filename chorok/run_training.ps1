# Parallel Ensemble Training Runner for Windows
# Usage: powershell -File chorok/run_training.ps1

Write-Host "========================================================================"
Write-Host "Starting Parallel Ensemble Training"
Write-Host "========================================================================"
Write-Host ""

# Change to project directory
Set-Location $PSScriptRoot\..

# Run parallel training with output to log file
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logfile = "chorok/results/training_$timestamp.log"

Write-Host "Training started at: $(Get-Date)"
Write-Host "Log file: $logfile"
Write-Host ""
Write-Host "Monitor progress with:"
Write-Host "  Get-Content $logfile -Wait"
Write-Host ""

# Run training and capture output
& C:\Users\hippo\miniconda3\envs\gnn_env\python.exe chorok\train_parallel.py --yes | Tee-Object -FilePath $logfile

Write-Host ""
Write-Host "========================================================================"
Write-Host "Training Complete!"
Write-Host "========================================================================"
Write-Host ""
Write-Host "Check results with:"
Write-Host "  python chorok/check_ensemble_status.py"
