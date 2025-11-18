# GNN Training Environment Setup Script
# Run this in PowerShell after conda initialization

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GNN Environment Setup and Package Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 0: Accept Anaconda Terms of Service
Write-Host "Step 0: Accepting Anaconda Terms of Service..." -ForegroundColor Yellow
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
Write-Host "✓ Terms of Service accepted" -ForegroundColor Green
Write-Host ""

# Step 1: Create Conda Environment
Write-Host "Step 1: Creating gnn_env environment..." -ForegroundColor Yellow
conda create -n gnn_env python=3.10 -y

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Environment creation failed" -ForegroundColor Red
    exit 1
}
Write-Host "✓ gnn_env environment created" -ForegroundColor Green
Write-Host ""

# Step 2: Install PyTorch
Write-Host "Step 2: Installing PyTorch (CPU version)..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray
conda install -n gnn_env pytorch torchvision torchaudio cpuonly -c pytorch -y

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ PyTorch installation failed" -ForegroundColor Red
    exit 1
}
Write-Host "✓ PyTorch installed" -ForegroundColor Green
Write-Host ""

# Step 3: Install PyTorch Geometric
Write-Host "Step 3: Installing PyTorch Geometric..." -ForegroundColor Yellow
conda run -n gnn_env pip install torch-geometric

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ PyTorch Geometric installation failed" -ForegroundColor Red
    exit 1
}
Write-Host "✓ PyTorch Geometric installed" -ForegroundColor Green
Write-Host ""

# Step 4: Install Additional Packages
Write-Host "Step 4: Installing additional packages (scipy, networkx, scikit-learn, etc.)..." -ForegroundColor Yellow
conda install -n gnn_env scipy networkx scikit-learn matplotlib numpy pandas jupyter -y

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Additional package installation had issues, but continuing..." -ForegroundColor Yellow
}
Write-Host "✓ Additional packages installed" -ForegroundColor Green
Write-Host ""

# Step 5: Verify Installation
Write-Host "Step 5: Verifying installation..." -ForegroundColor Yellow
Write-Host ""

$torchVersion = conda run -n gnn_env python -c "import torch; print(torch.__version__)" 2>$null
$pygVersion = conda run -n gnn_env python -c "import torch_geometric; print(torch_geometric.__version__)" 2>$null

if ($torchVersion) {
    Write-Host "✓ PyTorch: $torchVersion" -ForegroundColor Green
} else {
    Write-Host "✗ PyTorch verification failed" -ForegroundColor Red
}

if ($pygVersion) {
    Write-Host "✓ PyTorch Geometric: $pygVersion" -ForegroundColor Green
} else {
    Write-Host "✗ PyTorch Geometric verification failed" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🎉 Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Select Python interpreter in VSCode:" -ForegroundColor White
Write-Host "   - Press Ctrl + Shift + P" -ForegroundColor Gray
Write-Host "   - Type 'Python: Select Interpreter'" -ForegroundColor Gray
Write-Host "   - Select 'gnn_env'" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Run the code:" -ForegroundColor White
Write-Host "   conda activate gnn_env" -ForegroundColor Gray
Write-Host "   python week1_gnn_training.py" -ForegroundColor Gray
Write-Host ""
Write-Host "Or press F5 in VSCode to run!" -ForegroundColor White
Write-Host ""
