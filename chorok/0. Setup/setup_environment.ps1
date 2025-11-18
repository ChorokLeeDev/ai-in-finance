# GNN 학습 환경 자동 설정 스크립트
# PowerShell에서 실행하세요

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GNN 학습 환경 설정 시작" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Miniconda 설치 완료 대기
Write-Host "Step 1: Miniconda 설치 확인 중..." -ForegroundColor Yellow

# Conda 초기화 (설치 후 PATH 설정)
$condaPath = "$env:USERPROFILE\miniconda3"
if (Test-Path $condaPath) {
    Write-Host "✓ Miniconda 설치 확인됨: $condaPath" -ForegroundColor Green
} else {
    Write-Host "✗ Miniconda가 설치되지 않았습니다." -ForegroundColor Red
    Write-Host "먼저 Miniconda 설치 명령어를 실행하세요:" -ForegroundColor Yellow
    Write-Host 'Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -outfile ".\miniconda.exe"' -ForegroundColor White
    Write-Host 'Start-Process -FilePath ".\miniconda.exe" -ArgumentList "/S" -Wait' -ForegroundColor White
    Write-Host 'del .\miniconda.exe' -ForegroundColor White
    exit 1
}

Write-Host ""
Write-Host "Step 2: Conda 초기화 중..." -ForegroundColor Yellow

# Conda 초기화
& "$condaPath\Scripts\conda.exe" init powershell

Write-Host "✓ Conda 초기화 완료" -ForegroundColor Green
Write-Host ""
Write-Host "⚠️  중요: PowerShell을 닫고 다시 열어주세요!" -ForegroundColor Red
Write-Host "그런 다음 아래 명령어를 실행하세요:" -ForegroundColor Yellow
Write-Host ""
Write-Host "conda activate base" -ForegroundColor White
Write-Host "cd c:\Users\hippo\relbench\chorok" -ForegroundColor White
Write-Host ".\create_gnn_env.ps1" -ForegroundColor White
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
