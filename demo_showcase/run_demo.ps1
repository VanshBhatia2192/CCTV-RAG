# Launch Gait Biometrics Academic Showcase Demo on port 8501
$ErrorActionPreference = "Stop"
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Launching 2D Skeleton Gait Biometrics Academic Demo" -ForegroundColor Yellow
Write-Host "  Web Dashboard: http://127.0.0.1:8501" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

& .\.venv\Scripts\Activate.ps1
python -m uvicorn demo_showcase.app:app --host 127.0.0.1 --port 8501 --reload
