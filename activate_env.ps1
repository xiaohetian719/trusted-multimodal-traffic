# -*- coding: utf-8 -*-
# Trusted Multimodal Traffic System - Environment Activation Script
# Run: .\activate_env.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Trusted Multimodal Traffic AI System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

# Set ultralytics config to local directory (avoids permission issues)
$env:YOLO_CONFIG_DIR = $projectRoot

# Activate virtual environment
$venvActivate = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    . $venvActivate
    Write-Host "[OK] Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "[ERROR] .venv not found at $venvActivate" -ForegroundColor Red
    return
}

Write-Host ""
Write-Host "Environment ready!" -ForegroundColor Green
Write-Host ""
Write-Host "Quick start:" -ForegroundColor Yellow
Write-Host "  python run_pipeline.py                          # Process all videos"
Write-Host "  python run_pipeline.py --video videos\S06_c043.avi --max-secs 10  # Quick test (10s)"
Write-Host "  python perception_pipeline.py                   # Stage 1 only"
Write-Host ""
Write-Host "For Stage 3 (LLM narrative), install Ollama first:" -ForegroundColor Yellow
Write-Host "  https://ollama.com  ->  ollama pull qwen2.5:7b"
Write-Host ""
