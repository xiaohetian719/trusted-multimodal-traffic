@echo off
chcp 65001 >nul
echo ============================================================
echo   Trusted Multimodal Traffic System — One-Click Setup
echo ============================================================
echo.

REM ── Locate conda (check PATH first, then common install dirs) ──
set "CONDA_EXE="
where conda >nul 2>&1
if %errorlevel% equ 0 (
    set "CONDA_EXE=conda"
) else (
    for %%D in (
        "D:\anaconda3"
        "C:\Users\%USERNAME%\anaconda3"
        "C:\Users\%USERNAME%\miniconda3"
        "C:\ProgramData\anaconda3"
        "C:\anaconda3"
        "D:\miniconda3"
    ) do (
        if exist "%%~D\Scripts\conda.exe" (
            set "CONDA_EXE=%%~D\Scripts\conda.exe"
            echo [FOUND] conda at %%~D
        )
    )
)

if "%CONDA_EXE%"=="" (
    echo [ERROR] conda not found. Install Miniconda first:
    echo   https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

REM ── Check / create ai_lab environment ──
call %CONDA_EXE% info --envs | findstr "ai_lab" >nul
if %errorlevel% neq 0 (
    echo [SETUP] Creating conda environment: ai_lab (Python 3.10)
    call %CONDA_EXE% create -n ai_lab python=3.10 -y
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create ai_lab environment
        pause
        exit /b 1
    )
) else (
    echo [OK] conda environment "ai_lab" already exists
)

REM ── Install Python dependencies ──
echo.
echo [SETUP] Installing Python dependencies...
call %CONDA_EXE% run -n ai_lab pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] pip install failed
    pause
    exit /b 1
)

REM ── Check PyTorch ──
echo.
call %CONDA_EXE% run -n ai_lab python -c "import torch; print(f'[OK] PyTorch {torch.__version__}')"
if %errorlevel% neq 0 (
    echo [SETUP] Installing PyTorch (CUDA 12.1)...
    call %CONDA_EXE% run -n ai_lab pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
)

REM ── Verify core packages ──
echo.
echo [VERIFY] Checking installed packages...
call %CONDA_EXE% run -n ai_lab python -c "import numpy; import cv2; import torch; import ultralytics; print('[OK] All core packages ready')"
if %errorlevel% neq 0 (
    echo [WARN] Some packages may not be installed correctly
    echo [WARN] Try: conda activate ai_lab ^&^& pip install -r requirements.txt
    pause
    exit /b 1
)

REM ── Create models/ directory ──
if not exist "models\" (
    mkdir models
    echo [SETUP] Created models/ directory
)

REM ── Check model weight files ──
echo.
echo [CHECK] Model weight files:
if exist "models\depth_anything_v2_vits.pth" (
    echo   [OK] depth_anything_v2_vits.pth
) else (
    echo   [MISSING] depth_anything_v2_vits.pth  (~95 MB)
    echo     Download: https://github.com/DepthAnything/Depth-Anything-V2/releases
    echo     Place in: models\depth_anything_v2_vits.pth
)

if exist "models\best.pt" (
    echo   [OK] best.pt
) else if exist "backup\external\intelligent_traffic_ai\bdd_yolo_train_m_vs\weights\best.pt" (
    echo   [OK] best.pt (found in backup/)
) else (
    echo   [MISSING] best.pt  (YOLO BDD100K weights, ~20 MB)
    echo     Obtain from course shared drive and place in: models\best.pt
)

REM ── Check Depth-Anything-V2 ──
echo.
echo [CHECK] Depth-Anything-V2 library:
if exist "Depth-Anything-V2\depth_anything_v2\dpt.py" (
    echo   [OK] Depth-Anything-V2 found
) else (
    echo   [MISSING] Depth-Anything-V2
    echo     Run: git clone https://github.com/DepthAnything/Depth-Anything-V2
)

REM ── Check Ollama ──
echo.
echo [CHECK] Ollama (for Stage 3 LLM):
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Ollama is running
) else (
    echo   [INFO] Ollama not running or not installed
    echo     Download: https://ollama.com
    echo     Then run: ollama pull qwen2.5:7b
)

REM ── Done ──
echo.
echo ============================================================
echo   Setup complete!
echo.
echo   Next steps (see README.md for details):
echo     1. Clone Depth-Anything-V2 if missing
echo     2. Download model weights to models/
echo     3. Install Ollama and pull qwen2.5:7b ^(for Stage 3^)
echo.
echo   Run the pipeline:
echo     conda activate ai_lab
echo     python run_pipeline.py --video path\to\video.avi
echo ============================================================
pause
