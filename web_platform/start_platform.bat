@echo off
chcp 65001 >nul
echo ============================================================
echo   Traffic AI Platform - Web Dashboard
echo ============================================================
echo.
echo Starting server at http://localhost:5000
echo Press Ctrl+C to stop
echo.

cd /d "%~dp0.."
set YOLO_CONFIG_DIR=%~dp0..
.venv\Scripts\python.exe web_platform\app.py
pause
