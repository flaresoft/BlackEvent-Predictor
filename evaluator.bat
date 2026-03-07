@echo off
chcp 65001 >nul 2>&1
title BlackEvent Evaluator

cd /d "%~dp0"

echo.
echo  ========================================
echo   BlackEvent Evaluator
echo  ========================================
echo.
echo   1. Local only     (localhost:8501)
echo   2. External access (ngrok tunnel)
echo.

set /p choice="  Select (1/2): "

if "%choice%"=="2" (
    python serve.py
) else (
    python -m streamlit run web/app.py --server.headless true
)
