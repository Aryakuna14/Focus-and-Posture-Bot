@echo off
title Angelina Model Comparison (CNN vs SVM)
color 0B
cd /d "%~dp0"

rem -- Auto-detect Python: prefer local venv, fall back to system python --
if exist "%~dp0venv\Scripts\python.exe" (
    set PYTHON="%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set PYTHON="%~dp0.venv\Scripts\python.exe"
) else (
    set PYTHON=python
)

echo  ============================================================
echo     PROJECT ANGELINA — Architecture Comparison
echo  ============================================================
echo.

%PYTHON% 7_compare_models.py

echo.
echo  ============================================================
echo     Comparison Complete! Opening graphs...
echo  ============================================================
start "" "evaluation_results\cnn_vs_svm_comparison.png"
start "" "evaluation_results\cnn_vs_svm_roc.png"
pause
