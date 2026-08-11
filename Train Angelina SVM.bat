@echo off
title Angelina SVM Trainer
color 0A
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
echo     PROJECT ANGELINA — SVM Emergency Trainer
echo  ============================================================
echo.

%PYTHON% 2_train_svm.py

echo.
echo  ============================================================
echo     Training Complete! You can now run the SVM Inference bat.
echo  ============================================================
pause
