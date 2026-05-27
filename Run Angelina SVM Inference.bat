@echo off
title Angelina SVM Inference Engine
color 0A
echo.
echo  ============================================================
echo     PROJECT ANGELINA — SVM Emergency Inference
echo  ============================================================
echo.

cd /d "%~dp0"
set PYTHON="C:\Users\aryas\OneDrive\Desktop\codes\angelina\angelina_env\Scripts\python.exe"

echo  [1/3] Starting Dashboard Server...
start "Angelina Dashboard Server" /MIN %PYTHON% 5_dashboard.py
timeout /t 3 /nobreak >nul

echo  [2/3] Opening Web Dashboard...
start "" http://localhost:5001

echo  [3/3] Starting SVM Real-Time Inference...
%PYTHON% 3_svm_inference.py

taskkill /FI "WINDOWTITLE eq Angelina Dashboard Server*" /F >nul 2>&1
echo  ============================================================
echo     Inference Session Ended.
pause
