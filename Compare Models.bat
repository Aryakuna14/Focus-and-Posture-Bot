@echo off
title Angelina Model Comparison (CNN vs SVM)
color 0B
cd /d "%~dp0"
set PYTHON="C:\Users\aryas\OneDrive\Desktop\codes\angelina\angelina_env\Scripts\python.exe"

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
