@echo off
title Angelina SVM Trainer
color 0A
cd /d "%~dp0"
set PYTHON="C:\Users\aryas\OneDrive\Desktop\codes\angelina\angelina_env\Scripts\python.exe"

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
