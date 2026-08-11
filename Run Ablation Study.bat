@echo off
title Angelina IEEE Ablation Study
color 0B
echo ___________                        ___.                      .___ __________               __                         __________        __   
echo \_   _____/___   ____  __ __  _____\_ ^|__ _____    ____    __^| _/ \______   \____  _______/  ^|_ __ _________   ____   \______   \ _____/  ^|_ 
echo  ^|    __)/  _ \_/ ___\^|  ^|  \/  ___/^| __ \\__  \  /    \  / __ ^|   ^|     ___/  _ \/  ___/\   __\  ^|  \_  __ \_/ __ \   ^|    ^|  _//  _ \   __\
echo  ^|     \(  ^<_^> )  \___^|  ^|  /\___ \ ^| \_\ \/ __ \^|   ^|  \/ /_/ ^|   ^|    ^|  (  ^<_^> )___ \  ^|  ^| ^|  ^|  /^|  ^| \/\  ___/   ^|    ^|   (  ^<_^> )  ^|  
echo  \___  / \____/ \___  ^>____//____  ^>^|___  (____  /___^|  /\____ ^|   ^|____^|   \____/____  ^> ^|__^| ^|____/ ^|__^|    \___  ^>  ^|______  /\____/^|__^|  
echo      \/             \/           \/     \/     \/     \/      \/                      \/                          \/          \/             
cd /d "%~dp0"

rem -- Auto-detect Python: prefer local venv, fall back to system python --
if exist "%~dp0venv\Scripts\python.exe" (
    set PYTHON="%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set PYTHON="%~dp0.venv\Scripts\python.exe"
) else (
    set PYTHON=python
)

echo.
echo  ============================================================
echo     PROJECT ANGELINA — IEEE Ablation Study
echo  ============================================================
echo.

%PYTHON% 6b_ablation_study.py

echo.
echo  ============================================================
echo     Ablation Study Complete. Check evaluation_results\ folder.
echo  ============================================================
echo.
pause
