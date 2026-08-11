@echo off
title Angelina Inference Engine
color 0A
echo.
echo ___________                        ___.                      .___ __________               __                         __________        __   
echo \_   _____/___   ____  __ __  _____\_ ^|__ _____    ____    __^| _/ \______   \____  _______/  ^|_ __ _________   ____   \______   \ _____/  ^|_ 
echo  ^|    __)/  _ \_/ ___\^|  ^|  \/  ___/^| __ \\__  \  /    \  / __ ^|   ^|     ___/  _ \/  ___/\   __\  ^|  \_  __ \_/ __ \   ^|    ^|  _//  _ \   __\
echo  ^|     \(  ^<_^> )  \___^|  ^|  /\___ \ ^| \_\ \/ __ \^|   ^|  \/ /_/ ^|   ^|    ^|  (  ^<_^> )___ \  ^|  ^| ^|  ^|  /^|  ^| \/\  ___/   ^|    ^|   (  ^<_^> )  ^|  
echo  \___  / \____/ \___  ^>____//____  ^>^|___  (____  /___^|  /\____ ^|   ^|____^|   \____/____  ^> ^|__^| ^|____/ ^|__^|    \___  ^>  ^|______  /\____/^|__^|  
echo      \/             \/           \/     \/     \/     \/      \/                      \/                          \/          \/             
echo.
echo  ============================================================
echo     PROJECT ANGELINA — Real-Time Predictive Inference
echo  ============================================================
echo.

cd /d "%~dp0"

rem -- Auto-detect Python: prefer local venv, fall back to system python --
if exist "%~dp0venv\Scripts\python.exe" (
    set PYTHON="%~dp0venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set PYTHON="%~dp0.venv\Scripts\python.exe"
) else (
    set PYTHON=python
)

echo  [1/3] Starting Dashboard Server (port 5001)...
start "Angelina Dashboard Server" /MIN %PYTHON% 5_dashboard.py

:: Give the Flask server a moment to boot before opening the browser
echo  [2/3] Waiting for dashboard to initialize...
timeout /t 3 /nobreak >nul

echo  [3/3] Opening Web Dashboard + Starting Real-Time Inference...
start "" http://localhost:5001

%PYTHON% 3_realtime_inference.py

echo.
echo  ============================================================
echo     Inference Session Ended. Shutting down dashboard...
echo  ============================================================

:: Kill the dashboard server when inference ends
taskkill /FI "WINDOWTITLE eq Angelina Dashboard Server*" /F >nul 2>&1

echo     All processes stopped. Goodbye!
echo  ============================================================
echo.
pause
