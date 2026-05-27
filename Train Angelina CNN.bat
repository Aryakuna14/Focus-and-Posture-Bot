@echo off
title Angelina CNN Training Engine
color 0B
echo ___________                        ___.                      .___ __________               __                         __________        __   
echo \_   _____/___   ____  __ __  _____\_ ^|__ _____    ____    __^| _/ \______   \____  _______/  ^|_ __ _________   ____   \______   \ _____/  ^|_ 
echo  ^|    __)/  _ \_/ ___\^|  ^|  \/  ___/^| __ \\__  \  /    \  / __ ^|   ^|     ___/  _ \/  ___/\   __\  ^|  \_  __ \_/ __ \   ^|    ^|  _//  _ \   __\
echo  ^|     \(  ^<_^> )  \___^|  ^|  /\___ \ ^| \_\ \/ __ \^|   ^|  \/ /_/ ^|   ^|    ^|  (  ^<_^> )___ \  ^|  ^| ^|  ^|  /^|  ^| \/\  ___/   ^|    ^|   (  ^<_^> )  ^|  
echo  \___  / \____/ \___  ^>____//____  ^>^|___  (____  /___^|  /\____ ^|   ^|____^|   \____/____  ^> ^|__^| ^|____/ ^|__^|    \___  ^>  ^|______  /\____/^|__^|  
echo      \/             \/           \/     \/     \/     \/      \/                      \/                          \/          \/             
cd /d "%~dp0"

:RUN_SCRIPT
cls
"C:\Users\aryas\OneDrive\Desktop\codes\angelina\angelina_env\Scripts\python.exe" 2_train_cnn.py

:PROMPT
echo.
set /p choice="Do you want to run the training again? (Y/N): "
if /i "%choice%"=="Y" goto RUN_SCRIPT
if /i "%choice%"=="N" goto END
echo Invalid choice. Please enter Y or N.
goto PROMPT

:END
