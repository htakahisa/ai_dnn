@echo off
rem Python path for this workspace. Change this named setting if Python moves.
set "TL1_PYTHONW=D:\git\python\pythonw.exe"
if not exist "%TL1_PYTHONW%" (
    echo Python not found. Edit TL1_PYTHONW in this launcher.
    pause
    exit /b 1
)
start "" "%TL1_PYTHONW%" "%~dp0tl1_ui_tactics.pyw"
