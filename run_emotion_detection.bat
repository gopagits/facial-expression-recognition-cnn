
@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Python environment not found. Please repair or recreate .venv.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" webcam_cnn.py

pause
