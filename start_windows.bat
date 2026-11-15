@echo off
cd /d "%~dp0"
rem Stop any older OmniPart server still using port 8000
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8000 " ^| findstr LISTENING') do (
  echo Stopping old server on port 8000 ^(PID %%p^)...
  taskkill /PID %%p /F >nul 2>&1
)
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv || (echo Python not found. Install Python 3.11 from python.org and tick "Add to PATH". & pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\pip install -r requirements.txt || (echo Package install failed - check your internet connection. & pause & exit /b 1)
)
.venv\Scripts\python run.py
pause
