@echo off
rem Starts the app at http://127.0.0.1:8000. Extra arguments are passed to app.py (see --help).
rem Set EG2_PYTHON to use a Python environment other than .venv.
cd /d "%~dp0"
if defined EG2_PYTHON (set "PY=%EG2_PYTHON%") else (set "PY=.venv\Scripts\python.exe")
if not exist "%PY%" (
    echo Python environment not found - run setup.bat first.
    pause
    exit /b 1
)
"%PY%" app.py %*
pause
