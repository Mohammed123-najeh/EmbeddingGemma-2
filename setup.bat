@echo off
rem One-time setup on Windows: creates .venv and installs dependencies.
setlocal
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo Creating virtual environment in .venv ...
    python -m venv .venv || goto :error
)
set "PY=.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip || goto :error

where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
    echo NVIDIA GPU detected - installing the CUDA build of PyTorch ...
    "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 || goto :error
) else (
    echo No NVIDIA GPU detected - installing the CPU build of PyTorch.
)

"%PY%" -m pip install -r requirements.txt || goto :error

echo.
echo Setup complete.
echo   Optional sample photos:  .venv\Scripts\python.exe download_samples.py
echo   Start the app:           start.bat
exit /b 0

:error
echo.
echo Setup failed - see the messages above.
exit /b 1
