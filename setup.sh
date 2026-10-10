#!/usr/bin/env bash
# One-time setup on Linux / macOS: creates .venv and installs dependencies.
# Linux PyPI wheels of PyTorch already include CUDA; macOS uses Apple Silicon (MPS) or CPU.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "Creating virtual environment in .venv ..."
    python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Setup complete."
echo "  Optional sample photos:  .venv/bin/python download_samples.py"
echo "  Start the app:           ./start.sh"
