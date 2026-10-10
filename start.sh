#!/usr/bin/env bash
# Starts the app at http://127.0.0.1:8000. Extra arguments are passed to app.py (see --help).
# Set EG2_PYTHON to use a Python environment other than ./.venv.
cd "$(dirname "$0")"
PY="${EG2_PYTHON:-.venv/bin/python}"
if [ ! -x "$PY" ]; then
    echo "Python environment not found - run ./setup.sh first."
    exit 1
fi
exec "$PY" app.py "$@"
