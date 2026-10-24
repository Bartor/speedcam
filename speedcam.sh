#!/usr/bin/env sh
# speedcam launcher: ./speedcam.sh starts the app; ./speedcam.sh <command> runs any CLI command.
# Sets up .venv, dependencies and the web UI on first run (see launch.py).
HERE="$(cd "$(dirname "$0")" && pwd)"
if ! command -v python3 >/dev/null 2>&1; then
    echo "[speedcam] Python 3.11 or newer is needed and wasn't found."
    echo "[speedcam]   macOS: brew install python@3.12    Linux: sudo apt install python3 python3-venv"
    exit 1
fi
exec python3 "$HERE/launch.py" "$@"
