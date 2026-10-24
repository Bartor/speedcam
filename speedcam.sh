#!/usr/bin/env sh
# speedcam launcher: ./speedcam.sh starts the app; ./speedcam.sh <command> runs any CLI command.
# Sets up .venv, dependencies and the web UI on first run (see launch.py).
HERE="$(cd "$(dirname "$0")" && pwd)"
# the newest Python 3.11+ (on e.g. Ubuntu 22.04 "python3" is 3.10, python3.11 sits beside it)
for p in python3.13 python3.12 python3.11 python3; do
    if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
        exec "$p" "$HERE/launch.py" "$@"
    fi
done
echo "[speedcam] Python 3.11 or newer is needed and wasn't found."
echo "[speedcam]   macOS: brew install python@3.12    Linux: sudo apt install python3 python3-venv (3.11+)"
echo "[speedcam]   or run the installer: curl -fsSL https://raw.githubusercontent.com/Bartor/speedcam/main/install.sh | sh"
exit 1
