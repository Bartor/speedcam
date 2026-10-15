#!/usr/bin/env sh
# speedcam launcher: ./speedcam.sh starts the app; ./speedcam.sh <command> runs any CLI command.
# Sets up .venv, dependencies and the web UI on first run (see launch.py).
HERE="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$HERE/launch.py" "$@"
