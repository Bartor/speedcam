#!/bin/sh
# speedcam installer for macOS and Linux. Run in a terminal:
#
#   curl -fsSL https://raw.githubusercontent.com/Bartor/speedcam/main/install.sh | sh
#
# Installs Python 3.11+ and ffmpeg when they're missing (Homebrew on macOS; apt, dnf, pacman or
# zypper on Linux, asking for your password through sudo), downloads the latest speedcam
# release into ~/speedcam, installs its Python packages and starts the app. Run it again to
# update (installed packages and your projects in ~/speedcam-projects are kept).
#
# Environment: SPEEDCAM_HOME (install folder), SPEEDCAM_REPO (owner/name),
# SPEEDCAM_NO_START=1 (install only), SPEEDCAM_NO_SETUP=1 (download only; for testing).
set -eu

REPO="${SPEEDCAM_REPO:-Bartor/speedcam}"
DEST="${SPEEDCAM_HOME:-$HOME/speedcam}"

say() { printf '[speedcam] %s\n' "$*"; }
fail() { printf '[speedcam] %s\n' "$*" >&2; exit 1; }

# the newest Python 3.11+ on PATH
find_python() {
    for p in python3.13 python3.12 python3.11 python3; do
        if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
            command -v "$p"
            return 0
        fi
    done
    return 1
}

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

install_deps() {
    if [ "$(uname)" = "Darwin" ]; then
        if ! command -v brew >/dev/null 2>&1; then
            fail "Python 3.11+ and ffmpeg are needed. Install Homebrew first (https://brew.sh), then run this again."
        fi
        say "Installing Python and ffmpeg with Homebrew ..."
        brew install python@3.12 ffmpeg
    elif command -v apt-get >/dev/null 2>&1; then
        say "Installing Python and ffmpeg with apt (your password may be asked) ..."
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq python3 python3-venv ffmpeg
        # older releases (Ubuntu 22.04, Debian 12) ship Python 3.10/3.11 as python3
        if ! find_python >/dev/null; then
            $SUDO apt-get install -y -qq python3.11 python3.11-venv ||
                $SUDO apt-get install -y -qq python3.12 python3.12-venv ||
                fail "This system's packages have no Python 3.11+. Install it (e.g. from deadsnakes or pyenv), then run this again."
        fi
    elif command -v dnf >/dev/null 2>&1; then
        say "Installing Python and ffmpeg with dnf (your password may be asked) ..."
        $SUDO dnf install -y python3 ffmpeg-free || $SUDO dnf install -y python3 ffmpeg
    elif command -v pacman >/dev/null 2>&1; then
        say "Installing Python and ffmpeg with pacman (your password may be asked) ..."
        $SUDO pacman -S --needed --noconfirm python ffmpeg
    elif command -v zypper >/dev/null 2>&1; then
        say "Installing Python and ffmpeg with zypper (your password may be asked) ..."
        $SUDO zypper --non-interactive install python3 ffmpeg
    else
        fail "Python 3.11+ and ffmpeg are needed and no known package manager was found. Install them, then run this again."
    fi
}

say "Installing speedcam into $DEST"

# 1. Requirements: Python 3.11+ (with venv) and ffmpeg
if ! find_python >/dev/null || ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    install_deps
fi
PY="$(find_python)" || fail "No Python 3.11+ found after installing it. Open a new terminal and run this again."
command -v ffmpeg >/dev/null 2>&1 || fail "ffmpeg isn't on PATH after installing it. Open a new terminal and run this again."
if ! "$PY" -c 'import venv, ensurepip' 2>/dev/null; then
    # Debian/Ubuntu split venv out of python3
    v="$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    command -v apt-get >/dev/null 2>&1 && $SUDO apt-get install -y -qq "python$v-venv" || true
fi
say "Python: $("$PY" --version 2>&1) ($PY)"
say "ffmpeg: $(command -v ffmpeg)"

# 2. The latest release (downloaded and unpacked with Python: no curl/unzip versions to worry about)
"$PY" - "$REPO" "$DEST" <<'EOF'
import io, json, os, shutil, sys, tempfile, urllib.request, zipfile
from pathlib import Path

repo, dest = sys.argv[1], sys.argv[2]
req = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases/latest",
                             headers={"User-Agent": "speedcam-installer"})
rel = json.load(urllib.request.urlopen(req, timeout=30))
asset = next((a for a in rel["assets"] if a["name"].startswith("speedcam-") and a["name"].endswith(".zip")), None)
if asset is None:
    sys.exit(f"[speedcam] The latest release ({rel['tag_name']}) has no speedcam zip.")
print(f"[speedcam] Downloading speedcam {rel['tag_name']} ...", flush=True)
data = urllib.request.urlopen(asset["browser_download_url"], timeout=120).read()
with tempfile.TemporaryDirectory() as tmp:
    z = zipfile.ZipFile(io.BytesIO(data))
    z.extractall(tmp)
    for info in z.infolist():  # zipfile drops the executable bit: restore it (speedcam.sh)
        mode = info.external_attr >> 16
        if mode & 0o111:
            os.chmod(os.path.join(tmp, info.filename), mode & 0o777)
    top = next(p for p in Path(tmp).iterdir() if p.is_dir())
    # over the existing folder: an update keeps .venv (installed packages)
    shutil.copytree(top, dest, dirs_exist_ok=True)
print(f"[speedcam] speedcam {rel['tag_name']} is in {dest}")
EOF

if [ -n "${SPEEDCAM_NO_SETUP:-}" ]; then
    exit 0
fi

# 3. Python packages (a few minutes the first time; PyTorch with CUDA is ~3 GB), then start
"$PY" "$DEST/launch.py" --setup-only || fail "Setup didn't finish (see above). Run the same command again to retry."
if [ -n "${SPEEDCAM_NO_START:-}" ]; then
    say "Installed. Start it with: $DEST/speedcam.sh"
    exit 0
fi
say "Starting speedcam (next time: $DEST/speedcam.sh) ..."
exec "$DEST/speedcam.sh" </dev/null
