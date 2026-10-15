"""One-command launcher: sets everything up on first run, then starts speedcam.

    speedcam                 start the web app (default)
    speedcam analyze p.yaml  any other speedcam CLI command
    speedcam --shortcut      put a "speedcam" shortcut on the desktop (Windows)
    speedcam --setup-only    install / update without starting

What it takes care of, only when needed:
  * creates .venv (Python 3.11+) and installs the package, with CUDA PyTorch when an NVIDIA GPU is present
  * reinstalls when pyproject.toml changes
  * rebuilds the web UI when web/src is newer than the built copy (needs Node.js)

Uses only the standard library, so it runs with any system Python 3.11+.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
WIN = os.name == "nt"
VPY = VENV / ("Scripts/python.exe" if WIN else "bin/python")
STAMP = VENV / ".speedcam-installed"
WEB = ROOT / "web"
STATIC_INDEX = ROOT / "speedcam" / "server" / "static" / "index.html"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


def say(msg: str) -> None:
    print(f"[speedcam] {msg}", flush=True)


def run(cmd: list, **kw) -> None:
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def has_nvidia_gpu() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        return subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def deps_fingerprint() -> str:
    return hashlib.sha256((ROOT / "pyproject.toml").read_bytes()).hexdigest()


def ensure_python_env() -> None:
    if sys.version_info < (3, 11):
        sys.exit(f"speedcam needs Python 3.11 or newer (this is {sys.version.split()[0]}). "
                 "Install it from python.org, then run this again.")
    if not VPY.exists():
        say(f"First run: creating a virtual environment in {VENV} ...")
        run([sys.executable, "-m", "venv", VENV])
    if STAMP.exists() and STAMP.read_text().strip() == deps_fingerprint():
        return
    say("Installing dependencies (a few minutes the first time; PyTorch with CUDA is ~3 GB) ...")
    run([VPY, "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    has_torch = subprocess.run([VPY, "-c", "import torch"], capture_output=True).returncode == 0
    if not has_torch:
        if has_nvidia_gpu():
            say("NVIDIA GPU found: installing CUDA build of PyTorch ...")
            run([VPY, "-m", "pip", "install", "--quiet", "torch", "torchvision", "--index-url", TORCH_CUDA_INDEX])
        else:
            say("No NVIDIA GPU found: installing CPU PyTorch (detection will be slow).")
    run([VPY, "-m", "pip", "install", "--quiet", "-e", f"{ROOT}[dev]"])
    STAMP.write_text(deps_fingerprint())
    say("Dependencies installed.")


def newest_mtime(paths) -> float:
    m = 0.0
    for base in paths:
        base = Path(base)
        files = [base] if base.is_file() else base.rglob("*")
        for f in files:
            if f.is_file() and "node_modules" not in f.parts:
                m = max(m, f.stat().st_mtime)
    return m


def ensure_ui() -> None:
    sources = [WEB / "src", WEB / "index.html", WEB / "package.json", WEB / "vite.config.ts"]
    built = STATIC_INDEX.stat().st_mtime if STATIC_INDEX.exists() else 0.0
    if built and newest_mtime(sources) <= built:
        return
    npm = shutil.which("npm")
    if not npm:
        if STATIC_INDEX.exists():
            say("Web UI sources changed but Node.js isn't installed; using the existing build.")
            return
        sys.exit("The web UI hasn't been built and Node.js (npm) isn't installed. "
                 "Install Node.js 18+ from nodejs.org, then run this again.")
    if not (WEB / "node_modules").exists():
        say("Installing web UI packages ...")
        run([npm, "install", "--no-audit", "--no-fund", "--loglevel=error"], cwd=WEB)
    say("Building the web UI ...")
    run([npm, "run", "build", "--silent", "--", "--logLevel", "warn"], cwd=WEB)
    say("Web UI built.")


def make_shortcut() -> None:
    if not WIN:
        sys.exit("--shortcut is only implemented for Windows; on macOS/Linux alias ./speedcam.sh instead.")
    target = ROOT / "speedcam.cmd"
    ps = (
        "$d=[Environment]::GetFolderPath('Desktop');"
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d 'speedcam.lnk'));"
        f"$s.TargetPath='{target}';$s.WorkingDirectory='{ROOT}';"
        "$s.Description='Drone vehicle speed measurement';$s.WindowStyle=7;$s.Save()"
    )
    run(["powershell", "-NoProfile", "-Command", ps])
    say("Desktop shortcut 'speedcam' created. Double-click it to start the app.")


def main(argv: list[str]) -> int:
    if "--shortcut" in argv:
        make_shortcut()
        return 0
    setup_only = "--setup-only" in argv
    argv = [a for a in argv if a != "--setup-only"]
    try:
        ensure_python_env()
        ensure_ui()
    except subprocess.CalledProcessError as e:
        say(f"Setup step failed: {' '.join(map(str, e.cmd))}")
        return e.returncode or 1
    if setup_only:
        say("Setup complete.")
        return 0
    args = argv or ["serve"]
    exe = VENV / ("Scripts/speedcam.exe" if WIN else "bin/speedcam")
    try:
        return subprocess.run([str(exe), *args]).returncode
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
