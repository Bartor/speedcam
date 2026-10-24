"""One-command launcher: sets everything up on first run, then starts speedcam.

    speedcam                 start the web app (default)
    speedcam analyze p.yaml  any other speedcam CLI command
    speedcam --shortcut      put a "speedcam" shortcut on the desktop (Windows)
    speedcam --setup-only    install / update without starting
    speedcam --check         check the requirements and say what's missing (installs nothing)

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


MAC = sys.platform == "darwin"
# how to install each requirement, per platform
HOW = {
    "python": {"win": "winget install Python.Python.3.12   (or python.org/downloads; tick 'Add python.exe to PATH')",
               "mac": "brew install python@3.12   (or python.org/downloads)",
               "linux": "sudo apt install python3.12 python3.12-venv   (or your distribution's package)"},
    "ffmpeg": {"win": "winget install Gyan.FFmpeg   (then open a new terminal so PATH updates)",
               "mac": "brew install ffmpeg",
               "linux": "sudo apt install ffmpeg"},
    "node": {"win": "winget install OpenJS.NodeJS.LTS   (or nodejs.org)",
             "mac": "brew install node",
             "linux": "sudo apt install nodejs npm   (or nodejs.org)"},
}


def how(what: str) -> str:
    return HOW[what]["win" if WIN else "mac" if MAC else "linux"]


def check_requirements(report: bool = False) -> bool:
    """Everything needed before installing: says what's missing and how to get it. With `report`,
    also lists what's fine (speedcam --check)."""
    ok = True

    def line(good: bool, name: str, detail: str, fix: str = "") -> None:
        nonlocal ok
        if not good:
            ok = False
        if report or not good:
            say(f"{'OK     ' if good else 'MISSING'} {name}: {detail}")
            if fix and not good:
                say(f"        install: {fix}")

    v = sys.version_info
    line(v >= (3, 11), "Python 3.11+", f"this is {v.major}.{v.minor}.{v.micro}", how("python"))
    for exe in ("ffmpeg", "ffprobe"):
        path = shutil.which(exe)
        line(bool(path), exe, path or "not on PATH", how("ffmpeg"))
    built = STATIC_INDEX.exists()
    npm = shutil.which("npm")
    if built:
        if report:
            say(f"OK      web UI: built{' (Node.js found, rebuilt when the sources change)' if npm else ''}")
    else:
        line(bool(npm), "Node.js 18+ (to build the web UI once)", npm or "npm not on PATH", how("node"))
    free = shutil.disk_usage(ROOT).free / 1e9
    need = 6 if not VPY.exists() else 1
    if free < need:
        say(f"WARNING disk: {free:.1f} GB free here; the first install needs ~6 GB (PyTorch with CUDA is ~3 GB)")
    elif report:
        say(f"OK      disk: {free:.0f} GB free")
    gpu = has_nvidia_gpu()
    if report or not gpu:
        say(f"{'OK     ' if gpu else 'NOTE   '} NVIDIA GPU: "
            f"{'found, detection runs on it' if gpu else 'not found; it works on the CPU, but analysis is 10-50x slower'}")
    return ok


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
                 f"Install it ({how('python')}), then run this again.")
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
                 f"Install Node.js 18+ ({how('node')}), then run this again.")
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
    if "--check" in argv:
        good = check_requirements(report=True)
        say("Ready: run speedcam to install and start." if good else "Install what's missing above, then run this again.")
        return 0 if good else 1
    if "--shortcut" in argv:
        make_shortcut()
        return 0
    if not check_requirements():
        say("Install what's missing above, then run this again.")
        return 1
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
