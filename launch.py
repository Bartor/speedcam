"""One-command launcher: sets everything up on first run, then starts speedcam.

    speedcam                 start the web app (default)
    speedcam analyze p.yaml  any other speedcam CLI command
    speedcam --shortcut      put a "speedcam" shortcut on the desktop (Windows)
    speedcam --setup-only    install / update without starting
    speedcam --check         check the requirements and say what's missing (installs nothing)
    speedcam --ui-hash       print the version hash of the web UI sources (used by CI)

What it takes care of, only when needed:
  * creates .venv (Python 3.11+) and installs the package, with CUDA PyTorch when an NVIDIA GPU is present
  * reinstalls when pyproject.toml changes
  * gets the web UI matching this code: downloads the prebuilt one from the project's GitHub
    releases (built by .github/workflows/ui-release.yml), or builds it with Node.js when there is
    none (e.g. while editing the UI)

Uses only the standard library, so it runs with any system Python 3.11+.
"""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
WIN = os.name == "nt"
VPY = VENV / ("Scripts/python.exe" if WIN else "bin/python")
STAMP = VENV / ".speedcam-installed"
WEB = ROOT / "web"
STATIC = ROOT / "speedcam" / "server" / "static"
STATIC_INDEX = STATIC / "index.html"
UI_STAMP = STATIC / ".ui-hash"  # hash of the sources the built UI came from
# prebuilt UIs: one ui-<hash>.zip per version of the UI sources on this repository's rolling
# "ui-latest" release (forks set SPEEDCAM_UI_REPO=owner/name)
UI_REPO = os.environ.get("SPEEDCAM_UI_REPO", "Bartor/speedcam")
UI_RELEASE = "ui-latest"
UI_BASE = os.environ.get("SPEEDCAM_UI_BASE", f"https://github.com/{UI_REPO}/releases/download/{UI_RELEASE}")
UI_FILES = ["index.html", "package.json", "package-lock.json", "vite.config.ts", "tsconfig.json"]
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
    npm = shutil.which("npm")
    if report:
        current = built_ui_hash() == ui_hash()
        say(f"OK      web UI: {'built for this version' if current else 'downloaded (or built) on start'}"
            f"; Node.js {'found' if npm else 'not needed unless you change the UI'}")
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


def ui_hash() -> str:
    """Version of the web UI sources: a hash of web/src and the build files, with line endings
    normalised so a Windows checkout (CRLF) and CI (LF) agree."""
    files = sorted([f for f in (WEB / "src").rglob("*") if f.is_file()] +
                   [WEB / n for n in UI_FILES if (WEB / n).is_file()])
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(WEB).as_posix().encode() + b"\0")
        h.update(f.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:16]


def built_ui_hash() -> str:
    return UI_STAMP.read_text().strip() if UI_STAMP.exists() and STATIC_INDEX.exists() else ""


def download_ui(want: str) -> bool:
    """Fetch the prebuilt UI for these sources from the GitHub release; False when there is none."""
    url = f"{UI_BASE}/ui-{want}.zip"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code != 404:
            say(f"Couldn't download the prebuilt web UI ({e.code}).")
        return False
    except (urllib.error.URLError, OSError) as e:
        say(f"Couldn't reach GitHub for the prebuilt web UI ({e}).")
        return False
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        if "index.html" not in z.namelist():
            raise zipfile.BadZipFile("no index.html")
        tmp = STATIC.with_name("static.download")
        shutil.rmtree(tmp, ignore_errors=True)
        z.extractall(tmp)
    except (zipfile.BadZipFile, OSError) as e:
        say(f"The downloaded web UI is unusable ({e}).")
        return False
    (tmp / UI_STAMP.name).write_text(want)
    shutil.rmtree(STATIC, ignore_errors=True)
    tmp.rename(STATIC)
    return True


def ensure_ui() -> None:
    """The web UI built from exactly these sources: kept if it already is, else downloaded from
    the GitHub release (no Node.js needed), else built locally with Node.js."""
    want = ui_hash()
    if built_ui_hash() == want:
        return
    say("Getting the web UI for this version ...")
    if download_ui(want):
        say("Web UI downloaded.")
        return
    npm = shutil.which("npm")
    if not npm:
        if STATIC_INDEX.exists():
            say("No prebuilt web UI matches these sources and Node.js isn't installed; using the existing "
                "build, which may be from another version.")
            return
        sys.exit("No prebuilt web UI matches this version (e.g. the UI was changed locally, or the release "
                 f"build hasn't finished yet) and Node.js isn't installed to build it. Install Node.js 18+ "
                 f"({how('node')}), or wait a few minutes after a push and run this again.")
    if not (WEB / "node_modules").exists():
        say("Installing web UI packages ...")
        run([npm, "install", "--no-audit", "--no-fund", "--loglevel=error"], cwd=WEB)
    say("Building the web UI ...")
    run([npm, "run", "build", "--silent", "--", "--logLevel", "warn"], cwd=WEB)
    UI_STAMP.write_text(want)
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
    if "--ui-hash" in argv:
        print(ui_hash())
        return 0
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
