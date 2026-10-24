# speedcam installer for Windows. Run in PowerShell:
#
#   irm https://raw.githubusercontent.com/Bartor/speedcam/main/install.ps1 | iex
#
# Installs Python 3.12 and ffmpeg with winget when they're missing, downloads the latest
# speedcam release into %LOCALAPPDATA%\speedcam, installs its Python packages, puts a shortcut
# on the desktop and starts the app. Run it again to update (installed packages and your
# projects in ~\speedcam-projects are kept).
#
# Environment: SPEEDCAM_HOME (install folder), SPEEDCAM_REPO (owner/name),
# SPEEDCAM_NO_START=1 (install only), SPEEDCAM_NO_SETUP=1 (download only; for testing).

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest is very slow with its progress bar
$Repo = if ($env:SPEEDCAM_REPO) { $env:SPEEDCAM_REPO } else { "Bartor/speedcam" }
$Dest = if ($env:SPEEDCAM_HOME) { $env:SPEEDCAM_HOME } else { Join-Path $env:LOCALAPPDATA "speedcam" }

function Say($msg) { Write-Host "[speedcam] $msg" }
function Fail($msg) { Write-Host "[speedcam] $msg" -ForegroundColor Red; throw "speedcam install stopped" }
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}

# A Python 3.11+ that really runs ("python" can be the Microsoft Store stub): returns
# @{ exe = ...; args = @(...) }, or $null. (No script-scope variables: under "irm | iex" there's
# no script scope.)
function Find-Python {
    foreach ($c in @(@{ exe = "py"; args = @("-3") }, @{ exe = "python"; args = @() })) {
        if (-not (Get-Command $c.exe -ErrorAction SilentlyContinue)) { continue }
        $exe = $c["exe"]; $a = $c["args"]
        $ok = & $exe @a -c "import sys; print(sys.version_info >= (3, 11))" 2>$null
        if ("$ok".Trim() -eq "True") { return $c }
    }
    return $null
}

function Winget-Install($id, $what) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Fail "$what is needed and winget isn't available to install it. Install $what yourself (see https://github.com/$Repo#install), then run this again."
    }
    Say "Installing $what with winget ..."
    winget install --exact --id $id --silent --accept-package-agreements --accept-source-agreements --disable-interactivity
    if ($LASTEXITCODE -ne 0) { Say "winget reported code $LASTEXITCODE for $what; checking whether it's usable anyway ..." }
    Refresh-Path
}

Say "Installing speedcam into $Dest"

# 1. Requirements: Python 3.11+ and ffmpeg
$PyCmd = Find-Python
if (-not $PyCmd) {
    Winget-Install "Python.Python.3.12" "Python 3.12"
    $PyCmd = Find-Python
    if (-not $PyCmd) { Fail "Python was installed but isn't usable in this window yet. Open a new PowerShell window and run the same command again." }
}
$PyExe = $PyCmd["exe"]
$PyArgs = $PyCmd["args"]
Say ("Python: " + (& $PyExe @PyArgs -c "import sys; print(sys.version.split()[0])"))
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -or -not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    Winget-Install "Gyan.FFmpeg" "ffmpeg"
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { Fail "ffmpeg was installed but isn't on PATH in this window yet. Open a new PowerShell window and run the same command again." }
}
Say "ffmpeg: $((Get-Command ffmpeg).Source)"

# 2. The latest release
$release = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest" -Headers @{ "User-Agent" = "speedcam-installer" }
$asset = $release.assets | Where-Object { $_.name -like "speedcam-*.zip" } | Select-Object -First 1
if (-not $asset) { Fail "The latest release ($($release.tag_name)) has no speedcam zip." }
Say "Downloading speedcam $($release.tag_name) ..."
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("speedcam-install-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    $zip = Join-Path $tmp $asset.name
    Invoke-WebRequest $asset.browser_download_url -OutFile $zip -UseBasicParsing
    Expand-Archive $zip -DestinationPath $tmp
    $src = Get-ChildItem $tmp -Directory | Select-Object -First 1
    New-Item -ItemType Directory -Path $Dest -Force | Out-Null
    # over the existing folder: an update keeps .venv (installed packages)
    Copy-Item (Join-Path $src.FullName "*") $Dest -Recurse -Force
} finally {
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
Say "speedcam $($release.tag_name) is in $Dest"
if ($env:SPEEDCAM_NO_SETUP) { return }

# 3. Python packages (a few minutes the first time; PyTorch with CUDA is ~3 GB), shortcut, start
$launch = Join-Path $Dest "launch.py"
& $PyExe @PyArgs $launch --setup-only
if ($LASTEXITCODE -ne 0) { Fail "Setup didn't finish (see above). Run the same command again to retry." }
& $PyExe @PyArgs $launch --shortcut
if ($env:SPEEDCAM_NO_START) {
    Say "Installed. Start it with the 'speedcam' shortcut on the desktop."
    return
}
Say "Starting speedcam (next time: the 'speedcam' shortcut on the desktop) ..."
& (Join-Path $Dest "speedcam.cmd")
