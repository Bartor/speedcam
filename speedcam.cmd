@echo off
rem speedcam launcher: double-click to start the app, or run "speedcam <command>" from this folder.
rem Sets up .venv, dependencies and the web UI on first run (see launch.py).
setlocal
set "HERE=%~dp0"
set "PY=python"
where py >nul 2>nul && set "PY=py -3"

rem The server needs no keyboard input. Feeding it NUL stops cmd.exe from asking
rem "Terminate batch job (Y/N)?" after Ctrl+C. Other commands (e.g. calibrate) keep stdin.
set "SERVER="
if "%~1"=="" set "SERVER=1"
if /i "%~1"=="serve" set "SERVER=1"
if defined SERVER (
    call :run %* <nul
) else (
    call :run %*
)
set "CODE=%errorlevel%"
rem keep the window open on errors when started by double-click
if not "%CODE%"=="0" if "%~1"=="" pause
exit /b %CODE%

:run
%PY% "%HERE%launch.py" %*
exit /b %errorlevel%
