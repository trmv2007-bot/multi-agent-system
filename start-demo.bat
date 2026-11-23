@echo off
setlocal
cd /d "%~dp0"

title Multi-agent dashboard (demo)

where python >nul 2>&1
if errorlevel 1 (
  echo.
  echo   Python was not found on your PATH.
  echo   Install Python 3.10 or newer, tick "Add python.exe to PATH" in the
  echo   installer, then run this file again.
  echo.
  pause
  exit /b 1
)

echo.
echo   Starting the dashboard in demo mode: no model key needed.
echo   Every model call is answered from src/demo.py, so the answers are
echo   scripted and nothing leaves this machine.
echo   It opens your browser at http://127.0.0.1:8765
echo   Close this window, or press Ctrl+C, to stop it.
echo.

python -m src.demo
set "CODE=%ERRORLEVEL%"

if not "%CODE%"=="0" (
  echo.
  echo   The demo dashboard stopped with exit code %CODE%.
  echo   If the message above mentions address 10048 or "only one usage of each
  echo   socket address", another copy is already running on port 8765 - close
  echo   it, or set DASHBOARD_PORT to a free port.
  echo.
  pause
)

endlocal
