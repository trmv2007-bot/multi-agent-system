@echo off
setlocal
cd /d "%~dp0"

title Multi-agent dashboard

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

if not exist ".env" (
  echo.
  echo   No .env file here, so the agents have no model key.
  echo   Copy .env.example to .env and fill in KEYPOOL_API_KEY, then run this
  echo   file again.
  echo.
  pause
  exit /b 1
)

echo.
echo   Starting the multi-agent dashboard...
echo   It opens your browser at http://127.0.0.1:8765
echo   Close this window, or press Ctrl+C, to stop it.
echo.

python -m src.dashboard.server
set "CODE=%ERRORLEVEL%"

if not "%CODE%"=="0" (
  echo.
  echo   The dashboard stopped with exit code %CODE%.
  echo   If the message above mentions address 10048 or "only one usage of each
  echo   socket address", another copy is already running on port 8765 - close
  echo   it, or set DASHBOARD_PORT in .env to a free port.
  echo.
  pause
)

endlocal
