@echo off
cd /d "%~dp0"
if not exist "logs" mkdir logs
echo Starting NeuralPulse Bot in Background Mode...
powershell -Command "Start-Process -FilePath '.\.venv\Scripts\pythonw.exe' -ArgumentList 'main.py --live' -WorkingDirectory '%~dp0' -WindowStyle Hidden -RedirectStandardOutput 'logs\live_output.log' -RedirectStandardError 'logs\live_error.log'"
echo.
echo [SUCCESS] Bot is now running silently in the background!
echo You can safely close VS Code and this window.
timeout /t 4