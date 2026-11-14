@echo off
title NeuralPulse VIP Pro-Trader Live Scanner
cd /d "%~dp0"
echo ======================================================================
echo   NeuralPulse VIP Pro-Trader Live Scanner (BTC/USDT)
echo   Silently watching live market... Alerts fire ONLY on confirmed setups.
echo ======================================================================
echo.
".venv\Scripts\python.exe" main.py --live
pause

