@echo off
title Alpaca Paper Trading
cd /d "%~dp0"
echo Running paper trading (inference + paper rotation)...
echo.
python -m stock_predictor.main --run
echo.
echo Done. Press any key to close.
pause >nul
