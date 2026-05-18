@echo off
cd /d "%~dp0"

echo Installing dependencies...
python -m pip install -r requirements.txt

echo Starting Flask...
cd backend

start "" cmd /k python app.py

timeout /t 3 >nul

start http://127.0.0.1:5000