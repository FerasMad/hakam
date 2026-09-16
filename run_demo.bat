@echo off
rem Hakam local demo: starts the API and the website, then opens the browser.
rem Needs weights\final.pt, weights\thresholds.json, .env, and "npm ci" run once in web\.
cd /d "%~dp0"

if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat

if not exist weights\final.pt (
  echo weights\final.pt is missing - see weights\README.md
  pause
  exit /b 1
)

if not exist web\public\samples\incident-88.mp4 python scripts\copy_samples.py

start "Hakam API" cmd /k python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
start "Hakam website" cmd /k "cd web && npm run dev"

echo Waiting for the servers to start...
timeout /t 15 /nobreak >nul
start "" http://localhost:3000
