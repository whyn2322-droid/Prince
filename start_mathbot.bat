@echo off
setlocal

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

start "Rasa Server" cmd /k "cd /d %ROOT% && call .venv\Scripts\activate.bat && rasa run --enable-api --cors \"*\""
start "Rasa Actions" cmd /k "cd /d %ROOT% && call .venv\Scripts\activate.bat && rasa run actions"
start "MathBot Web" cmd /k "cd /d %ROOT%\web && python -m http.server 8080 --bind 127.0.0.1"

echo Open http://127.0.0.1:8080 in your browser.
endlocal
