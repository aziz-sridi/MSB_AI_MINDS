@echo off
echo ============================================================
echo   AI MINDS - Starting Backend + Frontend
echo ============================================================
echo.

REM Start the backend server
echo [1/2] Starting backend server on port 5000...
start "AI MINDS Backend" cmd /k "cd /d %~dp0backend && python server.py"

REM Wait for backend to initialize
timeout /t 5 /nobreak > nul

REM Start the Streamlit frontend
echo [2/2] Starting Streamlit dashboard...
start "AI MINDS Frontend" cmd /k "cd /d %~dp0frontend && streamlit run app.py --server.port 8501"

echo.
echo ============================================================
echo   Backend:    http://127.0.0.1:5000
echo   Frontend:   http://127.0.0.1:8501
echo   Extension:  Load from ai-minds/extension/ in Chrome
echo ============================================================
echo.
echo Press any key to close this window (servers keep running)...
pause > nul
