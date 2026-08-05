@echo off
REM ============================================================
REM  C-OWPF Optimal Water Flow -- one-click web UI launcher
REM  Double-click this file to start the app in your browser.
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo   Starting the C-OWPF Optimal Water Flow UI...
echo   (a browser tab will open at http://localhost:8501)
echo.

REM Prefer the Windows Python launcher; fall back to python on PATH.
where py >nul 2>nul
if %errorlevel%==0 (
    set "PY=py"
) else (
    set "PY=python"
)

REM Make sure Streamlit is installed; install requirements on first run.
%PY% -c "import streamlit" >nul 2>nul
if not %errorlevel%==0 (
    echo   First run: installing dependencies, please wait...
    %PY% -m pip install -r requirements.txt
)

%PY% -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false

echo.
echo   The app has stopped. You can close this window.
pause
endlocal
