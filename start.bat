@echo off
setlocal enabledelayedexpansion

REM ---------------------------------------
REM One-click start: backend + frontend
REM Requirements: Python 3, Node.js, API_KEY set in env or .env
REM ---------------------------------------

set "ROOT=%~dp0"
cd /d "%ROOT%"

REM ---- check python ----
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    pause
    exit /b 1
)

REM ---- create venv if missing ----
if not exist "venv\Scripts\python.exe" (
    echo [Backend] Creating venv...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
)

call "venv\Scripts\activate.bat"

REM ---- install backend deps ----
echo [Backend] Installing dependencies...
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt

REM ---- start backend ----
echo [Backend] Starting FastAPI...
start "FDC Backend" cmd /k "cd /d \"%ROOT%\" && call venv\Scripts\activate.bat && python server.py"

REM ---- frontend ----
if not exist "%ROOT%frontend" (
    echo [ERROR] frontend directory not found.
    goto end
)
pushd "%ROOT%frontend"

where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm not found in PATH.
    popd
    goto end
)

if not exist "node_modules" (
    echo [Frontend] Installing npm packages...
    npm install
)

echo [Frontend] Starting Vite dev server...
start "FDC Frontend" cmd /k "cd /d \"%ROOT%frontend\" && npm run dev"
popd

:end
echo.
echo Backend: http://127.0.0.1:8000
echo Frontend: http://localhost:5173
echo Close the started command windows to stop.
echo.
pause
