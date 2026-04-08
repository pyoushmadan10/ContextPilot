@echo off
REM ContextPilot launcher for Claude Code (Windows)
REM Usage: cpx.cmd [project_path] [initial_prompt]

setlocal enabledelayedexpansion

set "PROJECT_PATH=%~1"
if "%PROJECT_PATH%"=="" set "PROJECT_PATH=."

set "INITIAL_PROMPT=%~2"

REM Resolve absolute path
pushd "%PROJECT_PATH%" 2>nul
if errorlevel 1 (
    echo ERROR: Directory "%PROJECT_PATH%" not found.
    exit /b 1
)
set "PROJECT_PATH=%CD%"
popd

set "CTXPILOT_DIR=%PROJECT_PATH%\.ctxpilot"
set "PID_FILE=%CTXPILOT_DIR%\server.pid"
set "PORT_FILE=%CTXPILOT_DIR%\server.port"
set "LOG_FILE=%CTXPILOT_DIR%\server.log"

echo ================================================
echo          ContextPilot -- Claude Code
echo ================================================
echo.
echo   Project: %PROJECT_PATH%

REM Kill stale server if running
if exist "%PID_FILE%" (
    set /p OLD_PID=<"%PID_FILE%"
    taskkill /PID !OLD_PID! /F >nul 2>&1
    del /f "%PID_FILE%" "%PORT_FILE%" >nul 2>&1
    echo   Stopped stale server.
)

REM Ensure .ctxpilot dir exists
if not exist "%CTXPILOT_DIR%" mkdir "%CTXPILOT_DIR%"

REM Ensure log file exists for tail
if not exist "%LOG_FILE%" type nul > "%LOG_FILE%"

REM Start server in background
echo   Starting ContextPilot...
start /B uv run python -m contextpilot.server "%PROJECT_PATH%" >> "%LOG_FILE%" 2>&1

REM Wait for HTTP server to be reachable (not for scan to complete)
set WAIT_SECS=0
set MAX_WAIT=30

:wait_loop
if exist "%PORT_FILE%" (
    set /p PORT=<"%PORT_FILE%"
    curl -s "http://localhost:!PORT!/health" >nul 2>&1
    if not errorlevel 1 goto server_ready
)
if !WAIT_SECS! GEQ 5 (
    if !WAIT_SECS! EQU 5 echo   Waiting for server to start...
)
timeout /t 1 /nobreak >nul
set /a WAIT_SECS+=1
if %WAIT_SECS% GEQ %MAX_WAIT% (
    echo   ContextPilot server failed to start. Check .ctxpilot\server.log for details.
    exit /b 1
)
goto wait_loop

:server_ready
for /f "tokens=*" %%a in ('curl -s "http://localhost:!PORT!/health"') do set HEALTH_JSON=%%a

set "MCP_URL=http://localhost:%PORT%/mcp"
set "DASHBOARD_URL=http://localhost:%PORT%/dashboard"

echo.
echo   ContextPilot server is up (indexing may continue in background)
echo   Dashboard: %DASHBOARD_URL%
echo.

REM Register MCP server with Claude Code
echo   Registering MCP server with Claude Code...
call claude mcp add contextpilot "%MCP_URL%" >nul 2>&1
if errorlevel 1 (
    echo   Warning: Could not register MCP server with 'claude' CLI.
)

REM Start tail for cost lines in terminal (PowerShell Get-Content -Wait)
set "TAIL_PID_FILE=%CTXPILOT_DIR%\tail.pid"
start /B powershell -NoProfile -Command "$host.UI.RawUI.WindowTitle='cpx-tail'; Get-Content -Path '%LOG_FILE%' -Wait -Tail 0 | Select-String '\[ContextPilot\]' | ForEach-Object { Write-Host $_.Line }" >nul 2>&1

REM Capture tail's PID (last started background process)
for /f "tokens=2" %%p in ('wmic process where "commandline like '%%cpx-tail%%' and name='powershell.exe'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    echo %%p > "%TAIL_PID_FILE%"
)

echo.
echo   Launching Claude Code...
echo   ------------------------------------------------
echo.

if not "%INITIAL_PROMPT%"=="" (
    call claude "%INITIAL_PROMPT%"
) else (
    call claude
)

REM Cleanup
echo.
echo   Stopping ContextPilot server...
if exist "%PID_FILE%" (
    set /p SRV_PID=<"%PID_FILE%"
    taskkill /PID !SRV_PID! /F >nul 2>&1
    del /f "%PID_FILE%" "%PORT_FILE%" >nul 2>&1
)
REM Kill tail process
if exist "%TAIL_PID_FILE%" (
    set /p TAIL_PID=<"%TAIL_PID_FILE%"
    taskkill /PID !TAIL_PID! /F >nul 2>&1
    del /f "%TAIL_PID_FILE%" >nul 2>&1
)
REM Also kill by window title as fallback
taskkill /FI "WINDOWTITLE eq cpx-tail" /F >nul 2>&1
echo   Done.

endlocal
