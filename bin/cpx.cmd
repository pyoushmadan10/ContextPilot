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

REM Start server in background
echo   Starting ContextPilot server...
start /B "ctxpilot-server" uv run python -m contextpilot.server "%PROJECT_PATH%"

REM Wait for port file to appear
set WAIT_SECS=0
set MAX_WAIT=60

:wait_loop
if exist "%PORT_FILE%" goto server_ready
timeout /t 1 /nobreak >nul
set /a WAIT_SECS+=1
if %WAIT_SECS% GEQ %MAX_WAIT% (
    echo   ERROR: Server didn't start within %MAX_WAIT%s.
    exit /b 1
)
goto wait_loop

:server_ready
set /p PORT=<"%PORT_FILE%"
set "MCP_URL=http://localhost:%PORT%/mcp"
set "DASHBOARD_URL=http://localhost:%PORT%/dashboard"

echo.
echo   Server running on port %PORT%
echo   Dashboard: %DASHBOARD_URL%
echo.

REM Register MCP server with Claude Code
echo   Registering MCP server with Claude Code...
claude mcp add contextpilot "%MCP_URL%" >nul 2>&1
if errorlevel 1 (
    echo   Warning: Could not register MCP server with 'claude' CLI.
)

echo.
echo   Launching Claude Code...
echo   ------------------------------------------------
echo.

if not "%INITIAL_PROMPT%"=="" (
    claude "%INITIAL_PROMPT%"
) else (
    claude
)

REM Cleanup
echo.
echo   Stopping ContextPilot server...
if exist "%PID_FILE%" (
    set /p SRV_PID=<"%PID_FILE%"
    taskkill /PID !SRV_PID! /F >nul 2>&1
    del /f "%PID_FILE%" "%PORT_FILE%" >nul 2>&1
)
echo   Done.

endlocal
