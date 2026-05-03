# ContextPilot installer for Windows (PowerShell)
#
# - Checks for uv, installs if missing
# - Checks for Python 3.12+
# - Installs contextpilot as editable package
# - Copies launcher to a location on PATH

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "         ContextPilot -- Installer" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ScriptDir) { $ScriptDir = Get-Location }

# ---- Check uv ----
$uvPath = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvPath) {
    Write-Host "  uv not found. Installing..." -ForegroundColor Yellow
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    } catch {
        Write-Host "  ERROR: Failed to install uv." -ForegroundColor Red
        Write-Host "  See: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    }
    Write-Host "  OK: uv installed" -ForegroundColor Green
} else {
    $uvVersion = & uv --version 2>&1
    Write-Host "  OK: uv found ($uvVersion)" -ForegroundColor Green
}

# ---- Check Python ----
$pythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 12) {
                $pythonCmd = $cmd
                Write-Host "  OK: $ver" -ForegroundColor Green
                break
            }
        }
    } catch { }
}

if (-not $pythonCmd) {
    Write-Host "  ERROR: Python 3.12+ is required." -ForegroundColor Red
    Write-Host "  Download from: https://www.python.org/downloads/"
    exit 1
}

# ---- Install package ----
Write-Host ""
Write-Host "  Installing ContextPilot..."
Push-Location $ScriptDir
try {
    & uv pip install -e .
    Write-Host "  OK: Package installed" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Installation failed: $_" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}

# ---- Copy launcher ----
Write-Host ""
$binDir = Join-Path $env:USERPROFILE "bin"
if (-not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir -Force | Out-Null
}

Copy-Item "$ScriptDir\bin\cpx.cmd" "$binDir\cpx.cmd" -Force
Write-Host "  OK: Launcher installed to $binDir\cpx.cmd" -ForegroundColor Green
Copy-Item "$ScriptDir\bin\cpx-codex.cmd" "$binDir\cpx-codex.cmd" -Force
Write-Host "  OK: Launcher installed to $binDir\cpx-codex.cmd" -ForegroundColor Green

# ---- Add to PATH if needed ----
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$binDir*") {
    Write-Host ""
    Write-Host "  Adding $binDir to user PATH..." -ForegroundColor Yellow
    [Environment]::SetEnvironmentVariable(
        "Path",
        "$binDir;$currentPath",
        "User"
    )
    $env:Path = "$binDir;$env:Path"
    Write-Host "  OK: Added to PATH (restart terminal for full effect)" -ForegroundColor Green
}

# ---- Done ----
Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "  ContextPilot installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "  Usage:"
Write-Host "    cpx C:\path\to\project              # Launch with Claude Code"
Write-Host "    cpx-codex C:\path\to\project        # Launch with Codex CLI"
Write-Host ""
Write-Host "  The MCP server will:"
Write-Host "    1. Scan and index your project"
Write-Host "    2. Start serving on localhost:8090-8099"
Write-Host "    3. Register with your AI coding assistant"
Write-Host "    4. Open a live dashboard"
Write-Host "================================================"
