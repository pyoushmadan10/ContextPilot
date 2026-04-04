#!/usr/bin/env bash
# ContextPilot installer for Linux/macOS
#
# - Checks for uv, installs if missing
# - Checks for Python 3.12+
# - Installs contextpilot as editable package
# - Copies launchers to ~/bin/ and adds to PATH

set -euo pipefail

echo "╔══════════════════════════════════════════╗"
echo "║        ContextPilot — Installer          ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Get script directory (where contextpilot source is)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Check uv ----
if ! command -v uv &>/dev/null; then
    echo -e "${YELLOW}  uv not found. Installing...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        echo -e "${RED}  ERROR: Failed to install uv. Please install manually.${NC}"
        echo "  See: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    echo -e "${GREEN}  ✓ uv installed${NC}"
else
    echo -e "${GREEN}  ✓ uv found: $(uv --version)${NC}"
fi

# ---- Check Python ----
PYTHON_VERSION=$(python3 --version 2>/dev/null || python --version 2>/dev/null || echo "not found")
echo "  Python: $PYTHON_VERSION"

# Check Python version >= 3.12
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
    if ! python -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
        echo -e "${RED}  ERROR: Python 3.12+ is required but not found.${NC}"
        echo "  Please install Python 3.12 or later:"
        echo "    macOS:  brew install python@3.12"
        echo "    Ubuntu: sudo apt install python3.12"
        echo "    Other:  https://www.python.org/downloads/"
        exit 1
    fi
fi
echo -e "${GREEN}  ✓ Python 3.12+ found${NC}"

# ---- Install package ----
echo ""
echo "  Installing ContextPilot..."
cd "$SCRIPT_DIR"
uv pip install -e .
echo -e "${GREEN}  ✓ Package installed${NC}"

# ---- Copy launchers ----
echo ""
BIN_DIR="$HOME/bin"
mkdir -p "$BIN_DIR"

cp "$SCRIPT_DIR/bin/cpx" "$BIN_DIR/cpx"
cp "$SCRIPT_DIR/bin/cpx-codex" "$BIN_DIR/cpx-codex"
chmod +x "$BIN_DIR/cpx" "$BIN_DIR/cpx-codex"
echo -e "${GREEN}  ✓ Launchers installed to $BIN_DIR${NC}"

# ---- Add to PATH if needed ----
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo ""
    echo -e "${YELLOW}  Adding $BIN_DIR to PATH...${NC}"

    SHELL_RC=""
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_RC="$HOME/.bashrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_RC="$HOME/.bash_profile"
    fi

    if [ -n "$SHELL_RC" ]; then
        if ! grep -q "# ContextPilot" "$SHELL_RC" 2>/dev/null; then
            echo "" >> "$SHELL_RC"
            echo "# ContextPilot" >> "$SHELL_RC"
            echo "export PATH=\"\$HOME/bin:\$PATH\"" >> "$SHELL_RC"
            echo -e "${GREEN}  ✓ Added to $SHELL_RC${NC}"
            echo -e "${YELLOW}  Run 'source $SHELL_RC' or restart your terminal.${NC}"
        fi
    else
        echo -e "${YELLOW}  Add this to your shell profile:${NC}"
        echo "    export PATH=\"\$HOME/bin:\$PATH\""
    fi
fi

# ---- Done ----
echo ""
echo "══════════════════════════════════════════"
echo -e "${GREEN}  ContextPilot installed successfully!${NC}"
echo ""
echo "  Usage:"
echo "    cpx /path/to/project           # Launch with Claude Code"
echo "    cpx-codex /path/to/project     # Launch with Codex CLI"
echo ""
echo "  The MCP server will:"
echo "    1. Scan and index your project"
echo "    2. Start serving on localhost:8090-8099"
echo "    3. Register with your AI coding assistant"
echo "    4. Open a live dashboard"
echo "══════════════════════════════════════════"
