"""FastMCP server entry point for ContextPilot.

Exposes 4 MCP tools to Claude Code and Codex CLI:
- ctx_continue: First tool called every turn (three-tier resolution)
- ctx_retrieve: Direct semantic search
- ctx_read: File/symbol reader with budget enforcement
- ctx_register_edit: Edit tracking + incremental re-index

Also serves /stats and /dashboard HTTP endpoints.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

from fastmcp import FastMCP

from contextpilot.compressor import Compressor
from contextpilot.cost_guard import CostGuard, SessionSavings, estimate_tokens, estimate_cost_usd
from contextpilot.retriever import Retriever, count_tokens
from contextpilot.scanner import scan_project
from contextpilot.store import VectorStore

# ---------------------------------------------------------------------------
# Server configuration
# ---------------------------------------------------------------------------

PORT_RANGE_START = 8090
PORT_RANGE_END = 8099


def _find_free_port() -> int:
    """Find a free port in the configured range."""
    for port in range(PORT_RANGE_START, PORT_RANGE_END + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue
    # Fallback: let OS pick
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Global state — initialized in main()
# ---------------------------------------------------------------------------

_store: VectorStore | None = None
_retriever: Retriever | None = None
_session_stats: SessionSavings | None = None
_cost_guard: CostGuard | None = None
_compressor: Compressor | None = None
_project_root: str = ""
_server_status: str = "initializing"  # initializing → indexing → ready | error
_server_status_detail: str = ""

# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("contextpilot")


@mcp.tool()
def ctx_continue(query: str) -> dict:
    """First tool to call every turn. Provides surgical context for your query.

    Uses three-tier resolution:
    1. Session memory (files from previous turns)
    2. Exact file path match
    3. Exact symbol name match
    4. Semantic KNN search (fallback)

    Args:
        query: Natural language description of what you need context for.

    Returns:
        Context result with mode, confidence, matching symbols, and token count.
    """
    if _server_status != "ready" or _retriever is None:
        return {"error": f"Server is still indexing the project ({_server_status}). Please wait a moment and try again."}

    result = _retriever.ctx_continue(query)

    if _cost_guard:
        # Build context text for cost guard check
        context_parts = [s.get("body_preview", "") for s in result.get("symbols", [])]
        context_text = "\n\n".join(context_parts)
        guard_result = _cost_guard.check(context_text)

        result["guard"] = {
            "action": guard_result.action,
            "token_count": guard_result.token_count,
            "estimated_cost_usd": guard_result.estimated_cost_usd,
            "message": guard_result.message,
        }

        # If over hard limit, auto-compress if possible
        if guard_result.action == "interrupt" and _compressor:
            compressed = _compressor.compress(result.get("symbols", []))
            result["compressed"] = {
                "original_tokens": compressed["original_tokens"],
                "compressed_tokens": compressed["compressed_tokens"],
                "tokens_saved": compressed["tokens_saved"],
                "quality_report": compressed["quality_report"],
            }

    if _session_stats:
        traversal = result.pop("_traversal", None)
        if traversal:
            _session_stats.record_traversal(traversal)
            _session_stats.record_turn(
                traversal.get("total_tokens_raw", 0),
                traversal.get("total_tokens_sent", 0),
                _retriever.action_graph.turn,
            )
            saved_tok = traversal.get("total_tokens_raw", 0) - traversal.get("total_tokens_sent", 0)
            cost_saved = traversal.get("cost_raw_would_have_been_usd", 0) - traversal.get("cost_this_turn_usd", 0)
            print(
                f"[ContextPilot] Turn {traversal.get('turn', '?')} \u2014 "
                f"sent {traversal.get('total_tokens_sent', 0):,} tok "
                f"(${traversal.get('cost_this_turn_usd', 0):.3f}) \u00b7 "
                f"saved {saved_tok:,} tok (${cost_saved:.3f})",
                file=sys.stderr, flush=True,
            )

    return result


@mcp.tool()
def ctx_retrieve(query: str, top_k: int = 10) -> dict:
    """Semantic search for symbols related to your query.

    Embeds the query and finds the most relevant symbols via KNN search.
    Results are tiered:
    - Primary: High confidence matches with full body preview
    - Summaries: Medium confidence with CliffNotes only

    Args:
        query: Natural language description of what you're looking for.
        top_k: Maximum number of results to return (default 10).

    Returns:
        Retrieval result with primary symbols, summaries, and token counts.
    """
    if _server_status != "ready" or _retriever is None:
        return {"error": f"Server is still indexing the project ({_server_status}). Please wait a moment and try again."}

    result = _retriever.ctx_retrieve(query, top_k)

    # Attach cost info
    total_tokens = (
        result.get("tokens_primary", 0)
        + result.get("tokens_summary", 0)
        + result.get("tokens_neighbors", 0)
    )
    result["cost"] = {
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimate_cost_usd(total_tokens),
        "tokens_saved": result.get("tokens_saved", 0),
    }

    return resultmcp.tool()
def ctx_read(file_path: str, symbol_name: str | None = None) -> dict:
    """Read a file or specific symbol from the project.

    If symbol_name is provided, returns only that symbol's body (not the
    whole file), saving tokens.

    Enforces a per-turn read budget (default 18,000 chars) to prevent
    excessive context consumption.

    Args:
        file_path: Relative path from project root.
        symbol_name: Optional specific symbol to read (e.g., 'AuthService.login').

    Returns:
        Read result with content, token count, staleness indicator, and budget remaining.
    """
    if _server_status != "ready" or _retriever is None:
        return {"error": f"Server is still indexing the project ({_server_status}). Please wait a moment and try again."}

    result = _retriever.ctx_read(file_path, symbol_name)
    
    return result


@mcp.tool()
def ctx_register_edit(file_path: str, summary: str | None = None) -> dict:
    """Register that a file was edited. Triggers incremental re-indexing.

    Call this after editing a file so ContextPilot can update its index
    and keep future context retrievals accurate.

    Args:
        file_path: Relative path of the edited file from project root.
        summary: Optional brief description of what changed.

    Returns:
        Edit result with reindex status and symbol count changes.
    """
    if _server_status != "ready" or _retriever is None:
        return {"error": f"Server is still indexing the project ({_server_status}). Please wait a moment and try again."}

    return _retriever.ctx_register_edit(file_path, summary)


# ---------------------------------------------------------------------------
# HTTP endpoints (non-MCP)
# ---------------------------------------------------------------------------

@mcp.custom_route("/health", methods=["GET"])
async def health_endpoint(request):
    """Health check for launcher script."""
    from starlette.responses import JSONResponse
    
    symbols_indexed = 0
    if _store:
        try:
            stats = _store.get_stats()
            symbols_indexed = stats.get("total_symbols", 0)
        except Exception:
            pass
        
    port_val = getattr(mcp, "port", 8090)
    try:
        if _project_root:
            pf = Path(_project_root) / ".ctxpilot" / "server.port"
            if pf.exists():
                port_val = int(pf.read_text().strip())
    except Exception:
        pass

    return JSONResponse({
        "status": _server_status,
        "ready": _server_status == "ready",
        "detail": _server_status_detail,
        "symbols_indexed": symbols_indexed,
        "port": port_val
    })

@mcp.custom_route("/stats", methods=["GET"])
async def stats_endpoint(request):
    """Return session stats as JSON."""
    from starlette.responses import JSONResponse

    if _session_stats:
        # Add index stats
        data = dict(_session_stats.data)
        if _store:
            data["index"] = _store.get_stats()
            
        if _retriever:
            top_files = []
            for f in _retriever.action_graph.get_top_files(5):
                fp = f["file_path"]
                # Extract filename, cross-platform
                filename = fp.split('/')[-1].split('\\')[-1]
                top_files.append({"file": filename, "access_count": f["access_count"]})
            data["top_files"] = top_files

        # Rename 'per_turn' array to 'turns' and save raw count as 'turns_count'
        if "per_turn" in data:
            data["turns_count"] = data.get("turns", 0)
            data["turns"] = data.pop("per_turn")

        # Add traversal cost aggregates
        data["session_cost_usd"] = round(_session_stats.session_cost_usd, 6)
        data["session_cost_raw_would_have_been_usd"] = round(
            _session_stats.session_cost_raw_would_have_been_usd, 6
        )

        return JSONResponse(data)
    return JSONResponse({"error": "No session stats available"})


@mcp.custom_route("/dashboard", methods=["GET"])
async def dashboard_endpoint(request):
    """Serve the single-file HTML dashboard."""
    from starlette.responses import HTMLResponse

    dashboard_path = Path(__file__).parent / "dashboard" / "index.html"
    if dashboard_path.exists():
        html = dashboard_path.read_text(encoding="utf-8")
        return HTMLResponse(html)
    return HTMLResponse("<h1>Dashboard not found</h1><p>Build it in Phase 6.</p>")


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def _background_scan(project_root: str, store: VectorStore, ctxpilot_dir: Path):
    """Run scanning and indexing in a background thread."""
    global _retriever, _session_stats, _cost_guard, _compressor
    global _server_status, _server_status_detail

    try:
        _server_status = "indexing"
        _server_status_detail = "Scanning project files..."
        print("[ctxpilot] Running initial scan...", file=sys.stderr)

        scan_result = scan_project(project_root, store)
        print(
            f"[ctxpilot] Scan complete: {scan_result.files_indexed} files, "
            f"{scan_result.symbols_total} symbols in {scan_result.duration_seconds:.2f}s",
            file=sys.stderr,
        )

        _server_status_detail = "Initializing retriever..."
        _retriever = Retriever(store, project_root)
        _cost_guard = CostGuard()
        _compressor = Compressor()
        _session_stats = SessionSavings(ctxpilot_dir)

        _server_status = "ready"
        _server_status_detail = f"{scan_result.symbols_total} symbols indexed"
        print("[ctxpilot] Server is ready.", file=sys.stderr)

    except Exception as e:
        _server_status = "error"
        _server_status_detail = str(e)
        print(f"[ctxpilot] Background scan failed: {e}", file=sys.stderr)


def main():
    """Main entry point: start HTTP server immediately, scan in background."""
    global _store, _project_root, _server_status, _server_status_detail

    if len(sys.argv) < 2:
        print("Usage: python -m contextpilot.server <project_root>", file=sys.stderr)
        print("  Scans the project and starts the MCP server.", file=sys.stderr)
        sys.exit(1)

    _project_root = os.path.abspath(sys.argv[1])

    if not os.path.isdir(_project_root):
        print(f"[ctxpilot] Error: {_project_root} is not a directory.", file=sys.stderr)
        sys.exit(1)

    ctxpilot_dir = Path(_project_root) / ".ctxpilot"
    ctxpilot_dir.mkdir(exist_ok=True)

    # Initialize store (fast — just opens/creates the DB)
    print(f"[ctxpilot] Initializing store for {_project_root}...", file=sys.stderr)
    _store = VectorStore(_project_root)

    # Find free port and write PID/port files BEFORE scanning
    port = _find_free_port()

    pid_file = ctxpilot_dir / "server.pid"
    pid_file.write_text(str(os.getpid()), encoding="utf-8")

    port_file = ctxpilot_dir / "server.port"
    port_file.write_text(str(port), encoding="utf-8")

    print(f"[ctxpilot] Starting MCP server on localhost:{port}...", file=sys.stderr)
    print(f"[ctxpilot] Dashboard: http://localhost:{port}/dashboard", file=sys.stderr)
    print(f"[ctxpilot] Stats API: http://localhost:{port}/stats", file=sys.stderr)

    # Start scanning in background thread
    _server_status = "initializing"
    _server_status_detail = "Starting background scan..."
    scan_thread = threading.Thread(
        target=_background_scan,
        args=(_project_root, _store, ctxpilot_dir),
        daemon=True,
        name="ctxpilot-scanner",
    )
    scan_thread.start()
    print("[ctxpilot] Background scan started. Server is accepting connections.", file=sys.stderr)

    # Cleanup handler
    def _cleanup(signum=None, frame=None):
        print("\n[ctxpilot] Shutting down...", file=sys.stderr)
        if _session_stats:
            _session_stats.save()
        if _store:
            _store.close()
        if pid_file.exists():
            pid_file.unlink()
        if port_file.exists():
            port_file.unlink()
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    # Start HTTP server immediately — /health is available right away
    try:
        mcp.run(transport="streamable-http", host="localhost", port=port)
    except KeyboardInterrupt:
        _cleanup()
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
