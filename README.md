# ContextPilot

> Production-grade MCP tool that gives AI coding assistants **surgical, semantic context** — not entire files.

ContextPilot extracts symbols (functions, classes, methods) from your codebase, embeds them with **BAAI/bge-small-en-v1.5**, stores vectors in **sqlite-vec**, and serves context through 4 MCP tools. Instead of sending entire files to Claude or Codex, it sends only the relevant symbols — saving **60-80% of input tokens** per turn.

It is designed for real coding loops, not one-off search: each turn can reuse session memory, prioritize exact matches before vector search, and cap read/compression budgets so context stays useful and affordable.

## ✨ Features

- **Semantic retrieval** — KNN search across your entire codebase via embeddings
- **Three-tier resolution** — file path → symbol name → semantic search (fastest path wins)
- **Incremental indexing** — xxhash-based change detection, only re-indexes modified files
- **Cost guard** — token estimation with warn/hard limits and auto-compression
- **Live dashboard** — real-time Chart.js dashboard showing token savings and costs
- **Action graph** — session memory tracks which files you've read and edited
- **Import-neighbor context** — expands one hop from primary matches using import relationships
- **Per-turn read budget** — prevents oversized file dumps via `CTX_READ_BUDGET`
- **Staleness checks** — symbol reads can flag stale cached bodies after file changes
- **Session stats API** — exposes live turn metrics at `/stats` for dashboard polling
- **100% local** — no cloud, no telemetry, all data stays in `.ctxpilot/`
- **4 languages** — Python, JavaScript/TypeScript, and Go via tree-sitter

## 🧭 Why This Works Better Than Full-File Context

In most repos, an assistant rarely needs every line in a file. ContextPilot narrows input to the symbols that are most likely relevant, then enriches with neighboring summaries only when useful.

Retrieval strategy per query:

1. Reuse session memory (files recently read/edited)
2. Try exact file path match
3. Try exact symbol name match
4. Fall back to semantic KNN search

This layered approach improves precision and usually reduces token usage compared to sending full files by default.

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install

```bash
# Clone and install
git clone <repo-url> contextpilot
cd contextpilot
uv pip install -e .

# Or use the installer
./install.sh        # Linux/macOS
.\install.ps1       # Windows
```

### Launch

```bash
# With Claude Code
cpx /path/to/your/project

# With Codex CLI
cpx-codex /path/to/your/project

# Or start the server manually
python -m contextpilot serve /path/to/your/project
```

The server will:
1. Scan and index your project (~2s for a typical project)
2. Start the MCP server on `localhost:8090`
3. Open the live dashboard at `http://localhost:8090/dashboard`

## 🔧 Manual Setup Guide

Use this when you want to run ContextPilot without launcher scripts.

### 1) Install dependencies and package

Linux/macOS:

```bash
cd contextpilot
uv pip install -e .
```

Windows PowerShell:

```powershell
cd contextpilot
uv pip install -e .
```

### 2) (Optional) Run a standalone scan first

```bash
uv run python -m contextpilot scan /path/to/project
```

Windows PowerShell:

```powershell
uv run python -m contextpilot scan C:\path\to\project
```

### 3) Start server manually

```bash
uv run python -m contextpilot serve /path/to/project
```

Windows PowerShell:

```powershell
uv run python -m contextpilot serve C:\path\to\project
```

On startup ContextPilot will:

1. Incrementally scan/index project files
2. Pick the first free port in `8090-8099`
3. Write `.ctxpilot/server.port` and `.ctxpilot/server.pid`
4. Serve MCP endpoint at `http://localhost:<port>/mcp`

### 4) Register MCP endpoint in your assistant CLI

Claude CLI:

```bash
claude mcp add --transport http contextpilot http://localhost:<port>/mcp
```

Codex CLI:

```bash
codex mcp add contextpilot --url http://localhost:<port>/mcp
```

> **Note:** For Claude CLI, the `--transport http` flag is required (it defaults to stdio). For Codex CLI, use `--url` instead.

Cursor / Antigravity:

Search for your `mcp_config.json` file, add the following JSON configuration, and refresh in the **Manage MCP servers** tab:

```json
{
  "mcpServers": {
    "localhost-server": {
      "serverUrl": "http://localhost:<port/mcp"
    }
  }
}
```

### 5) Verify runtime endpoints

- Dashboard: `http://localhost:<port>/dashboard`
- Stats API: `http://localhost:<port>/stats`

## 🛠 MCP Tools

| Tool | Purpose | When to Use |
|---|---|---|
| `ctx_continue(query)` | First tool every turn | Start of each coding turn |
| `ctx_retrieve(query, top_k)` | Direct semantic search | Need specific symbols |
| `ctx_read(file, symbol)` | Read file or symbol | Need full source code |
| `ctx_register_edit(file, summary)` | Track edits | After modifying a file |

### Response Details You Can Rely On

- `ctx_continue` returns `mode`, `confidence`, `resolution`, and token totals
- `ctx_retrieve` returns `primary`, `summaries`, `neighbors`, and token split (`tokens_primary`, `tokens_summary`, `tokens_neighbors`, `tokens_saved`)
- `ctx_read` returns content with budget metadata and stale detection fields
- `ctx_register_edit` re-indexes only the changed file and reports symbol delta

### How `ctx_continue` Works

On each turn, it resolves context through three tiers (in order):

1. **Action graph memory** — files from previous turns matching the query
2. **Exact file path** — `"auth.py"` → all symbols in that file
3. **Exact symbol name** — `"send_notification"` → that specific symbol via SQL
4. **Semantic KNN search** — `"authentication logic"` → nearest neighbors via embeddings

Only reaches semantic search if the first three miss.

## 📊 Architecture

```
┌─────────────────────────────────────────────────┐
│                  MCP Server                      │
│         (FastMCP 3.2.0, streamable-http)         │
├──────────┬──────────┬──────────┬────────────────┤
│ctx_contin│ctx_retri │ctx_read  │ctx_register_   │
│   ue     │   eve    │          │   edit         │
├──────────┴──────────┴──────────┴────────────────┤
│              Retriever + Action Graph            │
│           (three-tier + session memory)           │
├───────────────────┬────────────────────────────-─┤
│    Cost Guard     │      Compressor              │
│  (token budgets)  │  (boilerplate removal)       │
├───────────────────┴──────────────────────────────┤
│              Vector Store (sqlite-vec)            │
│         symbols + vec_symbols (float[384])        │
├───────────────────┬──────────────────────────────┤
│  Symbol Extractor │     Embedder                 │
│  (tree-sitter)    │  (fastembed / bge-small)     │
├───────────────────┴──────────────────────────────┤
│           Scanner + Summarizer                   │
│    (pathspec, xxhash, ThreadPoolExecutor(4))     │
└─────────────────────────────────────────────────┘
```

## 📂 Project Structure

```
contextpilot/
├── pyproject.toml
├── install.sh / install.ps1
├── bin/
│   ├── cpx              # Claude Code launcher (bash)
│   ├── cpx-codex        # Codex CLI launcher (bash)
│   ├── cpx.cmd          # Claude Code launcher (Windows)
│   └── cpx-codex.cmd    # Codex CLI launcher (Windows)
└── contextpilot/
    ├── __main__.py       # CLI dispatcher (serve/scan/stats)
    ├── server.py         # FastMCP server + HTTP endpoints
    ├── retriever.py      # Three-tier resolution + action graph
    ├── extractor.py      # tree-sitter symbol extraction
    ├── embedder.py       # fastembed wrapper (bge-small-en-v1.5)
    ├── store.py          # sqlite-vec vector store
    ├── scanner.py        # Directory walker + incremental indexing
    ├── summarizer.py     # CliffNotes generator (no LLM)
    ├── cost_guard.py     # Token budgets + CLI interrupt
    ├── compressor.py     # Boilerplate removal + compression
    └── dashboard/
        └── index.html    # Live Chart.js dashboard
```

## ⚙️ Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `CTX_PRIMARY_THRESHOLD` | `0.80` | Max distance for full-body results |
| `CTX_SUMMARY_THRESHOLD` | `0.95` | Max distance for CliffNotes results |
| `CTX_WARN_THRESHOLD` | `15000` | Tokens before showing cost warning |
| `CTX_HARD_LIMIT` | `30000` | Tokens before auto-compress |
| `CTX_READ_BUDGET` | `18000` | Max chars per turn for ctx_read |
| `CTX_TURN_GAP_SECONDS` | `3.0` | Seconds of silence before a new turn is detected |
| `CTX_MAX_FILE_SIZE` | `512000` | Max file size to index (bytes) |
| `CTX_COMPRESS_BUDGET` | `15000` | Target tokens after compression |

### Runtime Artifacts

ContextPilot stores project-local runtime data in `.ctxpilot/`:

- `embeddings.db` — symbol and vector index
- `file_hashes.json` — incremental scan hash state
- `action_graph.json` — per-session memory of reads/edits/queries
- `session_stats.json` — token/cost history for dashboard and stats endpoint
- `server.pid` and `server.port` — active server process and selected port

## 🔒 Privacy

- **Zero telemetry** — no data ever leaves your machine
- **No cloud** — embeddings run locally via ONNX (CPU-only)
- **Local storage** — all data in `.ctxpilot/` (auto-gitignored)
- **No API keys** — no external services required

## 📈 Token Savings

Typical savings observed during development:

| Metric | Value |
|---|---|
| Tokens per turn (raw) | ~3,000–5,000 |
| Tokens per turn (sent) | ~400–800 |
| Reduction | **60–80%** |
| Cost savings at $3/1M | ~$0.01–0.02 per turn |

## 🧪 Manual Verification Checklist

1. Run scan twice and confirm second run indexes 0 changed files.
2. Edit one source file and call `ctx_register_edit` to confirm single-file re-index.
3. Query `ctx_retrieve("authentication logic")` and verify relevant symbols appear.
4. Open `/dashboard` and `/stats` and confirm turn stats update during tool usage.

## License

MIT
