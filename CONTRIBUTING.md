# Contributing to ContextPilot

Thank you for your interest in contributing! ContextPilot is an open-source MCP server that gives AI coding assistants surgical, semantic context about a codebase. All contributions — bug fixes, new language support, performance improvements, and documentation — are welcome.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Development Setup](#development-setup)
- [Running the Test Suite](#running-the-test-suite)
- [Project Structure](#project-structure)
- [Coding Conventions](#coding-conventions)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Issue Labels](#issue-labels)
- [License](#license)

---

## Project Overview

ContextPilot indexes your project's symbols (functions, classes, methods) using tree-sitter, embeds them with BAAI/bge-small-en-v1.5, stores vectors in sqlite-vec, and serves context through 4 MCP tools. The core retrieval pipeline resolves context in three tiers:

1. **Action graph memory** — files from previous turns
2. **Exact file path / symbol name match** — fast SQL lookup
3. **Semantic KNN search** — embedding similarity

Everything runs locally — no cloud, no telemetry.

---

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended)

### Steps

```bash
# 1. Fork and clone the repo
git clone https://github.com/<your-username>/ContextPilot.git
cd ContextPilot/contextpilot

# 2. Create a virtual environment
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install the package and dev dependencies
uv pip install -e ".[dev]"

# 4. Verify everything works
pytest tests/ -v
```

> **Tip:** `uv pip install -e ".[dev]"` installs `contextpilot` in editable mode plus `pytest` and `pytest-cov`. You do **not** need to install `fastembed` just to run the test suite — the tests mock the embedding layer.

---

## Running the Test Suite

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run a single test file
pytest tests/test_store.py -v

# Run with coverage report
pytest tests/ --cov=contextpilot --cov-report=term-missing

# Run tests matching a keyword
pytest tests/ -k "action_graph"
```

### Test Design

- Tests are **fast** (~5 seconds total). Heavy components (fastembed model, full MCP server) are mocked with deterministic fake vectors.
- Each test gets its own `tmp_path` — no shared state.
- Tests live in `tests/` at the package root.

### CI

Every push and pull request to `main` automatically runs the full test suite via GitHub Actions. PRs cannot be merged if tests fail.

---

## Project Structure

```
contextpilot/
├── contextpilot/          # Main package
│   ├── server.py          # FastMCP server + HTTP endpoints
│   ├── retriever.py       # Three-tier resolution + action graph
│   ├── store.py           # sqlite-vec vector store
│   ├── extractor.py       # tree-sitter symbol extraction
│   ├── embedder.py        # fastembed wrapper
│   ├── scanner.py         # Incremental file indexing
│   ├── summarizer.py      # CliffNotes generator (no LLM)
│   ├── cost_guard.py      # Token budgets
│   ├── compressor.py      # Boilerplate removal
│   └── session_manager.py # Session state
├── tests/                 # Test suite
│   ├── conftest.py        # Shared fixtures
│   ├── test_store.py
│   ├── test_retriever.py
│   ├── test_summarizer.py
│   ├── test_compressor.py
│   └── test_cost_guard.py
├── .github/workflows/
│   └── tests.yml          # CI definition
├── pyproject.toml
└── README.md
```

---

## Coding Conventions

### Python Style

- **Type hints** on all public functions and method signatures.
- **Docstrings** on all public classes and functions (single-sentence minimum).
- Use `from __future__ import annotations` at the top of every module for forward-reference compatibility.
- Format with standard Python conventions (PEP 8). Line length: 100 characters.

### Dependencies

- **Do not add new runtime dependencies without opening an issue first.** ContextPilot is intentionally dependency-light to keep installation fast.
- Dev-only dependencies (linters, testing tools) go in `[project.optional-dependencies] dev` in `pyproject.toml`.

### Testing

- Every new feature or bug fix must include a corresponding test.
- Tests must not make network requests or start the MCP server.
- Mock `embed_query` / `embed_symbols` using `unittest.mock.patch` or the `_fake_vector` helper in `conftest.py`.

### Commit Messages

Use the imperative mood and be concise:

```
Add Go method receiver parsing to extractor
Fix budget overflow in ctx_read when symbol body is large
Refactor ActionGraph._load to handle corrupted JSON
```

---

## Submitting a Pull Request

1. **Open an issue first** for non-trivial changes so we can discuss the approach.
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes, keeping commits focused.
4. Add or update tests to cover your change.
5. Run the full test suite locally: `pytest tests/ -v`
6. Push and open a PR against `main`.

### PR Checklist

- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] New code has tests
- [ ] New public functions/classes have docstrings
- [ ] Type hints added to new function signatures
- [ ] No new runtime dependencies added without prior discussion
- [ ] PR description explains **what** changed and **why**

---

## Issue Labels

| Label | Meaning |
|-------|---------|
| `bug` | Something is broken |
| `enhancement` | New feature or improvement |
| `language-support` | Adding a new language (Rust, Java, etc.) |
| `performance` | Indexing/retrieval speed improvements |
| `documentation` | Docs, README, CONTRIBUTING |
| `good first issue` | Beginner-friendly scope |
| `help wanted` | Community input needed |

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE) that covers this project.
