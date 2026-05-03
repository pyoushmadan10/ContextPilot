"""Shared pytest fixtures for the ContextPilot test suite.

Key design choices:
- Heavy I/O (fastembed model) is mocked with deterministic fake vectors so
  the suite runs in ~5 seconds.
- Each test that needs a store gets a fresh temp directory — no shared state.
"""

from __future__ import annotations

import numpy as np
import pytest

from contextpilot.extractor import Symbol
from contextpilot.store import VectorStore

# Embedding dimension must match what the real embedder produces
_EMBED_DIM = 384


def _make_symbol(
    name: str = "my_func",
    file_path: str = "src/foo.py",
    kind: str = "function",
    signature: str = "def my_func():",
    docstring: str | None = "Does something.",
    body_preview: str = "def my_func():\n    return 42",
    start_line: int = 1,
    end_line: int = 2,
    body_hash: str = "abc123",
    language: str = "python",
) -> Symbol:
    """Factory for Symbol test objects."""
    return Symbol(
        file_path=file_path,
        language=language,
        name=name,
        kind=kind,
        signature=signature,
        docstring=docstring,
        body_preview=body_preview,
        start_line=start_line,
        end_line=end_line,
        body_hash=body_hash,
    )


def _fake_vector(seed: int = 0) -> np.ndarray:
    """Return a deterministic unit-normalised float32 vector."""
    rng = np.random.default_rng(seed)
    v = rng.random(_EMBED_DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


@pytest.fixture()
def tmp_store(tmp_path):
    """A VectorStore backed by a fresh temporary directory."""
    store = VectorStore(str(tmp_path))
    yield store
    store.close()


@pytest.fixture()
def populated_store(tmp_store):
    """A VectorStore pre-loaded with three symbols across two files."""
    syms = [
        _make_symbol("authenticate", "auth.py", "function",
                     "def authenticate(user, pwd):", "Verify credentials.",
                     "def authenticate(user, pwd):\n    return True", 1, 2),
        _make_symbol("UserModel", "models.py", "class",
                     "class UserModel:", "ORM model for users.",
                     "class UserModel:\n    pass", 5, 10),
        _make_symbol("UserModel.save", "models.py", "method",
                     "def save(self):", "Persist to DB.",
                     "def save(self):\n    db.commit()", 12, 15),
    ]
    for i, sym in enumerate(syms):
        tmp_store.upsert_symbol(sym, _fake_vector(i))
    return tmp_store


# Re-export helpers for test modules
__all__ = ["_make_symbol", "_fake_vector"]
