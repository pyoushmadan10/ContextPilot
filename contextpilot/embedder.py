"""FastEmbed wrapper for symbol and query embedding.

Uses BAAI/bge-small-en-v1.5 (384-dim, ONNX, CPU-only) via fastembed.
Model is loaded once and cached by FastEmbed in ~/.cache/fastembed/.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from contextpilot.extractor import Symbol

# ---------------------------------------------------------------------------
# Lazy model loading — initialized on first use
# ---------------------------------------------------------------------------

_MODEL = None
_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _get_model():
    """Load the embedding model, cached after first call."""
    global _MODEL
    if _MODEL is None:
        try:
            from fastembed import TextEmbedding
            print(f"[ctxpilot] Loading embedding model: {_MODEL_NAME}...", file=sys.stderr)
            _MODEL = TextEmbedding(model_name=_MODEL_NAME)
            print("[ctxpilot] Embedding model loaded.", file=sys.stderr)
        except Exception as e:
            print(f"[ctxpilot] Error loading embedding model: {e}", file=sys.stderr)
            raise
    return _MODEL


# ---------------------------------------------------------------------------
# Symbol → text representation for embedding
# ---------------------------------------------------------------------------

def _symbol_to_text(symbol: Symbol) -> str:
    """Convert a Symbol to its text representation for embedding.

    Format: "{name}: {signature}\n{docstring}"
    Strips None values gracefully.
    """
    parts = []

    # Name and signature
    sig = symbol.signature or ""
    parts.append(f"{symbol.name}: {sig}")

    # Docstring if present
    if symbol.docstring:
        parts.append(symbol.docstring)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_symbol(symbol: Symbol) -> np.ndarray:
    """Embed a single symbol into a 384-dimensional vector.

    Concatenates "{name}: {signature}\n{docstring}" as input text.
    """
    model = _get_model()
    text = _symbol_to_text(symbol)
    # fastembed returns a generator, take the first result
    embeddings = list(model.embed([text]))
    return np.array(embeddings[0], dtype=np.float32)


def embed_symbols(symbols: list[Symbol]) -> list[np.ndarray]:
    """Embed multiple symbols using FastEmbed's internal batching.

    More efficient than calling embed_symbol() in a loop.
    Returns a list of 384-dim numpy arrays in the same order as input.
    """
    if not symbols:
        return []

    model = _get_model()
    texts = [_symbol_to_text(s) for s in symbols]
    # fastembed handles batching internally
    embeddings = list(model.embed(texts))
    return [np.array(e, dtype=np.float32) for e in embeddings]


def embed_query(query: str) -> np.ndarray:
    """Embed a search query into a 384-dimensional vector.

    Prefixes with "query: " as required by bge retrieval models
    for asymmetric search (query vs document).
    """
    model = _get_model()
    prefixed = f"query: {query}"
    embeddings = list(model.embed([prefixed]))
    return np.array(embeddings[0], dtype=np.float32)


# ---------------------------------------------------------------------------
# CLI entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick sanity check
    from contextpilot.extractor import Symbol

    test_symbol = Symbol(
        file_path="test.py",
        language="python",
        name="validate_jwt",
        kind="function",
        signature="def validate_jwt(token: str, secret: str) -> bool:",
        docstring="Verifies JWT signature and expiry.",
        body_preview="def validate_jwt(token, secret):\n    decoded = jwt.decode(token, secret)",
        start_line=1,
        end_line=10,
        body_hash="abc123",
    )

    print(f"Symbol text: {_symbol_to_text(test_symbol)}", file=sys.stderr)

    vec = embed_symbol(test_symbol)
    print(f"Symbol embedding shape: {vec.shape}", file=sys.stderr)
    print(f"Symbol embedding dtype: {vec.dtype}", file=sys.stderr)
    print(f"Symbol embedding norm: {np.linalg.norm(vec):.4f}", file=sys.stderr)

    qvec = embed_query("authentication logic")
    print(f"Query embedding shape: {qvec.shape}", file=sys.stderr)

    # Cosine similarity
    similarity = np.dot(vec, qvec) / (np.linalg.norm(vec) * np.linalg.norm(qvec))
    print(f"Cosine similarity (jwt vs 'authentication logic'): {similarity:.4f}", file=sys.stderr)

    print("\n[OK] Embedder working correctly.", file=sys.stderr)
