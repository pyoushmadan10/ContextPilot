"""sqlite-vec vector store for symbol embeddings.

Creates and manages .ctxpilot/embeddings.db with:
- symbols table: metadata for each extracted symbol
- vec_symbols: vec0 virtual table for KNN similarity search (float[384])
- summaries table: cached CliffNotes per file
"""

from __future__ import annotations

import os
import sqlite3
import struct
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sqlite_vec

if TYPE_CHECKING:
    from contextpilot.extractor import Symbol

# ---------------------------------------------------------------------------
# Retrieval thresholds — distance-based (lower = more similar)
#
# sqlite-vec returns distances where 0 = identical. The thresholds below
# are maximum distance values:
#   - PRIMARY: results closer than this get full body returned
#   - SUMMARY: results closer than this (but above primary) get CliffNotes
#   - Beyond summary threshold: dropped entirely
#
# Tuned from empirical testing with BAAI/bge-small-en-v1.5.
# ---------------------------------------------------------------------------

CTX_PRIMARY_THRESHOLD = float(os.environ.get("CTX_PRIMARY_THRESHOLD", "0.80"))
CTX_SUMMARY_THRESHOLD = float(os.environ.get("CTX_SUMMARY_THRESHOLD", "0.95"))

# Embedding dimension
_EMBED_DIM = 384


def _serialize_vector(vec: np.ndarray) -> bytes:
    """Serialize a numpy float32 vector to raw bytes for sqlite-vec."""
    return vec.astype(np.float32).tobytes()


def _deserialize_vector(blob: bytes) -> np.ndarray:
    """Deserialize raw bytes back to a numpy float32 vector."""
    return np.frombuffer(blob, dtype=np.float32).copy()


class VectorStore:
    """sqlite-vec backed vector store for symbol embeddings and summaries."""

    def __init__(self, project_root: str):
        """Initialize the store, creating .ctxpilot/ and DB if needed.

        Args:
            project_root: Absolute path to the project root directory.
        """
        self.project_root = Path(project_root).resolve()
        self.ctxpilot_dir = self.project_root / ".ctxpilot"
        self.db_path = self.ctxpilot_dir / "embeddings.db"

        # Ensure .ctxpilot directory exists
        self.ctxpilot_dir.mkdir(exist_ok=True)

        # Ensure .ctxpilot is in .gitignore
        self._ensure_gitignore()

        # Connect and initialize schema
        self._conn = self._connect()
        self._init_schema()

    def _ensure_gitignore(self):
        """Add .ctxpilot/ to .gitignore if not already present."""
        gitignore_path = self.project_root / ".gitignore"
        marker = ".ctxpilot/"

        if gitignore_path.exists():
            content = gitignore_path.read_text(encoding="utf-8", errors="replace")
            if marker in content:
                return
            # Append to existing .gitignore
            with open(gitignore_path, "a", encoding="utf-8") as f:
                if not content.endswith("\n"):
                    f.write("\n")
                f.write(f"\n# ContextPilot runtime data\n{marker}\n")
        else:
            # Create .gitignore
            gitignore_path.write_text(
                f"# ContextPilot runtime data\n{marker}\n",
                encoding="utf-8",
            )

    def _connect(self) -> sqlite3.Connection:
        """Create a connection with sqlite-vec extension loaded."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self):
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                language TEXT NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                signature TEXT NOT NULL DEFAULT '',
                docstring TEXT,
                body_preview TEXT NOT NULL DEFAULT '',
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                body_hash TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_symbols_file_path
                ON symbols(file_path);

            CREATE INDEX IF NOT EXISTS idx_symbols_name
                ON symbols(name);

            CREATE TABLE IF NOT EXISTS summaries (
                file_path TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS imports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT NOT NULL,
                imported_file TEXT NOT NULL,
                UNIQUE(source_file, imported_file)
            );
        """)

        # Create vec0 virtual table for vector similarity search
        # Using float[384] for full precision
        self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_symbols
            USING vec0(
                symbol_id INTEGER PRIMARY KEY,
                embedding float[{_EMBED_DIM}]
            )
        """)

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_imports_source
            ON imports(source_file)
        """)

        self._conn.commit()

    # ------------------------------------------------------------------
    # Symbol CRUD
    # ------------------------------------------------------------------

    def upsert_symbol(self, symbol: Symbol, vector: np.ndarray):
        """Insert or update a symbol and its embedding vector.

        If a symbol with the same file_path + name + kind already exists,
        it is replaced.
        """
        cursor = self._conn.cursor()

        # Check for existing symbol
        cursor.execute(
            "SELECT id FROM symbols WHERE file_path = ? AND name = ? AND kind = ?",
            (symbol.file_path, symbol.name, symbol.kind),
        )
        existing = cursor.fetchone()

        if existing:
            symbol_id = existing[0]
            # Update metadata
            cursor.execute("""
                UPDATE symbols SET
                    language = ?, signature = ?, docstring = ?,
                    body_preview = ?, start_line = ?, end_line = ?,
                    body_hash = ?
                WHERE id = ?
            """, (
                symbol.language, symbol.signature, symbol.docstring,
                symbol.body_preview, symbol.start_line, symbol.end_line,
                symbol.body_hash, symbol_id,
            ))
            # Update vector — delete and re-insert in vec0
            cursor.execute("DELETE FROM vec_symbols WHERE symbol_id = ?", (symbol_id,))
        else:
            # Insert new symbol
            cursor.execute("""
                INSERT INTO symbols
                    (file_path, language, name, kind, signature, docstring,
                     body_preview, start_line, end_line, body_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol.file_path, symbol.language, symbol.name, symbol.kind,
                symbol.signature, symbol.docstring, symbol.body_preview,
                symbol.start_line, symbol.end_line, symbol.body_hash,
            ))
            symbol_id = cursor.lastrowid

        # Insert vector
        cursor.execute(
            "INSERT INTO vec_symbols (symbol_id, embedding) VALUES (?, ?)",
            (symbol_id, _serialize_vector(vector)),
        )

        self._conn.commit()

    def upsert_symbols_batch(self, symbols: list[Symbol], vectors: list[np.ndarray]):
        """Batch upsert symbols and their vectors. More efficient than looping."""
        if len(symbols) != len(vectors):
            raise ValueError("symbols and vectors must have the same length")

        cursor = self._conn.cursor()

        for symbol, vector in zip(symbols, vectors):
            # Check for existing
            cursor.execute(
                "SELECT id FROM symbols WHERE file_path = ? AND name = ? AND kind = ?",
                (symbol.file_path, symbol.name, symbol.kind),
            )
            existing = cursor.fetchone()

            if existing:
                symbol_id = existing[0]
                cursor.execute("""
                    UPDATE symbols SET
                        language = ?, signature = ?, docstring = ?,
                        body_preview = ?, start_line = ?, end_line = ?,
                        body_hash = ?
                    WHERE id = ?
                """, (
                    symbol.language, symbol.signature, symbol.docstring,
                    symbol.body_preview, symbol.start_line, symbol.end_line,
                    symbol.body_hash, symbol_id,
                ))
                cursor.execute("DELETE FROM vec_symbols WHERE symbol_id = ?", (symbol_id,))
            else:
                cursor.execute("""
                    INSERT INTO symbols
                        (file_path, language, name, kind, signature, docstring,
                         body_preview, start_line, end_line, body_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol.file_path, symbol.language, symbol.name, symbol.kind,
                    symbol.signature, symbol.docstring, symbol.body_preview,
                    symbol.start_line, symbol.end_line, symbol.body_hash,
                ))
                symbol_id = cursor.lastrowid

            cursor.execute(
                "INSERT INTO vec_symbols (symbol_id, embedding) VALUES (?, ?)",
                (symbol_id, _serialize_vector(vector)),
            )

        self._conn.commit()

    def delete_symbols_for_file(self, file_path: str):
        """Delete all symbols and vectors for a given file path.

        Called before re-indexing a changed file.
        """
        cursor = self._conn.cursor()

        # Get symbol IDs for this file
        cursor.execute("SELECT id FROM symbols WHERE file_path = ?", (file_path,))
        ids = [row[0] for row in cursor.fetchall()]

        if ids:
            # Delete vectors
            placeholders = ",".join("?" * len(ids))
            cursor.execute(
                f"DELETE FROM vec_symbols WHERE symbol_id IN ({placeholders})", ids
            )
            # Delete symbols
            cursor.execute(
                f"DELETE FROM symbols WHERE id IN ({placeholders})", ids
            )

        self._conn.commit()
        return len(ids)

    def upsert_imports(self, source_file: str, imported_files: list[str]):
        """Replace one-hop import edges for source_file."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM imports WHERE source_file = ?", (source_file,))

        unique_files = sorted(set(imported_files))
        for imported in unique_files:
            cursor.execute(
                """
                INSERT OR IGNORE INTO imports (source_file, imported_file)
                VALUES (?, ?)
                """,
                (source_file, imported),
            )

        self._conn.commit()

    def delete_imports(self, file_path: str):
        """Delete import edges originating from file_path."""
        self._conn.execute("DELETE FROM imports WHERE source_file = ?", (file_path,))
        self._conn.commit()

    def get_imports(self, file_path: str) -> list[str]:
        """Return one-hop imported files for a source file."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT imported_file FROM imports WHERE source_file = ? ORDER BY imported_file",
            (file_path,),
        )
        return [row[0] for row in cursor.fetchall()]

    def get_imported_by(self, file_path: str) -> list[str]:
        """Return source files that directly import file_path."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT source_file FROM imports WHERE imported_file = ? ORDER BY source_file",
            (file_path,),
        )
        return [row[0] for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[dict]:
        """KNN search via sqlite-vec, returns symbols ordered by cosine distance.

        Returns a list of dicts with all symbol fields plus 'distance' (cosine).
        Lower distance = more similar.
        """
        cursor = self._conn.cursor()

        # sqlite-vec KNN query
        cursor.execute("""
            SELECT
                v.symbol_id,
                v.distance,
                s.file_path, s.language, s.name, s.kind, s.signature,
                s.docstring, s.body_preview, s.start_line, s.end_line,
                s.body_hash
            FROM vec_symbols v
            JOIN symbols s ON s.id = v.symbol_id
            WHERE v.embedding MATCH ?
                AND k = ?
            ORDER BY v.distance
        """, (_serialize_vector(query_vector), top_k))

        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "distance": row[1],
                "file_path": row[2],
                "language": row[3],
                "name": row[4],
                "kind": row[5],
                "signature": row[6],
                "docstring": row[7],
                "body_preview": row[8],
                "start_line": row[9],
                "end_line": row[10],
                "body_hash": row[11],
            })

        return results

    def search_with_tiers(self, query_vector: np.ndarray, top_k: int = 10) -> dict:
        """Search with two-tier threshold filtering.

        Returns:
            {
                "primary": [...],    # distance <= PRIMARY_THRESHOLD -> full body
                "summaries": [...],  # distance <= SUMMARY_THRESHOLD -> CliffNotes only
                "dropped": int,      # count of results beyond summary threshold
            }

        Thresholds are distance-based: lower distance = more similar.
        """
        results = self.search(query_vector, top_k)

        primary = []
        summaries = []
        dropped = 0

        for r in results:
            dist = r["distance"]
            if dist <= CTX_PRIMARY_THRESHOLD:
                primary.append(r)
            elif dist <= CTX_SUMMARY_THRESHOLD:
                # Attach CliffNotes summary if available
                summary = self.get_summary(r["file_path"])
                r["cliff_notes"] = summary
                summaries.append(r)
            else:
                dropped += 1

        return {
            "primary": primary,
            "summaries": summaries,
            "dropped": dropped,
        }

    def find_symbols_by_file(self, file_path: str) -> list[dict]:
        """Find all symbols in a given file path (exact or partial match).

        Supports exact match and suffix match (e.g., 'auth.py' matches
        'src/auth.py').
        """
        cursor = self._conn.cursor()

        # Try exact match first
        cursor.execute("""
            SELECT id, file_path, language, name, kind, signature,
                   docstring, body_preview, start_line, end_line, body_hash
            FROM symbols WHERE file_path = ?
            ORDER BY start_line
        """, (file_path,))

        rows = cursor.fetchall()

        # If no exact match, try suffix match
        if not rows:
            cursor.execute("""
                SELECT id, file_path, language, name, kind, signature,
                       docstring, body_preview, start_line, end_line, body_hash
                FROM symbols WHERE file_path LIKE ?
                ORDER BY start_line
            """, (f"%{file_path}",))
            rows = cursor.fetchall()

        return [
            {
                "id": r[0], "file_path": r[1], "language": r[2],
                "name": r[3], "kind": r[4], "signature": r[5],
                "docstring": r[6], "body_preview": r[7],
                "start_line": r[8], "end_line": r[9], "body_hash": r[10],
                "distance": 0.0,  # Exact match
            }
            for r in rows
        ]

    def find_symbols_by_name(self, name: str) -> list[dict]:
        """Find symbols by exact or partial name match via SQL.

        Matches exact name, or suffix (e.g., 'login' matches
        'AuthService.login').
        """
        cursor = self._conn.cursor()

        # Try exact match first
        cursor.execute("""
            SELECT id, file_path, language, name, kind, signature,
                   docstring, body_preview, start_line, end_line, body_hash
            FROM symbols WHERE name = ?
            ORDER BY file_path, start_line
        """, (name,))

        rows = cursor.fetchall()

        # If no exact match, try LIKE match (contains)
        if not rows:
            cursor.execute("""
                SELECT id, file_path, language, name, kind, signature,
                       docstring, body_preview, start_line, end_line, body_hash
                FROM symbols WHERE name LIKE ?
                ORDER BY file_path, start_line
            """, (f"%{name}%",))
            rows = cursor.fetchall()

        return [
            {
                "id": r[0], "file_path": r[1], "language": r[2],
                "name": r[3], "kind": r[4], "signature": r[5],
                "docstring": r[6], "body_preview": r[7],
                "start_line": r[8], "end_line": r[9], "body_hash": r[10],
                "distance": 0.0,  # Exact match
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # CliffNotes summaries
    # ------------------------------------------------------------------

    def get_summary(self, file_path: str) -> str | None:
        """Fetch cached CliffNotes summary for a file."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT summary FROM summaries WHERE file_path = ?", (file_path,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def set_summary(self, file_path: str, summary: str):
        """Cache a CliffNotes summary for a file."""
        self._conn.execute("""
            INSERT OR REPLACE INTO summaries (file_path, summary, updated_at)
            VALUES (?, ?, datetime('now'))
        """, (file_path, summary))
        self._conn.commit()

    def delete_summary(self, file_path: str):
        """Delete cached summary for a file."""
        self._conn.execute("DELETE FROM summaries WHERE file_path = ?", (file_path,))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get index statistics."""
        cursor = self._conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM symbols")
        total_symbols = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT file_path) FROM symbols")
        total_files = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM summaries")
        total_summaries = cursor.fetchone()[0]

        return {
            "total_symbols": total_symbols,
            "total_files": total_files,
            "total_summaries": total_summaries,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ---------------------------------------------------------------------------
# CLI entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from contextpilot.extractor import Symbol, extract_symbols
    from contextpilot.embedder import embed_symbols, embed_query

    if len(sys.argv) < 2:
        print("Usage: python -m contextpilot.store <project_root> [query...]", file=sys.stderr)
        print("  First run indexes all .py files in the project.", file=sys.stderr)
        print("  Add queries after the path to test search.", file=sys.stderr)
        sys.exit(1)

    project_root = sys.argv[1]
    queries = sys.argv[2:] if len(sys.argv) > 2 else ["authentication logic"]

    print(f"[ctxpilot] Opening store at {project_root}...", file=sys.stderr)

    with VectorStore(project_root) as store:
        # Find and index all Python files
        root = Path(project_root)
        py_files = list(root.rglob("*.py"))
        py_files = [f for f in py_files if ".ctxpilot" not in str(f) and ".venv" not in str(f)]

        print(f"[ctxpilot] Found {len(py_files)} Python files to index.", file=sys.stderr)

        all_symbols = []
        for py_file in py_files:
            symbols = extract_symbols(str(py_file))
            all_symbols.extend(symbols)

        print(f"[ctxpilot] Extracted {len(all_symbols)} symbols. Embedding...", file=sys.stderr)

        if all_symbols:
            vectors = embed_symbols(all_symbols)
            store.upsert_symbols_batch(all_symbols, vectors)
            print(f"[ctxpilot] Indexed {len(all_symbols)} symbols.", file=sys.stderr)

        # Run search queries
        for query in queries:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Query: \"{query}\"", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

            qvec = embed_query(query)
            results = store.search_with_tiers(qvec, top_k=5)

            print(f"\nPrimary results ({len(results['primary'])}):", file=sys.stderr)
            for r in results["primary"]:
                print(f"  [{r['distance']:.4f}] {r['name']} ({r['kind']}) — {r['file_path']}", file=sys.stderr)
                if r["docstring"]:
                    print(f"           {r['docstring'][:80]}", file=sys.stderr)

            print(f"\nSummary-tier results ({len(results['summaries'])}):", file=sys.stderr)
            for r in results["summaries"]:
                print(f"  [{r['distance']:.4f}] {r['name']} ({r['kind']}) — {r['file_path']}", file=sys.stderr)

            print(f"\nDropped: {results['dropped']}", file=sys.stderr)

        # Print stats
        stats = store.get_stats()
        print(f"\nStore stats: {json.dumps(stats)}", file=sys.stderr)
