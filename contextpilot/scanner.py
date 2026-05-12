"""Project scanner with incremental indexing via xxhash file tracking.

Walks a project directory, extracts symbols from supported files, embeds them,
and stores in the vector DB. Tracks file hashes for incremental re-indexing:
only changed/new files are processed on subsequent runs.

Scanner runs synchronously. Embedding uses ThreadPoolExecutor(4) for
parallelism, processing symbols in batches of 50.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pathspec
import xxhash

from contextpilot.embedder import embed_symbols
from contextpilot.extractor import extract_imports, extract_symbols, Symbol
from contextpilot.store import VectorStore
from contextpilot.summarizer import generate_summary

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Max file size to index (default 2MB) — larger files are usually
# generated/minified and not useful for context
MAX_FILE_SIZE = int(os.environ.get("CTX_MAX_FILE_SIZE", str(2 * 1024 * 1024)))

# Embedding batch size and worker count
EMBED_BATCH_SIZE = 50
EMBED_WORKERS = 4

# Directories and patterns to always skip
ALWAYS_SKIP_DIRS = {
    ".ctxpilot", ".git", "node_modules", "__pycache__",
    "dist", "build", ".venv", "venv", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "eggs", ".eggs",
}

ALWAYS_SKIP_PATTERNS = {
    "*.min.js", "*.lock", "*.map",
}

# Binary file extensions to skip
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a",
    ".pyc", ".pyo", ".class", ".jar",
    ".mp3", ".mp4", ".wav", ".avi", ".mkv",
    ".sqlite", ".db", ".db3",
    ".wasm",
}


@dataclass
class ScanResult:
    """Result of a project scan."""
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    files_deleted: int = 0
    symbols_total: int = 0
    duration_seconds: float = 0.0


def _load_gitignore(project_root: Path) -> pathspec.PathSpec | None:
    """Load .gitignore patterns if the file exists."""
    gitignore_path = project_root / ".gitignore"
    if not gitignore_path.exists():
        return None

    try:
        with open(gitignore_path, "r", encoding="utf-8", errors="replace") as f:
            patterns = f.read()
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns.splitlines())
    except OSError:
        return None


def _should_skip_file(file_path: Path, project_root: Path, gitignore_spec: pathspec.PathSpec | None) -> bool:
    """Check if a file should be skipped based on all skip rules."""
    # Check binary extensions
    if file_path.suffix.lower() in BINARY_EXTENSIONS:
        return True

    # Check always-skip patterns
    name = file_path.name
    for pattern in ALWAYS_SKIP_PATTERNS:
        if name.endswith(pattern.lstrip("*")):
            return True

    # Check file size
    try:
        if file_path.stat().st_size > MAX_FILE_SIZE:
            return True
        if file_path.stat().st_size == 0:
            return True
    except OSError:
        return True

    # Check .gitignore
    if gitignore_spec:
        try:
            rel_path = file_path.relative_to(project_root).as_posix()
            if gitignore_spec.match_file(rel_path):
                return True
        except ValueError:
            pass

    return False


def _should_skip_dir(dir_name: str) -> bool:
    """Check if a directory should be skipped entirely."""
    return dir_name in ALWAYS_SKIP_DIRS or dir_name.startswith(".")


def _compute_file_hash(file_path: Path) -> str:
    """Compute xxhash of raw file bytes for change detection."""
    try:
        return xxhash.xxh64(file_path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _load_file_hashes(ctxpilot_dir: Path) -> dict[str, str]:
    """Load previous file hashes from .ctxpilot/file_hashes.json."""
    hash_file = ctxpilot_dir / "file_hashes.json"
    if not hash_file.exists():
        return {}
    try:
        with open(hash_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_file_hashes(ctxpilot_dir: Path, hashes: dict[str, str]):
    """Save current file hashes to .ctxpilot/file_hashes.json."""
    hash_file = ctxpilot_dir / "file_hashes.json"
    with open(hash_file, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2)


def _embed_batch(symbols: list[Symbol]) -> list:
    """Embed a batch of symbols. Used as ThreadPoolExecutor task."""
    if not symbols:
        return []
    return embed_symbols(symbols)


# ---------------------------------------------------------------------------
# Supported file extensions (from extractor)
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".go"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_project(project_root: str, store: VectorStore | None = None) -> ScanResult:
    """Scan a project directory and index all supported source files.

    - Walks directory tree, respects .gitignore
    - Skips: .ctxpilot/, .git/, node_modules/, etc.
    - Computes xxhash per file, compares with previous scan
    - Re-indexes only changed/new files
    - Deletes embeddings for removed files
    - Generates CliffNotes summaries per file

    Args:
        project_root: Absolute path to the project root.
        store: VectorStore instance. Created if not provided.

    Returns:
        ScanResult with scan statistics.
    """
    start_time = time.time()
    root = Path(project_root).resolve()
    result = ScanResult()

    if not root.is_dir():
        print(f"[ctxpilot] Error: {project_root} is not a directory.", file=sys.stderr)
        return result

    # Setup
    ctxpilot_dir = root / ".ctxpilot"
    ctxpilot_dir.mkdir(exist_ok=True)

    own_store = store is None
    if own_store:
        store = VectorStore(str(root))

    gitignore_spec = _load_gitignore(root)
    old_hashes = _load_file_hashes(ctxpilot_dir)
    new_hashes: dict[str, str] = {}

    # Phase 1: Walk directory and identify files to process
    files_to_index: list[Path] = []
    all_current_files: set[str] = set()

    print(f"[ctxpilot] Scanning {root}...", file=sys.stderr)

    for dirpath, dirnames, filenames in os.walk(root):
        # Filter directories in-place to skip unwanted ones
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]

        for filename in filenames:
            file_path = Path(dirpath) / filename

            # Only process supported extensions
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            # Apply skip rules
            if _should_skip_file(file_path, root, gitignore_spec):
                result.files_skipped += 1
                continue

            result.files_scanned += 1

            # Use relative path as key for hash tracking
            rel_path = str(file_path.relative_to(root))
            all_current_files.add(rel_path)

            # Compute hash and check for changes
            file_hash = _compute_file_hash(file_path)
            new_hashes[rel_path] = file_hash

            if rel_path in old_hashes and old_hashes[rel_path] == file_hash:
                # File unchanged — skip
                continue

            files_to_index.append(file_path)

    # Phase 2: Delete embeddings for removed files
    removed_files = set(old_hashes.keys()) - all_current_files
    for removed_file in removed_files:
        count = store.delete_symbols_for_file(removed_file)
        store.delete_imports(removed_file)
        store.delete_summary(removed_file)
        if count > 0:
            result.files_deleted += 1
            print(f"[ctxpilot]   Removed: {removed_file} ({count} symbols)", file=sys.stderr)

    if not files_to_index:
        print(
            f"[ctxpilot] No changes detected. "
            f"Scanned {result.files_scanned} files, "
            f"skipped {result.files_skipped}.",
            file=sys.stderr,
        )
        _save_file_hashes(ctxpilot_dir, new_hashes)
        result.duration_seconds = time.time() - start_time
        if own_store:
            store.close()
        return result

    print(
        f"[ctxpilot] Found {len(files_to_index)} files to index "
        f"(of {result.files_scanned} scanned, {result.files_skipped} skipped).",
        file=sys.stderr,
    )

    # Phase 3: Extract symbols from changed files
    all_symbols: list[Symbol] = []
    file_symbols_map: dict[str, list[Symbol]] = {}

    for file_path in files_to_index:
        rel_path = str(file_path.relative_to(root))
        symbols = extract_symbols(str(file_path))
        imported_files_abs = extract_imports(str(file_path))

        imported_files_rel: list[str] = []
        for imported_abs in imported_files_abs:
            try:
                imported_rel = str(Path(imported_abs).resolve().relative_to(root))
                imported_files_rel.append(imported_rel)
            except ValueError:
                continue

        store.upsert_imports(rel_path, imported_files_rel)

        if symbols:
            # Use relative paths in symbols for portability
            adjusted_symbols = []
            for s in symbols:
                adjusted = Symbol(
                    file_path=rel_path,
                    language=s.language,
                    name=s.name,
                    kind=s.kind,
                    signature=s.signature,
                    docstring=s.docstring,
                    body_preview=s.body_preview,
                    start_line=s.start_line,
                    end_line=s.end_line,
                    body_hash=s.body_hash,
                )
                adjusted_symbols.append(adjusted)

            all_symbols.extend(adjusted_symbols)
            file_symbols_map[rel_path] = adjusted_symbols

        print(
            f"[ctxpilot]   Extracted: {rel_path} ({len(symbols)} symbols)",
            file=sys.stderr,
        )

    result.symbols_total = len(all_symbols)

    if not all_symbols:
        print("[ctxpilot] No symbols extracted from changed files.", file=sys.stderr)
        _save_file_hashes(ctxpilot_dir, new_hashes)
        result.duration_seconds = time.time() - start_time
        if own_store:
            store.close()
        return result

    # Phase 4: Embed symbols using ThreadPoolExecutor(4) in batches of 50
    print(
        f"[ctxpilot] Embedding {len(all_symbols)} symbols "
        f"(batch_size={EMBED_BATCH_SIZE}, workers={EMBED_WORKERS})...",
        file=sys.stderr,
    )

    # Split into batches
    batches = [
        all_symbols[i:i + EMBED_BATCH_SIZE]
        for i in range(0, len(all_symbols), EMBED_BATCH_SIZE)
    ]

    # Process batches in parallel
    all_vectors = [None] * len(all_symbols)

    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as executor:
        future_to_batch_idx = {}
        for batch_idx, batch in enumerate(batches):
            future = executor.submit(_embed_batch, batch)
            future_to_batch_idx[future] = batch_idx

        for future in as_completed(future_to_batch_idx):
            batch_idx = future_to_batch_idx[future]
            try:
                vectors = future.result()
                # Place vectors at the correct offset
                start = batch_idx * EMBED_BATCH_SIZE
                for i, vec in enumerate(vectors):
                    all_vectors[start + i] = vec
                print(
                    f"[ctxpilot]   Embedded batch {batch_idx + 1}/{len(batches)} "
                    f"({len(vectors)} symbols)",
                    file=sys.stderr,
                )
            except Exception as e:
                print(
                    f"[ctxpilot] Error embedding batch {batch_idx + 1}: {e}",
                    file=sys.stderr,
                )

    # Phase 5: Store in vector DB
    print("[ctxpilot] Storing embeddings...", file=sys.stderr)

    for file_path in files_to_index:
        rel_path = str(file_path.relative_to(root))
        # Delete old embeddings for this file before re-inserting
        store.delete_symbols_for_file(rel_path)

    # Batch upsert all symbols with their vectors
    valid_pairs = [
        (sym, vec) for sym, vec in zip(all_symbols, all_vectors)
        if vec is not None
    ]

    if valid_pairs:
        symbols_list, vectors_list = zip(*valid_pairs)
        store.upsert_symbols_batch(list(symbols_list), list(vectors_list))

    result.files_indexed = len(files_to_index)

    # Phase 6: Generate CliffNotes summaries
    print("[ctxpilot] Generating CliffNotes summaries...", file=sys.stderr)

    for rel_path, symbols in file_symbols_map.items():
        summary = generate_summary(rel_path, symbols)
        store.set_summary(rel_path, summary)

    # Save updated file hashes
    _save_file_hashes(ctxpilot_dir, new_hashes)

    result.duration_seconds = time.time() - start_time

    print(
        f"[ctxpilot] Scan complete: "
        f"{result.files_indexed} indexed, "
        f"{result.symbols_total} symbols, "
        f"{result.files_deleted} removed, "
        f"{result.duration_seconds:.2f}s",
        file=sys.stderr,
    )

    if own_store:
        store.close()

    return result


def scan_file(project_root: str, file_path: str, store: VectorStore | None = None) -> dict:
    """Incrementally re-index a single file.

    Used by ctx_register_edit to re-index after an edit.

    Returns:
        {"reindexed": bool, "new_symbols": int, "removed_symbols": int}
    """
    root = Path(project_root).resolve()
    fpath = Path(file_path)

    # Use relative path
    if fpath.is_absolute():
        rel_path = str(fpath.relative_to(root))
    else:
        rel_path = file_path
        fpath = root / file_path

    own_store = store is None
    if own_store:
        store = VectorStore(str(root))

    ctxpilot_dir = root / ".ctxpilot"
    hashes = _load_file_hashes(ctxpilot_dir)

    # Count old symbols
    old_count = 0
    cursor = store._conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM symbols WHERE file_path = ?", (rel_path,))
    old_count = cursor.fetchone()[0]

    # Delete old embeddings
    store.delete_symbols_for_file(rel_path)
    store.delete_imports(rel_path)

    if not fpath.exists():
        # File was deleted
        store.delete_summary(rel_path)
        if rel_path in hashes:
            del hashes[rel_path]
        _save_file_hashes(ctxpilot_dir, hashes)
        if own_store:
            store.close()
        return {"reindexed": True, "new_symbols": 0, "removed_symbols": old_count}

    # Extract symbols
    symbols = extract_symbols(str(fpath))
    imported_files_abs = extract_imports(str(fpath))
    imported_files_rel: list[str] = []
    for imported_abs in imported_files_abs:
        try:
            imported_rel = str(Path(imported_abs).resolve().relative_to(root))
            imported_files_rel.append(imported_rel)
        except ValueError:
            continue
    store.upsert_imports(rel_path, imported_files_rel)

    adjusted_symbols = []
    for s in symbols:
        adjusted = Symbol(
            file_path=rel_path,
            language=s.language,
            name=s.name,
            kind=s.kind,
            signature=s.signature,
            docstring=s.docstring,
            body_preview=s.body_preview,
            start_line=s.start_line,
            end_line=s.end_line,
            body_hash=s.body_hash,
        )
        adjusted_symbols.append(adjusted)

    if adjusted_symbols:
        vectors = embed_symbols(adjusted_symbols)
        store.upsert_symbols_batch(adjusted_symbols, vectors)

        # Update summary
        summary = generate_summary(rel_path, adjusted_symbols)
        store.set_summary(rel_path, summary)

    # Update hash
    hashes[rel_path] = _compute_file_hash(fpath)
    _save_file_hashes(ctxpilot_dir, hashes)

    if own_store:
        store.close()

    return {
        "reindexed": True,
        "new_symbols": len(adjusted_symbols),
        "removed_symbols": old_count,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m contextpilot.scanner <project_root>", file=sys.stderr)
        sys.exit(1)

    project_root = sys.argv[1]
    result = scan_project(project_root)

    # Output result as JSON to stdout
    print(json.dumps({
        "files_scanned": result.files_scanned,
        "files_indexed": result.files_indexed,
        "files_skipped": result.files_skipped,
        "files_deleted": result.files_deleted,
        "symbols_total": result.symbols_total,
        "duration_seconds": round(result.duration_seconds, 2),
    }, indent=2))
