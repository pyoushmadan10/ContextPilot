"""Semantic retrieval engine with three-tier resolution and action graph memory.

Resolution order for ctx_continue on cold start:
1. Exact file path match
2. Exact symbol name match via SQL
3. Semantic KNN search via embeddings

Also manages the per-session action_graph for memory-first retrieval.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tiktoken

from contextpilot.cost_guard import estimate_cost_usd
from contextpilot.embedder import embed_query
from contextpilot.store import VectorStore, CTX_PRIMARY_THRESHOLD, CTX_SUMMARY_THRESHOLD

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_ENCODING = None


def _get_encoding():
    """Lazy-load tiktoken encoding."""
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding."""
    if not text:
        return 0
    return len(_get_encoding().encode(text))


# ---------------------------------------------------------------------------
# Action Graph — per-session memory
# ---------------------------------------------------------------------------

class ActionGraph:
    """Tracks files read, edited, and queries made this session.

    Persisted to .ctxpilot/action_graph.json.
    """

    def __init__(self, ctxpilot_dir: Path):
        self._path = ctxpilot_dir / "action_graph.json"
        self._data: dict[str, Any] = {
            "session_start": time.time(),
            "turns": [],
            "files_read": {},      # file_path -> {count, last_turn}
            "files_edited": {},    # file_path -> {count, last_turn, summary}
            "queries": [],         # list of {query, turn, mode}
        }
        self._turn = 0
        self._load()

    def _load(self):
        """Load existing action graph if present."""
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self._data.update(saved)
                self._turn = len(self._data.get("turns", []))
            except (json.JSONDecodeError, OSError):
                pass

    def _save(self):
        """Persist action graph to disk."""
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, default=str)
        except OSError as e:
            print(f"[ctxpilot] Warning: could not save action graph: {e}", file=sys.stderr)

    def new_turn(self):
        """Start a new turn in the session."""
        self._turn += 1
        self._data["turns"].append({
            "turn": self._turn,
            "timestamp": time.time(),
            "actions": [],
        })
        self._save()

    @property
    def turn(self) -> int:
        return self._turn

    def record_read(self, file_path: str):
        """Record a file read."""
        reads = self._data["files_read"]
        if file_path in reads:
            reads[file_path]["count"] += 1
            reads[file_path]["last_turn"] = self._turn
        else:
            reads[file_path] = {"count": 1, "last_turn": self._turn}

        if self._data["turns"]:
            self._data["turns"][-1]["actions"].append({
                "type": "read", "file_path": file_path,
            })
        self._save()

    def record_edit(self, file_path: str, summary: str | None = None):
        """Record a file edit."""
        edits = self._data["files_edited"]
        if file_path in edits:
            edits[file_path]["count"] += 1
            edits[file_path]["last_turn"] = self._turn
            if summary:
                edits[file_path]["summary"] = summary
        else:
            edits[file_path] = {
                "count": 1, "last_turn": self._turn,
                "summary": summary or "",
            }

        if self._data["turns"]:
            self._data["turns"][-1]["actions"].append({
                "type": "edit", "file_path": file_path, "summary": summary,
            })
        self._save()

    def record_query(self, query: str, mode: str):
        """Record a search query."""
        self._data["queries"].append({
            "query": query, "turn": self._turn, "mode": mode,
        })
        self._save()

    def get_recent_files(self, last_n_turns: int = 3) -> list[str]:
        """Get files touched in the last N turns (reads + edits)."""
        min_turn = max(1, self._turn - last_n_turns + 1)
        recent = set()

        for fp, info in self._data["files_read"].items():
            if info["last_turn"] >= min_turn:
                recent.add(fp)

        for fp, info in self._data["files_edited"].items():
            if info["last_turn"] >= min_turn:
                recent.add(fp)

        return list(recent)

    def find_matching_files(self, query: str) -> list[str]:
        """Find files from action history that might match a query.

        Checks if query terms appear in file paths or edit summaries.
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        matches = []

        # Check edited files (highest priority — user was working on these)
        for fp, info in self._data["files_edited"].items():
            fp_lower = fp.lower()
            summary_lower = (info.get("summary") or "").lower()

            if any(term in fp_lower or term in summary_lower for term in query_terms):
                matches.append(fp)

        # Check read files
        for fp, info in self._data["files_read"].items():
            if fp not in matches:
                fp_lower = fp.lower()
                if any(term in fp_lower for term in query_terms):
                    matches.append(fp)

        return matches

    def get_top_files(self, n: int = 5) -> list[dict]:
        """Get the most-accessed files this session."""
        file_counts: dict[str, int] = {}

        for fp, info in self._data["files_read"].items():
            file_counts[fp] = file_counts.get(fp, 0) + info["count"]

        for fp, info in self._data["files_edited"].items():
            file_counts[fp] = file_counts.get(fp, 0) + info["count"] * 2  # Edits weight more

        sorted_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"file_path": fp, "access_count": count} for fp, count in sorted_files[:n]]

    def reset(self):
        """Reset for a new session."""
        self._data = {
            "session_start": time.time(),
            "turns": [],
            "files_read": {},
            "files_edited": {},
            "queries": [],
        }
        self._turn = 0
        self._save()


# ---------------------------------------------------------------------------
# Retriever — the core retrieval engine
# ---------------------------------------------------------------------------

# Per-turn read budget (chars)
CTX_READ_BUDGET = int(os.environ.get("CTX_READ_BUDGET", "18000"))


def expand_with_imports(
    primary_symbols: list[dict],
    store: VectorStore,
    already_included_files: set[str],
) -> list[str]:
    """Expand one hop along import edges from primary symbol files.

    Returns neighbor file paths not already included.
    """
    neighbor_files: list[str] = []
    seen: set[str] = set()

    source_files = {s.get("file_path", "") for s in primary_symbols if s.get("file_path")}
    for source_file in source_files:
        for imported in store.get_imports(source_file):
            if imported in already_included_files:
                continue
            if imported in seen:
                continue
            neighbor_files.append(imported)
            seen.add(imported)

    return neighbor_files


def _build_neighbor_summaries(
    neighbor_files: list[str],
    store: VectorStore,
    primary_token_budget: int,
) -> tuple[list[dict], int]:
    """Build neighbor summary tier under a strict token cap.

    Neighbor tier is kept cheaper than primary by capping total neighbor
    summary tokens below the primary token budget.
    """
    max_neighbor_tokens = max(0, primary_token_budget - 1)
    neighbors: list[dict] = []
    tokens_neighbors = 0

    if max_neighbor_tokens <= 0:
        return neighbors, tokens_neighbors

    for nfile in neighbor_files[:5]:
        summary = store.get_summary(nfile)
        if not summary:
            continue
        summary_tokens = count_tokens(summary)
        if tokens_neighbors + summary_tokens > max_neighbor_tokens:
            continue

        neighbors.append({
            "file_path": nfile,
            "label": "import_neighbor",
            "summary": summary,
        })
        tokens_neighbors += summary_tokens

    return neighbors, tokens_neighbors


class Retriever:
    """Semantic retrieval engine with action graph memory.

    Three-tier resolution for ctx_continue:
    1. Exact file path match from action graph / query
    2. Exact symbol name match via SQL
    3. Semantic KNN search via embeddings
    """

    def __init__(self, store: VectorStore, project_root: str):
        self.store = store
        self.project_root = Path(project_root).resolve()
        self.action_graph = ActionGraph(self.store.ctxpilot_dir)
        self._turn_chars_read = 0

    def new_turn(self):
        """Start a new turn — reset per-turn budgets."""
        self.action_graph.new_turn()
        self._turn_chars_read = 0

    # ------------------------------------------------------------------
    # Traversal record builder
    # ------------------------------------------------------------------

    def _build_traversal_record(
        self,
        query: str,
        total_tokens_sent: int,
        total_tokens_raw: int,
        context_sent: list[dict],
        context_excluded: list[dict] | None = None,
    ) -> dict:
        """Build a traversal record for session_stats."""
        preview = query[:60] + ("..." if len(query) > 60 else "")
        cost_sent = estimate_cost_usd(total_tokens_sent)
        cost_raw = estimate_cost_usd(total_tokens_raw)
        return {
            "turn": self.action_graph.turn,
            "prompt_preview": preview,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total_tokens_sent": total_tokens_sent,
            "total_tokens_raw": total_tokens_raw,
            "cost_this_turn_usd": round(cost_sent, 6),
            "cost_raw_would_have_been_usd": round(cost_raw, 6),
            "context_sent": context_sent,
            "context_excluded": context_excluded or [],
        }

    # ------------------------------------------------------------------
    # ctx_continue — three-tier resolution
    # ------------------------------------------------------------------

    def ctx_continue(self, query: str) -> dict:
        """First tool called every turn. Three-tier cold-start resolution:

        1. Check action_graph for files from previous turns matching query
        2. If no memory match → try exact file path match in DB
        3. If no file match → try exact symbol name match via SQL
        4. If no symbol match → fall back to semantic KNN search

        Returns ContextResult dict.
        """
        self.new_turn()

        # Tier 0: Check action graph memory
        memory_files = self.action_graph.find_matching_files(query)
        if memory_files:
            # Found in session memory — retrieve symbols from those files
            symbols = []
            for fp in memory_files[:5]:  # Limit to top 5 files
                file_symbols = self.store.find_symbols_by_file(fp)
                symbols.extend(file_symbols)

            if symbols:
                total_tokens = sum(count_tokens(s.get("body_preview", "")) for s in symbols)
                already_files = {s.get("file_path", "") for s in symbols if s.get("file_path")}
                neighbor_files = expand_with_imports(symbols, self.store, already_files)[:5]
                neighbors, tokens_neighbors = _build_neighbor_summaries(
                    neighbor_files, self.store, total_tokens
                )

                self.action_graph.record_query(query, "memory_first")

                # Build traversal context_sent for memory hits
                ctx_sent = []
                for s in symbols[:20]:
                    tok = count_tokens(s.get("body_preview", ""))
                    ctx_sent.append({
                        "file": s.get("file_path", ""),
                        "symbol": s.get("name"),
                        "kind": s.get("kind", "function"),
                        "tier": "memory",
                        "score": None,
                        "tokens": tok,
                        "reason": "action graph hit",
                    })
                for n in neighbors:
                    ntok = count_tokens(n.get("summary", ""))
                    ctx_sent.append({
                        "file": n.get("file_path", ""),
                        "symbol": None,
                        "kind": "file",
                        "tier": "neighbor",
                        "score": None,
                        "tokens": ntok,
                        "reason": f"import neighbor of memory file",
                    })

                tokens_sent = total_tokens + tokens_neighbors
                tokens_raw = tokens_sent * 5  # Estimate: full-file ~5x
                traversal = self._build_traversal_record(
                    query, tokens_sent, tokens_raw, ctx_sent
                )

                return {
                    "mode": "memory_first",
                    "confidence": "high",
                    "resolution": "action_graph",
                    "symbols": symbols[:20],  # Cap at 20
                    "neighbors": neighbors,
                    "total_tokens": tokens_sent,
                    "tokens_neighbors": tokens_neighbors,
                    "turn": self.action_graph.turn,
                    "_traversal": traversal,
                }

        # Tier 1: Exact file path match
        file_results = self.store.find_symbols_by_file(query)
        if file_results:
            total_tokens = sum(count_tokens(s.get("body_preview", "")) for s in file_results)
            self.action_graph.record_query(query, "file_path_match")
            return {
                "mode": "file_path_match",
                "confidence": "high",
                "resolution": "exact_file",
                "symbols": file_results,
                "total_tokens": total_tokens,
                "turn": self.action_graph.turn,
            }

        # Tier 2: Exact symbol name match via SQL
        name_results = self.store.find_symbols_by_name(query)
        if name_results:
            total_tokens = sum(count_tokens(s.get("body_preview", "")) for s in name_results)
            self.action_graph.record_query(query, "symbol_name_match")
            return {
                "mode": "symbol_name_match",
                "confidence": "high",
                "resolution": "exact_symbol",
                "symbols": name_results,
                "total_tokens": total_tokens,
                "turn": self.action_graph.turn,
            }

        # Tier 3: Semantic KNN search (fallback)
        return self._semantic_retrieve(query)

    def _semantic_retrieve(self, query: str, top_k: int = 10) -> dict:
        """Semantic KNN search with two-tier threshold filtering."""
        query_vector = embed_query(query)
        tiered = self.store.search_with_tiers(query_vector, top_k)

        primary_tokens = sum(
            count_tokens(s.get("body_preview", "")) for s in tiered["primary"]
        )
        summary_tokens = sum(
            count_tokens(s.get("cliff_notes") or s.get("signature", ""))
            for s in tiered["summaries"]
        )

        already_files = {
            *(s.get("file_path", "") for s in tiered["primary"] if s.get("file_path")),
            *(s.get("file_path", "") for s in tiered["summaries"] if s.get("file_path")),
        }
        neighbor_files = expand_with_imports(tiered["primary"], self.store, set(already_files))[:5]
        neighbors, tokens_neighbors = _build_neighbor_summaries(
            neighbor_files, self.store, primary_tokens
        )

        self.action_graph.record_query(query, "semantic_search")

        # Build traversal record
        ctx_sent = []
        for s in tiered["primary"]:
            tok = count_tokens(s.get("body_preview", ""))
            ctx_sent.append({
                "file": s.get("file_path", ""),
                "symbol": s.get("name"),
                "kind": s.get("kind", "function"),
                "tier": "primary",
                "score": round(s.get("distance", 0), 4),
                "tokens": tok,
                "reason": f"above primary threshold {CTX_PRIMARY_THRESHOLD}",
            })
        for s in tiered["summaries"]:
            tok = count_tokens(s.get("cliff_notes") or s.get("signature", ""))
            ctx_sent.append({
                "file": s.get("file_path", ""),
                "symbol": s.get("name"),
                "kind": s.get("kind", "function"),
                "tier": "summary",
                "score": round(s.get("distance", 0), 4),
                "tokens": tok,
                "reason": f"above summary threshold {CTX_SUMMARY_THRESHOLD}",
            })
        for n in neighbors:
            ntok = count_tokens(n.get("summary", ""))
            ctx_sent.append({
                "file": n.get("file_path", ""),
                "symbol": None,
                "kind": "file",
                "tier": "neighbor",
                "score": None,
                "tokens": ntok,
                "reason": f"import neighbor of {n.get('file_path', '')}",
            })

        # Excluded: results that were dropped by the store's tier search
        # We can reconstruct approximate exclusions from the raw search
        ctx_excluded = []
        raw_results = self.store.search(query_vector, top_k)
        for r in raw_results:
            dist = r.get("distance", 999)
            if dist > CTX_SUMMARY_THRESHOLD:
                ctx_excluded.append({
                    "file": r.get("file_path", ""),
                    "symbol": r.get("name"),
                    "score": round(dist, 4),
                    "reason": f"below summary threshold {CTX_SUMMARY_THRESHOLD}",
                })

        tokens_sent = primary_tokens + summary_tokens + tokens_neighbors
        tokens_raw = tokens_sent * 5  # Estimate: full-file ~5x
        traversal = self._build_traversal_record(
            query, tokens_sent, tokens_raw, ctx_sent, ctx_excluded
        )

        return {
            "mode": "semantic_search",
            "confidence": "medium" if tiered["primary"] else "low",
            "resolution": "knn",
            "symbols": tiered["primary"],
            "summaries": tiered["summaries"],
            "neighbors": neighbors,
            "total_tokens": tokens_sent,
            "tokens_primary": primary_tokens,
            "tokens_summary": summary_tokens,
            "tokens_neighbors": tokens_neighbors,
            "dropped": tiered["dropped"],
            "turn": self.action_graph.turn,
            "_traversal": traversal,
        }

    # ------------------------------------------------------------------
    # ctx_retrieve — direct semantic search
    # ------------------------------------------------------------------

    def ctx_retrieve(self, query: str, top_k: int = 10) -> dict:
        """Semantic retrieval with two-tier thresholds.

        Returns RetrievalResult dict.
        """
        query_vector = embed_query(query)
        tiered = self.store.search_with_tiers(query_vector, top_k)

        primary_tokens = sum(
            count_tokens(s.get("body_preview", "")) for s in tiered["primary"]
        )
        summary_tokens = sum(
            count_tokens(s.get("cliff_notes") or s.get("signature", ""))
            for s in tiered["summaries"]
        )

        already_files = {
            *(s.get("file_path", "") for s in tiered["primary"] if s.get("file_path")),
            *(s.get("file_path", "") for s in tiered["summaries"] if s.get("file_path")),
        }
        neighbor_files = expand_with_imports(tiered["primary"], self.store, set(already_files))[:5]
        neighbors, tokens_neighbors = _build_neighbor_summaries(
            neighbor_files, self.store, primary_tokens
        )

        # Estimate tokens saved: rough estimate of what full-file read would cost
        full_tokens = primary_tokens + summary_tokens + tokens_neighbors
        # Assume sending full files would be ~5x the symbol-level tokens
        tokens_saved = max(0, full_tokens * 4)

        self.action_graph.record_query(query, "semantic_search")

        return {
            "primary": tiered["primary"],
            "summaries": tiered["summaries"],
            "neighbors": neighbors,
            "tokens_primary": primary_tokens,
            "tokens_summary": summary_tokens,
            "tokens_neighbors": tokens_neighbors,
            "tokens_saved": tokens_saved,
            "turn": self.action_graph.turn,
        }

    # ------------------------------------------------------------------
    # ctx_read — file/symbol reader with budget
    # ------------------------------------------------------------------

    def ctx_read(self, file_path: str, symbol_name: str | None = None) -> dict:
        """Read a file or specific symbol from the project.

        Enforces per-turn read budget (default 18,000 chars).
        Tracks reads in action_graph.

        Returns ReadResult dict.
        """
        abs_path = self.project_root / file_path

        if not abs_path.exists():
            return {
                "error": f"File not found: {file_path}",
                "content": "",
                "tokens": 0,
                "from_cache": False,
                "stale": False,
            }

        if symbol_name:
            # Find specific symbol in the file
            symbols = self.store.find_symbols_by_file(file_path)
            target = None
            for s in symbols:
                if s["name"] == symbol_name or s["name"].endswith(f".{symbol_name}"):
                    target = s
                    break

            if target:
                content = target["body_preview"]
                tokens = count_tokens(content)

                # Check staleness
                from contextpilot.extractor import extract_symbols as _extract
                current_symbols = _extract(str(abs_path))
                stale = True
                for cs in current_symbols:
                    if cs.name == target["name"] or cs.name.endswith(f".{symbol_name}"):
                        stale = cs.body_hash != target["body_hash"]
                        break

                # Budget check
                if self._turn_chars_read + len(content) > CTX_READ_BUDGET:
                    return {
                        "error": f"Read budget exceeded ({self._turn_chars_read}/{CTX_READ_BUDGET} chars used). "
                                 f"Symbol '{symbol_name}' would add {len(content)} chars.",
                        "content": "",
                        "tokens": 0,
                        "from_cache": True,
                        "stale": stale,
                        "budget_remaining": CTX_READ_BUDGET - self._turn_chars_read,
                    }

                self._turn_chars_read += len(content)
                self.action_graph.record_read(file_path)

                return {
                    "content": content,
                    "tokens": tokens,
                    "from_cache": True,
                    "stale": stale,
                    "symbol": symbol_name,
                    "budget_remaining": CTX_READ_BUDGET - self._turn_chars_read,
                }
            else:
                return {
                    "error": f"Symbol '{symbol_name}' not found in {file_path}",
                    "content": "",
                    "tokens": 0,
                    "from_cache": False,
                    "stale": False,
                }
        else:
            # Read entire file
            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                return {
                    "error": f"Could not read {file_path}: {e}",
                    "content": "",
                    "tokens": 0,
                    "from_cache": False,
                    "stale": False,
                }

            # Budget check
            if self._turn_chars_read + len(content) > CTX_READ_BUDGET:
                return {
                    "error": f"Read budget exceeded ({self._turn_chars_read}/{CTX_READ_BUDGET} chars used). "
                             f"File '{file_path}' is {len(content)} chars.",
                    "content": "",
                    "tokens": 0,
                    "from_cache": False,
                    "stale": False,
                    "budget_remaining": CTX_READ_BUDGET - self._turn_chars_read,
                }

            tokens = count_tokens(content)
            self._turn_chars_read += len(content)
            self.action_graph.record_read(file_path)

            return {
                "content": content,
                "tokens": tokens,
                "from_cache": False,
                "stale": False,
                "budget_remaining": CTX_READ_BUDGET - self._turn_chars_read,
            }

    # ------------------------------------------------------------------
    # ctx_register_edit — edit tracking + incremental re-index
    # ------------------------------------------------------------------

    def ctx_register_edit(self, file_path: str, summary: str | None = None) -> dict:
        """Record an edit and trigger incremental re-index.

        Returns edit result dict.
        """
        from contextpilot.scanner import scan_file

        self.action_graph.record_edit(file_path, summary)

        # Trigger incremental re-index of the edited file
        result = scan_file(str(self.project_root), file_path, self.store)

        return {
            "reindexed": result["reindexed"],
            "new_symbols": result["new_symbols"],
            "removed_symbols": result["removed_symbols"],
            "file_path": file_path,
            "turn": self.action_graph.turn,
        }
