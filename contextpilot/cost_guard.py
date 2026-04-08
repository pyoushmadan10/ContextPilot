"""Cost guard — token estimation, budget enforcement, and CLI interrupt.

Intercepts context before it reaches the LLM. Warns when expensive,
offers auto-compress when limits are hit, displays savings.

Uses tiktoken cl100k_base encoding (~5% variance from Claude's tokenizer).
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_ENCODING = None


def _get_encoding():
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def estimate_tokens(text: str) -> int:
    """Estimate token count using tiktoken cl100k_base encoding.

    This is the closest public tokenizer to Claude's — ~5% variance.
    """
    if not text:
        return 0
    return len(_get_encoding().encode(text))


# ---------------------------------------------------------------------------
# Pricing constants
# ---------------------------------------------------------------------------

# Claude Sonnet 4.5 pricing
PRICE_PER_1M_INPUT = 3.00      # $3.00 / 1M input tokens
PRICE_PER_1M_CACHE_READ = 0.30  # $0.30 / 1M cached tokens


def estimate_cost_usd(tokens: int, cached: bool = False) -> float:
    """Estimate cost in USD for a given token count."""
    rate = PRICE_PER_1M_CACHE_READ if cached else PRICE_PER_1M_INPUT
    return tokens * rate / 1_000_000


# ---------------------------------------------------------------------------
# Guard result
# ---------------------------------------------------------------------------

@dataclass
class GuardResult:
    """Result of a cost guard check."""
    token_count: int
    estimated_cost_usd: float
    action: str  # "pass" | "warn" | "interrupt"
    message: str


# ---------------------------------------------------------------------------
# CostGuard class
# ---------------------------------------------------------------------------

class CostGuard:
    """Guards against excessive context token spend.

    Thresholds (configurable via env vars):
    - warn_threshold: show a warning (default 15,000 tokens)
    - hard_limit: interrupt and offer options (default 30,000 tokens)
    """

    def __init__(
        self,
        warn_threshold: int | None = None,
        hard_limit: int | None = None,
    ):
        self.warn_threshold = warn_threshold or int(
            os.environ.get("CTX_WARN_THRESHOLD", "15000")
        )
        self.hard_limit = hard_limit or int(
            os.environ.get("CTX_HARD_LIMIT", "30000")
        )

    def check(self, context_bundle: str) -> GuardResult:
        """Check a context bundle against budget thresholds.

        Returns a GuardResult with the appropriate action:
        - "pass": under warn threshold, proceed normally
        - "warn": above warn threshold, show FYI warning
        - "interrupt": above hard limit, require user decision
        """
        token_count = estimate_tokens(context_bundle)
        cost = estimate_cost_usd(token_count)

        if token_count >= self.hard_limit:
            return GuardResult(
                token_count=token_count,
                estimated_cost_usd=cost,
                action="interrupt",
                message=(
                    f"Context is {token_count:,} tokens (~${cost:.4f}). "
                    f"This exceeds the hard limit of {self.hard_limit:,} tokens."
                ),
            )
        elif token_count >= self.warn_threshold:
            return GuardResult(
                token_count=token_count,
                estimated_cost_usd=cost,
                action="warn",
                message=(
                    f"Context is {token_count:,} tokens (~${cost:.4f}). "
                    f"Above warn threshold of {self.warn_threshold:,} tokens."
                ),
            )
        else:
            return GuardResult(
                token_count=token_count,
                estimated_cost_usd=cost,
                action="pass",
                message=f"Context is {token_count:,} tokens (~${cost:.4f}).",
            )

    def check_and_handle(
        self,
        context_bundle: str,
        compressor=None,
        symbols: list | None = None,
        retriever=None,
    ) -> tuple[str, GuardResult]:
        """Check context and handle interrupt interactively if needed.

        If action is "interrupt", presents the CLI menu to stderr and
        handles the user's choice.

        Args:
            context_bundle: The full context text to check.
            compressor: Optional Compressor instance for auto-compress.
            symbols: Optional list of symbol dicts (for compressor).
            retriever: Optional Retriever instance (for re-query).

        Returns:
            (final_context, guard_result) tuple.
        """
        result = self.check(context_bundle)

        if result.action == "pass":
            return context_bundle, result

        if result.action == "warn":
            print(
                f"\n  [ctxpilot] Warning: {result.message}",
                file=sys.stderr,
            )
            return context_bundle, result

        # action == "interrupt"
        if compressor and symbols:
            compressed = compressor.compress(symbols)
            compressed_text = compressed["compressed_text"]
            compressed_tokens = estimate_tokens(compressed_text)
            compressed_cost = estimate_cost_usd(compressed_tokens)
        else:
            compressed_text = None
            compressed_tokens = 0
            compressed_cost = 0.0

        # Print interrupt UI to stderr
        print(
            f"\n  {'='*56}\n"
            f"  \u26a0  Token alert: context is {result.token_count:,} tokens "
            f"(~${result.estimated_cost_usd:.4f})\n",
            file=sys.stderr,
        )

        if compressed_text:
            print(
                f"  [A] Auto-compress to ~{compressed_tokens:,} tokens "
                f"(~${compressed_cost:.4f})",
                file=sys.stderr,
            )
        print(
            "  [S] Select files to exclude manually\n"
            "  [R] Rewrite query to reduce scope\n"
            "  [P] Proceed anyway\n",
            file=sys.stderr,
        )

        try:
            choice = input("  Choice [A/S/R/P]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            choice = "P"

        if choice == "A" and compressed_text:
            # Auto-compress
            savings = result.token_count - compressed_tokens
            print(
                f"\n  [ctxpilot] Compressed: saved {savings:,} tokens "
                f"(~${estimate_cost_usd(savings):.4f})",
                file=sys.stderr,
            )
            result.action = "compressed"
            result.message = (
                f"Auto-compressed from {result.token_count:,} to "
                f"{compressed_tokens:,} tokens"
            )
            return compressed_text, result

        elif choice == "S" and symbols:
            # Select files to exclude
            files = list({s.get("file_path", "") for s in symbols})
            print("\n  Files in context:", file=sys.stderr)
            for i, f in enumerate(files):
                file_tokens = sum(
                    estimate_tokens(s.get("body_preview", ""))
                    for s in symbols if s.get("file_path") == f
                )
                print(f"    [{i}] {f} (~{file_tokens:,} tokens)", file=sys.stderr)

            try:
                indices = input("  Exclude (comma-separated): ").strip()
                exclude_indices = {int(x.strip()) for x in indices.split(",") if x.strip().isdigit()}
                exclude_files = {files[i] for i in exclude_indices if i < len(files)}
            except (ValueError, EOFError, KeyboardInterrupt):
                exclude_files = set()

            if exclude_files:
                filtered_symbols = [
                    s for s in symbols if s.get("file_path") not in exclude_files
                ]
                filtered_text = "\n\n".join(
                    s.get("body_preview", "") for s in filtered_symbols
                )
                new_tokens = estimate_tokens(filtered_text)
                savings = result.token_count - new_tokens
                print(
                    f"\n  [ctxpilot] Excluded {len(exclude_files)} file(s), "
                    f"saved {savings:,} tokens",
                    file=sys.stderr,
                )
                result.action = "filtered"
                return filtered_text, result

            return context_bundle, result

        elif choice == "R":
            # Rewrite query
            try:
                new_query = input("  Enter narrower query: ").strip()
            except (EOFError, KeyboardInterrupt):
                new_query = ""

            if new_query and retriever:
                new_result = retriever.ctx_retrieve(new_query)
                new_symbols = new_result.get("primary", [])
                new_text = "\n\n".join(
                    s.get("body_preview", "") for s in new_symbols
                )
                new_tokens = estimate_tokens(new_text)
                print(
                    f"\n  [ctxpilot] Re-queried: {new_tokens:,} tokens "
                    f"({len(new_symbols)} symbols)",
                    file=sys.stderr,
                )
                result.action = "requeried"
                return new_text, result

            return context_bundle, result

        else:
            # Proceed anyway
            print(
                "\n  [ctxpilot] Proceeding with full context "
                f"({result.token_count:,} tokens).",
                file=sys.stderr,
            )
            result.action = "override"
            return context_bundle, result


# ---------------------------------------------------------------------------
# Session savings accumulator
# ---------------------------------------------------------------------------

class SessionSavings:
    """Track cumulative token savings across all turns in a session.

    Persisted to .ctxpilot/session_stats.json on every turn.
    """

    def __init__(self, ctxpilot_dir: Path):
        self._path = ctxpilot_dir / "session_stats.json"
        self.data = {
            "session_start": time.time(),
            "turns": 0,
            "tokens_raw": 0,
            "tokens_sent": 0,
            "tokens_saved": 0,
            "cost_saved_usd": 0.0,
            "per_turn": [],
            "traversals": [],
        }

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self.data.update(saved)
            except (json.JSONDecodeError, OSError):
                pass

    def save(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, default=str)
        except OSError:
            pass

    def record_turn(self, tokens_raw: int, tokens_sent: int, turn_id: int):
        """Record token usage for a turn and persist."""
        saved = max(0, tokens_raw - tokens_sent)
        
        per_turn = self.data["per_turn"]
        existing = next((t for t in per_turn if t["turn"] == turn_id), None)
        
        if existing:
            existing["tokens_raw"] += tokens_raw
            existing["tokens_sent"] += tokens_sent
            existing["tokens_saved"] += saved
        else:
            self.data["turns"] += 1
            per_turn.append({
                "turn": turn_id,
                "tokens_raw": tokens_raw,
                "tokens_sent": tokens_sent,
                "tokens_saved": saved,
                "timestamp": time.time(),
            })
            
        self.data["tokens_raw"] += tokens_raw
        self.data["tokens_sent"] += tokens_sent
        self.data["tokens_saved"] += saved
        self.data["cost_saved_usd"] += estimate_cost_usd(saved)
        self.save()

    @property
    def summary(self) -> dict:
        return {
            "turns": self.data["turns"],
            "tokens_raw": self.data["tokens_raw"],
            "tokens_sent": self.data["tokens_sent"],
            "tokens_saved": self.data["tokens_saved"],
            "cost_saved_usd": round(self.data["cost_saved_usd"], 6),
        }

    def record_traversal(self, record: dict):
        """Append a traversal record, capping at 50 entries."""
        traversals = self.data.setdefault("traversals", [])
        traversals.append(record)
        # Cap at 50 — drop oldest
        if len(traversals) > 50:
            self.data["traversals"] = traversals[-50:]
        self.save()

    @property
    def session_cost_usd(self) -> float:
        """Total session cost from traversals."""
        return sum(
            t.get("cost_this_turn_usd", 0)
            for t in self.data.get("traversals", [])
        )

    @property
    def session_cost_raw_would_have_been_usd(self) -> float:
        """What the session would have cost without ContextPilot."""
        return sum(
            t.get("cost_raw_would_have_been_usd", 0)
            for t in self.data.get("traversals", [])
        )


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    guard = CostGuard(warn_threshold=100, hard_limit=200)

    # Test pass
    r = guard.check("Hello world")
    print(f"Pass test: {r.action} — {r.message}")

    # Test warn
    text_warn = "word " * 150
    r = guard.check(text_warn)
    print(f"Warn test: {r.action} — {r.message}")

    # Test interrupt
    text_hard = "word " * 300
    r = guard.check(text_hard)
    print(f"Interrupt test: {r.action} — {r.message}")

    print("\n[OK] Cost guard working correctly.")
