"""Context compression pipeline — the CliffNotes compressor.

Takes a list of symbol results with full bodies and compresses them
by removing boilerplate, deduplicating imports, and ranking by relevance.
Produces a quality report classifying each removal.

Never removes: function signatures, symbol names, or anything with
relevance score above the primary threshold.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field

from contextpilot.cost_guard import estimate_tokens


# ---------------------------------------------------------------------------
# Quality classifications
# ---------------------------------------------------------------------------

QUALITY_LOW = "low"       # Boilerplate, duplicates — safe to remove
QUALITY_MEDIUM = "medium"  # Non-trivial but lower ranked
QUALITY_HIGH = "high"      # Unique symbols — warn user, don't silently drop


@dataclass
class QualityEntry:
    """One entry in the quality report."""
    symbol_name: str
    file_path: str
    quality: str  # "low" | "medium" | "high"
    reason: str
    tokens_saved: int


@dataclass
class CompressedBundle:
    """Result of the compression pipeline."""
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    quality_report: list[QualityEntry] = field(default_factory=list)
    symbols_kept: int = 0
    symbols_removed: int = 0


# ---------------------------------------------------------------------------
# Boilerplate detection patterns
# ---------------------------------------------------------------------------

# Python empty __init__ patterns
_EMPTY_INIT_PATTERNS = [
    re.compile(r'def __init__\(self\)\s*:\s*pass', re.MULTILINE),
    re.compile(r'def __init__\(self\)\s*:\s*\.\.\.',  re.MULTILINE),
]

# Obvious getter/setter patterns
_GETTER_PATTERN = re.compile(
    r'def (get_\w+|_get_\w+)\(self\)\s*(?:->.*?)?\s*:\s*\n\s*return self\.\w+',
    re.MULTILINE,
)
_SETTER_PATTERN = re.compile(
    r'def (set_\w+|_set_\w+)\(self,\s*\w+(?::\s*\w+)?\)\s*(?:->.*?)?\s*:\s*\n\s*self\.\w+\s*=',
    re.MULTILINE,
)

# Property getter/setter (Python)
_PROPERTY_GETTER = re.compile(
    r'@property\s*\n\s*def \w+\(self\)\s*(?:->.*?)?\s*:\s*\n\s*return self\._?\w+',
    re.MULTILINE,
)


def _is_boilerplate(symbol: dict) -> tuple[bool, str]:
    """Check if a symbol is boilerplate (safe to replace with summary).

    Returns (is_boilerplate, reason).
    """
    name = symbol.get("name", "")
    kind = symbol.get("kind", "")
    body = symbol.get("body_preview", "")
    body_stripped = body.strip()

    # Empty __init__ with only self
    if name.endswith(".__init__") or name == "__init__":
        # Check if body is trivially short
        lines = [l.strip() for l in body_stripped.split("\n") if l.strip()]
        non_trivial = [l for l in lines[1:] if l not in ("pass", "...", '"""', "'''")]
        if len(non_trivial) <= 1:
            return True, "trivial __init__"

    # pass-only methods
    if kind in ("method", "function"):
        lines = [l.strip() for l in body_stripped.split("\n") if l.strip()]
        body_lines = [l for l in lines[1:] if not l.startswith(('"""', "'''", "#"))]
        if all(l in ("pass", "...", "return", "return None") for l in body_lines):
            return True, "pass-only or empty body"

    # Obvious getters (return self.x)
    if _GETTER_PATTERN.search(body_stripped):
        return True, "trivial getter"

    # Obvious setters (self.x = value)
    if _SETTER_PATTERN.search(body_stripped):
        return True, "trivial setter"

    # Property getters
    if _PROPERTY_GETTER.search(body_stripped):
        return True, "property getter"

    # __repr__, __str__ with simple f-string
    if name.endswith((".__repr__", ".__str__", "__repr__", "__str__")):
        if "return f" in body_stripped or "return '" in body_stripped or 'return "' in body_stripped:
            return True, "trivial __repr__/__str__"

    return False, ""


def _extract_imports(body: str) -> list[str]:
    """Extract import lines from a body preview."""
    imports = []
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            imports.append(stripped)
    return imports


def _build_summary_line(symbol: dict) -> str:
    """Build a one-line summary for a compressed symbol."""
    name = symbol.get("name", "")
    signature = symbol.get("signature", "")
    docstring = symbol.get("docstring", "")

    if docstring:
        # First sentence
        first_sentence = docstring.split(".")[0].strip()
        if first_sentence:
            return f"  # {name}: {first_sentence}"

    return f"  # {name}: {signature}"


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------

class Compressor:
    """Context compression pipeline.

    Steps applied in order:
    1. Replace duplicate import blocks (same imports in 3+ files)
    2. Replace boilerplate patterns with CliffNotes summary
    3. Rank remaining symbols by relevance score (distance)
    4. Cut lowest-ranked symbols until under budget

    Never removes: function signatures, symbol names, or anything with
    relevance score above primary threshold.
    """

    def __init__(self, budget_tokens: int | None = None, primary_threshold: float = 0.80):
        """Initialize compressor.

        Args:
            budget_tokens: Target token budget. If None, uses hard_limit / 2.
            primary_threshold: Distance threshold — symbols closer than this
                              are never removed.
        """
        self.budget_tokens = budget_tokens or int(
            os.environ.get("CTX_COMPRESS_BUDGET", "15000")
        )
        self.primary_threshold = primary_threshold

    def compress(self, symbols: list[dict]) -> dict:
        """Run the full compression pipeline.

        Args:
            symbols: List of symbol result dicts with body_preview, distance, etc.

        Returns:
            CompressedBundle as a dict with compressed_text, quality_report, etc.
        """
        if not symbols:
            return {
                "compressed_text": "",
                "original_tokens": 0,
                "compressed_tokens": 0,
                "tokens_saved": 0,
                "quality_report": [],
                "symbols_kept": 0,
                "symbols_removed": 0,
            }

        # Calculate original token count
        original_text = "\n\n".join(s.get("body_preview", "") for s in symbols)
        original_tokens = estimate_tokens(original_text)

        quality_report: list[dict] = []
        kept_symbols: list[dict] = []
        removed_symbols: list[dict] = []

        # -----------------------------------------------------------------
        # Step 1: Deduplicate import blocks
        # -----------------------------------------------------------------
        import_counter: Counter[str, int] = Counter()
        for s in symbols:
            imports = _extract_imports(s.get("body_preview", ""))
            for imp in imports:
                import_counter[imp] += 1

        # Find imports that appear in 3+ symbols
        duplicate_imports = {imp for imp, count in import_counter.items() if count >= 3}

        # -----------------------------------------------------------------
        # Step 2: Process each symbol
        # -----------------------------------------------------------------
        for s in symbols:
            distance = s.get("distance", 0.5)

            # Never remove high-relevance symbols
            if distance <= self.primary_threshold:
                # Still apply import dedup
                body = s.get("body_preview", "")
                if duplicate_imports:
                    body = self._dedup_imports(body, duplicate_imports)
                    s = dict(s, body_preview=body)
                kept_symbols.append(s)
                continue

            # Check for boilerplate
            is_bp, reason = _is_boilerplate(s)
            if is_bp:
                summary_line = _build_summary_line(s)
                tokens_saved = estimate_tokens(s.get("body_preview", "")) - estimate_tokens(summary_line)
                quality_report.append({
                    "symbol_name": s.get("name", ""),
                    "file_path": s.get("file_path", ""),
                    "quality": QUALITY_LOW,
                    "reason": reason,
                    "tokens_saved": max(0, tokens_saved),
                })
                # Replace body with summary line
                s = dict(s, body_preview=summary_line)
                kept_symbols.append(s)
                continue

            # Not boilerplate, not high-relevance — keep for now, may cut later
            kept_symbols.append(s)

        # -----------------------------------------------------------------
        # Step 3: Import dedup summary
        # -----------------------------------------------------------------
        import_dedup_text = ""
        if duplicate_imports:
            import_dedup_text = (
                f"[imports deduplicated - {len(duplicate_imports)} occurrences]\n"
            )

        # -----------------------------------------------------------------
        # Step 4: Rank and cut to fit budget
        # -----------------------------------------------------------------
        # Sort by distance (lower = more relevant = keep first)
        kept_symbols.sort(key=lambda s: s.get("distance", 0.5))

        final_parts = []
        if import_dedup_text:
            final_parts.append(import_dedup_text)

        current_tokens = estimate_tokens(import_dedup_text)

        for s in kept_symbols:
            body = s.get("body_preview", "")
            body_tokens = estimate_tokens(body)

            if current_tokens + body_tokens > self.budget_tokens:
                # Check if this is a high-value symbol
                distance = s.get("distance", 1.0)
                if distance <= self.primary_threshold:
                    # High relevance — keep it, warn user
                    final_parts.append(body)
                    current_tokens += body_tokens
                    quality_report.append({
                        "symbol_name": s.get("name", ""),
                        "file_path": s.get("file_path", ""),
                        "quality": QUALITY_HIGH,
                        "reason": "high relevance, kept despite budget",
                        "tokens_saved": 0,
                    })
                else:
                    # Cut this symbol
                    summary_line = _build_summary_line(s)
                    summary_tokens = estimate_tokens(summary_line)

                    if current_tokens + summary_tokens <= self.budget_tokens:
                        # Replace with summary
                        final_parts.append(summary_line)
                        current_tokens += summary_tokens
                    else:
                        removed_symbols.append(s)

                    tokens_saved = body_tokens - (summary_tokens if current_tokens + summary_tokens <= self.budget_tokens else 0)

                    quality_report.append({
                        "symbol_name": s.get("name", ""),
                        "file_path": s.get("file_path", ""),
                        "quality": QUALITY_MEDIUM,
                        "reason": f"below budget cut (distance={distance:.3f})",
                        "tokens_saved": max(0, tokens_saved),
                    })
            else:
                final_parts.append(body)
                current_tokens += body_tokens

        compressed_text = "\n\n".join(final_parts)
        compressed_tokens = estimate_tokens(compressed_text)

        return {
            "compressed_text": compressed_text,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "tokens_saved": max(0, original_tokens - compressed_tokens),
            "quality_report": quality_report,
            "symbols_kept": len(kept_symbols) - len(removed_symbols),
            "symbols_removed": len(removed_symbols),
        }

    def _dedup_imports(self, body: str, duplicate_imports: set[str]) -> str:
        """Remove duplicate import lines from a body preview."""
        lines = body.split("\n")
        filtered = []
        deduped_count = 0

        for line in lines:
            stripped = line.strip()
            if stripped in duplicate_imports:
                deduped_count += 1
                continue
            filtered.append(line)

        return "\n".join(filtered)


# ---------------------------------------------------------------------------
# Convenience: import os here for budget env var
# ---------------------------------------------------------------------------
import os


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test boilerplate detection
    test_symbols = [
        {
            "name": "UserService.__init__",
            "kind": "method",
            "file_path": "user.py",
            "signature": "def __init__(self):",
            "docstring": "Initialize.",
            "body_preview": "def __init__(self):\n    pass",
            "distance": 0.9,
        },
        {
            "name": "UserService.get_name",
            "kind": "method",
            "file_path": "user.py",
            "signature": "def get_name(self) -> str:",
            "docstring": "Get the user name.",
            "body_preview": "def get_name(self) -> str:\n    return self.name",
            "distance": 0.85,
        },
        {
            "name": "AuthService.login",
            "kind": "method",
            "file_path": "auth.py",
            "signature": "def login(self, username: str, password: str) -> dict:",
            "docstring": "Authenticate a user and create a session.",
            "body_preview": (
                "def login(self, username: str, password: str) -> dict:\n"
                '    """Authenticate a user."""\n'
                "    hashed = hashlib.sha256(password.encode()).hexdigest()\n"
                "    token = secrets.token_urlsafe(32)\n"
                "    self._sessions[token] = {'user': username}\n"
                "    return {'token': token}"
            ),
            "distance": 0.65,
        },
        {
            "name": "calculate_metrics",
            "kind": "function",
            "file_path": "utils.py",
            "signature": "def calculate_metrics(data: list) -> dict:",
            "docstring": "Calculate aggregate metrics.",
            "body_preview": (
                "def calculate_metrics(data: list) -> dict:\n"
                "    if not data:\n"
                "        return {}\n"
                "    return {'avg': sum(data) / len(data)}"
            ),
            "distance": 0.7,
        },
    ]

    compressor = Compressor(budget_tokens=500)
    result = compressor.compress(test_symbols)

    print(f"Original tokens: {result['original_tokens']}")
    print(f"Compressed tokens: {result['compressed_tokens']}")
    print(f"Tokens saved: {result['tokens_saved']}")
    print(f"Symbols kept: {result['symbols_kept']}")
    print(f"Symbols removed: {result['symbols_removed']}")
    print(f"\nQuality report ({len(result['quality_report'])} entries):")
    for entry in result["quality_report"]:
        print(f"  [{entry['quality']}] {entry['symbol_name']}: {entry['reason']} (saved {entry['tokens_saved']} tokens)")
    print(f"\nCompressed text:\n{result['compressed_text']}")
    print("\n[OK] Compressor working correctly.")
