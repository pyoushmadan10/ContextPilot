"""Semantic summary generator — the CliffNotes engine.

Generates purely structural summaries from extracted symbols.
Does NOT call any LLM. One line per symbol with parameter names
and the first sentence of the docstring.
"""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextpilot.extractor import Symbol


def _extract_param_names(signature: str) -> str:
    """Extract parameter names from a function/method signature.

    Handles Python, JavaScript/TypeScript, and Go signatures.

    Examples:
        "def foo(self, bar: str, baz: int = 5) -> bool:" -> "bar, baz"
        "function fetchUser(id, name)" -> "id, name"
        "const validate = (email: string) =>" -> "email"
        "func (s *Server) Start(port int)" -> "port"
    """
    # Find content within parentheses
    match = re.search(r'\(([^)]*)\)', signature)
    if not match:
        return ""

    params_str = match.group(1).strip()
    if not params_str:
        return ""

    params = []
    for param in params_str.split(","):
        param = param.strip()
        if not param:
            continue

        # Skip 'self', 'cls' for Python
        if param in ("self", "cls") or param.startswith("self:") or param.startswith("cls:"):
            continue

        # Extract just the name (before : or space for type annotations)
        # Python: "bar: str = 5" -> "bar"
        # TypeScript: "email: string" -> "email"
        # Go: "port int" -> "port"

        # Handle destructured params in JS/TS: { userId }
        if param.startswith("{") or param.startswith("["):
            params.append(param.split("}")[0].split("]")[0].strip("{ ["))
            continue

        # Handle Go receiver: "*Server" or "s *Server"
        if param.startswith("*"):
            continue

        # Split on colon (Python/TS type annotation) or space (Go)
        name = param.split(":")[0].split("=")[0].strip()
        # For Go, the name is before the type: "port int"
        name = name.split()[0] if " " in name else name

        # Remove pointer/reference markers
        name = name.lstrip("*&")

        if name and name.isidentifier():
            params.append(name)

    return ", ".join(params)


def _first_sentence(text: str | None) -> str:
    """Extract the first sentence from a docstring, or return a default."""
    if not text:
        return "No description"

    # Clean up whitespace
    text = " ".join(text.split())

    # Find first sentence-ending punctuation
    for i, char in enumerate(text):
        if char in ".!?" and i > 0:
            return text[:i + 1]

    # No sentence-ending punctuation — use first line (capped at 80 chars)
    first_line = text.split("\n")[0].strip()
    if len(first_line) > 80:
        return first_line[:77] + "..."
    return first_line


def generate_summary(file_path: str, symbols: list[Symbol]) -> str:
    """Generate a CliffNotes summary for a file.

    Purely structural — one line per symbol:
        "{name}({param_names}): {docstring_first_sentence}"

    Example output:
        validate_jwt(token, secret): Verifies JWT signature and expiry.
        UserService.create(email, password): Creates new user record in DB.

    Args:
        file_path: Path to the source file (for context).
        symbols: List of Symbol objects extracted from the file.

    Returns:
        Multi-line summary string.
    """
    if not symbols:
        return f"# {file_path}\nNo symbols extracted."

    lines = []
    for symbol in symbols:
        param_names = _extract_param_names(symbol.signature)
        doc_sentence = _first_sentence(symbol.docstring)

        # Format: name(params): description
        if param_names:
            lines.append(f"{symbol.name}({param_names}): {doc_sentence}")
        else:
            lines.append(f"{symbol.name}: {doc_sentence}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from contextpilot.extractor import extract_symbols

    if len(sys.argv) < 2:
        print("Usage: python -m contextpilot.summarizer <file_path>", file=sys.stderr)
        sys.exit(1)

    for fpath in sys.argv[1:]:
        symbols = extract_symbols(fpath)
        summary = generate_summary(fpath, symbols)
        print(f"\n--- CliffNotes: {fpath} ---")
        print(summary)
