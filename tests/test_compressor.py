"""Tests for compressor.py — boilerplate detection and budget enforcement."""

from __future__ import annotations

import pytest

from contextpilot.compressor import (
    Compressor,
    _is_boilerplate,
    _build_summary_line,
    _extract_imports,
)


def _sym(
    name: str = "my_func",
    kind: str = "function",
    body: str = "def my_func():\n    return 42",
    distance: float = 0.85,
    file_path: str = "foo.py",
    signature: str = "def my_func():",
    docstring: str | None = "Does something.",
) -> dict:
    return {
        "name": name,
        "kind": kind,
        "file_path": file_path,
        "signature": signature,
        "docstring": docstring,
        "body_preview": body,
        "distance": distance,
    }


class TestIsBoilerplate:
    def test_trivial_init_is_boilerplate(self):
        sym = _sym(name="MyClass.__init__", kind="method", body="def __init__(self):\n    pass")
        is_bp, reason = _is_boilerplate(sym)
        assert is_bp
        assert "init" in reason

    def test_pass_only_method_is_boilerplate(self):
        sym = _sym(name="noop", kind="method", body="def noop(self):\n    pass")
        is_bp, _ = _is_boilerplate(sym)
        assert is_bp

    def test_trivial_getter_is_boilerplate(self):
        sym = _sym(name="get_name", kind="method", body="def get_name(self) -> str:\n    return self.name")
        is_bp, reason = _is_boilerplate(sym)
        assert is_bp
        assert "getter" in reason

    def test_trivial_setter_is_boilerplate(self):
        sym = _sym(name="set_name", kind="method", body="def set_name(self, value: str):\n    self.name = value")
        is_bp, reason = _is_boilerplate(sym)
        assert is_bp
        assert "setter" in reason

    def test_trivial_repr_is_boilerplate(self):
        sym = _sym(name="__repr__", kind="method", body='def __repr__(self):\n    return f"Foo({self.x})"')
        is_bp, _ = _is_boilerplate(sym)
        assert is_bp

    def test_real_logic_not_boilerplate(self):
        sym = _sym(
            name="authenticate",
            kind="function",
            body="def authenticate(user, pwd):\n    hashed = hash(pwd)\n    if hashed != user.hash:\n        raise AuthError\n    return token.generate(user)",
        )
        is_bp, _ = _is_boilerplate(sym)
        assert not is_bp


class TestExtractImports:
    def test_detects_import_lines(self):
        body = "import os\nfrom pathlib import Path\ndef foo():\n    pass"
        imports = _extract_imports(body)
        assert "import os" in imports
        assert "from pathlib import Path" in imports

    def test_no_imports_returns_empty(self):
        assert _extract_imports("def foo():\n    return 1") == []


class TestBuildSummaryLine:
    def test_uses_docstring_first_sentence(self):
        sym = _sym(docstring="Validates the token. Extra details.")
        line = _build_summary_line(sym)
        assert "Validates the token." in line
        assert sym["name"] in line

    def test_falls_back_to_signature_when_no_docstring(self):
        sym = _sym(docstring=None, signature="def my_func(x: int):")
        line = _build_summary_line(sym)
        assert sym["name"] in line


class TestCompressorCompress:
    def test_empty_input_returns_zero_bundle(self):
        c = Compressor(budget_tokens=10000)
        result = c.compress([])
        assert result["compressed_text"] == ""
        assert result["original_tokens"] == 0
        assert result["symbols_kept"] == 0

    def test_high_relevance_symbols_always_kept(self):
        c = Compressor(budget_tokens=50)
        sym = _sym(distance=0.50, body="def critical_func():\n    return important_value")
        result = c.compress([sym])
        assert "critical_func" in result["compressed_text"]

    def test_boilerplate_replaced_not_removed(self):
        c = Compressor(budget_tokens=10000)
        sym = _sym(name="Foo.__init__", kind="method", body="def __init__(self):\n    pass", distance=0.90)
        result = c.compress([sym])
        low_entries = [e for e in result["quality_report"] if e["quality"] == "low"]
        assert len(low_entries) >= 1

    def test_quality_report_structure(self):
        c = Compressor(budget_tokens=10000)
        sym = _sym(name="Foo.__repr__", kind="method", body='def __repr__(self):\n    return f"Foo()"', distance=0.90)
        result = c.compress([sym])
        for entry in result["quality_report"]:
            assert "symbol_name" in entry
            assert "file_path" in entry
            assert "quality" in entry
            assert "reason" in entry
            assert "tokens_saved" in entry
            assert entry["quality"] in ("low", "medium", "high")

    def test_tokens_saved_non_negative(self):
        c = Compressor(budget_tokens=10000)
        syms = [_sym(f"fn_{i}", distance=0.8 + i * 0.02) for i in range(5)]
        result = c.compress(syms)
        assert result["tokens_saved"] >= 0

    def test_dedup_imports_reduces_repetition(self):
        common_import = "import os"
        syms = [
            _sym(f"fn_{i}", body=f"{common_import}\ndef fn_{i}():\n    pass", distance=0.85)
            for i in range(4)
        ]
        c = Compressor(budget_tokens=10000)
        result = c.compress(syms)
        # Compressed result should have fewer occurrences of the import than 4x
        import_count = result["compressed_text"].count(common_import)
        assert import_count < 4
