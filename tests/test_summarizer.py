"""Tests for summarizer.py — CliffNotes generation."""

from __future__ import annotations

import pytest

from contextpilot.extractor import Symbol
from contextpilot.summarizer import _extract_param_names, _first_sentence, generate_summary


# ---------------------------------------------------------------------------
# _extract_param_names
# ---------------------------------------------------------------------------

class TestExtractParamNames:
    def test_python_no_self(self):
        sig = "def foo(self, bar: str, baz: int = 5) -> bool:"
        result = _extract_param_names(sig)
        assert "bar" in result
        assert "baz" in result
        assert "self" not in result

    def test_python_cls_excluded(self):
        sig = "def create(cls, name: str):"
        result = _extract_param_names(sig)
        assert "name" in result
        assert "cls" not in result

    def test_javascript_style(self):
        sig = "function fetchUser(id, name)"
        result = _extract_param_names(sig)
        assert "id" in result
        assert "name" in result

    def test_typescript_typed(self):
        sig = "const validate = (email: string, age: number) =>"
        result = _extract_param_names(sig)
        assert "email" in result
        assert "age" in result

    def test_go_style(self):
        sig = "func (s *Server) Start(port int)"
        result = _extract_param_names(sig)
        assert "port" in result

    def test_no_params_returns_empty(self):
        sig = "def no_args():"
        result = _extract_param_names(sig)
        assert result == ""

    def test_no_parens_returns_empty(self):
        sig = "class Foo:"
        result = _extract_param_names(sig)
        assert result == ""


# ---------------------------------------------------------------------------
# _first_sentence
# ---------------------------------------------------------------------------

class TestFirstSentence:
    def test_stops_at_period(self):
        text = "Does something. And more stuff here."
        assert _first_sentence(text) == "Does something."

    def test_stops_at_exclamation(self):
        text = "Raises an error! Extra info."
        assert _first_sentence(text) == "Raises an error!"

    def test_stops_at_question(self):
        text = "Is this valid? More details."
        assert _first_sentence(text) == "Is this valid?"

    def test_none_input_returns_default(self):
        assert _first_sentence(None) == "No description"

    def test_empty_string_returns_default(self):
        assert _first_sentence("") == "No description"

    def test_long_text_without_punctuation_capped(self):
        text = "a" * 200
        result = _first_sentence(text)
        assert len(result) <= 80

    def test_whitespace_collapsed(self):
        text = "  Does   something.  Extra."
        assert _first_sentence(text) == "Does something."


# ---------------------------------------------------------------------------
# generate_summary
# ---------------------------------------------------------------------------

def _make_sym(name: str, sig: str, doc: str | None) -> Symbol:
    return Symbol(
        file_path="test.py",
        language="python",
        name=name,
        kind="function",
        signature=sig,
        docstring=doc,
        body_preview=f"{sig}\n    pass",
        start_line=1,
        end_line=2,
        body_hash="hash1",
    )


class TestGenerateSummary:
    def test_empty_symbols_fallback(self):
        result = generate_summary("test.py", [])
        assert "test.py" in result
        assert "No symbols" in result

    def test_one_symbol_with_docstring(self):
        sym = _make_sym("do_thing", "def do_thing(x: int):", "Performs an action. Extra.")
        result = generate_summary("test.py", [sym])
        assert "do_thing" in result
        assert "Performs an action." in result

    def test_one_symbol_without_docstring(self):
        sym = _make_sym("bare_func", "def bare_func(a, b):", None)
        result = generate_summary("test.py", [sym])
        assert "bare_func" in result

    def test_params_appear_in_summary(self):
        sym = _make_sym("compute", "def compute(alpha: float, beta: float):", "Computes result.")
        result = generate_summary("test.py", [sym])
        assert "alpha" in result
        assert "beta" in result

    def test_multiple_symbols_one_line_each(self):
        syms = [
            _make_sym("func_a", "def func_a():", "First function."),
            _make_sym("func_b", "def func_b(x):", "Second function."),
        ]
        result = generate_summary("test.py", syms)
        lines = [l for l in result.strip().splitlines() if l.strip()]
        assert len(lines) == 2

    def test_no_params_omits_parens_from_name(self):
        sym = _make_sym("init_app", "def init_app():", "Initialize app.")
        result = generate_summary("test.py", [sym])
        # When no params, format is "name: description"
        assert "init_app" in result
        assert "Initialize app." in result
