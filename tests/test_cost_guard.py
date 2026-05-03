"""Tests for cost_guard.py — token estimation, CostGuard, and SessionSavings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextpilot.cost_guard import (
    CostGuard,
    GuardResult,
    SessionSavings,
    estimate_cost_usd,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty_string_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_none_like_empty_returns_zero(self):
        # The function guards on falsy input
        assert estimate_tokens("") == 0

    def test_non_empty_returns_positive(self):
        count = estimate_tokens("hello world")
        assert count > 0

    def test_longer_text_more_tokens(self):
        short = estimate_tokens("hi")
        long = estimate_tokens("hi " * 100)
        assert long > short

    def test_code_text(self):
        code = "def foo(bar: int) -> str:\n    return str(bar)\n"
        assert estimate_tokens(code) > 5


# ---------------------------------------------------------------------------
# estimate_cost_usd
# ---------------------------------------------------------------------------

class TestEstimateCostUsd:
    def test_zero_tokens_zero_cost(self):
        assert estimate_cost_usd(0) == 0.0

    def test_uncached_uses_higher_rate(self):
        uncached = estimate_cost_usd(1_000_000, cached=False)
        cached = estimate_cost_usd(1_000_000, cached=True)
        assert uncached > cached

    def test_one_million_tokens_uncached(self):
        # $3.00 per 1M tokens
        cost = estimate_cost_usd(1_000_000, cached=False)
        assert abs(cost - 3.00) < 0.01

    def test_one_million_tokens_cached(self):
        # $0.30 per 1M tokens
        cost = estimate_cost_usd(1_000_000, cached=True)
        assert abs(cost - 0.30) < 0.01


# ---------------------------------------------------------------------------
# CostGuard.check
# ---------------------------------------------------------------------------

class TestCostGuardCheck:
    def test_short_text_passes(self):
        guard = CostGuard(warn_threshold=500, hard_limit=1000)
        result = guard.check("hello")
        assert result.action == "pass"

    def test_text_above_warn_threshold(self):
        guard = CostGuard(warn_threshold=2, hard_limit=10000)
        # 3 tokens is above threshold of 2
        result = guard.check("one two three")
        assert result.action in ("warn", "interrupt")

    def test_text_above_hard_limit_interrupts(self):
        guard = CostGuard(warn_threshold=1, hard_limit=2)
        # Force a big token count above hard limit
        long_text = "word " * 500
        result = guard.check(long_text)
        assert result.action == "interrupt"

    def test_guard_result_has_required_fields(self):
        guard = CostGuard(warn_threshold=500, hard_limit=1000)
        result = guard.check("hello")
        assert isinstance(result.token_count, int)
        assert isinstance(result.estimated_cost_usd, float)
        assert result.action in ("pass", "warn", "interrupt")
        assert isinstance(result.message, str)

    def test_warn_threshold_boundary(self):
        """Text right at the warn threshold should warn, not interrupt."""
        guard = CostGuard(warn_threshold=100, hard_limit=10000)
        # Craft text that produces ~120 tokens (above warn, below hard)
        text = "token " * 120
        result = guard.check(text)
        assert result.action in ("warn",)

    def test_env_defaults_applied(self, monkeypatch):
        """CostGuard reads env vars when no explicit thresholds given."""
        monkeypatch.setenv("CTX_WARN_THRESHOLD", "50")
        monkeypatch.setenv("CTX_HARD_LIMIT", "100")
        guard = CostGuard()
        assert guard.warn_threshold == 50
        assert guard.hard_limit == 100


# ---------------------------------------------------------------------------
# SessionSavings
# ---------------------------------------------------------------------------

class TestSessionSavings:
    def test_initial_state_is_zeroed(self, tmp_path):
        ss = SessionSavings(tmp_path)
        summary = ss.summary
        assert summary["turns"] == 0
        assert summary["tokens_raw"] == 0
        assert summary["tokens_sent"] == 0
        assert summary["tokens_saved"] == 0

    def test_record_turn_accumulates(self, tmp_path):
        ss = SessionSavings(tmp_path)
        ss.record_turn(tokens_raw=1000, tokens_sent=200, turn_id=1)
        ss.record_turn(tokens_raw=500, tokens_sent=100, turn_id=2)

        summary = ss.summary
        assert summary["tokens_raw"] == 1500
        assert summary["tokens_sent"] == 300
        assert summary["tokens_saved"] == 1200
        assert summary["turns"] == 2

    def test_record_turn_same_id_aggregates(self, tmp_path):
        """Multiple tool calls in the same turn (same turn_id) should aggregate."""
        ss = SessionSavings(tmp_path)
        ss.record_turn(tokens_raw=500, tokens_sent=100, turn_id=1)
        ss.record_turn(tokens_raw=300, tokens_sent=50, turn_id=1)

        # Turn count should still be 1
        assert ss.summary["turns"] == 1
        assert ss.summary["tokens_raw"] == 800

    def test_saves_to_disk(self, tmp_path):
        ss = SessionSavings(tmp_path)
        ss.record_turn(tokens_raw=100, tokens_sent=20, turn_id=1)
        ss.save()

        stats_path = tmp_path / "session_stats.json"
        assert stats_path.exists()
        data = json.loads(stats_path.read_text())
        assert data["tokens_raw"] == 100

    def test_record_traversal_appends(self, tmp_path):
        ss = SessionSavings(tmp_path)
        ss.record_traversal({"turn": 1, "cost_this_turn_usd": 0.001})
        assert len(ss.data["traversals"]) == 1

    def test_record_traversal_caps_at_50(self, tmp_path):
        ss = SessionSavings(tmp_path)
        for i in range(55):
            ss.record_traversal({"turn": i, "cost_this_turn_usd": 0.0})
        assert len(ss.data["traversals"]) == 50

    def test_cost_saved_usd_positive(self, tmp_path):
        ss = SessionSavings(tmp_path)
        ss.record_turn(tokens_raw=100_000, tokens_sent=10_000, turn_id=1)
        assert ss.summary["cost_saved_usd"] > 0
