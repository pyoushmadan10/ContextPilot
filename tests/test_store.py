"""Tests for store.py — VectorStore CRUD, search, and lifecycle."""

from __future__ import annotations

import numpy as np
import pytest

from contextpilot.store import VectorStore, CTX_PRIMARY_THRESHOLD, CTX_SUMMARY_THRESHOLD
from tests.conftest import _fake_vector, _make_symbol


# ---------------------------------------------------------------------------
# Basic symbol CRUD
# ---------------------------------------------------------------------------

class TestUpsertAndFind:
    def test_upsert_then_find_by_file_exact(self, tmp_store):
        sym = _make_symbol("do_thing", "src/foo.py")
        tmp_store.upsert_symbol(sym, _fake_vector(0))
        results = tmp_store.find_symbols_by_file("src/foo.py")
        assert len(results) == 1
        assert results[0]["name"] == "do_thing"

    def test_find_by_file_suffix_match(self, tmp_store):
        """'foo.py' should match 'src/foo.py' via LIKE fallback."""
        sym = _make_symbol("helper", "src/utils/foo.py")
        tmp_store.upsert_symbol(sym, _fake_vector(1))
        results = tmp_store.find_symbols_by_file("foo.py")
        assert len(results) == 1

    def test_find_by_file_no_match(self, tmp_store):
        results = tmp_store.find_symbols_by_file("nonexistent.py")
        assert results == []

    def test_upsert_updates_existing(self, tmp_store):
        sym = _make_symbol("do_thing", "src/foo.py", body_preview="def do_thing():\n    return 1")
        tmp_store.upsert_symbol(sym, _fake_vector(0))

        updated = _make_symbol("do_thing", "src/foo.py", body_preview="def do_thing():\n    return 99")
        tmp_store.upsert_symbol(updated, _fake_vector(0))

        results = tmp_store.find_symbols_by_file("src/foo.py")
        assert len(results) == 1  # No duplicate
        assert "99" in results[0]["body_preview"]

    def test_upsert_batch_inserts_all(self, tmp_store):
        syms = [_make_symbol(f"fn_{i}", "batch.py") for i in range(5)]
        vecs = [_fake_vector(i) for i in range(5)]
        tmp_store.upsert_symbols_batch(syms, vecs)
        results = tmp_store.find_symbols_by_file("batch.py")
        assert len(results) == 5

    def test_upsert_batch_length_mismatch_raises(self, tmp_store):
        syms = [_make_symbol("fn_a", "x.py")]
        vecs = [_fake_vector(0), _fake_vector(1)]
        with pytest.raises(ValueError, match="same length"):
            tmp_store.upsert_symbols_batch(syms, vecs)


# ---------------------------------------------------------------------------
# find_symbols_by_name
# ---------------------------------------------------------------------------

class TestFindByName:
    def test_exact_name_match(self, populated_store):
        results = populated_store.find_symbols_by_name("authenticate")
        assert len(results) >= 1
        assert results[0]["name"] == "authenticate"

    def test_partial_name_match(self, populated_store):
        """'save' should match 'UserModel.save' via LIKE."""
        results = populated_store.find_symbols_by_name("save")
        names = [r["name"] for r in results]
        assert any("save" in n for n in names)

    def test_no_name_match(self, populated_store):
        results = populated_store.find_symbols_by_name("zzz_nonexistent")
        assert results == []


# ---------------------------------------------------------------------------
# delete_symbols_for_file
# ---------------------------------------------------------------------------

class TestDeleteSymbols:
    def test_delete_removes_all_symbols_for_file(self, tmp_store):
        for i in range(3):
            tmp_store.upsert_symbol(_make_symbol(f"fn_{i}", "victim.py"), _fake_vector(i))

        count = tmp_store.delete_symbols_for_file("victim.py")
        assert count == 3
        assert tmp_store.find_symbols_by_file("victim.py") == []

    def test_delete_nonexistent_file_returns_zero(self, tmp_store):
        count = tmp_store.delete_symbols_for_file("ghost.py")
        assert count == 0

    def test_delete_leaves_other_files_intact(self, populated_store):
        populated_store.delete_symbols_for_file("auth.py")
        # models.py symbols should still be there
        results = populated_store.find_symbols_by_file("models.py")
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Summaries (CliffNotes)
# ---------------------------------------------------------------------------

class TestSummaryCRUD:
    def test_set_and_get_summary(self, tmp_store):
        tmp_store.set_summary("src/foo.py", "This is a summary.")
        result = tmp_store.get_summary("src/foo.py")
        assert result == "This is a summary."

    def test_get_missing_summary_returns_none(self, tmp_store):
        assert tmp_store.get_summary("missing.py") is None

    def test_delete_summary(self, tmp_store):
        tmp_store.set_summary("src/foo.py", "Summary here.")
        tmp_store.delete_summary("src/foo.py")
        assert tmp_store.get_summary("src/foo.py") is None

    def test_upsert_summary_replaces_existing(self, tmp_store):
        tmp_store.set_summary("src/foo.py", "Old summary.")
        tmp_store.set_summary("src/foo.py", "New summary.")
        assert tmp_store.get_summary("src/foo.py") == "New summary."


# ---------------------------------------------------------------------------
# Import edges
# ---------------------------------------------------------------------------

class TestImportEdges:
    def test_upsert_and_get_imports(self, tmp_store):
        tmp_store.upsert_imports("a.py", ["b.py", "c.py"])
        result = tmp_store.get_imports("a.py")
        assert set(result) == {"b.py", "c.py"}

    def test_get_imported_by(self, tmp_store):
        tmp_store.upsert_imports("a.py", ["shared.py"])
        tmp_store.upsert_imports("b.py", ["shared.py"])
        importers = tmp_store.get_imported_by("shared.py")
        assert set(importers) == {"a.py", "b.py"}

    def test_upsert_replaces_existing_edges(self, tmp_store):
        tmp_store.upsert_imports("a.py", ["old.py"])
        tmp_store.upsert_imports("a.py", ["new.py"])
        result = tmp_store.get_imports("a.py")
        assert result == ["new.py"]
        assert "old.py" not in result

    def test_delete_imports(self, tmp_store):
        tmp_store.upsert_imports("a.py", ["b.py"])
        tmp_store.delete_imports("a.py")
        assert tmp_store.get_imports("a.py") == []

    def test_no_imports_returns_empty(self, tmp_store):
        assert tmp_store.get_imports("unknown.py") == []


# ---------------------------------------------------------------------------
# Search with tiers
# ---------------------------------------------------------------------------

class TestSearchWithTiers:
    def test_search_returns_tiered_structure(self, populated_store):
        qvec = _fake_vector(0)  # Same vector as 'authenticate' → should be primary
        result = populated_store.search_with_tiers(qvec, top_k=10)
        assert "primary" in result
        assert "summaries" in result
        assert "dropped" in result

    def test_primary_tier_distance_within_threshold(self, populated_store):
        """All primary results must have distance <= CTX_PRIMARY_THRESHOLD."""
        qvec = _fake_vector(0)
        result = populated_store.search_with_tiers(qvec, top_k=10)
        for item in result["primary"]:
            assert item["distance"] <= CTX_PRIMARY_THRESHOLD

    def test_summary_tier_distance_within_threshold(self, populated_store):
        """All summary results must have distance in (PRIMARY, SUMMARY]."""
        qvec = _fake_vector(99)  # Unrelated vector → likely summaries or dropped
        result = populated_store.search_with_tiers(qvec, top_k=10)
        for item in result["summaries"]:
            assert item["distance"] <= CTX_SUMMARY_THRESHOLD

    def test_raw_search_returns_list(self, populated_store):
        result = populated_store.search(_fake_vector(0), top_k=3)
        assert isinstance(result, list)
        assert len(result) <= 3


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_empty_store_stats(self, tmp_store):
        stats = tmp_store.get_stats()
        assert stats["total_symbols"] == 0
        assert stats["total_files"] == 0
        assert stats["total_summaries"] == 0

    def test_stats_after_insert(self, populated_store):
        populated_store.set_summary("auth.py", "Auth summary.")
        stats = populated_store.get_stats()
        assert stats["total_symbols"] == 3
        assert stats["total_files"] == 2
        assert stats["total_summaries"] == 1


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_context_manager_closes_cleanly(self, tmp_path):
        with VectorStore(str(tmp_path)) as store:
            sym = _make_symbol("fn", "x.py")
            store.upsert_symbol(sym, _fake_vector(0))
        # After __exit__, connection is None
        assert store._conn is None

    def test_double_close_is_safe(self, tmp_path):
        store = VectorStore(str(tmp_path))
        store.close()
        store.close()  # Second close should not raise
