"""Tests for caching.py — TTL-aware result recall."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import caching
import sqlite as db_module
from models import LocationResult, OSINTBundle, GathererResult


# ─── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _fresh_conn(monkeypatch, tmp_path):
    """Redirect caching to a temp DB and reset module-level connection before each test."""
    db_path = str(tmp_path / "test_cache.db")
    monkeypatch.setattr(
        "config.load",
        lambda: {
            "cache_ttl_hours": 24,
            "db_path": db_path,
        },
    )
    caching._reset_conn()
    yield
    caching._reset_conn()


@pytest.fixture()
def result() -> LocationResult:
    return LocationResult(
        location="Paris, France",
        confidence=0.85,
        reasoning="Multiple sources.",
        sources=["news"],
    )


@pytest.fixture()
def bundle(person) -> OSINTBundle:
    return OSINTBundle(
        person=person,
        results=[GathererResult(source_name="news", raw_text="Seen in Paris.")],
        iteration=1,
    )


# ─── get_cached ───────────────────────────────────────────────────────────────


class TestGetCached:
    def test_returns_none_when_no_entry(self):
        assert caching.get_cached("Nobody") is None

    def test_returns_result_when_fresh(self, bundle, result):
        caching.store("Jane Doe", bundle, result)
        cached = caching.get_cached("Jane Doe")
        assert cached is not None
        assert cached.location == result.location

    def test_returns_none_when_expired(self, bundle, result, monkeypatch):
        caching.store("Jane Doe", bundle, result)
        # Make the stored row look old by patching datetime.now inside caching
        old_time = datetime.now(timezone.utc) - timedelta(hours=48)

        original_fromisoformat = datetime.fromisoformat

        def fake_now(tz=None):
            return old_time + timedelta(hours=48)  # effectively now is 48h later

        # Patch at the caching module level
        with patch("caching.datetime") as mock_dt:
            mock_dt.fromisoformat.side_effect = original_fromisoformat
            mock_dt.now.return_value = datetime.now(timezone.utc) + timedelta(hours=48)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            cached = caching.get_cached("Jane Doe")
        assert cached is None

    def test_case_insensitive_lookup(self, bundle, result):
        caching.store("Jane Doe", bundle, result)
        assert caching.get_cached("jane doe") is not None

    def test_result_fields_match(self, bundle, result):
        caching.store("Test Person", bundle, result)
        cached = caching.get_cached("Test Person")
        assert cached.location == result.location  # type: ignore
        assert cached.confidence == pytest.approx(result.confidence)  # type: ignore
        assert cached.reasoning == result.reasoning  # type: ignore
        assert cached.sources == result.sources  # type: ignore

    def test_ttl_zero_always_stale(self, bundle, result, monkeypatch, tmp_path):
        db_path = str(tmp_path / "zero_ttl.db")
        monkeypatch.setattr(
            "config.load",
            lambda: {"cache_ttl_hours": 0, "db_path": db_path},
        )
        caching._reset_conn()
        caching.store("Jane Doe", bundle, result)
        # TTL=0 means any age is expired
        assert caching.get_cached("Jane Doe") is None


# ─── store ────────────────────────────────────────────────────────────────────


class TestStore:
    def test_persists_to_db(self, bundle, result):
        caching.store("Jane Doe", bundle, result)
        conn = caching._get_conn()
        row = db_module.get_latest(conn, "Jane Doe")
        assert row is not None
        assert row["location"] == "Paris, France"

    def test_multiple_stores_accumulate(self, bundle, result):
        caching.store("Jane Doe", bundle, result)
        caching.store("Jane Doe", bundle, result)
        conn = caching._get_conn()
        rows = db_module.list_all(conn)
        assert len(rows) == 2


# ─── invalidate ───────────────────────────────────────────────────────────────


class TestInvalidate:
    def test_returns_zero_when_nothing_to_delete(self):
        assert caching.invalidate("Nobody") == 0

    def test_removes_entries(self, bundle, result):
        caching.store("Jane Doe", bundle, result)
        caching.store("Jane Doe", bundle, result)
        deleted = caching.invalidate("Jane Doe")
        assert deleted == 2
        assert caching.get_cached("Jane Doe") is None

    def test_case_insensitive_invalidate(self, bundle, result):
        caching.store("Jane Doe", bundle, result)
        assert caching.invalidate("JANE DOE") == 1
