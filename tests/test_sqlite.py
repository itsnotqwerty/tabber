"""Tests for sqlite.py — persistence layer."""

from __future__ import annotations

import json
import sqlite3

import pytest

from tabber import sqlite as db_module
from tabber.models import LocationResult, OSINTBundle, PersonProfile, GathererResult


# ─── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_path):
    """Return a fresh in-memory-backed connection via tmp_path."""
    return db_module.init_db(tmp_path / "test.db")


@pytest.fixture()
def sample_bundle(person) -> OSINTBundle:
    return OSINTBundle(
        person=person,
        results=[GathererResult(source_name="news", raw_text="Seen in Paris.")],
        iteration=1,
    )


@pytest.fixture()
def sample_result() -> LocationResult:
    return LocationResult(
        location="Paris, France",
        confidence=0.85,
        reasoning="Multiple sources.",
        sources=["news"],
    )


# ─── init_db ──────────────────────────────────────────────────────────────────


class TestInitDb:
    def test_creates_file(self, tmp_path):
        db_path = tmp_path / "sub" / "results.db"
        conn = db_module.init_db(db_path)
        assert db_path.exists()
        conn.close()

    def test_returns_connection(self, tmp_path):
        conn = db_module.init_db(tmp_path / "results.db")
        assert isinstance(conn, sqlite3.Connection)
        conn.close()

    def test_creates_lookups_table(self, db):
        row = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='lookups'"
        ).fetchone()
        assert row is not None

    def test_idempotent(self, tmp_path):
        db_path = tmp_path / "results.db"
        conn1 = db_module.init_db(db_path)
        conn1.close()
        # Second call must not raise
        conn2 = db_module.init_db(db_path)
        conn2.close()


# ─── save_result ──────────────────────────────────────────────────────────────


class TestSaveResult:
    def test_returns_integer_id(self, db, sample_bundle, sample_result):
        row_id = db_module.save_result(db, "Jane Doe", sample_bundle, sample_result)
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_increments_id(self, db, sample_bundle, sample_result):
        id1 = db_module.save_result(db, "Jane Doe", sample_bundle, sample_result)
        id2 = db_module.save_result(db, "Jane Doe", sample_bundle, sample_result)
        assert id2 > id1

    def test_stored_fields_match(self, db, sample_bundle, sample_result):
        db_module.save_result(db, "jane doe", sample_bundle, sample_result)
        row = db.execute("SELECT * FROM lookups").fetchone()
        assert row["query_name"] == "jane doe"
        assert row["canon_name"] == sample_bundle.person.name
        assert row["location"] == sample_result.location
        assert row["confidence"] == pytest.approx(sample_result.confidence)
        assert json.loads(row["sources"]) == sample_result.sources

    def test_sources_stored_as_json(self, db, sample_bundle, sample_result):
        db_module.save_result(db, "Jane", sample_bundle, sample_result)
        raw = db.execute("SELECT sources FROM lookups").fetchone()["sources"]
        parsed = json.loads(raw)
        assert isinstance(parsed, list)

    def test_created_at_is_iso(self, db, sample_bundle, sample_result):
        db_module.save_result(db, "Jane", sample_bundle, sample_result)
        row = db.execute("SELECT created_at FROM lookups").fetchone()
        # Should be parseable as datetime
        from datetime import datetime

        dt = datetime.fromisoformat(row["created_at"])
        assert dt is not None


# ─── get_latest ───────────────────────────────────────────────────────────────


class TestGetLatest:
    def test_returns_none_when_empty(self, db):
        assert db_module.get_latest(db, "Unknown") is None

    def test_returns_row_for_canon_name(self, db, sample_bundle, sample_result):
        db_module.save_result(db, "jane", sample_bundle, sample_result)
        row = db_module.get_latest(db, sample_bundle.person.name)
        assert row is not None
        assert row["canon_name"] == sample_bundle.person.name

    def test_matches_query_name(self, db, sample_bundle, sample_result):
        db_module.save_result(db, "jd alias", sample_bundle, sample_result)
        row = db_module.get_latest(db, "jd alias")
        assert row is not None

    def test_case_insensitive_lookup(self, db, sample_bundle, sample_result):
        db_module.save_result(db, "Jane Doe", sample_bundle, sample_result)
        row = db_module.get_latest(db, "jane doe")
        assert row is not None

    def test_returns_most_recent_when_multiple(self, db, sample_bundle):
        old_result = LocationResult(
            location="Berlin, Germany", confidence=0.5, reasoning="old", sources=[]
        )
        new_result = LocationResult(
            location="Paris, France", confidence=0.9, reasoning="new", sources=["news"]
        )
        db_module.save_result(db, "Jane", sample_bundle, old_result)
        db_module.save_result(db, "Jane", sample_bundle, new_result)
        row = db_module.get_latest(db, "Jane")
        assert row["location"] == "Paris, France"  # type: ignore


# ─── list_all ─────────────────────────────────────────────────────────────────


class TestListAll:
    def test_empty_returns_empty_list(self, db):
        assert db_module.list_all(db) == []

    def test_returns_all_rows(self, db, sample_bundle, sample_result):
        db_module.save_result(db, "Jane", sample_bundle, sample_result)
        db_module.save_result(db, "Jane", sample_bundle, sample_result)
        rows = db_module.list_all(db)
        assert len(rows) == 2

    def test_ordered_newest_first(self, db, sample_bundle):
        r1 = LocationResult(
            location="Berlin, Germany", confidence=0.5, reasoning="old", sources=[]
        )
        r2 = LocationResult(
            location="Paris, France", confidence=0.9, reasoning="new", sources=[]
        )
        db_module.save_result(db, "Jane", sample_bundle, r1)
        db_module.save_result(db, "Jane", sample_bundle, r2)
        rows = db_module.list_all(db)
        assert rows[0]["location"] == "Paris, France"

    def test_limit_respected(self, db, sample_bundle, sample_result):
        for _ in range(5):
            db_module.save_result(db, "Jane", sample_bundle, sample_result)
        rows = db_module.list_all(db, limit=3)
        assert len(rows) == 3


# ─── delete_by_name ───────────────────────────────────────────────────────────


class TestDeleteByName:
    def test_returns_zero_when_nothing_deleted(self, db):
        assert db_module.delete_by_name(db, "Nonexistent") == 0

    def test_deletes_matching_rows(self, db, sample_bundle, sample_result):
        db_module.save_result(db, "Jane", sample_bundle, sample_result)
        db_module.save_result(db, "Jane", sample_bundle, sample_result)
        count = db_module.delete_by_name(db, sample_bundle.person.name)
        assert count == 2
        assert db_module.list_all(db) == []

    def test_case_insensitive_delete(self, db, sample_bundle, sample_result):
        db_module.save_result(db, "Jane Doe", sample_bundle, sample_result)
        count = db_module.delete_by_name(db, "jane doe")
        assert count == 1

    def test_does_not_delete_other_names(self, db, sample_bundle, sample_result):
        other_person = PersonProfile(name="John Smith", known_roles=["actor"])
        other_bundle = OSINTBundle(person=other_person, results=[], iteration=1)
        db_module.save_result(db, "Jane Doe", sample_bundle, sample_result)
        db_module.save_result(db, "John Smith", other_bundle, sample_result)
        db_module.delete_by_name(db, "Jane Doe")
        remaining = db_module.list_all(db)
        assert len(remaining) == 1
        assert remaining[0]["canon_name"] == "John Smith"
