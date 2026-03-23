"""Tests for api.py — FastAPI REST server endpoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tabber import caching
from tabber import sqlite as db_module
from tabber.models import LocationResult, OSINTBundle, GathererResult, PersonProfile


# ─── setup ────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _fresh_conn(monkeypatch, tmp_path):
    """Point caching at a temp DB for each test."""
    db_path = str(tmp_path / "api_test.db")
    monkeypatch.setattr(
        "tabber.config.load",
        lambda: {"cache_ttl_hours": 24, "db_path": db_path},
    )
    caching._reset_conn()
    yield
    caching._reset_conn()


@pytest.fixture()
def client():
    from api import app

    return TestClient(app)


@pytest.fixture()
def stored_result(tmp_path):
    """Pre-populate the cache with one result and return (bundle, result)."""
    person = PersonProfile(name="Jane Doe", known_roles=["CEO"])
    bundle = OSINTBundle(
        person=person,
        results=[GathererResult(source_name="news", raw_text="Paris trip.")],
        iteration=1,
    )
    result = LocationResult(
        location="Paris, France",
        confidence=0.85,
        reasoning="Multiple sources.",
        sources=["news"],
    )
    caching.store("Jane Doe", bundle, result)
    return bundle, result


# ─── /health ──────────────────────────────────────────────────────────────────


class TestHealth:
    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ─── /lookup (cached path) ────────────────────────────────────────────────────


class TestLookupCached:
    """When a cached bundle exists it is used as a starting point for a fresh run."""

    def _make_fresh_result(self):
        person = PersonProfile(name="Jane Doe", known_roles=["CEO"])
        bundle = OSINTBundle(person=person, results=[], iteration=2)
        result = LocationResult(
            location="Paris, France",
            confidence=0.90,
            reasoning="Enriched result.",
            sources=["news"],
        )
        return bundle, result

    def test_prior_bundle_passed_to_identification(self, client, stored_result):
        bundle, result = self._make_fresh_result()
        with (
            patch("api.identification.run", return_value=bundle) as mock_run,
            patch("api.location_analysis.analyse", return_value=result),
        ):
            resp = client.post("/lookup", json={"name": "Jane Doe"})
        assert resp.status_code == 200
        assert mock_run.call_args.kwargs.get("prior_bundle") is not None

    def test_returns_fresh_not_cached(self, client, stored_result):
        bundle, result = self._make_fresh_result()
        with (
            patch("api.identification.run", return_value=bundle),
            patch("api.location_analysis.analyse", return_value=result),
        ):
            resp = client.post("/lookup", json={"name": "Jane Doe"})
        assert resp.status_code == 200
        assert resp.json()["cached"] is False

    def test_query_name_echoed(self, client, stored_result):
        bundle, result = self._make_fresh_result()
        with (
            patch("api.identification.run", return_value=bundle),
            patch("api.location_analysis.analyse", return_value=result),
        ):
            resp = client.post("/lookup", json={"name": "Jane Doe"})
        assert resp.json()["query_name"] == "Jane Doe"

    def test_canon_name_in_response(self, client, stored_result):
        bundle, result = self._make_fresh_result()
        with (
            patch("api.identification.run", return_value=bundle),
            patch("api.location_analysis.analyse", return_value=result),
        ):
            resp = client.post("/lookup", json={"name": "Jane Doe"})
        assert resp.json()["canon_name"] == "Jane Doe"

    def test_case_insensitive_cache_hit(self, client, stored_result):
        bundle, result = self._make_fresh_result()
        with (
            patch("api.identification.run", return_value=bundle),
            patch("api.location_analysis.analyse", return_value=result),
        ):
            resp = client.post("/lookup", json={"name": "jane doe"})
        assert resp.status_code == 200


# ─── /lookup (fresh path) ─────────────────────────────────────────────────────


class TestLookupFresh:
    def _mock_pipeline(self):
        person = PersonProfile(name="Jane Doe", known_roles=["CEO"])
        bundle = OSINTBundle(person=person, results=[], iteration=1)
        result = LocationResult(
            location="London, UK",
            confidence=0.75,
            reasoning="Event confirmed.",
            sources=["news"],
        )
        return bundle, result

    def test_no_cache_flag_bypasses_cache(self, client, stored_result):
        bundle, result = self._mock_pipeline()
        with (
            patch("api.identification.run", return_value=bundle),
            patch("api.location_analysis.analyse", return_value=result),
        ):
            resp = client.post("/lookup", json={"name": "Jane Doe", "no_cache": True})
        assert resp.status_code == 200
        body = resp.json()
        assert body["cached"] is False

    def test_fresh_result_stored(self, client):
        bundle, result = self._mock_pipeline()
        with (
            patch("api.identification.run", return_value=bundle),
            patch("api.location_analysis.analyse", return_value=result),
        ):
            resp = client.post("/lookup", json={"name": "Fresh Person"})
        assert resp.status_code == 200
        # Should now be in cache
        cached = caching.get_cached("Fresh Person")
        assert cached is not None

    def test_runtime_error_returns_502(self, client):
        with patch("api.identification.run", side_effect=RuntimeError("LLM down")):
            resp = client.post("/lookup", json={"name": "Bad Person", "no_cache": True})
        assert resp.status_code == 502
        assert "LLM down" in resp.json()["detail"]

    def test_empty_name_returns_422(self, client):
        resp = client.post("/lookup", json={"name": "   ", "no_cache": True})
        assert resp.status_code == 422

    def test_missing_name_returns_422(self, client):
        resp = client.post("/lookup", json={})
        assert resp.status_code == 422


# ─── GET /results ─────────────────────────────────────────────────────────────


class TestListResults:
    def test_empty_returns_empty_list(self, client):
        resp = client.get("/results")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_stored_results(self, client, stored_result):
        resp = client.get("/results")
        assert resp.status_code == 200
        rows = resp.json()
        assert len(rows) == 1
        assert rows[0]["location"] == "Paris, France"

    def test_limit_query_param(self, client, stored_result):
        # store one more
        person = PersonProfile(name="John Smith", known_roles=["actor"])
        bundle2 = OSINTBundle(person=person, results=[], iteration=1)
        result2 = LocationResult(
            location="NYC", confidence=0.6, reasoning="x", sources=[]
        )
        caching.store("John Smith", bundle2, result2)

        resp = client.get("/results?limit=1")
        assert len(resp.json()) == 1

    def test_ordered_newest_first(self, client):
        person = PersonProfile(name="Jane Doe", known_roles=[])
        bundle = OSINTBundle(person=person, results=[], iteration=1)
        r1 = LocationResult(
            location="Berlin", confidence=0.5, reasoning="old", sources=[]
        )
        r2 = LocationResult(
            location="Paris", confidence=0.9, reasoning="new", sources=[]
        )
        caching.store("Jane Doe", bundle, r1)
        caching.store("Jane Doe", bundle, r2)
        rows = client.get("/results").json()
        assert rows[0]["location"] == "Paris"

    def test_invalid_limit_returns_422(self, client):
        resp = client.get("/results?limit=0")
        assert resp.status_code == 422


# ─── GET /results/{name} ──────────────────────────────────────────────────────


class TestGetResultByName:
    def test_returns_404_when_not_found(self, client):
        resp = client.get("/results/Unknown%20Person")
        assert resp.status_code == 404

    def test_returns_result_for_known_name(self, client, stored_result):
        resp = client.get("/results/Jane%20Doe")
        assert resp.status_code == 200
        assert resp.json()["location"] == "Paris, France"

    def test_returns_most_recent(self, client):
        person = PersonProfile(name="Jane Doe", known_roles=[])
        bundle = OSINTBundle(person=person, results=[], iteration=1)
        r1 = LocationResult(
            location="Berlin", confidence=0.5, reasoning="old", sources=[]
        )
        r2 = LocationResult(
            location="Paris", confidence=0.9, reasoning="new", sources=[]
        )
        caching.store("Jane Doe", bundle, r1)
        caching.store("Jane Doe", bundle, r2)
        resp = client.get("/results/Jane%20Doe")
        assert resp.json()["location"] == "Paris"


# ─── DELETE /results/{name} ───────────────────────────────────────────────────


class TestDeleteResult:
    def test_delete_existing_returns_count(self, client, stored_result):
        resp = client.delete("/results/Jane%20Doe")
        assert resp.status_code == 200
        body = resp.json()
        assert body["deleted"] == 1
        assert body["name"] == "Jane Doe"

    def test_delete_nonexistent_returns_zero(self, client):
        resp = client.delete("/results/Nobody")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 0

    def test_result_gone_after_delete(self, client, stored_result):
        client.delete("/results/Jane%20Doe")
        resp = client.get("/results/Jane%20Doe")
        assert resp.status_code == 404
