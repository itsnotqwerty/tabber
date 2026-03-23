"""Shared pytest fixtures for the tabber test suite."""
from __future__ import annotations

import pytest

from models import (
    GathererResult,
    HintsList,
    LocationResult,
    OSINTBundle,
    PersonProfile,
    SignalEvaluation,
)


@pytest.fixture()
def person() -> PersonProfile:
    return PersonProfile(
        name="Jane Doe",
        aliases=["JD"],
        known_roles=["CEO of Acme Corp"],
        disambiguation_notes="A fictional public figure used in tests.",
    )


@pytest.fixture()
def gatherer_result() -> GathererResult:
    return GathererResult(
        source_name="news",
        items=[{"title": "Jane spotted in Paris", "body": "Seen at conference."}],
        raw_text="Jane Doe was seen at the Paris conference last week.",
    )


@pytest.fixture()
def bundle(person: PersonProfile, gatherer_result: GathererResult) -> OSINTBundle:
    return OSINTBundle(
        person=person,
        results=[gatherer_result],
        iteration=1,
    )


@pytest.fixture()
def signal_eval() -> SignalEvaluation:
    return SignalEvaluation(
        confidence=0.9,
        reason="Strong evidence from news articles.",
    )


@pytest.fixture()
def location_result() -> LocationResult:
    return LocationResult(
        location="Paris, France",
        confidence=0.85,
        reasoning="Multiple news sources place the subject in Paris.",
        sources=["news"],
    )


@pytest.fixture(autouse=True)
def _redirect_config(monkeypatch, tmp_path):
    """Redirect config file I/O to a temp directory so no test touches ~/.tabber."""
    import config as cfg_module

    config_dir = tmp_path / "tabber"
    monkeypatch.setattr(cfg_module, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cfg_module, "CONFIG_FILE", config_dir / "config.json")
