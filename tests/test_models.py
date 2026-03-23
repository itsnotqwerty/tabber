"""Tests for Pydantic data models in models.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import (
    GathererResult,
    HintsList,
    LocationResult,
    OSINTBundle,
    PersonProfile,
    SignalEvaluation,
)


class TestPersonProfile:
    def test_required_name_field(self):
        p = PersonProfile(name="Jane Doe")
        assert p.name == "Jane Doe"

    def test_optional_fields_default_to_empty(self):
        p = PersonProfile(name="Jane")
        assert p.aliases == []
        assert p.known_roles == []
        assert p.disambiguation_notes == ""

    def test_full_construction(self):
        p = PersonProfile(
            name="Jane Doe",
            aliases=["JD", "@janedoe"],
            known_roles=["CEO", "author"],
            disambiguation_notes="Notable person.",
        )
        assert len(p.aliases) == 2
        assert len(p.known_roles) == 2

    def test_missing_name_raises_validation_error(self):
        with pytest.raises(ValidationError):
            PersonProfile()  # type: ignore


class TestHintsList:
    def test_holds_list_of_strings(self):
        h = HintsList(hints=["London visit", "NYC conference"])
        assert h.hints == ["London visit", "NYC conference"]

    def test_empty_hints_allowed(self):
        h = HintsList(hints=[])
        assert h.hints == []

    def test_missing_hints_field_raises(self):
        with pytest.raises(ValidationError):
            HintsList()  # type: ignore


class TestGathererResult:
    def test_required_source_name(self):
        r = GathererResult(source_name="news")
        assert r.source_name == "news"

    def test_items_and_raw_text_default_empty(self):
        r = GathererResult(source_name="news")
        assert r.items == []
        assert r.raw_text == ""

    def test_full_construction(self):
        r = GathererResult(
            source_name="twitter",
            items=[{"text": "Jane is in Paris"}],
            raw_text="Jane is in Paris",
        )
        assert r.items[0]["text"] == "Jane is in Paris"


class TestSignalEvaluation:
    def test_fields(self):
        s = SignalEvaluation(confidence=0.75, reason="Sufficient evidence.")
        assert s.confidence == 0.75
        assert s.reason == "Sufficient evidence."

    def test_zero_confidence_allowed(self):
        s = SignalEvaluation(confidence=0.0, reason="No data.")
        assert s.confidence == 0.0

    def test_full_confidence_allowed(self):
        s = SignalEvaluation(confidence=1.0, reason="Certainty.")
        assert s.confidence == 1.0


class TestOSINTBundle:
    def test_defaults(self, person):
        b = OSINTBundle(person=person)
        assert b.results == []
        assert b.iteration == 0
        assert b.signal_evaluation is None

    def test_with_results_and_iteration(self, person, gatherer_result):
        b = OSINTBundle(person=person, results=[gatherer_result], iteration=2)
        assert len(b.results) == 1
        assert b.iteration == 2

    def test_signal_evaluation_optional(self, person, signal_eval):
        b = OSINTBundle(person=person, signal_evaluation=signal_eval)
        assert b.signal_evaluation.confidence == 0.9  # type: ignore


class TestLocationResult:
    def test_all_fields(self):
        r = LocationResult(
            location="London, UK",
            confidence=0.8,
            reasoning="Confirmed by news.",
            sources=["news", "twitter"],
        )
        assert r.location == "London, UK"
        assert r.confidence == 0.8
        assert "news" in r.sources

    def test_sources_defaults_to_empty(self):
        r = LocationResult(location="Unknown", confidence=0.1, reasoning="No data.")
        assert r.sources == []
