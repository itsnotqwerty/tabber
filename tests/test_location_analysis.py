"""Tests for modules/location_analysis.py."""

from __future__ import annotations

from unittest.mock import patch

from tabber.models import GathererResult, LocationResult, OSINTBundle, PersonProfile
from tabber.modules import location_analysis


class TestAnalyse:
    def test_returns_location_result(self, bundle):
        expected = LocationResult(
            location="Paris, France",
            confidence=0.85,
            reasoning="Multiple sources confirmed.",
            sources=["news"],
        )
        with patch("tabber.llm.complete", return_value=expected):
            result = location_analysis.analyse(bundle)
        assert result.location == "Paris, France"
        assert result.confidence == 0.85

    def test_passes_location_result_as_response_format(self, bundle):
        expected = LocationResult(
            location="NYC", confidence=0.5, reasoning="x", sources=[]
        )
        with patch("tabber.llm.complete", return_value=expected) as mock_complete:
            location_analysis.analyse(bundle)
        assert mock_complete.call_args.kwargs["response_format"] is LocationResult

    def test_person_name_in_prompt(self, bundle):
        expected = LocationResult(
            location="NYC", confidence=0.5, reasoning="x", sources=[]
        )
        with patch("tabber.llm.complete", return_value=expected) as mock_complete:
            location_analysis.analyse(bundle)
        prompt = mock_complete.call_args.args[0]
        assert bundle.person.name in prompt

    def test_raw_text_snippets_included_in_prompt(self, bundle):
        expected = LocationResult(
            location="NYC", confidence=0.5, reasoning="x", sources=[]
        )
        with patch("tabber.llm.complete", return_value=expected) as mock_complete:
            location_analysis.analyse(bundle)
        prompt = mock_complete.call_args.args[0]
        # The fixture's gatherer_result raw_text contains "Paris conference"
        assert "Paris conference" in prompt

    def test_empty_results_sends_no_data_placeholder(self, person):
        empty_bundle = OSINTBundle(person=person, results=[], iteration=1)
        expected = LocationResult(
            location="Unknown", confidence=0.1, reasoning="No data.", sources=[]
        )
        with patch("tabber.llm.complete", return_value=expected) as mock_complete:
            result = location_analysis.analyse(empty_bundle)
        assert result.location == "Unknown"
        assert "no data gathered" in mock_complete.call_args.args[0].lower()

    def test_multiple_sources_all_appear_in_prompt(self, person):
        results = [
            GathererResult(source_name="news", raw_text="News text."),
            GathererResult(source_name="wikipedia", raw_text="Wiki text."),
        ]
        multi_bundle = OSINTBundle(person=person, results=results, iteration=1)
        expected = LocationResult(
            location="London", confidence=0.7, reasoning="x", sources=["news"]
        )
        with patch("tabber.llm.complete", return_value=expected) as mock_complete:
            location_analysis.analyse(multi_bundle)
        prompt = mock_complete.call_args.args[0]
        assert "News text." in prompt
        assert "Wiki text." in prompt
