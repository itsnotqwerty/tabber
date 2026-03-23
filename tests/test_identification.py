"""Tests for modules/identification.py."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tabber.models import (
    GathererResult,
    HintsList,
    OSINTBundle,
    PersonProfile,
    SignalEvaluation,
)
from tabber.modules.identification import (
    _disambiguate,
    _evaluate_signal,
    _generate_hints,
    _generate_verification_hints,
    run,
)


# ─── helpers ──────────────────────────────────────────────────────────────────


def _route_by_format(profile, hints_obj, eval_obj):
    """Returns a side_effect function that dispatches on response_format."""

    def _side_effect(prompt, system="", *, response_format=None):
        if response_format is PersonProfile:
            return profile
        if response_format is HintsList:
            return hints_obj
        if response_format is SignalEvaluation:
            return eval_obj
        return ""

    return _side_effect


# ─── _disambiguate ────────────────────────────────────────────────────────────


class TestDisambiguate:
    def test_returns_person_profile(self, person):
        with patch("tabber.llm.complete", return_value=person) as mock_complete:
            result = _disambiguate("Jane Doe")
        assert result is person
        assert mock_complete.call_args.kwargs["response_format"] is PersonProfile

    def test_name_appears_in_prompt(self, person):
        with patch("tabber.llm.complete", return_value=person) as mock_complete:
            _disambiguate("Unique Name XYZ")
        assert "Unique Name XYZ" in mock_complete.call_args.args[0]


# ─── _generate_hints ──────────────────────────────────────────────────────────


class TestGenerateHints:
    def test_returns_list_of_strings(self, person):
        hints_obj = HintsList(hints=["London", "NYC", "Paris", "Dubai", "Tokyo"])
        with patch("tabber.llm.complete", return_value=hints_obj) as mock_complete:
            result = _generate_hints(person, prior_bundle=None)
        assert result == hints_obj.hints
        assert mock_complete.call_args.kwargs["response_format"] is HintsList

    def test_no_prior_bundle_yields_none_yet_context(self, person):
        hints_obj = HintsList(hints=["h1"])
        with patch("tabber.llm.complete", return_value=hints_obj) as mock_complete:
            _generate_hints(person, prior_bundle=None)
        assert "none yet" in mock_complete.call_args.args[0].lower()

    def test_prior_bundle_raw_text_included_in_prompt(self, person, bundle):
        hints_obj = HintsList(hints=["h1"])
        with patch("tabber.llm.complete", return_value=hints_obj) as mock_complete:
            _generate_hints(person, prior_bundle=bundle)
        # The fixture's gatherer_result contains "Paris conference"
        assert "Paris conference" in mock_complete.call_args.args[0]


# ─── _generate_verification_hints ────────────────────────────────────────────


class TestGenerateVerificationHints:
    def test_returns_list_of_strings(self, person, signal_eval):
        hints_obj = HintsList(hints=["verify 1", "verify 2", "verify 3", "v4", "v5"])
        with patch("tabber.llm.complete", return_value=hints_obj):
            result = _generate_verification_hints(person, signal_eval)
        assert result == hints_obj.hints

    def test_evaluation_reason_in_prompt(self, person, signal_eval):
        hints_obj = HintsList(hints=["h1"])
        with patch("tabber.llm.complete", return_value=hints_obj) as mock_complete:
            _generate_verification_hints(person, signal_eval)
        assert signal_eval.reason in mock_complete.call_args.args[0]

    def test_uses_hints_response_format(self, person, signal_eval):
        hints_obj = HintsList(hints=["h1"])
        with patch("tabber.llm.complete", return_value=hints_obj) as mock_complete:
            _generate_verification_hints(person, signal_eval)
        assert mock_complete.call_args.kwargs["response_format"] is HintsList


# ─── _evaluate_signal ─────────────────────────────────────────────────────────


class TestEvaluateSignal:
    def test_returns_signal_evaluation(self, bundle):
        expected = SignalEvaluation(confidence=0.85, reason="Good data.")
        with patch("tabber.llm.complete", return_value=expected):
            result = _evaluate_signal(bundle)
        assert result.confidence == 0.85
        assert result.reason == "Good data."

    def test_uses_signal_evaluation_response_format(self, bundle):
        ev = SignalEvaluation(confidence=0.5, reason="ok")
        with patch("tabber.llm.complete", return_value=ev) as mock_complete:
            _evaluate_signal(bundle)
        assert mock_complete.call_args.kwargs["response_format"] is SignalEvaluation

    def test_empty_results_returns_zero_confidence(self, person):
        empty = OSINTBundle(person=person, results=[], iteration=1)
        result = _evaluate_signal(empty)
        assert result.confidence == 0.0
        assert result.reason == "No gathered text"

    def test_blank_raw_text_returns_zero_confidence(self, person):
        blank_bundle = OSINTBundle(
            person=person,
            results=[GathererResult(source_name="news", raw_text="   ")],
            iteration=1,
        )
        result = _evaluate_signal(blank_bundle)
        assert result.confidence == 0.0

    def test_normalises_100_scale_confidence(self, bundle):
        scaled = SignalEvaluation(confidence=87.0, reason="scaled")
        with patch("tabber.llm.complete", return_value=scaled):
            result = _evaluate_signal(bundle)
        assert result.confidence == pytest.approx(0.87)

    def test_clamps_confidence_at_1_after_normalisation(self, bundle):
        over = SignalEvaluation(confidence=110.0, reason="over")
        with patch("tabber.llm.complete", return_value=over):
            result = _evaluate_signal(bundle)
        assert result.confidence == 1.0

    def test_clamps_confidence_cannot_be_negative(self, bundle):
        neg = SignalEvaluation(confidence=-0.1, reason="negative")
        with patch("tabber.llm.complete", return_value=neg):
            result = _evaluate_signal(bundle)
        assert result.confidence == 0.0

    def test_llm_failure_with_large_text_returns_assumed_high_confidence(self, person):
        large = GathererResult(source_name="news", raw_text="x" * 600)
        big_bundle = OSINTBundle(person=person, results=[large], iteration=1)
        with patch("tabber.llm.complete", side_effect=Exception("API timeout")):
            result = _evaluate_signal(big_bundle)
        assert result.confidence == 0.85

    def test_llm_failure_with_small_text_returns_zero_confidence(self, person):
        small = GathererResult(source_name="news", raw_text="tiny")
        small_bundle = OSINTBundle(person=person, results=[small], iteration=1)
        with patch("tabber.llm.complete", side_effect=Exception("API error")):
            result = _evaluate_signal(small_bundle)
        assert result.confidence == 0.0


# ─── run ──────────────────────────────────────────────────────────────────────


class TestRun:
    def test_returns_osint_bundle(self, person):
        hints_obj = HintsList(hints=["h1", "h2", "h3", "h4", "h5"])
        eval_low = SignalEvaluation(confidence=0.2, reason="not enough")
        gather_result = GathererResult(
            source_name="news", raw_text="Jane seen in Paris."
        )
        bundle_out = OSINTBundle(person=person, results=[gather_result], iteration=1)

        with patch(
            "tabber.llm.complete",
            side_effect=_route_by_format(person, hints_obj, eval_low),
        ), patch(
            "tabber.modules.information_gathering.gather", return_value=bundle_out
        ):
            result = run("Jane Doe", max_iter=1)

        assert isinstance(result, OSINTBundle)
        assert result.person.name == "Jane Doe"

    def test_exits_after_verification_when_confidence_high(self, person):
        hints_obj = HintsList(hints=["h1", "h2", "h3", "h4", "h5"])
        # confidence ≥ 0.80 triggers verification; ≥ 0.85 exits the loop
        eval_high = SignalEvaluation(confidence=0.9, reason="great evidence")
        gather_result = GathererResult(source_name="news", raw_text="x" * 100)
        bundle_out = OSINTBundle(person=person, results=[gather_result], iteration=1)

        with patch(
            "tabber.llm.complete",
            side_effect=_route_by_format(person, hints_obj, eval_high),
        ), patch(
            "tabber.modules.information_gathering.gather", return_value=bundle_out
        ) as mock_gather:
            run("Jane Doe", max_iter=3)

        # One main gather + one verification gather, then exits
        assert mock_gather.call_count == 2

    def test_respects_max_iter_cap(self, person):
        hints_obj = HintsList(hints=["h1", "h2", "h3", "h4", "h5"])
        eval_low = SignalEvaluation(confidence=0.1, reason="nothing yet")
        gather_result = GathererResult(source_name="news", raw_text="")
        bundle_out = OSINTBundle(person=person, results=[gather_result], iteration=1)

        with patch(
            "tabber.llm.complete",
            side_effect=_route_by_format(person, hints_obj, eval_low),
        ), patch(
            "tabber.modules.information_gathering.gather", return_value=bundle_out
        ) as mock_gather:
            run("Jane Doe", max_iter=2)

        assert mock_gather.call_count == 2

    def test_bundle_includes_signal_evaluation(self, person):
        hints_obj = HintsList(hints=["h1", "h2", "h3", "h4", "h5"])
        eval_obj = SignalEvaluation(confidence=0.3, reason="partial")
        gather_result = GathererResult(source_name="news", raw_text="some text")
        bundle_out = OSINTBundle(person=person, results=[gather_result], iteration=1)

        with patch(
            "tabber.llm.complete",
            side_effect=_route_by_format(person, hints_obj, eval_obj),
        ), patch(
            "tabber.modules.information_gathering.gather", return_value=bundle_out
        ):
            result = run("Jane Doe", max_iter=1)

        assert result.signal_evaluation is not None
        assert result.signal_evaluation.confidence == pytest.approx(0.3)

    def test_prior_bundle_used_as_initial_hint_context(self, person, bundle):
        """When prior_bundle is provided, first hint generation receives it as context."""
        hints_obj = HintsList(hints=["h1", "h2", "h3", "h4", "h5"])
        eval_low = SignalEvaluation(confidence=0.2, reason="not enough")
        gather_result = GathererResult(source_name="news", raw_text="Paris trip.")
        bundle_out = OSINTBundle(person=person, results=[gather_result], iteration=2)

        captured_hints_calls = []

        def _side_effect(prompt, system="", *, response_format=None):
            if response_format is PersonProfile:
                return person
            if response_format is HintsList:
                captured_hints_calls.append(prompt)
                return hints_obj
            if response_format is SignalEvaluation:
                return eval_low
            return ""

        with patch("tabber.llm.complete", side_effect=_side_effect), patch(
            "tabber.modules.information_gathering.gather", return_value=bundle_out
        ):
            run("Jane Doe", max_iter=1, prior_bundle=bundle)

        # The fixture bundle's gatherer_result raw_text should appear in the first hints prompt
        assert any(
            "Paris conference" in call for call in captured_hints_calls
        ), "prior_bundle raw_text should seed the first hints prompt"
