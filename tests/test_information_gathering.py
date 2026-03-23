"""Tests for modules/information_gathering.py."""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest

from tabber.models import GathererResult, OSINTBundle, PersonProfile
from tabber.modules import information_gathering


def _result(name: str) -> GathererResult:
    return GathererResult(source_name=name, raw_text=f"{name} gathered data")


def _patch_enabled_gatherers(cfg: dict):
    """Context manager that patches gather() on all gatherers is_configured for cfg."""
    patches = [
        patch.object(g, "gather", return_value=_result(g.name))
        for g in information_gathering._ALL_GATHERERS
        if g.is_configured(cfg)
    ]

    @contextlib.contextmanager
    def ctx():
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            yield

    return ctx()


class TestGather:
    def test_returns_osint_bundle(self, person, monkeypatch):
        cfg = {"llm_provider": "openai"}
        monkeypatch.setattr("tabber.config.load", lambda: cfg)
        with _patch_enabled_gatherers(cfg):
            bundle = information_gathering.gather(person, ["hint1"])
        assert isinstance(bundle, OSINTBundle)
        assert bundle.person == person

    def test_iteration_stored_in_bundle(self, person, monkeypatch):
        cfg = {"llm_provider": "openai"}
        monkeypatch.setattr("tabber.config.load", lambda: cfg)
        with _patch_enabled_gatherers(cfg):
            bundle = information_gathering.gather(person, [], iteration=3)
        assert bundle.iteration == 3

    def test_results_contain_enabled_gatherer_names(self, person, monkeypatch):
        cfg = {"llm_provider": "openai"}
        monkeypatch.setattr("tabber.config.load", lambda: cfg)
        enabled = {
            g.name for g in information_gathering._ALL_GATHERERS if g.is_configured(cfg)
        }
        with _patch_enabled_gatherers(cfg):
            bundle = information_gathering.gather(person, [])
        result_names = {r.source_name for r in bundle.results}
        assert result_names == enabled


class TestIsConfiguredFiltering:
    """Verify which gatherers are enabled/disabled based on config keys."""

    def test_unauthenticated_gatherers_always_enabled(self):
        cfg: dict = {}
        enabled = [
            g for g in information_gathering._ALL_GATHERERS if g.is_configured(cfg)
        ]
        enabled_names = {g.name for g in enabled}
        assert {"news", "wikipedia", "events"}.issubset(enabled_names)

    def test_twitter_disabled_without_token(self):
        cfg: dict = {}
        disabled = [
            g for g in information_gathering._ALL_GATHERERS if not g.is_configured(cfg)
        ]
        assert any(g.name == "twitter" for g in disabled)

    def test_reddit_disabled_without_credentials(self):
        cfg: dict = {}
        disabled = [
            g for g in information_gathering._ALL_GATHERERS if not g.is_configured(cfg)
        ]
        assert any(g.name == "reddit" for g in disabled)

    def test_instagram_disabled_without_token(self):
        cfg: dict = {}
        disabled = [
            g for g in information_gathering._ALL_GATHERERS if not g.is_configured(cfg)
        ]
        assert any(g.name == "instagram" for g in disabled)

    def test_twitter_enabled_with_token(self):
        cfg = {"twitter_bearer_token": "tok"}
        enabled = [
            g for g in information_gathering._ALL_GATHERERS if g.is_configured(cfg)
        ]
        assert any(g.name == "twitter" for g in enabled)

    def test_reddit_enabled_with_both_credentials(self):
        cfg = {"reddit_client_id": "id", "reddit_client_secret": "sec"}
        enabled = [
            g for g in information_gathering._ALL_GATHERERS if g.is_configured(cfg)
        ]
        assert any(g.name == "reddit" for g in enabled)
