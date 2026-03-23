"""Tests for individual gatherer classes — is_configured and static properties."""
from __future__ import annotations

import pytest

from gatherers.base import BaseGatherer
from gatherers.events import EventsGatherer
from gatherers.instagram import InstagramGatherer
from gatherers.news import NewsGatherer
from gatherers.reddit import RedditGatherer
from gatherers.twitter import TwitterGatherer
from gatherers.wikipedia import WikipediaGatherer


ALL_GATHERER_CLASSES = [
    NewsGatherer,
    WikipediaGatherer,
    EventsGatherer,
    TwitterGatherer,
    InstagramGatherer,
    RedditGatherer,
]


class TestBaseGathererInterface:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseGatherer()  # type: ignore

    def test_default_is_configured_returns_true(self):
        class ConcreteGatherer(BaseGatherer):
            name = "test"

            def gather(self, profile, hints):
                ...

        assert ConcreteGatherer().is_configured({}) is True


class TestGathererNames:
    def test_each_class_has_nonempty_name(self):
        for cls in ALL_GATHERER_CLASSES:
            assert cls.name, f"{cls.__name__}.name must not be empty"

    def test_all_names_are_unique(self):
        names = [cls.name for cls in ALL_GATHERER_CLASSES]
        assert len(names) == len(set(names)), "Gatherer names must be unique"


class TestIsConfigured:
    # — unauthenticated gatherers —

    def test_news_always_configured(self):
        assert NewsGatherer().is_configured({}) is True

    def test_wikipedia_always_configured(self):
        assert WikipediaGatherer().is_configured({}) is True

    def test_events_always_configured(self):
        assert EventsGatherer().is_configured({}) is True

    # — Twitter —

    def test_twitter_not_configured_without_token(self):
        assert TwitterGatherer().is_configured({}) is False

    def test_twitter_configured_with_token(self):
        assert TwitterGatherer().is_configured({"twitter_bearer_token": "tok"}) is True

    def test_twitter_not_configured_with_empty_token(self):
        assert TwitterGatherer().is_configured({"twitter_bearer_token": ""}) is False

    # — Instagram —

    def test_instagram_not_configured_without_token(self):
        assert InstagramGatherer().is_configured({}) is False

    def test_instagram_configured_with_token(self):
        assert (
            InstagramGatherer().is_configured({"instagram_access_token": "tok"}) is True
        )

    def test_instagram_not_configured_with_empty_token(self):
        assert (
            InstagramGatherer().is_configured({"instagram_access_token": ""}) is False
        )

    # — Reddit —

    def test_reddit_not_configured_without_credentials(self):
        assert RedditGatherer().is_configured({}) is False

    def test_reddit_not_configured_with_only_id(self):
        assert RedditGatherer().is_configured({"reddit_client_id": "id"}) is False

    def test_reddit_not_configured_with_only_secret(self):
        assert RedditGatherer().is_configured({"reddit_client_secret": "sec"}) is False

    def test_reddit_configured_with_both_credentials(self):
        assert (
            RedditGatherer().is_configured(
                {"reddit_client_id": "id", "reddit_client_secret": "sec"}
            )
            is True
        )

    def test_reddit_not_configured_with_empty_credentials(self):
        assert (
            RedditGatherer().is_configured(
                {"reddit_client_id": "", "reddit_client_secret": ""}
            )
            is False
        )
