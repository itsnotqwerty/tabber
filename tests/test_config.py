"""Tests for config.py — load, save, set_key, masked."""
from __future__ import annotations

import json

import pytest

import config as cfg_module


class TestLoad:
    def test_returns_defaults_when_file_absent(self):
        cfg = cfg_module.load()
        assert cfg["max_iterations"] == 3
        assert cfg["llm_provider"] == "openai"

    def test_merges_file_values_with_defaults(self):
        cfg_module.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        cfg_module.CONFIG_FILE.write_text(
            json.dumps({"openai_api_key": "sk-test", "max_iterations": 5})
        )
        cfg = cfg_module.load()
        assert cfg["openai_api_key"] == "sk-test"
        assert cfg["max_iterations"] == 5
        assert cfg["llm_provider"] == "openai"  # default preserved

    def test_file_values_override_defaults(self):
        cfg_module.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        cfg_module.CONFIG_FILE.write_text(json.dumps({"llm_provider": "anthropic"}))
        cfg = cfg_module.load()
        assert cfg["llm_provider"] == "anthropic"


class TestSetKey:
    def test_persists_string_value(self):
        cfg_module.set_key("openai_api_key", "sk-abc123")
        assert cfg_module.load()["openai_api_key"] == "sk-abc123"

    def test_converts_numeric_string_to_int(self):
        cfg_module.set_key("max_iterations", "7")
        cfg = cfg_module.load()
        assert cfg["max_iterations"] == 7
        assert isinstance(cfg["max_iterations"], int)

    def test_non_numeric_string_not_converted(self):
        cfg_module.set_key("llm_provider", "anthropic")
        assert cfg_module.load()["llm_provider"] == "anthropic"

    def test_overwriting_key_replaces_previous_value(self):
        cfg_module.set_key("max_iterations", "2")
        cfg_module.set_key("max_iterations", "5")
        assert cfg_module.load()["max_iterations"] == 5


class TestMasked:
    def test_partially_masks_long_secret(self):
        cfg = {"openai_api_key": "sk-abcdefghijklmno"}
        result = cfg_module.masked(cfg)
        val = result["openai_api_key"]
        assert "..." in val
        assert val.startswith("sk-a")
        assert val.endswith("lmno")

    def test_fully_masks_short_secret(self):
        cfg = {"openai_api_key": "short"}
        assert cfg_module.masked(cfg)["openai_api_key"] == "****"

    def test_leaves_empty_secret_unchanged(self):
        cfg = {"openai_api_key": ""}
        assert cfg_module.masked(cfg)["openai_api_key"] == ""

    def test_preserves_non_secret_keys(self):
        cfg = {"llm_provider": "openai", "max_iterations": 3}
        result = cfg_module.masked(cfg)
        assert result["llm_provider"] == "openai"
        assert result["max_iterations"] == 3

    def test_reddit_client_id_is_not_masked(self):
        # reddit_client_id is public; only reddit_client_secret is sensitive
        cfg = {"reddit_client_id": "pub_id_value"}
        result = cfg_module.masked(cfg)
        assert result["reddit_client_id"] == "pub_id_value"

    def test_all_secret_keys_are_masked(self):
        secret_keys = [
            "openai_api_key",
            "anthropic_api_key",
            "twitter_bearer_token",
            "instagram_access_token",
            "reddit_client_secret",
        ]
        cfg = {k: "supersecretvalue123" for k in secret_keys}
        result = cfg_module.masked(cfg)
        for k in secret_keys:
            assert result[k] != "supersecretvalue123", f"{k} was not masked"
