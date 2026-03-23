"""Tests for llm.py — OpenAI and Anthropic provider routing, response_format."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import llm
from models import PersonProfile


# ─── mock builders ────────────────────────────────────────────────────────────


def _text_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = content
    return resp


def _parsed_response(obj) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.parsed = obj
    return resp


def _cfg_openai(monkeypatch) -> None:
    monkeypatch.setattr(
        "tabber.config.load",
        lambda: {"llm_provider": "openai", "openai_api_key": "sk-test"},
    )


def _cfg_anthropic(monkeypatch) -> None:
    monkeypatch.setattr(
        "tabber.config.load",
        lambda: {"llm_provider": "anthropic", "anthropic_api_key": "sk-ant-test"},
    )


# ─── OpenAI — plain text ──────────────────────────────────────────────────────


class TestOpenAIPlainText:
    def test_returns_string(self, monkeypatch):
        _cfg_openai(monkeypatch)
        with patch("llm.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = (
                _text_response("hello world")
            )
            result = llm.complete("test prompt")
        assert result == "hello world"

    def test_uses_create_not_parse(self, monkeypatch):
        _cfg_openai(monkeypatch)
        with patch("llm.openai.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.chat.completions.create.return_value = _text_response("ok")
            llm.complete("prompt")
        mock_client.chat.completions.create.assert_called_once()
        mock_client.beta.chat.completions.parse.assert_not_called()

    def test_includes_system_message(self, monkeypatch):
        _cfg_openai(monkeypatch)
        with patch("llm.openai.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.chat.completions.create.return_value = _text_response("ok")
            llm.complete("user prompt", system="be concise")
            msgs = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert msgs[0] == {"role": "system", "content": "be concise"}
        assert msgs[1] == {"role": "user", "content": "user prompt"}

    def test_omits_system_message_when_empty(self, monkeypatch):
        _cfg_openai(monkeypatch)
        with patch("llm.openai.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.chat.completions.create.return_value = _text_response("ok")
            llm.complete("prompt")
            msgs = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_missing_api_key_raises_runtime_error(self, monkeypatch):
        monkeypatch.setattr("tabber.config.load", lambda: {"llm_provider": "openai"})
        with pytest.raises(RuntimeError, match="openai_api_key"):
            llm.complete("test")


# ─── OpenAI — structured output ───────────────────────────────────────────────


class TestOpenAIStructured:
    def test_returns_pydantic_instance(self, monkeypatch):
        _cfg_openai(monkeypatch)
        profile = PersonProfile(name="Jane Doe")
        with patch("llm.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.beta.chat.completions.parse.return_value = (
                _parsed_response(profile)
            )
            result = llm.complete("test", response_format=PersonProfile)
        assert isinstance(result, PersonProfile)
        assert result.name == "Jane Doe"

    def test_calls_beta_parse_not_create(self, monkeypatch):
        _cfg_openai(monkeypatch)
        profile = PersonProfile(name="Jane")
        with patch("llm.openai.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.beta.chat.completions.parse.return_value = _parsed_response(
                profile
            )
            llm.complete("test", response_format=PersonProfile)
        mock_client.beta.chat.completions.parse.assert_called_once()
        mock_client.chat.completions.create.assert_not_called()

    def test_passes_response_format_to_parse(self, monkeypatch):
        _cfg_openai(monkeypatch)
        profile = PersonProfile(name="Jane")
        with patch("llm.openai.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.beta.chat.completions.parse.return_value = _parsed_response(
                profile
            )
            llm.complete("test", response_format=PersonProfile)
            kwargs = mock_client.beta.chat.completions.parse.call_args.kwargs
        assert kwargs["response_format"] is PersonProfile


# ─── Anthropic — plain text ───────────────────────────────────────────────────


class TestAnthropicPlainText:
    def test_returns_string(self, monkeypatch):
        _cfg_anthropic(monkeypatch)
        with patch("llm.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = (
                _text_response("hello")
            )
            result = llm.complete("test")
        assert result == "hello"

    def test_missing_api_key_raises_runtime_error(self, monkeypatch):
        monkeypatch.setattr("tabber.config.load", lambda: {"llm_provider": "anthropic"})
        with pytest.raises(RuntimeError, match="anthropic_api_key"):
            llm.complete("test")

    def test_uses_anthropic_base_url(self, monkeypatch):
        _cfg_anthropic(monkeypatch)
        with patch("llm.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = (
                _text_response("ok")
            )
            llm.complete("test")
            _, init_kwargs = MockOpenAI.call_args
        assert "api.anthropic.com" in init_kwargs["base_url"]


# ─── Anthropic — structured output ────────────────────────────────────────────


class TestAnthropicStructured:
    def test_returns_pydantic_instance(self, monkeypatch):
        _cfg_anthropic(monkeypatch)
        profile = PersonProfile(name="Jane Doe")
        with patch("llm.openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.chat.completions.create.return_value = (
                _text_response(profile.model_dump_json())
            )
            result = llm.complete("test", response_format=PersonProfile)
        assert isinstance(result, PersonProfile)
        assert result.name == "Jane Doe"

    def test_passes_json_schema_response_format(self, monkeypatch):
        _cfg_anthropic(monkeypatch)
        profile = PersonProfile(name="Jane")
        with patch("llm.openai.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.chat.completions.create.return_value = _text_response(
                profile.model_dump_json()
            )
            llm.complete("test", response_format=PersonProfile)
            kwargs = mock_client.chat.completions.create.call_args.kwargs
        rf = kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "PersonProfile"
        assert "schema" in rf["json_schema"]

    def test_sets_strict_true_in_schema(self, monkeypatch):
        _cfg_anthropic(monkeypatch)
        profile = PersonProfile(name="Jane")
        with patch("llm.openai.OpenAI") as MockOpenAI:
            mock_client = MockOpenAI.return_value
            mock_client.chat.completions.create.return_value = _text_response(
                profile.model_dump_json()
            )
            llm.complete("test", response_format=PersonProfile)
            kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"]["json_schema"]["strict"] is True
