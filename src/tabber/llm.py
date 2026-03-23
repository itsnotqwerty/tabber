"""Thin LLM wrapper — supports OpenAI and Anthropic providers."""

from __future__ import annotations

from typing import TypeVar, overload

import openai
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@overload
def complete(prompt: str, system: str = ..., *, response_format: None = ...) -> str: ...


@overload
def complete(prompt: str, system: str = ..., *, response_format: type[T]) -> T: ...


def complete(prompt: str, system: str = "", *, response_format=None):
    """Send a prompt to the configured LLM provider and return the response text."""
    from tabber import config as cfg_module

    cfg = cfg_module.load()
    provider = cfg.get("llm_provider", "openai")
    if provider == "anthropic":
        return _anthropic(prompt, system, cfg, response_format)
    return _openai(prompt, system, cfg, response_format)


def _openai(prompt: str, system: str, cfg: dict, response_format=None):
    api_key = cfg.get("openai_api_key")
    if not api_key:
        raise RuntimeError(
            "openai_api_key is not set. "
            "Run: python tabber.py config set openai_api_key YOUR_KEY"
        )
    client = openai.OpenAI(api_key=api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    if response_format is not None:
        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            temperature=0.2,
            response_format=response_format,
        )
        return response.choices[0].message.parsed
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content  # type: ignore


def _anthropic(prompt: str, system: str, cfg: dict, response_format=None):
    api_key = cfg.get("anthropic_api_key")
    if not api_key:
        raise RuntimeError(
            "anthropic_api_key is not set. "
            "Run: python tabber.py config set anthropic_api_key YOUR_KEY"
        )
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.anthropic.com/v1",
        default_headers={"anthropic-version": "2023-06-01"},
    )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    if response_format is not None:
        schema = response_format.model_json_schema()
        response = client.chat.completions.create(
            model="claude-opus-4-6",
            messages=messages,
            temperature=0.2,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": schema,
                    "strict": True,
                },
            },
        )
        return response_format.model_validate_json(response.choices[0].message.content)
    response = client.chat.completions.create(
        model="claude-opus-4-6",
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content  # type: ignore
