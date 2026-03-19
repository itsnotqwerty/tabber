"""Thin LLM wrapper — calls OpenAI (gpt-4o)."""
from __future__ import annotations
import openai


def complete(prompt: str, system: str = "") -> str:
    """Send a prompt to OpenAI and return the response text."""
    import config as cfg_module

    cfg = cfg_module.load()
    return _openai(prompt, system, cfg)


def _openai(prompt: str, system: str, cfg: dict) -> str:
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
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content # type: ignore



