"""Thin LLM wrapper — dispatches to OpenAI or Anthropic based on config."""
from __future__ import annotations


def complete(prompt: str, system: str = "") -> str:
    """Send a prompt to the configured LLM and return the response text."""
    import config as cfg_module

    cfg = cfg_module.load()
    provider = cfg.get("llm_provider", "openai")

    if provider == "openai":
        return _openai(prompt, system, cfg)
    if provider == "anthropic":
        return _anthropic(prompt, system, cfg)
    raise ValueError(
        f"Unknown llm_provider '{provider}'. "
        "Run: python tabber.py config set llm_provider openai|anthropic"
    )


def _openai(prompt: str, system: str, cfg: dict) -> str:
    import openai

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
    return response.choices[0].message.content


def _anthropic(prompt: str, system: str, cfg: dict) -> str:
    import anthropic

    api_key = cfg.get("anthropic_api_key")
    if not api_key:
        raise RuntimeError(
            "anthropic_api_key is not set. "
            "Run: python tabber.py config set anthropic_api_key YOUR_KEY"
        )
    client = anthropic.Anthropic(api_key=api_key)
    kwargs: dict = {
        "model": "claude-opus-4-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return response.content[0].text
