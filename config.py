from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".tabber"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Keys whose values are masked in `config show`
_SECRET_KEYS = {
    "openai_api_key",
    "anthropic_api_key",
    "twitter_bearer_token",
    "instagram_access_token",
    "reddit_client_secret",
}

_DEFAULTS: dict[str, Any] = {
    "max_iterations": 3,
    "llm_provider": "openai",
}


def load() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        return dict(_DEFAULTS)
    with CONFIG_FILE.open() as f:
        data = json.load(f)
    return {**_DEFAULTS, **data}


def save(cfg: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_FILE.open("w") as f:
        json.dump(cfg, f, indent=2)


def set_key(key: str, value: Any) -> None:
    cfg = load()
    # Preserve int type for numeric keys
    try:
        value = int(value)
    except (ValueError, TypeError):
        pass
    cfg[key] = value
    save(cfg)


def masked(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of cfg with secret values partially obscured."""
    out = {}
    for k, v in cfg.items():
        if k in _SECRET_KEYS and v:
            s = str(v)
            out[k] = s[:4] + "..." + s[-4:] if len(s) > 8 else "****"
        else:
            out[k] = v
    return out
