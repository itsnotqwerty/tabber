"""Instagram gatherer.

NOTE: The Instagram Basic Display API only grants access to the *authenticated
user's own* media — it cannot retrieve another public figure's posts.  This
gatherer is therefore only useful if the target person's own access token has
been configured.  It will return empty results (with a note) when the token
belongs to a different account, and is skipped entirely when no token is set.
"""

from __future__ import annotations

import requests

from tabber import config as cfg_module
from tabber.gatherers.base import BaseGatherer
from tabber.models import GathererResult, PersonProfile

_TIMEOUT = 10


class InstagramGatherer(BaseGatherer):
    name = "instagram"

    def is_configured(self, cfg: dict) -> bool:
        return bool(cfg.get("instagram_access_token"))

    def gather(self, profile: PersonProfile, hints: list[str]) -> GathererResult:
        cfg = cfg_module.load()
        token = cfg.get("instagram_access_token", "")
        items: list[dict] = []
        raw_parts: list[str] = []

        try:
            resp = requests.get(
                "https://graph.instagram.com/me/media",
                params={
                    "fields": "caption,media_type,timestamp,permalink",
                    "access_token": token,
                },
                timeout=_TIMEOUT,
            )
            if resp.ok:
                for post in resp.json().get("data", [])[:10]:
                    caption = post.get("caption", "")
                    items.append(
                        {
                            "caption": caption,
                            "timestamp": post.get("timestamp", ""),
                            "url": post.get("permalink", ""),
                        }
                    )
                    if caption:
                        raw_parts.append(caption)
        except Exception:
            pass

        return GathererResult(
            source_name=self.name,
            items=items,
            raw_text="\n".join(raw_parts),
        )
