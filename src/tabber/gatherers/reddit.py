from __future__ import annotations

from tabber import config as cfg_module
from tabber.gatherers.base import BaseGatherer
from tabber.models import GathererResult, PersonProfile


class RedditGatherer(BaseGatherer):
    name = "reddit"

    def is_configured(self, cfg: dict) -> bool:
        return bool(cfg.get("reddit_client_id") and cfg.get("reddit_client_secret"))

    def gather(self, profile: PersonProfile, hints: list[str]) -> GathererResult:
        import praw

        cfg = cfg_module.load()
        reddit = praw.Reddit(
            client_id=cfg["reddit_client_id"],
            client_secret=cfg["reddit_client_secret"],
            user_agent="tabber/1.0",
        )

        query = f"{profile.name} location"
        items: list[dict] = []
        raw_parts: list[str] = []
        try:
            for submission in reddit.subreddit("all").search(
                query, sort="new", limit=15
            ):
                snippet = (
                    submission.title
                    + ". "
                    + (submission.selftext[:200] if submission.selftext else "")
                )
                items.append(
                    {
                        "title": submission.title,
                        "url": submission.url,
                        "created_utc": submission.created_utc,
                        "snippet": snippet,
                    }
                )
                raw_parts.append(snippet)
        except Exception:
            pass

        return GathererResult(
            source_name=self.name,
            items=items,
            raw_text="\n".join(raw_parts),
        )
