from __future__ import annotations

import config as cfg_module
from gatherers.base import BaseGatherer
from models import GathererResult, PersonProfile


class TwitterGatherer(BaseGatherer):
    name = "twitter"

    def is_configured(self, cfg: dict) -> bool:
        return bool(cfg.get("twitter_bearer_token"))

    def gather(self, profile: PersonProfile, hints: list[str]) -> GathererResult:
        import tweepy

        cfg = cfg_module.load()
        client = tweepy.Client(bearer_token=cfg["twitter_bearer_token"])

        # Build a concise query – quoted name, no retweets, English only
        query = f'"{profile.name}" -is:retweet lang:en'
        items: list[dict] = []
        try:
            response: tweepy.Response = client.search_recent_tweets(
                query=query,
                max_results=20,
                tweet_fields=["created_at", "text", "geo"],
            ) # type: ignore
            if response and response.data:
                for tweet in response.data:
                    items.append(
                        {
                            "text": tweet.text,
                            "created_at": str(tweet.created_at),
                            "geo": str(tweet.geo) if tweet.geo else "",
                        }
                    )
        except Exception:
            pass

        raw_text = "\n".join(i["text"] for i in items)
        return GathererResult(source_name=self.name, items=items, raw_text=raw_text)
