from __future__ import annotations

from gatherers.base import BaseGatherer
from models import GathererResult, PersonProfile


class NewsGatherer(BaseGatherer):
    name = "news"

    def gather(self, profile: PersonProfile, hints: list[str]) -> GathererResult:
        from ddgs import DDGS

        query = f"{profile.name} location recent " + " ".join(hints[:3])
        items: list[dict] = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.news(query, max_results=10))
            items = [
                {
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "date": r.get("date", ""),
                    "url": r.get("url", ""),
                }
                for r in results
            ]
        except Exception:
            pass

        raw_text = "\n".join(
            f"{i['date']} {i['title']}: {i['body']}" for i in items
        )
        return GathererResult(source_name=self.name, items=items, raw_text=raw_text)
