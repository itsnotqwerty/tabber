"""Events gatherer — searches for upcoming/recent public appearances via
DuckDuckGo (no API key required).  Uses two targeted queries:
        1. General event / appearance search
        2. Tour dates / schedule search
Avoids Google scraping (brittle) and Eventbrite API (requires OAuth).
"""

from __future__ import annotations

from gatherers.base import BaseGatherer
from models import GathererResult, PersonProfile


class EventsGatherer(BaseGatherer):
    name = "events"

    def gather(self, profile: PersonProfile, hints: list[str]) -> GathererResult:
        from ddgs import DDGS

        items: list[dict] = []
        raw_parts: list[str] = []

        queries = [
            f"{profile.name} upcoming events appearance location 2025 2026",
            f"{profile.name} tour schedule concert venue city",
        ]

        for attempt in range(2):
            try:
                with DDGS() as ddgs:
                    for q in queries:
                        for r in ddgs.text(q, max_results=8):
                            entry = {
                                "title": r.get("title", ""),
                                "body": r.get("body", ""),
                                "url": r.get("href", ""),
                            }
                            items.append(entry)
                            raw_parts.append(f"{entry['title']}: {entry['body']}")
            except Exception:
                pass
            if items:
                break
            if attempt == 0:
                import time

                time.sleep(2)

        return GathererResult(
            source_name=self.name,
            items=items,
            raw_text="\n".join(raw_parts),
        )
