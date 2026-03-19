from __future__ import annotations

from abc import ABC, abstractmethod

from models import GathererResult, PersonProfile


class BaseGatherer(ABC):
    name: str = ""

    def is_configured(self, cfg: dict) -> bool:
        """Return True if this gatherer has all required config keys set."""
        return True

    @abstractmethod
    def gather(self, profile: PersonProfile, hints: list[str]) -> GathererResult: ...
