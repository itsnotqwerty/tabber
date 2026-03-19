from __future__ import annotations

from pydantic import BaseModel


class PersonProfile(BaseModel):
    name: str
    aliases: list[str] = []
    known_roles: list[str] = []
    disambiguation_notes: str = ""


class GathererResult(BaseModel):
    source_name: str
    items: list[dict] = []
    raw_text: str = ""


class OSINTBundle(BaseModel):
    person: PersonProfile
    results: list[GathererResult] = []
    iteration: int = 0


class LocationResult(BaseModel):
    location: str
    confidence: float
    reasoning: str
    sources: list[str] = []
