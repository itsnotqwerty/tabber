from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class PersonProfile(BaseModel):
    name: str
    aliases: list[str] = []
    known_roles: list[str] = []
    disambiguation_notes: str = ""


class HintsList(BaseModel):
    hints: list[str]


class GathererResult(BaseModel):
    source_name: str
    items: list[dict] = []
    raw_text: str = ""


class SignalEvaluation(BaseModel):
    confidence: float  # 0.0 – 1.0
    reason: str


class OSINTBundle(BaseModel):
    person: PersonProfile
    results: list[GathererResult] = []
    iteration: int = 0
    signal_evaluation: Optional[SignalEvaluation] = None


class LocationResult(BaseModel):
    location: str
    confidence: float
    reasoning: str
    sources: list[str] = []


class LookupResponse(BaseModel):
    query_name: str
    canon_name: str
    result: LocationResult
    cached: bool
    timestamp: str
