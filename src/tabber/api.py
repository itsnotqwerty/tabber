"""FastAPI REST server for Tabber.

Start with:
    tabber server
or:
    uvicorn tabber.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from tabber import caching
from tabber import sqlite as db_module
from tabber.models import LocationResult, LookupResponse
from tabber.modules import identification, location_analysis

app = FastAPI(title="Tabber API", version="1.0.0")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class LookupRequest(BaseModel):
    name: str
    no_cache: bool = False


class HistoryRow(BaseModel):
    id: int
    query_name: str
    canon_name: str
    location: str
    confidence: float
    reasoning: str
    sources: list[str]
    cached: bool = True
    timestamp: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_history(row) -> HistoryRow:
    return HistoryRow(
        id=row["id"],
        query_name=row["query_name"],
        canon_name=row["canon_name"],
        location=row["location"],
        confidence=row["confidence"],
        reasoning=row["reasoning"],
        sources=json.loads(row["sources"]),
        timestamp=row["created_at"],
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/lookup", response_model=LookupResponse)
def lookup(req: LookupRequest) -> LookupResponse:
    """Run a location lookup, using cache unless *no_cache* is set."""
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")

    cached_result: Optional[LocationResult] = None
    if not req.no_cache:
        cached_result = caching.get_cached(name)

    if cached_result is not None:
        conn = caching._get_conn()
        row = db_module.get_latest(conn, name)
        canon_name = row["canon_name"] if row else name
        return LookupResponse(
            query_name=name,
            canon_name=canon_name,
            result=cached_result,
            cached=True,
            timestamp=(
                row["created_at"] if row else datetime.now(timezone.utc).isoformat()
            ),
        )

    # Fresh lookup
    try:
        bundle = identification.run(name)
        result = location_analysis.analyse(bundle)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    caching.store(name, bundle, result)

    return LookupResponse(
        query_name=name,
        canon_name=bundle.person.name,
        result=result,
        cached=False,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/results", response_model=list[HistoryRow])
def list_results(limit: int = Query(default=50, ge=1, le=500)) -> list[HistoryRow]:
    """List stored lookup results, newest first."""
    conn = caching._get_conn()
    rows = db_module.list_all(conn, limit=limit)
    return [_row_to_history(r) for r in rows]


@app.get("/results/{name}", response_model=HistoryRow)
def get_result(name: str) -> HistoryRow:
    """Return the most recent stored result for *name*."""
    conn = caching._get_conn()
    row = db_module.get_latest(conn, name)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No result found for '{name}'")
    return _row_to_history(row)


@app.delete("/results/{name}")
def delete_result(name: str) -> dict:
    """Invalidate (delete) all cached results for *name*."""
    deleted = caching.invalidate(name)
    return {"deleted": deleted, "name": name}
