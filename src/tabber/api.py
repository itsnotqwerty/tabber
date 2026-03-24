"""FastAPI REST server for Tabber.

Start with:
    tabber server
or:
    uvicorn tabber.api:app --host 127.0.0.1 --port 8000

To enable the web dashboard:
    tabber server --webui
or:
    TABBER_WEBUI=1 uvicorn tabber.api:create_app --factory --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.routing import APIRouter
from pydantic import BaseModel

from tabber import caching
from tabber import sqlite as db_module
from tabber.models import LookupResponse
from tabber.modules import identification, location_analysis


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
# API router  (all REST endpoints live under /api)
# ---------------------------------------------------------------------------

api_router = APIRouter(prefix="/api")


@api_router.get("/health")
def health():
    return {"status": "ok"}


@api_router.post("/lookup", response_model=LookupResponse)
def lookup(req: LookupRequest) -> LookupResponse:
    """Run a location lookup, using cache unless *no_cache* is set."""
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")
    try:
        name = identification._sanitize_name(name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    prior_bundle = None
    if not req.no_cache:
        prior_bundle = caching.get_cached_bundle(name)

    # Fresh lookup, optionally seeded with cached OSINT data
    try:
        bundle = identification.run(name, prior_bundle=prior_bundle)
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


@api_router.get("/results", response_model=list[HistoryRow])
def list_results(limit: int = Query(default=50, ge=1, le=500)) -> list[HistoryRow]:
    """List stored lookup results, newest first."""
    conn = caching._get_conn()
    rows = db_module.list_all(conn, limit=limit)
    return [_row_to_history(r) for r in rows]


@api_router.get("/results/{name}", response_model=HistoryRow)
def get_result(name: str) -> HistoryRow:
    """Return the most recent stored result for *name*."""
    conn = caching._get_conn()
    row = db_module.get_latest(conn, name)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No result found for '{name}'")
    return _row_to_history(row)


@api_router.delete("/results/{name}")
def delete_result(name: str) -> dict:
    """Invalidate (delete) all cached results for *name*."""
    deleted = caching.invalidate(name)
    return {"deleted": deleted, "name": name}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(webui: bool | None = None) -> FastAPI:
    """Build and return the FastAPI application.

    *webui* defaults to the ``TABBER_WEBUI`` environment variable when not
    supplied explicitly (``"1"`` enables the dashboard).
    """
    if webui is None:
        webui = os.environ.get("TABBER_WEBUI") == "1"

    _app = FastAPI(title="Tabber API", version="1.0.0")
    _app.include_router(api_router)

    if webui:
        from fastapi.staticfiles import StaticFiles
        from fastapi.templating import Jinja2Templates

        _static_dir = Path(__file__).parent / "static"
        _app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

        _templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

        @_app.get("/", include_in_schema=False)
        def dashboard(request: Request):
            return _templates.TemplateResponse(request, "dashboard.html")

    return _app


# Module-level app for direct uvicorn invocation and backwards compatibility.
app = create_app()
