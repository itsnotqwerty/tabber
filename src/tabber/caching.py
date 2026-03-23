"""Caching façade — TTL-aware result recall backed by SQLite."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import config as cfg_module
import sqlite as db_module
from models import LocationResult, OSINTBundle

_conn = None  # module-level cached connection


def _get_conn():
    global _conn
    if _conn is None:
        cfg = cfg_module.load()
        db_path = cfg.get("db_path", str(Path.home() / ".tabber" / "results.db"))
        _conn = db_module.init_db(db_path)
    return _conn


def _reset_conn() -> None:
    """Force the next call to _get_conn() to open a fresh connection.

    Useful in tests that redirect the db_path between calls.
    """
    global _conn
    _conn = None


def get_cached(name: str) -> Optional[LocationResult]:
    """Return a fresh (non-expired) LocationResult for *name*, or None.

    Expiry is controlled by the ``cache_ttl_hours`` config key (default 24 h).
    """
    cfg = cfg_module.load()
    ttl_hours: int = int(cfg.get("cache_ttl_hours", 24))

    conn = _get_conn()
    row = db_module.get_latest(conn, name)
    if row is None:
        return None

    created_at = datetime.fromisoformat(row["created_at"])
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - created_at
    if age > timedelta(hours=ttl_hours):
        return None

    import json

    return LocationResult(
        location=row["location"],
        confidence=row["confidence"],
        reasoning=row["reasoning"],
        sources=json.loads(row["sources"]),
    )


def store(query_name: str, bundle: OSINTBundle, result: LocationResult) -> None:
    """Persist a lookup result to the database."""
    conn = _get_conn()
    db_module.save_result(conn, query_name, bundle, result)


def invalidate(name: str) -> int:
    """Delete all cached rows for *name*. Returns number of deleted rows."""
    conn = _get_conn()
    return db_module.delete_by_name(conn, name)
