"""SQLite persistence layer for Tabber lookup results."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tabber.models import LocationResult, OSINTBundle

_DDL = """
CREATE TABLE IF NOT EXISTS lookups (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    query_name   TEXT    NOT NULL,
    canon_name   TEXT    NOT NULL,
    location     TEXT    NOT NULL,
    confidence   REAL    NOT NULL,
    reasoning    TEXT    NOT NULL,
    sources      TEXT    NOT NULL,
    profile_json TEXT    NOT NULL,
    bundle_json  TEXT,
    created_at   TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_lookups_canon_name ON lookups(canon_name);
CREATE INDEX IF NOT EXISTS ix_lookups_created_at ON lookups(created_at);
"""


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Open (or create) the SQLite database at *db_path* and return a connection."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    conn.commit()
    return conn


def save_result(
    conn: sqlite3.Connection,
    query_name: str,
    bundle: OSINTBundle,
    result: LocationResult,
) -> int:
    """Persist a lookup result. Returns the new row id."""
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """
        INSERT INTO lookups
            (query_name, canon_name, location, confidence, reasoning,
             sources, profile_json, bundle_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            query_name,
            bundle.person.name,
            result.location,
            result.confidence,
            result.reasoning,
            json.dumps(result.sources),
            bundle.person.model_dump_json(),
            bundle.model_dump_json(),
            now,
        ),
    )
    conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


def get_latest(conn: sqlite3.Connection, name: str) -> Optional[sqlite3.Row]:
    """Return the most recent row whose query_name or canon_name matches *name* (case-insensitive)."""
    return conn.execute(
        """
        SELECT * FROM lookups
        WHERE lower(canon_name) = lower(?)
           OR lower(query_name) = lower(?)
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (name, name),
    ).fetchone()


def list_all(conn: sqlite3.Connection, limit: int = 100) -> list[sqlite3.Row]:
    """Return up to *limit* rows ordered newest-first."""
    return conn.execute(
        "SELECT * FROM lookups ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()


def delete_by_name(conn: sqlite3.Connection, name: str) -> int:
    """Delete all rows for *name* (query or canonical). Returns number of deleted rows."""
    cursor = conn.execute(
        """
        DELETE FROM lookups
        WHERE lower(canon_name) = lower(?)
           OR lower(query_name) = lower(?)
        """,
        (name, name),
    )
    conn.commit()
    return cursor.rowcount
