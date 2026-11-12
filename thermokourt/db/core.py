"""Connection and schema helpers for the behaviour database."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (and create if needed) the database with foreign keys enforced."""
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    return con


def init_db(path: str | Path) -> sqlite3.Connection:
    """Create every table and view from ``schema.sql`` (idempotent)."""
    con = connect(path)
    con.executescript(SCHEMA_PATH.read_text())
    con.commit()
    return con


def schema_version(con: sqlite3.Connection) -> int:
    row = con.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()
    return int(row[0]) if row else 0


def now_iso() -> str:
    """Local time, ISO-8601 with offset (whitepaper convention)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def register_file(
    con: sqlite3.Connection, path: str | Path, kind: str, hash_it: bool = True
) -> int:
    """Insert (or find) a row in ``source_files`` and return its id."""
    p = Path(path)
    row = con.execute("SELECT file_id FROM source_files WHERE path = ?", (str(p),)).fetchone()
    if row:
        return int(row[0])
    size = p.stat().st_size if p.exists() else None
    digest = sha256_of(p) if (hash_it and p.exists() and p.is_file()) else None
    cur = con.execute(
        "INSERT INTO source_files (path, kind, sha256, size_bytes, ingested_at) VALUES (?,?,?,?,?)",
        (str(p), kind, digest, size, now_iso()),
    )
    return int(cur.lastrowid)
