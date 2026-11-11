"""Stage 6 — the behaviour database.

A single SQLite file that is *generated* from source files (recordings,
temperature logs, the trials sheet, DrosoDB snapshots, tracker output and
scoring exports) and never edited by hand. See ``schema.sql`` for the
layout and :mod:`thermokourt.db.ddb` for the DrosoDB copy that makes the
database standalone.
"""

from thermokourt.db.core import connect, init_db, schema_version

__all__ = ["connect", "init_db", "schema_version"]
