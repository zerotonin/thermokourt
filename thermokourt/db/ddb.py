"""Copy fly-line metadata from a DrosoDB snapshot into the behaviour DB.

DrosoDB (``zerotonin/drosodb``) is the lab's vial-tracking system. Its
hourly backup ``ddb.latest.sqlite3`` is synced to the recording PC. At trial
set-up the operator picks each fly's source vial by its 4-character print
code; this module copies that vial, its ancestors (flip chain and cross
parents), and their genotypes and donors into the ``ddb_*`` tables, stamped
with the snapshot's SHA-256. After that the behaviour DB stands alone.

Only the columns the analysis needs are copied (no owners, org units or
audit events). The live DrosoDB is never written to.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from thermokourt.db.core import now_iso, sha256_of

DDB_TABLES = ("genotype", "vial", "vial_parentage", "donor")


@dataclass
class VialInfo:
    vial_id: int
    print_code: str
    genotype_id: int
    genotype_name: str
    generation: int | None
    is_active: bool


def open_snapshot(path: str | Path) -> sqlite3.Connection:
    """Open a DrosoDB snapshot read-only and check it looks like one."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"DrosoDB snapshot not found: {p}")
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = [t for t in DDB_TABLES if t not in have]
    if missing:
        raise ValueError(f"{p} is not a DrosoDB file: missing tables {missing}")
    return con


def find_vials(ddb: sqlite3.Connection, query: str, active_only: bool = True) -> list[VialInfo]:
    """Vials whose print code starts with ``query`` (case-insensitive), or
    whose genotype name contains it. Used by the trial set-up picker."""
    q = query.strip()
    sql = """
        SELECT v.id, v.print_code, v.genotype_id, g.name, v.generation, v.is_active
        FROM vial v JOIN genotype g ON g.id = v.genotype_id
        WHERE (UPPER(v.print_code) LIKE UPPER(?) OR g.name LIKE ?)
    """
    params: list = [q + "%", "%" + q + "%"]
    if active_only:
        sql += " AND v.is_active = 1"
    sql += " ORDER BY v.print_code LIMIT 50"
    return [VialInfo(r[0], r[1], r[2], r[3], r[4], bool(r[5])) for r in ddb.execute(sql, params)]


def vial_by_print_code(ddb: sqlite3.Connection, print_code: str) -> VialInfo:
    rows = find_vials(ddb, print_code, active_only=False)
    exact = [v for v in rows if v.print_code.upper() == print_code.strip().upper()]
    if not exact:
        raise LookupError(f"no vial with print code {print_code!r} in the DrosoDB snapshot")
    if len(exact) > 1:
        # Print codes of decommissioned vials may be reused; prefer the active one.
        active = [v for v in exact if v.is_active]
        if len(active) == 1:
            return active[0]
        raise LookupError(f"print code {print_code!r} is ambiguous ({len(exact)} vials)")
    return exact[0]


def ancestor_ids(ddb: sqlite3.Connection, vial_id: int, max_depth: int = 50) -> set[int]:
    """The vial plus every ancestor by flip (``flipped_from_id``) and by
    cross (``vial_parentage``)."""
    seen: set[int] = set()
    frontier = [vial_id]
    depth = 0
    while frontier and depth < max_depth:
        depth += 1
        nxt = []
        for vid in frontier:
            if vid in seen:
                continue
            seen.add(vid)
            row = ddb.execute("SELECT flipped_from_id FROM vial WHERE id = ?", (vid,)).fetchone()
            if row and row[0] is not None:
                nxt.append(int(row[0]))
            for r in ddb.execute("SELECT parent_id FROM vial_parentage WHERE child_id = ?", (vid,)):
                nxt.append(int(r[0]))
        frontier = nxt
    return seen


def register_snapshot(con: sqlite3.Connection, path: str | Path) -> int:
    """Insert (or find) the ``ddb_snapshots`` row for this file; return its id."""
    p = Path(path)
    digest = sha256_of(p)
    row = con.execute(
        "SELECT snapshot_id FROM ddb_snapshots WHERE sha256 = ?", (digest,)
    ).fetchone()
    if row:
        return int(row[0])
    taken = datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    cur = con.execute(
        "INSERT INTO ddb_snapshots (sha256, source_path, taken_at, ingested_at) VALUES (?,?,?,?)",
        (digest, str(p), taken, now_iso()),
    )
    return int(cur.lastrowid)


def copy_vials(
    con: sqlite3.Connection, ddb: sqlite3.Connection, snapshot_id: int, vial_ids: set[int]
) -> dict[str, int]:
    """Copy the given vials (already expanded to ancestors), their genotypes,
    donors and parentage edges. Existing rows are replaced (newer snapshot
    wins). Returns row counts per table."""
    if not vial_ids:
        return {}
    ids = sorted(vial_ids)
    ph = ",".join("?" * len(ids))

    vials = ddb.execute(
        f"""SELECT id, print_code, genotype_id, flipped_from_id, generation, is_active,
                   created_at, decommissioned_at, notes
            FROM vial WHERE id IN ({ph})""",
        ids,
    ).fetchall()
    geno_ids = sorted({int(v["genotype_id"]) for v in vials})
    gph = ",".join("?" * len(geno_ids))
    genos = ddb.execute(
        f"""SELECT id, name, chromosome_x, chromosome_2, chromosome_3, chromosome_4,
                   chromosome_y, phenotype, is_wildtype, donor_strain_id, donor_id
            FROM genotype WHERE id IN ({gph})""",
        geno_ids,
    ).fetchall()
    donor_ids = sorted({int(g["donor_id"]) for g in genos if g["donor_id"] is not None})
    donors = []
    if donor_ids:
        dph = ",".join("?" * len(donor_ids))
        donors = ddb.execute(
            f"SELECT id, name FROM donor WHERE id IN ({dph})", donor_ids
        ).fetchall()
    edges = ddb.execute(
        f"SELECT child_id, parent_id, role FROM vial_parentage WHERE child_id IN ({ph})", ids
    ).fetchall()

    con.executemany(
        "INSERT OR REPLACE INTO ddb_donor (id, name, snapshot_id) VALUES (?,?,?)",
        [(d["id"], d["name"], snapshot_id) for d in donors],
    )
    con.executemany(
        """INSERT OR REPLACE INTO ddb_genotype
           (id, name, chromosome_x, chromosome_2, chromosome_3, chromosome_4, chromosome_y,
            phenotype, is_wildtype, donor_strain_id, donor_id, snapshot_id)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                g["id"],
                g["name"],
                g["chromosome_x"],
                g["chromosome_2"],
                g["chromosome_3"],
                g["chromosome_4"],
                g["chromosome_y"],
                g["phenotype"],
                int(bool(g["is_wildtype"])),
                g["donor_strain_id"],
                g["donor_id"],
                snapshot_id,
            )
            for g in genos
        ],
    )
    con.executemany(
        """INSERT OR REPLACE INTO ddb_vial
           (id, print_code, genotype_id, flipped_from_id, generation, is_active,
            created_at, decommissioned_at, notes, snapshot_id)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                v["id"],
                v["print_code"],
                v["genotype_id"],
                v["flipped_from_id"],
                v["generation"],
                int(bool(v["is_active"])) if v["is_active"] is not None else None,
                _iso(v["created_at"]),
                _iso(v["decommissioned_at"]),
                v["notes"],
                snapshot_id,
            )
            for v in vials
        ],
    )
    con.executemany(
        "INSERT OR REPLACE INTO ddb_vial_parentage"
        " (child_id, parent_id, role, snapshot_id) VALUES (?,?,?,?)",
        [(e["child_id"], e["parent_id"], e["role"], snapshot_id) for e in edges],
    )
    return {
        "ddb_vial": len(vials),
        "ddb_genotype": len(genos),
        "ddb_donor": len(donors),
        "ddb_vial_parentage": len(edges),
    }


def import_vials_by_print_code(
    con: sqlite3.Connection, snapshot_path: str | Path, print_codes: list[str]
) -> dict[str, VialInfo]:
    """One call for the trial set-up: resolve print codes in the snapshot,
    copy the vials + lineage, return ``{print_code: VialInfo}``."""
    ddb = open_snapshot(snapshot_path)
    try:
        snap_id = register_snapshot(con, snapshot_path)
        resolved = {pc: vial_by_print_code(ddb, pc) for pc in print_codes}
        ids: set[int] = set()
        for v in resolved.values():
            ids |= ancestor_ids(ddb, v.vial_id)
        copy_vials(con, ddb, snap_id, ids)
        con.commit()
        return resolved
    finally:
        ddb.close()


def _iso(value):
    """DDB stores naive UTC datetimes; keep text as-is but make it ISO."""
    if value is None:
        return None
    s = str(value)
    return s.replace(" ", "T", 1) if " " in s and "T" not in s else s
