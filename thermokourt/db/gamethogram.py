"""Import GameThogram manual scoring into ``behaviour_events``.

GameThogram's TXT export is a frames × behaviours 0/1 matrix written by
``numpy.savetxt`` with header lines ``# COL1: <label>``. Labels are
``<animal>_<behaviour>`` or ``<behaviour>`` (single-animal sets). Columns
named ``delete`` are editing actions, not behaviours, and are skipped.

Each column is mapped to (actor role, behaviour) by name:

* behaviour: case-insensitive match on the ``behaviours`` table (spaces,
  hyphens and camel case normalised to ``snake_case``), e.g.
  ``WingExtension`` → ``wing_extension``;
* actor: the animal part matched against ``male_1`` / ``male_2`` /
  ``female_decap`` or the aliases below (``MaleLeft`` → ``male_1`` …).

Anything unresolved is reported and must be given explicitly with
``mapping={"<label>": ("male_2", "lunge")}``.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from thermokourt.db.core import now_iso, register_file

ROLE_ALIASES = {
    "male_1": (
        "male1",
        "male_1",
        "m1",
        "maleleft",
        "male_left",
        "left",
        "a",
        "animal0",
        "animal_0",
    ),
    "male_2": (
        "male2",
        "male_2",
        "m2",
        "maleright",
        "male_right",
        "right",
        "b",
        "animal1",
        "animal_1",
    ),
    "female_decap": ("female", "female_decap", "fem", "f", "decap", "animal2", "animal_2"),
}
_COL = re.compile(r"COL\d+:\s*(.+?)\s*$")


def _snake(s: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s)
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


@dataclass
class Column:
    label: str
    role: str | None
    behaviour: str | None


def read_txt(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Return (frames × columns int matrix, labels)."""
    p = Path(path)
    header = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.startswith("#")]
    data = np.loadtxt(p, dtype=int, ndmin=2)
    labels = [m.group(1) for ln in header if (m := _COL.search(ln))]
    if len(labels) != data.shape[1]:
        labels = [f"COL{i + 1}" for i in range(data.shape[1])]
    return data, labels


def resolve_columns(
    labels: list[str], behaviour_names: list[str], mapping: dict[str, tuple[str, str]] | None = None
) -> list[Column]:
    mapping = mapping or {}
    beh_by_snake = {_snake(b): b for b in behaviour_names}
    out = []
    for lab in labels:
        if lab in mapping:
            r, b = mapping[lab]
            out.append(Column(lab, r, b))
            continue
        sl = _snake(lab)
        if "delete" in sl:
            out.append(Column(lab, None, None))
            continue
        role, beh = None, None
        # behaviour: longest matching behaviour name at the end of the label
        for bs, bname in sorted(beh_by_snake.items(), key=lambda kv: -len(kv[0])):
            if sl == bs or sl.endswith("_" + bs):
                beh, prefix = bname, sl[: -len(bs)].strip("_") if sl != bs else ""
                break
        if beh is not None:
            for r, aliases in ROLE_ALIASES.items():
                if prefix and prefix.replace("_", "") in [a.replace("_", "") for a in aliases]:
                    role = r
                    break
        out.append(Column(lab, role, beh))
    return out


def import_txt(
    con: sqlite3.Connection,
    path: str | Path,
    trial_id: str,
    annotator_name: str,
    fps: float | None = None,
    mapping: dict[str, tuple[str, str]] | None = None,
    default_role: str | None = None,
    replace: bool = True,
) -> dict:
    """Insert the file's bouts as events of a human annotator. Returns a summary."""
    data, labels = read_txt(path)
    row = con.execute("SELECT fps FROM trials WHERE trial_id=?", (trial_id,)).fetchone()
    if row is None:
        raise LookupError(f"trial {trial_id} not in the database")
    fps = fps or row[0] or 25.0
    beh_ids = {r[0]: r[1] for r in con.execute("SELECT name, behaviour_id FROM behaviours")}
    fly_ids = {
        r[0]: r[1]
        for r in con.execute("SELECT role, fly_id FROM flies WHERE trial_id=?", (trial_id,))
    }
    cols = resolve_columns(labels, list(beh_ids), mapping)
    unresolved = [c.label for c in cols if c.behaviour is None and "delete" not in _snake(c.label)]
    if unresolved:
        raise ValueError(f"columns not mapped to a behaviour: {unresolved}; pass mapping=...")

    con.execute(
        "INSERT OR IGNORE INTO annotators (kind, name, version) VALUES ('human', ?, NULL)",
        (annotator_name,),
    )
    ann_id = con.execute(
        "SELECT annotator_id FROM annotators WHERE kind='human' AND name=? AND version IS NULL",
        (annotator_name,),
    ).fetchone()[0]
    fid = register_file(con, path, "ethogram")
    if replace:
        con.execute(
            "DELETE FROM behaviour_events WHERE trial_id=? AND annotator_id=?", (trial_id, ann_id)
        )

    n_events = 0
    rows = []
    for j, c in enumerate(cols):
        if c.behaviour is None:
            continue
        role = c.role or default_role
        actor = fly_ids.get(role) if role else None
        m = np.concatenate([[0], data[:, j] > 0, [0]]).astype(int)
        d = np.diff(m)
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0], strict=True):
            rows.append(
                (trial_id, ann_id, beh_ids[c.behaviour], actor, None, s / fps, e / fps, None, fid)
            )
            n_events += 1
    con.executemany(
        """INSERT INTO behaviour_events
           (trial_id, annotator_id, behaviour_id, actor_fly_id, target_fly_id, start_s, end_s,
            confidence, source_file_id) VALUES (?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    con.commit()
    return {
        "trial_id": trial_id,
        "annotator_id": ann_id,
        "n_frames": int(data.shape[0]),
        "fps": fps,
        "n_events": n_events,
        "columns": [(c.label, c.role or default_role, c.behaviour) for c in cols],
        "imported_at": now_iso(),
    }
