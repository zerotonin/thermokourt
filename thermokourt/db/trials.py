"""Register trials from a Motif recording + arenas JSON + the operator's form.

One Motif recording folder holds all arenas of one session; each arena with
flies in it becomes one trial. The trial ID is coded (no genotype in it):

    <YYYYMMDD>_<HHMM>_A<arena>     e.g. 20261204_1412_A03

Video file names keep that ID, so scorers never see the group.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from thermokourt.db.core import now_iso, register_file


@dataclass
class FlySpec:
    role: str  # male_1 | male_2 | female_decap
    print_code: str | None = None  # DrosoDB vial print code
    ddb_vial_id: int | None = None
    eclosion_date: str | None = None
    age_days: float | None = None
    marker_scored: str | None = None
    pcr_tube_label: str | None = None


@dataclass
class TrialSpec:
    arena_index: int
    experiment_id: str
    group_code: str
    test_temp_c: float
    stimulus: str = "triad"
    ramp_c_per_h: float | None = None
    rearing_temp_c: float | None = None
    treatment: str = "none"
    operator: str | None = None
    notes: str | None = None
    flies: list[FlySpec] = field(default_factory=list)


def read_motif_metadata(recording_dir: str | Path) -> dict:
    """Motif's ``metadata.yaml``: camera, fps, start time, frame shape."""
    p = Path(recording_dir) / "metadata.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"no metadata.yaml in {recording_dir}")
    with open(p) as f:
        return yaml.safe_load(f)


def trial_id_for(recording_dir: str | Path, arena_index: int, start_iso: str | None = None) -> str:
    """Coded trial ID from the recording start time and the arena index."""
    stamp = None
    if start_iso:
        s = start_iso.replace("-", "").replace(":", "")
        if "T" in s:
            d, t = s.split("T", 1)
            stamp = f"{d[:8]}_{t[:4]}"
    if stamp is None:
        # Motif folders are named <YYYYMMDD>_<HHMMSS> (possibly with a prefix)
        name = Path(recording_dir).name
        parts = name.split("_")
        if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
            stamp = f"{parts[-2][:8]}_{parts[-1][:4]}"
        else:
            stamp = name
    return f"{stamp}_A{arena_index:02d}"


def load_arenas_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def ensure_group(
    con: sqlite3.Connection,
    group_code: str,
    experiment_id: str,
    cross_label: str | None = None,
    ddb_genotype_id: int | None = None,
    description: str | None = None,
) -> None:
    con.execute(
        """INSERT OR IGNORE INTO coded_groups
           (group_code, experiment_id, cross_label, ddb_genotype_id, description)
           VALUES (?,?,?,?,?)""",
        (group_code, experiment_id, cross_label, ddb_genotype_id, description),
    )


def add_trial(
    con: sqlite3.Connection,
    recording_dir: str | Path,
    arenas_json: str | Path,
    spec: TrialSpec,
    clip_path: str | Path | None = None,
    temp_log: str | Path | None = None,
    mm_per_pixel: float | None = None,
) -> str:
    """Insert one trial (and its flies). Returns the coded trial ID.

    Raises ``sqlite3.IntegrityError`` if that recording/arena already exists.
    """
    rec = Path(recording_dir)
    meta = read_motif_metadata(rec)
    arenas = load_arenas_json(arenas_json)
    by_idx = {a.get("idx", i): a for i, a in enumerate(arenas["arenas"])}
    if spec.arena_index not in by_idx:
        raise ValueError(f"arena {spec.arena_index} not in {arenas_json}")
    arena = by_idx[spec.arena_index]

    start = meta.get("__store", {}).get("created_utc")
    start_iso = str(start) + (
        "+00:00" if start and not str(start).endswith("Z") and "+" not in str(start) else ""
    )
    tid = trial_id_for(rec, spec.arena_index, start_iso)
    fps = float(meta.get("acquisitionframerate") or 0) or None

    video_id = register_file(con, rec, "video", hash_it=False)
    arenas_id = register_file(con, arenas_json, "arenas_json")
    clip_id = register_file(con, clip_path, "arena_clip") if clip_path else None
    temp_id = register_file(con, temp_log, "temp_log") if temp_log else None

    con.execute(
        """INSERT INTO trials
           (trial_id, experiment_id, group_code, stimulus, test_temp_c, ramp_c_per_h,
            rearing_temp_c, treatment, start_time, duration_s, fps, recording_dir,
            arena_index, arena_cx, arena_cy, arena_r, mm_per_pixel,
            video_file_id, clip_file_id, temp_log_file_id, arenas_file_id,
            operator, notes, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            tid,
            spec.experiment_id,
            spec.group_code,
            spec.stimulus,
            spec.test_temp_c,
            spec.ramp_c_per_h,
            spec.rearing_temp_c,
            spec.treatment,
            start_iso,
            None,
            fps,
            str(rec),
            spec.arena_index,
            arena["cx"],
            arena["cy"],
            arena["r"],
            mm_per_pixel,
            video_id,
            clip_id,
            temp_id,
            arenas_id,
            spec.operator,
            spec.notes,
            now_iso(),
        ),
    )
    con.executemany(
        """INSERT INTO flies
           (trial_id, role, ddb_vial_id, ddb_print_code, eclosion_date, age_days,
            marker_scored, pcr_tube_label)
           VALUES (?,?,?,?,?,?,?,?)""",
        [
            (
                tid,
                f.role,
                f.ddb_vial_id,
                f.print_code,
                f.eclosion_date,
                f.age_days,
                f.marker_scored,
                f.pcr_tube_label,
            )
            for f in spec.flies
        ],
    )
    con.commit()
    return tid


def set_qc(con: sqlite3.Connection, trial_id: str, flag: str, reason: str | None = None) -> None:
    con.execute(
        "UPDATE trials SET qc_flag = ?, exclusion_reason = ? WHERE trial_id = ?",
        (flag, reason, trial_id),
    )
    con.commit()
