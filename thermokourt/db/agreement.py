"""Human-vs-automatic agreement per genotype × temperature cell.

The crossing guide requires the classifier to be validated per cell, not
overall. Two measures per behaviour and cell:

* **frame-level** precision / recall / F1 — both annotators' events are
  rasterised onto the trial's frame grid (restricted to the window the human
  scored) and compared frame by frame;
* **bout-level** precision / recall — a predicted bout counts as a hit when it
  overlaps a reference bout by at least ``min_overlap_s`` (so one 15 s bout
  touching one frame of a reference does not count, cf. the DANCE review).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np


@dataclass
class Agreement:
    group_code: str
    test_temp_c: float
    behaviour: str
    n_trials: int
    frame_precision: float
    frame_recall: float
    frame_f1: float
    bout_precision: float
    bout_recall: float
    n_ref_bouts: int
    n_pred_bouts: int


def _events(con, trial_id, annotator_id, behaviour_id):
    return con.execute(
        "SELECT start_s, end_s FROM behaviour_events"
        " WHERE trial_id=? AND annotator_id=? AND behaviour_id=?",
        (trial_id, annotator_id, behaviour_id),
    ).fetchall()


def _raster(events, t0, t1, fps):
    n = int(round((t1 - t0) * fps))
    m = np.zeros(max(n, 1), bool)
    for s, e in events:
        a = max(0, int(round((s - t0) * fps)))
        b = min(n, int(round((e - t0) * fps)))
        if b > a:
            m[a:b] = True
    return m


def _bout_hits(pred, ref, min_overlap_s):
    hits = 0
    for ps, pe in pred:
        if any(min(pe, re) - max(ps, rs) >= min_overlap_s for rs, re in ref):
            hits += 1
    return hits


def agreement(
    con: sqlite3.Connection,
    ref_annotator_id: int,
    pred_annotator_id: int,
    windows: dict[str, tuple[float, float]] | None = None,
    min_overlap_s: float = 0.1,
) -> list[Agreement]:
    """Compare ``pred`` (rules/model) against ``ref`` (human) per cell.

    ``windows`` maps trial_id -> (t0, t1) of the scored window; trials
    without a window are compared over their whole duration.
    """
    beh = {r[0]: r[1] for r in con.execute("SELECT behaviour_id, name FROM behaviours")}
    trials = con.execute(
        """SELECT DISTINCT t.trial_id, t.group_code, t.test_temp_c, t.fps, t.duration_s
           FROM trials t JOIN behaviour_events e ON e.trial_id = t.trial_id
           WHERE e.annotator_id = ? AND t.qc_flag <> 'exclude'""",
        (ref_annotator_id,),
    ).fetchall()
    cells: dict[tuple, dict] = {}
    for tid, group, temp, fps, dur in trials:
        fps = fps or 25.0
        t0, t1 = (windows or {}).get(tid, (0.0, dur or 0.0))
        if t1 <= t0:
            # no duration known: span the union of events
            evs = con.execute(
                "SELECT MIN(start_s), MAX(end_s) FROM behaviour_events WHERE trial_id=?", (tid,)
            ).fetchone()
            t0, t1 = float(evs[0] or 0), float(evs[1] or 0)
        for bid, name in beh.items():
            ref = _events(con, tid, ref_annotator_id, bid)
            pred = _events(con, tid, pred_annotator_id, bid)
            if not ref and not pred:
                continue
            pred_w = [(s, e) for s, e in pred if e > t0 and s < t1]
            rr, pp = _raster(ref, t0, t1, fps), _raster(pred_w, t0, t1, fps)
            c = cells.setdefault(
                (group, temp, name),
                {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "trials": set(),
                    "bh": 0,
                    "bp": 0,
                    "br": 0,
                    "nref": 0,
                    "npred": 0,
                },
            )
            c["tp"] += int(np.sum(rr & pp))
            c["fp"] += int(np.sum(~rr & pp))
            c["fn"] += int(np.sum(rr & ~pp))
            c["trials"].add(tid)
            c["bh"] += _bout_hits(
                pred_w, ref, min_overlap_s
            )  # predicted bouts that hit a reference
            c["br"] += _bout_hits(ref, pred_w, min_overlap_s)  # reference bouts that were found
            c["nref"] += len(ref)
            c["npred"] += len(pred_w)
    out = []
    for (group, temp, name), c in sorted(cells.items()):
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * p * r / (p + r) if (p + r) and not np.isnan(p + r) else float("nan")
        out.append(
            Agreement(
                group,
                temp,
                name,
                len(c["trials"]),
                p,
                r,
                f1,
                c["bh"] / c["npred"] if c["npred"] else float("nan"),
                c["br"] / c["nref"] if c["nref"] else float("nan"),
                c["nref"],
                c["npred"],
            )
        )
    return out


def format_table(rows: list[Agreement]) -> str:
    hdr = (
        f"{'group':6s} {'°C':>5s} {'behaviour':16s} {'n':>3s} {'fP':>5s} {'fR':>5s} "
        f"{'F1':>5s} {'bP':>5s} {'bR':>5s} {'ref':>4s} {'pred':>4s}"
    )
    lines = [hdr, "-" * len(hdr)]
    for a in rows:
        f = lambda x: "  -  " if np.isnan(x) else f"{x:5.2f}"  # noqa: E731
        lines.append(
            f"{a.group_code:6s} {a.test_temp_c:5.1f} {a.behaviour:16s} {a.n_trials:3d} "
            f"{f(a.frame_precision)} {f(a.frame_recall)} {f(a.frame_f1)} "
            f"{f(a.bout_precision)} {f(a.bout_recall)} {a.n_ref_bouts:4d} {a.n_pred_bouts:4d}"
        )
    return "\n".join(lines)
