# ruff: noqa: E501
"""Frame- and bout-level agreement between a human and the rules annotator."""

from __future__ import annotations

import numpy as np
import pytest

from thermokourt.db import init_db
from thermokourt.db.agreement import agreement, format_table
from thermokourt.db.seed import seed


@pytest.fixture
def db(tmp_path):
    con = init_db(tmp_path / "a.sqlite3")
    seed(con)
    con.execute("INSERT INTO coded_groups (group_code, experiment_id) VALUES ('G01','E1')")
    con.executemany(
        """INSERT INTO trials (trial_id, experiment_id, group_code, stimulus, test_temp_c, start_time,
           duration_s, fps, recording_dir, arena_index, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        [("T1", "E1", "G01", "triad", 28.0, "2026-12-04T01:00:00+13:00", 120.0, 25.0, "/r1", 0, "x"),
         ("T2", "E1", "G01", "triad", 33.0, "2026-12-04T02:00:00+13:00", 120.0, 25.0, "/r1", 1, "x")],
    )
    con.execute("INSERT INTO annotators (kind, name, version) VALUES ('human','JC',NULL)")
    con.commit()
    return con


def _ids(con):
    human = con.execute("SELECT annotator_id FROM annotators WHERE kind='human'").fetchone()[0]
    rules = con.execute("SELECT annotator_id FROM annotators WHERE kind='rule'").fetchone()[0]
    lunge = con.execute("SELECT behaviour_id FROM behaviours WHERE name='lunge'").fetchone()[0]
    wing = con.execute("SELECT behaviour_id FROM behaviours WHERE name='wing_extension'").fetchone()[0]
    return human, rules, lunge, wing


def test_perfect_agreement(db):
    human, rules, lunge, _ = _ids(db)
    ev = [("T1", a, lunge, 10.0, 10.4) for a in (human, rules)]
    db.executemany("INSERT INTO behaviour_events (trial_id, annotator_id, behaviour_id, start_s, end_s) VALUES (?,?,?,?,?)", ev)
    db.commit()
    rows = agreement(db, human, rules)
    assert len(rows) == 1
    a = rows[0]
    assert a.group_code == "G01" and a.test_temp_c == 28.0 and a.behaviour == "lunge"
    assert a.frame_f1 == pytest.approx(1.0) and a.bout_precision == 1.0 and a.bout_recall == 1.0


def test_partial_overlap_and_misses(db):
    human, rules, lunge, wing = _ids(db)
    # T1: human 2 lunges; rules finds one of them (overlapping) and one false bout
    db.executemany(
        "INSERT INTO behaviour_events (trial_id, annotator_id, behaviour_id, start_s, end_s) VALUES (?,?,?,?,?)",
        [("T1", human, lunge, 10.0, 10.4), ("T1", human, lunge, 50.0, 50.4),
         ("T1", rules, lunge, 10.1, 10.5), ("T1", rules, lunge, 80.0, 80.3),
         # T2 (33 °C): a 10 s human wing extension, rules reports 20 s covering it
         ("T2", human, wing, 20.0, 30.0), ("T2", rules, wing, 15.0, 35.0)],
    )
    db.commit()
    rows = {(r.test_temp_c, r.behaviour): r for r in agreement(db, human, rules)}
    lg = rows[(28.0, "lunge")]
    assert lg.n_ref_bouts == 2 and lg.n_pred_bouts == 2
    assert lg.bout_recall == 0.5 and lg.bout_precision == 0.5
    assert lg.frame_recall == pytest.approx(0.3 / 0.8, abs=0.06)  # 0.3 s hit of 0.8 s reference
    w = rows[(33.0, "wing_extension")]
    assert w.frame_recall == pytest.approx(1.0)
    assert w.frame_precision == pytest.approx(0.5, abs=0.02)
    assert w.bout_precision == 1.0 and w.bout_recall == 1.0
    txt = format_table(list(rows.values()))
    assert "wing_extension" in txt and "lunge" in txt


def test_window_restricts_comparison(db):
    human, rules, lunge, _ = _ids(db)
    db.executemany(
        "INSERT INTO behaviour_events (trial_id, annotator_id, behaviour_id, start_s, end_s) VALUES (?,?,?,?,?)",
        [("T1", human, lunge, 10.0, 10.4), ("T1", rules, lunge, 10.0, 10.4), ("T1", rules, lunge, 100.0, 100.4)],
    )
    db.commit()
    # the human only scored 0–60 s: the rules' bout at 100 s must not count as a false positive
    a = agreement(db, human, rules, windows={"T1": (0.0, 60.0)})[0]
    assert a.n_pred_bouts == 1 and a.frame_precision == pytest.approx(1.0)
    b = agreement(db, human, rules)[0]
    assert b.n_pred_bouts == 2 and b.bout_precision == 0.5


def test_min_overlap_rejects_touching_bouts(db):
    human, rules, lunge, _ = _ids(db)
    db.executemany(
        "INSERT INTO behaviour_events (trial_id, annotator_id, behaviour_id, start_s, end_s) VALUES (?,?,?,?,?)",
        [("T1", human, lunge, 10.0, 10.4), ("T1", rules, lunge, 10.38, 12.0)],
    )
    db.commit()
    a = agreement(db, human, rules, min_overlap_s=0.1)[0]
    assert a.bout_recall == 0.0 and a.bout_precision == 0.0
    assert not np.isnan(a.frame_recall)
