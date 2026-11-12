# ruff: noqa: E501
"""GameThogram TXT import → behaviour_events."""

from __future__ import annotations

import numpy as np
import pytest

from thermokourt.db import init_db
from thermokourt.db.gamethogram import import_txt, read_txt, resolve_columns
from thermokourt.db.seed import seed


@pytest.fixture
def db(tmp_path):
    con = init_db(tmp_path / "g.sqlite3")
    seed(con)
    con.execute("INSERT INTO coded_groups (group_code, experiment_id) VALUES ('G01','E1')")
    con.execute(
        """INSERT INTO trials (trial_id, experiment_id, group_code, stimulus, test_temp_c, start_time,
           duration_s, fps, recording_dir, arena_index, created_at)
           VALUES ('T1','E1','G01','triad',28.0,'2026-12-04T01:00:00+13:00',10.0,25.0,'/r',0,'x')"""
    )
    con.executemany("INSERT INTO flies (trial_id, role) VALUES ('T1', ?)", [("male_1",), ("male_2",), ("female_decap",)])
    con.commit()
    return con


def write_export(path, labels, data):
    header = "\n".join(f"COL{i + 1}: {lab}" for i, lab in enumerate(labels))
    np.savetxt(path, data, fmt="%2i", header=header)


def test_read_and_resolve(tmp_path):
    p = tmp_path / "score.txt"
    data = np.zeros((50, 4), int)
    write_export(p, ["MaleLeft_lunge", "MaleRight_WingExtension", "MaleLeft_delete", "wing threat"], data)
    d, labels = read_txt(p)
    assert d.shape == (50, 4) and labels[1] == "MaleRight_WingExtension"
    cols = resolve_columns(labels, ["lunge", "wing_extension", "wing_threat", "following"])
    assert (cols[0].role, cols[0].behaviour) == ("male_1", "lunge")
    assert (cols[1].role, cols[1].behaviour) == ("male_2", "wing_extension")
    assert cols[2].behaviour is None  # delete column skipped
    assert (cols[3].role, cols[3].behaviour) == (None, "wing_threat")  # no animal in label


def test_import_creates_bouts(db, tmp_path):
    p = tmp_path / "T1_JC.txt"
    data = np.zeros((250, 2), int)
    data[25:35, 0] = 1  # male_1 lunge 1.0–1.4 s
    data[100:103, 0] = 1  # second lunge
    data[50:150, 1] = 1  # male_2 wing extension 2–6 s
    write_export(p, ["MaleLeft_lunge", "MaleRight_wing_extension"], data)
    summary = import_txt(db, p, "T1", "JC")
    assert summary["n_events"] == 3 and summary["fps"] == 25.0
    rows = db.execute(
        """SELECT b.name, f.role, e.start_s, e.end_s FROM behaviour_events e
           JOIN behaviours b USING (behaviour_id) LEFT JOIN flies f ON f.fly_id = e.actor_fly_id
           ORDER BY e.start_s"""
    ).fetchall()
    assert [tuple(r) for r in rows] == [
        ("lunge", "male_1", 1.0, 1.4),
        ("wing_extension", "male_2", 2.0, 6.0),
        ("lunge", "male_1", 4.0, 4.12),
    ]
    # re-import replaces, does not duplicate
    import_txt(db, p, "T1", "JC")
    assert db.execute("SELECT COUNT(*) FROM behaviour_events").fetchone()[0] == 3
    ann = db.execute("SELECT kind, name FROM annotators WHERE name='JC'").fetchone()
    assert tuple(ann) == ("human", "JC")


def test_unmapped_column_is_an_error(db, tmp_path):
    p = tmp_path / "x.txt"
    write_export(p, ["MaleLeft_headbutt"], np.zeros((10, 1), int))
    with pytest.raises(ValueError):
        import_txt(db, p, "T1", "JC")
    import_txt(db, p, "T1", "JC", mapping={"MaleLeft_headbutt": ("male_1", "lunge")})
