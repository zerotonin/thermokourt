# ruff: noqa: E501  -- long SQL / fixture strings
"""Behaviour-database tests: schema, DrosoDB copy, trial registration, views."""

from __future__ import annotations

import sqlite3

import pytest
import yaml

from thermokourt.db import ddb as ddbmod
from thermokourt.db import init_db, schema_version
from thermokourt.db.seed import seed
from thermokourt.db.trials import FlySpec, TrialSpec, add_trial, ensure_group, trial_id_for

# ── a mock DrosoDB with the columns thermokourt reads ───────────────────

DDB_SCHEMA = """
CREATE TABLE donor (id INTEGER PRIMARY KEY, name TEXT, address TEXT, email TEXT);
CREATE TABLE genotype (
  id INTEGER PRIMARY KEY, name TEXT, chromosome_x TEXT, chromosome_2 TEXT,
  chromosome_3 TEXT, chromosome_4 TEXT, chromosome_y TEXT, phenotype TEXT, notes TEXT,
  is_wildtype INTEGER, is_in_stock INTEGER, donor_strain_id TEXT, donor_id INTEGER,
  created_by_id INTEGER, created_at TEXT);
CREATE TABLE vial (
  id INTEGER PRIMARY KEY, print_code TEXT, genotype_id INTEGER, org_unit_id INTEGER,
  owner_id INTEGER, is_active INTEGER, flipped_from_id INTEGER, generation INTEGER,
  notes TEXT, created_at TEXT, decommissioned_at TEXT);
CREATE TABLE vial_parentage (child_id INTEGER, parent_id INTEGER, role TEXT, created_at TEXT,
  PRIMARY KEY (child_id, parent_id, role));
CREATE TABLE audit_event (id INTEGER PRIMARY KEY, payload TEXT);
"""


@pytest.fixture
def ddb_file(tmp_path):
    p = tmp_path / "ddb.latest.sqlite3"
    con = sqlite3.connect(p)
    con.executescript(DDB_SCHEMA)
    con.execute("INSERT INTO donor VALUES (1,'Bloomington','','')")
    con.executemany(
        "INSERT INTO genotype (id,name,chromosome_x,chromosome_3,is_wildtype,donor_strain_id,donor_id) VALUES (?,?,?,?,?,?,?)",
        [
            (10, "Canton-S", "+", "+", 1, "64349", 1),
            (11, "w[1118]", "w[1118]", "+", 0, "5905", 1),
            (12, "w[1118]; TI{TrpA1[1]}", "w[1118]", "TrpA1[1]", 0, "26504", 1),
            (13, "TrpA1[1]/Df(3L)ED4415 (C1 F1)", "w[1118]", "TrpA1[1]/Df", 0, None, None),
        ],
    )
    con.executemany(
        "INSERT INTO vial (id,print_code,genotype_id,is_active,flipped_from_id,generation,created_at) VALUES (?,?,?,?,?,?,?)",
        [
            (100, "Ab12", 10, 0, None, 1, "2026-10-01 10:00:00"),
            (101, "Ab13", 10, 1, 100, 2, "2026-10-20 10:00:00"),  # CS flipped from 100
            (200, "Cd44", 12, 1, None, 1, "2026-11-01 09:00:00"),  # TrpA1 mother stock
            (201, "Cd45", 11, 1, None, 1, "2026-11-01 09:00:00"),  # w1118 father stock
            (300, "Ef77", 13, 1, None, 1, "2026-11-20 09:00:00"),  # cross vial
            (301, "Ab12", 11, 0, None, 1, "2025-01-01 09:00:00"),
        ],  # old decommissioned vial reusing a code
    )
    con.executemany(
        "INSERT INTO vial_parentage VALUES (?,?,?,?)",
        [(300, 200, "mother", ""), (300, 201, "father", "")],
    )
    con.commit()
    con.close()
    return p


@pytest.fixture
def db(tmp_path):
    con = init_db(tmp_path / "heatagg.sqlite3")
    seed(con)
    return con


# ── tests ────────────────────────────────────────────────────────────────


def test_schema_initialises_and_is_idempotent(tmp_path):
    con = init_db(tmp_path / "x.sqlite3")
    assert schema_version(con) == 1
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for t in (
        "trials",
        "flies",
        "behaviour_events",
        "window_summary",
        "ddb_vial",
        "temperature_samples",
    ):
        assert t in tables
    init_db(tmp_path / "x.sqlite3")  # second run must not fail
    assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_seed_rows(db):
    assert db.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 8
    cats = {r[0] for r in db.execute("SELECT DISTINCT category FROM behaviours")}
    assert {"aggression", "courtship", "locomotion", "outcome"} <= cats


def test_ddb_lookup_by_print_code(ddb_file):
    ddb = ddbmod.open_snapshot(ddb_file)
    v = ddbmod.vial_by_print_code(ddb, "ab13")  # case-insensitive
    assert v.vial_id == 101 and v.genotype_name == "Canton-S" and v.is_active
    assert [x.print_code for x in ddbmod.find_vials(ddb, "Cd")] == ["Cd44", "Cd45"]
    with pytest.raises(LookupError):
        ddbmod.vial_by_print_code(ddb, "Zz99")


def test_ddb_ambiguous_code_raises(ddb_file):
    ddb = ddbmod.open_snapshot(ddb_file)
    with pytest.raises(LookupError):
        ddbmod.vial_by_print_code(ddb, "Ab12")


def test_ddb_import_copies_lineage(db, ddb_file):
    res = ddbmod.import_vials_by_print_code(db, ddb_file, ["Ef77", "Ab13"])
    assert res["Ef77"].genotype_name.startswith("TrpA1")
    vials = {r[0] for r in db.execute("SELECT id FROM ddb_vial")}
    assert vials == {300, 200, 201, 101, 100}  # cross parents + flip ancestor
    assert db.execute("SELECT COUNT(*) FROM ddb_vial_parentage").fetchone()[0] == 2
    genos = {r[0] for r in db.execute("SELECT name FROM ddb_genotype")}
    assert {"Canton-S", "w[1118]"} <= genos
    assert db.execute("SELECT COUNT(*) FROM ddb_snapshots").fetchone()[0] == 1
    # re-import with the same snapshot: no duplicate snapshot, rows replaced not doubled
    ddbmod.import_vials_by_print_code(db, ddb_file, ["Ef77"])
    assert db.execute("SELECT COUNT(*) FROM ddb_snapshots").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM ddb_vial").fetchone()[0] == 5


def test_open_snapshot_rejects_non_ddb(tmp_path):
    p = tmp_path / "other.sqlite3"
    sqlite3.connect(p).executescript("CREATE TABLE foo (x);")
    with pytest.raises(ValueError):
        ddbmod.open_snapshot(p)


@pytest.fixture
def recording(tmp_path):
    rec = tmp_path / "20261204_141233"
    rec.mkdir()
    (rec / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "__store": {"created_utc": "2026-12-04T01:12:33.740206", "imgshape": [2160, 2600]},
                "acquisitionframerate": 25.0,
                "camera_serial": "40366151",
            }
        )
    )
    (rec / "000000.mp4").write_bytes(b"")
    arenas = rec / "20261204_141233_arenas.json"
    arenas.write_text(
        '{"recording": "20261204_141233", "frame_width": 2600, "frame_height": 2160, '
        '"arenas": [{"cx": 841, "cy": 266, "r": 233, "idx": 0}, {"cx": 1619, "cy": 267, "r": 241, "idx": 1}]}'
    )
    return rec, arenas


def test_trial_id_is_coded():
    assert trial_id_for("/x/20261204_141233", 3, "2026-12-04T01:12:33+00:00") == "20261204_0112_A03"
    assert trial_id_for("/x/droso_20261204_141233", 0) == "20261204_1412_A00"


def test_add_trial_and_views(db, ddb_file, recording):
    rec, arenas = recording
    ensure_group(db, "G01", "E1", description="pilot CS")
    vials = ddbmod.import_vials_by_print_code(db, ddb_file, ["Ab13"])
    spec = TrialSpec(
        arena_index=1,
        experiment_id="E1",
        group_code="G01",
        test_temp_c=28.0,
        operator="JC",
        flies=[
            FlySpec("male_1", "Ab13", vials["Ab13"].vial_id, age_days=3),
            FlySpec("male_2", "Ab13", vials["Ab13"].vial_id, age_days=3),
            FlySpec("female_decap", "Ab13", vials["Ab13"].vial_id),
        ],
    )
    tid = add_trial(db, rec, arenas, spec)
    assert tid == "20261204_0112_A01"
    t = db.execute("SELECT * FROM trials WHERE trial_id = ?", (tid,)).fetchone()
    assert t["arena_r"] == 241 and t["fps"] == 25.0 and t["qc_flag"] == "pending"
    assert db.execute("SELECT COUNT(*) FROM flies WHERE trial_id = ?", (tid,)).fetchone()[0] == 3
    # same recording + arena twice is an error
    with pytest.raises(sqlite3.IntegrityError):
        add_trial(db, rec, arenas, spec)
    # events + windows feed the rate view
    m1 = db.execute(
        "SELECT fly_id FROM flies WHERE trial_id=? AND role='male_1'", (tid,)
    ).fetchone()[0]
    ann = db.execute("SELECT annotator_id FROM annotators WHERE name='rules'").fetchone()[0]
    lunge = db.execute("SELECT behaviour_id FROM behaviours WHERE name='lunge'").fetchone()[0]
    db.executemany(
        "INSERT INTO behaviour_events (trial_id, annotator_id, behaviour_id, actor_fly_id, start_s, end_s) VALUES (?,?,?,?,?,?)",
        [(tid, ann, lunge, m1, 10.0, 10.2), (tid, ann, lunge, m1, 40.0, 40.3)],
    )
    db.execute(
        "INSERT INTO window_summary (trial_id, fly_id, window_start_s, speed_mm_s, distance_mm, n_encounters) VALUES (?,?,?,?,?,?)",
        (tid, m1, 0, 2.5, 150, 4),
    )
    db.commit()
    row = db.execute(
        "SELECT n_lunges, n_encounters, aggression_per_encounter FROM v_trial_rates WHERE trial_id=?",
        (tid,),
    ).fetchone()
    assert (
        row["n_lunges"] == 2 and row["n_encounters"] == 4 and row["aggression_per_encounter"] == 0.5
    )
    agg = db.execute("SELECT * FROM v_agreement").fetchall()
    assert len(agg) == 1 and agg[0]["behaviour"] == "lunge" and agg[0]["n_events"] == 2
