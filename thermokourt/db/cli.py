"""``thermokourt-db`` — command-line front end for the behaviour database.

    thermokourt-db init     DB                       create tables + seed rows
    thermokourt-db vials    DB SNAPSHOT QUERY        search DrosoDB vials (picker)
    thermokourt-db import-ddb DB SNAPSHOT CODE...    copy vials + lineage
    thermokourt-db add-trial DB SPEC.json            register one trial
    thermokourt-db export   DB OUT.sqlite3 [--blind] copy without genotype tables
    thermokourt-db status   DB                       row counts
    thermokourt-db import-gamethogram DB TXT --trial T --annotator JC
                                    [--map LABEL=ROLE:BEHAVIOUR ...]
    thermokourt-db agreement DB --ref JC --pred rules  per-cell human vs automatic agreement

``SPEC.json`` for add-trial (written by the trial GUI or by hand):

    {"recording_dir": "/mnt/DATA/20261204_141233",
     "arenas_json": "/mnt/DATA/20261204_141233/20261204_141233_arenas.json",
     "ddb_snapshot": "/home/lab/data/drosodb/ddb.latest.sqlite3",
     "experiment_id": "E1", "group_code": "G01", "test_temp_c": 28,
     "stimulus": "triad", "operator": "JC",
     "trials": [{"arena_index": 0, "flies": [
         {"role": "male_1", "print_code": "Ab13", "age_days": 3},
         {"role": "male_2", "print_code": "Ab13", "age_days": 3},
         {"role": "female_decap", "print_code": "Cd44"}]}]}
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from thermokourt.db import ddb as ddbmod
from thermokourt.db import gamethogram
from thermokourt.db.agreement import agreement, format_table
from thermokourt.db.core import connect, init_db
from thermokourt.db.seed import seed
from thermokourt.db.trials import FlySpec, TrialSpec, add_trial, ensure_group

BLIND_DROP_TABLES = ("ddb_genotype", "ddb_vial", "ddb_vial_parentage", "ddb_donor", "ddb_snapshots")


def cmd_init(a):
    con = init_db(a.db)
    seed(con)
    print(f"initialised {a.db}")


def cmd_vials(a):
    ddb = ddbmod.open_snapshot(a.snapshot)
    for v in ddbmod.find_vials(ddb, a.query, active_only=not a.all):
        flag = "" if v.is_active else "  (decommissioned)"
        print(f"{v.print_code:6s} gen {v.generation or '?':>2}  {v.genotype_name}{flag}")


def cmd_import_ddb(a):
    con = connect(a.db)
    res = ddbmod.import_vials_by_print_code(con, a.snapshot, a.codes)
    for code, v in res.items():
        print(f"{code}: vial {v.vial_id} -> {v.genotype_name}")


def cmd_add_trial(a):
    spec = json.loads(Path(a.spec).read_text())
    con = connect(a.db)
    ensure_group(con, spec["group_code"], spec["experiment_id"], spec.get("cross_label"))
    codes = sorted(
        {f["print_code"] for t in spec["trials"] for f in t.get("flies", []) if f.get("print_code")}
    )
    vials = {}
    if codes:
        if not spec.get("ddb_snapshot"):
            sys.exit("print codes given but no 'ddb_snapshot' in the spec")
        vials = ddbmod.import_vials_by_print_code(con, spec["ddb_snapshot"], codes)
    for t in spec["trials"]:
        flies = [
            FlySpec(
                role=f["role"],
                print_code=f.get("print_code"),
                ddb_vial_id=vials[f["print_code"]].vial_id if f.get("print_code") else None,
                eclosion_date=f.get("eclosion_date"),
                age_days=f.get("age_days"),
                marker_scored=f.get("marker_scored"),
                pcr_tube_label=f.get("pcr_tube_label"),
            )
            for f in t.get("flies", [])
        ]
        ts = TrialSpec(
            arena_index=int(t["arena_index"]),
            experiment_id=spec["experiment_id"],
            group_code=spec["group_code"],
            test_temp_c=float(spec["test_temp_c"]),
            stimulus=spec.get("stimulus", "triad"),
            ramp_c_per_h=spec.get("ramp_c_per_h"),
            rearing_temp_c=spec.get("rearing_temp_c"),
            treatment=spec.get("treatment", "none"),
            operator=spec.get("operator"),
            notes=t.get("notes") or spec.get("notes"),
            flies=flies,
        )
        try:
            tid = add_trial(
                con,
                spec["recording_dir"],
                spec["arenas_json"],
                ts,
                clip_path=t.get("clip_path"),
                temp_log=spec.get("temp_log"),
                mm_per_pixel=spec.get("mm_per_pixel"),
            )
        except sqlite3.IntegrityError:
            print(f"arena {t['arena_index']}: already registered, skipped")
            continue
        print(f"arena {t['arena_index']}: trial {tid}")


def cmd_export(a):
    src = connect(a.db)
    dst = sqlite3.connect(a.out)
    src.backup(dst)
    if a.blind:
        for t in BLIND_DROP_TABLES:
            dst.execute(f"DROP TABLE IF EXISTS {t}")
        dst.execute("UPDATE coded_groups SET ddb_genotype_id = NULL")
        dst.execute("UPDATE flies SET ddb_vial_id = NULL, ddb_print_code = NULL")
        dst.execute("INSERT OR REPLACE INTO schema_info VALUES ('blinded', '1')")
        dst.commit()
        dst.execute("VACUUM")
    dst.close()
    print(f"wrote {a.out}{' (blinded)' if a.blind else ''}")


def cmd_import_gamethogram(a):
    con = connect(a.db)
    mapping = {}
    for m in a.map or []:
        lab, rest = m.split("=", 1)
        role, beh = rest.split(":", 1)
        mapping[lab] = (role or None, beh)
    r = gamethogram.import_txt(
        con, a.txt, a.trial, a.annotator, fps=a.fps, mapping=mapping, default_role=a.default_role
    )
    print(
        f"{r['n_events']} events from {r['n_frames']} frames @ {r['fps']} fps "
        f"into trial {r['trial_id']}"
    )
    for lab, role, beh in r["columns"]:
        print(f"  {lab:30s} -> {role or '-':13s} {beh or '(skipped)'}")


def cmd_agreement(a):
    con = connect(a.db)

    def ann_id(name):
        row = con.execute(
            "SELECT annotator_id FROM annotators WHERE name=? ORDER BY version DESC", (name,)
        ).fetchone()
        if row is None:
            sys.exit(f"no annotator named {name!r}")
        return row[0]

    rows = agreement(con, ann_id(a.ref), ann_id(a.pred), min_overlap_s=a.min_overlap)
    print(format_table(rows) if rows else "no trials scored by both annotators")


def cmd_status(a):
    con = connect(a.db)
    for t in (
        "trials",
        "flies",
        "behaviour_events",
        "window_summary",
        "temperature_samples",
        "ddb_vial",
        "ddb_genotype",
        "source_files",
    ):
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"{t:20s} {n}")
    for r in con.execute("SELECT qc_flag, COUNT(*) FROM trials GROUP BY qc_flag"):
        print(f"  trials {r[0]}: {r[1]}")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="thermokourt-db",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init")
    s.add_argument("db")
    s.set_defaults(fn=cmd_init)
    s = sub.add_parser("vials")
    s.add_argument("db")
    s.add_argument("snapshot")
    s.add_argument("query")
    s.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_vials)
    s = sub.add_parser("import-ddb")
    s.add_argument("db")
    s.add_argument("snapshot")
    s.add_argument("codes", nargs="+")
    s.set_defaults(fn=cmd_import_ddb)
    s = sub.add_parser("add-trial")
    s.add_argument("db")
    s.add_argument("spec")
    s.set_defaults(fn=cmd_add_trial)
    s = sub.add_parser("export")
    s.add_argument("db")
    s.add_argument("out")
    s.add_argument("--blind", action="store_true")
    s.set_defaults(fn=cmd_export)
    s = sub.add_parser("import-gamethogram")
    s.add_argument("db")
    s.add_argument("txt")
    s.add_argument("--trial", required=True)
    s.add_argument("--annotator", required=True, help="initials of the scorer")
    s.add_argument("--fps", type=float, default=None)
    s.add_argument(
        "--default-role", default=None, help="actor when the column label names no animal"
    )
    s.add_argument("--map", action="append", metavar="LABEL=ROLE:BEHAVIOUR")
    s.set_defaults(fn=cmd_import_gamethogram)
    s = sub.add_parser("agreement")
    s.add_argument("db")
    s.add_argument("--ref", required=True, help="reference annotator name (human initials)")
    s.add_argument("--pred", default="rules", help="annotator to evaluate (default: rules)")
    s.add_argument("--min-overlap", type=float, default=0.1)
    s.set_defaults(fn=cmd_agreement)
    s = sub.add_parser("status")
    s.add_argument("db")
    s.set_defaults(fn=cmd_status)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
