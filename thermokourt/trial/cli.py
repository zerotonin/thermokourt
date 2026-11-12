"""``thermokourt-trial`` — run or resume a recording session's pipeline.

    thermokourt-trial init-config                 write ~/.config/thermokourt/rig.yaml
    thermokourt-trial run SESSION.json [--from STAGE] [--to STAGE]
    thermokourt-trial status RECORDING_DIR
    thermokourt-trial gui                         open the operator window

SESSION.json is what the GUI writes; by hand it looks like:

    {"recording_dir": "/mnt/DATA/20261204_141233",
     "experiment_id": "E1", "group_code": "G01", "test_temp_c": 28,
     "operator": "JC",
     "arenas": {"0": [{"role": "male_1", "print_code": "Ab13", "age_days": 3},
                      {"role": "male_2", "print_code": "Ab13", "age_days": 3},
                      {"role": "female_decap", "print_code": "Cd44"}]}}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thermokourt.trial.config import RigConfig, write_example
from thermokourt.trial.runner import STAGES, Session, SessionSpec


def cmd_init_config(a):
    try:
        p = write_example(a.path, overwrite=a.force)
    except FileExistsError as e:
        raise SystemExit(f"{e} exists; use --force to overwrite") from None
    print(f"wrote {p}; edit the paths before the first run")


def cmd_run(a):
    cfg = RigConfig.load(a.config)
    spec = SessionSpec.from_json(json.loads(Path(a.session).read_text()))
    i0 = STAGES.index(a.from_stage) if a.from_stage else 0
    i1 = STAGES.index(a.to_stage) + 1 if a.to_stage else len(STAGES)
    Session(cfg, spec).run(STAGES[i0:i1], poll_interval_s=a.poll)


def cmd_status(a):
    p = Path(a.recording_dir) / "thermokourt_session.json"
    if not p.is_file():
        print("no session state in", a.recording_dir)
        return
    st = json.loads(p.read_text())
    for s in STAGES:
        d = st["done"].get(s)
        print(f"{s:9s} {'done ' + d['at'] if d else '-'}")
    if st.get("job_id"):
        print("job", st["job_id"])
    for idx, tid in st.get("trials", {}).items():
        print(f"arena {idx}: {tid}")


def cmd_gui(a):
    from thermokourt.trial.gui import main as gui_main

    gui_main(a.config)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="thermokourt-trial",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", default=None, help="rig.yaml (default: ~/.config/thermokourt/rig.yaml)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init-config")
    s.add_argument("--path", default=None)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init_config)
    s = sub.add_parser("run")
    s.add_argument("session")
    s.add_argument("--from", dest="from_stage", choices=STAGES, default=None)
    s.add_argument("--to", dest="to_stage", choices=STAGES, default=None)
    s.add_argument("--poll", type=float, default=60, help="seconds between job checks")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("status")
    s.add_argument("recording_dir")
    s.set_defaults(fn=cmd_status)
    s = sub.add_parser("gui")
    s.set_defaults(fn=cmd_gui)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
