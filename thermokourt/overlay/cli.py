# ╔══════════════════════════════════════════════════════════════════╗
# ║  ThermoKourt — overlay.cli                                       ║
# ║  « thermokourt-overlay: one annotated clip »                     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Called by Rig Conductor per arena; prints '@progress' lines.    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""``thermokourt-overlay``: draw skeletons and/or events onto an arena clip."""

from __future__ import annotations

import argparse
from pathlib import Path

from thermokourt.overlay.render import DEFAULT_SCALE, render_trial_clip


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("clip", type=Path, help="arena clip (mp4)")
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--h5", type=Path, default=None, help="SLEAP analysis file of this arena")
    p.add_argument("--db", type=Path, required=True, help="ThermoKourt database")
    p.add_argument("--trial", required=True, help="trial id")
    p.add_argument("--fps", type=float, required=True, help="real frame rate of the clip")
    p.add_argument("--skeleton", action="store_true")
    p.add_argument("--events", action="store_true")
    p.add_argument("--scale", type=int, default=DEFAULT_SCALE)
    p.add_argument("--n-animals", type=int, default=3)
    p.add_argument("--tag", default="", help="label in progress lines (e.g. 03)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if not (a.skeleton or a.events):
        raise SystemExit("choose --skeleton, --events or both")

    def progress(done, total):
        frac = done / total if total else 0.0
        print(f"@progress clip={a.tag} frac={min(1.0, frac):.3f}", flush=True)

    out = render_trial_clip(
        a.clip,
        a.out,
        a.h5,
        a.db,
        a.trial,
        a.fps,
        skeleton=a.skeleton,
        show_events=a.events,
        n_animals=a.n_animals,
        scale=a.scale,
        progress=progress,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
