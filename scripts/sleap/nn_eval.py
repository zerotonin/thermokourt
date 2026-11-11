#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  ThermoKourt — nn_eval                                           ║
# ║  « one model on one validation set, as numbers »                 ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Compare predictions with hand labels and write the headline numbers as JSON.

Usage (SLEAP 1.6 env)::

    python nn_eval.py VAL.slp PRED.slp OUT.json --model lab_v3 --set val_440px
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def headline(metrics: dict) -> dict:
    """Pick the numbers we compare models by from an Evaluator result."""
    voc = metrics.get("voc_metrics", {})
    dist = metrics.get("distance_metrics", {})
    det = metrics.get("detection_metrics", {})

    def num(x):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)

    return {
        "mAP": num(voc.get("oks_voc.mAP")),
        "mAR": num(voc.get("oks_voc.mAR")),
        "mOKS": num(metrics.get("mOKS", {}).get("mOKS")),
        "dist_p50_px": num(dist.get("p50")),
        "dist_p90_px": num(dist.get("p90")),
        "recall": num(det.get("recall")),
        "precision": num(det.get("precision")),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gt")
    ap.add_argument("pred")
    ap.add_argument("out")
    ap.add_argument("--model", default="")
    ap.add_argument("--set", dest="val_set", default="")
    a = ap.parse_args(argv)

    import sleap_io as sio
    from sleap_nn.evaluation import Evaluator

    gt = sio.load_slp(a.gt)
    ev = Evaluator(gt, sio.load_slp(a.pred), user_labels_only=True)
    row = {"model": a.model, "set": a.val_set, "frames": len(gt.user_labeled_frames)}
    row.update(headline(ev.evaluate()))
    Path(a.out).write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(json.dumps(row))


if __name__ == "__main__":
    main()
