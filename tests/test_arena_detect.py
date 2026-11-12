"""Regression tests for arena detection on real rig projections.

Fixtures in ``tests/data`` are quarter-scale max-projections of two
recordings from the lab's Motif rig:

* ``projection_oct2025_10arenas`` — arenas fill the frame, grey discs on a
  white backlight (Hough-friendly).
* ``projection_jun2026_10arenas`` — zoomed out, arenas are bright holes in a
  dim plate, with a bright light pad and incubator lid in view (the case
  that broke the Hough-only detector).

Each ``.json`` holds the full-resolution detection result scaled to the
fixture, used as the reference for centre positions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from thermokourt.extract import arena_detect as ad
from thermokourt.extract.arena_extractor import detect_arenas

DATA = Path(__file__).parent / "data"
FIXTURES = ["projection_oct2025_10arenas", "projection_jun2026_10arenas"]


def _load(name: str):
    img = cv2.imread(str(DATA / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"missing fixture {name}.png"
    ref = json.loads((DATA / f"{name}.json").read_text())
    return img, ref


def _matched(found, ref_arenas, tol_frac=0.35):
    """Number of reference arenas with a found centre within tol_frac * r."""
    n = 0
    for b in ref_arenas:
        if any(math.hypot(a.cx - b["cx"], a.cy - b["cy"]) < b["r"] * tol_frac for a in found):
            n += 1
    return n


@pytest.mark.parametrize("name", FIXTURES)
def test_detects_all_arenas(name):
    img, ref = _load(name)
    arenas = detect_arenas(img, n_expected=ref["n"])
    assert len(arenas) == ref["n"]
    assert _matched(arenas, ref["arenas"]) >= ref["n"] - 1
    radii = [a.r for a in arenas]
    assert max(radii) / min(radii) < 1.5, "radii should be consistent across arenas"


@pytest.mark.parametrize("name", FIXTURES)
def test_detects_with_roi(name):
    img, ref = _load(name)
    xs = [a["cx"] for a in ref["arenas"]]
    ys = [a["cy"] for a in ref["arenas"]]
    r = max(a["r"] for a in ref["arenas"])
    roi = ad.parse_roi(
        f"{min(xs) - 2 * r},{min(ys) - 2 * r},{max(xs) - min(xs) + 4 * r},{max(ys) - min(ys) + 4 * r}",
        *img.shape,
    )
    arenas = detect_arenas(img, n_expected=ref["n"], roi=roi)
    assert len(arenas) == ref["n"]
    assert _matched(arenas, ref["arenas"]) >= ref["n"] - 1
    # results must be in full-frame coordinates, not ROI-relative
    assert all(roi[0] <= a.cx <= roi[0] + roi[2] for a in arenas)


def test_rgb_and_grey_agree():
    img, ref = _load(FIXTURES[0])
    grey = detect_arenas(img, n_expected=ref["n"])
    rgb = detect_arenas(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), n_expected=ref["n"])
    assert len(grey) == len(rgb)


def test_empty_frame_gives_nothing():
    blank = np.full((300, 400), 200, np.uint8)
    assert detect_arenas(blank, n_expected=10) == []


def test_radius_windows_cover_range():
    wins = ad.radius_windows(64, 432)
    assert wins[0][0] == 64
    assert wins[-1][1] >= 432
    for (lo, hi), (nlo, _) in zip(wins, wins[1:]):
        assert nlo < hi, "windows must overlap"
    assert len(ad.radius_windows(100, 110)) == 1


def test_parse_roi():
    assert ad.parse_roi(None, 100, 200) == (0, 0, 200, 100)
    assert ad.parse_roi("10,20,50,30", 100, 200) == (10, 20, 50, 30)
    assert ad.parse_roi("150,80,500,500", 100, 200) == (150, 80, 50, 20)  # clamped
    with pytest.raises(ValueError):
        ad.parse_roi("1,2,3", 100, 200)
    with pytest.raises(ValueError):
        ad.parse_roi("300,0,10,10", 100, 200)


def test_min_r_must_be_below_max_r():
    img, _ = _load(FIXTURES[0])
    with pytest.raises(ValueError):
        detect_arenas(img, n_expected=10, min_r=50, max_r=40)
