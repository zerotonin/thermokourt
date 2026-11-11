"""Arena circle detection on a background (max-projection) frame.

Three complementary detectors feed one consensus step:

* ``hough_circles``     – edge-based (OpenCV HoughCircles). Works when arena
  rims are crisp against a plain background (e.g. backlit arenas that fill
  the frame).
* ``blob_circles``      – intensity-based. Finds round patches that are
  brighter *or* darker than their surroundings after large-scale background
  subtraction. Works when arenas are bright holes in a dimmer plate, and
  gives a reliable radius estimate.
* ``template_circles``  – normalised cross-correlation with a disc of the
  estimated radius. Given a radius it ranks every disc-like spot in the
  image; the top ``n_expected`` are the arenas when the field of view is
  restricted to the plate (``roi``).

``detect_arenas`` combines them: blob/Hough give the radius estimate,
template matching gives the final ranked positions, and a radius-consistency
filter removes outliers. Everything runs on a downscaled copy (``work``
pixels on the long side), so a 2600×2160 frame takes well under a second.

Coordinates returned are always in *full-frame* pixels, even when an ``roi``
was used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

WORK_SIZE = 800  # long-side pixels used for detection
RADIUS_TOL = 0.25  # accept radii within ±25 % of the median
BLOB_R_PX = 40  # blob detection works at ~40 px arena radius
HULL_DILATE_R = 3.0  # template search area: seed hull + 3 radii


@dataclass
class Circle:
    cx: float
    cy: float
    r: float
    score: float = 0.0  # template score (0–1) or 0 for Hough/blob-only
    source: str = ""


# ── helpers ────────────────────────────────────────────────────────────


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return frame


def _downscale(gray: np.ndarray, work: int) -> tuple[np.ndarray, float]:
    h, w = gray.shape
    s = min(1.0, work / max(h, w))
    if s < 1.0:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return gray, s


def default_radius_bounds(h: int, w: int) -> tuple[int, int]:
    """Radius window as a fraction of the short side (3 %–20 %)."""
    m = min(h, w)
    return int(m * 0.03), int(m * 0.20)


def parse_roi(text: str | None, h: int, w: int) -> tuple[int, int, int, int]:
    """Parse ``x,y,w,h`` (pixels) into a clamped tuple; ``None`` → whole frame."""
    if not text:
        return 0, 0, w, h
    try:
        x, y, rw, rh = (int(float(v)) for v in text.split(","))
    except ValueError as e:
        raise ValueError(f"ROI must be 'x,y,w,h' in pixels, got {text!r}") from e
    x, y = max(0, x), max(0, y)
    rw, rh = min(rw, w - x), min(rh, h - y)
    if rw <= 0 or rh <= 0:
        raise ValueError(f"ROI {text!r} lies outside the {w}x{h} frame")
    return x, y, rw, rh


def radius_windows(min_r: int, max_r: int, step: float = 1.6) -> list[tuple[int, int]]:
    """Split ``[min_r, max_r]`` into overlapping windows of ratio ``step``.

    Blob detection needs a narrow radius window (its background blur and
    morphology scale with the upper bound), so a wide user range is scanned
    in octaves: (64, 102), (102, 164), ... each one cheap.
    """
    out, lo = [], float(min_r)
    while lo < max_r:
        hi = min(float(max_r), lo * step)
        out.append((int(lo), int(math.ceil(hi))))
        if hi >= max_r:
            break
        lo = hi / 1.15  # small overlap so a radius on the boundary is not missed
    return out


def _median_radius(circles: list[Circle]) -> float:
    return float(np.median([c.r for c in circles])) if circles else 0.0


def filter_radius(circles: list[Circle], tol: float = RADIUS_TOL, keep: int = 3) -> list[Circle]:
    """Drop circles whose radius is far from the median (if enough remain)."""
    if len(circles) < keep:
        return circles
    med = _median_radius(circles)
    f = [c for c in circles if med * (1 - tol) <= c.r <= med * (1 + tol)]
    return f if len(f) >= keep else circles


# ── detectors ──────────────────────────────────────────────────────────


def hough_circles(
    gray: np.ndarray, n_expected: int, min_r: int, max_r: int, work: int = WORK_SIZE
) -> list[Circle]:
    """Edge-based circle detection; sweeps the accumulator threshold down
    until ``n_expected`` circles appear (or it bottoms out)."""
    g, s = _downscale(gray, work)
    mn, mx = max(3, int(min_r * s)), max(4, int(max_r * s))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    blurred = cv2.GaussianBlur(clahe.apply(g), (7, 7), 1.5)
    best = None
    for p2 in range(60, 8, -3):
        c = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=mn * 2,
            param1=100, param2=p2, minRadius=mn, maxRadius=mx,
        )
        if c is not None:
            if best is None or c.shape[1] > best.shape[1]:
                best = c
            if c.shape[1] >= n_expected:
                break
    if best is None:
        return []
    return [Circle(float(x) / s, float(y) / s, float(r) / s, 0.0, "hough") for x, y, r in best[0]]


def blob_circles(
    gray: np.ndarray, min_r: int, max_r: int, polarity: str, work: int = WORK_SIZE
) -> list[Circle]:
    """Round patches brighter (``polarity='bright'``) or darker (``'dark'``)
    than the large-scale background.

    The image is downscaled so that ``max_r`` is about ``BLOB_R_PX`` pixels,
    which keeps the background blur cheap whatever the arena size."""
    work = min(work, int(max(gray.shape) * BLOB_R_PX / max(1, max_r)))
    g, s = _downscale(gray, max(64, work))
    mn, mx = min_r * s, max_r * s
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    if polarity == "dark":
        g = 255 - g
    bg = cv2.GaussianBlur(g, (0, 0), max(1.0, mx * 1.5))
    d = cv2.subtract(g, bg)
    _, th = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, int(mn / 3) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, ker)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, ker)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a, p = cv2.contourArea(c), cv2.arcLength(c, True)
        if p == 0:
            continue
        circ = 4 * math.pi * a / (p * p)
        (cx, cy), r = cv2.minEnclosingCircle(c)
        fill = a / (math.pi * r * r) if r > 0 else 0
        if mn <= r <= mx and circ > 0.6 and fill > 0.6:
            out.append(Circle(cx / s, cy / s, r / s, 0.0, f"blob-{polarity}"))
    return out


def template_circles(
    gray: np.ndarray,
    r: float,
    polarity: str,
    n_max: int,
    work: int = WORK_SIZE,
    mask: np.ndarray | None = None,
) -> list[Circle]:
    """Rank disc-shaped spots of radius ``r`` by normalised cross-correlation.

    Returns up to ``n_max`` circles, best first, non-overlapping (centres at
    least 1.5 r apart). ``mask`` (same shape as ``gray``, non-zero = search
    here) restricts where centres may lie."""
    g, s = _downscale(gray, work)
    g = g.astype(np.float32)
    if polarity == "dark":
        g = 255 - g
    rs = max(3, int(round(r * s)))
    pad = rs + 2
    t = np.zeros((2 * pad + 1, 2 * pad + 1), np.float32)
    cv2.circle(t, (pad, pad), rs, 1.0, -1)
    t = cv2.GaussianBlur(t, (0, 0), 1.0)
    if g.shape[0] <= t.shape[0] or g.shape[1] <= t.shape[1]:
        return []
    res = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
    if mask is not None:
        m = cv2.resize(mask, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = m[pad:pad + res.shape[0], pad:pad + res.shape[1]]
        res[m == 0] = -1.0
    out: list[Circle] = []
    for _ in range(n_max):
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx <= 0:
            break
        out.append(Circle((loc[0] + pad) / s, (loc[1] + pad) / s, r, float(mx), "template"))
        cv2.circle(res, loc, int(rs * 1.5), -1.0, -1)
    return out


# ── consensus ──────────────────────────────────────────────────────────


def seed_hull_mask(shape: tuple[int, int], seeds: list[Circle]) -> np.ndarray | None:
    """Mask covering the convex hull of the seeds, grown by ``HULL_DILATE_R``
    radii. Arenas sit on one plate, so the missing ones are near the found
    ones; everything else (lamps, screws, lids) is excluded."""
    if len(seeds) < 3:
        return None
    r = _median_radius(seeds)
    pts = np.array([[c.cx, c.cy] for c in seeds], np.float32)
    hull = cv2.convexHull(pts).astype(np.int32)
    hull_img = np.full(shape, 255, np.uint8)
    cv2.fillConvexPoly(hull_img, hull, 0)
    # distance from the hull, in pixels; cheap even for large radii
    dist = cv2.distanceTransform(hull_img, cv2.DIST_L2, 5)
    return (dist <= HULL_DILATE_R * r).astype(np.uint8) * 255


def _polarity_of(gray: np.ndarray, circles: list[Circle]) -> str:
    """Are the circle interiors brighter or darker than the rest of the image?"""
    mask = np.zeros(gray.shape, np.uint8)
    for c in circles:
        cv2.circle(mask, (int(c.cx), int(c.cy)), int(c.r * 0.8), 255, -1)
    inside = float(np.mean(gray[mask > 0])) if mask.any() else 0.0
    outside = float(np.mean(gray[mask == 0])) if (mask == 0).any() else inside
    return "bright" if inside >= outside else "dark"


def detect_arenas(
    frame: np.ndarray,
    n_expected: int = 10,
    min_r: int = 0,
    max_r: int = 0,
    roi: tuple[int, int, int, int] | None = None,
    polarity: str = "auto",
    work: int = WORK_SIZE,
) -> list[Circle]:
    """Find up to ``n_expected`` circular arenas in a background frame.

    Args:
        frame: RGB or grey background image (ideally a max-projection).
        n_expected: number of arenas on the plate.
        min_r, max_r: radius window in full-frame pixels (0 = automatic,
            3–20 % of the short side).
        roi: ``(x, y, w, h)`` in full-frame pixels restricting the search to
            the arena plate. Strongly recommended when lamps, lids or other
            round objects are visible.
        polarity: ``'bright'`` if arenas are brighter than the plate,
            ``'dark'`` if darker, ``'auto'`` to decide from the data.
        work: long-side size (pixels) of the downscaled working image.

    Returns:
        Circles in full-frame coordinates, unsorted. Each carries a
        ``score`` (template correlation, 0–1) and ``source``.
    """
    gray = _to_gray(frame)
    H, W = gray.shape
    x0, y0, rw, rh = roi if roi else (0, 0, W, H)
    g = gray[y0:y0 + rh, x0:x0 + rw]
    if min_r <= 0 or max_r <= 0:
        # Bounds come from the *frame*, not the ROI: a tight ROI around the
        # plate must not shrink the window onto screw holes and grid cells.
        a_min, a_max = default_radius_bounds(H, W)
        min_r = min_r or a_min
        max_r = max_r or a_max
    if min_r >= max_r:
        raise ValueError(f"min_r ({min_r}) must be smaller than max_r ({max_r})")

    # 1. seeds: blob over a few radius octaves (both polarities unless
    #    given), Hough as fallback
    seeds: list[Circle] = []
    pols = ("bright", "dark") if polarity == "auto" else (polarity,)
    for lo, hi in radius_windows(min_r, max_r):
        for pol in pols:
            c = filter_radius(blob_circles(g, lo, hi, pol, work))
            if len(c) > len(seeds):
                seeds = c
    # Blob seeds are specific (round *and* contrasting); Hough fires on any
    # edge arc, so it is only the fallback when blobs find too little.
    if len(seeds) < 3:
        hough = filter_radius(hough_circles(g, n_expected, min_r, max_r, work))
        if len(hough) > len(seeds):
            seeds = hough

    if not seeds:
        return []

    # 2. radius estimate + polarity, then template ranking
    r_est = _median_radius(seeds)
    pol = polarity if polarity != "auto" else _polarity_of(g, seeds)
    ranked = template_circles(g, r_est, pol, n_expected, work, mask=seed_hull_mask(g.shape, seeds))

    # 3. prefer template positions; keep seed radius where a seed matches
    out: list[Circle] = []
    for t in ranked:
        near = [s for s in seeds if math.hypot(s.cx - t.cx, s.cy - t.cy) < r_est * 0.5]
        r = float(np.median([s.r for s in near])) if near else r_est
        out.append(Circle(t.cx, t.cy, r, t.score, "template+seed" if near else "template"))
    if len(out) < len(seeds):
        out = seeds  # template failed (tiny image); fall back to seeds
    out = filter_radius(out)

    # back to full-frame coordinates
    for c in out:
        c.cx += x0
        c.cy += y0
    return out
