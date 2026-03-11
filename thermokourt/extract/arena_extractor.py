#!/usr/bin/env python3
"""
 ╔═══════════════════════════════════════════════════════════════════════╗
 ║  ░█▀█░█▀▄░█▀▀░█▀█░█▀█░░░█▀▀░█░█░▀█▀░█▀▄░█▀█░█▀▀░▀█▀░█▀█░█▀▄░░         ║
 ║  ░█▀█░█▀▄░█▀▀░█░█░█▀█░░░█▀▀░▄▀▄░░█░░█▀▄░█▀█░█░░░░█░░█░█░█▀▄░░         ║
 ║  ░▀░▀░▀░▀░▀▀▀░▀░▀░▀░▀░░░▀▀▀░▀░▀░░▀░░▀░▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀░▀░░         ║
 ║                                                                       ║
 ║   Detect, verify & extract circular arenas from Motif recordings      ║
 ║   ── detect. adjust. stitch. crop. science. ──                v1.1    ║
 ╚═══════════════════════════════════════════════════════════════════════╝

Pipeline for Drosophila arena recordings captured with Motif (FLIR/Basler).
Given a recording directory of sequential .mp4 chunks + metadata.yaml:

  1. Reads the first frame of the first chunk
  2. Auto-detects circular arenas via adaptive thresholding + contour fitting
  3. Opens an interactive matplotlib GUI for verification
  4. Concatenates all mp4 chunks via ffmpeg
  5. Crops each arena (with 10% padding) into individual videos

Usage:
    python arena_extractor.py /path/to/droso_18deg_20251002_102448
    python arena_extractor.py /path/to/recording --output_dir /path/to/output
    python arena_extractor.py /path/to/recording --arenas arenas.json
    python arena_extractor.py /path/to/recording --n_arenas 8

Requirements:
    pip install matplotlib numpy Pillow pyyaml opencv-python
    ffmpeg must be on PATH

License: MIT
Author: Bart R.H. Geurten
"""

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as MplCircle, Rectangle as MplRect
from PIL import Image

try:
    matplotlib.rcParams["toolbar"] = "None"
except TypeError:
    pass

# ┌─────────────────────────────────────────────────────────────────────┐
# │  CONSTANTS                                   « system defaults »    │
# └─────────────────────────────────────────────────────────────────────┘
PICK_RADIUS_PX = 18
CENTRE_SIZE = 90
RIM_HANDLE_SIZE = 60
COLOURS = [
    "#00ff88", "#ff4466", "#44aaff", "#ffaa00", "#aa44ff",
    "#ff66cc", "#00ddcc", "#ffdd44", "#88ff44", "#ff8844",
]
EDGE_ALPHA = 0.9
FILL_ALPHA = 0.18
LABEL_FONTSIZE = 10
DEFAULT_N_ARENAS = 10
CROP_PADDING = 0.10
RADIUS_TOLERANCE = 0.25
MIN_ARENAS_FOR_FILTER = 5


# ┌─────────────────────────────────────────────────────────────────────┐
# │  ARENA DATA                                   « circle geometry »   │
# └─────────────────────────────────────────────────────────────────────┘
@dataclass
class Arena:
    """A circular arena: centre (cx, cy) and radius r in pixel coordinates."""
    cx: float
    cy: float
    r: float
    idx: int = 0

    def bbox(self, padding: float = CROP_PADDING,
             frame_w: int = 0, frame_h: int = 0) -> Tuple[int, int, int, int]:
        """Return (x, y, w, h) bounding box with padding, clamped to frame."""
        padded_r = self.r * (1.0 + padding)
        x = int(self.cx - padded_r)
        y = int(self.cy - padded_r)
        w = int(2 * padded_r)
        h = int(2 * padded_r)
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0
        if frame_w > 0 and x + w > frame_w:
            w = frame_w - x
        if frame_h > 0 and y + h > frame_h:
            h = frame_h - y
        w = max(2, w - (w % 2))
        h = max(2, h - (h % 2))
        return x, y, w, h

    def to_dict(self) -> dict:
        return {"cx": self.cx, "cy": self.cy, "r": self.r, "idx": self.idx}

    @classmethod
    def from_dict(cls, d: dict) -> "Arena":
        return cls(cx=d["cx"], cy=d["cy"], r=d["r"], idx=d.get("idx", 0))


def sort_arenas_row_major(arenas: List[Arena]) -> List[Arena]:
    """Sort top-to-bottom, left-to-right. Always re-numbers from 0."""
    if not arenas:
        return arenas
    arenas_copy = list(arenas)
    arenas_copy.sort(key=lambda a: a.cy)
    median_r = float(np.median([a.r for a in arenas_copy]))
    rows: List[List[Arena]] = []
    current_row: List[Arena] = [arenas_copy[0]]
    for a in arenas_copy[1:]:
        if abs(a.cy - current_row[-1].cy) > median_r:
            rows.append(current_row)
            current_row = [a]
        else:
            current_row.append(a)
    rows.append(current_row)
    sorted_arenas: List[Arena] = []
    idx = 0
    for row in rows:
        row.sort(key=lambda a: a.cx)
        for a in row:
            a.idx = idx
            sorted_arenas.append(a)
            idx += 1
    return sorted_arenas


def filter_by_radius(arenas: List[Arena],
                     tolerance: float = RADIUS_TOLERANCE,
                     min_keep: int = MIN_ARENAS_FOR_FILTER) -> List[Arena]:
    """Remove arenas with radius outside ±tolerance of median.
    Skips filtering if result would have fewer than min_keep arenas."""
    if len(arenas) < 3:
        return sort_arenas_row_major(arenas)
    radii = [a.r for a in arenas]
    median_r = float(np.median(radii))
    lo = median_r * (1.0 - tolerance)
    hi = median_r * (1.0 + tolerance)
    filtered = [a for a in arenas if lo <= a.r <= hi]
    if len(filtered) < min_keep:
        return sort_arenas_row_major(arenas)
    return sort_arenas_row_major(filtered)


# ┌─────────────────────────────────────────────────────────────────────┐
# │  FRAME EXTRACTION                                                   │
# └─────────────────────────────────────────────────────────────────────┘
def get_mp4_chunks(recording_dir: str) -> List[str]:
    chunks = sorted(glob.glob(os.path.join(recording_dir, "*.mp4")))
    if not chunks:
        raise FileNotFoundError(f"No .mp4 files found in {recording_dir}")
    return chunks


def extract_first_frame(mp4_path: str) -> np.ndarray:
    """Extract first frame via ffmpeg. Returns RGB (H, W, 3) uint8."""
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", mp4_path,
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {mp4_path}: {probe.stderr}")
    info = json.loads(probe.stdout)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    w, h = int(vstream["width"]), int(vstream["height"])
    cmd = [
        "ffmpeg", "-i", mp4_path, "-frames:v", "1",
        "-f", "image2pipe", "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo", "-loglevel", "error", "-"
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape((h, w, 3))


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CIRCLE DETECTION                                                   │
# │  Primary: adaptive threshold + contour + minEnclosingCircle (cv2)   │
# │  Fallback: Hough (cv2 or scipy)                                     │
# └─────────────────────────────────────────────────────────────────────┘
def detect_arenas(
    frame: np.ndarray,
    n_expected: int = DEFAULT_N_ARENAS,
    min_radius: int = 0,
    max_radius: int = 0,
) -> List[Arena]:
    """Detect circular arenas. Tries contour method first, then Hough."""
    try:
        import cv2
        arenas = _detect_contour_cv2(frame, n_expected, min_radius, max_radius)
        if len(arenas) >= max(n_expected // 2, 3):
            return filter_by_radius(arenas)
        print(f"  Contour method found {len(arenas)}, trying Hough fallback...")
        arenas_h = _detect_hough_cv2(frame, n_expected, min_radius, max_radius)
        best = arenas_h if len(arenas_h) > len(arenas) else arenas
        return filter_by_radius(best)
    except ImportError:
        print("[arena_extractor] OpenCV not found, using scipy fallback.")
        return filter_by_radius(
            _detect_hough_scipy(frame, n_expected, min_radius, max_radius)
        )


def _detect_contour_cv2(frame, n_expected, min_radius, max_radius):
    """Adaptive threshold + contour fitting — robust for bright-on-dark arenas."""
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame.copy()
    h, w = gray.shape[:2]
    if min_radius == 0:
        min_radius = int(min(h, w) * 0.04)
    if max_radius == 0:
        max_radius = int(min(h, w) * 0.20)
    min_area = math.pi * min_radius ** 2 * 0.5
    max_area = math.pi * max_radius ** 2 * 1.5

    # CLAHE + blur
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 1.5)

    # Adaptive threshold
    block_size = max_radius * 4 + 1
    if block_size % 2 == 0:
        block_size += 1
    block_size = min(block_size, min(h, w) - 2)
    if block_size % 2 == 0:
        block_size -= 1
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, -5,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def _extract_circles(cnts, existing=None):
        existing = existing or []
        results = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            circ = 4.0 * math.pi * area / (peri * peri)
            if circ < 0.65:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < min_radius or radius > max_radius:
                continue
            fill = area / (math.pi * radius * radius) if radius > 0 else 0
            if fill < 0.55:
                continue
            # Check for duplicates
            if any(np.hypot(cx - e.cx, cy - e.cy) < min_radius for e in existing + results):
                continue
            results.append(Arena(cx=float(cx), cy=float(cy), r=float(radius)))
        return results

    candidates = _extract_circles(contours)

    # Otsu fallback if too few
    if len(candidates) < n_expected // 2:
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours2, _ = cv2.findContours(otsu_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates.extend(_extract_circles(contours2, existing=candidates))

    return sort_arenas_row_major(candidates)


def _detect_hough_cv2(frame, n_expected, min_radius, max_radius):
    """Hough circle detection fallback."""
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame.copy()
    h, w = gray.shape[:2]
    if min_radius == 0:
        min_radius = int(min(h, w) * 0.04)
    if max_radius == 0:
        max_radius = int(min(h, w) * 0.20)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    blurred = cv2.GaussianBlur(clahe.apply(gray), (9, 9), 2)
    best = None
    for p2 in range(80, 8, -3):
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_radius * 2,
            param1=100, param2=p2, minRadius=min_radius, maxRadius=max_radius,
        )
        if circles is not None:
            if best is None or circles.shape[1] > best.shape[1]:
                best = circles
            if circles.shape[1] >= n_expected:
                break
    arenas = []
    if best is not None:
        for c in best[0]:
            arenas.append(Arena(cx=float(c[0]), cy=float(c[1]), r=float(c[2])))
    return sort_arenas_row_major(arenas)


def _detect_hough_scipy(frame, n_expected, min_radius, max_radius):
    """Scipy/skimage fallback (no OpenCV)."""
    try:
        from skimage.transform import hough_circle, hough_circle_peaks
        from skimage.feature import canny
        from skimage.color import rgb2gray
    except ImportError:
        warnings.warn("Neither OpenCV nor scikit-image found.")
        return []
    gray = rgb2gray(frame) if len(frame.shape) == 3 else frame.astype(float) / 255.0
    h, w = gray.shape[:2]
    if min_radius == 0:
        min_radius = int(min(h, w) * 0.04)
    if max_radius == 0:
        max_radius = int(min(h, w) * 0.20)
    edges = canny(gray, sigma=2.0)
    radii = np.arange(min_radius, max_radius, 3)
    hough_res = hough_circle(edges, radii)
    _, cx_arr, cy_arr, r_arr = hough_circle_peaks(
        hough_res, radii, min_xdistance=min_radius * 2,
        min_ydistance=min_radius * 2, total_num_peaks=n_expected + 5,
    )
    arenas = [Arena(cx=float(cx), cy=float(cy), r=float(r))
              for cx, cy, r in zip(cx_arr, cy_arr, r_arr)]
    return sort_arenas_row_major(arenas[:n_expected + 5])


# ┌─────────────────────────────────────────────────────────────────────┐
# │  MULTI-CIRCLE ANNOTATOR GUI                                         │
# │  Performance: background image rendered ONCE, overlay artists       │
# │  redrawn via matplotlib blitting during drag operations.            │
# └─────────────────────────────────────────────────────────────────────┘
class MultiCircleEditor:
    """Interactive matplotlib GUI for editing multiple circular arenas.

    Controls:
      Drag centre (+)       Move arena
      Drag rim handle (diamond)  Resize arena
      Right-click empty     Add new arena
      Right-click centre    Delete arena
      A   Re-run detection   +/-  Adjust radii   U  Uniform radii
      L   Labels   F  Fill   H  Help   S  Save   Q/Enter  Accept
    """

    WINDOW_NAME = "Arena Extractor"

    def __init__(self, frame, arenas, recording_name="", json_path=""):
        self.frame = frame
        self.arenas = sort_arenas_row_major(arenas)
        self.recording_name = recording_name
        self.json_path = json_path
        self._dragging = None
        self._drag_mode = ""
        self._drag_offset = (0.0, 0.0)
        self.show_labels = True
        self.show_fill = True
        self.show_help = False
        self.accepted = False
        self._background = None
        self._overlay_artists = []

        h, w = frame.shape[:2]
        dpi = 100
        self.fig, self.ax = plt.subplots(
            1, 1, figsize=(min(w / dpi, 18), min(h / dpi, 12)), dpi=dpi
        )
        self.fig.canvas.manager.set_window_title(self.WINDOW_NAME)
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01)

        if len(frame.shape) == 2:
            self.ax.imshow(frame, cmap="gray", aspect="equal")
        else:
            self.ax.imshow(frame, aspect="equal")
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.ax.set_axis_off()

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)

        self._render_overlay()
        self.fig.canvas.draw()
        plt.show()

    @staticmethod
    def _colour(idx):
        return COLOURS[idx % len(COLOURS)]

    def _on_draw(self, event):
        self._background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def _clear_overlay(self):
        for a in self._overlay_artists:
            try:
                a.remove()
            except ValueError:
                pass
        self._overlay_artists.clear()

    def _render_overlay(self):
        self._clear_overlay()
        h, w = self.frame.shape[:2]
        self.arenas = sort_arenas_row_major(self.arenas)

        for a in self.arenas:
            col = self._colour(a.idx)
            cp = MplCircle(
                (a.cx, a.cy), a.r,
                fill=self.show_fill,
                facecolor=col if self.show_fill else "none",
                edgecolor=col,
                alpha=FILL_ALPHA if self.show_fill else EDGE_ALPHA,
                linewidth=2, animated=True,
            )
            self.ax.add_patch(cp)
            self._overlay_artists.append(cp)

            bx, by, bw, bh = a.bbox(padding=CROP_PADDING, frame_w=w, frame_h=h)
            rect = MplRect(
                (bx, by), bw, bh, fill=False, edgecolor=col,
                linewidth=1, linestyle="--", alpha=0.5, animated=True,
            )
            self.ax.add_patch(rect)
            self._overlay_artists.append(rect)

            sc = self.ax.scatter(
                [a.cx], [a.cy], s=CENTRE_SIZE, c=[col],
                edgecolors="white", linewidth=1.5, zorder=5,
                marker="+", animated=True,
            )
            self._overlay_artists.append(sc)

            rim_x = a.cx + a.r
            sr = self.ax.scatter(
                [rim_x], [a.cy], s=RIM_HANDLE_SIZE, c=[col],
                edgecolors="white", linewidth=1.5, zorder=6,
                marker="D", animated=True,
            )
            self._overlay_artists.append(sr)

            if self.show_labels:
                txt = self.ax.annotate(
                    f"#{a.idx:02d}", (a.cx, a.cy - a.r - 8),
                    fontsize=LABEL_FONTSIZE, fontweight="bold",
                    color="white", ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=col, alpha=0.85),
                    zorder=7, animated=True,
                )
                self._overlay_artists.append(txt)

        if self.show_help:
            txt = self.ax.text(
                0.02, 0.98,
                "=== ARENA EXTRACTOR HELP ===\n\n"
                "MOUSE\n"
                "  Drag centre (+)      Move arena\n"
                "  Drag rim handle      Resize arena\n"
                "  Right-click empty    Add new arena\n"
                "  Right-click centre   Delete arena\n\n"
                "KEYS\n"
                "  A   Re-run detection    +/- Adjust radii\n"
                "  U   Uniform radii       L   Toggle labels\n"
                "  F   Toggle fill         H   Toggle help\n"
                "  S   Save positions      Q   Accept & extract\n\n"
                f"Arenas: {len(self.arenas)}   Frame: {w} x {h} px",
                transform=self.ax.transAxes, fontsize=10,
                fontfamily="monospace", verticalalignment="top",
                color="white", zorder=10, animated=True,
                bbox=dict(boxstyle="round,pad=0.8", facecolor="black",
                          alpha=0.88, edgecolor="#00ff88", linewidth=1.5),
            )
            self._overlay_artists.append(txt)

        status = (f"{self.recording_name}  |  {len(self.arenas)} arenas  |  "
                  f"drag=move/resize  right-click=add/delete  Q=accept")
        self.ax.set_title(status, fontsize=9, loc="left", pad=6)

    def _blit_overlay(self):
        if self._background is None:
            self._render_overlay()
            self.fig.canvas.draw()
            return
        self.fig.canvas.restore_region(self._background)
        for a in self._overlay_artists:
            self.ax.draw_artist(a)
        self.fig.canvas.blit(self.ax.bbox)

    def _full_redraw(self):
        self._render_overlay()
        self.fig.canvas.draw()

    def _hit_test(self, event):
        if event.xdata is None:
            return None, ""
        for a in reversed(self.arenas):
            rim_x = a.cx + a.r
            dr = self.ax.transData.transform((rim_x, a.cy))
            if np.hypot(dr[0] - event.x, dr[1] - event.y) < PICK_RADIUS_PX:
                return a, "resize"
            dc = self.ax.transData.transform((a.cx, a.cy))
            if np.hypot(dc[0] - event.x, dc[1] - event.y) < PICK_RADIUS_PX:
                return a, "move"
        return None, ""

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if event.button == 1:
            arena, mode = self._hit_test(event)
            if arena:
                self._dragging = arena
                self._drag_mode = mode
                self._drag_offset = (event.xdata - arena.cx, event.ydata - arena.cy)
        elif event.button == 3:
            arena, mode = self._hit_test(event)
            if arena and mode == "move":
                self.arenas.remove(arena)
                self._full_redraw()
            else:
                mr = float(np.median([a.r for a in self.arenas])) if self.arenas else 100.0
                self.arenas.append(Arena(cx=event.xdata, cy=event.ydata, r=mr))
                self._full_redraw()

    def _on_release(self, event):
        if self._dragging is not None:
            self._dragging = None
            self._drag_mode = ""
            self._full_redraw()

    def _on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self._dragging is not None:
            a = self._dragging
            h, w = self.frame.shape[:2]
            if self._drag_mode == "move":
                a.cx = float(np.clip(event.xdata - self._drag_offset[0], 0, w))
                a.cy = float(np.clip(event.ydata - self._drag_offset[1], 0, h))
            elif self._drag_mode == "resize":
                a.r = max(20.0, float(np.hypot(event.xdata - a.cx, event.ydata - a.cy)))
            self._render_overlay()
            self._blit_overlay()

    def _on_key(self, event):
        key = event.key
        if key in ("q", "enter"):
            self.accepted = True
            plt.close(self.fig)
        elif key == "escape":
            self.accepted = False
            plt.close(self.fig)
        elif key == "a":
            self.arenas = detect_arenas(self.frame)
            print(f"[arena_extractor] Re-detected {len(self.arenas)} arenas")
            self._full_redraw()
        elif key in ("+", "="):
            for a in self.arenas: a.r += 5
            self._full_redraw()
        elif key == "-":
            for a in self.arenas: a.r = max(20, a.r - 5)
            self._full_redraw()
        elif key == "u":
            if self.arenas:
                mr = float(np.median([a.r for a in self.arenas]))
                for a in self.arenas: a.r = mr
                self._full_redraw()
        elif key == "l":
            self.show_labels = not self.show_labels
            self._full_redraw()
        elif key == "f":
            self.show_fill = not self.show_fill
            self._full_redraw()
        elif key == "h":
            self.show_help = not self.show_help
            self._full_redraw()
        elif key == "s":
            self._save_json()

    def _save_json(self):
        path = self.json_path or f"{self.recording_name}_arenas.json"
        data = {
            "recording": self.recording_name,
            "frame_width": self.frame.shape[1],
            "frame_height": self.frame.shape[0],
            "arenas": [a.to_dict() for a in sort_arenas_row_major(self.arenas)],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[arena_extractor] Saved {len(self.arenas)} arenas -> {path}")

    def get_arenas(self):
        if self.accepted:
            return sort_arenas_row_major(self.arenas)
        return None


# ┌─────────────────────────────────────────────────────────────────────┐
# │  FFMPEG CONCAT + CROP                                               │
# └─────────────────────────────────────────────────────────────────────┘
def build_concat_file(chunks, tmpdir):
    path = os.path.join(tmpdir, "concat.txt")
    with open(path, "w") as f:
        for chunk in chunks:
            f.write(f"file '{os.path.abspath(chunk)}'\n")
    return path


def concat_and_crop(chunks, arenas, output_dir, recording_name,
                    codec="libx264", crf=18, frame_height=0, frame_width=0,
                    keep_full=False):
    """Concatenate chunks and crop each arena with padding."""
    os.makedirs(output_dir, exist_ok=True)
    outputs = []
    with tempfile.TemporaryDirectory(prefix="arena_extract_") as tmpdir:
        concat_file = build_concat_file(chunks, tmpdir)
        if keep_full:
            full_path = os.path.join(output_dir, f"{recording_name}_full.mp4")
            print(f"  Concatenating full video -> {full_path}")
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_file, "-c", "copy", full_path,
            ], check=True, capture_output=True)
            outputs.append(full_path)

        n = len(arenas)
        for i, arena in enumerate(arenas):
            x, y, w, h = arena.bbox(
                padding=CROP_PADDING, frame_w=frame_width, frame_h=frame_height,
            )
            out_path = os.path.join(
                output_dir, f"{recording_name}_arena_{arena.idx:02d}.mp4"
            )
            print(f"  [{i+1}/{n}] Arena #{arena.idx:02d}  "
                  f"crop={w}x{h}+{x}+{y} (r={arena.r:.0f}, pad={CROP_PADDING:.0%})  "
                  f"-> {os.path.basename(out_path)}")
            result = subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
                "-vf", f"crop={w}:{h}:{x}:{y}",
                "-c:v", codec, "-crf", str(crf), "-preset", "fast",
                "-pix_fmt", "yuv420p", "-an", out_path,
            ], capture_output=True)
            if result.returncode != 0:
                print(f"    Warning: ffmpeg error: {result.stderr.decode()[-200:]}")
            else:
                outputs.append(out_path)
    return outputs


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CLI + MAIN                                                         │
# └─────────────────────────────────────────────────────────────────────┘
def _die(msg):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        prog="arena_extractor",
        description="Arena Extractor v1.1 — detect, verify & crop circular arenas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s /data/droso_18deg_20251002_102448\n"
            "  %(prog)s /data/recording -o /results/\n"
            "  %(prog)s /data/recording --arenas arenas.json\n"
            "  %(prog)s /data/recording -n 8 --padding 0.15\n"
        ),
    )
    p.add_argument("recording_dir", help="Motif recording directory.")
    p.add_argument("-o", "--output_dir", default=None)
    p.add_argument("-n", "--n_arenas", type=int, default=DEFAULT_N_ARENAS)
    p.add_argument("--arenas", default=None, help="JSON with arena positions (skip GUI).")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--codec", default="libx264")
    p.add_argument("--padding", type=float, default=CROP_PADDING,
                   help=f"Fractional padding around arena (default: {CROP_PADDING}).")
    p.add_argument("--keep_full", action="store_true")
    p.add_argument("--no_gui", action="store_true")
    return p.parse_args()


def load_arenas_json(path):
    with open(path) as f:
        data = json.load(f)
    ad = data if isinstance(data, list) else data.get("arenas", data)
    if isinstance(ad, list):
        return [Arena.from_dict(d) for d in ad]
    _die(f"Could not parse arenas from {path}")


def main():
    args = parse_args()
    global CROP_PADDING
    CROP_PADDING = args.padding

    recording_dir = args.recording_dir.rstrip("/")
    recording_name = os.path.basename(recording_dir)
    output_dir = args.output_dir or recording_dir

    print("+" + "-" * 44 + "+")
    print("|  ARENA EXTRACTOR v1.1                      |")
    print("|  detect. adjust. stitch. crop.             |")
    print("+" + "-" * 44 + "+")

    chunks = get_mp4_chunks(recording_dir)
    print(f"  Recording: {recording_name}")
    print(f"  Chunks:    {len(chunks)} .mp4 files")
    print(f"  Output:    {output_dir}")
    print(f"  Padding:   {CROP_PADDING:.0%}")

    if args.arenas:
        arenas = load_arenas_json(args.arenas)
        print(f"  Loaded {len(arenas)} arenas from {args.arenas}")
        frame = extract_first_frame(chunks[0])
        frame_h, frame_w = frame.shape[:2]
    else:
        print("  Extracting first frame...")
        frame = extract_first_frame(chunks[0])
        frame_h, frame_w = frame.shape[:2]
        print(f"  Frame size: {frame_w} x {frame_h}")
        print(f"  Running circle detection (expecting ~{args.n_arenas})...")
        arenas = detect_arenas(frame, n_expected=args.n_arenas)
        print(f"  Detected {len(arenas)} arenas (after radius filtering)")

        if not args.no_gui:
            json_path = os.path.join(output_dir, f"{recording_name}_arenas.json")
            editor = MultiCircleEditor(
                frame, arenas, recording_name=recording_name, json_path=json_path,
            )
            arenas = editor.get_arenas()
            if arenas is None:
                print("\n  Aborted by user.")
                sys.exit(0)

    if not arenas:
        _die("No arenas defined.")

    arenas = sort_arenas_row_major(arenas)
    json_out = os.path.join(output_dir, f"{recording_name}_arenas.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump({
            "recording": recording_name,
            "frame_width": frame_w, "frame_height": frame_h,
            "padding": CROP_PADDING,
            "arenas": [a.to_dict() for a in arenas],
        }, f, indent=2)
    print(f"\n  Arena positions saved -> {json_out}")

    print(f"\n  Starting extraction ({len(arenas)} arenas x {len(chunks)} chunks)...")
    outputs = concat_and_crop(
        chunks=chunks, arenas=arenas, output_dir=output_dir,
        recording_name=recording_name, codec=args.codec, crf=args.crf,
        frame_height=frame_h, frame_width=frame_w, keep_full=args.keep_full,
    )
    print(f"\n  Done! {len(outputs)} files written to {output_dir}")
    for p in outputs:
        print(f"    -> {os.path.basename(p)}")


if __name__ == "__main__":
    main()