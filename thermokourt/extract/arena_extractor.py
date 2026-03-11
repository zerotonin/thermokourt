#!/usr/bin/env python3
"""
 ╔═══════════════════════════════════════════════════════════════════════╗
 ║  ░█▀█░█▀▄░█▀▀░█▀█░█▀█░░░█▀▀░█░█░▀█▀░█▀▄░█▀█░█▀▀░▀█▀░█▀█░█▀▄░░   ║
 ║  ░█▀█░█▀▄░█▀▀░█░█░█▀█░░░█▀▀░▄▀▄░░█░░█▀▄░█▀█░█░░░░█░░█░█░█▀▄░░   ║
 ║  ░▀░▀░▀░▀░▀▀▀░▀░▀░▀░▀░░░▀▀▀░▀░▀░░▀░░▀░▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀░▀░░   ║
 ║                                                                      ║
 ║   Detect, verify & extract circular arenas from Motif recordings     ║
 ║   ── detect. adjust. stitch. crop. science. ──                v1.0   ║
 ╚═══════════════════════════════════════════════════════════════════════╝

Pipeline for Drosophila arena recordings captured with Motif (FLIR/Basler).
Given a recording directory of sequential .mp4 chunks + metadata.yaml:

  1. Reads the first frame of the first chunk
  2. Auto-detects circular arenas via Hough Circle Transform
  3. Opens an interactive matplotlib GUI (reusing circle_annotator patterns)
     for the user to verify, move, resize, add, or delete circles
  4. Concatenates all mp4 chunks via ffmpeg
  5. Crops each arena into individual videos:
       <dirname>_arena_00.mp4  …  <dirname>_arena_09.mp4

Usage:
    python arena_extractor.py /path/to/droso_18deg_20251002_102448
    python arena_extractor.py /path/to/recording --output_dir /path/to/output
    python arena_extractor.py /path/to/recording --arenas arenas.json  # skip GUI
    python arena_extractor.py /path/to/recording --n_arenas 8          # expect 8

Requirements:
    pip install matplotlib numpy Pillow pyyaml
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as MplCircle
from PIL import Image

try:
    matplotlib.rcParams["toolbar"] = "None"
except TypeError:
    pass

# ┌─────────────────────────────────────────────────────────────────────┐
# │  CONSTANTS                                   « system defaults »    │
# └─────────────────────────────────────────────────────────────────────┘
PICK_RADIUS_PX = 18          # display-space pixels for hit-testing handles
CENTRE_SIZE = 90              # scatter marker size for centre dots
RIM_HANDLE_SIZE = 60          # scatter marker size for rim diamonds
COLOURS = [                   # per-arena colours, cycled
    "#00ff88", "#ff4466", "#44aaff", "#ffaa00", "#aa44ff",
    "#ff66cc", "#00ddcc", "#ffdd44", "#88ff44", "#ff8844",
]
EDGE_ALPHA = 0.9
FILL_ALPHA = 0.18
LABEL_FONTSIZE = 10
DEFAULT_N_ARENAS = 10


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

    def bbox(self) -> Tuple[int, int, int, int]:
        """Return (x, y, w, h) bounding box for crop, clamped to >= 0."""
        x = max(0, int(self.cx - self.r))
        y = max(0, int(self.cy - self.r))
        w = int(2 * self.r)
        h = int(2 * self.r)
        return x, y, w, h

    def to_dict(self) -> dict:
        return {"cx": self.cx, "cy": self.cy, "r": self.r, "idx": self.idx}

    @classmethod
    def from_dict(cls, d: dict) -> "Arena":
        return cls(cx=d["cx"], cy=d["cy"], r=d["r"], idx=d.get("idx", 0))


def sort_arenas_row_major(arenas: List[Arena]) -> List[Arena]:
    """Sort arenas top-to-bottom, left-to-right (row-major order).

    Clusters arenas into rows by y-coordinate (gap > median_radius = new row),
    then sorts each row by x. Re-assigns .idx in the sorted order.
    """
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


# ┌─────────────────────────────────────────────────────────────────────┐
# │  FRAME EXTRACTION                         « reading the first frame » │
# └─────────────────────────────────────────────────────────────────────┘
def get_mp4_chunks(recording_dir: str) -> List[str]:
    """Return sorted list of .mp4 chunk file paths in recording_dir."""
    chunks = sorted(glob.glob(os.path.join(recording_dir, "*.mp4")))
    if not chunks:
        raise FileNotFoundError(f"No .mp4 files found in {recording_dir}")
    return chunks


def extract_first_frame(mp4_path: str) -> np.ndarray:
    """Extract the first frame from an mp4 file using ffmpeg.

    Returns an RGB numpy array (H, W, 3) uint8.
    Falls back to PIL/Pillow pipe if ffprobe is available.
    """
    # Use ffmpeg to pipe a single frame as raw RGB
    cmd = [
        "ffmpeg", "-i", mp4_path,
        "-frames:v", "1",
        "-f", "image2pipe",
        "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo",
        "-loglevel", "error",
        "-"
    ]
    # First get dimensions via ffprobe
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

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr.decode()}")

    frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape((h, w, 3))
    return frame


# ┌─────────────────────────────────────────────────────────────────────┐
# │  HOUGH CIRCLE DETECTION               « finding the arenas »        │
# └─────────────────────────────────────────────────────────────────────┘
def detect_arenas_hough(
    frame: np.ndarray,
    n_expected: int = DEFAULT_N_ARENAS,
    min_radius: int = 0,
    max_radius: int = 0,
) -> List[Arena]:
    """Detect circular arenas via Hough Circle Transform.

    Auto-estimates radius bounds from frame dimensions if not provided.
    Iteratively loosens the accumulator threshold until >= n_expected
    circles are found (or gives up and returns whatever was detected).

    Args:
        frame: RGB or grayscale image array.
        n_expected: Target number of arenas.
        min_radius: Minimum circle radius in pixels (0 = auto).
        max_radius: Maximum circle radius in pixels (0 = auto).

    Returns:
        List of Arena objects sorted in row-major order.
    """
    try:
        import cv2
        return _detect_hough_cv2(frame, n_expected, min_radius, max_radius)
    except ImportError:
        print("[arena_extractor] OpenCV not found, using scipy fallback for Hough detection.")
        return _detect_hough_scipy(frame, n_expected, min_radius, max_radius)


def _detect_hough_cv2(
    frame: np.ndarray,
    n_expected: int,
    min_radius: int,
    max_radius: int,
) -> List[Arena]:
    """Hough circle detection using OpenCV (preferred, faster)."""
    import cv2

    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame.copy()

    h, w = gray.shape[:2]
    if min_radius == 0:
        min_radius = int(min(h, w) * 0.05)
    if max_radius == 0:
        max_radius = int(min(h, w) * 0.18)

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    best_circles = None
    for param2 in range(80, 8, -3):
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_radius * 2,
            param1=100,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is not None:
            if best_circles is None or circles.shape[1] > best_circles.shape[1]:
                best_circles = circles
            if circles.shape[1] >= n_expected:
                break

    arenas: List[Arena] = []
    if best_circles is not None:
        for c in best_circles[0]:
            arenas.append(Arena(cx=float(c[0]), cy=float(c[1]), r=float(c[2])))

    return sort_arenas_row_major(arenas)


def _detect_hough_scipy(
    frame: np.ndarray,
    n_expected: int,
    min_radius: int,
    max_radius: int,
) -> List[Arena]:
    """Fallback Hough circle detection using scipy/skimage."""
    try:
        from skimage.transform import hough_circle, hough_circle_peaks
        from skimage.feature import canny
        from skimage.color import rgb2gray
    except ImportError:
        warnings.warn(
            "Neither OpenCV nor scikit-image found. "
            "Install one of: opencv-python, scikit-image"
        )
        return []

    if len(frame.shape) == 3:
        gray = rgb2gray(frame)
    else:
        gray = frame.astype(float) / 255.0

    h, w = gray.shape[:2]
    if min_radius == 0:
        min_radius = int(min(h, w) * 0.05)
    if max_radius == 0:
        max_radius = int(min(h, w) * 0.18)

    edges = canny(gray, sigma=2.0)
    radii = np.arange(min_radius, max_radius, 3)
    hough_res = hough_circle(edges, radii)
    _, cx_arr, cy_arr, r_arr = hough_circle_peaks(
        hough_res, radii,
        min_xdistance=min_radius * 2,
        min_ydistance=min_radius * 2,
        total_num_peaks=n_expected + 5,
    )

    arenas = [Arena(cx=float(cx), cy=float(cy), r=float(r))
              for cx, cy, r in zip(cx_arr, cy_arr, r_arr)]
    return sort_arenas_row_major(arenas[:n_expected + 5])


# ┌─────────────────────────────────────────────────────────────────────┐
# │  MULTI-CIRCLE ANNOTATOR GUI       « interactive verification »      │
# └─────────────────────────────────────────────────────────────────────┘
class MultiCircleEditor:
    """Interactive matplotlib GUI for editing multiple circular arenas.

    Built on the same interaction patterns as circle_annotator.py but
    extended for N simultaneous circles with auto-detection.

    Controls:
      Left-click + drag centre marker   → move that arena
      Left-click + drag rim handle (◇)  → resize that arena
      Right-click on empty space         → add new arena (median radius)
      Right-click on a centre            → delete that arena
      A                                  → re-run auto-detection
      +/=                                → increase all radii by 5 px
      -                                  → decrease all radii by 5 px
      U                                  → uniform radii (set all to median)
      L                                  → toggle labels
      F                                  → toggle fill
      H                                  → toggle help overlay
      S                                  → save arena JSON
      Q / Enter / Escape                 → accept and continue pipeline
    """

    WINDOW_NAME = "Arena Extractor"

    def __init__(self, frame: np.ndarray, arenas: List[Arena],
                 recording_name: str = "", json_path: str = ""):
        self.frame = frame
        self.arenas = arenas
        self.recording_name = recording_name
        self.json_path = json_path

        # Interaction state
        self._dragging: Optional[Arena] = None
        self._drag_mode: str = ""   # "move" or "resize"
        self._drag_offset: Tuple[float, float] = (0.0, 0.0)
        self._mouse_xy: Tuple[float, float] = (0.0, 0.0)

        # Display toggles
        self.show_labels = True
        self.show_fill = True
        self.show_help = False

        # Result
        self.accepted = False

        # Build the figure
        h, w = frame.shape[:2]
        dpi = 100
        fig_w = min(w / dpi, 18)
        fig_h = min(h / dpi, 12)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
        self.fig.canvas.manager.set_window_title(self.WINDOW_NAME)
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._render()
        plt.show()

    # ── colour helper ────────────────────────────────────────────────
    @staticmethod
    def _colour(idx: int) -> str:
        return COLOURS[idx % len(COLOURS)]

    # ── rendering « painting the screen » ────────────────────────────
    def _render(self):
        self.ax.clear()
        h, w = self.frame.shape[:2]

        if len(self.frame.shape) == 2:
            self.ax.imshow(self.frame, cmap="gray", aspect="equal")
        else:
            self.ax.imshow(self.frame, aspect="equal")

        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)

        self.arenas = sort_arenas_row_major(self.arenas)

        for a in self.arenas:
            col = self._colour(a.idx)

            # Circle patch
            circle_patch = MplCircle(
                (a.cx, a.cy), a.r,
                fill=self.show_fill,
                facecolor=col if self.show_fill else "none",
                edgecolor=col,
                alpha=FILL_ALPHA if self.show_fill else EDGE_ALPHA,
                linewidth=2,
            )
            self.ax.add_patch(circle_patch)

            # Bounding box (the actual crop region)
            bx, by, bw, bh = a.bbox()
            self.ax.add_patch(plt.Rectangle(
                (bx, by), bw, bh,
                fill=False, edgecolor=col, linewidth=1,
                linestyle="--", alpha=0.5,
            ))

            # Centre marker
            self.ax.scatter(
                [a.cx], [a.cy], s=CENTRE_SIZE, c=[col],
                edgecolors="white", linewidth=1.5, zorder=5, marker="+",
            )

            # Rim handle at 3-o'clock
            rim_x, rim_y = a.cx + a.r, a.cy
            self.ax.scatter(
                [rim_x], [rim_y], s=RIM_HANDLE_SIZE, c=[col],
                edgecolors="white", linewidth=1.5, zorder=6, marker="D",
            )

            # Label
            if self.show_labels:
                self.ax.annotate(
                    f"#{a.idx:02d}",
                    (a.cx, a.cy - a.r - 8),
                    fontsize=LABEL_FONTSIZE, fontweight="bold",
                    color="white", ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=col, alpha=0.85),
                    zorder=7,
                )

        # Help overlay
        if self.show_help:
            self._render_help()

        # Status bar
        status = (
            f"{self.recording_name}  |  "
            f"{len(self.arenas)} arenas  |  "
            f"drag=move/resize  right-click=add/delete  Q=accept"
        )
        self.ax.set_title(status, fontsize=9, loc="left", pad=6)
        self.ax.set_axis_off()
        self.fig.canvas.draw_idle()

    def _render_help(self):
        text = (
            "=== ARENA EXTRACTOR HELP ===\n"
            "\n"
            "MOUSE\n"
            "  Drag centre (+)      Move arena\n"
            "  Drag rim handle (◇)  Resize arena\n"
            "  Right-click empty    Add new arena\n"
            "  Right-click centre   Delete arena\n"
            "\n"
            "KEYS\n"
            "  A            Re-run auto-detection\n"
            "  +/=          Increase all radii by 5 px\n"
            "  -            Decrease all radii by 5 px\n"
            "  U            Uniform radii (set all to median)\n"
            "  L            Toggle arena labels\n"
            "  F            Toggle circle fill\n"
            "  H            Toggle this help\n"
            "  S            Save arena positions to JSON\n"
            "  Q / Enter    Accept & continue to extraction\n"
            "  Escape       Abort (no extraction)\n"
            "\n"
            f"Arenas: {len(self.arenas)}\n"
            f"Frame:  {self.frame.shape[1]} x {self.frame.shape[0]} px"
        )
        self.ax.text(
            0.02, 0.98, text,
            transform=self.ax.transAxes,
            fontsize=10, fontfamily="monospace",
            verticalalignment="top", color="white", zorder=10,
            bbox=dict(
                boxstyle="round,pad=0.8",
                facecolor="black", alpha=0.88,
                edgecolor="#00ff88", linewidth=1.5,
            ),
        )

    # ── hit-testing « who did you click? » ───────────────────────────
    def _hit_test(self, event) -> Tuple[Optional[Arena], str]:
        """Find which arena handle (if any) is under the cursor.

        Returns (arena, mode) where mode is 'move', 'resize', or ''.
        """
        if event.xdata is None:
            return None, ""

        # Check in reverse so topmost (last drawn) wins
        for a in reversed(self.arenas):
            # Rim handle
            rim_x, rim_y = a.cx + a.r, a.cy
            disp_rim = self.ax.transData.transform((rim_x, rim_y))
            if np.hypot(disp_rim[0] - event.x, disp_rim[1] - event.y) < PICK_RADIUS_PX:
                return a, "resize"

            # Centre
            disp_c = self.ax.transData.transform((a.cx, a.cy))
            if np.hypot(disp_c[0] - event.x, disp_c[1] - event.y) < PICK_RADIUS_PX:
                return a, "move"

        return None, ""

    # ── event handlers « reacting to the user » ──────────────────────
    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        if event.button == 1:  # left click
            arena, mode = self._hit_test(event)
            if arena:
                self._dragging = arena
                self._drag_mode = mode
                self._drag_offset = (event.xdata - arena.cx, event.ydata - arena.cy)

        elif event.button == 3:  # right click
            arena, mode = self._hit_test(event)
            if arena and mode == "move":
                # Delete arena
                self.arenas.remove(arena)
                self._render()
            else:
                # Add new arena
                median_r = float(np.median([a.r for a in self.arenas])) if self.arenas else 100.0
                new = Arena(cx=event.xdata, cy=event.ydata, r=median_r)
                self.arenas.append(new)
                self._render()

    def _on_release(self, event):
        if self._dragging is not None:
            self._dragging = None
            self._drag_mode = ""
            self._render()

    def _on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        self._mouse_xy = (event.xdata, event.ydata)

        if self._dragging is not None:
            a = self._dragging
            h, w = self.frame.shape[:2]

            if self._drag_mode == "move":
                a.cx = float(np.clip(event.xdata - self._drag_offset[0], 0, w))
                a.cy = float(np.clip(event.ydata - self._drag_offset[1], 0, h))
                self._render()

            elif self._drag_mode == "resize":
                new_r = float(np.hypot(event.xdata - a.cx, event.ydata - a.cy))
                a.r = max(20.0, new_r)
                self._render()

    def _on_key(self, event):
        key = event.key

        if key in ("q", "enter"):
            self.accepted = True
            plt.close(self.fig)

        elif key == "escape":
            self.accepted = False
            plt.close(self.fig)

        elif key == "a":
            # Re-run auto-detection
            self.arenas = detect_arenas_hough(self.frame)
            print(f"[arena_extractor] Re-detected {len(self.arenas)} arenas")
            self._render()

        elif key in ("+", "="):
            for a in self.arenas:
                a.r += 5
            self._render()

        elif key == "-":
            for a in self.arenas:
                a.r = max(20, a.r - 5)
            self._render()

        elif key == "u":
            if self.arenas:
                median_r = float(np.median([a.r for a in self.arenas]))
                for a in self.arenas:
                    a.r = median_r
                self._render()

        elif key == "l":
            self.show_labels = not self.show_labels
            self._render()

        elif key == "f":
            self.show_fill = not self.show_fill
            self._render()

        elif key == "h":
            self.show_help = not self.show_help
            self._render()

        elif key == "s":
            self._save_json()

    def _save_json(self):
        """Persist arena positions to JSON."""
        path = self.json_path or f"{self.recording_name}_arenas.json"
        data = {
            "recording": self.recording_name,
            "frame_width": self.frame.shape[1],
            "frame_height": self.frame.shape[0],
            "arenas": [a.to_dict() for a in sort_arenas_row_major(self.arenas)],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[arena_extractor] Saved {len(self.arenas)} arenas → {path}")

    def get_arenas(self) -> Optional[List[Arena]]:
        """Return the final arena list, or None if user aborted."""
        if self.accepted:
            return sort_arenas_row_major(self.arenas)
        return None


# ┌─────────────────────────────────────────────────────────────────────┐
# │  FFMPEG CONCAT + CROP                  « the heavy lifting »        │
# └─────────────────────────────────────────────────────────────────────┘
def build_concat_file(chunks: List[str], tmpdir: str) -> str:
    """Write an ffmpeg concat demuxer file listing all chunks.

    Returns path to the concat file.
    """
    concat_path = os.path.join(tmpdir, "concat.txt")
    with open(concat_path, "w") as f:
        for chunk in chunks:
            # ffmpeg concat demuxer needs absolute paths with escaped quotes
            f.write(f"file '{os.path.abspath(chunk)}'\n")
    return concat_path


def concat_and_crop(
    chunks: List[str],
    arenas: List[Arena],
    output_dir: str,
    recording_name: str,
    codec: str = "libx264",
    crf: int = 18,
    frame_height: int = 0,
    frame_width: int = 0,
    keep_full: bool = False,
) -> List[str]:
    """Concatenate mp4 chunks and crop each arena region.

    Strategy:
      - Use ffmpeg concat demuxer to stitch chunks (no re-encode)
      - For each arena, run a single ffmpeg pass that reads from the
        concat demuxer and applies a crop filter

    This avoids creating a huge intermediate full-size video unless
    keep_full=True.

    Args:
        chunks: Sorted list of mp4 chunk paths.
        arenas: List of Arena objects with crop coordinates.
        output_dir: Directory for output files.
        recording_name: Base name for output files.
        codec: Video codec for re-encoding cropped regions.
        crf: Constant rate factor (quality; lower = better).
        frame_height: Original frame height (for clamping).
        frame_width: Original frame width (for clamping).
        keep_full: If True, also save the full concatenated video.

    Returns:
        List of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs: List[str] = []

    with tempfile.TemporaryDirectory(prefix="arena_extract_") as tmpdir:
        concat_file = build_concat_file(chunks, tmpdir)

        # Optionally save full concat
        if keep_full:
            full_path = os.path.join(output_dir, f"{recording_name}_full.mp4")
            print(f"  Concatenating full video → {full_path}")
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_file,
                "-c", "copy",
                full_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            outputs.append(full_path)

        # Crop each arena
        n = len(arenas)
        for i, arena in enumerate(arenas):
            x, y, w, h = arena.bbox()

            # Clamp to frame bounds
            if frame_width > 0:
                if x + w > frame_width:
                    w = frame_width - x
            if frame_height > 0:
                if y + h > frame_height:
                    h = frame_height - y

            # Force even dimensions (required by most codecs)
            w = w - (w % 2)
            h = h - (h % 2)

            out_path = os.path.join(
                output_dir, f"{recording_name}_arena_{arena.idx:02d}.mp4"
            )
            print(f"  [{i + 1}/{n}] Arena #{arena.idx:02d}  "
                  f"crop={w}x{h}+{x}+{y}  → {os.path.basename(out_path)}")

            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0", "-i", concat_file,
                "-vf", f"crop={w}:{h}:{x}:{y}",
                "-c:v", codec,
                "-crf", str(crf),
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-an",  # drop audio (Basler cameras have none)
                out_path,
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                print(f"    ⚠ ffmpeg error: {result.stderr.decode()[-200:]}")
            else:
                outputs.append(out_path)

    return outputs


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CLI + MAIN                            « parsing & orchestration »  │
# └─────────────────────────────────────────────────────────────────────┘
def _die(msg: str):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="arena_extractor",
        description=(
            "Arena Extractor — detect, verify & crop circular arenas\n"
            "from Motif recording directories (sequential .mp4 chunks)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s /data/droso_18deg_20251002_102448\n"
            "  %(prog)s /data/recording --output_dir /results/\n"
            "  %(prog)s /data/recording --arenas arenas.json   # skip GUI\n"
            "  %(prog)s /data/recording --n_arenas 8 --crf 23\n"
            "\n"
            "GUI key bindings:\n"
            "  Drag centre (+)      Move arena\n"
            "  Drag rim handle (◇)  Resize arena\n"
            "  Right-click empty    Add new arena\n"
            "  Right-click centre   Delete arena\n"
            "  A   Re-run auto-detection\n"
            "  +/- Adjust all radii\n"
            "  U   Uniform radii (median)\n"
            "  H   Help overlay\n"
            "  S   Save positions\n"
            "  Q   Accept & extract\n"
        ),
    )
    parser.add_argument(
        "recording_dir",
        help="Path to the Motif recording directory containing .mp4 chunks.",
    )
    parser.add_argument(
        "-o", "--output_dir",
        default=None,
        help="Output directory for cropped videos (default: same as recording).",
    )
    parser.add_argument(
        "-n", "--n_arenas",
        type=int, default=DEFAULT_N_ARENAS,
        help=f"Expected number of arenas (default: {DEFAULT_N_ARENAS}).",
    )
    parser.add_argument(
        "--arenas",
        default=None,
        help="JSON file with pre-defined arena positions (skips GUI).",
    )
    parser.add_argument(
        "--crf",
        type=int, default=18,
        help="FFmpeg CRF quality for output (default: 18, lower = better).",
    )
    parser.add_argument(
        "--codec",
        default="libx264",
        help="Video codec (default: libx264).",
    )
    parser.add_argument(
        "--keep_full",
        action="store_true",
        help="Also save the full concatenated (uncropped) video.",
    )
    parser.add_argument(
        "--no_gui",
        action="store_true",
        help="Skip GUI even without --arenas (use auto-detection as-is).",
    )
    return parser.parse_args()


def load_arenas_json(path: str) -> List[Arena]:
    """Load arena positions from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    arenas_data = data if isinstance(data, list) else data.get("arenas", data)
    if isinstance(arenas_data, list):
        return [Arena.from_dict(d) for d in arenas_data]
    _die(f"Could not parse arenas from {path}")


def main():
    args = parse_args()

    recording_dir = args.recording_dir.rstrip("/")
    recording_name = os.path.basename(recording_dir)
    output_dir = args.output_dir or recording_dir

    # ── Discover chunks ──────────────────────────────────────────────
    print("┌──────────────────────────────────────────┐")
    print("│  ARENA EXTRACTOR v1.0                    │")
    print("│  « detect. adjust. stitch. crop. »       │")
    print("└──────────────────────────────────────────┘")

    chunks = get_mp4_chunks(recording_dir)
    print(f"  Recording: {recording_name}")
    print(f"  Chunks:    {len(chunks)} .mp4 files")
    print(f"  Output:    {output_dir}")

    # ── Get arena positions ──────────────────────────────────────────
    if args.arenas:
        # Load from JSON (skip detection + GUI)
        arenas = load_arenas_json(args.arenas)
        print(f"  Loaded {len(arenas)} arenas from {args.arenas}")
        frame = extract_first_frame(chunks[0])
        frame_h, frame_w = frame.shape[:2]
    else:
        # Extract first frame and detect
        print("  Extracting first frame...")
        frame = extract_first_frame(chunks[0])
        frame_h, frame_w = frame.shape[:2]
        print(f"  Frame size: {frame_w} x {frame_h}")

        print(f"  Running Hough circle detection (expecting ~{args.n_arenas})...")
        arenas = detect_arenas_hough(frame, n_expected=args.n_arenas)
        print(f"  Detected {len(arenas)} arenas")

        if not args.no_gui:
            # Launch interactive editor
            json_path = os.path.join(output_dir, f"{recording_name}_arenas.json")
            editor = MultiCircleEditor(
                frame, arenas,
                recording_name=recording_name,
                json_path=json_path,
            )
            arenas = editor.get_arenas()
            if arenas is None:
                print("\n  Aborted by user.")
                sys.exit(0)

    if not arenas:
        _die("No arenas defined. Nothing to extract.")

    # Save final arena positions
    arenas = sort_arenas_row_major(arenas)
    json_out = os.path.join(output_dir, f"{recording_name}_arenas.json")
    data = {
        "recording": recording_name,
        "frame_width": frame_w,
        "frame_height": frame_h,
        "arenas": [a.to_dict() for a in arenas],
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Arena positions saved → {json_out}")

    # ── Concatenate & crop ───────────────────────────────────────────
    print(f"\n  Starting extraction ({len(arenas)} arenas × {len(chunks)} chunks)...")
    outputs = concat_and_crop(
        chunks=chunks,
        arenas=arenas,
        output_dir=output_dir,
        recording_name=recording_name,
        codec=args.codec,
        crf=args.crf,
        frame_height=frame_h,
        frame_width=frame_w,
        keep_full=args.keep_full,
    )

    print(f"\n  ✓ Done! {len(outputs)} files written to {output_dir}")
    for p in outputs:
        print(f"    → {os.path.basename(p)}")


if __name__ == "__main__":
    main()
