#!/usr/bin/env python3
"""
 ╔═══════════════════════════════════════════════════════════════════════╗
 ║   Arena Extractor v1.2                                               ║
 ║   Detect, verify & extract circular arenas from Motif recordings     ║
 ║   ── detect. adjust. stitch. crop. science. ──                       ║
 ╚═══════════════════════════════════════════════════════════════════════╝

Pipeline for Drosophila arena recordings (Motif / FLIR / Basler).

  1. Builds a max-projection image (100 frames, 25-100% of duration)
     to erase dark fly bodies and reveal clean arena boundaries
  2. Detects circles via Hough transform on the max-projection
  3. Opens a fast OpenCV GUI for verification / adjustment
  4. Concatenates .mp4 chunks via ffmpeg concat demuxer
  5. Crops each arena with 10% padding + white corner masks
     (5% triangles to hide neighbouring arenas from the tracker)

Requirements:
    pip install numpy opencv-python pyyaml
    ffmpeg on PATH

License: MIT · Author: Bart R.H. Geurten
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ┌─────────────────────────────────────────────────────────────────────┐
# │  CONSTANTS                                                          │
# └─────────────────────────────────────────────────────────────────────┘
DEFAULT_N_ARENAS = 10
CROP_PADDING = 0.10       # 10 % padding around arena circle
CORNER_MASK = 0.05        # 5 % of crop side length for corner triangles
RADIUS_TOLERANCE = 0.25   # reject circles with radius ± 25 % of median
MIN_ARENAS_FOR_FILTER = 5
N_PROJECTION_FRAMES = 10  # frames sampled for max-projection
PROJECTION_START = 0.25   # start at 25 % of total duration


# ┌─────────────────────────────────────────────────────────────────────┐
# │  ARENA DATACLASS                                                    │
# └─────────────────────────────────────────────────────────────────────┘
@dataclass
class Arena:
    cx: float
    cy: float
    r: float
    idx: int = 0

    def bbox(self, padding: float = CROP_PADDING,
             frame_w: int = 0, frame_h: int = 0) -> Tuple[int, int, int, int]:
        """(x, y, w, h) with padding, clamped to frame, even dimensions."""
        pr = self.r * (1.0 + padding)
        x, y = int(self.cx - pr), int(self.cy - pr)
        w = h = int(2 * pr)
        if x < 0: w += x; x = 0
        if y < 0: h += y; y = 0
        if frame_w > 0 and x + w > frame_w: w = frame_w - x
        if frame_h > 0 and y + h > frame_h: h = frame_h - y
        return x, y, max(2, w - w % 2), max(2, h - h % 2)

    def to_dict(self):
        return {"cx": self.cx, "cy": self.cy, "r": self.r, "idx": self.idx}

    @classmethod
    def from_dict(cls, d):
        return cls(cx=d["cx"], cy=d["cy"], r=d["r"], idx=d.get("idx", 0))


def sort_arenas_row_major(arenas: List[Arena]) -> List[Arena]:
    """Sort top→bottom, left→right. Re-numbers 0..N-1."""
    if not arenas:
        return arenas
    arenas = sorted(arenas, key=lambda a: a.cy)
    med_r = float(np.median([a.r for a in arenas]))
    rows, cur = [], [arenas[0]]
    for a in arenas[1:]:
        if abs(a.cy - cur[-1].cy) > med_r:
            rows.append(cur); cur = [a]
        else:
            cur.append(a)
    rows.append(cur)
    out, idx = [], 0
    for row in rows:
        for a in sorted(row, key=lambda a: a.cx):
            a.idx = idx; out.append(a); idx += 1
    return out


def filter_by_radius(arenas, tol=RADIUS_TOLERANCE, keep=MIN_ARENAS_FOR_FILTER):
    if len(arenas) < 3:
        return sort_arenas_row_major(arenas)
    med = float(np.median([a.r for a in arenas]))
    f = [a for a in arenas if med * (1 - tol) <= a.r <= med * (1 + tol)]
    return sort_arenas_row_major(f if len(f) >= keep else arenas)


# ┌─────────────────────────────────────────────────────────────────────┐
# │  VIDEO I/O helpers                                                  │
# └─────────────────────────────────────────────────────────────────────┘
def get_mp4_chunks(d):
    c = sorted(glob.glob(os.path.join(d, "*.mp4")))
    if not c: raise FileNotFoundError(f"No .mp4 in {d}")
    return c


def _probe(mp4):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", mp4],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr}")
    info = json.loads(r.stdout)
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    w, h = int(vs["width"]), int(vs["height"])
    dur = float(info.get("format", {}).get("duration", 0))
    if dur <= 0:
        dur = float(vs.get("duration", 0))
    fps_parts = vs.get("r_frame_rate", "25/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 25.0
    return w, h, dur, fps


def extract_frame_at(mp4, t, w, h):
    """Extract a single RGB frame at time t (seconds)."""
    cmd = [
        "ffmpeg", "-ss", f"{t:.3f}", "-i", mp4,
        "-frames:v", "1", "-f", "image2pipe",
        "-pix_fmt", "rgb24", "-vcodec", "rawvideo",
        "-loglevel", "error", "-"
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or len(r.stdout) < h * w * 3:
        return None
    return np.frombuffer(r.stdout[:h * w * 3], dtype=np.uint8).reshape((h, w, 3))


# ┌─────────────────────────────────────────────────────────────────────┐
# │  MAX-PROJECTION                                                     │
# │  Sample N frames from 25–100% of each chunk, take pixel-wise max.  │
# │  Dark flies move → max keeps the bright arena background.           │
# └─────────────────────────────────────────────────────────────────────┘
def build_max_projection(chunks: List[str], n_frames: int = N_PROJECTION_FRAMES,
                         start_frac: float = PROJECTION_START) -> np.ndarray:
    """Build a max-intensity projection across sampled frames.

    Samples n_frames evenly from start_frac..1.0 of total duration across
    all chunks. Because flies are dark and move, the per-pixel maximum
    recovers the bright, static arena background.
    """
    # Compute total duration
    chunk_info = []  # (path, w, h, start_time, end_time)
    cumulative = 0.0
    for c in chunks:
        w, h, dur, fps = _probe(c)
        chunk_info.append((c, w, h, cumulative, cumulative + dur))
        cumulative += dur

    total_dur = cumulative
    if total_dur <= 0:
        raise RuntimeError("Could not determine video duration")

    W, H = chunk_info[0][1], chunk_info[0][2]
    t_start = total_dur * start_frac
    t_end = total_dur * 0.999  # avoid exact end

    sample_times = np.linspace(t_start, t_end, n_frames)
    projection = np.zeros((H, W, 3), dtype=np.uint8)

    print(f"  Building max-projection ({n_frames} frames, "
          f"{start_frac:.0%}–100% of {total_dur:.1f}s)...")

    for i, t in enumerate(sample_times):
        # Find which chunk this time falls in
        for path, cw, ch, cs, ce in chunk_info:
            if cs <= t < ce or (t >= ce and path == chunk_info[-1][0]):
                local_t = t - cs
                frame = extract_frame_at(path, local_t, W, H)
                if frame is not None:
                    projection = np.maximum(projection, frame)
                break
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{n_frames} frames sampled")

    return projection


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CIRCLE DETECTION on max-projection                                 │
# └─────────────────────────────────────────────────────────────────────┘
def detect_arenas(frame, n_expected=DEFAULT_N_ARENAS, min_r=0, max_r=0):
    """Hough circle detection on (ideally max-projected) frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
    fh, fw = gray.shape[:2]
    if min_r == 0: min_r = int(min(fh, fw) * 0.04)
    if max_r == 0: max_r = int(min(fh, fw) * 0.20)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    blurred = cv2.GaussianBlur(clahe.apply(gray), (9, 9), 2)

    best = None
    for p2 in range(80, 8, -3):
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r * 2,
            param1=100, param2=p2, minRadius=min_r, maxRadius=max_r,
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
    return filter_by_radius(sort_arenas_row_major(arenas))


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CORNER MASKING                                                     │
# │  White-out triangular corners in the crop to hide neighbouring      │
# │  arenas that might confuse identity trackers.                       │
# └─────────────────────────────────────────────────────────────────────┘
def corner_mask_triangles(crop_w, crop_h, frac=CORNER_MASK):
    """Return 4 triangle vertex arrays for the corners of a crop.

    Each triangle has legs of length frac * side_length, measured
    from the corner along each edge.

    Returns list of 4 numpy arrays, each shape (3, 2) int32.
    """
    dx = int(crop_w * frac)
    dy = int(crop_h * frac)
    return [
        np.array([[0, 0], [dx, 0], [0, dy]], np.int32),                          # top-left
        np.array([[crop_w, 0], [crop_w - dx, 0], [crop_w, dy]], np.int32),        # top-right
        np.array([[0, crop_h], [dx, crop_h], [0, crop_h - dy]], np.int32),        # bottom-left
        np.array([[crop_w, crop_h], [crop_w - dx, crop_h],
                  [crop_w, crop_h - dy]], np.int32),                              # bottom-right
    ]


def apply_corner_mask_to_frame(frame, frac=CORNER_MASK, colour=255):
    """White-out corner triangles on a frame (in-place)."""
    h, w = frame.shape[:2]
    for tri in corner_mask_triangles(w, h, frac):
        cv2.fillPoly(frame, [tri], (colour, colour, colour))
    return frame


# ┌─────────────────────────────────────────────────────────────────────┐
# │  OPENCV GUI                                                         │
# │  Uses cv2.imshow — instant rendering, no matplotlib overhead.       │
# └─────────────────────────────────────────────────────────────────────┘
COLOURS_BGR = [
    (136, 255, 0), (102, 68, 255), (255, 170, 68), (0, 170, 255), (255, 68, 170),
    (204, 102, 255), (204, 221, 0), (68, 221, 255), (68, 255, 136), (68, 136, 255),
]


class ArenaEditorCV2:
    """Fast OpenCV-based arena editor.

    Controls:
      Left-drag on centre dot    → move arena
      Left-drag on rim diamond   → resize arena
      Right-click empty          → add arena
      Right-click on centre      → delete arena
      A   re-run detection
      +/= increase all radii     -   decrease all radii
      U   uniform radii (median)
      C   toggle corner mask preview
      H   toggle help overlay
      Q / Enter  accept          Esc  abort
    """

    WIN = "Arena Extractor"
    HANDLE_R = 12  # radius for centre/rim hit-testing in display pixels

    def __init__(self, frame_rgb, arenas, recording_name="", json_path=""):
        self.orig = frame_rgb.copy()
        self.arenas = sort_arenas_row_major(arenas)
        self.rec_name = recording_name
        self.json_path = json_path
        self.show_corners = True
        self.show_help = False
        self.accepted = False

        # Display: render at full res, let OpenCV window handle scaling
        h, w = frame_rgb.shape[:2]
        self.scale = 1.0  # all coordinates are in full-res frame space
        target_w = min(w, 1920)
        target_h = int(h * (target_w / w))

        # Drag state
        self._drag = None   # (arena, mode)  mode='move'|'resize'
        self._drag_off = (0.0, 0.0)

        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN, target_w, target_h)
        cv2.setMouseCallback(self.WIN, self._mouse_cb)
        self._render()
        self._loop()

    # ── coordinate conversion ────────────────────────────────────────
    def _d2f(self, dx, dy):
        """Display coords → frame coords.
        With WINDOW_NORMAL, OpenCV maps mouse coords to image coords
        automatically, so display coords == frame coords."""
        return float(dx), float(dy)

    def _f2d(self, fx, fy):
        """Frame coords → display coords (identity with WINDOW_NORMAL)."""
        return int(fx), int(fy)

    # ── rendering ────────────────────────────────────────────────────
    def _render(self):
        vis = cv2.cvtColor(self.orig, cv2.COLOR_RGB2BGR)
        fh, fw = vis.shape[:2]
        self.arenas = sort_arenas_row_major(self.arenas)

        for a in self.arenas:
            col = COLOURS_BGR[a.idx % len(COLOURS_BGR)]
            cx, cy, r = int(a.cx), int(a.cy), int(a.r)

            # Filled circle (semi-transparent via overlay)
            overlay = vis.copy()
            cv2.circle(overlay, (cx, cy), r, col, -1)
            cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)

            # Circle edge
            cv2.circle(vis, (cx, cy), r, col, 2)

            # Bounding box (padded crop)
            bx, by, bw, bh = a.bbox(padding=CROP_PADDING, frame_w=fw, frame_h=fh)
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), col, 1)

            # Corner mask preview — diagonal lines in each corner of the bbox
            if self.show_corners:
                dx_c = int(bw * CORNER_MASK)
                dy_c = int(bh * CORNER_MASK)
                # top-left
                cv2.line(vis, (bx + dx_c, by), (bx, by + dy_c), (200, 200, 200), 1)
                # top-right
                cv2.line(vis, (bx + bw - dx_c, by), (bx + bw, by + dy_c), (200, 200, 200), 1)
                # bottom-left
                cv2.line(vis, (bx + dx_c, by + bh), (bx, by + bh - dy_c), (200, 200, 200), 1)
                # bottom-right
                cv2.line(vis, (bx + bw - dx_c, by + bh), (bx + bw, by + bh - dy_c), (200, 200, 200), 1)

            # Centre dot
            cv2.circle(vis, (cx, cy), 6, col, -1)
            cv2.circle(vis, (cx, cy), 6, (255, 255, 255), 1)

            # Rim handle (diamond at 3 o'clock)
            rim_x = cx + r
            pts = np.array([
                [rim_x, cy - 6], [rim_x + 6, cy],
                [rim_x, cy + 6], [rim_x - 6, cy],
            ], np.int32)
            cv2.fillPoly(vis, [pts], col)
            cv2.polylines(vis, [pts], True, (255, 255, 255), 1)

            # Label
            label = f"#{a.idx:02d}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            lx, ly = cx - tw // 2, cy - r - 12
            cv2.rectangle(vis, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4), col, -1)
            cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

        # Status bar
        status = (f"{self.rec_name}  |  {len(self.arenas)} arenas  |  "
                  f"drag=move/resize  RMB=add/del  Q=accept")
        cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        # Help overlay
        if self.show_help:
            lines = [
                "MOUSE: drag centre=move, drag rim=resize",
                "  right-click empty=add, right-click centre=delete",
                "KEYS: A=re-detect  +/-=radii  U=uniform  C=corners",
                "  H=help  S=save  Q/Enter=accept  Esc=abort",
            ]
            for i, ln in enumerate(lines):
                cv2.putText(vis, ln, (10, 55 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 136), 1)

        # WINDOW_NORMAL handles display scaling — show at full resolution
        cv2.imshow(self.WIN, vis)

    # ── hit testing ──────────────────────────────────────────────────
    def _hit(self, fx, fy):
        """Returns (arena, 'move'|'resize') or (None, '')."""
        # Hit radius in frame pixels — scale with image size
        hit_r = max(15, self.orig.shape[1] * 0.008)
        for a in reversed(self.arenas):
            if abs(fx - (a.cx + a.r)) < hit_r and abs(fy - a.cy) < hit_r:
                return a, "resize"
            if np.hypot(fx - a.cx, fy - a.cy) < hit_r:
                return a, "move"
        return None, ""

    # ── mouse callback ───────────────────────────────────────────────
    def _mouse_cb(self, event, x, y, flags, param):
        fx, fy = self._d2f(x, y)
        fh, fw = self.orig.shape[:2]

        if event == cv2.EVENT_LBUTTONDOWN:
            arena, mode = self._hit(fx, fy)
            if arena:
                self._drag = (arena, mode)
                self._drag_off = (fx - arena.cx, fy - arena.cy)

        elif event == cv2.EVENT_MOUSEMOVE and self._drag:
            a, mode = self._drag
            if mode == "move":
                a.cx = float(np.clip(fx - self._drag_off[0], 0, fw))
                a.cy = float(np.clip(fy - self._drag_off[1], 0, fh))
            elif mode == "resize":
                a.r = max(20.0, float(np.hypot(fx - a.cx, fy - a.cy)))
            self._render()

        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None
            self._render()

        elif event == cv2.EVENT_RBUTTONDOWN:
            arena, mode = self._hit(fx, fy)
            if arena and mode == "move":
                self.arenas.remove(arena)
            else:
                mr = float(np.median([a.r for a in self.arenas])) if self.arenas else 100.0
                self.arenas.append(Arena(cx=fx, cy=fy, r=mr))
            self._render()

    # ── keyboard loop ────────────────────────────────────────────────
    def _loop(self):
        while True:
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q") or k == 13:  # q or Enter
                self.accepted = True; break
            elif k == 27:  # Esc
                self.accepted = False; break
            elif k == ord("a"):
                self.arenas = detect_arenas(self.orig)
                print(f"  Re-detected {len(self.arenas)} arenas")
                self._render()
            elif k in (ord("+"), ord("=")):
                for a in self.arenas: a.r += 5
                self._render()
            elif k == ord("-"):
                for a in self.arenas: a.r = max(20, a.r - 5)
                self._render()
            elif k == ord("u"):
                if self.arenas:
                    mr = float(np.median([a.r for a in self.arenas]))
                    for a in self.arenas: a.r = mr
                    self._render()
            elif k == ord("c"):
                self.show_corners = not self.show_corners
                self._render()
            elif k == ord("h"):
                self.show_help = not self.show_help
                self._render()
            elif k == ord("s"):
                self._save_json()
        cv2.destroyAllWindows()

    def _save_json(self):
        path = self.json_path or f"{self.rec_name}_arenas.json"
        with open(path, "w") as f:
            json.dump({
                "recording": self.rec_name,
                "frame_width": self.orig.shape[1],
                "frame_height": self.orig.shape[0],
                "arenas": [a.to_dict() for a in sort_arenas_row_major(self.arenas)],
            }, f, indent=2)
        print(f"  Saved {len(self.arenas)} arenas -> {path}")

    def get_arenas(self):
        return sort_arenas_row_major(self.arenas) if self.accepted else None


# ┌─────────────────────────────────────────────────────────────────────┐
# │  FFMPEG CONCAT + CROP + CORNER MASK                                 │
# └─────────────────────────────────────────────────────────────────────┘
def build_concat_file(chunks, tmpdir):
    p = os.path.join(tmpdir, "concat.txt")
    with open(p, "w") as f:
        for c in chunks:
            f.write(f"file '{os.path.abspath(c)}'\n")
    return p


def concat_and_crop(chunks, arenas, output_dir, recording_name,
                    codec="libx264", crf=18, frame_height=0, frame_width=0,
                    keep_full=False, corner_frac=CORNER_MASK):
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
            print(f"  [{i+1}/{n}] #{arena.idx:02d}  "
                  f"crop={w}x{h}+{x}+{y} (r={arena.r:.0f})  "
                  f"-> {os.path.basename(out_path)}")

            if corner_frac > 0:
                # Crop with ffmpeg, then apply corner mask with OpenCV
                out_tmp = os.path.join(tmpdir, f"tmp_{arena.idx:02d}.mp4")
                result = subprocess.run([
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
                    "-vf", f"crop={w}:{h}:{x}:{y}",
                    "-c:v", codec, "-crf", str(crf), "-preset", "fast",
                    "-pix_fmt", "yuv420p", "-an", out_tmp,
                ], capture_output=True)
                if result.returncode != 0:
                    print(f"    Warning: ffmpeg crop failed: {result.stderr.decode()[-200:]}")
                    continue
                _postprocess_corner_mask(out_tmp, out_path, w, h, corner_frac, codec, crf)
            else:
                # No corner mask — just crop
                result = subprocess.run([
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
                    "-vf", f"crop={w}:{h}:{x}:{y}",
                    "-c:v", codec, "-crf", str(crf), "-preset", "fast",
                    "-pix_fmt", "yuv420p", "-an", out_path,
                ], capture_output=True)
                if result.returncode != 0:
                    print(f"    Warning: ffmpeg error: {result.stderr.decode()[-200:]}")
                    continue

            outputs.append(out_path)
    return outputs


def _postprocess_corner_mask(in_path, out_path, crop_w, crop_h, frac, codec, crf):
    """Read video, white-out corners frame by frame, re-encode."""
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (crop_w, crop_h))
    triangles = corner_mask_triangles(crop_w, crop_h, frac)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for tri in triangles:
            cv2.fillPoly(frame, [tri], (255, 255, 255))
        writer.write(frame)
    cap.release()
    writer.release()


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CLI + MAIN                                                         │
# └─────────────────────────────────────────────────────────────────────┘
def _die(msg):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        prog="arena_extractor",
        description="Arena Extractor v1.2 — detect, verify & crop circular arenas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("recording_dir", help="Motif recording directory.")
    p.add_argument("-o", "--output_dir", default=None)
    p.add_argument("-n", "--n_arenas", type=int, default=DEFAULT_N_ARENAS)
    p.add_argument("--arenas", default=None, help="JSON with arena positions (skip GUI).")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--codec", default="libx264")
    p.add_argument("--padding", type=float, default=CROP_PADDING)
    p.add_argument("--corner_mask", type=float, default=CORNER_MASK,
                   help=f"Corner triangle size as fraction of crop side (default: {CORNER_MASK}).")
    p.add_argument("--no_corner_mask", action="store_true",
                   help="Disable corner white-out.")
    p.add_argument("--keep_full", action="store_true")
    p.add_argument("--no_gui", action="store_true")
    p.add_argument("--n_proj_frames", type=int, default=N_PROJECTION_FRAMES,
                   help=f"Frames for max-projection (default: {N_PROJECTION_FRAMES}).")
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
    global CROP_PADDING, CORNER_MASK
    CROP_PADDING = args.padding
    if args.no_corner_mask:
        CORNER_MASK = 0.0
    else:
        CORNER_MASK = args.corner_mask

    recording_dir = args.recording_dir.rstrip("/")
    recording_name = os.path.basename(recording_dir)
    output_dir = args.output_dir or recording_dir

    print("+" + "-" * 44 + "+")
    print("|  ARENA EXTRACTOR v1.2                      |")
    print("|  detect. adjust. stitch. crop.             |")
    print("+" + "-" * 44 + "+")

    chunks = get_mp4_chunks(recording_dir)
    print(f"  Recording: {recording_name}")
    print(f"  Chunks:    {len(chunks)} .mp4 files")
    print(f"  Output:    {output_dir}")

    if args.arenas:
        arenas = load_arenas_json(args.arenas)
        print(f"  Loaded {len(arenas)} arenas from {args.arenas}")
        # Still need a frame for display
        w, h, dur, fps = _probe(chunks[0])
        proj = extract_frame_at(chunks[-1], dur * 0.9 if dur > 0 else 0, w, h)
        if proj is None:
            proj = extract_frame_at(chunks[0], 0, w, h)
        frame_h, frame_w = h, w
    else:
        # Build max-projection for detection
        proj = build_max_projection(chunks, n_frames=args.n_proj_frames)
        frame_h, frame_w = proj.shape[:2]

        print(f"  Frame size: {frame_w} x {frame_h}")
        print(f"  Running Hough detection on max-projection...")
        arenas = detect_arenas(proj, n_expected=args.n_arenas)
        print(f"  Detected {len(arenas)} arenas")

        if not args.no_gui:
            json_path = os.path.join(output_dir, f"{recording_name}_arenas.json")
            try:
                os.makedirs(output_dir, exist_ok=True)
            except PermissionError:
                _die(f"Cannot create output directory '{output_dir}' — permission denied.\n"
                     f"       Try a path you own, e.g.  -o ~/results/")
            editor = ArenaEditorCV2(
                proj, arenas, recording_name=recording_name, json_path=json_path,
            )
            arenas = editor.get_arenas()
            if arenas is None:
                print("\n  Aborted by user.")
                sys.exit(0)

    if not arenas:
        _die("No arenas defined.")

    arenas = sort_arenas_row_major(arenas)
    json_out = os.path.join(output_dir, f"{recording_name}_arenas.json")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError:
        _die(f"Cannot create output directory '{output_dir}' — permission denied.\n"
             f"       Try a path you own, e.g.  -o ~/results/")
    with open(json_out, "w") as f:
        json.dump({
            "recording": recording_name,
            "frame_width": frame_w, "frame_height": frame_h,
            "padding": CROP_PADDING, "corner_mask": CORNER_MASK,
            "arenas": [a.to_dict() for a in arenas],
        }, f, indent=2)
    print(f"\n  Arena positions saved -> {json_out}")

    print(f"\n  Extracting ({len(arenas)} arenas x {len(chunks)} chunks)...")
    outputs = concat_and_crop(
        chunks=chunks, arenas=arenas, output_dir=output_dir,
        recording_name=recording_name, codec=args.codec, crf=args.crf,
        frame_height=frame_h, frame_width=frame_w,
        keep_full=args.keep_full, corner_frac=CORNER_MASK,
    )
    print(f"\n  Done! {len(outputs)} files written to {output_dir}")
    for p in outputs:
        print(f"    -> {os.path.basename(p)}")


if __name__ == "__main__":
    main()