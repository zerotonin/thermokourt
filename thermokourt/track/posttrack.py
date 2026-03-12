#!/usr/bin/env python3
"""
 ╔═══════════════════════════════════════════════════════════════════════╗
 ║   ThermoKourt — Post-Tracking Pipeline                    v1.0      ║
 ║   Assign identities, crop individuals, render overlay videos        ║
 ╚═══════════════════════════════════════════════════════════════════════╝

After idtracker.ai has produced trajectories for a 3-fly arena, this tool:

  1. IDENTITY ASSIGNMENT — Opens an interactive GUI where the user clicks
     on each fly to assign: male_teal, male_orange, female.
     Saves the mapping as a JSON sidecar.

  2. INDIVIDUAL CROPS — For each identified animal, produces a centred
     crop video following that animal. These are suitable as input for
     SLEAP or DeepLabCut single-animal pose estimation.

  3. OVERLAY VIDEO — Renders teal and orange semi-transparent auras
     around the two males on the full arena video, for manual ethogram
     scoring in GameThogram.

Usage:
    thermokourt-posttrack arena_video.mp4 --session session_folder/
    thermokourt-posttrack arena_video.mp4 --trajectories without_gaps.npy
    thermokourt-posttrack arena_video.mp4 --session session_folder/ --skip_gui
        (reuses previously saved identity assignment)

Requirements:
    pip install numpy opencv-python
    (idtracker.ai output files)

License: MIT · Author: Bart R.H. Geurten
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CONSTANTS                                                          │
# └─────────────────────────────────────────────────────────────────────┘
TEAL_BGR = (210, 180, 0)      # teal in BGR
ORANGE_BGR = (0, 107, 255)    # orange in BGR
FEMALE_BGR = (180, 180, 180)  # grey for female
AURA_ALPHA = 0.35             # overlay transparency
AURA_RADIUS_MULT = 2.5        # aura radius = body_length * this
INDIVIDUAL_CROP_SIZE = 256    # px, side length of individual crop videos
IDENTITY_NAMES = ["male_teal", "male_orange", "female"]
IDENTITY_COLOURS = {
    "male_teal": TEAL_BGR,
    "male_orange": ORANGE_BGR,
    "female": FEMALE_BGR,
}


# ┌─────────────────────────────────────────────────────────────────────┐
# │  TRAJECTORY LOADING                                                 │
# └─────────────────────────────────────────────────────────────────────┘
def load_trajectories(session_dir: str = "", traj_path: str = "") -> np.ndarray:
    """Load trajectories from idtracker.ai output.

    Returns array of shape (N_frames, N_animals, 2) — xy centroids.
    NaN values indicate frames where the animal was not detected.
    """
    if traj_path and os.path.exists(traj_path):
        path = traj_path
    elif session_dir:
        # Try standard idtracker.ai output locations
        # v6 uses trajectories.npy, older versions use without_gaps.npy
        candidates = [
            os.path.join(session_dir, "trajectories", "trajectories.npy"),
            os.path.join(session_dir, "trajectories", "without_gaps.npy"),
            os.path.join(session_dir, "trajectories", "with_gaps.npy"),
        ]
        h5_candidates = [
            os.path.join(session_dir, "trajectories", "trajectories.h5"),
            os.path.join(session_dir, "trajectories", "without_gaps.h5"),
        ]
        path = None
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
        if path is None:
            for c in h5_candidates:
                if os.path.exists(c):
                    path = c
                    break
        if path is None:
            raise FileNotFoundError(
                f"No trajectory file found in {session_dir}/trajectories/. "
                f"Looked for: {candidates}"
            )
    else:
        raise ValueError("Provide either --session or --trajectories")

    print(f"  Loading trajectories from {path}")

    if path.endswith(".npy"):
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            # idtracker.ai stores a dict inside the npy
            data = data.item()
        if isinstance(data, dict):
            traj = data["trajectories"]
        else:
            traj = data
    elif path.endswith(".h5"):
        import h5py
        with h5py.File(path, "r") as f:
            traj = f["trajectories"][:]
    else:
        raise ValueError(f"Unsupported trajectory format: {path}")

    # Shape should be (N_frames, N_animals, 2)
    if traj.ndim != 3 or traj.shape[2] != 2:
        raise ValueError(
            f"Expected trajectory shape (frames, animals, 2), got {traj.shape}"
        )

    print(f"  Trajectories: {traj.shape[0]} frames, {traj.shape[1]} animals")
    return traj.astype(np.float64)


def find_clear_frame(traj: np.ndarray, video_path: str,
                     min_distance_frac: float = 0.15) -> Tuple[int, np.ndarray]:
    """Find a frame where all animals are well-separated and detected.

    Searches for a frame where:
    - All animals have valid (non-NaN) positions
    - The minimum pairwise distance is > min_distance_frac * frame_diagonal

    Returns (frame_index, frame_image_rgb).
    """
    cap = cv2.VideoCapture(video_path)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    diag = np.hypot(fw, fh)
    min_dist = diag * min_distance_frac
    n_frames = traj.shape[0]
    n_animals = traj.shape[1]

    # Score each frame: minimum pairwise distance (higher = better)
    best_frame = 0
    best_score = 0

    # Sample frames to avoid scanning the entire video
    sample_indices = np.linspace(n_frames // 4, n_frames - 1, 200, dtype=int)

    for fi in sample_indices:
        positions = traj[fi]  # (N_animals, 2)
        if np.any(np.isnan(positions)):
            continue
        # Minimum pairwise distance
        min_pair = np.inf
        for a in range(n_animals):
            for b in range(a + 1, n_animals):
                d = np.hypot(positions[a, 0] - positions[b, 0],
                             positions[a, 1] - positions[b, 1])
                min_pair = min(min_pair, d)
        if min_pair > best_score:
            best_score = min_pair
            best_frame = fi

    # Read that frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {best_frame} from {video_path}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(f"  Best frame for identity assignment: #{best_frame} "
          f"(min distance: {best_score:.0f} px)")
    return best_frame, frame_rgb


# ┌─────────────────────────────────────────────────────────────────────┐
# │  IDENTITY ASSIGNMENT GUI                                            │
# │  User clicks on each fly to assign male_teal, male_orange.          │
# │  Third fly is automatically assigned as female.                     │
# └─────────────────────────────────────────────────────────────────────┘
class IdentityAssigner:
    """OpenCV GUI for assigning identities to tracked animals.

    Shows a frame with animal positions marked. User clicks near each
    animal to assign identities in order: male_teal first, then male_orange.
    The remaining animal is auto-assigned as female.
    """

    WIN = "Identity Assignment"

    def __init__(self, frame_rgb: np.ndarray, positions: np.ndarray,
                 animal_indices: List[int]):
        """
        Args:
            frame_rgb: The frame image (RGB).
            positions: (N_animals, 2) array of centroid positions.
            animal_indices: list of idtracker animal indices (0-based).
        """
        self.frame = frame_rgb.copy()
        self.positions = positions
        self.animal_indices = list(animal_indices)
        self.assignments: Dict[str, int] = {}  # identity_name -> animal_index
        self._current_role = 0  # 0 = male_teal, 1 = male_orange
        self._done = False
        self._aborted = False

        h, w = frame_rgb.shape[:2]
        self.S = max(0.4, w / 1500.0)  # smaller scale for cropped arenas

        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        target_w = min(w, 1400)
        target_h = int(h * (target_w / w))
        cv2.resizeWindow(self.WIN, target_w, target_h)
        cv2.setMouseCallback(self.WIN, self._mouse_cb)

        self._render()
        self._loop()

    def _render(self):
        vis = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        h, w = vis.shape[:2]
        S = self.S
        dot_r = max(8, int(12 * S))
        font_scale = max(0.5, 0.7 * S)
        font_thick = max(1, int(2 * S))
        line_w = max(2, int(3 * S))

        # Draw all animals
        for i, aidx in enumerate(self.animal_indices):
            px, py = int(self.positions[aidx, 0]), int(self.positions[aidx, 1])

            # Determine colour based on assignment
            assigned_name = None
            for name, idx in self.assignments.items():
                if idx == aidx:
                    assigned_name = name
                    break

            if assigned_name:
                col = IDENTITY_COLOURS[assigned_name]
                label = assigned_name.replace("_", " ")
            else:
                col = (200, 200, 200)
                label = f"animal {aidx + 1}"

            # Circle around animal
            cv2.circle(vis, (px, py), int(40 * S), col, line_w)
            cv2.circle(vis, (px, py), dot_r, col, -1)

            # Label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale, font_thick)
            lx = px - tw // 2
            ly = py - int(50 * S)
            cv2.rectangle(vis, (lx - 4, ly - th - 4), (lx + tw + 4, ly + 4),
                          (0, 0, 0), -1)
            cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, col, font_thick)

        # Instructions
        if self._current_role < 2 and not self._done:
            role_name = IDENTITY_NAMES[self._current_role]
            role_col = IDENTITY_COLOURS[role_name]
            instructions = f"Click on the {role_name.replace('_', ' ').upper()}"
            inst_font = max(0.6, 0.9 * S)
            inst_thick = max(2, int(2.5 * S))
        elif self._done:
            instructions = "Assignment complete! Press Q to continue, R to redo"
            role_col = (0, 255, 0)
            inst_font = max(0.6, 0.8 * S)
            inst_thick = max(2, int(2.5 * S))
        else:
            instructions = ""
            role_col = (255, 255, 255)
            inst_font = max(0.6, 0.8 * S)
            inst_thick = max(2, int(2.5 * S))

        if instructions:
            # Bottom-centre instruction bar
            (iw, ih), _ = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX,
                                          inst_font, inst_thick)
            ix = (w - iw) // 2
            iy = h - int(30 * S)
            cv2.rectangle(vis, (ix - 10, iy - ih - 10), (ix + iw + 10, iy + 10),
                          (0, 0, 0), -1)
            cv2.putText(vis, instructions, (ix, iy),
                        cv2.FONT_HERSHEY_SIMPLEX, inst_font, role_col, inst_thick)

        cv2.imshow(self.WIN, vis)

    def _find_nearest_animal(self, x, y):
        """Find the unassigned animal closest to click position."""
        assigned_indices = set(self.assignments.values())
        best_dist = float("inf")
        best_idx = None
        for aidx in self.animal_indices:
            if aidx in assigned_indices:
                continue
            px, py = self.positions[aidx]
            d = np.hypot(x - px, y - py)
            if d < best_dist:
                best_dist = d
                best_idx = aidx
        return best_idx, best_dist

    def _mouse_cb(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self._done:
            return

        aidx, dist = self._find_nearest_animal(float(x), float(y))
        if aidx is None:
            return

        # Assign current role
        role = IDENTITY_NAMES[self._current_role]
        self.assignments[role] = aidx
        print(f"    {role} -> animal {aidx + 1} (idtracker index {aidx})")
        self._current_role += 1

        if self._current_role >= 2:
            # Auto-assign female as the remaining animal
            assigned = set(self.assignments.values())
            remaining = [i for i in self.animal_indices if i not in assigned]
            if remaining:
                self.assignments["female"] = remaining[0]
                print(f"    female -> animal {remaining[0] + 1} (auto-assigned)")
            self._done = True

        self._render()

    def _loop(self):
        while True:
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q") and self._done:
                break
            elif k == ord("r"):
                # Reset
                self.assignments.clear()
                self._current_role = 0
                self._done = False
                self._render()
            elif k == 27:  # Esc
                self._aborted = True
                break
        cv2.destroyAllWindows()

    def get_assignments(self) -> Optional[Dict[str, int]]:
        """Returns {identity_name: animal_index} or None if aborted."""
        if self._aborted or not self._done:
            return None
        return self.assignments


# ┌─────────────────────────────────────────────────────────────────────┐
# │  INDIVIDUAL CROP VIDEOS                                             │
# │  Centred crop around each tracked animal for SLEAP/DLC input.       │
# └─────────────────────────────────────────────────────────────────────┘
def create_individual_crops(
    video_path: str,
    traj: np.ndarray,
    assignments: Dict[str, int],
    output_dir: str,
    crop_size: int = INDIVIDUAL_CROP_SIZE,
    basename: str = "",
):
    """Create centred crop videos for each identified animal.

    For each frame, extracts a crop_size x crop_size window centred on
    the animal's centroid. Pads with white if the crop extends beyond
    frame boundaries. Writes one video per identity.

    Args:
        video_path: Path to the arena video.
        traj: Trajectory array (N_frames, N_animals, 2).
        assignments: {identity_name: animal_index} mapping.
        output_dir: Directory for output videos.
        crop_size: Side length of the square crop.
        basename: Prefix for output filenames.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    half = crop_size // 2

    # Open one writer per identity
    writers = {}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for name, aidx in assignments.items():
        out_path = os.path.join(output_dir, f"{basename}_{name}.mp4")
        writers[name] = cv2.VideoWriter(out_path, fourcc, fps,
                                        (crop_size, crop_size))
        print(f"    {name} (animal {aidx + 1}) -> {os.path.basename(out_path)}")

    frame_idx = 0
    last_positions = {name: None for name in assignments}
    t0 = time.monotonic()
    last_report = t0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for name, aidx in assignments.items():
            if frame_idx < traj.shape[0]:
                cx, cy = traj[frame_idx, aidx]
                if not np.isnan(cx):
                    last_positions[name] = (int(cx), int(cy))

            pos = last_positions[name]
            if pos is None:
                # No position yet — write white frame
                crop = np.full((crop_size, crop_size, 3), 255, dtype=np.uint8)
            else:
                px, py = pos
                # Extract crop with boundary padding
                x1 = px - half
                y1 = py - half
                x2 = x1 + crop_size
                y2 = y1 + crop_size

                # Source region (clamped to frame)
                sx1 = max(0, x1)
                sy1 = max(0, y1)
                sx2 = min(fw, x2)
                sy2 = min(fh, y2)

                # Destination offsets in the crop
                dx1 = sx1 - x1
                dy1 = sy1 - y1
                dx2 = dx1 + (sx2 - sx1)
                dy2 = dy1 + (sy2 - sy1)

                crop = np.full((crop_size, crop_size, 3), 255, dtype=np.uint8)
                if sx2 > sx1 and sy2 > sy1:
                    crop[dy1:dy2, dx1:dx2] = frame[sy1:sy2, sx1:sx2]

            writers[name].write(crop)

        frame_idx += 1
        now = time.monotonic()
        if now - last_report >= 5.0 and n_frames > 0:
            pct = frame_idx * 100 // n_frames
            print(f"    Individual crops: {pct}% ({frame_idx}/{n_frames})",
                  flush=True)
            last_report = now

    cap.release()
    for w in writers.values():
        w.release()

    elapsed = time.monotonic() - t0
    print(f"    Individual crops done in {elapsed:.0f}s")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  OVERLAY VIDEO                                                      │
# │  Teal and orange auras around the males for GameThogram scoring.    │
# └─────────────────────────────────────────────────────────────────────┘
def create_overlay_video(
    video_path: str,
    traj: np.ndarray,
    assignments: Dict[str, int],
    output_path: str,
    aura_alpha: float = AURA_ALPHA,
    aura_radius_mult: float = AURA_RADIUS_MULT,
):
    """Render an overlay video with coloured auras around identified animals.

    Males get teal/orange semi-transparent circles. The female gets a
    subtle grey marker. Output is suitable for GameThogram annotation.

    Args:
        video_path: Path to the arena video.
        traj: Trajectory array (N_frames, N_animals, 2).
        assignments: {identity_name: animal_index} mapping.
        output_path: Path for the output video.
        aura_alpha: Transparency of the coloured overlay.
        aura_radius_mult: Aura radius as multiple of estimated body length.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Estimate body length from trajectory data (median displacement between
    # consecutive frames when the animal is moving)
    body_lengths = []
    for aidx in assignments.values():
        diffs = np.diff(traj[:, aidx, :], axis=0)
        speeds = np.sqrt(np.nansum(diffs ** 2, axis=1))
        # Use 90th percentile of speeds as a rough body-length proxy
        valid = speeds[~np.isnan(speeds)]
        if len(valid) > 0:
            body_lengths.append(np.percentile(valid, 90))
    body_length = max(20, np.median(body_lengths) if body_lengths else 30)
    aura_r = int(body_length * aura_radius_mult)
    print(f"    Estimated body length: {body_length:.0f} px, aura radius: {aura_r} px")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))

    # Pre-compute a normalized colour array for multiply blend
    # In multiply mode: result = (frame * tint) / 255
    # Where tint = white everywhere except the aura regions which get the colour
    # We mix towards the colour: tint = lerp(white, colour, strength)
    # strength controls how saturated the tint is (0 = no effect, 1 = full multiply)
    MULTIPLY_STRENGTH = 0.45  # how strong the colour tint is

    frame_idx = 0
    last_positions = {name: None for name in assignments}
    t0 = time.monotonic()
    last_report = t0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Start with a white tint layer (no effect when multiplied)
        tint = np.full_like(frame, 255, dtype=np.float32)

        for name, aidx in assignments.items():
            if frame_idx < traj.shape[0]:
                cx, cy = traj[frame_idx, aidx]
                if not np.isnan(cx):
                    last_positions[name] = (int(cx), int(cy))

            pos = last_positions[name]
            if pos is None:
                continue

            col = IDENTITY_COLOURS[name]
            px, py = pos

            if name.startswith("male"):
                r = aura_r
            else:
                r = max(8, aura_r // 3)

            # Create a mask for this animal's aura region
            mask = np.zeros((fh, fw), dtype=np.float32)
            cv2.circle(mask, (px, py), r, 1.0, -1)

            # Apply a soft radial falloff for a nicer look
            # (optional: Gaussian blur the hard circle edge)
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=r * 0.3)
            # Renormalize so peak is 1.0
            peak = mask.max()
            if peak > 0:
                mask = mask / peak

            # Blend: tint pixel = lerp(255, colour, mask * strength)
            for c_idx in range(3):
                tint[:, :, c_idx] = np.where(
                    mask > 0.01,
                    np.minimum(tint[:, :, c_idx],
                               255.0 - mask * MULTIPLY_STRENGTH * (255.0 - col[c_idx])),
                    tint[:, :, c_idx],
                )

        # Multiply blend: result = frame * tint / 255
        frame_f = frame.astype(np.float32)
        result = (frame_f * tint) / 255.0
        frame = np.clip(result, 0, 255).astype(np.uint8)

        # Small identity labels near each animal
        for name, aidx in assignments.items():
            pos = last_positions[name]
            if pos is None:
                continue
            col = IDENTITY_COLOURS[name]
            px, py = pos
            short = "T" if name == "male_teal" else ("O" if name == "male_orange" else "F")
            cv2.putText(frame, short, (px + 5, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        writer.write(frame)
        frame_idx += 1

        now = time.monotonic()
        if now - last_report >= 5.0 and n_frames > 0:
            pct = frame_idx * 100 // n_frames
            print(f"    Overlay: {pct}% ({frame_idx}/{n_frames})", flush=True)
            last_report = now

    cap.release()
    writer.release()
    elapsed = time.monotonic() - t0
    sz = os.path.getsize(output_path)
    sz_str = f"{sz / 1e6:.1f} MB" if sz < 1e9 else f"{sz / 1e9:.1f} GB"
    print(f"    Overlay video done in {elapsed:.0f}s -> {sz_str}")


# ┌─────────────────────────────────────────────────────────────────────┐
# │  CLI + MAIN                                                         │
# └─────────────────────────────────────────────────────────────────────┘
def parse_args():
    p = argparse.ArgumentParser(
        prog="thermokourt-posttrack",
        description="Post-tracking pipeline: assign identities, crop individuals, "
                    "render overlay videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video", help="Arena video file (cropped by arena_extractor).")
    p.add_argument("--session", default=None,
                   help="idtracker.ai session folder (contains trajectories/).")
    p.add_argument("--trajectories", default=None,
                   help="Direct path to trajectory file (.npy or .h5).")
    p.add_argument("-o", "--output_dir", default=None,
                   help="Output directory (default: same as video).")
    p.add_argument("--skip_gui", action="store_true",
                   help="Skip identity assignment GUI (reuse saved JSON).")
    p.add_argument("--identity_json", default=None,
                   help="Path to identity assignment JSON (auto-detected if not given).")
    p.add_argument("--crop_size", type=int, default=INDIVIDUAL_CROP_SIZE,
                   help=f"Individual crop size in pixels (default: {INDIVIDUAL_CROP_SIZE}).")
    p.add_argument("--no_crops", action="store_true",
                   help="Skip individual crop video generation.")
    p.add_argument("--no_overlay", action="store_true",
                   help="Skip overlay video generation.")
    return p.parse_args()


def main():
    args = parse_args()
    video_path = os.path.abspath(args.video)
    video_dir = os.path.dirname(video_path)
    video_stem = Path(video_path).stem
    output_dir = args.output_dir or video_dir

    print("+" + "-" * 50 + "+")
    print("|  THERMOKOURT POST-TRACKING PIPELINE v1.0        |")
    print("|  assign. crop. overlay. science.                 |")
    print("+" + "-" * 50 + "+")
    print(f"  Video:  {video_path}")
    print(f"  Output: {output_dir}")

    # ── Load trajectories ────────────────────────────────────────────
    traj = load_trajectories(
        session_dir=args.session or "",
        traj_path=args.trajectories or "",
    )
    n_animals = traj.shape[1]
    if n_animals != 3:
        print(f"  WARNING: Expected 3 animals, got {n_animals}. "
              f"Proceeding but identity assignment assumes 3.")

    # ── Identity assignment ──────────────────────────────────────────
    id_json_path = args.identity_json or os.path.join(
        output_dir, f"{video_stem}_identities.json"
    )

    if args.skip_gui and os.path.exists(id_json_path):
        print(f"  Loading identity assignment from {id_json_path}")
        with open(id_json_path) as f:
            assignments = json.load(f)
        # Convert string keys to proper format, values to int
        assignments = {k: int(v) for k, v in assignments.items()}
        for name, aidx in assignments.items():
            print(f"    {name} -> animal {aidx + 1}")
    else:
        print("\n  === IDENTITY ASSIGNMENT ===")
        print("  Finding a clear frame with all flies separated...")
        frame_idx, frame_rgb = find_clear_frame(traj, video_path)
        positions = traj[frame_idx]

        animal_indices = list(range(n_animals))
        assigner = IdentityAssigner(frame_rgb, positions, animal_indices)
        assignments = assigner.get_assignments()

        if assignments is None:
            print("\n  Identity assignment aborted.")
            sys.exit(0)

        # Save assignment
        os.makedirs(output_dir, exist_ok=True)
        with open(id_json_path, "w") as f:
            json.dump(assignments, f, indent=2)
        print(f"  Identities saved -> {id_json_path}")

    # ── Individual crop videos ───────────────────────────────────────
    if not args.no_crops:
        print(f"\n  === INDIVIDUAL CROP VIDEOS ({args.crop_size}x{args.crop_size}) ===")
        crops_dir = os.path.join(output_dir, "individual_crops")
        create_individual_crops(
            video_path, traj, assignments, crops_dir,
            crop_size=args.crop_size, basename=video_stem,
        )

    # ── Overlay video ────────────────────────────────────────────────
    if not args.no_overlay:
        print("\n  === OVERLAY VIDEO ===")
        overlay_path = os.path.join(output_dir, f"{video_stem}_overlay.mp4")
        create_overlay_video(video_path, traj, assignments, overlay_path)

    print(f"\n  Done! Outputs in {output_dir}")


if __name__ == "__main__":
    main()