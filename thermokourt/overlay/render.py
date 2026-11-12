# ╔══════════════════════════════════════════════════════════════════╗
# ║  ThermoKourt — overlay.render                                    ║
# ║  « see what the pose model and the rules saw »                   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Draws the SLEAP skeletons (coloured by role) and the scored     ║
# ║  behaviour events onto an arena clip. Frames are piped to ffmpeg ║
# ║  (libx264), so any frame rate works.                             ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Annotated arena clips: skeletons and behaviour events."""

from __future__ import annotations

import sqlite3
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np

from thermokourt.pose.analysis import THORAX, Pose, assign_roles, load_sleap_h5

# Wong (2011) colours, BGR for OpenCV
ROLE_BGR = {
    "male_1": (178, 114, 0),  # Wong blue
    "male_2": (115, 158, 0),  # Wong bluish green
    "female_decap": (0, 159, 230),  # Wong orange
}
UNKNOWN_BGR = (150, 150, 150)
TEXT_BG = (0, 0, 0)
TEXT_FG = (255, 255, 255)
DEFAULT_SCALE = 2  # clips are 224-350 px; text needs room
CRF = 20
PROGRESS_EVERY = 250  # frames


@dataclass(frozen=True)
class Event:
    start_s: float
    end_s: float
    behaviour: str
    actor: str | None
    target: str | None

    def label(self) -> str:
        who = self.actor or "?"
        if self.target:
            who += f" -> {self.target}"
        return f"{self.behaviour}: {who}"


def load_events(db: Path, trial_id: str) -> list[Event]:
    """Scored events of one trial with actor/target roles."""
    con = sqlite3.connect(db)
    rows = con.execute(
        """SELECT e.start_s, e.end_s, b.name, fa.role, ft.role
           FROM behaviour_events e
           JOIN behaviours b ON b.behaviour_id = e.behaviour_id
           LEFT JOIN flies fa ON fa.fly_id = e.actor_fly_id
           LEFT JOIN flies ft ON ft.fly_id = e.target_fly_id
           WHERE e.trial_id = ? ORDER BY e.start_s""",
        (trial_id,),
    ).fetchall()
    con.close()
    return [Event(float(a), float(b), c, d, e) for a, b, c, d, e in rows]


def load_roles(db: Path, trial_id: str) -> dict[int, str]:
    """track index -> role, as stored by scoring (``flies.track_id``)."""
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT track_id, role FROM flies WHERE trial_id = ? AND track_id IS NOT NULL",
        (trial_id,),
    ).fetchall()
    con.close()
    return {int(t): r for t, r in rows}


def load_edges(h5: Path) -> list[tuple[int, int]]:
    with h5py.File(h5, "r") as f:
        return [tuple(map(int, e)) for e in f["edge_inds"][:]] if "edge_inds" in f else []


def active_events(events: list[Event], t: float) -> list[Event]:
    return [e for e in events if e.start_s <= t <= e.end_s]


def _probe(clip: Path) -> tuple[int, int, float, int]:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-of",
            "csv=p=0",
            str(clip),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    w, h, rate, n = (r.stdout.strip().split(",") + ["0"])[:4]
    num, _, den = rate.partition("/")
    return int(w), int(h), float(num) / float(den or 1), int(n) if n.isdigit() else 0


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float, marker=None) -> int:
    """White text on a black box, optionally led by a coloured square; returns height."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, 1)
    x, y = org
    pad = th + 6 if marker is not None else 0
    cv2.rectangle(img, (x - 3, y - th - 4), (x + pad + tw + 3, y + base + 1), TEXT_BG, -1)
    if marker is not None:
        cv2.rectangle(img, (x, y - th), (x + th, y), marker, -1)
    cv2.putText(img, text, (x + pad, y), font, scale, TEXT_FG, 1, cv2.LINE_AA)
    return th + base + 6


def draw_skeletons(
    img, pose: Pose, frame: int, roles: dict[int, str], edges, scale: float, highlight: set[str]
):
    th = pose.node(THORAX) if THORAX in pose.node_names else 0
    for track in range(pose.n_tracks):
        pts = pose.tracks[track, frame] * scale  # (n_nodes, 2)
        role = roles.get(track)
        colour = ROLE_BGR.get(role, UNKNOWN_BGR)
        for a, b in edges:
            if not (np.isnan(pts[a]).any() or np.isnan(pts[b]).any()):
                cv2.line(
                    img, tuple(np.int32(pts[a])), tuple(np.int32(pts[b])), colour, 2, cv2.LINE_AA
                )
        for p in pts:
            if not np.isnan(p).any():
                cv2.circle(img, tuple(np.int32(p)), 3, colour, -1, cv2.LINE_AA)
        if not np.isnan(pts[th]).any():
            c = tuple(np.int32(pts[th]))
            if role in highlight:
                cv2.circle(img, c, int(14 * scale), colour, 2, cv2.LINE_AA)
            if role:
                _text(img, role.replace("_decap", ""), (c[0] + 10, c[1] - 10), 0.45, colour)


def mark_actors(
    img, pose: Pose | None, frame: int, roles: dict[int, str], scale: float, highlight: set[str]
):
    """Events without skeletons: a ring around each fly taking part."""
    if pose is None or THORAX not in pose.node_names:
        return
    th = pose.node(THORAX)
    for track, role in roles.items():
        if role in highlight and track < pose.n_tracks:
            p = pose.tracks[track, frame, th] * scale
            if not np.isnan(p).any():
                cv2.circle(
                    img,
                    tuple(np.int32(p)),
                    int(14 * scale),
                    ROLE_BGR.get(role, UNKNOWN_BGR),
                    2,
                    cv2.LINE_AA,
                )


def render_clip(
    clip: Path,
    out: Path,
    *,
    pose: Pose | None = None,
    roles: dict[int, str] | None = None,
    edges=(),
    events: list[Event] = (),
    skeleton: bool = True,
    show_events: bool = True,
    scale: int = DEFAULT_SCALE,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Write ``out``: ``clip`` enlarged ``scale``× with skeletons and/or events drawn."""
    w, h, fps, n_frames = _probe(clip)
    W, H = w * scale, h * scale
    roles = roles or {}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.mp4")
    enc = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{W}x{H}",
            "-r",
            f"{fps:g}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(CRF),
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(tmp),
        ],
        stdin=subprocess.PIPE,
    )
    cap = cv2.VideoCapture(str(clip))
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            img = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
            t = i / fps
            now = active_events(events, t) if show_events else []
            busy = {r for e in now for r in (e.actor, e.target) if r}
            if skeleton and pose is not None and i < pose.n_frames:
                draw_skeletons(img, pose, i, roles, edges, scale, busy)
            elif show_events and pose is not None and i < pose.n_frames:
                mark_actors(img, pose, i, roles, scale, busy)
            if show_events:
                _text(img, f"t {t:7.2f} s", (6, H - 8), 0.5)
                y = 22
                for e in now[:4]:
                    y += _text(img, e.label(), (6, y), 0.55, ROLE_BGR.get(e.actor, UNKNOWN_BGR))
            enc.stdin.write(img.tobytes())
            i += 1
            if progress and (i % PROGRESS_EVERY == 0):
                progress(i, n_frames)
    finally:
        cap.release()
        enc.stdin.close()
        rc = enc.wait()
    if rc != 0 or i == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"rendering {clip.name} failed (ffmpeg exit {rc}, {i} frames)")
    tmp.replace(out)
    if progress:
        progress(n_frames or i, n_frames or i)
    return out


def render_trial_clip(
    clip: Path,
    out: Path,
    h5: Path | None,
    db: Path,
    trial_id: str,
    fps: float,
    *,
    skeleton: bool,
    show_events: bool,
    n_animals: int = 3,
    scale: int = DEFAULT_SCALE,
    progress=None,
) -> Path:
    """Everything for one arena: pose and roles (as scored), events from the DB."""
    pose = load_sleap_h5(str(h5), fps) if h5 and Path(h5).is_file() else None
    roles = load_roles(db, trial_id) if pose is not None else {}
    if pose is not None and not roles:
        roles = {t: r for r, t in assign_roles(pose, n_animals).items()}
    events = load_events(db, trial_id) if show_events else []
    edges = load_edges(h5) if pose is not None else []
    return render_clip(
        clip,
        out,
        pose=pose,
        roles=roles,
        edges=edges,
        events=events,
        skeleton=skeleton,
        show_events=show_events,
        scale=scale,
        progress=progress,
    )
