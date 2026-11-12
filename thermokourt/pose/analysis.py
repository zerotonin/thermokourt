"""Pose loading, role assignment, per-frame features and rule-based events."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Skeleton node names we rely on (flies13 naming).
HEAD, THORAX, ABDOMEN, WING_L, WING_R = "head", "thorax", "abdomen", "wingL", "wingR"
ROLES = ("male_1", "male_2", "female_decap")


@dataclass
class Pose:
    """Tracks as (n_tracks, n_frames, n_nodes, 2) in pixels; NaN = missing."""

    tracks: np.ndarray
    node_names: list[str]
    fps: float
    scores: np.ndarray | None = None  # (n_tracks, n_frames, n_nodes)

    @property
    def n_tracks(self) -> int:
        return self.tracks.shape[0]

    @property
    def n_frames(self) -> int:
        return self.tracks.shape[1]

    def node(self, name: str) -> int:
        return self.node_names.index(name)

    def xy(self, track: int, name: str) -> np.ndarray:
        """(n_frames, 2) coordinates of one node of one track."""
        return self.tracks[track, :, self.node(name), :]


def load_sleap_h5(path: str, fps: float) -> Pose:
    """Read a SLEAP analysis HDF5 (``tracks`` is stored n_tracks×2×n_nodes×n_frames)."""
    import h5py

    with h5py.File(path, "r") as f:
        t = f["tracks"][:]  # (n_tracks, 2, n_nodes, n_frames)
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["node_names"][:]]
        scores = (
            f["point_scores"][:] if "point_scores" in f else None
        )  # (n_tracks, n_nodes, n_frames)
    tracks = np.transpose(t, (0, 3, 2, 1)).astype(float)
    sc = np.transpose(scores, (0, 2, 1)).astype(float) if scores is not None else None
    return Pose(tracks=tracks, node_names=names, fps=fps, scores=sc)


# ── helpers ────────────────────────────────────────────────────────────


def _interp_nan(a: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly fill NaN runs up to ``max_gap`` frames (per column)."""
    a = a.copy()
    n = a.shape[0]
    idx = np.arange(n)
    for c in range(a.shape[1]):
        col = a[:, c]
        ok = ~np.isnan(col)
        if ok.sum() < 2:
            continue
        filled = np.interp(idx, idx[ok], col[ok])
        # re-mask gaps longer than max_gap
        bad = ~ok
        start = None
        for i in range(n + 1):
            if i < n and bad[i]:
                start = i if start is None else start
            elif start is not None:
                if i - start > max_gap:
                    filled[start:i] = np.nan
                start = None
        a[:, c] = filled
    return a


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Unsigned angle in degrees between row vectors (NaN where undefined)."""
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(v1 * v2, axis=1) / (n1 * n2)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(int))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts, ends, strict=True))


# ── roles ──────────────────────────────────────────────────────────────


def assign_roles(pose: Pose, n_expected: int = 3) -> dict[str, int]:
    """Map roles to track indices.

    The decapitated female is the track that moves least; the two males are
    the most-occupied remaining tracks (``male_1`` = more occupied). With two
    tracks only (E2 two-male trials) no female is assigned.
    """
    th = pose.node(THORAX)
    occupancy = np.sum(~np.isnan(pose.tracks[:, :, th, 0]), axis=1)
    order = np.argsort(-occupancy)[: min(n_expected, pose.n_tracks)]
    if len(order) == 0:
        return {}
    travel = []
    for t in order:
        xy = pose.tracks[t, :, th, :]
        d = np.nansum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
        travel.append(d)
    roles: dict[str, int] = {}
    cand = list(order)
    if len(cand) >= 3:
        fem = cand[int(np.argmin(travel))]
        roles["female_decap"] = int(fem)
        cand = [c for c in cand if c != fem]
    males = sorted(cand, key=lambda t: -occupancy[t])[:2]
    for role, t in zip(("male_1", "male_2"), males, strict=False):
        roles[role] = int(t)
    return roles


# ── features ───────────────────────────────────────────────────────────


@dataclass
class Features:
    """Per-frame features for one male, in body lengths (BL) and seconds."""

    role: str
    t_s: np.ndarray
    centroid: np.ndarray  # (n, 2) px
    body_len_px: float
    speed_bl_s: np.ndarray  # scalar speed
    heading: np.ndarray  # unit vector thorax->head (n, 2)
    wing_l_deg: np.ndarray
    wing_r_deg: np.ndarray
    dist_other_male_bl: np.ndarray
    dist_female_bl: np.ndarray
    approach_other_bl_s: np.ndarray  # + = closing in on the other male
    facing_other_deg: np.ndarray  # angle between heading and vector to other male
    other_speed_bl_s: np.ndarray
    valid: np.ndarray = field(default=None)


def compute_features(
    pose: Pose, roles: dict[str, int], max_gap_frames: int = 5
) -> dict[str, Features]:
    """Features for each male present in ``roles``."""
    fps = pose.fps
    n = pose.n_frames
    t_s = np.arange(n) / fps

    def node_xy(track, name):
        return _interp_nan(pose.xy(track, name), max_gap_frames)

    males = [r for r in ("male_1", "male_2") if r in roles]
    female = roles.get("female_decap")
    fem_xy = node_xy(female, THORAX) if female is not None else None

    out: dict[str, Features] = {}
    for role in males:
        tr = roles[role]
        head, thorax, abd = node_xy(tr, HEAD), node_xy(tr, THORAX), node_xy(tr, ABDOMEN)
        wl, wr = node_xy(tr, WING_L), node_xy(tr, WING_R)
        axis = head - abd
        bl = float(np.nanmedian(np.linalg.norm(axis, axis=1)))
        if not np.isfinite(bl) or bl <= 0:
            bl = 1.0
        heading = head - thorax
        hn = np.linalg.norm(heading, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            heading = heading / hn
        back = thorax - head  # body axis pointing backwards; wings fold along it
        wing_l = _angle_between(wl - thorax, back)
        wing_r = _angle_between(wr - thorax, back)
        vel = np.gradient(thorax, axis=0) * fps  # px/s
        speed = np.linalg.norm(vel, axis=1) / bl

        other = [r for r in males if r != role]
        if other:
            o_th = node_xy(roles[other[0]], THORAX)
            to_other = o_th - thorax
            dist_o = np.linalg.norm(to_other, axis=1) / bl
            approach = -np.gradient(dist_o) * fps  # BL/s, positive when closing
            facing = _angle_between(heading, to_other)
            o_speed = np.linalg.norm(np.gradient(o_th, axis=0) * fps, axis=1) / bl
        else:
            dist_o = np.full(n, np.nan)
            approach = np.full(n, np.nan)
            facing = np.full(n, np.nan)
            o_speed = np.full(n, np.nan)
        dist_f = (
            np.linalg.norm(fem_xy - thorax, axis=1) / bl
            if fem_xy is not None
            else np.full(n, np.nan)
        )
        valid = ~np.isnan(thorax[:, 0]) & ~np.isnan(head[:, 0])
        out[role] = Features(
            role=role,
            t_s=t_s,
            centroid=thorax,
            body_len_px=bl,
            speed_bl_s=speed,
            heading=heading,
            wing_l_deg=wing_l,
            wing_r_deg=wing_r,
            dist_other_male_bl=dist_o,
            dist_female_bl=dist_f,
            approach_other_bl_s=approach,
            facing_other_deg=facing,
            other_speed_bl_s=o_speed,
            valid=valid,
        )
    return out


# ── rule-based events ──────────────────────────────────────────────────


@dataclass
class RuleParams:
    """Thresholds for rules-v0. Tune in the pilot; keep the version string."""

    version: str = "v0"
    wing_ext_min_deg: float = 45.0  # extended wing
    wing_ext_other_max_deg: float = 25.0  # the other wing stays folded
    wing_ext_min_s: float = 0.2
    wing_threat_min_deg: float = 30.0  # both wings
    wing_threat_max_dist_bl: float = 2.5
    wing_threat_max_facing_deg: float = 60.0
    wing_threat_min_s: float = 0.1
    lunge_min_speed_bl_s: float = 4.0
    lunge_min_approach_bl_s: float = 2.0
    lunge_max_dist_bl: float = 1.5
    lunge_max_s: float = 0.4
    lunge_min_s: float = 0.04
    follow_max_dist_bl: float = 2.0
    follow_max_facing_deg: float = 45.0
    follow_min_other_speed_bl_s: float = 0.5
    follow_min_s: float = 1.0
    merge_gap_s: float = 0.1  # join events separated by less than this


@dataclass
class Event:
    behaviour: str
    actor: str
    target: str | None
    start_s: float
    end_s: float
    confidence: float = 1.0


def _events_from_mask(mask, t_s, fps, behaviour, actor, target, min_s, max_s=None, merge_gap_s=0.0):
    runs = _runs(np.nan_to_num(mask, nan=False))
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and (s - merged[-1][1]) / fps <= merge_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        dur = (e - s) / fps
        if dur < min_s or (max_s is not None and dur > max_s):
            continue
        out.append(
            Event(
                behaviour, actor, target, float(t_s[s]), float(t_s[min(e, len(t_s) - 1)]) + 1 / fps
            )
        )
    return out


def detect_events(
    feats: dict[str, Features], fps: float, p: RuleParams | None = None
) -> list[Event]:
    p = p or RuleParams()
    events: list[Event] = []
    for role, f in feats.items():
        other = next((r for r in feats if r != role), None)
        with np.errstate(invalid="ignore"):
            ext_l = (f.wing_l_deg >= p.wing_ext_min_deg) & (
                f.wing_r_deg <= p.wing_ext_other_max_deg
            )
            ext_r = (f.wing_r_deg >= p.wing_ext_min_deg) & (
                f.wing_l_deg <= p.wing_ext_other_max_deg
            )
            threat = (
                (f.wing_l_deg >= p.wing_threat_min_deg)
                & (f.wing_r_deg >= p.wing_threat_min_deg)
                & (f.dist_other_male_bl <= p.wing_threat_max_dist_bl)
                & (f.facing_other_deg <= p.wing_threat_max_facing_deg)
            )
            lunge = (
                (f.speed_bl_s >= p.lunge_min_speed_bl_s)
                & (f.approach_other_bl_s >= p.lunge_min_approach_bl_s)
                & (f.dist_other_male_bl <= p.lunge_max_dist_bl)
            )
            follow = (
                (f.dist_other_male_bl <= p.follow_max_dist_bl)
                & (f.facing_other_deg <= p.follow_max_facing_deg)
                & (f.other_speed_bl_s >= p.follow_min_other_speed_bl_s)
                & (f.speed_bl_s >= p.follow_min_other_speed_bl_s * 0.5)
            )
        # courtship wing extension is directed at the female when she is near, else at the male
        events += _events_from_mask(
            (ext_l | ext_r) & f.valid,
            f.t_s,
            fps,
            "wing_extension",
            role,
            None,
            p.wing_ext_min_s,
            merge_gap_s=p.merge_gap_s,
        )
        events += _events_from_mask(
            threat & f.valid,
            f.t_s,
            fps,
            "wing_threat",
            role,
            other,
            p.wing_threat_min_s,
            merge_gap_s=p.merge_gap_s,
        )
        events += _events_from_mask(
            lunge & f.valid, f.t_s, fps, "lunge", role, other, p.lunge_min_s, p.lunge_max_s
        )
        events += _events_from_mask(
            follow & f.valid,
            f.t_s,
            fps,
            "following",
            role,
            other,
            p.follow_min_s,
            merge_gap_s=p.merge_gap_s * 5,
        )
    events.sort(key=lambda e: (e.start_s, e.actor))
    return events


# ── per-window summary ─────────────────────────────────────────────────


def window_summary(
    feats: dict[str, Features],
    fps: float,
    mm_per_pixel: float | None,
    window_s: float = 60.0,
    encounter_bl: float = 1.5,
) -> list[dict]:
    """Speed, distance, tracked fraction and encounter count per male per window."""
    rows = []
    for role, f in feats.items():
        n = len(f.t_s)
        w = int(round(window_s * fps))
        px_to_mm = mm_per_pixel if mm_per_pixel else np.nan
        step = np.linalg.norm(np.diff(f.centroid, axis=0), axis=1)  # px per frame
        close = np.nan_to_num(f.dist_other_male_bl <= encounter_bl, nan=False)
        enc_starts = np.array([s for s, _ in _runs(close)])
        for start in range(0, n, w):
            sl = slice(start, min(start + w, n))
            st = step[max(sl.start - 1, 0) : sl.stop - 1]
            dist_px = float(np.nansum(st))
            dur = (sl.stop - sl.start) / fps
            rows.append(
                {
                    "role": role,
                    "window_start_s": float(f.t_s[sl.start]),
                    "window_s": float(dur),
                    "speed_mm_s": dist_px * px_to_mm / dur if dur > 0 else np.nan,
                    "distance_mm": dist_px * px_to_mm,
                    "frac_tracked": float(np.mean(f.valid[sl])),
                    "n_encounters": int(np.sum((enc_starts >= sl.start) & (enc_starts < sl.stop))),
                }
            )
    return rows
