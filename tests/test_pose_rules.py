"""Synthetic-pose tests for role assignment, features and rules-v0."""

from __future__ import annotations

import numpy as np
import pytest

from thermokourt.pose.analysis import (
    Pose,
    RuleParams,
    assign_roles,
    compute_features,
    detect_events,
    window_summary,
)

NODES = ["head", "thorax", "abdomen", "wingL", "wingR"]
FPS = 25.0
BL = 40.0  # px


def fly(n, x, y, heading_deg=0.0, wing_l=10.0, wing_r=10.0):
    """(n, 5, 2) node coordinates for a fly at (x, y) arrays, facing heading_deg."""
    x, y = np.broadcast_to(x, (n,)).astype(float), np.broadcast_to(y, (n,)).astype(float)
    h = np.deg2rad(np.broadcast_to(heading_deg, (n,)).astype(float))
    fwd = np.stack([np.cos(h), np.sin(h)], 1)
    back = -fwd
    out = np.full((n, 5, 2), np.nan)
    thorax = np.stack([x, y], 1)
    out[:, 1] = thorax
    out[:, 0] = thorax + fwd * BL * 0.5
    out[:, 2] = thorax + back * BL * 0.5
    for i, (ang, sign) in enumerate(((wing_l, 1), (wing_r, -1)), start=3):
        a = np.deg2rad(np.broadcast_to(ang, (n,)).astype(float)) * sign
        # wing vector = backwards axis rotated by ±angle
        rot = np.stack(
            [
                np.cos(a) * back[:, 0] - np.sin(a) * back[:, 1],
                np.sin(a) * back[:, 0] + np.cos(a) * back[:, 1],
            ],
            1,
        )
        out[:, i] = thorax + rot * BL * 0.6
    return out


def make_pose(*flies):
    return Pose(tracks=np.stack(flies, 0), node_names=NODES, fps=FPS)


def test_roles_pick_stationary_female():
    n = 200
    t = np.arange(n)
    m1 = fly(n, 100 + t, 100)
    m2 = fly(n, 300 - t * 0.5, 200)
    fem = fly(n, 250, 250)
    roles = assign_roles(make_pose(m1, m2, fem))
    assert roles["female_decap"] == 2
    assert {roles["male_1"], roles["male_2"]} == {0, 1}


def test_wing_extension_detected_once():
    n = 250
    wl = np.full(n, 10.0)
    wl[100:130] = 60.0  # 1.2 s unilateral extension
    d = np.arange(n) * 0.2  # slow drift keeps the males distinguishable from the female
    m1 = fly(n, 100 + d, 100, wing_l=wl)
    m2 = fly(n, 400 + d, 400)
    fem = fly(n, 120, 100)
    pose = make_pose(m1, m2, fem)
    roles = assign_roles(pose)
    assert roles == {"female_decap": 2, "male_1": 0, "male_2": 1}
    feats = compute_features(pose, roles)
    assert feats["male_1"].wing_l_deg[110] == pytest.approx(60.0, abs=1.0)
    assert feats["male_1"].wing_r_deg[110] == pytest.approx(10.0, abs=1.0)
    ev = [e for e in detect_events(feats, FPS) if e.behaviour == "wing_extension"]
    assert len(ev) == 1
    assert ev[0].actor == "male_1"
    assert ev[0].start_s == pytest.approx(100 / FPS, abs=1 / FPS)
    assert ev[0].end_s - ev[0].start_s == pytest.approx(30 / FPS, abs=2 / FPS)


def test_lunge_is_a_fast_burst_toward_the_other_male():
    n = 200
    d = np.arange(n) * 0.2
    x = 100 + d
    # male_1 jumps ~1 BL toward male_2 (at x=160, same y) over 4 frames at t=50
    x[50:54] += np.array([10, 20, 30, 38])
    x[54:] += 38
    m1 = fly(n, x, 100, heading_deg=0)
    m2 = fly(n, 160 + d, 100, heading_deg=180)
    fem = fly(n, 300, 300)
    pose = make_pose(m1, m2, fem)
    feats = compute_features(pose, assign_roles(pose))
    ev = [e for e in detect_events(feats, FPS) if e.behaviour == "lunge"]
    assert len(ev) == 1 and ev[0].actor == "male_1" and ev[0].target == "male_2"
    assert 49 / FPS <= ev[0].start_s <= 52 / FPS


def test_wing_threat_requires_facing_and_both_wings():
    n = 100
    d = np.arange(n) * 0.2
    m1 = fly(n, 100 + d, 100, heading_deg=0, wing_l=40, wing_r=40)  # faces +x, male_2 is at +x
    m2 = fly(n, 150 + d, 100, heading_deg=180)
    fem = fly(n, 300, 300)
    pose = make_pose(m1, m2, fem)
    feats = compute_features(pose, assign_roles(pose))
    ev = detect_events(feats, FPS)
    kinds = {(e.behaviour, e.actor) for e in ev}
    assert ("wing_threat", "male_1") in kinds
    assert ("wing_threat", "male_2") not in kinds
    assert ("wing_extension", "male_1") not in kinds  # both wings up is not a courtship extension


def test_following_and_windows():
    n = 500
    t = np.arange(n)
    m2 = fly(n, 100 + t * 1.0, 100, heading_deg=0)  # walks +x at 1 px/frame
    m1 = fly(n, 60 + t * 1.0, 100, heading_deg=0)  # 1 BL behind, same speed
    fem = fly(n, 300, 300)
    pose = make_pose(m1, m2, fem)
    roles = assign_roles(pose)
    feats = compute_features(pose, roles)
    ev = [e for e in detect_events(feats, FPS) if e.behaviour == "following"]
    assert ev and ev[0].end_s - ev[0].start_s > 10
    rows = window_summary(feats, FPS, mm_per_pixel=0.05, window_s=10)
    r = [x for x in rows if x["role"] == "male_1"]
    assert len(r) == 2
    assert r[0]["frac_tracked"] == 1.0
    assert r[0]["speed_mm_s"] == pytest.approx(1.0 * FPS * 0.05, rel=0.05)


def test_nan_gaps_are_bridged_but_long_gaps_stay_invalid():
    n = 100
    d = np.arange(n) * 0.2
    m1 = fly(n, 100 + d, 100)
    m1[20:23] = np.nan  # 3-frame dropout -> interpolated
    m1[50:70] = np.nan  # 20-frame dropout -> stays invalid
    m2 = fly(n, 300 + d, 300)
    fem = fly(n, 200, 200)
    pose = make_pose(m1, m2, fem)
    roles = assign_roles(pose)
    role_of_track0 = next(r for r, t in roles.items() if t == 0)  # the gappy male
    feats = compute_features(pose, roles, max_gap_frames=5)
    f = feats[role_of_track0]
    assert f.valid[21] and not f.valid[60]


def test_rule_params_have_a_version():
    assert RuleParams().version == "v0"
