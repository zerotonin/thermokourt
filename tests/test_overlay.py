import shutil
import subprocess

import numpy as np
import pytest

from thermokourt.overlay.render import ROLE_BGR, Event, active_events, render_clip
from thermokourt.pose.analysis import Pose


def test_event_label_and_activity():
    e = Event(1.0, 2.0, "lunge", "male_2", "male_1")
    assert e.label() == "lunge: male_2 -> male_1"
    assert active_events([e], 1.5) == [e] and active_events([e], 2.5) == []


def test_role_colours_are_wong():
    assert ROLE_BGR["male_1"] == (178, 114, 0)  # blue
    assert ROLE_BGR["male_2"] == (115, 158, 0)  # bluish green
    assert ROLE_BGR["female_decap"] == (0, 159, 230)  # orange


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")
def test_render_clip_keeps_every_frame_at_the_real_rate(tmp_path):
    clip = tmp_path / "a.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=white:size=64x64:rate=69",
            "-frames:v",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            str(clip),
        ],
        check=True,
    )
    tracks = np.full((1, 30, 2, 2), np.nan)
    tracks[0, :, 0] = [20, 20]
    tracks[0, :, 1] = [30, 30]
    pose = Pose(tracks=tracks, node_names=["head", "thorax"], fps=69.0)
    out = render_clip(
        clip,
        tmp_path / "o.mp4",
        pose=pose,
        roles={0: "male_1"},
        edges=[(1, 0)],
        events=[Event(0.1, 0.2, "lunge", "male_1", None)],
        scale=2,
    )
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=width,nb_read_frames,r_frame_rate",
            "-of",
            "csv=p=0",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert r.stdout.strip() == "128,69/1,30"
