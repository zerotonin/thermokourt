"""Basic tests for ThermoKourt arena extraction."""

import json

from thermokourt.extract.arena_extractor import (
    Arena,
    sort_arenas_row_major,
)


class TestArena:
    def test_bbox(self):
        # Use padding=0.0 so the bbox is the raw circle bounding box;
        # the default CROP_PADDING is exercised in test_bbox_with_padding.
        a = Arena(cx=100.0, cy=100.0, r=50.0, idx=0)
        x, y, w, h = a.bbox(padding=0.0)
        assert x == 50
        assert y == 50
        assert w == 100
        assert h == 100

    def test_bbox_with_padding(self):
        # Default padding adds CROP_PADDING * r to each side.
        a = Arena(cx=100.0, cy=100.0, r=50.0, idx=0)
        x, y, w, h = a.bbox()
        # Symmetric padding, even-sized result.
        assert w == h
        assert w >= 100
        assert w % 2 == 0

    def test_bbox_clamps_negative(self):
        a = Arena(cx=10.0, cy=10.0, r=50.0, idx=0)
        x, y, w, h = a.bbox()
        assert x == 0
        assert y == 0

    def test_to_dict_roundtrip(self):
        a = Arena(cx=123.4, cy=567.8, r=90.1, idx=3)
        d = a.to_dict()
        a2 = Arena.from_dict(d)
        assert a2.cx == a.cx
        assert a2.cy == a.cy
        assert a2.r == a.r
        assert a2.idx == a.idx


class TestSortArenas:
    def test_row_major_order(self):
        # Two rows of 3 arenas, deliberately unsorted
        arenas = [
            Arena(cx=300, cy=100, r=40, idx=0),  # row 0, col 2
            Arena(cx=100, cy=100, r=40, idx=0),  # row 0, col 0
            Arena(cx=200, cy=100, r=40, idx=0),  # row 0, col 1
            Arena(cx=200, cy=250, r=40, idx=0),  # row 1, col 1
            Arena(cx=100, cy=250, r=40, idx=0),  # row 1, col 0
        ]
        sorted_a = sort_arenas_row_major(arenas)
        assert len(sorted_a) == 5
        # Check indices assigned correctly
        assert [a.idx for a in sorted_a] == [0, 1, 2, 3, 4]
        # Check order: row 0 left-to-right, then row 1
        assert sorted_a[0].cx == 100 and sorted_a[0].cy == 100
        assert sorted_a[1].cx == 200 and sorted_a[1].cy == 100
        assert sorted_a[2].cx == 300 and sorted_a[2].cy == 100
        assert sorted_a[3].cx == 100 and sorted_a[3].cy == 250
        assert sorted_a[4].cx == 200 and sorted_a[4].cy == 250

    def test_empty(self):
        assert sort_arenas_row_major([]) == []


def test_bbox_min_size_grows_small_arenas_centred():
    from thermokourt.extract.arena_extractor import Arena

    a = Arena(cx=1254.5, cy=1147.25, r=93.3, idx=3)
    x, y, w, h = a.bbox(padding=0.1, frame_w=2600, frame_h=2160)
    assert w < 224  # the 11 Nov problem: 204 px clip
    x, y, w, h = a.bbox(padding=0.1, frame_w=2600, frame_h=2160, min_size=224)
    assert w >= 224 and h >= 224 and w % 2 == 0
    assert abs((x + w / 2) - a.cx) <= 1.5 and abs((y + h / 2) - a.cy) <= 1.5
    big = Arena(cx=500, cy=500, r=150)
    assert big.bbox(0.1, 2600, 2160, min_size=224) == big.bbox(0.1, 2600, 2160)


def test_shift_for_reduced(tmp_path):
    from thermokourt.extract.arena_extractor import Arena, shift_for_reduced

    lay = tmp_path / "arenas.json"
    lay.write_text(json.dumps({"frame_width": 2600, "frame_height": 2160,
                               "arenas": [{"cx": 1000, "cy": 900, "r": 100, "idx": 0}]}))
    arenas = [Arena(cx=1000, cy=900, r=100, idx=0)]
    assert shift_for_reduced(arenas, str(lay), str(tmp_path), 944, 976) == arenas  # not reduced
    (tmp_path / "reduced.json").write_text(json.dumps(
        {"x": 784, "y": 830, "w": 944, "h": 976, "full_width": 2600, "full_height": 2160}))
    (a,) = shift_for_reduced(arenas, str(lay), str(tmp_path), 944, 976)
    assert (a.cx, a.cy, a.r) == (216, 70, 100)
    # video still full-frame (crop not applied yet): no shift
    assert shift_for_reduced(arenas, str(lay), str(tmp_path), 2600, 2160) == arenas


def test_reduced_offset_and_shift(tmp_path):
    import json

    from thermokourt.extract.arena_extractor import Arena, reduced_offset, shift_arenas

    lay = tmp_path / "arenas.json"
    lay.write_text(json.dumps({"frame_width": 2600, "arenas": []}), encoding="utf-8")
    assert reduced_offset(lay, tmp_path, 944, 976) is None
    (tmp_path / "reduced.json").write_text(json.dumps(
        {"x": 784, "y": 830, "w": 944, "h": 976, "full_width": 2600, "full_height": 2160}),
        encoding="utf-8")
    assert reduced_offset(lay, tmp_path, 944, 976) == (784, 830, 2600, 2160)
    (a,) = shift_arenas([Arena(cx=10, cy=20, r=5, idx=3)], 784, 830)
    assert (a.cx, a.cy, a.r, a.idx) == (794, 850, 5, 3)


def _tiny_mp4(path, frames):
    import subprocess

    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=25",
         "-frames:v", str(frames), "-pix_fmt", "yuv420p", "-c:v", "libx264", str(path)],
        check=True,
    )


def test_chunks_are_motif_names_only(tmp_path):
    from thermokourt.extract.arena_extractor import get_mp4_chunks

    for name in ("000001.mp4", "000000.mp4", "000000.reduced.mp4", "export.mp4"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.split("/")[-1] for p in get_mp4_chunks(tmp_path)] == ["000000.mp4", "000001.mp4"]
    other = tmp_path / "plain"
    other.mkdir()
    (other / "clip.mp4").write_bytes(b"x")
    assert get_mp4_chunks(other)[0].endswith("clip.mp4")


def test_clip_frame_check(tmp_path):
    import shutil

    import pytest

    from thermokourt.extract.arena_extractor import check_clip_frames

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg missing")
    chunks = [tmp_path / "000000.mp4", tmp_path / "000001.mp4"]
    for c in chunks:
        _tiny_mp4(c, 10)
    good, short = tmp_path / "good.mp4", tmp_path / "short.mp4"
    _tiny_mp4(good, 20)
    _tiny_mp4(short, 12)
    check_clip_frames([str(good)], [str(c) for c in chunks])
    with pytest.raises(RuntimeError, match="short.mp4 has 12"):
        check_clip_frames([str(good), str(short)], [str(c) for c in chunks])
