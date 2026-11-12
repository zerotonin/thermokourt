import numpy as np
import yaml

from thermokourt.extract.arena_extractor import recorded_fps


def test_recorded_fps_prefers_motif_timestamps(tmp_path):
    (tmp_path / "metadata.yaml").write_text(yaml.safe_dump({"acquisitionframerate": 25.0}))
    np.savez(tmp_path / "000000.npz", frame_time=1000 + np.arange(200) / 69.0)
    assert recorded_fps(str(tmp_path)) == 69.0


def test_recorded_fps_falls_back_to_metadata_then_none(tmp_path):
    assert recorded_fps(str(tmp_path)) is None
    (tmp_path / "metadata.yaml").write_text(yaml.safe_dump({"acquisitionframerate": 50.0}))
    assert recorded_fps(str(tmp_path)) == 50.0


def test_recorded_fps_snaps_jitter_to_the_set_rate(tmp_path):
    (tmp_path / "metadata.yaml").write_text(yaml.safe_dump({"acquisitionframerate": 69.0}),
                                            encoding="utf-8")
    np.savez(tmp_path / "000000.npz", frame_time=1000 + np.arange(200) / 69.001)
    assert recorded_fps(str(tmp_path)) == 69.0  # 69.001 -> time base 1000/69001 breaks MPEG-4


def test_recorded_fps_without_metadata_is_rounded(tmp_path):
    np.savez(tmp_path / "000000.npz", frame_time=1000 + np.arange(200) / 29.9701)
    assert recorded_fps(str(tmp_path)) == 29.97
