#!/usr/bin/env python
"""Build a SLEAP labelling project for the lab's arena clips.

Run inside the `sleap` conda environment (SLEAP 1.4.1):

    python scripts/sleap/make_label_project.py OUT.slp CLIP.mp4 [CLIP.mp4 ...] \
        --skeleton-from flies13_train.pkg.slp --n-per-clip 15 --invert

* uses the flies13 skeleton (13 nodes) so the public weights can be reused;
* samples `n-per-clip` frames per clip, spread over the recording with a
  bias towards frames where the flies are close to each other (interaction
  frames are the ones the detectors need);
* optionally writes an inverted copy of each clip next to it (`*_inv.mp4`)
  and labels that, because the flies13-based models expect bright flies on
  a dark background;
* if `--predict-with centroid_dir centered_instance_dir` is given, the
  current models' predictions are added so the labeller only corrects them.

Open the result with `sleap-label OUT.slp`.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


def interaction_scores(path: str, step: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Rough 'flies close together' score per sampled frame: dark-blob count
    (fewer blobs than flies ⇒ overlapping/close)."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx, scores = [], []
    for f in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in cnts if cv2.contourArea(c) > 300]
        idx.append(f)
        scores.append(1.0 / max(1, len(big)))  # 1 blob -> 1.0 (merged), 3 blobs -> 0.33
    cap.release()
    return np.array(idx), np.array(scores)


def sample_frames(path: str, n: int, rng: np.random.Generator) -> list[int]:
    idx, sc = interaction_scores(path)
    if len(idx) == 0:
        return []
    k_inter = n // 2
    order = np.argsort(-sc)
    chosen = set(idx[order[:k_inter]].tolist())
    rest = [i for i in idx.tolist() if i not in chosen]
    chosen |= set(rng.choice(rest, size=min(n - len(chosen), len(rest)), replace=False).tolist())
    return sorted(chosen)


def invert_clip(src: Path) -> Path:
    dst = src.with_name(src.stem + "_inv.mp4")
    if not dst.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(src),
                "-vf",
                "negate",
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "16",
                "-an",
                str(dst),
            ],
            check=True,
        )
    return dst


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("out")
    ap.add_argument("clips", nargs="+")
    ap.add_argument(
        "--skeleton-from", required=True, help="a .slp with the skeleton to reuse (flies13)"
    )
    ap.add_argument("--n-per-clip", type=int, default=15)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--predict-with", nargs=2, metavar=("CENTROID_DIR", "CI_DIR"), default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import sleap
    from sleap import Labels, Video
    from sleap.gui.suggestions import SuggestionFrame

    skel = sleap.load_file(a.skeleton_from).skeletons[0]
    rng = np.random.default_rng(a.seed)
    labels = Labels(skeletons=[skel])
    suggestions = []
    for c in a.clips:
        src = Path(c).resolve()  # absolute: SLEAP resolves relative paths against its cwd
        vid_path = invert_clip(src) if a.invert else src
        video = Video.from_filename(str(vid_path))
        labels.add_video(video)
        frames = sample_frames(str(vid_path), a.n_per_clip, rng)
        suggestions += [SuggestionFrame(video, f) for f in frames]
        print(f"{vid_path.name}: {len(frames)} frames suggested")
    labels.suggestions = suggestions

    if a.predict_with:
        from sleap.nn.data.providers import VideoReader
        from sleap.nn.inference import TopDownPredictor

        predictor = TopDownPredictor.from_trained_models(
            centroid_model_path=a.predict_with[0],
            confmap_model_path=a.predict_with[1],
            peak_threshold=0.1,
        )
        for video in labels.videos:
            frames = sorted(s.frame_idx for s in suggestions if s.video is video)
            if not frames:
                continue
            preds = predictor.predict(VideoReader(video, example_indices=frames))
            for lf in preds.labeled_frames:
                labels.append(lf)
            print(
                f"{Path(video.backend.filename).name}: predictions added for {len(frames)} frames"
            )

    labels.save(a.out)
    print(
        f"saved {a.out}: {len(labels.videos)} videos, {len(labels.suggestions)} suggested frames,"
        f" {len(labels.labeled_frames)} frames with predictions"
    )


if __name__ == "__main__":
    main()
