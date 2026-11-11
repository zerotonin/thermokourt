# Changelog

All notable changes to ThermoKourt are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `extract.arena_detect`: new detector combining intensity blobs (radius
  octave scan, both polarities), Hough seeds, and disc template matching
  restricted to the seed hull. Finds all 10 arenas on both the zoomed-in
  (Oct 2025) and the zoomed-out, lamp-in-view (Jun 2026) rig layouts, where
  Hough alone found 0/10 on the latter. ~1 s per 2600×2160 frame.
- `thermokourt-extract` options: `--min_r/--max_r` (pixel radius window),
  `--roi X,Y,W,H` (restrict detection to the arena plate), `--polarity`
  (`bright`/`dark`/`auto`), `--detect_only` (detect, verify, save JSON +
  preview PNG, no video extraction).
- `<recording>_arenas.png` preview written next to the arenas JSON.
- `thermokourt-extract --fps`: clips are labelled with the real camera rate,
  by default taken from the Motif frame timestamps (`recorded_fps`). Motif had
  written 25 fps into a 69 fps recording, so clips played 2.76× slow; every
  frame is kept and re-timed. A measured rate within 1 % of the camera's set
  rate uses the set rate: jitter such as 69.001 fps gave a time base (1000/69001)
  that OpenCV's MPEG-4 writer refuses, so the corner-masked clips were silently
  not written. The writer now fails loudly instead.
- `thermokourt-extract --min_size` (runner default `extract.min_size: 224`):
  clips are grown to at least this side, centred on the arena, so the pose
  model's 224 px instance crop fits (arenas detected at r≈93 px gave 204 px).
- Re-cutting a recording cropped by Rig Conductor (`reduced.json`) shifts a
  full-frame arena layout into the cropped video; the arenas JSON is always
  written in full camera coordinates (`video_offset` records the crop).
- `sleap_track.sh`: a clip without animals writes `<clip>.empty` and exits 0
  (the analysis export crashed on empty predictions); `fetch` reports such
  arenas as "no animals found".
- Opportunistic SLEAP on Aoraki: `sleap_submit.sh` + `sleap_worker.sh` (pull
  queue, one sbatch per GPU partition, 3–5 clips packed per GPU with TF memory
  growth, atomic claims with heartbeat, 3 tries per clip). The runner submits
  through it and stops polling once every clip is done, cancelling workers that
  are still waiting for a GPU.
- Live cut progress: the cutter prints `@progress arena=NN of=N phase=P frac=F`
  every 2 s; the runner streams it and logs `@progress cut <fraction> <detail>`.
- The cut prefers a layout saved with the recording (`clips/<rec>_arenas.json`)
  over the rig default, so older recordings from other camera positions work.
- Chunks: only Motif's `NNNNNN.mp4` files are joined (a stray `*.reduced.mp4`
  or export in the folder used to be stitched in); after the cut every clip is
  checked to hold all frames of all chunks (`check_clip_frames`).
- `submit` sends only arenas without poses (.h5) or a no-animals marker
  (.empty) on the cluster, and skips the cluster when none are left.
- `poll` cancels only workers still pending (`scancel` failed on ids that had
  already finished) and judges a run finished by its clips, so workers cancelled
  as "not needed" no longer make a complete run look failed.
- `thermokourt-overlay` (was a placeholder): draws the SLEAP skeletons (Wong
  colours: male 1 blue, male 2 bluish green, female orange) and/or the scored
  events onto an arena clip, enlarged 2×, every frame kept at the real rate.
- `score` treats an arena with a `.empty` marker as "no animals found, nothing to
  score" instead of a missing pose file. RTX6000 (Blackwell) and H200 stay out until a
  smoke test: the sleap env is TF 2.7 / CUDA 11.2.
- Regression fixtures (`tests/data/projection_*_10arenas.png/.json`) and
  `tests/test_arena_detect.py`.
- `thermokourt.db` (stage 6): SQLite star schema (`schema.sql`), DrosoDB
  snapshot reader + lineage copy (`ddb.py`), trial registration with coded
  IDs (`trials.py`), seed rows for experiments E1–E7 and the pilot ethogram,
  `thermokourt-db` CLI (`init`, `vials`, `import-ddb`, `add-trial`,
  `export --blind`, `status`), views `v_trial_rates` and `v_agreement`.
- `thermokourt.pose` (stage 2.5 / 5-v0): SLEAP analysis-HDF5 reader, role
  assignment (stationary female), per-male features (speed, wing angles,
  distance/approach/facing), rule-based detectors for wing extension, wing
  threat, lunge and following (`RuleParams` v0), per-window locomotion and
  encounter summary.
- `thermokourt.trial` (stage 0): rig config (`~/.config/thermokourt/rig.yaml`),
  idempotent session runner (extract → register → upload → submit → poll →
  fetch → score) with on-disk state, `thermokourt-trial` CLI and the PyQt6
  operator window with DrosoDB print-code completion.
- `thermokourt.db.gamethogram`: import of GameThogram TXT exports as human
  events; `thermokourt.db.agreement`: frame- and bout-level agreement per
  group × temperature × behaviour; `thermokourt-db import-gamethogram` and
  `agreement` commands.
- `scripts/slurm/sleap_train.sh` and `sleap_track.sh`: Aoraki jobs for
  top-down SLEAP training (flies13 + lab labels) and per-arena-clip
  inference with analysis export.

### Changed

- Max-projection samples the whole clip densely for recordings shorter than
  two minutes (previously 10 frames from 25 % onwards, which left fly
  bodies in the projection of short test clips).
- GitHub Actions workflows: `tests.yml`, `docs.yml`, `release.yml`.
- Sphinx sidebar populated with grouped navigation (User guide,
  Development, API reference, Project).

## v0.1.0 — 2026-03-11

Initial public release.

### Added

- `arena_extractor`: Hough-transform-based arena detection with an
  interactive OpenCV verification GUI; ffmpeg concat + crop pipeline.
- `posttrack`: post-tracking GUI for reviewing idtracker.ai output.
- `scripts/slurm/train_scorer.sh`: HPC job template for stage 5 training.
- Sphinx documentation scaffold (Furo theme, MyST parser, autodoc).
- CITATION.cff and GPL-3.0-or-later licence.

### Known limitations

- `track.cli` and `overlay.cli` are placeholder entry points.
- Automated scorer (`thermokourt.score`) is not yet implemented.
