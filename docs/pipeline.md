# Pipeline guide

This page walks through each stage of the ThermoKourt pipeline with practical
examples and tips for common problems.

## Stage 1 — Arena extraction

### What it does

Motif records long sessions as sequential `.mp4` chunks (typically 18 000 frames
each at 25 fps). `arena_extractor` concatenates these chunks and crops each
circular arena into an individual video file.

### Arena detection

The first frame is analysed with a Hough Circle Transform. The detector sweeps
accumulator thresholds from strict to loose until it finds the expected number
of arenas (default: 10). It tries OpenCV first and falls back to
scikit-image if OpenCV is not installed.

### Interactive editor

The detected circles are presented in a matplotlib window. Each arena has:

- A **centre marker** (+) — drag to reposition
- A **rim handle** (◇) — drag to resize
- A **dashed bounding box** — shows the actual crop region

Key bindings:

| Key | Action |
|-----|--------|
| Drag centre | Move arena |
| Drag rim | Resize arena |
| Right-click empty | Add new arena |
| Right-click centre | Delete arena |
| A | Re-run auto-detection |
| +/- | Adjust all radii ±5 px |
| U | Set all radii to median |
| S | Save arena positions to JSON |
| Q / Enter | Accept and extract |
| Escape | Abort |

### Output

For a recording named `droso_18deg_20251002_102448` with 10 arenas:

```
droso_18deg_20251002_102448_arenas.json   # arena positions (reusable)
droso_18deg_20251002_102448_arena_00.mp4
droso_18deg_20251002_102448_arena_01.mp4
...
droso_18deg_20251002_102448_arena_09.mp4
```

### Reusing arena positions

If arena positions are stable across recordings (same camera setup):

```bash
thermokourt-extract /data/new_recording --arenas old_recording_arenas.json
```

## Stage 2 — Identity tracking

### Why idtracker.ai?

For 3 unmarked *Drosophila* in a small arena, idtracker.ai v6 achieves
>99.9% identity accuracy on comparable benchmarks. It works by learning a
visual fingerprint for each individual, which means it can distinguish the
two males and the headless female without any physical markers.

Alternatives considered:

- **DeepLabCut (multi-animal)**: excellent for pose estimation but identity
  tracking is less robust for visually similar flies
- **SLEAP**: fast and modular, but flow-shift tracking has higher ID-switch
  rates than idtracker.ai for small groups
- **STCS**: promising newer approach but less battle-tested

### Running the tracker

```bash
thermokourt-track arena_00.mp4 --n_animals 3 --backend idtracker
```

Output: an HDF5 file with per-frame centroid coordinates and identity labels.

## Stage 3 — Identity overlay

Renders colour-coded auras around each tracked individual:

- **Male 1**: teal (semi-transparent halo)
- **Male 2**: orange (semi-transparent halo)
- **Headless female**: automatically assigned remaining colour

```bash
thermokourt-overlay arena_00.mp4 --tracks arena_00_tracks.h5
```

## Stage 4 — Manual ethogram annotation

Open overlay videos in [GameThogram](https://github.com/zerotonin/GameThogram).
Define behaviours (courtship wing extension, lunge, chase, etc.) and score
frame-by-frame with a gamepad.

## Stage 5 — Automated scoring

Train a temporal CNN on the manual annotations, then batch-infer across all
remaining videos on the Aoraki HPC cluster:

```bash
# On Aoraki
sbatch scripts/slurm/train_scorer.sh --annotations /data/manual/ --epochs 100
sbatch scripts/slurm/infer_scorer.sh --model best.pt --videos /data/overlays/
```
