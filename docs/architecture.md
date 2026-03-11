# Architecture & design rationale

This document serves both as internal documentation and as an architecture
white paper for the ThermoKourt pipeline. It is intended for a scientific
audience and doubles as a development roadmap.

---

## 1 — Motivation

Temperature is a potent modulator of *Drosophila* social behaviour. Courtship
and aggression share overlapping motor programmes and neural substrates, and
their balance shifts with ambient temperature in ways that are not yet fully
characterised. Studying this requires:

1. High-throughput video acquisition (multiple arenas, long recordings)
2. Reliable individual identification (2 males + 1 headless female)
3. Frame-accurate behavioural annotation
4. Scalable automated classification for large datasets

No single existing tool covers this full workflow. ThermoKourt chains
purpose-built and established open-source tools into a reproducible pipeline,
with each stage producing self-describing output that can be audited or
re-processed independently.

## 2 — Design principles

**Modularity.** Each stage is a standalone CLI tool with JSON/HDF5 interchange
formats. Stages can be run independently, replaced, or extended without
affecting the rest of the pipeline.

**Minimal dependencies per stage.** Stage 1 (arena extraction) requires only
matplotlib, numpy, and ffmpeg—no GPU. GPU-dependent stages (tracking, auto-scoring)
are optional extras that can run on a different machine.

**Human-in-the-loop verification.** The pipeline never silently commits to a
processing decision. Arena detection opens an interactive editor. Tracking
produces overlay videos for visual inspection. Manual scoring precedes and
validates automated scoring.

**Reproducibility.** Every processing step writes a sidecar JSON recording
input files, parameters, software versions, and timestamps.

## 3 — Stage 1: Arena extraction

### 3.1 Problem

Motif (Loopbio) records sessions as sequential `.mp4` chunks with a
`metadata.yaml` manifest. A typical session produces 5–20 chunks at 2160 × 2600
pixels, 25 fps. Only the circular arena regions (~10 per frame) contain useful
data; the surrounding area is wasted storage and bandwidth.

### 3.2 Approach

**Circle detection.** We use the Hough Circle Transform on the first frame of
the first chunk. The detector sweeps the accumulator threshold from strict
(param2 = 80) to loose (param2 = 11) until it finds ≥ N circles, where N is
the expected arena count (default 10). This iterative loosening handles
variable contrast across experiments without requiring manual parameter tuning.

The implementation preferentially uses OpenCV's `HoughCircles` (faster, more
robust) but falls back to scikit-image's `hough_circle` + `hough_circle_peaks`
if OpenCV is not installed. This keeps the base dependency footprint small.

**Interactive verification.** Detected circles are presented in a matplotlib
GUI adapted from the `circle_annotator` tool. Users can drag centres, resize
radii, add or delete circles. This takes ~30 seconds per experiment and
eliminates false positives that would otherwise waste hours of downstream
processing.

**Video concatenation and cropping.** We use ffmpeg's concat demuxer to avoid
creating a monolithic intermediate file. Each arena is cropped in a single
ffmpeg pass that reads all chunks sequentially:

```
ffmpeg -f concat -safe 0 -i chunks.txt -vf crop=W:H:X:Y -c:v libx264 arena.mp4
```

This is I/O-efficient: the source chunks are read N times (once per arena)
but never copied to an intermediate full-frame video unless explicitly
requested.

### 3.3 Arena position persistence

Arena positions are saved as JSON. When the physical setup does not change
between recordings (same camera, same arena plate), positions can be reused
with `--arenas previous_arenas.json`, skipping detection and GUI entirely.

### 3.4 Alternatives considered

| Approach | Why not |
|----------|---------|
| Template matching | Requires a reference template; less robust to lighting changes |
| Blob detection | Finds arenas but doesn't give circle parameters directly |
| Manual ROI in ImageJ | Not scriptable, not reproducible, slower for 10 arenas |

## 4 — Stage 2: Identity tracking

### 4.1 Problem

Each arena contains three unmarked *Drosophila*: two intact males and one
headless female. We need to maintain individual identity across the full
recording (potentially hours) to attribute courtship and aggression behaviours
to specific individuals.

### 4.2 Tool selection: idtracker.ai v6

We selected idtracker.ai v6 as the default tracking backend for the following
reasons:

**Identity accuracy on small groups of flies.** The idtracker.ai benchmark
includes Drosophila videos with ≤ 15 individuals where accuracy exceeds 99.9%
(Romero-Ferrero et al., 2019; v6 eLife preprint, 2025). Our scenario (3 flies)
is well within this regime. The headless female is morphologically distinct,
which further aids discrimination.

**No training data required.** idtracker.ai learns individual fingerprints
from the video itself using self-supervised representation learning. This
eliminates the need for manually labelled identity data, which would be
prohibitively expensive across hundreds of recordings.

**Crossing resolution.** idtracker.ai explicitly detects and resolves animal
crossings (occlusions) using a dedicated crossing-detection network. With only
3 animals and relatively brief occlusions, this is highly reliable.

### 4.3 Alternatives considered

| Tool | Strength | Limitation for our use case |
|------|----------|---------------------------|
| **DeepLabCut (maDLC)** | State-of-the-art pose estimation | Identity tracking less robust for visually similar animals; requires labelled training frames |
| **SLEAP** | Fast, modular, good pose estimation | Flow-shift tracking has higher ID-switch rates; appearance-based ID needs labelled examples |
| **TRex** | Fast classical tracking | Less accurate identity maintenance than idtracker.ai on benchmarks |
| **STCS** | New segmentation-based approach | Less mature; fewer Drosophila benchmarks |
| **vmTracking** | Creative virtual-marker approach | Adds complexity; best for crowded scenes beyond our 3-fly case |

### 4.4 Integration strategy

ThermoKourt wraps idtracker.ai behind an abstract `TrackerBase` interface:

```python
class TrackerBase(ABC):
    @abstractmethod
    def track(self, video_path: str, n_animals: int) -> TrackingResult: ...
```

This allows swapping to a different backend (e.g., a custom CNN tracker
fine-tuned on our data) without changing downstream stages. The
`TrackingResult` is a standardised HDF5 file containing per-frame centroid
coordinates and identity labels.

### 4.5 Roadmap

- **v0.1**: idtracker.ai wrapper with CLI
- **v0.2**: Quality metrics (crossing count, identity confidence per frame)
- **v0.3**: Optional custom CNN tracker trained on accumulated idtracker.ai
  outputs, for faster inference on the HPC

## 5 — Stage 3: Identity overlay

### 5.1 Purpose

Overlay videos serve two purposes:

1. **Visual QC** — spot tracking errors before committing to annotation
2. **Scorer input** — GameThogram annotators need to know which fly is which

### 5.2 Rendering

Each tracked individual receives a semi-transparent radial gradient ("aura")
centred on their centroid:

- **Male 1**: teal (#00B4D8, alpha 0.4)
- **Male 2**: orange (#FF6B35, alpha 0.4)
- **Headless female**: assigned automatically

The aura radius scales with the animal's bounding box. Colours were chosen
for readability on greyscale backgrounds and for accessibility (teal/orange
is distinguishable under the most common colour vision deficiencies).

### 5.3 Implementation

OpenCV `addWeighted` blending per frame. For a typical 25 fps, 10-minute
arena video (15 000 frames at ~400 × 400 px), rendering takes approximately
2 minutes on a single CPU core.

## 6 — Stage 4: Manual ethogram annotation

### 6.1 Tool: GameThogram

GameThogram is an existing open-source tool for gamepad-driven ethogram
annotation, developed in the same lab. It supports:

- Frame-by-frame video stepping
- Multi-animal behaviour scoring with colour-coded icons
- Behaviour compatibility constraints (mutual exclusion)
- Export to text, Excel, MATLAB, and pickle

### 6.2 Behaviour vocabulary

The initial behaviour set for courtship vs. aggression experiments:

| Behaviour | Category | Description |
|-----------|----------|-------------|
| Wing extension | Courtship | Unilateral wing extension (vibration song) |
| Following | Courtship | Oriented pursuit of the female |
| Licking | Courtship | Proboscis contact with female abdomen |
| Attempted copulation | Courtship | Mounting attempt |
| Lunge | Aggression | Rapid forward thrust toward opponent |
| Wing threat | Aggression | Bilateral wing raise |
| Chase | Aggression | Oriented pursuit of the male opponent |
| Boxing / fencing | Aggression | Foreleg strikes |
| Tussle | Aggression | Grappling / rolling |

This vocabulary is configurable via GameThogram's JSON settings export.

### 6.3 Annotation protocol

Each student annotator scores a video independently. Inter-rater reliability
is computed (Cohen's kappa per behaviour) and videos with low agreement are
re-scored or discussed. Agreed annotations become training data for stage 5.

## 7 — Stage 5: Automated behavioural classification

### 7.1 Problem

Manual scoring is the bottleneck. A single 10-minute arena video takes
~30 minutes to score. With 10 arenas × multiple temperature conditions ×
replicates, the annotation workload quickly exceeds what a small team can
manage.

### 7.2 Approach

We train a temporal convolutional network (TCN) on the manually scored subset.
The model takes as input a window of consecutive frames (or pose features, if
available from the tracking stage) and predicts the active behaviours for the
centre frame.

**Architecture options under evaluation:**

| Architecture | Input | Pros | Cons |
|-------------|-------|------|------|
| **Frame-based TCN** | Raw cropped frames | No feature engineering | Needs more data, GPU-heavy |
| **Pose-based TCN** | Centroid trajectories + relative angles | Lightweight, interpretable | Requires accurate tracking |
| **Hybrid** | Frames + pose features | Best of both | Most complex |

### 7.3 Training on Aoraki (HPC)

The University of Otago's Aoraki cluster uses Slurm for job management. We
provide job scripts in `scripts/slurm/` that handle:

- Data staging from shared filesystem to local scratch
- Multi-GPU training with PyTorch DistributedDataParallel
- Automatic checkpoint saving and early stopping
- Result export back to shared filesystem

### 7.4 Active learning loop

After the first round of auto-scoring, annotators review the predictions in
GameThogram. Corrected frames are fed back into the training set, and the
model is retrained. This active learning loop progressively improves accuracy
while focusing annotator effort on the most informative (uncertain) samples.

### 7.5 Roadmap

- **v0.1**: Dataset loader + basic TCN on centroid features
- **v0.2**: Frame-based model, HPC training scripts
- **v0.3**: Active learning loop with GameThogram integration
- **v1.0**: Validated model with published accuracy benchmarks

## 8 — Data flow and file formats

```
Recording directory
  └── 000000.mp4, 000001.mp4, ..., metadata.yaml
        │
        ▼  arena_extractor
  <name>_arenas.json          ← arena positions (reusable)
  <name>_arena_00.mp4         ← cropped arena videos
  <name>_arena_01.mp4
  ...
        │
        ▼  identity_tracker
  arena_00_tracks.h5           ← HDF5: frames × animals × (x, y, id)
        │
        ▼  identity_overlay
  arena_00_overlay.mp4         ← video with coloured auras
        │
        ▼  GameThogram (manual)
  arena_00.gamethogram.pkl     ← ethogram annotations
  arena_00_export.xlsx
        │
        ▼  auto_scorer
  arena_00_predictions.h5      ← per-frame behaviour probabilities
```

### HDF5 schema for tracking output

```
/trajectories
  /animal_0
    /centroid          float64  (N_frames, 2)   # x, y in arena pixels
    /identity_label    string                    # "male_1", "male_2", "female"
    /confidence        float32  (N_frames,)     # identity confidence
  /animal_1
    ...
/metadata
  /n_animals           int
  /video_path          string
  /tracker_backend     string
  /tracker_version     string
  /parameters          JSON string
```

## 9 — Development roadmap

| Milestone | Target | Stages | Status |
|-----------|--------|--------|--------|
| **v0.1.0** — Arena extraction | March 2026 | 1 | 🔨 In progress |
| **v0.2.0** — Identity tracking | May 2026 | 1–2 | Planned |
| **v0.3.0** — Overlay + GameThogram integration | July 2026 | 1–3 | Planned |
| **v0.4.0** — Auto-scorer prototype | October 2026 | 1–5 | Planned |
| **v1.0.0** — Validated pipeline | February 2027 | 1–5 | Planned |

## 10 — References

1. Romero-Ferrero, F., Bergomi, M. G., Hinz, R. C., Heras, F. J. H. &
   de Polavieja, G. G. idtracker.ai: tracking all individuals in small or
   large collectives of unmarked animals. *Nat. Methods* **16**, 179–182 (2019).

2. Mathis, A. et al. DeepLabCut: markerless pose estimation of user-defined
   body parts with deep learning. *Nat. Neurosci.* **21**, 1281–1289 (2018).

3. Pereira, T. D. et al. SLEAP: A deep learning system for multi-animal pose
   tracking. *Nat. Methods* **19**, 486–495 (2022).

4. Lauer, J. et al. Multi-animal pose estimation, identification and tracking
   with DeepLabCut. *Nat. Methods* **19**, 496–504 (2022).

5. Walter, T. & Couzin, I. D. TRex, a fast multi-animal tracking system with
   markerless identification, and 2D estimation of posture and visual fields.
   *eLife* **10**, e64000 (2021).

6. Chen, Z. et al. Segmentation tracking and clustering system enables
   accurate multi-animal tracking of social behaviors. *Patterns* **5**,
   101071 (2024).

7. Azechi, H. & Takahashi, S. vmTracking enables highly accurate multi-animal
   pose tracking in crowded environments. *PLoS Biol.* **23**, e3003002 (2025).
