# ThermoKourt

**A modular pipeline for analysing *Drosophila* courtship and aggression under thermal manipulation.**

ThermoKourt is a toolbox for behavioural neuroscientists studying how temperature modulates social behaviour in *Drosophila melanogaster*. It covers the full arc from raw multi-camera recordings to publication-ready ethograms and automated behavioural classifiers.

> **Name:** *Thermo* (temperature) + *Kourt* (courtship, with a nod to the German *kurz*—short, as in short behavioural bouts). The K also disambiguates from generic "court" in search results.

---

## Pipeline overview

```
 ┌─────────────────┐     ┌─────────────────┐     ┌────────────────────┐
 │  1 · ARENA       │     │  2 · IDENTITY    │     │  3 · OVERLAY       │
 │  arena_extractor │────▶│  idtracker.ai /  │────▶│  identity_overlay  │
 │  stitch + crop   │     │  CNN tracker     │     │  colour-coded IDs  │
 └─────────────────┘     └─────────────────┘     └────────────────────┘
                                                          │
 ┌─────────────────┐     ┌─────────────────┐              ▼
 │  5 · AUTO-SCORE  │◀────│  4 · ETHOGRAM    │◀─── overlay videos
 │  CNN classifier  │     │  GameThogram     │     ready for scoring
 │  (Aoraki HPC)    │     │  manual scoring  │
 └─────────────────┘     └─────────────────┘
```

| Stage | Module | Purpose | Key dependency |
|-------|--------|---------|----------------|
| **1** | `arena_extractor` | Detect circular arenas (Hough transform), interactive verification GUI, ffmpeg concat + crop | matplotlib, ffmpeg |
| **2** | `identity_tracker` | Assign persistent IDs to 2 males + 1 headless female per arena | [idtracker.ai](https://idtracker.ai) ≥ v6 |
| **3** | `identity_overlay` | Render ID-coloured auras (teal / orange) onto arena videos | OpenCV |
| **4** | — | Manual ethogram annotation via [GameThogram](https://github.com/zerotonin/GameThogram) | GameThogram |
| **5** | `auto_scorer` | Train a temporal CNN on manual annotations, batch-infer on HPC | PyTorch, Slurm |

## Installation

### Prerequisites

- Python ≥ 3.10
- ffmpeg ≥ 5.0 on `PATH`
- A working display for the interactive arena editor (matplotlib backend)

### Install from source

```bash
git clone https://github.com/zerotonin/thermokourt.git
cd thermokourt
pip install -e ".[dev]"
```

### Conda (recommended for GPU stages)

```bash
conda env create -f environment.yml
conda activate thermokourt
pip install -e ".[dev]"
```

## Quick start

### 1 — Extract arenas from a Motif recording

```bash
thermokourt-extract /data/droso_18deg_20251002_102448

# Or with options:
thermokourt-extract /data/droso_18deg_20251002_102448 \
    --output_dir /results/ \
    --n_arenas 10 \
    --crf 18
```

This opens an interactive matplotlib window showing detected circles. Drag centres to reposition, drag rim handles to resize, right-click to add/delete. Press **Q** to accept and begin extraction.

### 2 — Track identities

```bash
thermokourt-track /results/droso_18deg_20251002_102448_arena_00.mp4 \
    --n_animals 3 \
    --backend idtracker
```

### 3 — Overlay identity colours

```bash
thermokourt-overlay /results/droso_18deg_20251002_102448_arena_00.mp4 \
    --tracks /results/arena_00_tracks.h5
```

### 4 — Manual scoring

Open the overlay videos in [GameThogram](https://github.com/zerotonin/GameThogram) and score courtship / aggression behaviours with a gamepad.

### 5 — Train auto-scorer (HPC)

```bash
sbatch scripts/slurm/train_scorer.sh \
    --annotations /data/annotations/ \
    --videos /data/overlays/ \
    --epochs 100
```

## Experimental context

This toolbox was developed for experiments investigating how ambient temperature (range 18–32 °C) modulates the balance between courtship and aggression in *Drosophila melanogaster*. Each arena contains two intact males and one headless female. The headless female serves as a standardised social stimulus that elicits both courtship and aggression without contributing active behaviour of her own.

## Project structure

```
thermokourt/
├── thermokourt/                 # Main package
│   ├── extract/                 # Stage 1: arena detection + video stitching
│   │   ├── arena_extractor.py   # Hough detection, interactive GUI, ffmpeg crop
│   │   └── hough.py             # Circle detection backends (OpenCV / scipy)
│   ├── track/                   # Stage 2: identity tracking
│   │   ├── idtracker_wrapper.py # idtracker.ai integration
│   │   └── tracker_base.py      # Abstract tracker interface
│   ├── overlay/                 # Stage 3: identity visualisation
│   │   └── identity_overlay.py  # Colour-coded aura rendering
│   ├── score/                   # Stage 5: automated scoring
│   │   ├── model.py             # Temporal CNN architecture
│   │   ├── dataset.py           # Ethogram + video dataset loader
│   │   └── train.py             # Training loop
│   └── utils/                   # Shared utilities
│       ├── video_io.py          # ffmpeg / ffprobe helpers
│       └── arena.py             # Arena dataclass + serialisation
├── scripts/
│   └── slurm/                   # HPC job scripts for Aoraki
├── docs/                        # Sphinx documentation
├── tests/                       # pytest suite
├── pyproject.toml               # PEP 621 packaging
├── environment.yml              # Conda environment
├── CITATION.cff                 # Citation metadata
└── LICENSE                      # GPL-3.0-or-later
```

## Relation to other tools

ThermoKourt does **not** replace any of the tools it wraps. It provides:

- A **consistent CLI** that chains them together
- **Sane defaults** for the specific experimental design (3 flies, circular arenas, Motif recordings)
- **Reproducibility metadata** — every processing step logs parameters and software versions to a sidecar JSON

| Tool | Role in ThermoKourt | Reference |
|------|---------------------|-----------|
| [idtracker.ai](https://idtracker.ai) | Identity tracking (stage 2) | Romero-Ferrero et al., *Nat. Methods* 2019 |
| [GameThogram](https://github.com/zerotonin/GameThogram) | Manual ethogram scoring (stage 4) | Geurten & Kuhlemann |
| [arena_annotator](https://github.com/zerotonin/arena_annotator) | Circle annotation patterns reused in stage 1 | Geurten |

## Contributing

1. Fork and create a feature branch
2. Install dev dependencies: `pip install -e ".[dev]"`
3. Run tests: `pytest`
4. Run lints: `ruff check .`
5. Open a pull request

## Authors

Bart R.H. Geurten — Department of Zoology, University of Otago, Dunedin, New Zealand

## License

GPL-3.0-or-later
