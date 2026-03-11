# Getting started

## Installation

### From source (recommended)

```bash
git clone https://github.com/zerotonin/thermokourt.git
cd thermokourt
pip install -e ".[dev]"
```

### Conda environment

```bash
conda env create -f environment.yml
conda activate thermokourt
pip install -e ".[dev]"
```

### System dependencies

ThermoKourt requires **ffmpeg ≥ 5.0** on your `PATH`:

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Conda
conda install -c conda-forge ffmpeg
```

## Verifying the installation

```bash
python -c "import thermokourt; print(thermokourt.__version__)"
thermokourt-extract --help
```

## Your first extraction

Given a Motif recording directory:

```
droso_18deg_20251002_102448/
├── 000000.mp4
├── 000001.mp4
├── ...
└── metadata.yaml
```

Run:

```bash
thermokourt-extract /path/to/droso_18deg_20251002_102448
```

The interactive arena editor will open. Verify the detected circles, adjust
if needed, then press **Q** to accept and begin extraction.
