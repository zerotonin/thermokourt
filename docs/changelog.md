# Changelog

All notable changes to ThermoKourt are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

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
