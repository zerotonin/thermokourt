# Continuous integration

ThermoKourt uses three GitHub Actions workflows. All workflow files live in
`.github/workflows/`.

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `tests.yml` | push / PR to `main` | Run pytest on 3 OS × 3 Python versions, plus `ruff check` |
| `docs.yml` | push / PR to `main` | Build Sphinx docs; deploy to GitHub Pages on push to `main` |
| `release.yml` | push of `v*.*.*` tag | Gate on tests, build sdist + wheel, attach to GitHub Release |

## `tests.yml`

Runs the full test suite across the support matrix:

- **OS:** `ubuntu-latest`, `macos-latest`, `windows-latest`
- **Python:** `3.10`, `3.11`, `3.12`

Each job installs the package with the `dev` and `track` extras and runs
`pytest` with coverage. Coverage XML is uploaded as an artifact from the
Ubuntu × Python 3.12 job (canonical run).

A separate `lint` job runs `ruff check .` on Ubuntu / Python 3.12.

## `docs.yml`

Builds the Sphinx site (`sphinx-build -b html docs/ docs/_build/html`) using
the pinned dependencies in `docs/requirements.txt`. On every push to `main`,
the rendered HTML is uploaded as a Pages artifact and deployed via
`actions/deploy-pages@v4`.

PR runs build the docs but do not deploy — this catches doc regressions
before they reach `main`.

### Required one-time repo settings

These are configured in the GitHub web UI for the repository:

1. **Settings → Pages → Source:** "GitHub Actions"
2. **Settings → Actions → General → Workflow permissions:** "Read and
   write permissions"
3. **Settings → Environments:** create `github-pages` (auto-created by the
   first deploy)

## `release.yml`

Triggers on any tag matching `v*.*.*`. The workflow:

1. Runs the test suite (gate — release fails if tests fail).
2. Builds `sdist` + `wheel` via `python -m build`.
3. Creates a GitHub Release and attaches the distributions, with
   auto-generated release notes.

Zenodo is wired up via the GitHub webhook at
[zenodo.org/account/settings/github/](https://zenodo.org/account/settings/github/);
each new release auto-mints a DOI from `CITATION.cff`.

## Branch protection

`main` is protected with the following requirements:

- All `tests` matrix jobs must pass.
- The `docs` build job must pass.
- At least one approving review for PRs.
