# Contributing

We welcome contributions from collaborators, students, and the broader
behavioural neuroscience community.

## Development setup

```bash
git clone https://github.com/zerotonin/thermokourt.git
cd thermokourt
pip install -e ".[dev,track]"
```

For GPU-dependent stages (auto-scoring), also install the `score` extra:

```bash
pip install -e ".[dev,track,score]"
```

## Running the test suite

```bash
pytest
```

To match the CI matrix locally, run with coverage:

```bash
pytest --cov=thermokourt --cov-report=term
```

## Linting

```bash
ruff check .
```

Auto-fix where possible:

```bash
ruff check . --fix
```

## Building the docs locally

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/ docs/_build/html
open docs/_build/html/index.html   # macOS
xdg-open docs/_build/html/index.html  # Linux
```

## Commit conventions

Atomic commits, one logical change per commit. Follow Conventional
Commits-ish prefixes:

| Prefix | Use for |
|--------|---------|
| `feat(<module>):` | New feature in `<module>` |
| `fix(<module>):` | Bug fix in `<module>` |
| `refactor(<module>):` | Restructure without behaviour change |
| `docs(<topic>):` | Documentation only |
| `test(<module>):` | Test additions or fixes |
| `ci:` | Continuous integration changes |
| `chore:` | Tooling, dependencies, housekeeping |

## Pull request workflow

1. Fork and create a feature branch from `main`.
2. Run `pytest` and `ruff check .` locally.
3. Open a PR against `main`. The `tests` and `docs` workflows must pass.
4. Tag a reviewer (Bart for stage 1–3 changes; Alex for scoring).
