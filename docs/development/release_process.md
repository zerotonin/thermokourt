# Release process

## 1. Pre-release checks

```bash
pytest
ruff check .
sphinx-build -b html docs/ docs/_build/html
```

All three must pass cleanly before tagging.

## 2. Update version metadata

Update the version string in **three** places — these must stay in lockstep:

| File | Field |
|------|-------|
| `pyproject.toml` | `[project] version = "X.Y.Z"` |
| `thermokourt/__init__.py` | `__version__ = "X.Y.Z"` |
| `CITATION.cff` | `version: "X.Y.Z"` (quoted) and `date-released: "YYYY-MM-DD"` |

Validate `CITATION.cff` with [`cffconvert`](https://github.com/citation-file-format/cffconvert):

```bash
cffconvert --validate
```

Common pitfalls that break Zenodo minting:

- Unquoted version (`version: 1.0.0` instead of `version: "1.0.0"`).
- Em-dashes in the abstract — replace with `--` in plain text.
- Wrong ORCID format — must be a full URL (`https://orcid.org/0000-...`).

## 3. Update the changelog

Add a section to {doc}`../changelog` for the new version with the format:

```markdown
## vX.Y.Z — YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...
```

## 4. Tag and push

```bash
git tag -a vX.Y.Z -m "vX.Y.Z: <one-line summary>"
git push origin vX.Y.Z
```

The `release.yml` workflow runs automatically:

1. Tests pass on Ubuntu / Python 3.12.
2. `sdist` + `wheel` are built.
3. A GitHub Release is created with the distributions attached.
4. Zenodo's GitHub webhook mints a new DOI.

## 5. Post-release

- Verify the Zenodo DOI resolves (allow ~10 min for the webhook).
- Update the README badge if the DOI is being added for the first time.
- Announce on the lab channel.

## Rolling back a broken release

Zenodo DOIs are immutable once published, but draft deposits can be
deleted. If the CFF was broken:

1. Delete the GitHub Release (this leaves the tag in place).
2. Delete the draft Zenodo deposit at zenodo.org.
3. Delete the local and remote tag:

   ```bash
   git tag -d vX.Y.Z
   git push origin :refs/tags/vX.Y.Z
   ```

4. Fix the CFF, commit, re-tag, re-push.
