"""Rig configuration: where things are on this PC and on the cluster.

Loaded from ``~/.config/thermokourt/rig.yaml`` (or ``$THERMOKOURT_RIG``).
Machine-specific paths and cluster names live here, never in the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_PATH = Path("~/.config/thermokourt/rig.yaml").expanduser()

EXAMPLE = """\
# thermokourt rig configuration
data_root: /mnt/DATA                      # Motif recordings
db: ~/data/heatagg.sqlite3                # behaviour database (NVMe, not the data disk)
ddb_snapshot: ~/data/drosodb/ddb.latest.sqlite3   # synced DrosoDB backup
arenas_json: ~/data/rig/arenas.json       # verified arena positions for this rig
mm_per_pixel: 0.035                       # from the arena diameter
n_animals: 3
extract:
  crf: 22
  jobs: 4
cluster:
  ssh_host: aoraki                        # alias in ~/.ssh/config
  project: /projects/sciences/zoology/geurten_lab/HeatAggression
  sleap_run: flies13_base                 # models/<run>.centroid, .centered_instance
  array_time: "04:00:00"
  invert: true                            # dark flies on white; flies13 models want the opposite
"""


@dataclass
class ClusterConfig:
    ssh_host: str = "aoraki"
    project: str = "/projects/sciences/zoology/geurten_lab/HeatAggression"
    sleap_run: str = "flies13_base"
    array_time: str = "04:00:00"
    invert: bool = True  # flies13-based models expect bright flies on a dark background
    # how to reach the cluster; a shared rig sets this to the rig-ssh helper so that
    # operators never hold the account's key (see rigconductor deploy/)
    ssh_cmd: str = "ssh -o BatchMode=yes"


@dataclass
class ExtractConfig:
    crf: int = 22
    jobs: int = 4
    min_size: int = 224  # px a side; the pose models crop 224 px around each fly


@dataclass
class RigConfig:
    data_root: Path
    db: Path
    ddb_snapshot: Path | None
    arenas_json: Path
    mm_per_pixel: float | None = None
    n_animals: int = 3
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> RigConfig:
        p = Path(path or os.environ.get("THERMOKOURT_RIG") or DEFAULT_PATH).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"rig config not found: {p}\nCreate it with:  thermokourt-trial init-config"
            )
        with open(p) as f:
            d = yaml.safe_load(f) or {}
        exp = lambda k: Path(d[k]).expanduser() if d.get(k) else None  # noqa: E731
        return cls(
            data_root=exp("data_root"),
            db=exp("db"),
            ddb_snapshot=exp("ddb_snapshot"),
            arenas_json=exp("arenas_json"),
            mm_per_pixel=d.get("mm_per_pixel"),
            n_animals=int(d.get("n_animals", 3)),
            extract=ExtractConfig(**(d.get("extract") or {})),
            cluster=ClusterConfig(**(d.get("cluster") or {})),
        )


def write_example(path: str | Path | None = None, overwrite: bool = False) -> Path:
    p = Path(path or DEFAULT_PATH).expanduser()
    if p.exists() and not overwrite:
        raise FileExistsError(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(EXAMPLE)
    return p
