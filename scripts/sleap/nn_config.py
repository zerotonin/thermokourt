#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  ThermoKourt — nn_config                                         ║
# ║  « sleap-nn training config from a trained model »               ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Runs in the SLEAP 1.6 env. Reads a model's config (legacy JSON  ║
# ║  or sleap-nn YAML) and writes a YAML for `sleap-nn train`.       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Write a sleap-nn training config that fine-tunes from an existing model.

Usage::

    python nn_config.py BASE_MODEL_DIR TRAIN.slp OUT.yaml --val VAL.slp \\
        --ckpt-dir $PROJ/models --run lab_v3.centered_instance

``BASE_MODEL_DIR`` is a trained model (``best_model.h5`` + ``training_config.json``
from SLEAP 1.4, or ``best.ckpt`` + ``training_config.yaml`` from sleap-nn). Its
architecture is kept and its weights are the starting point. For centred-instance
models the crop size is grown to fit the largest labelled fly.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

CROP_MARGIN_PX = 16  # free pixels around the largest fly in a crop
CROP_MULTIPLE = 32  # UNet strides need crop sizes divisible by this
SCALE_AUG = (0.8, 1.25)  # random zoom per training sample (one set, several camera scales)
EPOCHS = 200
EARLY_STOP_PATIENCE = 20


def load_base(model_dir: Path):
    """Return (OmegaConf config, weights path) of a trained model."""
    from omegaconf import OmegaConf
    from sleap_nn.config.training_job_config import TrainingJobConfig

    if (model_dir / "best.ckpt").is_file():
        return OmegaConf.load(model_dir / "training_config.yaml"), model_dir / "best.ckpt"
    cfg = TrainingJobConfig.load_sleap_config(str(model_dir / "training_config.json"))
    return cfg, model_dir / "best_model.h5"


def largest_fly_px(labels_path: Path) -> float:
    """Longest bounding-box side over all user instances (px)."""
    import sleap_io as sio

    labels = sio.load_slp(str(labels_path))
    sides = []
    for lf in labels.user_labeled_frames:
        for inst in lf.user_instances:
            xy = inst.numpy()
            xy = xy[~np.isnan(xy).any(axis=1)]
            if len(xy) > 1:
                sides.append(float((xy.max(axis=0) - xy.min(axis=0)).max()))
    return max(sides) if sides else 0.0


def crop_size_for(fly_px: float, current: int) -> int:
    """Crop that fits ``fly_px`` plus margin, never smaller than ``current``."""
    need = math.ceil((fly_px + 2 * CROP_MARGIN_PX) / CROP_MULTIPLE) * CROP_MULTIPLE
    return max(int(current or 0), need)


def build(a) -> None:
    from omegaconf import OmegaConf

    cfg, weights = load_base(Path(a.base))
    d, m, t = cfg.data_config, cfg.model_config, cfg.trainer_config
    d.train_labels_path = [str(a.train)]
    d.val_labels_path = [str(a.val)] if a.val else None
    d.validation_fraction = 0.1
    d.user_instances_only = True
    d.preprocessing.max_height = None
    d.preprocessing.max_width = None
    geo = d.augmentation_config.geometric
    geo.scale_min, geo.scale_max = SCALE_AUG
    geo.scale_p = 1.0
    if m.head_configs.get("centered_instance"):
        fly = largest_fly_px(Path(a.train))
        d.preprocessing.crop_size = crop_size_for(fly * SCALE_AUG[1], d.preprocessing.crop_size)
        print(f"largest fly {fly:.0f} px -> crop {d.preprocessing.crop_size} px")
    m.pretrained_backbone_weights = str(weights)
    m.pretrained_head_weights = str(weights)
    t.ckpt_dir = str(a.ckpt_dir)
    t.run_name = a.run
    t.save_ckpt = True
    t.use_wandb = False
    t.max_epochs = a.epochs
    t.early_stopping.stop_training_on_plateau = True
    t.early_stopping.patience = EARLY_STOP_PATIENCE
    t.trainer_accelerator = a.accelerator
    OmegaConf.save(cfg, a.out)
    print(f"config written: {a.out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base", help="trained model dir to start from")
    ap.add_argument("train", help="training labels (.slp / .pkg.slp)")
    ap.add_argument("out", help="YAML to write")
    ap.add_argument("--val", default="", help="validation labels (default: 10 %% of train)")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--accelerator", default="auto")
    build(ap.parse_args(argv))


if __name__ == "__main__":
    main()
