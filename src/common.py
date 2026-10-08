from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def config(filename="configs/config.yaml"):
    with path(filename).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def digest(filename):
    h = hashlib.sha256()
    with Path(filename).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(filename, value):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp = filename.with_suffix(filename.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(filename)


def load_split(cfg, groups=("train", "val", "test")):
    split = json.loads(path(cfg["split_file"]).read_text(encoding="utf-8"))
    all_ids = [item["id"] for group in ("train", "val", "test") for item in split[group]]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Subject leakage: duplicate IDs across splits")
    if [len(split[g]) for g in ("train", "val", "test")] != [85, 15, 25]:
        raise ValueError("The NFBS protocol requires an 85/15/25 split")
    return {g: [{**item, "image": str(path(item["image"])), "label": str(path(item["label"]))}
                for item in split[g]] for g in groups}


def check_grid(image, label):
    if image.shape != label.shape or not np.allclose(image.affine, label.affine, atol=1e-4):
        raise ValueError("Shape or affine mismatch: masks must occupy the same physical grid")
