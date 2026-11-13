"""Locked inputs and native geometry for the adult T1w validation."""
import hashlib
import subprocess

import nibabel as nib
import numpy as np

from src.common import ROOT, digest, path
from src.freesurfer_common import read_json, verify_frozen as verify_nfbs

OUTPUT = path("results/external/synthstrip_v15_t1")
WORK = path("results/external_work/synthstrip_v15_t1")
DATA = path("data/synthstrip_v15_t1")
ARCHIVE_SHA = "c9f070c40266790d97fce35b8864dfcd96f039b598311030c9d707671b3b0aba"
ARCHIVE_URL = "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/data/synthstrip_data_v1.5.tar"
PROTOCOL = "b161e009b73567cabee339d97a252f0050ce1ad3"
COHORTS = {"ixi": 48, "asl": 41}
METHODS = {"unet": "NFBS U-Net", "bet_r": "BET -R", "bet_crop": "robustfov + BET -R", "synthstrip": "SynthStrip"}
SYNTH_SCRIPT = "d674926f56393e3ead4eeb4a53a3b41de16f822469d653e33cdfe9d69c3185a2"
SYNTH_WEIGHTS = "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33"


def verify_frozen():
    state = verify_nfbs()
    frozen = read_json(OUTPUT / "frozen_before.json")
    for name, checksum in frozen["freesurfer_files"].items():
        if digest(path(name)) != checksum:
            raise ValueError(f"Prior FreeSurfer artifact changed: {name}")
    readme = (ROOT / "README.md").read_bytes()
    start = b"## Supplementary baseline (post hoc): FreeSurfer autorecon1\n"
    block = start + readme.split(start, 1)[1].split(b"\n## ", 1)[0]
    if hashlib.sha256(block).hexdigest() != frozen["freesurfer_readme_sha256"]:
        raise ValueError("Prior FreeSurfer README section changed")
    return {**state, "freesurfer_files_unchanged": len(frozen["freesurfer_files"]), "readme_sections_unchanged": 3}


def code_commit():
    subprocess.run(["git", "merge-base", "--is-ancestor", PROTOCOL, "HEAD"], cwd=ROOT, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def native_grid(image):
    if len(image.shape) != 3 or not np.isfinite(image.affine).all():
        raise ValueError("Expected finite 3D native image geometry")
    spacing = np.array(image.header.get_zooms()[:3], dtype=float)
    if image.header.get_xyzt_units()[0] != "mm" or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError("Expected positive native voxel spacing in mm")
    basis = image.affine[:3, :3] / spacing
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-4, rtol=0):
        raise ValueError("Nonorthogonal native axes are unsupported by the frozen HD95 definition")
    return spacing.tolist()


def require_same_grid(image, other):
    if image.shape != other.shape or not np.allclose(image.affine, other.affine, atol=1e-4, rtol=0):
        raise ValueError("Shape or scanner-RAS affine differs from the native input")


def binary_values(image):
    values = np.asarray(image.dataobj)
    if not np.isin(values, (0, 1)).all():
        raise ValueError("Expected a finite binary mask")
    return values.astype(bool)


def dataset():
    manifest = read_json(OUTPUT / "dataset.json")
    if manifest["archive_sha256"] != ARCHIVE_SHA or manifest["cohorts"] != COHORTS:
        raise ValueError("External data manifest does not match the protocol")
    return manifest["subjects"]


def validate_sources(items):
    for item in items:
        for name in ("image", "label"):
            if digest(path(item[name])) != item["sha256"][name]:
                raise ValueError(f"External {name} changed: {item['id']}")
