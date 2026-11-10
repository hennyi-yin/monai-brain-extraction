"""Paths and integrity checks for the independent FreeSurfer comparison."""
import hashlib
import json
import subprocess

from src.common import ROOT, digest, path
from src.supplementary_common import frozen_state as primary_state

METHOD = "freesurfer_autorecon1"
VERSION = "7.4.1"
MASKS = path(f"results/{METHOD}")
OUTPUT = path(f"results/supplementary/{METHOD}")
PREREGISTRATION = "6602c78d24ace61c231091214a10b77248a47d52"
ROBUSTFOV_START = b"## Supplementary baseline (post hoc): robustfov + BET -R\n"


def read_json(filename):
    return json.loads(filename.read_text(encoding="utf-8"))


def relative(filename):
    return filename.relative_to(ROOT).as_posix()


def verify_frozen():
    expected = read_json(OUTPUT / "frozen_before.json")
    state = primary_state()
    for key in state:
        if state[key] != expected[key]:
            raise RuntimeError(f"Frozen primary evidence differs: {key}")
    changed = [name for name, checksum in expected["robustfov_files"].items()
               if digest(path(name)) != checksum]
    readme = path("README.md").read_bytes()
    block = ROBUSTFOV_START + readme.split(ROBUSTFOV_START, 1)[1].split(b"\n## ", 1)[0]
    if hashlib.sha256(block).hexdigest() != expected["robustfov_readme_sha256"]:
        changed.append("README robustfov section")
    if changed:
        raise RuntimeError(f"Frozen supplementary evidence differs: {changed}")
    return {"primary_files_unchanged": len(state["files"]),
            "robustfov_files_unchanged": len(expected["robustfov_files"]),
            "readme_sections_unchanged": 2}


def protocol_commit():
    subprocess.run(["git", "merge-base", "--is-ancestor", PREREGISTRATION, "HEAD"],
                   cwd=ROOT, check=True)
    return PREREGISTRATION
