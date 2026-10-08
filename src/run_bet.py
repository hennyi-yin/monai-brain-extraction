import argparse
import os
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from src.common import check_grid, config, digest, load_split, path, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = config(args.config)
    executable = shutil.which(cfg["bet"]["executable"])
    if not executable:
        raise RuntimeError("FSL BET unavailable. Run in WSL with the FSL environment on PATH.")
    env = {**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"}
    result_dir = path(cfg["results_dir"])
    manifest_file = result_dir / "bet_manifest.json"
    import json
    signature = {"split_sha256": digest(path(cfg["split_file"])),
                 "bet_sha256": digest(executable), "bet2_sha256": digest(Path(executable).with_name("bet2")),
                 "f": cfg["bet"]["fractional_threshold"]}
    if manifest_file.exists() and json.loads(manifest_file.read_text())["signature"] != signature:
        raise ValueError("Existing BET outputs belong to a different baseline protocol")
    commands = []
    for item in load_split(cfg, ("test",))["test"]:
        for method in ("bet_default", "bet_robust"):
            destination = result_dir / method / item["id"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            output = destination.parent / (destination.name + "_mask.nii.gz")
            command = [executable, item["image"], str(destination), "-f",
                       str(cfg["bet"]["fractional_threshold"]), "-m", "-n"]
            if method == "bet_robust":
                command.append("-R")
            done_file = output.with_suffix(".done.json")
            run_signature = {"signature": signature, "method": method, "image_sha256": digest(item["image"])}
            done = json.loads(done_file.read_text()) if done_file.exists() else None
            if not (output.exists() and done and done["run"] == run_signature
                    and done["mask_sha256"] == digest(output)):
                print("BET", method, item["id"], flush=True)
                subprocess.run(command, env=env, check=True)
            image, mask = nib.load(item["image"]), nib.load(output)
            check_grid(image, mask)
            values = np.asarray(mask.dataobj)
            if not np.isfinite(values).all() or not set(np.unique(values)).issubset({0, 1}):
                raise ValueError(f"Invalid BET mask: {output}")
            write_json(done_file, {"run": run_signature, "mask_sha256": digest(output)})
            commands.append({"id": item["id"], "method": method, "argv": command,
                             "mask_sha256": digest(output)})
    write_json(manifest_file, {"signature": signature, "commands": commands})


if __name__ == "__main__":
    main()
