"""Package verified weights/masks for release; does not publish anything."""
import json
import shutil
import tarfile

import torch

from src.common import config, digest, path, write_json


def main():
    cfg = config()
    results = path(cfg["results_dir"])
    frozen = json.loads((results / "evaluation_protocol.json").read_text())
    best = path(cfg["checkpoint_dir"]) / "best.pt"
    if digest(best) != frozen["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed since evaluation")
    release = path("artifacts/release")
    release.mkdir(parents=True, exist_ok=True)
    # Keep the full checkpoint for exact evaluate.py reproduction.
    shutil.copy2(best, release / "best.pt")
    state = torch.load(best, map_location="cpu", weights_only=False)
    torch.save(state["model"], release / "unet_weights.pt")
    archive = release / "test_predictions.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for method in ("unet", "bet_default", "bet_robust"):
            for item in sorted((results / method).glob("*_mask.nii.gz")):
                bundle.add(item, arcname=f"results/{method}/{item.name}")
        for manifest in (results / "unet" / "manifest.json", results / "bet_manifest.json"):
            bundle.add(manifest, arcname=str(manifest.relative_to(path("."))).replace("\\", "/"))
    hashes = {filename.name: digest(filename) for filename in release.iterdir() if filename.name != "SHA256SUMS.json"}
    write_json(release / "SHA256SUMS.json", hashes)
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
