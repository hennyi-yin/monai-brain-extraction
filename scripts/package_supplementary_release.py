"""Prepare a complete release without rerunning or changing either analysis."""
import json
import shutil
import subprocess
import tarfile

from scripts.verify_release import verify
from src.common import digest, path, write_json
from src.supplementary_common import OUTPUT, frozen_state


def main():
    frozen = frozen_state()
    original = path("artifacts/release")
    verify(original)
    release = path("artifacts/release-v1.1.1")
    release.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "unet_weights.pt", "test_predictions.tar.gz"):
        shutil.copy2(original / name, release / name)
    manifest = json.loads(path(f"{OUTPUT}/manifest.json").read_text())
    scored = json.loads(path(f"{OUTPUT}/evaluation_once.json").read_text())
    assert manifest["complete"] and scored["status"] == "complete"
    assert len(manifest["subjects"]) == 25 and scored["scoring_passes"] == 1
    masks = []
    for subject in manifest["subjects"]:
        mask = path(subject["output"])
        if digest(mask) != subject["output_sha256"] or digest(mask) != scored["supplementary_masks_sha256"][subject["id"]]:
            raise ValueError(f"Supplementary mask changed: {subject['id']}")
        masks.append(mask)
    archive = release / "robustfov_predictions.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for mask in sorted(masks):
            bundle.add(mask, arcname=mask.relative_to(path(".")).as_posix())
    # The full frozen audit also covers last.pt and cached BET intermediates.
    # Preserve the original primary archive, and ship missing files separately.
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=path(".")).decode().split("\0"))
    with tarfile.open(release / "test_predictions.tar.gz") as bundle:
        covered = tracked | set(bundle.getnames()) | {"checkpoints/best.pt"}
    support = sorted(set(frozen["files"]) - covered)
    with tarfile.open(release / "frozen_audit_support.tar.gz", "w:gz") as bundle:
        for name in support:
            bundle.add(path(name), arcname=name)
    with tarfile.open(release / "frozen_audit_support.tar.gz") as bundle:
        assert set(bundle.getnames()) == set(support)
        assert set(frozen["files"]).issubset(covered | set(bundle.getnames()))
    assets = ("best.pt", "unet_weights.pt", "test_predictions.tar.gz", "robustfov_predictions.tar.gz", "frozen_audit_support.tar.gz")
    write_json(release / "SHA256SUMS.json", {name: digest(release / name) for name in assets})
    verify(release)
    with tarfile.open(archive) as bundle:
        assert len(bundle.getmembers()) == 25
        assert all(item.name.startswith("results/bet_robustfov/") and item.isfile() for item in bundle.getmembers())
    print(f"Complete release prepared: {release} (75 primary + 25 supplementary masks; {len(support)} frozen support files)")


if __name__ == "__main__":
    main()
