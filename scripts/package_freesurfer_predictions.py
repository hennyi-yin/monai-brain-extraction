"""Package the 25 saved native masks without rerunning either analysis."""
import hashlib
import tarfile

from src.common import ROOT, digest, write_json
from src.freesurfer_common import MASKS, OUTPUT, read_json, verify_frozen


def main():
    verify_frozen()
    manifest_file = OUTPUT / "manifest.json"
    manifest = read_json(manifest_file)
    marker = read_json(OUTPUT / "evaluation_once.json")
    if (not read_json(OUTPUT / "verification.json")["passed"] or not manifest["complete"]
            or marker["status"] != "complete" or marker["scoring_passes"] != 1
            or marker["manifest_sha256"] != digest(manifest_file)):
        raise ValueError("Finish the FreeSurfer evaluation and audit first")
    files = {}
    for row in manifest["subjects"]:
        mask = MASKS / "test" / f"{row['id']}_mask.nii.gz"
        name = mask.relative_to(ROOT).as_posix()
        if name != row["output"] or digest(mask) != row["output_sha256"]:
            raise ValueError(f"FreeSurfer mask changed: {row['id']}")
        files[name] = row["output_sha256"]
    if len(files) != 25:
        raise ValueError("Expected 25 distinct native test masks")
    release = ROOT / "artifacts/freesurfer"
    release.mkdir(parents=True, exist_ok=True)
    archive = release / "freesurfer_predictions.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name in sorted(files):
            bundle.add(ROOT / name, arcname=name)
    with tarfile.open(archive) as bundle:
        assert set(bundle.getnames()) == set(files)
        for item in bundle.getmembers():
            assert hashlib.sha256(bundle.extractfile(item).read()).hexdigest() == files[item.name]
    write_json(release / "SHA256SUMS.json", {archive.name: digest(archive)})
    print(f"{archive} — 25 verified native-grid test masks")


if __name__ == "__main__":
    main()
