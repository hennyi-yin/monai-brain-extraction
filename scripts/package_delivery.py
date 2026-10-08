"""Create a local source-and-release bundle from committed files and verified assets."""
import json
import subprocess
import zipfile

from src.common import digest, path


def main():
    root = path(".")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root):
        raise RuntimeError("Commit the final project before creating the delivery bundle")
    release = root / "artifacts" / "release"
    hashes = json.loads((release / "SHA256SUMS.json").read_text())
    assets = []
    for name, expected in hashes.items():
        asset = release / name
        if digest(asset) != expected:
            raise ValueError(f"Release asset hash mismatch: {name}")
        assets.append(asset)
    assets.append(release / "SHA256SUMS.json")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    output = root / "artifacts" / "monai-brain-extraction-delivery.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for filename in [root / name for name in tracked if name] + assets:
            bundle.write(filename, "monai-brain-extraction/" + filename.relative_to(root).as_posix())
    print(f"{output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
