"""Download the official NFBS archive and extract only source T1w and brain masks."""
import hashlib
import tarfile
import urllib.request
from pathlib import Path

from src.common import path, write_json

URL = "https://fcp-indi.s3.amazonaws.com/data/Projects/RocklandSample/NFBS_Dataset.tar.gz"


def main():
    archive = path("data/NFBS_Dataset.tar.gz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        urllib.request.urlretrieve(URL, partial)
        partial.replace(archive)
    destination = path("data/NFBS")
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            parts = Path(member.name).parts
            if not member.isfile() or len(parts) != 3 or not parts[1].startswith("A"):
                continue
            name = parts[-1]
            sid = parts[1]
            if name not in (f"sub-{sid}_ses-NFB3_T1w.nii.gz", f"sub-{sid}_ses-NFB3_T1w_brainmask.nii.gz"):
                continue
            target = destination / sid / name
            if target.exists() and target.stat().st_size == member.size:
                count += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, target.open("wb") as output:
                import shutil
                shutil.copyfileobj(source, output)
            count += 1
    if count != 250:
        raise ValueError(f"Expected 250 files, extracted {count}")
    from src.common import digest
    write_json(path("results/data_source.json"), {"url": URL, "archive_sha256": digest(archive), "files": count})
    print("NFBS: 125 T1w / brain-mask pairs extracted.")


if __name__ == "__main__":
    main()
