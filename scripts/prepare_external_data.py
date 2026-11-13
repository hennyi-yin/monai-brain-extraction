"""Extract only the protocol's IXI and ASL T1w test pairs from the official archive."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
from pathlib import Path
import re
import shutil
import tarfile

import nibabel as nib
import numpy as np

from src.common import ROOT, config, digest, load_split, write_json
from src.external_common import (ARCHIVE_SHA, ARCHIVE_URL, COHORTS, DATA, OUTPUT, PROTOCOL,
                                 binary_values, code_commit, native_grid, require_same_grid)

PREFIX = "synthstrip_data_v1.5"


def select_ids(text):
    ids = [line.strip() for line in text.splitlines() if re.fullmatch(r"(ixi|asl)_t1_\d+", line.strip())]
    counts = Counter(sid.split("_", 1)[0] for sid in ids)
    if len(set(ids)) != len(ids) or counts != COHORTS:
        raise ValueError(f"Expected distinct test IDs with counts {COHORTS}; found {dict(counts)}")
    return sorted(ids)


def decompressed_sha(filename):
    checksum = hashlib.sha256()
    with gzip.open(filename, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def inspect(sid):
    image, label = DATA / sid / "image.nii.gz", DATA / sid / "mask.nii.gz"
    native, truth = nib.load(image), nib.load(label)
    spacing = native_grid(native)
    require_same_grid(native, truth)
    mask = binary_values(truth)
    values = np.asarray(native.dataobj)
    if not mask.any() or mask.all() or not np.isfinite(values).all() or np.ptp(values.astype(np.float32)) <= 0:
        raise ValueError(f"Invalid shared source for {sid}")
    return {"id": sid, "cohort": sid.split("_", 1)[0], "image": image.relative_to(ROOT).as_posix(),
            "label": label.relative_to(ROOT).as_posix(), "shape": list(native.shape), "spacing_mm": spacing,
            "sha256": {"image": digest(image), "label": digest(label)}, "uncompressed_image_sha256": decompressed_sha(image)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    code_commit()
    if digest(args.archive) != ARCHIVE_SHA:
        raise ValueError("Official 3D archive SHA-256 mismatch")
    with tarfile.open(args.archive, "r:") as bundle:
        split = bundle.extractfile(f"{PREFIX}/scans_test.txt").read()
        ids = select_ids(split.decode())
        names = {f"{PREFIX}/{sid}/{name}": DATA / sid / name for sid in ids for name in ("image.nii.gz", "mask.nii.gz")}
        for member in bundle.getmembers():
            if member.name not in names:
                continue
            if not member.isfile():
                raise ValueError(f"Nonregular archive entry: {member.name}")
            destination = names.pop(member.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
        if names:
            raise ValueError(f"Missing expected archive entries: {sorted(names)}")
    with ThreadPoolExecutor(max_workers=4) as workers:
        subjects = list(workers.map(inspect, ids))
        split_nfbs = load_split(config())
        hashes = set(workers.map(decompressed_sha, (item["image"] for group in split_nfbs.values() for item in group)))
    sources = {item["uncompressed_image_sha256"] for item in subjects}
    if len(sources) != len(subjects) or sources & hashes:
        raise ValueError("Duplicate external input or an exact NFBS input match")
    output = OUTPUT / "dataset.json"
    record = {"archive_url": ARCHIVE_URL, "archive_sha256": ARCHIVE_SHA, "protocol_commit": PROTOCOL,
              "publisher_test_list_sha256": hashlib.sha256(split).hexdigest(), "cohorts": COHORTS,
              "exact_duplicate_check": {"external_unique": len(sources), "nfbs_unique": len(hashes), "matches": 0},
              "subjects": subjects}
    if output.exists():
        from src.freesurfer_common import read_json
        if read_json(output) != record:
            raise ValueError("Prepared dataset differs from the locked manifest")
    else:
        write_json(output, record)
    print(f"Verified {len(subjects)} paired adult T1w scans from {dict(COHORTS)}; no exact NFBS duplicates")


if __name__ == "__main__":
    main()
