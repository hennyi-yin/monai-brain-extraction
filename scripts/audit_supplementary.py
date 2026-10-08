"""Validate supplementary artifacts and frozen hashes without scoring test scans again."""
import json

import nibabel as nib
import numpy as np
import pandas as pd

from src.common import check_grid, config, digest, load_split, path, write_json
from src.supplementary_common import METHOD, OUTPUT, verify_frozen


def main():
    frozen = verify_frozen()
    out = path(OUTPUT)
    manifest = json.loads((out / "manifest.json").read_text())
    marker = json.loads((out / "evaluation_once.json").read_text())
    assert marker["status"] == "complete" and marker["scoring_passes"] == 1
    assert digest(out / "per_subject.csv") == marker["supplementary_csv_sha256"]
    frame = pd.read_csv(out / "per_subject.csv")
    items = {item["id"]: item for item in load_split(config(), ("test",))["test"]}
    assert len(frame) == 25 and not frame.id.duplicated().any() and set(frame.id) == set(items)
    assert set(frame.method) == {METHOD}
    assert len(manifest["subjects"]) == 25 and manifest["complete"]
    for row in manifest["subjects"]:
        filename = path(row["output"])
        assert digest(filename) == row["output_sha256"] == marker["supplementary_masks_sha256"][row["id"]]
        native, mask = nib.load(items[row["id"]]["image"]), nib.load(filename)
        check_grid(native, mask)
        values = np.asarray(mask.dataobj)
        assert set(np.unique(values)).issubset({0, 1}) and mask.get_data_dtype() == np.dtype("uint8")
        score_row = frame[frame.id == row["id"]].iloc[0]
        if row["status"] == "failed":
            assert not values.any() and score_row.dice == 0 and np.isinf(score_row.hd95)
        else:
            checks = row["mapping"]
            assert checks["passed"] and checks["roi_t1w_max_abs_diff"] == 0
            assert checks["mask_voxels_original"] == checks["mask_voxels_native"] == int(values.sum())
            assert checks["roi_footprint_voxels_original"] == checks["roi_footprint_voxels_native"]
            assert 0 <= row["roi_brain_retention"] <= 1
    for name in ("paired_dice.png", "qc_previous_failures.png"):
        assert (out / "figures" / name).stat().st_size > 1000
    result = {"passed": True, "frozen_files_unchanged": len(frozen["files"]),
              "readme_results_block_unchanged": True, "supplementary_subjects": 25,
              "supplementary_test_scoring_passes": 1,
              "failed_subjects": [row["id"] for row in manifest["subjects"] if row["status"] == "failed"]}
    write_json(out / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
