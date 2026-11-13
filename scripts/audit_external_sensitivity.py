"""Verify the separate map-back sensitivity without recalculating any score."""
import json

import nibabel as nib
import pandas as pd

from src.common import digest, path, write_json
from src.evaluate_external import summarize
from src.external_common import OUTPUT, WORK, binary_values, dataset, require_same_grid, verify_frozen
from src.freesurfer_common import read_json


def main():
    frozen = verify_frozen()
    root = OUTPUT / "geometry_sensitivity"
    state = read_json(root / "evaluation_once.json")
    primary = OUTPUT / "per_subject.csv"
    changed_csv = root / "per_subject_bet_crop.csv"
    assert state["status"] == "complete" and state["primary_csv_sha256"] == digest(primary)
    assert state["csv_sha256"] == digest(changed_csv)
    frame, changed = (pd.read_csv(file, dtype={"failure": "string"}) for file in (primary, changed_csv))
    original_crop = frame[frame.method == "bet_crop"]
    assert len(changed) == len(original_crop) and not changed.id.duplicated().any()
    assert set(changed.id) == set(original_crop.id) and set(changed.method) == {"bet_crop"}
    records = {row["id"]: row for row in state["subjects"]}
    recovered = {sid for sid, row in records.items() if row["recovered"]}
    assert len(records) == len(state["subjects"]) and set(records) == set(original_crop.loc[original_crop.status == "failed", "id"])
    a = original_crop[~original_crop.id.isin(recovered)].sort_values("id").reset_index(drop=True)
    b = changed[~changed.id.isin(recovered)].sort_values("id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    items = {item["id"]: item for item in dataset()}
    for sid in recovered:
        row = records[sid]
        work = WORK / "test/bet_crop" / sid
        assert all(digest(work / name) == expected for name, expected in row["saved_input_sha256"].items())
        destination = work / "sensitivity_native_mask.nii.gz"
        assert digest(destination) == row["output_sha256"]
        image, mask = nib.load(path(items[sid]["image"])), nib.load(destination)
        require_same_grid(image, mask)
        assert int(binary_values(mask).sum()) == row["geometry"]["foreground_voxels_preserved"]
        assert row["geometry"]["corner_error_voxels"] <= .001
        assert row["geometry"]["roi_intensities_exact"] or row["geometry"]["roi_matches_after_native_float32_cast"]
    recomputed, _ = summarize(pd.concat([frame[frame.method != "bet_crop"], changed]))
    saved = read_json(root / "summary.json")
    # The sensitivity summary used full precision for recovered rows before CSV serialization.
    for group, expected in saved["groups"].items():
        actual = recomputed["groups"][group]
        for method in expected["methods"]:
            assert abs(actual["methods"][method]["dice_mean"] - expected["methods"][method]["dice_mean"]) < 1e-10
        expected_mean = expected["unet_minus_comparator"]["bet_crop"]["mean_points"]
        actual_mean = actual["unet_minus_comparator"]["bet_crop"]["mean_points"]
        assert abs(expected_mean - actual_mean) < 1e-8
    result = {"passed": True, **frozen, "primary_failures": len(records), "recovered_masks": len(recovered),
              "primary_scores_unchanged": True, "unchanged_crop_rows": len(a)}
    write_json(root / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
