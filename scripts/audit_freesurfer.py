"""Audit saved FreeSurfer outputs without rescoring or requiring FreeSurfer."""
import json

import nibabel as nib
import numpy as np
import pandas as pd

from src.common import ROOT, check_grid, config, digest, load_split, write_json
from src.freesurfer_common import METHOD, OUTPUT, read_json, verify_frozen


def main():
    frozen = verify_frozen()
    manifest = read_json(OUTPUT / "manifest.json")
    marker = read_json(OUTPUT / "evaluation_once.json")
    assert marker["status"] == "complete" and marker["scoring_passes"] == 1
    assert marker["manifest_sha256"] == digest(OUTPUT / "manifest.json")
    assert marker["csv_sha256"] == digest(OUTPUT / "per_subject.csv")
    scores = pd.read_csv(OUTPUT / "per_subject.csv")
    items = {item["id"]: item for item in load_split(config(), ("test",))["test"]}
    assert len(scores) == 25 and not scores.id.duplicated().any()
    assert set(scores.id) == set(items) and set(scores.method) == {METHOD}
    assert manifest["complete"] and manifest["group"] == "test" and manifest["method"] == METHOD
    assert len(manifest["subjects"]) == 25 and {row["id"] for row in manifest["subjects"]} == set(items)
    assert manifest["signature"]["omp_threads"] == 1
    smoke_path = OUTPUT / "training_smoke_manifest.json"
    smoke = read_json(smoke_path)
    assert digest(smoke_path) == manifest["reviewed_training_smoke_sha256"]
    assert smoke["complete"] and smoke["group"] == "train" and smoke["signature"] == manifest["signature"]
    assert [row["id"] for row in smoke["subjects"]] == [item["id"] for item in load_split(config(), ("train",))["train"][:3]]
    assert all(row["status"] == "ok" for row in smoke["subjects"])
    assert digest(OUTPUT / "figures/training_qc.png") == smoke["qc_sha256"]
    for row in manifest["subjects"]:
        item = items[row["id"]]
        filename = ROOT / row["output"]
        assert digest(filename) == row["output_sha256"]
        assert all(digest(item[key]) == row["inputs"][key] for key in ("image", "label"))
        native, mask = nib.load(item["image"]), nib.load(filename)
        check_grid(native, mask)
        assert mask.get_data_dtype() == np.uint8
        values = np.asarray(mask.dataobj)
        assert np.isin(values, (0, 1)).all()
        score = scores[scores.id == row["id"]].iloc[0]
        if row["status"] == "failed":
            assert not values.any() and score.dice == 0 and np.isinf(score.hd95)
        else:
            assert row["status"] == "ok" and row["returncode"] == 0 and values.any()
            assert "mri_watershed" in row["watershed_command"] and "brainmask.auto.mgz" in row["watershed_command"]
            assert row["mapping"]["imported_grid_matches_native"] and row["mapping"]["mask_grid_matches_conformed_orig"]
            assert row["mapping"]["native_foreground_voxels"] == int(values.sum())
    for filename in ("paired_dice.png", "native_qc.png"):
        assert (OUTPUT / "figures" / filename).stat().st_size > 1000
    result = {"passed": True, **frozen, "subjects": 25, "scoring_passes": 1,
              "failed_subjects": [row["id"] for row in manifest["subjects"] if row["status"] == "failed"]}
    write_json(OUTPUT / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
