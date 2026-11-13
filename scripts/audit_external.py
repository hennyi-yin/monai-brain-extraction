"""Verify the external cohort, saved masks, one-pass score record and frozen NFBS artifacts."""
import json

import nibabel as nib
import numpy as np
import pandas as pd

from src.common import digest, path, write_json
from src.evaluate_external import load_predictions, summarize
from src.external_common import METHODS, OUTPUT, binary_values, dataset, require_same_grid, validate_sources, verify_frozen
from src.freesurfer_common import read_json


def main():
    frozen = verify_frozen()
    items = dataset()
    validate_sources(items)
    predictions, hashes = load_predictions(items)
    marker = read_json(OUTPUT / "evaluation_once.json")
    csv = OUTPUT / "per_subject.csv"
    assert marker["status"] == "complete" and marker["scoring_passes"] == 1
    assert marker["dataset_sha256"] == digest(OUTPUT / "dataset.json") and marker["prediction_manifest_sha256"] == hashes
    assert marker["csv_sha256"] == digest(csv) and marker["rows"] == len(items) * len(METHODS)
    scores = pd.read_csv(csv)
    assert len(scores) == marker["rows"] and not scores.duplicated(["id", "method"]).any()
    assert set(scores.id) == {item["id"] for item in items} and set(scores.method) == set(METHODS)
    lookup = scores.set_index(["id", "method"])
    for item in items:
        native = nib.load(path(item["image"]))
        for method, records in predictions.items():
            row = records[item["id"]]
            image = nib.load(path(row["output"]))
            require_same_grid(native, image)
            assert image.get_data_dtype() == np.uint8
            values = binary_values(image)
            assert int(values.sum()) == row["foreground"]
            score = lookup.loc[item["id"], method]
            assert score.status == row["status"] and score.cohort == item["cohort"]
            if row["status"] == "failed":
                assert not values.any() and score.dice == 0 and np.isposinf(score.hd95_mm)
            else:
                assert values.any()
    summary, _ = summarize(scores)
    assert read_json(OUTPUT / "summary.json") == summary
    assert (OUTPUT / "figures/paired_dice_differences.png").stat().st_size > 1000
    assert read_json(OUTPUT / "frozen_after.json") == frozen
    result = {"passed": True, **frozen, "external_subjects": len(items), "native_masks": len(scores),
              "scoring_passes": marker["scoring_passes"], "processing_failures":
                  {method: summary["groups"]["pooled"]["methods"][method]["processing_failures"] for method in METHODS}}
    write_json(OUTPUT / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
