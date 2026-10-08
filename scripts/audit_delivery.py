"""Audit real completed experiment artifacts against the primary project requirements."""
import json

import nibabel as nib
import numpy as np
import pandas as pd

from src.common import check_grid, config, digest, load_split, path, write_json
from src.report import summarize


def main():
    cfg = config()
    split = load_split(cfg)
    results = path(cfg["results_dir"])
    data_audit = json.loads((results / "data_audit.json").read_text())
    assert data_audit["count"] == 125 and all(s["grid_aligned"] for s in data_audit["subjects"])
    smoke = json.loads((results / "smoke.json").read_text())
    assert smoke["output_shape"][1:] == [2, 96, 96, 96] and not smoke["test_data_used"]
    history = pd.read_csv(path(cfg["log_dir"]) / "training.csv")
    completed = json.loads((path(cfg["log_dir"]) / "completed.json").read_text())
    assert history.epoch.tolist() == list(range(1, int(history.epoch.max()) + 1))
    assert int(history.epoch.max()) <= cfg["train"]["max_epochs"]
    valid = history.dropna(subset=["val_dice"])
    assert all(valid.epoch % cfg["train"]["val_interval"] == 0)
    assert np.isclose(completed["best_val_dice"], float(valid.val_dice.max()), atol=1e-12)
    assert completed["best_epoch"] == int(valid.loc[valid.val_dice.idxmax(), "epoch"])
    assert completed["checkpoint_sha256"] == digest(path(cfg["checkpoint_dir"]) / "best.pt")
    frame = pd.read_csv(results / "per_subject.csv")
    assert len(frame) == 75 and not frame.duplicated(["id", "method"]).any()
    assert set(frame.id) == {item["id"] for item in split["test"]}
    assert set(frame.method) == {"unet", "bet_default", "bet_robust"}
    assert frame.dice.between(0, 1).all() and (frame.hd95 >= 0).all()
    stats = summarize(frame)
    stored = json.loads((results / "summary.json").read_text())
    assert stats["resume"]["X"] == stored["resume"]["X"]
    assert stats["resume"]["Y"] == stored["resume"]["Y"]
    evidence = json.loads((results / "evaluation_inputs.json").read_text())
    for item in split["test"]:
        image = nib.load(item["image"])
        for method in ("unet", "bet_default", "bet_robust"):
            filename = results / method / f"{item['id']}_mask.nii.gz"
            mask = nib.load(filename)
            check_grid(image, mask)
            assert set(np.unique(np.asarray(mask.dataobj))).issubset({0, 1})
            assert digest(filename) == evidence[f"{item['id']}/{method}"]
    for filename in ("paired_dice.png", "overlays.png", "training_curves.png", "preprocessing_qc.png"):
        assert (results / "figures" / filename).stat().st_size > 1000
    readme = path("README.md").read_text(encoding="utf-8")
    assert "Training and held-out evaluation are pending" not in readme
    assert f"**{stored['resume']['X']}**" in readme
    report = {"R1_to_R10": "passed", "subjects": 125, "test_subjects": 25,
              "metric_rows": 75, "native_binary_masks": 75,
              "best_epoch": completed["best_epoch"], "last_epoch": completed["last_epoch"],
              "checkpoint_sha256": completed["checkpoint_sha256"],
              "resume": stored["resume"], "publication_and_resume_compilation": "checked separately"}
    write_json(results / "acceptance.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
