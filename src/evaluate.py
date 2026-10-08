import argparse
import json

import nibabel as nib
import numpy as np
import pandas as pd

from src.common import check_grid, config, digest, load_split, path, write_json
from src.metrics import score
from src.report import build_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = config(args.config)
    output = path(cfg["results_dir"])
    checkpoint = path(cfg["checkpoint_dir"]) / "best.pt"
    completed = path(cfg["log_dir"]) / "completed.json"
    if not completed.exists():
        raise RuntimeError("Finish training before opening the held-out evaluation")
    completion = json.loads(completed.read_text())
    if completion["checkpoint_sha256"] != digest(checkpoint):
        raise ValueError("Best checkpoint has changed since training completion")
    items = load_split(cfg, ("test",))["test"]
    freeze = output / "evaluation_protocol.json"
    protocol = {"config": cfg, "checkpoint_sha256": digest(checkpoint),
                "split_sha256": digest(path(cfg["split_file"])), "test_ids": sorted(x["id"] for x in items)}
    if freeze.exists():
        if json.loads(freeze.read_text()) != protocol:
            raise RuntimeError("Held-out evaluation is frozen; changed model/config/split rejected")
    else:
        write_json(freeze, protocol)
    prediction_manifest = output / "unet" / "manifest.json"
    if not prediction_manifest.exists():
        from src.infer import run
        run(cfg, checkpoint)
    manifest = json.loads(prediction_manifest.read_text())
    if (manifest["checkpoint_sha256"] != protocol["checkpoint_sha256"] or
        manifest["split_sha256"] != protocol["split_sha256"] or manifest["config"] != cfg or manifest["group"] != "test"):
        raise ValueError("Prediction manifest does not match the frozen evaluation")
    if not (output / "bet_manifest.json").exists():
        raise RuntimeError("Generate both BET baselines first: python -m src.run_bet")
    evidence = {}
    # Fail before computing any test score if even one required prediction is missing.
    for item in items:
        for method in ("unet", "bet_default", "bet_robust"):
            filename = output / method / f"{item['id']}_mask.nii.gz"
            if not filename.exists():
                raise FileNotFoundError(filename)
            evidence[f"{item['id']}/{method}"] = digest(filename)
        evidence[f"{item['id']}/truth"] = digest(item["label"])
    evidence_file = output / "evaluation_inputs.json"
    if evidence_file.exists() and json.loads(evidence_file.read_text()) != evidence:
        raise RuntimeError("Evaluation input masks changed after the first scoring pass")
    write_json(evidence_file, evidence)
    rows = []
    for item in items:
        truth_native = nib.load(item["label"])
        truth = nib.as_closest_canonical(truth_native)
        g = np.asarray(truth.dataobj) >= 0.5
        for method in ("unet", "bet_default", "bet_robust"):
            native = nib.load(output / method / f"{item['id']}_mask.nii.gz")
            check_grid(truth_native, native)
            prediction = nib.as_closest_canonical(native)
            check_grid(truth, prediction)
            values = np.asarray(prediction.dataobj)
            if not np.isfinite(values).all() or not set(np.unique(values)).issubset({0, 1}):
                raise ValueError(f"{item['id']}/{method}: invalid binary prediction")
            dice, hd95 = score(values > 0.5, g, truth.header.get_zooms())
            rows.append({"id": item["id"], "method": method, "dice": dice, "hd95": hd95})
        print(f"Scored {item['id']}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["id", "method"])
    target = output / "per_subject.csv"
    tmp = target.with_suffix(".tmp")
    frame.to_csv(tmp, index=False, float_format="%.10g")
    tmp.replace(target)
    result = build_report(frame, cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
