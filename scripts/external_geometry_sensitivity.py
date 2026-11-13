"""Map already-generated BET masks exactly; keep all primary scores immutable."""
import itertools

import nibabel as nib
import numpy as np
import pandas as pd

from src.common import digest, path, write_json
from src.evaluate_external import GROUPS, plot_paired, score_native, summarize
from src.external_common import OUTPUT, WORK, binary_values, code_commit, dataset, native_grid, require_same_grid, verify_frozen
from src.freesurfer_common import read_json
from src.run_bet_robustfov import save_mask


def integer_map(roi, raw_mask, native):
    require_same_grid(roi, raw_mask)
    orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(roi.affine),
                                                  nib.orientations.io_orientation(native.affine))
    transform = np.linalg.solve(native.affine, roi.affine @ nib.orientations.inv_ornt_aff(orientation, roi.shape))
    intensities = nib.orientations.apply_orientation(np.asarray(roi.dataobj), orientation)
    mask = nib.orientations.apply_orientation(binary_values(raw_mask), orientation)
    start = np.rint(transform[:3, 3]).astype(int)
    corners = np.array(list(itertools.product(*((0, size - 1) for size in mask.shape))))
    actual = nib.affines.apply_affine(transform, corners)
    error = float(np.max(np.abs(actual - corners - start)))
    if error > .001:
        raise ValueError(f"Not an exact voxel permutation/translation: {error:.6g} voxels")
    stop = start + mask.shape
    if np.any(start < 0) or np.any(stop > native.shape):
        raise ValueError("Mapped crop lies outside the native grid")
    slices = tuple(slice(a, b) for a, b in zip(start, stop))
    original = np.asarray(native.dataobj)[slices]
    exact = np.array_equal(original, intensities)
    float32_roi = roi.get_data_dtype().kind == "f" and roi.get_data_dtype().itemsize == 4
    if not exact and (not float32_roi or not np.array_equal(original.astype(np.float32), intensities)):
        raise ValueError("Entire ROI does not match native intensities in the stored precision")
    if not mask.any():
        raise ValueError("Saved raw BET mask is empty")
    output = np.zeros(native.shape, dtype=np.uint8)
    output[slices] = mask
    return output, slices, {"corner_error_voxels": error, "native_start": start.tolist(), "native_stop": stop.tolist(),
                            "roi_intensities_exact": exact, "roi_matches_after_native_float32_cast": not exact,
                            "foreground_voxels_preserved": int(mask.sum())}


def main():
    verify_frozen()
    marker = OUTPUT / "evaluation_once.json"
    prior = read_json(marker)
    csv = OUTPUT / "per_subject.csv"
    crop = OUTPUT / "predictions/bet_crop.json"
    if (prior["status"] != "complete" or digest(csv) != prior["csv_sha256"]
            or digest(crop) != prior["prediction_manifest_sha256"]["bet_crop"]):
        raise ValueError("Sensitivity requires the original completed analysis")
    items = {item["id"]: item for item in dataset()}
    records = read_json(crop)["subjects"]
    directory = OUTPUT / "geometry_sensitivity"
    directory.mkdir(parents=True, exist_ok=True)
    state = {"status": "started", "primary_csv_sha256": prior["csv_sha256"], "code_commit": code_commit(), "subjects": []}
    lock = directory / "evaluation_once.json"
    with lock.open("x", encoding="utf-8"):
        pass
    write_json(lock, state)
    frame = pd.read_csv(csv)
    for row in records:
        if row["status"] != "failed":
            continue
        sid, reason = row["id"], row["failure"]
        item = items[sid]
        record = {"id": sid, "cohort": item["cohort"], "primary_failure": reason, "recovered": False}
        state["subjects"].append(record)
        source, label = path(item["image"]), path(item["label"])
        if digest(source) != item["sha256"]["image"] or digest(label) != item["sha256"]["label"]:
            raise ValueError(f"External sources changed for {sid}")
        if not reason.startswith("Native crop map-back failed:"):
            continue
        work = WORK / "test/bet_crop" / sid
        roi, mask = work / "roi.nii.gz", work / "bet_mask.nii.gz"
        if not roi.is_file() or not mask.is_file():
            record["recovery_failure"] = "Saved raw ROI or BET mask is absent"
            continue
        record["saved_input_sha256"] = {file.name: digest(file) for file in (roi, mask)}
        native = nib.load(source)
        try:
            output, slices, geometry = integer_map(nib.load(roi), nib.load(mask), native)
        except ValueError as error:
            record["recovery_failure"] = str(error)
            continue
        destination = work / "sensitivity_native_mask.nii.gz"
        save_mask(output, native, destination)
        truth = nib.load(label)
        require_same_grid(native, truth)
        reference = np.asarray(truth.dataobj) >= .5
        dice, hd = score_native(output != 0, reference, native_grid(native))
        retention = float(reference[slices].sum() / reference.sum())
        idx = (frame.id == sid) & (frame.method == "bet_crop")
        frame.loc[idx, ["dice", "hd95_mm", "status", "failure", "crop_reference_retention"]] = [dice, hd, "ok", None, retention]
        record.update(recovered=True, geometry=geometry, output_sha256=digest(destination),
                      dice=dice, hd95_mm=hd, crop_reference_retention=retention)
        print(f"Recovered {sid} using an exact whole-ROI map in the stored precision", flush=True)
    results = directory / "per_subject_bet_crop.csv"
    frame.loc[frame.method == "bet_crop"].to_csv(results, index=False, float_format="%.12g")
    summary, pivot = summarize(frame)
    write_json(directory / "summary.json", summary)
    plot_paired(pivot, summary, directory / "paired_dice_differences.png")
    write_json(lock, {**state, "status": "complete", "csv_sha256": digest(results)})
    assert digest(csv) == prior["csv_sha256"]
    verify_frozen()
    recovered = sum(row["recovered"] for row in state["subjects"])
    cast = sum(row.get("geometry", {}).get("roi_matches_after_native_float32_cast", False) for row in state["subjects"])
    lines = ["# Post-hoc crop map-back sensitivity", "",
             f"Recovered **{recovered}/{len(state['subjects'])}** primary cropped BET failures by exact voxel placement "
             f"({cast} required matching the ROI's float32 intensity representation). "
             "No extraction method was rerun, and the original predictions and scores were not changed.", "",
             "| Cohort | Cropped BET Dice mean ± SD | Remaining failures | U-Net minus cropped BET, mean [95% CI] | Wins / ties / losses |",
             "|---|---:|---:|---:|---:|"]
    for group, data in summary["groups"].items():
        stat = data["methods"]["bet_crop"]
        paired = data["unet_minus_comparator"]["bet_crop"]
        lo, hi = paired["ci95_points"]
        lines.append(f"| {GROUPS[group]} | {stat['dice_mean']:.4f} ± {stat['dice_sd']:.4f} | {stat['processing_failures']} | "
                     f"{paired['mean_points']:+.2f} [{lo:+.2f}, {hi:+.2f}] | {paired['wins']} / {paired['ties']} / {paired['losses']} |")
    remaining = [row["id"] for row in state["subjects"] if not row["recovered"]]
    lines += ["", "Unrecovered cases: " + (", ".join(remaining) or "none") + ".", "",
              "![Paired differences after the map-back sensitivity](paired_dice_differences.png)", "",
              "[Sensitivity protocol](../../../../docs/external_geometry_sensitivity.md) · [Evidence and hashes](evaluation_once.json) · [Primary analysis](../summary.md)", ""]
    (directory / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
