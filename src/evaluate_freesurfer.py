"""Score the 25 FreeSurfer predictions once, without touching earlier results."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
import pandas as pd

from src.common import ROOT, check_grid, config, digest, load_split, write_json
from src.freesurfer_common import METHOD, OUTPUT, read_json, verify_frozen
from src.metrics import paired_test, score

NAMES = {"unet": "U-Net (frozen)", "bet_robust": "BET -R (frozen)",
         "bet_robustfov": "robustfov + BET -R (post hoc)",
         METHOD: "FreeSurfer 7.4.1 autorecon1 (post hoc)"}


def plot_overlays(items, records, filename):
    outputs = {row["id"]: row["output"] for row in records}
    fig, axes = plt.subplots(len(items), 3, figsize=(9, 2.65 * len(items)), squeeze=False,
                            layout="constrained")
    for item, row_axes in zip(items, axes):
        image, truth, mask = [np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj) for p in
                              (item["image"], item["label"], ROOT / outputs[item["id"]])]
        truth, mask = truth >= 0.5, mask != 0
        center = [int(np.mean(np.flatnonzero(truth.any(axis=tuple(i for i in range(3) if i != axis)))))
                  for axis in range(3)]
        limits = np.percentile(image[image != 0], [1, 99])
        for axis, ax in enumerate(row_axes):
            ax.imshow(np.take(image, center[axis], axis=axis).T, origin="lower", cmap="gray",
                      vmin=limits[0], vmax=limits[1])
            for volume, color in ((truth, "#36d591"), (mask, "#ffab58")):
                section = np.take(volume, center[axis], axis=axis)
                if section.any() and not section.all():
                    ax.contour(section.T, [.5], colors=[color], linewidths=.85)
            ax.set_title(f"{item['id']} · {('sagittal', 'coronal', 'axial')[axis]}", fontsize=10)
            ax.axis("off")
    fig.legend([Line2D([0], [0], color=c) for c in ("#36d591", "#ffab58")],
               ["NFBS reference", "FreeSurfer 7.4.1"], loc="outside lower center", ncol=2)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def summarize(frame):
    pivot = frame.pivot(index="id", columns="method", values="dice")
    if len(pivot) != 25 or set(pivot.columns) != set(NAMES) or pivot.isna().any().any():
        raise ValueError("Expected four methods on exactly 25 shared subject IDs")
    stats, paired = {}, {}
    for method in NAMES:
        rows = frame[frame.method == method]
        median_hd = float(rows.hd95.median())
        stats[method] = {"dice_mean": float(rows.dice.mean()), "dice_sd": float(rows.dice.std(ddof=1)),
                         "dice_median": float(rows.dice.median()), "dice_min": float(rows.dice.min()),
                         "dice_below_0_8_count": int((rows.dice < .8).sum()),
                         "zero_overlap_count": int((rows.dice == 0).sum()),
                         "hd95_median_mm": median_hd if np.isfinite(median_hd) else None,
                         "hd95_infinite_count": int(np.isinf(rows.hd95).sum())}
        if method != METHOD:
            gain = 100 * (pivot[METHOD] - pivot[method])
            paired[method] = {"mean_points": float(gain.mean()), "median_points": float(gain.median()),
                              "freesurfer_wins": int((gain > 0).sum()), "ties": int((gain == 0).sum())}
    return {"post_hoc": True, "n": 25, "methods": stats, "freesurfer_minus_comparator": paired,
            "wilcoxon_two_sided_p_vs_unet_unadjusted": paired_test(pivot.unet, pivot[METHOD])}, pivot


def plot_paired(pivot, filename):
    order = ("bet_robust", "bet_robustfov", METHOD, "unet")
    colors = ("#94a3b8", "#64748b", "#dc8a35", "#167f9e")
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for _, row in pivot.iterrows():
        ax.plot(range(4), row[list(order)], color="#94a3b8", alpha=.38, lw=.8)
    for i, (method, color) in enumerate(zip(order, colors)):
        ax.scatter(np.full(len(pivot), i), pivot[method], color=color, s=22, zorder=3)
    ax.set(xticks=range(4), xticklabels=["BET -R", "robustfov + BET -R\n(post hoc)",
                                        "FreeSurfer 7.4.1\n(post hoc)", "U-Net"],
           ylabel="Foreground Dice", title="Paired NFBS test scores (n=25)")
    ax.grid(axis="y", alpha=.2)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def report(result):
    lines = ["# Post-hoc supplementary baseline: FreeSurfer autorecon1", "",
             "| Method | Dice mean ± SD | Median | Min | Dice < 0.8 | Zero overlap | HD95 median (mm) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, name in NAMES.items():
        s = result["methods"][method]
        hd = "∞" if s["hd95_median_mm"] is None else f"{s['hd95_median_mm']:.2f}"
        lines.append(f"| {name} | {s['dice_mean']:.4f} ± {s['dice_sd']:.4f} | {s['dice_median']:.4f} | "
                     f"{s['dice_min']:.4f} | {s['dice_below_0_8_count']} | {s['zero_overlap_count']} | {hd} |")
    lines.extend(["", "| FreeSurfer minus comparator | Mean gain (points) | Median gain (points) | FreeSurfer wins | Ties |",
                  "|---|---:|---:|---:|---:|"])
    for method, gain in result["freesurfer_minus_comparator"].items():
        lines.append(f"| {NAMES[method]} | {gain['mean_points']:.2f} | {gain['median_points']:.2f} | "
                     f"{gain['freesurfer_wins']}/25 | {gain['ties']} |")
    lines.extend(["", f"Two-sided paired Wilcoxon p vs U-Net: **{result['wilcoxon_two_sided_p_vs_unet_unadjusted']:.6g}** "
                  "(descriptive and unadjusted).", "",
                  "The workflow was chosen after the original test results were available. All 25 cases were retained. "
                  "Failed processing is scored as an empty mask (Dice 0 and HD95 infinity).", "",
                  "Processing failures: " + (", ".join(row["id"] for row in result["failed_subjects"]) or "none") + "."])
    for row in result["failed_subjects"]:
        lines.append(f"- `{row['id']}`: {row['reason']}")
    lines.extend(["", "[Protocol](../../../docs/protocol.md#addendum-2026-11-10-post-hoc-freesurfer-baseline) · "
                  "[Manifest](manifest.json) · [Subject scores](per_subject.csv)", "",
                  "![Paired Dice](figures/paired_dice.png)", "",
                  "![Native overlays: lowest, median and highest FreeSurfer Dice](figures/native_qc.png)", ""])
    return "\n".join(lines)


def main():
    verify_frozen()
    marker = OUTPUT / "evaluation_once.json"
    if marker.exists():
        raise RuntimeError("FreeSurfer scoring has already started; refusing a second scoring pass")
    items = load_split(config(), ("test",))["test"]
    manifest_file = OUTPUT / "manifest.json"
    manifest = read_json(manifest_file)
    records = {row["id"]: row for row in manifest["subjects"]}
    if (manifest["method"] != METHOD or manifest["group"] != "test" or not manifest["complete"]
            or len(manifest["subjects"]) != 25 or set(records) != {item["id"] for item in items}):
        raise ValueError("Complete the 25-subject FreeSurfer test run first")
    primary = pd.read_csv(ROOT / "results/per_subject.csv")
    crop = pd.read_csv(ROOT / "results/supplementary/robustfov_bet/per_subject.csv")
    comparators = pd.concat([primary[primary.method.isin(("unet", "bet_robust"))], crop])
    for method in ("unet", "bet_robust", "bet_robustfov"):
        scores = comparators[comparators.method == method]
        if len(scores) != 25 or set(scores.id) != set(records) or scores[["dice", "hd95"]].isna().any().any():
            raise ValueError(f"Incomplete frozen scores for {method}")
    for item in items:
        row = records[item["id"]]
        if (any(digest(item[key]) != row["inputs"][key] for key in ("image", "label"))
                or digest(ROOT / row["output"]) != row["output_sha256"]):
            raise ValueError(f"FreeSurfer score inputs changed for {item['id']}")
    state = {"status": "started", "manifest_sha256": digest(manifest_file)}
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2)
    rows = []
    for item in items:
        row = records[item["id"]]
        native, truth, prediction = (nib.load(p) for p in (item["image"], item["label"], ROOT / row["output"]))
        check_grid(native, truth)
        check_grid(native, prediction)
        values = np.asarray(prediction.dataobj)
        if (prediction.get_data_dtype() != np.uint8 or not np.isin(values, (0, 1)).all()
                or bool(values.any()) != (row["status"] == "ok")):
            raise ValueError(f"Invalid binary prediction for {item['id']}")
        dice, hd = score(values != 0, np.asarray(truth.dataobj) >= .5, native.header.get_zooms())
        rows.append({"id": item["id"], "method": METHOD, "dice": dice, "hd95": hd})
        print(f"Scored {item['id']}", flush=True)
    predictions = pd.DataFrame(rows).sort_values("id")
    csv = OUTPUT / "per_subject.csv"
    predictions.to_csv(csv, index=False, float_format="%.10g")
    result, pivot = summarize(pd.concat([comparators, predictions]))
    result["failed_subjects"] = [{"id": row["id"], "reason": row["failure"]}
                                 for row in records.values() if row["status"] == "failed"]
    write_json(OUTPUT / "summary.json", result)
    (OUTPUT / "summary.md").write_text(report(result), encoding="utf-8")
    figure_dir = OUTPUT / "figures"
    plot_paired(pivot, figure_dir / "paired_dice.png")
    lookup = {item["id"]: item for item in items}
    ids = predictions.sort_values(["dice", "id"]).id.iloc[[0, 12, 24]]
    plot_overlays([lookup[sid] for sid in ids], manifest["subjects"], figure_dir / "native_qc.png")
    write_json(OUTPUT / "frozen_after.json", verify_frozen())
    write_json(marker, {**state, "status": "complete", "scoring_passes": 1, "csv_sha256": digest(csv)})
    print(report(result))


if __name__ == "__main__":
    main()
