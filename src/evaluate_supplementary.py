"""Score the post-hoc baseline once without rewriting the frozen primary analysis."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
import pandas as pd

from src.common import check_grid, config, digest, load_split, path, write_json
from src.metrics import paired_test, score
from src.supplementary_common import MASKS, METHOD, OUTPUT, frozen_state, verify_frozen

NAMES = {"unet": "U-Net (frozen)", "bet_robust": "BET -R (frozen)", METHOD: "robustfov + BET -R (post hoc)"}
SECTION = "## Supplementary baseline (post hoc): robustfov + BET -R"


def summarize(primary, supplementary, manifest):
    pivot = pd.concat([primary[primary.method.isin(("unet", "bet_robust"))], supplementary]).pivot(
        index="id", columns="method", values="dice").sort_index()
    if len(pivot) != 25 or pivot.isna().any().any():
        raise ValueError("Exactly 25 ID-aligned subjects are required")
    stats = {}
    for method in ("unet", "bet_robust", METHOD):
        rows = supplementary if method == METHOD else primary[primary.method == method]
        median_hd = float(rows.hd95.median())
        stats[method] = {"dice_mean": float(rows.dice.mean()), "dice_sd": float(rows.dice.std(ddof=1)),
                         "dice_median": float(rows.dice.median()), "dice_min": float(rows.dice.min()),
                         "dice_below_0_8_count": int((rows.dice < 0.8).sum()),
                         "zero_overlap_count": int((rows.dice == 0).sum()),
                         "hd95_median_mm": median_hd if np.isfinite(median_hd) else None,
                         "hd95_nonfinite_count": int((~np.isfinite(rows.hd95)).sum()),
                         "p_vs_unet": None if method == "unet" else paired_test(pivot.unet, pivot[method])}
    gain = (pivot.unet - pivot[METHOD]) * 100
    crop_gain = (pivot[METHOD] - pivot.bet_robust) * 100
    old_failures = pivot[pivot.bet_robust < 0.8]
    retention = [row["roi_brain_retention"] for row in manifest["subjects"] if row["roi_brain_retention"] is not None]
    return {"post_hoc": True, "n": 25, "methods": stats,
            "unet_vs_robustfov_bet": {"mean_gain_points": float(gain.mean()),
                                     "median_gain_points": float(gain.median()),
                                     "wins": int((gain > 0).sum()), "ties": int((gain == 0).sum())},
            "robustfov_bet_vs_plain_bet_R": {"mean_gain_points": float(crop_gain.mean()),
                                          "median_gain_points": float(crop_gain.median()),
                                          "wins": int((crop_gain > 0).sum()),
                                          "paired_wilcoxon_p": paired_test(pivot[METHOD], pivot.bet_robust),
                                          "previous_failures_recovered_to_0_8": int((old_failures[METHOD] >= 0.8).sum()),
                                          "previous_failure_count": len(old_failures),
                                          "previous_failure_cases": [
                                              {"id": sid, "plain_bet_R_dice": float(row.bet_robust),
                                               "robustfov_bet_dice": float(row[METHOD])}
                                              for sid, row in old_failures.iterrows()]},
            "minimum_roi_brain_retention": min(retention) if retention else None,
            "roi_retention_available_count": len(retention),
            "failed_subjects": [{"id": row["id"], "reason": row["failure"]}
                                for row in manifest["subjects"] if row["status"] == "failed"]}


def report(result):
    table = ["| Method | Dice mean ± SD | Median | Min | n < 0.8 | HD95 median (mm) | p vs U-Net |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method in ("unet", "bet_robust", METHOD):
        s = result["methods"][method]
        p = "—" if s["p_vs_unet"] is None else f"{s['p_vs_unet']:.6g}"
        hd = "∞" if s["hd95_median_mm"] is None else f"{s['hd95_median_mm']:.2f}"
        table.append(f"| {NAMES[method]} | {s['dice_mean']:.4f} ± {s['dice_sd']:.4f} | "
                     f"{s['dice_median']:.4f} | {s['dice_min']:.4f} | {s['dice_below_0_8_count']} | {hd} | {p} |")
    table = "\n".join(table)
    s = result["methods"][METHOD]
    u = result["unet_vs_robustfov_bet"]
    c = result["robustfov_bet_vs_plain_bet_R"]
    sentence = (f"On the same 25 held-out scans, the post-hoc robustfov + BET -R baseline achieves Dice "
                f"{s['dice_mean']:.4f} ± {s['dice_sd']:.4f}; the frozen U-Net's mean paired gain is "
                f"{u['mean_gain_points']:.2f} points ({u['wins']}/25 wins).")
    retention = result["minimum_roi_brain_retention"]
    retained = "unavailable" if retention is None else f"{retention:.8%}"
    failures = result["failed_subjects"]
    text = ("# Post-hoc supplementary baseline: robustfov + BET -R\n\n" + sentence + "\n\n" + table + "\n\n"
            f"U-Net mean / median paired gain: **{u['mean_gain_points']:.4f} / {u['median_gain_points']:.4f} points**; "
            f"wins **{u['wins']}/25**, ties {u['ties']}.\n\n"
            f"Cropping vs plain BET -R: mean / median paired gain **{c['mean_gain_points']:.4f} / "
            f"{c['median_gain_points']:.4f} points**, wins {c['wins']}/25, paired two-sided Wilcoxon "
            f"p={c['paired_wilcoxon_p']:.6g}. Recovered to Dice ≥0.8: "
            f"{c['previous_failures_recovered_to_0_8']}/{c['previous_failure_count']} previous failure cases.\n\n"
            f"Minimum ground-truth brain retention in the ROI: **{retained}** "
            f"({result['roi_retention_available_count']}/25 available). Failed subjects: "
            f"{', '.join(row['id'] for row in failures) if failures else 'none'}. "
            "Failed steps remain in the denominator with empty masks, Dice 0 and HD95 infinity.\n\n"
            "Zero-overlap counts (U-Net / plain BET -R / robustfov+BET): "
            + " / ".join(str(result['methods'][method]['zero_overlap_count']) for method in ("unet", "bet_robust", METHOD))
            + ".\n\nThis pipeline was chosen after viewing primary test scores. The supplementary p-values are "
            "descriptive and unadjusted; this does not replace the frozen primary table or resume claim. "
            "There was no parameter tuning, -B, subject exclusion or model change. "
            "The first three training subjects were used only to repair installation/geometry and smoke-test implementation.\n\n"
            "Dice and symmetric HD95 use src/metrics.py; truth is thresholded at ≥0.5 in its native grid. "
            "SD uses ddof=1, paired comparisons align by subject ID, and wins require a strictly positive difference. "
            "The native grid spacing is used for HD95 in mm. ROI retention is a diagnostic only.\n\n"
            "See [protocol addendum](../../../docs/protocol.md#addendum-2026-10-08-post-hoc-supplementary-baseline), "
            "[manifest](manifest.json), [frozen hashes before](frozen_before.json), "
            "[frozen hashes after](frozen_after.json).\n\n"
            "![Paired Dice](figures/paired_dice.png)\n\n"
            "![Three previous failures](figures/qc_previous_failures.png)\n")
    if failures:
        text += "\nFailure details:\n\n" + "\n".join(f"- {row['id']}: {row['reason']}" for row in failures) + "\n"
    return sentence, table, text


def figures(cfg, primary, supplementary):
    out = path(OUTPUT) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    pivot = pd.concat([primary[primary.method.isin(("unet", "bet_robust"))], supplementary]).pivot(
        index="id", columns="method", values="dice").sort_index()
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    for _, row in pivot.iterrows():
        ax.plot([0, 1], [row[METHOD], row.unet], color="#9aabb9", alpha=.65)
    ax.scatter(np.zeros(25), pivot[METHOD], color="#ee8844", zorder=3)
    ax.scatter(np.ones(25), pivot.unet, color="#2288bb", zorder=3)
    ax.set(xticks=[0, 1], xticklabels=["robustfov + BET -R\n(post hoc)", "Frozen U-Net"],
           ylabel="Foreground Dice", title="Supplementary paired scores (n=25)")
    fig.savefig(out / "paired_dice.png", dpi=180)
    plt.close(fig)
    lookup = {item["id"]: item for item in load_split(cfg, ("test",))["test"]}
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), layout="constrained")
    for ax, sid in zip(axes, ("A00039391", "A00063368", "A00063326")):
        item = lookup[sid]
        image = nib.as_closest_canonical(nib.load(item["image"])).get_fdata(dtype=np.float32)
        truth = nib.as_closest_canonical(nib.load(item["label"])).get_fdata() >= .5
        mask = nib.as_closest_canonical(nib.load(path(MASKS) / f"{sid}_mask.nii.gz")).get_fdata() > .5
        center = image.shape[0] // 2
        nonzero = image[image != 0]
        ax.imshow(image[center].T, origin="lower", cmap="gray",
                  vmin=np.percentile(nonzero, 1), vmax=np.percentile(nonzero, 99))
        for values, color in ((truth, "#33cc77"), (mask, "#ee8844")):
            if values[center].any() and not values[center].all():
                ax.contour(values[center].T, levels=[.5], colors=[color])
        ax.set_title(f"{sid}\nPlain BET -R {pivot.loc[sid, 'bet_robust']:.4f} → "
                     f"crop + BET {pivot.loc[sid, METHOD]:.4f}")
        ax.axis("off")
    fig.legend([Line2D([0], [0], color=color) for color in ("#33cc77", "#ee8844")],
               ["Ground truth", "robustfov + BET -R (post hoc)"], loc="outside lower center", ncol=2)
    fig.savefig(out / "qc_previous_failures.png", dpi=180)
    plt.close(fig)


def main():
    cfg = config()
    out = path(OUTPUT)
    frozen_state()
    marker = out / "evaluation_once.json"
    if marker.exists():
        raise RuntimeError("Supplementary test scoring has already started; refusing a second scoring pass")
    manifest = json.loads((out / "manifest.json").read_text())
    items = load_split(cfg, ("test",))["test"]
    if not manifest["complete"] or {row["id"] for row in manifest["subjects"]} != {item["id"] for item in items}:
        raise ValueError("Generate all 25 supplementary outputs before scoring")
    evidence = {item["id"]: digest(path(MASKS) / f"{item['id']}_mask.nii.gz") for item in items}
    if any(evidence[row["id"]] != row["output_sha256"] for row in manifest["subjects"]):
        raise ValueError("Supplementary masks changed since generation")
    primary = pd.read_csv(path("results/per_subject.csv"))
    primary_hash = digest(path("results/per_subject.csv"))
    write_json(marker, {"status": "started", "primary_csv_sha256": primary_hash, "supplementary_masks_sha256": evidence})
    rows = []
    for item in items:
        native, truth = nib.load(item["image"]), nib.load(item["label"])
        prediction = nib.load(path(MASKS) / f"{item['id']}_mask.nii.gz")
        check_grid(native, truth)
        check_grid(native, prediction)
        values = np.asarray(prediction.dataobj)
        if not set(np.unique(values)).issubset({0, 1}):
            raise ValueError("Supplementary mask is not binary")
        dice, hd95 = score(values > .5, np.asarray(truth.dataobj) >= .5, native.header.get_zooms())
        rows.append({"id": item["id"], "method": METHOD, "dice": dice, "hd95": hd95})
        print(f"Scored supplementary {item['id']}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["id", "method"])
    frame.to_csv(out / "per_subject.csv", index=False, float_format="%.10g")
    result = summarize(primary, frame, manifest)
    write_json(out / "summary.json", result)
    sentence, table, text = report(result)
    (out / "summary.md").write_text(text, encoding="utf-8")
    figures(cfg, primary, frame)
    readme = path("README.md")
    original = readme.read_bytes()
    if SECTION.encode() in original:
        raise ValueError("Supplementary README section already exists")
    addition = ("\n\n" + SECTION + "\n\n" + sentence + "\n\n" + table + "\n\n"
                "Chosen after viewing test scores; all 25 subjects and the pre-declared pipeline are retained. "
                "The frozen primary analysis above is unchanged. "
                "See the [addendum](docs/protocol.md#addendum-2026-10-08-post-hoc-supplementary-baseline), "
                "[supplementary summary](results/supplementary/robustfov_bet/summary.md) and "
                "[command/geometry manifest](results/supplementary/robustfov_bet/manifest.json).\n").encode()
    readme.write_bytes(original + addition)
    verify_frozen()
    write_json(marker, {"status": "complete", "scoring_passes": 1, "primary_csv_sha256": primary_hash,
                        "supplementary_masks_sha256": evidence, "supplementary_csv_sha256": digest(out / "per_subject.csv")})
    print(text)


if __name__ == "__main__":
    main()
