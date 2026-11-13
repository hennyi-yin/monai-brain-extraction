"""Lock external predictions, score each pair once, and report paired uncertainty."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from src.common import digest, path, write_json
from src.external_common import (COHORTS, METHODS, OUTPUT, PROTOCOL, binary_values, code_commit,
                                 dataset, native_grid, require_same_grid, validate_sources, verify_frozen)
from src.freesurfer_common import read_json
from src.metrics import score

GROUPS = {"ixi": "IXI", "asl": "ASL", "pooled": "Pooled"}
COMPARATORS = ("bet_crop", "synthstrip", "bet_r")


def load_predictions(items):
    expected = {item["id"]: item for item in items}
    if len(expected) != len(items) or Counter(item["cohort"] for item in items) != COHORTS:
        raise ValueError("Dataset no longer contains all 89 unique subjects")
    runs, hashes = {}, {}
    data_sha = digest(OUTPUT / "dataset.json")
    for method in METHODS:
        filename = OUTPUT / "predictions" / f"{method}.json"
        data = read_json(filename)
        rows = {row["id"]: row for row in data["subjects"]}
        if (not data["complete"] or set(rows) != set(expected) or len(data["subjects"]) != len(expected)
                or data["dataset_sha256"] != data_sha or data["signature"]["protocol_commit"] != PROTOCOL
                or data["signature"]["method"] != method):
            raise ValueError(f"Incomplete or mismatched manifest: {method}")
        for sid, row in rows.items():
            if (row["method"] != method or row["source_sha256"] != expected[sid]["sha256"]["image"]
                    or row["status"] not in {"ok", "failed"} or digest(path(row["output"])) != row["output_sha256"]):
                raise ValueError(f"Prediction changed or invalid for {method}/{sid}")
            if row.get("footprint") and digest(path(row["footprint"])) != row["footprint_sha256"]:
                raise ValueError(f"Crop footprint changed for {sid}")
        runs[method] = rows
        hashes[method] = digest(filename)
    return runs, hashes


def score_case(item, runs):
    native = nib.load(path(item["image"]))
    truth = nib.load(path(item["label"]))
    require_same_grid(native, truth)
    reference = np.asarray(truth.dataobj) >= .5
    if not reference.any():
        raise ValueError(f"Empty shared reference: {item['id']}")
    spacing = native_grid(native)
    for method in METHODS:
        result = runs[method][item["id"]]
        pred = nib.load(path(result["output"]))
        require_same_grid(native, pred)
        values = binary_values(pred)
        if pred.get_data_dtype() != np.uint8 or int(values.sum()) != result["foreground"] or bool(values.any()) != (result["status"] == "ok"):
            raise ValueError(f"Inconsistent prediction status for {method}/{item['id']}")
        dice, hd = score(values, reference, spacing)
        retention = None
        if result.get("footprint"):
            footprint = nib.load(path(result["footprint"]))
            require_same_grid(native, footprint)
            retention = float(np.count_nonzero(reference & binary_values(footprint)) / reference.sum())
        yield {"id": item["id"], "cohort": item["cohort"], "method": method, "dice": dice, "hd95_mm": hd,
               "status": result["status"], "failure": result["failure"], "crop_reference_retention": retention}


def paired_interval(strata, repetitions=10_000, seed=42):
    """Percentile interval of a subject-weighted mean with fixed stratum sizes."""
    rng = np.random.default_rng(seed)
    boot = np.zeros(repetitions, dtype=float)
    total = 0
    for stratum in strata:
        values = np.asarray(stratum, dtype=float)
        n = len(values)
        if not n or not np.isfinite(values).all():
            raise ValueError("Each paired bootstrap stratum must be nonempty and finite")
        boot += values[rng.integers(n, size=(repetitions, n))].sum(axis=1)
        total += n
    if not total:
        raise ValueError("Bootstrap requires paired differences")
    return np.percentile(boot / total, [2.5, 97.5]).tolist()


def hd_quantile(values, probability):
    ordered = np.sort(np.asarray(values, dtype=float))
    position = probability * (len(ordered) - 1)
    lo, hi = int(np.floor(position)), int(np.ceil(position))
    if np.isposinf(ordered[hi]):
        return "inf"
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo))


def summarize(frame):
    pivot = frame.pivot(index=["cohort", "id"], columns="method", values="dice")
    if (set(pivot.columns) != set(METHODS) or Counter(pivot.index.get_level_values("cohort")) != COHORTS
            or pivot.isna().any().any() or not frame.dice.between(0, 1).all()
            or frame.hd95_mm.isna().any() or (frame.hd95_mm < 0).any()):
        raise ValueError("External results do not form the full valid paired design")
    summary = {"n": len(pivot), "primary_comparator": "bet_crop", "bootstrap": {"repetitions": 10_000, "seed": 42}, "groups": {}}
    for group in GROUPS:
        part = frame if group == "pooled" else frame[frame.cohort == group]
        pairs = pivot if group == "pooled" else pivot.loc[[group]]
        methods, contrasts = {}, {}
        for method in METHODS:
            rows = part[part.method == method]
            methods[method] = {"dice_mean": float(rows.dice.mean()), "dice_sd": float(rows.dice.std(ddof=1)),
                               "dice_median": float(rows.dice.median()), "dice_min": float(rows.dice.min()),
                               "dice_below_0_8": int((rows.dice < .8).sum()), "zero_overlap": int((rows.dice == 0).sum()),
                               "hd95_median_mm": hd_quantile(rows.hd95_mm, .5),
                               "hd95_iqr_bounds_mm": [hd_quantile(rows.hd95_mm, p) for p in (.25, .75)],
                               "hd95_infinite": int(np.isposinf(rows.hd95_mm).sum()),
                               "processing_failures": int((rows.status == "failed").sum())}
        for method in COMPARATORS:
            delta = (pairs.unet - pairs[method]) * 100
            strata = [delta.loc[cohort].to_numpy() for cohort in COHORTS if cohort in delta.index.get_level_values("cohort")]
            contrasts[method] = {"mean_points": float(delta.mean()), "median_points": float(delta.median()),
                                 "ci95_points": paired_interval(strata), "wins": int((delta > 0).sum()),
                                 "ties": int((delta == 0).sum()), "losses": int((delta < 0).sum())}
        summary["groups"][group] = {"n": len(pairs), "methods": methods, "unet_minus_comparator": contrasts}
    crop = frame.loc[frame.method == "bet_crop", "crop_reference_retention"].dropna()
    summary["crop_retention"] = {"available": len(crop), "minimum": float(crop.min()) if len(crop) else None,
                                 "below_0_99": int((crop < .99).sum())}
    summary["low_dice"] = frame.loc[frame.dice < .8, ["id", "cohort", "method", "dice", "status"]].to_dict("records")
    return summary, pivot


def plot_paired(pivot, summary, filename):
    colors = ("#117e8b", "#cc803a", "#718199")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True, layout="constrained")
    rng = np.random.default_rng(42)
    for ax, cohort in zip(axes, COHORTS):
        subset = pivot.loc[cohort]
        stats = summary["groups"][cohort]["unet_minus_comparator"]
        for index, (method, color) in enumerate(zip(COMPARATORS, colors)):
            points = 100 * (subset.unet - subset[method])
            ax.scatter(points, index + rng.uniform(-.16, .16, len(points)), s=13, color=color, alpha=.35)
            mean, (lo, hi) = stats[method]["mean_points"], stats[method]["ci95_points"]
            ax.hlines(index, lo, hi, color=color, linewidth=3)
            ax.scatter(mean, index, color=color, s=48, edgecolors="white", linewidth=.7, zorder=4)
        ax.axvline(0, color="#1f2937", linestyle="--", linewidth=1)
        ax.set_yticks(range(3), [METHODS[method] for method in COMPARATORS])
        ax.set_ylim(2.55, -.55)
        ax.set_title(f"{GROUPS[cohort]} · {len(subset)} subjects")
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("External T1w validation · paired Dice differences", fontsize=16, fontweight="bold")
    fig.supxlabel("NFBS U-Net minus comparator (Dice points) · dots: subjects; bars: bootstrap 95% CI of mean")
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def format_hd(value):
    return "∞" if value == "inf" else f"{value:.2f}"


def report(summary):
    lines = ["# External adult T1w validation: IXI and ASL", "",
             "The NFBS checkpoint was applied unchanged to 89 independent adult T1w scans. "
             "The protocol and case selection were fixed before any external prediction. The original NFBS analysis remains unchanged.",
             "", "## Absolute performance", "",
             "| Cohort | Method | Dice mean ± SD | Median | Min | Dice <0.8 | Zero overlap | HD95 median [Q1, Q3] (mm) | Execution failures |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group, data in summary["groups"].items():
        for method, stat in data["methods"].items():
            lo, hi = map(format_hd, stat["hd95_iqr_bounds_mm"])
            lines.append(f"| {GROUPS[group]} ({data['n']}) | {METHODS[method]} | {stat['dice_mean']:.4f} ± {stat['dice_sd']:.4f} | "
                         f"{stat['dice_median']:.4f} | {stat['dice_min']:.4f} | {stat['dice_below_0_8']} | {stat['zero_overlap']} | "
                         f"{format_hd(stat['hd95_median_mm'])} [{lo}, {hi}] | {stat['processing_failures']} |")
    lines += ["", "## Paired comparisons", "",
              "Differences are U-Net minus comparator, in Dice points. Intervals are paired percentile bootstrap 95% intervals for the mean; "
              "the pooled analysis preserves the cohort sizes. The pre-declared primary contrast is pooled U-Net vs robustfov + BET -R.",
              "", "| Cohort | Comparator | Mean difference [95% CI] | Median difference | U-Net wins / ties / losses |",
              "|---|---|---:|---:|---:|"]
    for group, data in summary["groups"].items():
        for method, stat in data["unet_minus_comparator"].items():
            lo, hi = stat["ci95_points"]
            lines.append(f"| {GROUPS[group]} | {METHODS[method]} | {stat['mean_points']:+.2f} [{lo:+.2f}, {hi:+.2f}] | "
                         f"{stat['median_points']:+.2f} | {stat['wins']} / {stat['ties']} / {stat['losses']} |")
    lines += ["", "![Paired external Dice differences](figures/paired_dice_differences.png)", "",
              "## Scope and reproducibility", "",
              "These are adult T1w data from two external source cohorts, not evidence of performance in children, pathology, or other contrasts. "
              "The benchmark masks began as classical-method consensus and were manually refined. Differences from NFBS can reflect "
              "acquisition, cohort, and reference conventions. SynthStrip is assessed on its own publisher's held-out benchmark. "
              "Secondary contrasts are descriptive; intervals are not multiplicity adjusted. Empty predictions are retained with Dice 0 and HD95 ∞.", "",
              "[Protocol](../../../docs/external_protocol.md) · [Source manifest](dataset.json) · "
              "[Case scores](per_subject.csv) · [Scoring lock](evaluation_once.json)", ""]
    return "\n".join(lines)


def render(csv):
    frame = pd.read_csv(csv)
    summary, pivot = summarize(frame)
    write_json(OUTPUT / "summary.json", summary)
    (OUTPUT / "summary.md").write_text(report(summary), encoding="utf-8")
    plot_paired(pivot, summary, OUTPUT / "figures/paired_dice_differences.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error("Use one to four scoring workers")
    marker, csv = OUTPUT / "evaluation_once.json", OUTPUT / "per_subject.csv"
    if args.report_only:
        state = read_json(marker)
        if state["status"] != "complete" or state["csv_sha256"] != digest(csv):
            raise RuntimeError("Cannot render without a completed, unchanged score file")
        render(csv)
        return
    verify_frozen()
    if marker.exists():
        raise RuntimeError("External scoring already started; use --report-only to redraw completed results")
    items = dataset()
    validate_sources(items)
    runs, hashes = load_predictions(items)
    state = {"status": "started", "dataset_sha256": digest(OUTPUT / "dataset.json"), "prediction_manifest_sha256": hashes,
             "scoring_code_commit": code_commit(), "scoring_source_sha256": digest(path("src/evaluate_external.py"))}
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2)
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as workers:
        for index, result in enumerate(workers.map(lambda item: list(score_case(item, runs)), items), 1):
            rows.extend(result)
            print(f"Scored {index}/{len(items)}: {result[0]['id']}", flush=True)
    pd.DataFrame(rows).to_csv(csv, index=False, float_format="%.12g")
    write_json(marker, {**state, "status": "complete", "scoring_passes": 1, "rows": len(rows), "csv_sha256": digest(csv)})
    render(csv)
    write_json(OUTPUT / "frozen_after.json", verify_frozen())


if __name__ == "__main__":
    main()
