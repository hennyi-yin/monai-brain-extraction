import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from src.common import load_split, path, write_json
from src.metrics import paired_test

COLORS = {"Ground truth": "#33cc77", "U-Net": "#22bbee", "BET": "#ee8844"}


def summarize(frame):
    pivot = frame.pivot(index="id", columns="method", values="dice").sort_index()
    if pivot.isna().any().any() or set(pivot.columns) != {"unet", "bet_default", "bet_robust"}:
        raise ValueError("Every subject must have exactly three method scores")
    baseline = max(("bet_default", "bet_robust"), key=lambda method: pivot[method].mean())
    stats = {}
    for method in ("unet", "bet_default", "bet_robust"):
        rows = frame[frame.method == method]
        hd = rows.hd95.to_numpy()
        stats[method] = {"dice_mean": float(rows.dice.mean()), "dice_sd": float(rows.dice.std(ddof=1)),
                         "dice_median": float(rows.dice.median()),
                         "hd95_mean_mm": float(hd.mean()) if np.isfinite(hd).all() else None,
                         "hd95_sd_mm": float(hd.std(ddof=1)) if np.isfinite(hd).all() else None,
                         "hd95_nonfinite_count": int((~np.isfinite(hd)).sum()),
                         "p_vs_unet": None if method == "unet" else paired_test(pivot.unet, pivot[method])}
    x, y = stats["unet"]["dice_mean"], (stats["unet"]["dice_mean"] - stats[baseline]["dice_mean"]) * 100
    return {"n": len(pivot), "selected_bet": baseline, "methods": stats,
            "resume": {"X": f"{x:.2f}", "N": len(pivot), "Y": f"{y:.1f}",
                       "delta_points_unrounded": y}}


def build_report(frame, cfg):
    out = path(cfg["results_dir"])
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    result = summarize(frame)
    write_json(out / "summary.json", result)
    selected = result["selected_bet"]
    rows = ["| Method | Dice mean ± SD | Dice median | HD95 mm mean ± SD | p vs U-Net |",
            "|---|---:|---:|---:|---:|"]
    for method in ("unet", "bet_default", "bet_robust"):
        s = result["methods"][method]
        hd = f"{s['hd95_mean_mm']:.2f} ± {s['hd95_sd_mm']:.2f}" if s["hd95_mean_mm"] is not None else "∞ (empty prediction)"
        p = "—" if s["p_vs_unet"] is None else f"{s['p_vs_unet']:.6g}"
        rows.append(f"| {method}{' (selected)' if method == selected else ''} | "
                    f"{s['dice_mean']:.4f} ± {s['dice_sd']:.4f} | {s['dice_median']:.4f} | {hd} | {p} |")
    resume = result["resume"]
    direction = "above" if resume["delta_points_unrounded"] >= 0 else "below"
    sentence = (f"A MONAI 3D U-Net achieves Dice **{resume['X']}** on **{resume['N']}** held-out NFBS T1w scans, "
                f"**{abs(float(resume['Y'])):.1f} points {direction}** FSL BET ({selected}; "
                f"paired Wilcoxon p = {result['methods'][selected]['p_vs_unet']:.6g}).")
    table = "\n".join(rows)
    text = (sentence + "\n\n" + table + "\n\n"
            "SD is the sample standard deviation (ddof=1). Two-sided paired Wilcoxon tests align by subject ID; "
            "zero differences are excluded, and an all-zero difference returns p=1. "
            "HD95 is the maximum of the two directed 95th-percentile surface distances, in mm.\n\n"
            "Both BET settings were fixed before testing. The BET variant with higher test mean Dice is selected "
            "as required by the project brief. This is test-based baseline selection; the selected comparison's "
            "p-value is descriptive and unadjusted for selection. Both variants are reported for transparency. "
            "The U-Net checkpoint and all model decisions use only training/validation data.\n\n"
            "Resume wording (rounded mean Dice; delta computed from unrounded means):\n\n"
            f"> Achieved a Dice of {resume['X']} on NFBS brain extraction ({resume['N']} held-out T1w scans), "
            f"{abs(float(resume['Y'])):.1f} points {direction} FSL BET, by training a MONAI 3D U-Net "
            "with intensity normalization and spatial augmentation.\n")
    (out / "summary.md").write_text(text, encoding="utf-8")
    readme = path("README.md")
    if readme.exists():
        current = readme.read_text(encoding="utf-8")
        start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        before, tail = current.split(start, 1)
        _, after = tail.split(end, 1)
        readme.write_text(before + start + "\n" + sentence + "\n\n" + table + "\n" + end + after, encoding="utf-8")
    pivot = frame.pivot(index="id", columns="method", values="dice").sort_index()
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    for _, row in pivot.iterrows():
        ax.plot([0, 1], [row[selected], row.unet], color="#9aabb9", alpha=0.65, linewidth=1)
    ax.scatter(np.zeros(len(pivot)), pivot[selected], color="#ee8844", label=selected, zorder=3)
    ax.scatter(np.ones(len(pivot)), pivot.unet, color="#2288bb", label="U-Net", zorder=3)
    ax.set(xticks=[0, 1], xticklabels=[selected, "MONAI U-Net"], ylabel="Foreground Dice",
           title=f"Paired held-out scores (n={len(pivot)})")
    fig.savefig(figures / "paired_dice.png", dpi=180)
    plt.close(fig)
    ranked = pivot.sort_values(["unet"], kind="stable")
    ids = [ranked.index[-1], ranked.index[len(ranked)//2], ranked.index[0]]
    lookup = {item["id"]: item for item in load_split(cfg, ("test",))["test"]}
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), layout="constrained")
    for ax, sid, rank in zip(axes, ids, ("Best", "Median", "Worst")):
        item = lookup[sid]
        image = nib.as_closest_canonical(nib.load(item["image"])).get_fdata(dtype=np.float32)
        masks = {"Ground truth": nib.as_closest_canonical(nib.load(item["label"])).get_fdata() >= 0.5}
        for label, method in (("U-Net", "unet"), ("BET", selected)):
            masks[label] = nib.as_closest_canonical(nib.load(out / method / f"{sid}_mask.nii.gz")).get_fdata() > 0.5
        # Fixed central sagittal plane, never selected to improve appearance.
        index = image.shape[0] // 2
        nonzero = image[image != 0]
        ax.imshow(image[index].T, cmap="gray", origin="lower", vmin=np.percentile(nonzero, 1), vmax=np.percentile(nonzero, 99))
        for label, mask in masks.items():
            if mask[index].any() and not mask[index].all():
                ax.contour(mask[index].T, levels=[0.5], colors=[COLORS[label]], linewidths=1.0)
        ax.set_title(f"{rank}: {sid}\nU-Net Dice {pivot.loc[sid, 'unet']:.4f}")
        ax.axis("off")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0], [0], color=c) for c in COLORS.values()], list(COLORS), loc="outside lower center", ncol=3)
    fig.savefig(figures / "overlays.png", dpi=180)
    plt.close(fig)
    training_curves(cfg)
    return result


def training_curves(cfg):
    filename = path(cfg["log_dir"]) / "training.csv"
    if not filename.exists():
        return
    history = pd.read_csv(filename)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].plot(history.epoch, history.train_loss, color="#2288bb")
    valid = history.dropna(subset=["val_dice"])
    axes[1].plot(valid.epoch, valid.val_dice, marker="o", color="#33aa77")
    axes[0].set(xlabel="Epoch", ylabel="Mean DiceCE loss", title="Training")
    axes[1].set(xlabel="Epoch", ylabel="Full-volume foreground Dice", title="Validation only")
    fig.savefig(path(cfg["results_dir"]) / "figures" / "training_curves.png", dpi=180)
    plt.close(fig)
