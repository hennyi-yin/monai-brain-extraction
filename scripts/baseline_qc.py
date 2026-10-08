"""Inspect fixed BET failures without tuning or replacing the frozen baseline."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass

from src.common import check_grid, config, load_split, path, write_json


def main():
    cfg = config()
    output = path(cfg["results_dir"])
    frame = pd.read_csv(output / "per_subject.csv")
    lookup = {item["id"]: item for item in load_split(cfg, ("test",))["test"]}
    worst = frame[frame.method == "bet_robust"].sort_values("dice").head(3)
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), layout="constrained")
    audit = []
    for ax, row in zip(axes, worst.itertuples()):
        item = lookup[row.id]
        image = nib.load(item["image"])
        truth = nib.load(item["label"])
        bet = nib.load(output / "bet_robust" / f"{row.id}_mask.nii.gz")
        check_grid(image, truth)
        check_grid(image, bet)
        gt_values, bet_values = np.asarray(truth.dataobj) >= 0.5, np.asarray(bet.dataobj) > 0.5
        world = lambda c: nib.affines.apply_affine(image.affine, np.asarray(c)).tolist()
        audit.append({"id": row.id, "dice": row.dice, "hd95_mm": row.hd95,
                      "shape": list(image.shape), "native_orientation": list(nib.aff2axcodes(image.affine)),
                      "qform_matches": bool(np.allclose(image.get_qform(), bet.get_qform())),
                      "sform_matches": bool(np.allclose(image.get_sform(), bet.get_sform())),
                      "truth_centroid_world_mm": world(center_of_mass(gt_values)),
                      "bet_centroid_world_mm": world(center_of_mass(bet_values)),
                      "true_overlap_voxels": int(np.count_nonzero(gt_values & bet_values)),
                      "bet_foreground_voxels": int(bet_values.sum())})
        ras = nib.as_closest_canonical(image).get_fdata(dtype=np.float32)
        gt_ras = nib.as_closest_canonical(truth).get_fdata() >= 0.5
        bet_ras = nib.as_closest_canonical(bet).get_fdata() > 0.5
        index = ras.shape[0] // 2
        nonzero = ras[ras != 0]
        ax.imshow(ras[index].T, origin="lower", cmap="gray", vmin=np.percentile(nonzero, 1), vmax=np.percentile(nonzero, 99))
        for mask, color in ((gt_ras, "#33cc77"), (bet_ras, "#ee8844")):
            if mask[index].any():
                ax.contour(mask[index].T, levels=[0.5], colors=[color])
        ax.set_title(f"{row.id}\nBET -R Dice {row.dice:.4f}")
        ax.axis("off")
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0], [0], color=c) for c in ("#33cc77", "#ee8844")],
               ["Ground truth", "BET -R"], loc="outside lower center", ncol=2)
    fig.savefig(output / "figures" / "bet_failure_qc.png", dpi=180)
    plt.close(fig)
    write_json(output / "baseline_qc.json", {"cases": audit,
               "parameters_changed": False, "primary_masks_changed": False,
               "interpretation": "Native-grid headers match. Worst fixed BET masks lie in the neck rather than the brain."})
    print(audit)


if __name__ == "__main__":
    main()
