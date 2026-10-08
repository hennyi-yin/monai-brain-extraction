import argparse
import random

import nibabel as nib
import numpy as np

from src.common import check_grid, config, path, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = config(args.config)
    folders = sorted(path(cfg["data_dir"]).glob("A*"))
    if len(folders) != 125:
        raise ValueError(f"Expected 125 subject directories, found {len(folders)}")
    items, audit = [], []
    for folder in folders:
        sid = folder.name
        img = folder / f"sub-{sid}_ses-NFB3_T1w.nii.gz"
        mask = folder / f"sub-{sid}_ses-NFB3_T1w_brainmask.nii.gz"
        image, label = nib.load(img), nib.load(mask)
        check_grid(image, label)
        if len(image.shape) != 3:
            raise ValueError(f"{sid}: expected a 3D scan")
        for volume in (image, label):
            if not np.allclose(volume.header.get_zooms(), [1, 1, 1], atol=1e-4):
                raise ValueError(f"{sid}: expected 1 mm isotropic spacing")
            values = np.asarray(volume.dataobj)
            if not np.isfinite(values).all():
                raise ValueError(f"{sid}: nonfinite voxels")
        label_values = np.asarray(label.dataobj)
        if label_values.min() < 0 or label_values.max() > 1 or not np.any(label_values >= 0.5):
            raise ValueError(f"{sid}: label must be in [0,1] with foreground at threshold 0.5")
        items.append({"id": sid, "image": f"{cfg['data_dir']}/{sid}/{img.name}",
                      "label": f"{cfg['data_dir']}/{sid}/{mask.name}"})
        audit.append({"id": sid, "shape": list(image.shape), "spacing_mm": [1, 1, 1],
                      "orientation": list(nib.aff2axcodes(image.affine)), "grid_aligned": True,
                      "fractional_label_voxels": int(np.count_nonzero((label_values > 0) & (label_values < 1))),
                      "label_threshold": 0.5})
    random.Random(cfg["seed"]).shuffle(items)
    proposed = {"train": items[:85], "val": items[85:100], "test": items[100:]}
    target = path(cfg["split_file"])
    if target.exists():
        import json
        if json.loads(target.read_text()) != proposed:
            raise ValueError("Existing split differs; refusing to replace the frozen protocol")
    else:
        write_json(target, proposed)
    write_json(path("results/data_audit.json"), {"count": 125, "subjects": audit})
    print("All 125 pairs passed shape, affine, label range, finite voxel and 1 mm checks.")
    print("Frozen seed-42 split: 85 train / 15 validation / 25 test.")


if __name__ == "__main__":
    main()
