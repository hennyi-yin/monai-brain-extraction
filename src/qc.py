import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common import config, load_split, path
from src.data import transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = config(args.config)
    items = load_split(cfg, ("train",))["train"][:3]
    transform = transforms(cfg)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), layout="constrained")
    for row, item in enumerate(items):
        batch = transform(item)
        image, label = batch["image"][0].numpy(), batch["label"][0].numpy()
        if image.shape != label.shape or not np.allclose(batch["image"].affine, batch["label"].affine):
            raise ValueError("RAS preprocessing lost image/label alignment")
        for dimension, name in enumerate(("Sagittal", "Coronal", "Axial")):
            ax = axes[row, dimension]
            plane = np.take(image, image.shape[dimension] // 2, axis=dimension).T
            mask = np.take(label, label.shape[dimension] // 2, axis=dimension).T
            ax.imshow(plane, cmap="gray", origin="lower", vmin=-1, vmax=3)
            ax.contour(mask, levels=[0.5], colors=["#33cc77"], linewidths=0.8)
            ax.set_title(f"{item['id']} · {name}")
            ax.axis("off")
    output = path(cfg["results_dir"]) / "figures" / "preprocessing_qc.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
