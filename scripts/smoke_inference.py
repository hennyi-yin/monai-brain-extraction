"""Check a real training scan's full-volume inference and native-grid output."""
import nibabel as nib
import numpy as np
import torch

from src.common import check_grid, config, load_split, path, write_json
from src.data import transforms
from src.infer import predict_ras, save_native
from src.model import make_model


def main():
    cfg = config()
    torch.set_num_threads(cfg["train"]["cpu_threads"])
    state = torch.load(path(cfg["checkpoint_dir"]) / "best.pt", map_location="cpu", weights_only=False)
    model = make_model(cfg).to(cfg["device"])
    model.load_state_dict(state["model"])
    model.eval()
    item = load_split(cfg, ("train",))["train"][0]
    batch = transforms(cfg, labels=False)({"image": item["image"]})
    mask = predict_ras(model, batch["image"], cfg, torch.device(cfg["device"]))
    target = path("results/smoke") / f"{item['id']}_mask.nii.gz"
    save_native(mask, item["image"], target)
    original, saved = nib.load(item["image"]), nib.load(target)
    check_grid(original, saved)
    values = np.asarray(saved.dataobj)
    assert set(np.unique(values)).issubset({0, 1}) and values.any()
    evidence = {"id": item["id"], "group": "train", "shape": list(saved.shape),
                "native_affine_restored": True, "binary_uint8": saved.get_data_dtype() == np.dtype("uint8"),
                "test_data_used": False, "checkpoint_epoch": state["epoch"]}
    write_json(path("results/inference_smoke.json"), evidence)
    print(evidence)


if __name__ == "__main__":
    main()
