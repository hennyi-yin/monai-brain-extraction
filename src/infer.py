import argparse
from contextlib import nullcontext

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from scipy import ndimage

from src.common import config, digest, load_split, path, write_json
from src.data import transforms
from src.model import make_model


def postprocess(mask, connectivity=1):
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, connectivity))
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == sizes.argmax()
    return ndimage.binary_fill_holes(mask).astype(np.uint8)


def predict_ras(model, image, cfg, device):
    amp = cfg["train"]["amp"] and device.type == "cuda"
    with torch.inference_mode(), (torch.autocast("cuda") if amp else nullcontext()):
        logits = sliding_window_inference(
            image.unsqueeze(0).to("cpu"), tuple(cfg["infer"]["roi_size"]),
            cfg["infer"]["sw_batch_size"], model, overlap=cfg["infer"]["overlap"],
            mode=cfg["infer"]["blend_mode"], sw_device=device, device="cpu")
    mask = logits[0].argmax(dim=0).numpy()
    return postprocess(mask, cfg["postprocess"]["connectivity"])


def save_native(mask_ras, input_file, output_file):
    original = nib.load(input_file)
    original_ornt = nib.orientations.io_orientation(original.affine)
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    inverse = nib.orientations.ornt_transform(ras_ornt, original_ornt)
    native = nib.orientations.apply_orientation(mask_ras, inverse).astype(np.uint8)
    if native.shape != original.shape:
        raise ValueError("Inverse orientation did not restore native shape")
    header = original.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    result = nib.Nifti1Image(native, original.affine, header)
    result.set_qform(original.get_qform(), int(original.header["qform_code"]))
    result.set_sform(original.get_sform(), int(original.header["sform_code"]))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, output_file)


def run(cfg, checkpoint, group="test"):
    device = torch.device(cfg["device"])
    torch.set_num_threads(cfg["train"]["cpu_threads"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("split_sha256") != digest(path(cfg["split_file"])) or state.get("config") != cfg:
        raise ValueError("Checkpoint belongs to a different split or configuration")
    model = make_model(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    transform = transforms(cfg, labels=False)
    output_dir = path(cfg["results_dir"]) / "unet"
    for item in load_split(cfg, (group,))[group]:
        batch = transform({"image": item["image"]})
        prediction = predict_ras(model, batch["image"], cfg, device)
        save_native(prediction, item["image"], output_dir / f"{item['id']}_mask.nii.gz")
        print(f"Predicted {item['id']}", flush=True)
    write_json(output_dir / "manifest.json", {"checkpoint_sha256": digest(checkpoint),
               "split_sha256": digest(path(cfg["split_file"])), "config": cfg,
               "group": group, "epoch": state["epoch"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    cfg = config(args.config)
    run(cfg, path(args.checkpoint) if args.checkpoint else path(cfg["checkpoint_dir"]) / "best.pt")


if __name__ == "__main__":
    main()
