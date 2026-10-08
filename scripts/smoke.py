"""One real training-subject forward/backward check; never consumes test data."""
import json
import time

import torch
from monai.data import list_data_collate
from monai.losses import DiceCELoss
from monai.utils import set_determinism

from src.common import config, load_split, path, write_json
from src.data import transforms
from src.model import make_model


def main():
    cfg = config()
    set_determinism(seed=cfg["seed"])
    torch.set_num_threads(cfg["train"]["cpu_threads"])
    device = torch.device(cfg["device"])
    transform = transforms(cfg, training=True)
    subjects = load_split(cfg, ("train",))["train"][:cfg["train"]["batch_size"]]
    batch = list_data_collate([transform(item) for item in subjects])
    model = make_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    x, y = batch["image"].to(device), batch["label"].to(device)
    torch.cuda.reset_peak_memory_stats()
    tic = time.monotonic()
    with torch.autocast("cuda"):
        output = model(x)
        loss = DiceCELoss(to_onehot_y=True, softmax=True)(output, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    evidence = {"input_shape": list(x.shape), "output_shape": list(output.shape),
                "loss": loss.item(), "seconds": time.monotonic() - tic,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu": torch.cuda.get_device_name(), "test_data_used": False}
    write_json(path("results/smoke.json"), evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
