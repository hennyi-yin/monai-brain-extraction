import argparse
import csv
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
from monai.data import DataLoader, PersistentDataset
from monai.losses import DiceCELoss
from monai.utils import set_determinism

from src.common import config, digest, load_split, path, write_json
from src.data import transforms
from src.infer import predict_ras
from src.model import make_model


def atomic_checkpoint(destination, state):
    tmp = destination.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = config(args.config)
    set_determinism(seed=cfg["seed"])
    torch.set_num_threads(cfg["train"]["cpu_threads"])
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install the pinned CUDA 12.8 PyTorch build.")
    split = load_split(cfg, ("train", "val"))
    train_tf, val_tf = transforms(cfg, training=True), transforms(cfg)
    train_data = PersistentDataset(split["train"], train_tf, path(cfg["cache_dir"]) / "train",
                                   hash_transform=None)
    val_data = PersistentDataset(split["val"], val_tf, path(cfg["cache_dir"]) / "val")
    generator = torch.Generator().manual_seed(cfg["seed"])
    # Save all per-epoch random states. Worker seeds derive from this generator.
    loader = DataLoader(train_data, batch_size=cfg["train"]["batch_size"], shuffle=True,
                        num_workers=cfg["train"]["num_workers"], generator=generator,
                        pin_memory=device.type == "cuda", persistent_workers=False)
    model = make_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"],
                                 weight_decay=cfg["train"]["weight_decay"])
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    amp = cfg["train"]["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ckpt_dir, log_dir = path(cfg["checkpoint_dir"]), path(cfg["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    start, best, best_epoch = 1, -1.0, 0
    split_hash = digest(path(cfg["split_file"]))
    if (ckpt_dir / "last.pt").exists() and not args.resume:
        raise RuntimeError("Existing training run: use --resume instead of overwriting it")
    if args.resume:
        saved = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        if saved["config"] != cfg or saved["split_sha256"] != split_hash:
            raise ValueError("Cannot resume with a changed config or split")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start, best, best_epoch = saved["epoch"] + 1, saved["best_dice"], saved["best_epoch"]
        random.setstate(saved["random_state"])
        np.random.set_state(saved["numpy_state"])
        torch.set_rng_state(saved["torch_state"])
        generator.set_state(saved["loader_state"])
        train_tf.R.set_state(saved["transform_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_state"])
        if start > cfg["train"]["max_epochs"] or start - 1 - best_epoch >= cfg["train"]["patience_epochs"]:
            write_json(log_dir / "completed.json", {"last_epoch": start - 1, "best_epoch": best_epoch,
                       "best_val_dice": best, "checkpoint_sha256": digest(ckpt_dir / "best.pt")})
            print("Training already finished; keeping the frozen best checkpoint.")
            return
    write_json(log_dir / "run_config.json", cfg)
    write_json(log_dir / "environment.json", {"torch": torch.__version__,
               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if amp else str(device),
               "split_sha256": split_hash, "seed": cfg["seed"]})
    logfile = log_dir / "training.csv"
    if args.resume and logfile.exists():
        with logfile.open(newline="") as stream:
            history = list(csv.DictReader(stream))
        # A crash after logging but before checkpointing must not duplicate epochs.
        history = [row for row in history if int(row["epoch"]) < start]
        with logfile.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, ["epoch", "train_loss", "val_dice", "seconds"])
            writer.writeheader()
            writer.writerows(history)
    with logfile.open("a" if args.resume else "w", newline="", buffering=1) as stream:
        writer = csv.writer(stream)
        if not args.resume:
            writer.writerow(["epoch", "train_loss", "val_dice", "seconds"])
        for epoch in range(start, cfg["train"]["max_epochs"] + 1):
            tic = time.monotonic()
            model.train()
            total, count = 0.0, 0
            for batch in loader:
                x, y = batch["image"].to(device), batch["label"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda") if amp else nullcontext():
                    loss = loss_fn(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += loss.item() * len(x)
                count += len(x)
            val_dice = None
            if epoch % cfg["train"]["val_interval"] == 0:
                model.eval()
                scores = []
                for index in range(len(val_data)):
                    batch = val_data[index]
                    pred = predict_ras(model, batch["image"], cfg, device)
                    gt = batch["label"][0].numpy() >= 0.5
                    scores.append(2 * np.count_nonzero(pred & gt) / (np.count_nonzero(pred) + gt.sum()))
                val_dice = float(np.mean(scores))
                if val_dice > best:
                    best, best_epoch = val_dice, epoch
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch, "best_dice": best,
                     "best_epoch": best_epoch, "config": cfg, "split_sha256": split_hash,
                     "random_state": random.getstate(), "numpy_state": np.random.get_state(),
                     "torch_state": torch.get_rng_state(), "loader_state": generator.get_state(),
                     "transform_state": train_tf.R.get_state(),
                     "cuda_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
            writer.writerow([epoch, total / count, val_dice, time.monotonic() - tic])
            stream.flush()
            atomic_checkpoint(ckpt_dir / "last.pt", state)
            if val_dice is not None and epoch == best_epoch:
                atomic_checkpoint(ckpt_dir / "best.pt", state)
            print(f"epoch={epoch} loss={total/count:.5f} val={val_dice} best={best:.5f} "
                  f"seconds={time.monotonic()-tic:.1f}", flush=True)
            if val_dice is not None and epoch - best_epoch >= cfg["train"]["patience_epochs"]:
                print(f"Early stop: {cfg['train']['patience_epochs']} epochs without validation improvement.")
                break
    write_json(log_dir / "completed.json", {"last_epoch": epoch, "best_epoch": best_epoch,
               "best_val_dice": best, "checkpoint_sha256": digest(ckpt_dir / "best.pt")})


if __name__ == "__main__":
    main()
