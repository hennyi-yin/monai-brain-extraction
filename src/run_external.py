"""Predict the fixed external T1w cohort without reading reference masks for inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import time

import nibabel as nib
import numpy as np

from src.common import ROOT, config, digest, load_split, path, write_json
from src.external_common import (METHODS, OUTPUT, PROTOCOL, SYNTH_SCRIPT, SYNTH_WEIGHTS, WORK, binary_values,
                                 code_commit, dataset, native_grid, require_same_grid, validate_sources, verify_frozen)
from src.freesurfer_common import read_json, relative
from src.run_bet_robustfov import environment, fallback, map_back, mapping_checks, save_mask
from src.supplementary_common import OUTPUT as BET_SOURCE

SOURCES = ("src/run_external.py", "src/external_common.py", "src/infer.py", "src/data.py", "src/model.py",
           "src/run_bet_robustfov.py", "src/common.py")


def signature(method):
    result = {"method": method, "protocol_commit": PROTOCOL,
              "source_sha256": {name: digest(path(name)) for name in SOURCES},
              "python": sys.version.split()[0], "numpy": np.__version__, "nibabel": nib.__version__}
    if method.startswith("bet"):
        env = environment()
        prior = read_json(path(BET_SOURCE) / "manifest.json")["environment"]
        if env["executable_sha256"] != prior["executable_sha256"] or env["standard_orientation_reference"]["sha256"] != prior["standard_orientation_reference"]["sha256"]:
            raise ValueError("External BET must use the original FSL binaries and orientation template")
        result["fsl"] = {key: env[key] for key in ("executables", "executable_sha256", "standard_orientation_reference", "fsl_packages")}
    else:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("The locked external learned methods use CUDA")
        result.update(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
    if method == "unet":
        cfg = config()
        checkpoint = path(cfg["checkpoint_dir"]) / "best.pt"
        frozen = read_json(path("results/evaluation_protocol.json"))
        if frozen["config"] != cfg or digest(checkpoint) != frozen["checkpoint_sha256"]:
            raise ValueError("The NFBS model or configuration changed")
        result.update(checkpoint_sha256=digest(checkpoint), config_sha256=digest(path("configs/config.yaml")),
                      monai=importlib.metadata.version("monai"))
    if method == "synthstrip":
        home = Path(os.environ["FREESURFER_HOME"])
        script, weights = home / "python/scripts/mri_synthstrip", home / "models/synthstrip.1.pt"
        if digest(script) != SYNTH_SCRIPT or digest(weights) != SYNTH_WEIGHTS:
            raise ValueError("Unexpected SynthStrip script or weight hash")
        result["synthstrip"] = {"script": str(script), "weights": str(weights), "script_sha256": SYNTH_SCRIPT,
                               "weights_sha256": SYNTH_WEIGHTS, "surfa": importlib.metadata.version("surfa"),
                               "backend": "CUDA", "border_mm": 1, "csf_inclusive": True}
    return result


def unet_predictor():
    import torch
    from monai.utils import set_determinism
    from src.data import transforms
    from src.infer import predict_ras, save_native
    from src.model import make_model

    cfg = config()
    set_determinism(seed=cfg["seed"])
    torch.set_num_threads(cfg["train"]["cpu_threads"])
    device = torch.device(cfg["device"])
    state = torch.load(path(cfg["checkpoint_dir"]) / "best.pt", map_location="cpu", weights_only=False)
    if state["config"] != cfg or state["split_sha256"] != digest(path(cfg["split_file"])) or state["epoch"] != 190:
        raise ValueError("Unexpected NFBS checkpoint provenance")
    model = make_model(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    transform = transforms(cfg, labels=False)

    def predict(source, destination):
        mask = predict_ras(model, transform({"image": str(source)})["image"], cfg, device)
        save_native(mask, source, destination)

    return predict


def execute(argv, records):
    argv = [str(arg) for arg in argv]
    result = subprocess.run(argv, capture_output=True, text=True, env={**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"})
    records.append({"argv": argv, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    if result.returncode:
        output = (result.stdout + result.stderr).lower()
        shared = ("no space left", "out of memory", "cannot allocate memory", "modulenotfounderror", "shared libraries")
        if result.returncode < 0 or result.returncode == 137 or any(message in output for message in shared):
            raise RuntimeError(f"Shared runtime failure: {argv[0]} exited {result.returncode}: {result.stderr}")
        raise ValueError(f"Subject command exited {result.returncode}: {argv[0]}: {result.stderr[-700:]}")


def crop_bet(source, native, work, tools, record):
    commands = record["commands"]
    reoriented, roi_file = work / "reoriented.nii.gz", work / "roi.nii.gz"
    execute([tools["fslreorient2std"], source, reoriented], commands)
    execute([tools["robustfov"], "-i", reoriented, "-r", roi_file, "-b", "170", "-m", work / "roi_to_reoriented.mat"], commands)
    execute([tools["bet"], roi_file, work / "bet", "-f", "0.5", "-R", "-m", "-n"], commands)
    mask_file = work / "bet_mask.nii.gz"
    roi, mask = nib.load(roi_file), nib.load(mask_file)
    require_same_grid(roi, mask)
    binary_values(mask)
    mapped = map_back(roi, mask, native)
    checks = mapping_checks(roi, mask, native, mapped)
    record["mapping"] = {"method": "nibabel nearest neighbor", **checks}
    if not checks["passed"]:
        # Reuse the earlier basis conversion and exact ROI checks; no registration is estimated.
        try:
            mapped = fallback(roi_file, mask_file, source, native, work, tools, commands)
        except RuntimeError as error:
            if any(word in str(error).lower() for word in ("no space left", "out of memory", "cannot allocate memory")):
                raise
            raise ValueError(str(error)) from error
        checks = mapping_checks(roi, mask, native, mapped)
        record["mapping"] = {"method": "FSL matrices and FLIRT nearest neighbor", **checks}
    if not checks["passed"]:
        raise ValueError(f"Native crop map-back failed: {checks}")
    footprint = work / "native_footprint.nii.gz"
    save_mask(binary_values(mapped[2]), native, footprint)
    record.update(footprint=relative(footprint), footprint_sha256=digest(footprint))
    return binary_values(mapped[1])


def run_one(item, method, sig, directory, predict):
    sid = item["id"]
    work = directory / sid
    work.mkdir(parents=True, exist_ok=True)
    marker = work / "attempt.json"
    output = work / "native_mask.nii.gz"
    source = path(item["image"])
    native = nib.load(source)
    record = {"id": sid, "method": method, "source_sha256": digest(source), "commands": [], "status": "started"}
    previous = marker.exists()
    if previous:
        saved = read_json(marker)
        if saved["source_sha256"] != record["source_sha256"]:
            raise ValueError(f"Attempt source differs for {method}/{sid}")
        if saved["status"] in ("ok", "failed"):
            if digest(output) != saved["output_sha256"]:
                raise ValueError(f"Saved mask changed for {method}/{sid}")
            return saved
    else:
        write_json(marker, record)
    start = time.perf_counter()
    try:
        if previous:
            raise ValueError("Interrupted earlier attempt; not rerun")
        if method == "unet":
            raw = work / "unet.nii.gz"
            predict(source, raw)
        elif method == "bet_r":
            raw = work / "bet_mask.nii.gz"
            execute([sig["fsl"]["executables"]["bet"], source, work / "bet", "-f", "0.5", "-R", "-m", "-n"], record["commands"])
        elif method == "synthstrip":
            raw = work / "synthstrip.nii.gz"
            spec = sig["synthstrip"]
            execute([sys.executable, spec["script"], "-i", source, "-m", raw, "-b", "1", "--model", spec["weights"], "-g"], record["commands"])
        if method == "bet_crop":
            mask = crop_bet(source, native, work, sig["fsl"]["executables"], record)
        else:
            if not raw.is_file():
                raise ValueError("Method did not produce a mask")
            predicted = nib.load(raw)
            require_same_grid(native, predicted)
            mask = binary_values(predicted)
        if not mask.any():
            raise ValueError("Method produced an empty prediction")
        record.update(status="ok", failure=None)
    except (ValueError, nib.filebasedimages.ImageFileError) as error:
        record.update(status="failed", failure=str(error))
        mask = np.zeros(native.shape, dtype=np.uint8)
    save_mask(mask, native, output)
    saved = nib.load(output)
    require_same_grid(native, saved)
    if saved.get_data_dtype() != np.uint8:
        raise ValueError("Native mask must be uint8")
    record.update(output=relative(output), output_sha256=digest(output), foreground=int(mask.sum()),
                  seconds=round(time.perf_counter() - start, 3))
    write_json(marker, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    method = args.method
    if not 1 <= args.jobs <= 4 or (not method.startswith("bet") and args.jobs != 1):
        parser.error("Use one GPU process or up to four independent BET workers")
    verify_frozen()
    commit = code_commit()
    sig = signature(method)
    if args.smoke:
        items = load_split(config(), ("train",))["train"][:1]
        directory = WORK / "smoke" / method
        output = OUTPUT / "smoke" / f"{method}.json"
    else:
        if (OUTPUT / "evaluation_once.json").exists():
            raise RuntimeError("External scoring already started")
        smoke = read_json(OUTPUT / "smoke" / f"{method}.json")
        if smoke["signature"] != sig or any(row["status"] != "ok" for row in smoke["subjects"]):
            raise RuntimeError(f"A matching NFBS training smoke is required for {method}")
        items = dataset()
        validate_sources(items)
        directory = WORK / "test" / method
        output = OUTPUT / "predictions" / f"{method}.json"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("x"):
        try:
            _run(method, sig, items, directory, output, commit, args)
        finally:
            (directory / "run.lock").unlink()


def _run(method, sig, items, directory, output, commit, args):
    if output.exists():
        manifest = read_json(output)
        if manifest["signature"] != sig:
            raise ValueError(f"Runtime or code changed for {method}")
        if manifest["complete"]:
            print(f"{method}: existing complete manifest; no prediction rerun")
            return
    else:
        manifest = {"signature": sig, "code_commit": commit, "complete": False, "subjects": [],
                    "dataset_sha256": None if args.smoke else digest(OUTPUT / "dataset.json")}
        write_json(output, manifest)
    predict = unet_predictor() if method == "unet" else None

    def run(item):
        return run_one(item, method, sig, directory, predict)

    if args.jobs == 1:
        rows = list(_progress(map(run, items), method, len(items)))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as workers:
            rows = list(_progress(workers.map(run, items), method, len(items)))
    manifest.update(complete=True, subjects=rows)
    write_json(output, manifest)
    if args.smoke and rows[0]["status"] != "ok":
        raise SystemExit(f"Training smoke failed: {rows[0]['failure']}")
    verify_frozen()


def _progress(iterator, method, n):
    for index, row in enumerate(iterator, 1):
        print(f"{method} {index}/{n} {row['id']} {row['status']} ({row['seconds']:.1f}s)", flush=True)
        yield row


if __name__ == "__main__":
    main()
